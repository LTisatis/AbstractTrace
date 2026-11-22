import hashlib
import json
from copy import deepcopy
from urllib.parse import parse_qs, urlsplit

import pymupdf
import pytest

from abstract_review.references import (
    HttpResponse,
    ReferenceResolutionError,
    _public_addresses,
    check_external,
    check_omissions,
    resolve_pdf,
)


DOI = "10.1000/mobility"
ARXIV = "2401.01234"
CONTACT = "reviewer@example.test"
PDF_URL = "https://papers.example.org/mobility.pdf"


@pytest.fixture
def article():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "Results")
        page.insert_text((40, 65), "The treatment did not improve weekly mobility.")
        return pdf.tobytes()


def json_response(value, status=200):
    return HttpResponse(status, {}, json.dumps(value).encode())


def atom_response(identifier):
    return HttpResponse(
        200,
        {},
        (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            f"<id>http://arxiv.org/abs/{identifier}</id><title>Mobility trial</title>"
            "<published>2024-01-03T12:00:00Z</published></entry></feed>"
        ).encode(),
    )


def fail_request(*args):
    pytest.fail("unexpected request")


def test_arxiv_binds_latest_to_exact_version_and_reuses_exact_cache(tmp_path, article):
    urls = []

    def request(url, limit, timeout):
        urls.append(url)
        parts = urlsplit(url)
        if parts.hostname == "export.arxiv.org":
            assert parse_qs(parts.query)["id_list"] == [ARXIV]
            return atom_response(ARXIV + "v2")
        assert url == f"https://arxiv.org/pdf/{ARXIV}v2"
        return HttpResponse(200, {}, article)

    pdf = resolve_pdf(f"https://arxiv.org/abs/{ARXIV}", tmp_path, transport=request)
    assert pdf.read_bytes() == article
    record = json.loads(pdf.with_suffix(".json").read_bytes())
    assert record["identifier"] == ARXIV + "v2"
    assert record["requested_identifier"] == ARXIV
    assert record["version"] == "v2"
    assert record["sha256"] == hashlib.sha256(article).hexdigest()
    assert len(urls) == 2
    assert resolve_pdf(f"arXiv:{ARXIV}v2", tmp_path, transport=fail_request) == pdf


def test_arxiv_rejects_a_different_version_from_metadata_or_download(tmp_path, article):
    def request(url, limit, timeout):
        assert url.startswith("https://export.arxiv.org/")
        return atom_response(ARXIV + "v2")

    with pytest.raises(ReferenceResolutionError, match="different article or version"):
        resolve_pdf(ARXIV + "v1", tmp_path, transport=request)

    def redirected_version(url, limit, timeout):
        if url.startswith("https://export.arxiv.org/"):
            return atom_response(ARXIV + "v1")
        if url.endswith("v1"):
            return HttpResponse(302, {"Location": f"/pdf/{ARXIV}v2"}, b"")
        return HttpResponse(200, {}, article)

    with pytest.raises(ReferenceResolutionError, match="different article or version"):
        resolve_pdf(ARXIV + "v1", tmp_path, transport=redirected_version)
    assert not list(tmp_path.glob("*.pdf"))


def test_doi_uses_unpaywall_and_records_hash_and_version_without_contact(tmp_path, article):
    urls = []

    def request(url, limit, timeout):
        urls.append(url)
        if url.startswith("https://api.unpaywall.org/"):
            parts = urlsplit(url)
            assert parts.path == f"/v2/{DOI}"
            assert parse_qs(parts.query)["email"] == [CONTACT]
            return json_response({
                "doi": DOI, "title": "Weekly mobility", "published_date": "2024-01-01",
                "best_oa_location": {"url_for_pdf": PDF_URL, "version": "acceptedVersion"},
            })
        if url == PDF_URL:
            return HttpResponse(302, {"Location": "/download.pdf?shortlived=credential"}, b"")
        return HttpResponse(200, {}, article)

    pdf = resolve_pdf(f"https://doi.org/{DOI}", tmp_path, transport=request, contact_email=CONTACT)
    sidecar = pdf.with_suffix(".json").read_text(encoding="utf-8")
    metadata = json.loads(sidecar)
    assert metadata["identifier"] == DOI
    assert metadata["version"] == "acceptedVersion"
    assert metadata["pdf_url"] == "https://papers.example.org/download.pdf"
    assert CONTACT not in sidecar and "credential" not in sidecar
    assert pdf.read_bytes() == article and len(urls) == 3
    assert resolve_pdf(f"DOI:{DOI.upper()}", tmp_path, transport=fail_request) == pdf


def test_reference_transport_blocks_disallowed_redirects_and_overlong_metadata(tmp_path):
    urls = []

    def request(url, limit, timeout):
        urls.append(url)
        if url.startswith("https://api.unpaywall.org/"):
            return json_response({"doi": DOI, "best_oa_location": {"url_for_pdf": PDF_URL}})
        assert url == PDF_URL
        return HttpResponse(302, {"Location": "https://127.0.0.1/document.pdf"}, b"")

    with pytest.raises(ReferenceResolutionError, match="public address"):
        resolve_pdf(DOI, tmp_path, transport=request, contact_email=CONTACT)
    assert len(urls) == 2

    def oversized(url, limit, timeout):
        return HttpResponse(200, {}, b"x" * (limit + 1))

    with pytest.raises(ReferenceResolutionError, match="size limit"):
        resolve_pdf(DOI, tmp_path, transport=oversized, contact_email=CONTACT)
    assert list(tmp_path.iterdir()) == []


def test_dns_answers_must_all_be_public(monkeypatch):
    answers = [
        (2, 1, 6, "", ("93.184.215.14", 443)),
        (2, 1, 6, "", ("127.0.0.1", 443)),
    ]
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: answers)
    with pytest.raises(ReferenceResolutionError, match="public address"):
        _public_addresses("public.example.org")
    answers.pop()
    assert _public_addresses("public.example.org") == ["93.184.215.14"]


def make_report(abstract, *evidence):
    return {
        "reviewed_abstract": {"version_id": "selected", "text": abstract},
        "documents": [{"doc_id": "paper", "role": "main"}],
        "evidence": list(evidence),
    }


def evidence(identity, text, section="Results"):
    return {
        "evidence_id": identity, "doc_id": "paper", "kind": "body_text", "text": text,
        "locator": {"page_index": 1, "page_number": 2, "section": section, "bbox": [15, 20, 60, 35]},
    }


def test_citation_separates_publication_access_and_content_for_numbered_reference(tmp_path, article):
    report = make_report(
        "The treatment improved weekly mobility [2].",
        evidence("ref-2", "[2] Read, A. Weekly mobility.", "References"),
        evidence("ref-doi", f"https://doi.org/{DOI}", "References"),
        evidence("ref-3", "[3] 10.1000/uncited", "References"),
    )
    original = deepcopy(report)

    class Model:
        def call(self, stage, material, schema):
            assert stage == "external"
            assert material["reference"]["identifier"] == DOI
            allowed = material["evidence"]
            assert len(allowed) == 1 and allowed[0]["doc_id"] != "paper"
            return schema(support_status="not_supported", reason="The source reports no improvement.",
                          evidence_ids=[allowed[0]["evidence_id"]])

    def request(url, limit, timeout):
        if url.startswith("https://api.unpaywall.org/"):
            assert f"/v2/{DOI}" in url
            return json_response({"doi": DOI, "best_oa_location": {"url_for_pdf": PDF_URL}})
        return HttpResponse(200, {}, article)

    row, = check_external(report, model=Model(), cache_dir=tmp_path, transport=request, contact_email=CONTACT)
    assert (row["citation_status"], row["access_status"], row["support_status"]) == (
        "exists", "accessible", "not_supported",
    )
    assert row["citation_evidence_ids"] == ["ref-doi"]
    assert row["evidence_ids"] == [row["evidence"][0]["evidence_id"]]
    assert row["evidence"][0]["locator"]["page_index"] == 0
    assert report == original


def test_unavailable_full_text_and_unknown_reference_have_distinct_fields(tmp_path):
    def request(url, limit, timeout):
        if f"/v2/{DOI}?" in url:
            return json_response({"doi": DOI, "best_oa_location": None})
        return json_response({}, 404)

    report = make_report(f"Study cited {DOI}. A replication used 10.1000/no-record.")
    first, second = check_external(report, cache_dir=tmp_path, transport=request, contact_email=CONTACT)
    assert (first["citation_status"], first["access_status"], first["support_status"]) == (
        "exists", "no_open_fulltext", "not_checked",
    )
    assert (second["citation_status"], second["access_status"], second["support_status"]) == (
        "not_found", "not_checked", "not_checked",
    )


def test_no_identifiers_do_not_call_network_and_novelty_search_is_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    assert check_external(make_report("This paper measured mobility."), cache_dir=tmp_path,
                          transport=fail_request) == []
    urls = []

    def request(url, limit, timeout):
        urls.append(url)
        query = parse_qs(urlsplit(url).query)
        assert query["per_page"] == ["4"]
        return json_response({"results": [{"display_name": f"Study {index}"} for index in range(7)]})

    row, = check_external(
        make_report("We are the first to test weekly mobility. We are the first to do this in hospitals."),
        cache_dir=tmp_path, transport=request,
    )
    assert row["kind"] == "novelty_search"
    assert row["support_status"] == "not_checked" and len(row["candidates"]) == 4 and len(urls) == 1


def test_omissions_require_designation_and_check_the_selected_abstract():
    report = make_report(
        "There was no improvement in weekly mobility.",
        evidence("method", "The primary outcome was weekly mobility.", "Methods"),
        evidence("result", "Weekly mobility was unchanged."),
        evidence("limit", "Critical limitation: Only one hospital was included.", "Limitations"),
        evidence("ordinary", "The team recruited fifty patients."),
        evidence("reference", "A primary outcome was mortality in another trial.", "References"),
    )
    row, = check_omissions(report)
    assert row["status"] == "possible_omission" and row["evidence_ids"] == ["limit"]
    report["reviewed_abstract"]["text"] = "The program tested a new treatment."
    assert [row["evidence_ids"] for row in check_omissions(report)] == [["method", "result"], ["limit"]]

    class InvalidOmissionModel:
        def call(self, stage, material, schema):
            assert stage == "omissions"
            return schema(omissions=[{"reason": "Unguarded", "evidence_ids": ["reference"]}])

    assert check_omissions(report, model=InvalidOmissionModel()) == []
