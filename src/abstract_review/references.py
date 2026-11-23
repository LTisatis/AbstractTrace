"""Explicit open-access resolution and evidence-scoped optional checks."""

import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Literal
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import pymupdf

from .audit import blind_evidence, sentences
from .evidence import EvidenceIndex
from .ingest import MAX_PDF_BYTES, MAX_PDF_PAGES, parse_pdf
from .models import ModelPaused, ModelUnavailable, Schema


_DOI = r"10\.\d{4,9}/[^\s<>\"'\[\]{}]+"
_ARXIV_ID = r"(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}|[a-z][a-z.\-]*/\d{7})(?:v[1-9][0-9]*)?"
_CITATION = re.compile(rf"{_DOI}|(?:arxiv\s*:\s*|https://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/){_ARXIV_ID}", re.I)
_NUMBERED = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_BIB_ENTRY = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+")
_BIB_SECTION = re.compile(r"\b(?:references|bibliography)\b|参考文献", re.I)
_NOVELTY = re.compile(r"\bfirst\s+(?:to|study|work|report|evidence)\b|首次|首项|首个|率先", re.I)
_PRIMARY = re.compile(
    r"(?:\b(?:primary (?:outcome|endpoint|result|finding)|main (?:result|finding)|critical limitation)\b"
    r"|主要(?:结果|终点|结局)|核心发现|关键局限)(?:\s+(?:was|were|is|of|showed|indicated)\b)?[\s:：]*",
    re.I,
)
_RESULT_SECTION = re.compile(r"results?|conclusions?|discussion|limitations?|结果|结论|讨论|局限", re.I)
_ARXIV_HOSTS = frozenset(("arxiv.org", "www.arxiv.org", "export.arxiv.org"))
_METADATA_BYTES = 1024 * 1024
_MAX_REFERENCE_CHECKS = 6
_MAX_REDIRECTS = 3
_NETWORK_SECONDS = 30
_ARXIV_API_LOCK = Lock()
_arxiv_request_started = float("-inf")


class ReferenceResolutionError(ValueError):
    """An external source cannot be resolved as requested."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


type Transport = Callable[[str, int, float], HttpResponse]


@dataclass(frozen=True, slots=True)
class _Reference:
    provider: Literal["unpaywall", "arxiv"]
    identifier: str


@dataclass(frozen=True, slots=True)
class _Resolution:
    citation_status: str
    access_status: str
    metadata: dict
    reason: str
    path: Path | None = None


class _ExternalResult(Schema):
    support_status: Literal["supported", "not_supported", "unclear"]
    reason: str
    evidence_ids: list[str]


class _Omission(Schema):
    reason: str
    evidence_ids: list[str]
    anchor_sentence_id: str | None = None


class _OmissionResult(Schema):
    omissions: list[_Omission]


def _reference(source: str) -> _Reference:
    value = source.strip()
    if len(value) > 400:
        raise ReferenceResolutionError("Reference identifier exceeds the size limit.")
    try:
        parts = urlsplit(value) if value.lower().startswith(("https://", "http://")) else None
    except ValueError as exc:
        raise ReferenceResolutionError("The reference link is malformed.") from exc
    if parts is not None:
        host = parts.hostname
        if parts.username is not None:
            raise ReferenceResolutionError("The reference link is malformed.")
        if host in {"doi.org", "dx.doi.org"}:
            value = unquote(parts.path).lstrip("/")
        elif host in _ARXIV_HOSTS and re.match(r"^/(abs|pdf|html)/", parts.path):
            value = unquote(parts.path).split("/", 2)[-1].removesuffix(".pdf")
        else:
            raise ReferenceResolutionError("Expected a DOI or arXiv identifier or canonical link.")
    value = re.sub(r"^(doi|arxiv)\s*:\s*", "", value, flags=re.I)
    if re.fullmatch(_DOI, value, re.I) and not {".", ".."}.intersection(value.split("/")):
        return _Reference("unpaywall", value.lower())
    if re.fullmatch(_ARXIV_ID, value, re.I):
        return _Reference("arxiv", value.lower())
    raise ReferenceResolutionError("Expected a DOI or arXiv identifier or canonical link.")


def _public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _validate_url(url: str, hosts: frozenset[str] | None = None):
    if len(url) > 4096 or "\\" in url or any(character.isspace() or ord(character) < 32 for character in url):
        raise ReferenceResolutionError("External URL is malformed.")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ReferenceResolutionError("External URL is malformed.") from exc
    if parts.scheme != "https" or parts.username is not None or not host or port not in {None, 443}:
        raise ReferenceResolutionError("External downloads require public HTTPS.")
    if hosts is not None and host not in hosts:
        raise ReferenceResolutionError("The external provider redirected outside its allowed hosts.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.rstrip(".").endswith((".localhost", ".local", ".internal")):
            raise ReferenceResolutionError("External downloads require a public address.")
    else:
        if not _public(address):
            raise ReferenceResolutionError("External downloads require a public address.")
    return parts


def _public_addresses(host: str) -> list[str]:
    addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
    if not addresses or any(not _public(ipaddress.ip_address(address)) for address in addresses):
        raise ReferenceResolutionError("External downloads require a public address.")
    return addresses


class _PinnedConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, timeout: float) -> None:
        self._tls = ssl.create_default_context()
        super().__init__(host, port=443, timeout=timeout, context=self._tls)
        self._address = address

    def connect(self) -> None:
        # Pin the validated address while preserving the HTTPS host and TLS verification.
        raw = socket.create_connection((self._address, 443), self.timeout)
        try:
            self.sock = self._tls.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def _request_once(url: str, limit: int, timeout: float) -> HttpResponse:
    deadline = time.monotonic() + timeout
    parts = urlsplit(url)
    host = parts.hostname.encode("idna").decode("ascii")
    addresses = _public_addresses(host)
    path = urlunsplit((
        "", "", quote(parts.path or "/", safe="/%:;@()!$&'*,+~=-_."),
        quote(parts.query, safe="=&?/:;+,%@$()!*'~.-_"), "",
    ))
    for address in addresses[:3]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        connection = _PinnedConnection(host, address, remaining)
        try:
            connection.request("GET", path, headers={"User-Agent": "AbstractTrace/0.1", "Accept-Encoding": "identity"})
            with connection.getresponse() as response:
                headers = {name.lower(): value for name, value in response.getheaders()}
                if response.status != 200:
                    return HttpResponse(response.status, headers, b"")
                length = headers.get("content-length", "")
                if length.isdecimal() and int(length) > limit:
                    raise ReferenceResolutionError("External response exceeds the size limit.")
                body = bytearray()
                while len(body) <= limit:
                    if time.monotonic() >= deadline:
                        raise ReferenceResolutionError("External request exceeded the time budget.")
                    chunk = response.read1(min(65536, limit + 1 - len(body)))
                    if not chunk:
                        return HttpResponse(response.status, headers, bytes(body))
                    body.extend(chunk)
                raise ReferenceResolutionError("External response exceeds the size limit.")
        except (OSError, http.client.HTTPException):
            continue
        finally:
            connection.close()
    raise ReferenceResolutionError("External HTTPS request failed.")


def _request(url: str, limit: int, timeout: float) -> HttpResponse:
    if urlsplit(url).hostname != "export.arxiv.org":
        return _request_once(url, limit, timeout)
    # The arXiv legacy API permits one connection and one request every three seconds.
    started = time.monotonic()
    if not _ARXIV_API_LOCK.acquire(timeout=timeout):
        raise ReferenceResolutionError("arXiv request exceeded the time budget.")
    global _arxiv_request_started
    try:
        now = time.monotonic()
        delay = max(0, _arxiv_request_started + 3 - now)
        remaining = timeout - (now - started) - delay
        if remaining <= 0:
            raise ReferenceResolutionError("arXiv request exceeded the time budget.")
        if delay:
            time.sleep(delay)
        _arxiv_request_started = time.monotonic()
        return _request_once(url, limit, remaining)
    finally:
        _ARXIV_API_LOCK.release()


def _fetch(url: str, limit: int, deadline: float, transport: Transport, *, hosts=None) -> tuple[HttpResponse, str]:
    for _ in range(_MAX_REDIRECTS + 1):
        _validate_url(url, hosts)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReferenceResolutionError("External request exceeded the time budget.")
        try:
            response = transport(url, limit, min(remaining, 10))
        except (OSError, http.client.HTTPException, UnicodeError) as exc:
            raise ReferenceResolutionError("External HTTPS request failed.") from exc
        if time.monotonic() >= deadline:
            raise ReferenceResolutionError("External request exceeded the time budget.")
        if len(response.body) > limit:
            raise ReferenceResolutionError("External response exceeds the size limit.")
        if response.status not in {301, 302, 303, 307, 308}:
            return response, url
        location = next((value for name, value in response.headers.items() if name.lower() == "location"), None)
        if not location:
            raise ReferenceResolutionError("External redirect has no location.")
        url = urljoin(url, location)
    raise ReferenceResolutionError("External redirect limit exceeded.")


def _json(body: bytes) -> dict:
    try:
        document = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise ReferenceResolutionError("External provider returned malformed metadata.") from exc
    if not isinstance(document, dict):
        raise ReferenceResolutionError("External provider returned malformed metadata.")
    return document


def _arxiv_metadata(reference: _Reference, deadline: float, transport: Transport) -> tuple[dict, list[tuple[str, str]]]:
    params = urlencode({"id_list": reference.identifier, "max_results": 1})
    response, _ = _fetch(
        "https://export.arxiv.org/api/query?" + params, _METADATA_BYTES, deadline, transport,
        hosts=frozenset(("export.arxiv.org",)),
    )
    if response.status == 404:
        return {}, []
    if response.status != 200:
        raise ReferenceResolutionError(f"arXiv metadata returned HTTP {response.status}.")
    try:
        feed = ElementTree.fromstring(response.body)
    except ElementTree.ParseError as exc:
        raise ReferenceResolutionError("arXiv returned malformed metadata.") from exc
    atom = {"a": "http://www.w3.org/2005/Atom"}
    entry = feed.find("a:entry", atom)
    if entry is None:
        return {}, []
    entry_identifier = _reference(entry.findtext("a:id", "", atom))
    identifier = entry_identifier.identifier
    actual = re.fullmatch(rf"(?P<base>{_ARXIV_ID})v(?P<version>[1-9][0-9]*)", identifier)
    if entry_identifier.provider != "arxiv" or actual is None:
        raise ReferenceResolutionError("arXiv did not confirm an exact article version.")
    if reference.identifier not in {identifier, actual.group("base")}:
        raise ReferenceResolutionError("arXiv returned a different article or version.")
    metadata = {
        "provider": "arxiv", "identifier": identifier, "requested_identifier": reference.identifier,
        "title": " ".join(entry.findtext("a:title", "", atom).split()),
        "published": entry.findtext("a:published", "", atom),
        "version": "v" + actual.group("version"),
    }
    return metadata, [(f"https://arxiv.org/pdf/{identifier}", metadata["version"])]


def _doi_metadata(reference: _Reference, email: str | None, deadline: float, transport: Transport) -> tuple[dict, list[tuple[str, str | None]]]:
    contact = email if email is not None else os.environ.get("UNPAYWALL_EMAIL")
    if not contact or not re.fullmatch(r"[^@\s]+@[^@\s]+", contact):
        raise ReferenceResolutionError("Set UNPAYWALL_EMAIL to resolve DOI metadata.")
    identifier = quote(reference.identifier, safe="/")
    response, _ = _fetch(
        f"https://api.unpaywall.org/v2/{identifier}?" + urlencode({"email": contact}),
        _METADATA_BYTES, deadline, transport, hosts=frozenset(("api.unpaywall.org",)),
    )
    if response.status == 404:
        return {}, []
    if response.status != 200:
        raise ReferenceResolutionError(f"Unpaywall metadata returned HTTP {response.status}.")
    record = _json(response.body)
    returned = record.get("doi")
    if not isinstance(returned, str) or returned.lower() != reference.identifier:
        raise ReferenceResolutionError("Unpaywall returned a different DOI.")
    metadata = {key: record[key] for key in ("title", "published_date", "journal_name") if isinstance(record.get(key), str)}
    metadata.update(provider="unpaywall", identifier=reference.identifier)
    other_locations = record.get("oa_locations")
    locations = [record.get("best_oa_location"), *(other_locations if isinstance(other_locations, list) else [])]
    urls = {}
    for location in locations:
        if isinstance(location, dict) and isinstance(location.get("url_for_pdf"), str):
            url = location["url_for_pdf"]
            version = location.get("version")
            urls.setdefault(url, version if isinstance(version, str) else None)
    return metadata, list(urls.items())[:2]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_key(reference: _Reference) -> str:
    return _digest(f"{reference.provider}/{reference.identifier}".encode())


def _cached(reference: _Reference, cache_dir: Path) -> _Resolution | None:
    for sidecar in sorted(cache_dir.glob(_cache_key(reference) + "-*.json")):
        pdf = sidecar.with_suffix(".pdf")
        try:
            with sidecar.open("rb") as stream:
                metadata = _json(stream.read(_METADATA_BYTES))
            if metadata.get("identifier") != reference.identifier or metadata.get("provider") != reference.provider:
                continue
            with pdf.open("rb") as stream:
                data = stream.read(MAX_PDF_BYTES + 1)
            if len(data) <= MAX_PDF_BYTES and _digest(data) == metadata.get("sha256"):
                return _Resolution("exists", "accessible", metadata, "Found the verified cached PDF.", pdf)
        except (OSError, ReferenceResolutionError):
            continue
    return None


def _publish(path: Path, data: bytes) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _check_pdf(data: bytes) -> None:
    if b"%PDF-" not in data[:1024]:
        raise ReferenceResolutionError("The external response did not contain a PDF.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if not pdf.is_pdf or pdf.needs_pass or pdf.page_count < 1:
                raise ReferenceResolutionError("The external PDF is not readable.")
            if pdf.page_count > MAX_PDF_PAGES:
                raise ReferenceResolutionError("The external PDF exceeds the page limit.")
    except (pymupdf.mupdf.FzErrorBase, RuntimeError) as exc:
        raise ReferenceResolutionError("The external PDF is not readable.") from exc


def _reported_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _download(reference: _Reference, url: str, deadline: float, transport: Transport) -> tuple[bytes, str]:
    response, final_url = _fetch(
        url, MAX_PDF_BYTES, deadline, transport, hosts=_ARXIV_HOSTS if reference.provider == "arxiv" else None,
    )
    if response.status != 200:
        raise ReferenceResolutionError(f"Open PDF download returned HTTP {response.status}.")
    if reference.provider == "arxiv" and _reference(final_url) != reference:
        raise ReferenceResolutionError("arXiv redirected the PDF to a different article or version.")
    _check_pdf(response.body)
    return response.body, _reported_url(final_url)


def _resolve(reference: _Reference, cache_dir: Path, email: str | None, deadline: float, transport: Transport) -> _Resolution:
    label = {"provider": reference.provider, "identifier": reference.identifier}
    if cached := _cached(reference, cache_dir):
        return cached
    try:
        if reference.provider == "arxiv":
            metadata, urls = _arxiv_metadata(reference, deadline, transport)
        else:
            metadata, urls = _doi_metadata(reference, email, deadline, transport)
    except ReferenceResolutionError as exc:
        return _Resolution("unverified", "not_checked", label, str(exc))
    if not metadata:
        return _Resolution("not_found", "not_checked", label, "The external provider found no matching record.")
    if not urls:
        return _Resolution("exists", "no_open_fulltext", metadata, "The provider supplied no open PDF location.")
    canonical = _Reference(reference.provider, metadata["identifier"])
    if canonical != reference and (cached := _cached(canonical, cache_dir)):
        return cached
    reason = "Open PDF download failed."
    for url, version in urls:
        try:
            data, final_url = _download(canonical, url, deadline, transport)
        except ReferenceResolutionError as exc:
            reason = str(exc)
            continue
        sha256 = _digest(data)
        identified = {
            **metadata, "version": version, "sha256": sha256, "pdf_url": final_url,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        pdf = cache_dir / f"{_cache_key(canonical)}-{sha256}.pdf"
        _publish(pdf, data)
        _publish(pdf.with_suffix(".json"), json.dumps(identified, ensure_ascii=False, indent=2).encode("utf-8"))
        return _Resolution("exists", "accessible", identified, "Downloaded the provider's open PDF.", pdf)
    return _Resolution("exists", "unavailable", metadata, reason)


def resolve_pdf(
    source: str, cache_dir: Path, *, transport: Transport | None = None, contact_email: str | None = None,
) -> Path:
    """Resolve a DOI or arXiv ID to a cached local PDF and a same-stem JSON sidecar."""
    result = _resolve(_reference(source), cache_dir, contact_email, time.monotonic() + _NETWORK_SECONDS,
                      transport or _request)
    if result.path is None:
        raise ReferenceResolutionError(result.reason)
    return result.path


def _identifiers(text: str) -> list[_Reference]:
    references = {}
    for match in _CITATION.finditer(text):
        mention = match.group().rstrip(".,;:")
        while mention.endswith(")") and mention.count(")") > mention.count("("):
            mention = mention[:-1]
        try:
            reference = _reference(mention)
        except ReferenceResolutionError:
            continue
        references[reference] = None
    return list(references)


def _primary_evidence(report: dict, *, include_supplements: bool = False) -> list[dict]:
    roles = {"main", "supplement"} if include_supplements else {"main"}
    scope = {doc["doc_id"] for doc in report.get("documents", []) if doc.get("role") in roles}
    return [item for item in report.get("evidence", []) if item.get("doc_id") in scope]


def _bibliography(report: dict) -> dict[str, dict[_Reference, list[str]]]:
    references: dict[str, dict[_Reference, list[str]]] = {}
    number = None
    for item in _primary_evidence(report):
        section = item.get("locator", {}).get("section", "")
        if not _BIB_SECTION.search(section):
            number = None
            continue
        text = item["text"]
        if match := _BIB_ENTRY.match(text):
            number = match.group(1) or match.group(2)
        if number is not None:
            listed = references.setdefault(number, {})
            for reference in _identifiers(text):
                ids = listed.setdefault(reference, [])
                if item["evidence_id"] not in ids:
                    ids.append(item["evidence_id"])
    return references


def _attributions(report: dict) -> list[tuple[dict, _Reference, list[str]]]:
    bibliography = _bibliography(report)
    text = report.get("reviewed_abstract", {}).get("text", "")
    attributions = []
    for sentence in sentences(text):
        cited = {reference: [] for reference in _identifiers(sentence["text"])}
        for group in _NUMBERED.findall(sentence["text"]):
            for number in group.replace(" ", "").split(","):
                for reference, ids in bibliography.get(number, {}).items():
                    cited.setdefault(reference, []).extend(ids)
        attributions.extend((sentence, reference, list(dict.fromkeys(ids))) for reference, ids in cited.items())
    return attributions


def _content_check(row: dict, path: Path, model, archive_pdf=None) -> None:
    try:
        parsed = parse_pdf(path, role="external")
    except (OSError, ValueError) as exc:
        row.update(access_status="unavailable", reason=f"Could not read the open PDF: {exc}")
        return
    available = [
        item for item in parsed.evidence
        if not _BIB_SECTION.search(item["locator"].get("section", ""))
    ]
    if not available:
        row.update(access_status="unreadable", reason="The open PDF has no readable content evidence.")
        return
    query = _NUMBERED.sub(" ", _CITATION.sub(" ", row["attribution"]["text"]))
    index = EvidenceIndex(available)
    try:
        candidates = index.search(query, limit=8)["evidence"]
    finally:
        index.close()
    row["evidence"] = blind_evidence(candidates)
    if not candidates:
        row["reason"] = "The open PDF is readable, but no relevant passage was located."
        return
    if archive_pdf is not None:
        sha256 = parsed.document["sha256"]
        row["document"] = {**parsed.document, "blob_path": archive_pdf(parsed.data, sha256)}
    if model is None:
        row["reason"] = "The open PDF is readable and candidate passages are available for review."
        return
    material = {
        "attribution": row["attribution"]["text"], "reference": row["reference"],
        "evidence": row["evidence"],
    }
    try:
        result = model.call("external", material, _ExternalResult)
    except ModelPaused:
        raise
    except ModelUnavailable:
        row["reason"] = "The external evidence classifier was unavailable."
        return
    ids = list(dict.fromkeys(result.evidence_ids))
    allowed = {item["evidence_id"] for item in candidates}
    if not allowed.issuperset(ids) or (result.support_status != "unclear" and not ids):
        row["reason"] = "The classifier did not supply valid evidence for its decision."
        return
    row.update(support_status=result.support_status, reason=result.reason, evidence_ids=ids)


def _search_novelty(report: dict, transport: Transport, deadline: float) -> dict | None:
    parts = sentences(report.get("reviewed_abstract", {}).get("text", ""))
    sentence = next((part for part in parts if _NOVELTY.search(part["text"])), None)
    if sentence is None:
        return None
    query = " ".join(_NUMBERED.sub(" ", _CITATION.sub(" ", sentence["text"])).split())[:220]
    params = {"search": query, "per_page": "4", "select": "id,doi,display_name,publication_date"}
    if key := os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = key
    result = {
        "kind": "novelty_search", "attribution": sentence, "citation_status": "unverified",
        "access_status": "not_checked", "support_status": "not_checked", "evidence_ids": [],
        "evidence": [], "reference": {"provider": "openalex"}, "query": query,
        "searched_at": datetime.now(UTC).isoformat(timespec="seconds"), "candidates": [],
    }
    try:
        response, _ = _fetch(
            "https://api.openalex.org/works?" + urlencode(params), _METADATA_BYTES, deadline, transport,
            hosts=frozenset(("api.openalex.org",)),
        )
        if response.status != 200:
            raise ReferenceResolutionError(f"OpenAlex search returned HTTP {response.status}.")
        matches = _json(response.body).get("results")
        if not isinstance(matches, list):
            raise ReferenceResolutionError("OpenAlex returned malformed results.")
    except ReferenceResolutionError as exc:
        result["reason"] = str(exc)
    else:
        result["candidates"] = [
            {field: match.get(field) for field in ("id", "doi", "display_name", "publication_date")}
            for match in matches[:4] if isinstance(match, dict)
        ]
        result["reason"] = "A one-query metadata search cannot establish scientific priority."
    return result


def check_external(
    report: dict, *, model=None, cache_dir: Path, transport: Transport | None = None,
    contact_email: str | None = None, archive_pdf=None,
) -> list[dict]:
    """Check cited identifiers and optionally search for prior work on novelty claims."""
    request = transport or _request
    deadline = time.monotonic() + _NETWORK_SECONDS
    checks = []
    resolved: dict[_Reference, _Resolution] = {}
    for sentence, reference, citation_ids in _attributions(report)[:_MAX_REFERENCE_CHECKS]:
        if reference not in resolved:
            resolved[reference] = _resolve(reference, cache_dir, contact_email, deadline, request)
        result = resolved[reference]
        row = {
            "kind": "explicit_citation", "attribution": sentence, "reference": result.metadata,
            "citation_status": result.citation_status, "access_status": result.access_status,
            "support_status": "not_checked", "reason": result.reason,
            "citation_evidence_ids": citation_ids, "evidence_ids": [], "evidence": [],
        }
        if result.path is not None:
            row["pdf_file"] = result.path.name
            _content_check(row, result.path, model, archive_pdf)
        checks.append(row)
    if novelty := _search_novelty(report, request, deadline):
        checks.append(novelty)
    return checks


def _omission_candidates(report: dict) -> list[tuple[list[dict], str]]:
    body = [
        item for item in _primary_evidence(report, include_supplements=True)
        if item.get("kind") == "body_text"
        and not _BIB_SECTION.search(item.get("locator", {}).get("section", ""))
    ]
    results = [item for item in body if _RESULT_SECTION.search(item.get("locator", {}).get("section", ""))]
    candidates = []
    seen = set()
    for item in body:
        match = _PRIMARY.search(item["text"])
        if match is None:
            continue
        phrase = item["text"][match.end():].strip(" :：.")
        if len(phrase) < 8:
            continue
        if item in results:
            items = [item]
        else:
            # A named method endpoint needs an independently located result.
            outcome = next((row for row in results if phrase.casefold() in row["text"].casefold()), None)
            if outcome is None:
                continue
            items = [item, outcome]
        result_id = items[-1]["evidence_id"]
        if result_id not in seen:
            seen.add(result_id)
            candidates.append((items, phrase))
        if len(candidates) >= 8:
            break
    return candidates


def check_omissions(report: dict, model=None) -> list[dict]:
    """Check explicitly designated findings or limitations against the selected abstract."""
    abstract = report.get("reviewed_abstract", {}).get("text", "")
    if not abstract:
        return []
    candidates = _omission_candidates(report)
    if model is None:
        normalized = " ".join(abstract.casefold().split())
        return [
            {
                "status": "possible_omission", "evidence_ids": [item["evidence_id"] for item in items],
                "reason": "The paper explicitly designates this finding or limitation. Its wording was not "
                          "found in the selected abstract. Check whether the abstract paraphrases it.",
            }
            for items, phrase in candidates if " ".join(phrase.casefold().split()) not in normalized
        ]
    evidence = {item["evidence_id"]: item for items, _ in candidates for item in items}
    if not evidence:
        return []
    parts = sentences(abstract)
    sentence_ids = {part["sentence_id"] for part in parts}
    try:
        result = model.call("omissions", {"abstract": abstract, "abstract_sentences": parts,
                                         "evidence": blind_evidence(list(evidence.values()))}, _OmissionResult)
    except ModelPaused:
        raise
    except ModelUnavailable:
        return []
    return [
        {"status": "possible_omission", "evidence_ids": list(dict.fromkeys(item.evidence_ids)), "reason": item.reason,
         "anchor_sentence_id": item.anchor_sentence_id if item.anchor_sentence_id in sentence_ids else None}
        for item in result.omissions if item.evidence_ids and evidence.keys() >= set(item.evidence_ids)
    ]
