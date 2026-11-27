import hashlib

import pytest

from abstract_review.audit import extract_claims, sentences


class Model:
    def __init__(self, quote, proposition, support="original source", counter="alternative source"):
        self.quote = quote
        self.proposition = proposition
        self.support = support
        self.counter = counter
        self.source = None

    def call(self, stage, material, schema):
        assert stage == "extract"
        self.source = material["sentences"]
        return schema.model_validate({"claims": [{
            "sentence_id": self.source[0]["sentence_id"], "quote": self.quote, "proposition": self.proposition,
            "kind": "quantitative", "requirements": ["observed result"],
            "support_query": self.support, "counter_query": self.counter,
        }], "compounds": []})


@pytest.mark.parametrize(("source", "quote", "proposed", "blocker"), [
    ("The new model improved accuracy by 2%.", "improved accuracy by 2%",
     "The new model improved accuracy by 20%.", "quantity_changed"),
    ("The new model improved accuracy by 2%.", "improved accuracy",
     "The new model improved accuracy.", "quantity_changed"),
    ("The new model improved accuracy by 2%.", "improved accuracy by 2%",
     "The new model improved accuracy by 2 percentage points.", "quantity_changed"),
    ("Processing took 5 hours.", "5 hours", "Processing took 5 days.", "quantity_changed"),
    ("The model handles up to 128k tokens.", "128k tokens",
     "The model handles up to 128m tokens.", "quantity_changed"),
    ("The measured offset was −2%.", "−2%", "The measured offset was 2%.", "quantity_changed"),
    ("The new model did not improve accuracy.", "improve accuracy",
     "The new model improved accuracy.", "qualification_changed"),
    ("The effect was observed only in adults.", "observed only in adults",
     "The effect was observed in adults.", "qualification_changed"),
    ("The effect was observed in adults.", "observed in adults",
     "The effect was observed in children.", "qualification_changed"),
    ("Accuracy increased by 2%.", "increased by 2%", "Accuracy decreased by 2%.", "direction_changed"),
    ("准确率提高了2%。", "提高了2%", "准确率提高了20%。", "quantity_changed"),
])
def test_ungrounded_changes_recover_the_original_assertion(source, quote, proposed, blocker):
    model = Model(quote, proposed, support=proposed, counter=proposed)
    claim, = extract_claims(source, model)
    assert claim["proposition"] == claim["quote"] == source
    assert source[claim["start"]:claim["end"]] == claim["quote"]
    assert claim["support_query"] == source and claim["counter_query"].endswith(source)
    assert claim["extraction_guard"]["status"] == "source_fallback"
    assert blocker in claim["extraction_guard"]["blocker_codes"]


def test_supported_unit_alias_preserves_proposition_and_rebuilds_ungrounded_search_query():
    source = "Accuracy increased by 2.0 percent in adults."
    quote = "increased by 2.0 percent in adults"
    proposition = "Accuracy increased by 2% in adults."
    model = Model(quote, proposition, support="accuracy change 20%")
    claim, = extract_claims(source, model)
    expected_id = hashlib.sha256(f"{source.index(quote)}:{quote}:{proposition}".encode()).hexdigest()[:20]
    assert claim["claim_id"] == expected_id and claim["proposition"] == proposition
    assert claim["support_query"] == proposition and claim["counter_query"].endswith(proposition)
    assert claim["extraction_guard"] == {"status": "query_rebuilt", "blocker_codes": ["ungrounded_query_quantity"]}


@pytest.mark.parametrize("gap", [" ", "\n"])
def test_detached_terminal_marker_is_filtered_without_changing_raw_sentence_view(gap):
    first = "The analysis also has implications for theoretical linguistics."
    raw = f"{first}{gap}1"
    standalone = extract_claims(first)[0]
    assert [part["text"] for part in sentences(raw)] == [first, "1"]
    assert extract_claims(raw) == [standalone]
    model = Model(first, first)
    claim, = extract_claims(raw, model)
    assert [part["text"] for part in model.source] == [first]
    assert claim["claim_id"] == standalone["claim_id"]
    assert raw[claim["start"]:claim["end"]] == first


def test_standalone_or_nonterminal_numbers_are_not_reinterpreted_as_footnotes():
    assert [claim["quote"] for claim in extract_claims("1")] == ["1"]
    assert [claim["quote"] for claim in extract_claims("The system stopped.\n1\nThe second system continued.")] == [
        "The system stopped.", "1", "The second system continued.",
    ]
