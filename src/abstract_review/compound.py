"""Ground compound operands in their source and combine their independent findings."""

import hashlib
import json
import re


_OR = re.compile(r"\bor\b|或者|或", re.I)
_AND = re.compile(r"\band\b|并且|且|以及|和|及|与", re.I)
_UNSAFE_LOGIC = re.compile(
    r"\b(?:neither|nor|unless|not\s+(?:both|all|either)|or\s+rather)\b|并非|未必都|不能同时|不同时成立", re.I,
)
_POOLED = re.compile(r"\b(?:pooled|combined|aggregate|average|mean|overall|joint|may|might|could|whether)\b|合并|汇总|平均|总体|可能|预计", re.I)
_NEGATED_SHARED = re.compile(r"\b(?:no|not|never)\b|未|没有|不曾|无(?:变化|增加|提升|下降|提高)", re.I)
_POINT_CHANGE = re.compile(r"\d+(?:\.\d+)?\s*(?:percentage points?\b|个?百分点)", re.I)
_EN_DATASETS = re.compile(
    r"(?P<join>\b(?:on|across|in|for)\s+)(?:(?:the|all|both|either|two|three|four|five|\d+)\s+)*"
    r"data\s*sets?\s*:?[ \t]+(?P<items>[^.!?;]+?)\s*(?P<terminal>[.!?]?)$", re.I,
)
_CN_ITEMS = r"[A-Za-z0-9_\-甲乙丙丁戊己庚辛壬癸\s、，,和与及或以者]+?"
_CN_DATASETS = re.compile(rf"数据集\s*(?P<items>{_CN_ITEMS})(?=\s*(?:的|上|中)[^。！？]*[。！？]?$)")
_CN_LIST_FIRST = re.compile(rf"(?P<items>{_CN_ITEMS})\s*(?:[0-9两三四五六七八九十]+个)?数据集(?=\s*(?:上|中|的))")
_EN_NAME = r"(?:data\s*set\s+)?([A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)"
_CN_NAME = r"(?:数据集\s*)?([A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*|[甲乙丙丁戊己庚辛壬癸])"
_LIST_JOIN = re.compile(r"\s*(,\s*(?:and\b|or\b)?|and\b|or\b|[、，]|以及|或者|或|和|与|及)\s*", re.I)
_CLAUSE_JOIN = re.compile(r"\s*(?:[;,，；]\s*(?:and\b|or\b|并且|且|或者|或)?|\b(?:and|or)\b|并且|且|或者|或)\s*", re.I)
_EN_CLAUSE = re.compile(
    r"^(?:either\s+)?[A-Za-z0-9][A-Za-z0-9_()'./-]*(?:\s+[A-Za-z0-9][A-Za-z0-9_()'./-]*){0,6}\s+"
    r"(?:is|are|was|were|has|have|had|increas\w*|improv\w*|reduc\w*|decreas\w*|dropp?\w*|"
    r"rais\w*|fell|remain\w*|reach\w*|achiev\w*|report\w*|yield\w*)\b", re.I,
)
_CN_CLAUSE = re.compile(r"^[^，；。！？]{2,35}(?:准确率|召回率|精确率|误差率|延迟|吞吐量|覆盖率|性能).{0,8}(?:为|是|提高|增加|下降|降低|达到|保持)")
_PROCEDURE_SENTENCE = re.compile(
    r"(?P<subject>we)[ \t]+(?P<left>[^,.!?;:]+?)(?P<comma>[ \t]*,)?[ \t]+and[ \t]+"
    r"(?P<right>[^,.!?;:]+)(?P<terminal>\.?)", re.I,
)
_PROCEDURE_ACTION = re.compile(
    r"(?P<adverb>(?:automatically|manually)[ \t]+)?"
    r"(?P<verb>accepted|applied|approved|classified|collected|corrected|counted|excluded|extracted|"
    r"inspected|measured|recorded|recruited|rejected|reviewed|screened|selected|tested|trained|used)"
    r"[ \t]+(?P<object>[^\W_][\w \t'’–-]*)", re.I,
)
_PROCEDURE_SCOPE = re.compile(
    r"\b(?:after|although|and|assuming|before|because|both|but|can|could|either|except|if|jointly|"
    r"may|might|must|neither|nor|once|only|or|provided|respectively|shall|should|since|than|that|then|"
    r"together|unless|until|how|when|where|whereas|whether|which|while|who|whom|whose|will|would)\b", re.I,
)
_TRAILING_PROCEDURE_SCOPE = re.compile(
    r"\b(?:as|at|automatically|by|during|for|from|in|into|manually|on|over|through|to|under|using|via|"
    r"with|within|without)\b", re.I,
)
_REVIEWLESS_ADJUDICATION = re.compile(
    r"(?P<action>(?:accepted|approved|rejected)[ \t]+.+)[ \t]+without[ \t]+manual[ \t]+review", re.I,
)
_PROCEDURE_COMPLEMENT_VERBS = {"accepted", "measured", "recorded", "reviewed", "tested"}
_DETERMINED_PROCEDURE_OBJECT = re.compile(
    r"(?:a|an|the|all|each|every|any|my|our|its|their|these|those|multiple|several|\d+)[ \t]+\S", re.I,
)
_APPLIED_DETECTOR = re.compile(r"[\w-]+[ \t]+(?:to detect|for (?:the )?detection of)\b", re.I)
_MANUAL_CORRECTION = re.compile(r"manually[ \t]+corrected[ \t]+false[ \t]+positives", re.I)
_REPORTING_PREFIX = re.compile(
    r"(?P<prefix>we\s+(?:also\s+)?(?:demonstrate|(?:fi|ﬁ)nd|observe|report|show)"
    r"(?:\s+|\s*—[^—.!?]+—\s*)that\s+)", re.I,
)
_REPORTED_PREDICATE = re.compile(
    r"(?P<subject>[\w’'-]+(?:\s+[\w’'-]+){0,7}?)\s+"
    r"(?:(?:also|better|further|significantly)\s+)?"
    r"(?:acts?|aligns?|changes?|causes?|decreases?|functions?|improves?|increases?|leads?|prevents?|reduces?|regularizes?|"
    r"results?|serves?|worsens?)\s+\S", re.I,
)
_LOCAL_ROLE = re.compile(r"(?P<subject>[\w-]+)\s+(?:acts|functions|serves)\s+as\s+(?:an?\s+)?[\w-]+(?:\s+[\w-]+){0,3}", re.I)
_ROLE_LINK = re.compile(r"\b(?:against|at|by|for|from|in|of|on|to|via|with)\b", re.I)
_REMOVAL = re.compile(r"^(?P<action>ablating|disabling|omitting|removing)\s+it\b", re.I)
_REPORT_JOIN = re.compile(r",\s+and\s+|\s+and\s+", re.I)
_LIMITED_REPORTED_SCOPE = re.compile(r"\b(?:although|and|because|but|either|if|neither|nor|or|since|unless|whereas|while)\b", re.I)
_NON_ASSERTED_REPORT = re.compile(r"\b(?:deny|doubt|if|question|whether)\b", re.I)
_TRAILING_CLAUSE_SCOPE = re.compile(r",\s*(?:when|where|provided|unless|under)\b", re.I)
_MEASUREMENT = r"\d+(?:\.\d+)?\s*%"
_CHANGE = r"\d+(?:\.\d+)?\s*(?:%|percentage points?)"
_ABSOLUTE_AND_RELATIVE = re.compile(
    r"(?:(?P<evaluation>we\s+(?:assess|evaluate|test)\s+[^,;.!?]+),\s+and\s+)?"
    r"(?P<absolute>(?P<subjects>[^,;.!?]+?)\s+(?P<verb>achieve|attain|obtain|record)\s+"
    rf"(?:(?:an?\s+)?(?:average|mean)\s+of\s+)?{_MEASUREMENT}\s+and\s+{_MEASUREMENT}\s+"
    r"(?P<metric>[\w’'-]+(?:\s+[\w’'-]+){0,5}?)\s+respectively),\s+"
    rf"(?P<relative>(?P<delta>(?:an?\s+)?{_CHANGE}\s+and\s+{_CHANGE}\s+"
    r"(?:gains?|improvements?|increases?))\s+(?P<baseline>(?:compared\s+(?:to|with)|over)\s+[^,;.!?]+))"
    r"[.!?]?", re.I,
)
_SHARED_EXCEPTION = re.compile(
    r"(?P<prefix>[^;!?]+?\bcannot\s+(?:capture|compute|detect|express|learn|model|predict|recognize|represent)\s+)"
    r"(?P<left>[\w-]+(?:\s+[\w-]+){0,9}),\s+"
    r"(?P<right_anchor>nor\s+(?P<right>[\w-]+(?:\s+[\w-]+){0,9})),\s+"
    r"(?P<condition>unless\s+[^,;.!?]+)[.!?]?", re.I,
)


def _dataset_names(value: str, name_pattern: str) -> tuple[list[tuple[int, int, str]], str] | None:
    matcher = re.compile(name_pattern, re.I)
    names = []
    connectors = []
    offset = len(value) - len(value.lstrip())
    while name := matcher.match(value, offset):
        if name[1].casefold() in {"and", "or", "dataset", "datasets", "the", "all", "either", "others"}:
            return None
        names.append((*name.span(1), name[1]))
        offset = name.end()
        if not value[offset:].strip():
            break
        if not (connector := _LIST_JOIN.match(value, offset)):
            return None
        connectors.append(connector[1])
        offset = connector.end()
    if value[offset:].strip() or len(names) < 2 or len(connectors) != len(names) - 1:
        return None
    if len({name.casefold() for _, _, name in names}) != len(names):
        return None
    separators = " ".join(connectors)
    if _OR.search(separators) and _AND.search(separators):
        return None
    return names, "or" if _OR.search(separators) else "and"


def _dataset_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    if not _POINT_CHANGE.search(text) or _POOLED.search(text) or _UNSAFE_LOGIC.search(text) or _NEGATED_SHARED.search(text):
        return None
    if match := _EN_DATASETS.search(text):
        layout = "english"
    elif match := _CN_DATASETS.search(text):
        layout = "chinese_after"
    elif match := _CN_LIST_FIRST.search(text):
        layout = "chinese_before"
    else:
        return None
    names = _dataset_names(match["items"], _EN_NAME if layout == "english" else _CN_NAME)
    if names is None:
        return None
    parts, operator = names
    start = match.start("items")
    atoms = []
    for left, right, name in parts:
        if layout == "english":
            proposition = f"{text[:match.start()]}{match['join']}dataset {name}{match['terminal']}"
        elif layout == "chinese_after":
            proposition = f"{text[:start].rstrip()} {name} {text[match.end('items'):].lstrip()}"
        else:
            proposition = f"{text[:start]}数据集 {name} {text[match.end():].lstrip()}"
        atoms.append((start + left, start + right, proposition))
    return atoms, operator


def _clause_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    if (_UNSAFE_LOGIC.search(text) or _TRAILING_CLAUSE_SCOPE.search(text)
            or re.search(r"\b(?:if|whether|because|than|whereas|with)\b|若|如果|因为", text, re.I)):
        return None
    for match in _CLAUSE_JOIN.finditer(text):
        left = text[:match.start()].strip()
        right = text[match.end():].strip()
        if not ((_EN_CLAUSE.match(left) and _EN_CLAUSE.match(right)) or (_CN_CLAUSE.match(left) and _CN_CLAUSE.match(right))):
            continue
        trailing_connectors = [item for item in _CLAUSE_JOIN.finditer(right) if (
            _EN_CLAUSE.match(right[item.end():]) or _CN_CLAUSE.match(right[item.end():])
        )]
        if trailing_connectors:
            # Nested or multi-clause coordination belongs to explicit structured extraction.
            return None
        if re.match(r"^either\b", left, re.I):
            if not _OR.search(match[0]):
                continue
            left = re.sub(r"^either\s+", "", left, flags=re.I)
        offset = text.index(left)
        right_offset = match.end() + len(text[match.end():]) - len(text[match.end():].lstrip())
        operator = "or" if _OR.search(match[0]) else "and"
        return [(offset, offset + len(left), left), (right_offset, right_offset + len(right), right)], operator
    return None


def shared_procedure_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    """Ground two explicit procedural actions with one first-person subject."""
    match = _PROCEDURE_SENTENCE.fullmatch(text)
    if match is None or _NEGATED_SHARED.search(text):
        return None

    left, right = match["left"], match["right"]
    if _PROCEDURE_SCOPE.search(left) or _PROCEDURE_SCOPE.search(right):
        return None

    # With an adjudication verb, the absence of manual review describes that decision.
    adjudication = _REVIEWLESS_ADJUDICATION.fullmatch(right)
    right_action = adjudication["action"] if adjudication else right
    left_match = _PROCEDURE_ACTION.fullmatch(left)
    right_match = _PROCEDURE_ACTION.fullmatch(right_action)
    if not left_match or left_match["adverb"] or not right_match:
        return None
    left_verb = left_match["verb"].casefold()
    if left_verb in _PROCEDURE_COMPLEMENT_VERBS:
        return None
    if _TRAILING_PROCEDURE_SCOPE.search(right_match["object"]):
        return None
    if not (match["comma"] or _DETERMINED_PROCEDURE_OBJECT.match(right_match["object"])):
        # A bare phrase such as "selected controls" could be a second object.
        detects_errors = left_verb == "applied" and _APPLIED_DETECTOR.match(left_match["object"])
        if not detects_errors or not _MANUAL_CORRECTION.fullmatch(right):
            return None

    return [
        (match.start("subject"), match.end("left"), text[:match.end("left")]),
        (*match.span("right"), f"{match['subject']} {right}{match['terminal']}"),
    ], "and"


def _reported_conjunction_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    """Split two reported findings when both predicate boundaries are explicit."""
    if not (report := _REPORTING_PREFIX.match(text)) or _UNSAFE_LOGIC.search(text):
        return None
    for connector in _REPORT_JOIN.finditer(text, report.end()):
        left = text[report.end():connector.start()]
        right = text[connector.end():]
        if ("," in right or not _REPORTED_PREDICATE.match(left) or not _REPORTED_PREDICATE.match(right)
                or _LIMITED_REPORTED_SCOPE.search(left) or _LIMITED_REPORTED_SCOPE.search(right)):
            continue
        if removal := _REMOVAL.match(right):
            role = _LOCAL_ROLE.fullmatch(left)
            if role is None or _ROLE_LINK.search(left):
                return None
            right = f"{removal['action']} {role['subject']}{right[removal.end():]}"
        elif not connector[0].startswith(","):
            # Without a clause delimiter the second noun may be part of the left object.
            continue
        return [
            (report.end(), connector.start(), f"{report['prefix']}{left}"),
            (connector.end(), len(text), f"{report['prefix']}{right}"),
        ], "and"
    return None


def _absolute_and_relative_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    """Separate a matched pair of absolute measurements from its appositive comparison."""
    match = _ABSOLUTE_AND_RELATIVE.fullmatch(text)
    if not match or len(re.findall(r"\band\b", match["subjects"], re.I)) != 1:
        return None
    subjects = re.sub(r"\band\b", "", match["subjects"], flags=re.I)
    if _UNSAFE_LOGIC.search(text) or _LIMITED_REPORTED_SCOPE.search(subjects) or _NEGATED_SHARED.search(subjects):
        return None
    evaluation = f"{match['evaluation']}, and " if match["evaluation"] else ""
    return [
        (*match.span("absolute"), f"{evaluation}{match['absolute']}"),
        (*match.span("relative"),
         f"{evaluation}{match['subjects']} {match['verb']} {match['delta']} in {match['metric']} {match['baseline']}"),
    ], "and"


def _shared_exception_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    """Keep a terminal exception on both simple objects of one negated verb."""
    match = _SHARED_EXCEPTION.fullmatch(text)
    if not match or any(_LIMITED_REPORTED_SCOPE.search(match[part]) for part in ("left", "right")):
        return None
    prefix = match["prefix"]
    if (_UNSAFE_LOGIC.search(prefix) or _NEGATED_SHARED.search(prefix) or _NON_ASSERTED_REPORT.search(prefix)
            or _UNSAFE_LOGIC.search(match["condition"][len("unless "):])):
        return None
    return [
        (0, match.end("left"), f"{match['prefix']}{match['left']}, {match['condition']}"),
        (*match.span("right_anchor"), f"{match['prefix']}{match['right']}, {match['condition']}"),
    ], "and"


def reported_compound_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    return _reported_conjunction_parts(text) or _absolute_and_relative_parts(text) or _shared_exception_parts(text)


def offline_compound_parts(text: str) -> tuple[list[tuple[int, int, str]], str] | None:
    """Return exact operand ranges plus self-contained propositions for conservative patterns."""
    return _dataset_parts(text) or shared_procedure_parts(text) or reported_compound_parts(text) or _clause_parts(text)


def requires_whole_sentence(text: str) -> bool:
    if _UNSAFE_LOGIC.search(text):
        return True
    if not (_POOLED.search(text) or _NEGATED_SHARED.search(text)):
        return False
    match = _EN_DATASETS.search(text) or _CN_DATASETS.search(text) or _CN_LIST_FIRST.search(text)
    return match is not None and bool(_LIST_JOIN.search(match["items"]))


def attach_compound(part: dict, atoms: list[dict], operator: str) -> None:
    if operator not in {"and", "or"} or len(atoms) < 2:
        return
    atoms = sorted(atoms, key=lambda atom: (atom["start"], atom["end"]))
    if any(atom.get("compound") for atom in atoms) or len({atom["claim_id"] for atom in atoms}) != len(atoms):
        return
    text = part["text"]
    unsafe = _UNSAFE_LOGIC.search(text)
    pooled_or_negated = _POOLED.search(text) or _NEGATED_SHARED.search(text)
    if unsafe or (pooled_or_negated and not all(
        _EN_CLAUSE.match(atom["text"]) or _CN_CLAUSE.match(atom["text"]) for atom in atoms
    )):
        expected = _shared_exception_parts(text) if unsafe else _absolute_and_relative_parts(text)
        observed = [
            (atom["start"] - part["start"], atom["end"] - part["start"], atom["proposition"]) for atom in atoms
        ]
        if expected != (observed, operator):
            return
    gaps = []
    for left, right in zip(atoms, atoms[1:]):
        if left["end"] > right["start"]:
            return
        gaps.append(part["text"][left["end"] - part["start"] : right["start"] - part["start"]])
    connectors = " ".join(gaps)
    if operator == "or" and (not _OR.search(connectors) or _AND.search(connectors)):
        return
    if operator == "and" and (_OR.search(connectors) or not (_AND.search(connectors) or re.search(r"[,，；;、]", connectors))):
        return
    members = [atom["claim_id"] for atom in atoms]
    key = json.dumps([part["start"], part["end"], operator, members], separators=(",", ":"))
    compound = {"compound_id": "compound-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                "operator": operator, "member_claim_ids": members, **part}
    for atom in atoms:
        atom["compound"] = compound


def aggregate_compound_findings(claims: list[dict], findings: list[dict]) -> list[dict]:
    """Return separate sentence rollups. Atomic rows stay available to scorers and revision."""
    by_id = {claim["claim_id"]: claim for claim in claims}
    judged = {finding["claim_id"]: finding for finding in findings}
    compounds = {claim["compound"]["compound_id"]: claim["compound"] for claim in claims if "compound" in claim}
    result = []
    for group in compounds.values():
        operator = group["operator"]
        ids = group["member_claim_ids"]
        if operator not in {"and", "or"} or len(ids) < 2 or any(eid not in by_id for eid in ids):
            continue
        members = []
        for eid in ids:
            claim, finding = by_id[eid], judged.get(eid, {})
            status = finding.get("review_status", "pending")
            members.append({**{key: claim[key] for key in ("claim_id", "text", "proposition", "start", "end")},
                            "review_status": status, "verdict": finding.get("verdict") if status == "completed" else None,
                            "reason": finding.get("reason"), "evidence_ids": finding.get("evidence_ids", [])})
        true = [member for member in members if member["verdict"] == "supported"]
        false = [member for member in members if member["verdict"] == "refuted"]
        statuses = {member["review_status"] for member in members}
        reason = "A required operand has not been resolved."
        proof = members
        if operator == "and" and false:
            status, verdict, proof = "completed", "refuted", false
            reason = "A refuted operand refutes the conjunction."
        elif operator == "or" and true:
            status, verdict, proof = "completed", "supported", true
            reason = "A supported alternative establishes the disjunction."
        elif operator == "and" and len(true) == len(members):
            status, verdict = "completed", "supported"
            reason = "All required operands of the conjunction are supported."
        elif operator == "or" and len(false) == len(members):
            status, verdict = "completed", "refuted"
            reason = "All alternatives in the disjunction are refuted."
        elif "manual_conflict" in statuses:
            status, verdict = "manual_conflict", None
            reason = "An operand has unresolved conflicting evidence."
        elif statuses == {"completed"} and any(member["verdict"] == "insufficient" for member in members):
            status, verdict = "completed", "insufficient"
            reason = "An operand lacks sufficient evidence and the other outcomes do not settle the compound claim."
        else:
            status, verdict = "pending", None
        evidence = list(dict.fromkeys(eid for member in proof for eid in member["evidence_ids"]))
        result.append({**group, "review_status": status, "verdict": verdict, "reason": reason,
                       "evidence_ids": evidence, "members": members})
    return result
