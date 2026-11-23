"""Estimate recorded model usage against an explicit, versioned rate table."""

import re
from decimal import Decimal


_RATE_FIELDS = ("input_per_million", "cached_input_per_million", "output_per_million")
_MILLION = Decimal(1_000_000)
_RATE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _rates(raw: object) -> tuple[str, dict]:
    if (not isinstance(raw, dict) or raw.keys() != {"schema_version", "pricing_id", "models"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or not isinstance(raw["pricing_id"], str) or not raw["pricing_id"].strip()
            or not isinstance(raw["models"], dict)):
        raise ValueError("pricing requires schema_version 1, a pricing_id, and a models object")
    rates = {}
    for model, fields in raw["models"].items():
        if (not isinstance(model, str) or not model.strip() or not isinstance(fields, dict)
                or fields.keys() != set(_RATE_FIELDS) or not all(isinstance(value, str) for value in fields.values())):
            raise ValueError("each model price must contain input, cached input, and output decimal strings per million")
        if any(not _RATE.fullmatch(value) for value in fields.values()):
            raise ValueError(f"prices for {model!r} must be finite nonnegative decimals")
        rates[model] = tuple(Decimal(fields[key]) for key in _RATE_FIELDS)
    return raw["pricing_id"], rates


def _event_cost(event: dict, rates: dict) -> Decimal | None:
    prices = rates.get(event.get("model")) if isinstance(event.get("model"), str) else None
    supplied, output, cached = (event.get(key) for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    if prices is None or any(type(value) is not int or value < 0 for value in (supplied, output)):
        return None
    input_rate, cached_rate, output_rate = prices
    if cached is None and input_rate == cached_rate:
        cached = 0
    if type(cached) is not int or not 0 <= cached <= supplied:
        return None
    return ((supplied - cached) * input_rate + cached * cached_rate + output * output_rate) / _MILLION


def score_costs(run_record: dict, price_table: object) -> dict:
    pricing_id, rates = _rates(price_table)
    samples = []
    observed_cost = Decimal(0)
    priced, unknown, missing = 0, 0, 0
    for sample in run_record["samples"]:
        events = sample.get("request_usage")
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            missing += 1
            samples.append({"sample_id": sample["sample_id"], "known_cost_usd": None,
                            "priced_requests": 0, "unpriced_requests": None})
            continue
        known = [cost for event in events if (cost := _event_cost(event, rates)) is not None]
        cost = sum(known, Decimal(0))
        unpriced = len(events) - len(known)
        reported = sample.get("model_requests")
        if type(reported) is int and reported > len(events):
            unpriced += reported - len(events)
        samples.append({"sample_id": sample["sample_id"], "known_cost_usd": format(cost, "f"),
                        "priced_requests": len(known), "unpriced_requests": unpriced})
        observed_cost += cost
        priced += len(known)
        unknown += unpriced
    return {"pricing_id": pricing_id, "currency": "USD", "known_cost_usd": format(observed_cost, "f"),
            "priced_requests": priced, "unpriced_requests": unknown, "samples_without_usage": missing,
            "complete": not (unknown or missing), "samples": samples}
