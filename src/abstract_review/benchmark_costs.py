"""Estimate recorded model usage against an explicit, versioned rate table."""

import re
from decimal import Decimal


_RATE_FIELDS = ("input_per_million", "cached_input_per_million", "output_per_million")
_WRITE_RATES = ("cache_write_30m_per_million", "cache_write_12h_per_million")
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
                or not set(_RATE_FIELDS) <= fields.keys() <= set((*_RATE_FIELDS, *_WRITE_RATES))
                or not all(isinstance(value, str) for value in fields.values())):
            raise ValueError("each model price must contain input, cached input, and output rates and may contain cache write rates")
        if any(not _RATE.fullmatch(value) for value in fields.values()):
            raise ValueError(f"prices for {model!r} must be finite nonnegative decimals")
        rates[model] = {key: Decimal(value) for key, value in fields.items()}
    return raw["pricing_id"], rates


def _event_cost(event: dict, rates: dict) -> Decimal | None:
    prices = rates.get(event.get("model")) if isinstance(event.get("model"), str) else None
    supplied, output, cached = (event.get(key) for key in ("input_tokens", "output_tokens", "cached_input_tokens"))
    if prices is None or any(type(value) is not int or value < 0 for value in (supplied, output)):
        return None
    input_rate, cached_rate, output_rate = (prices[key] for key in _RATE_FIELDS)
    if cached is None and input_rate == cached_rate:
        cached = 0
    if type(cached) is not int or not 0 <= cached <= supplied:
        return None
    total_write = event.get("cache_write_input_tokens")
    write_30m = event.get("cache_write_30m_input_tokens")
    write_12h = event.get("cache_write_12h_input_tokens")
    if total_write is None and write_30m is None and write_12h is None:
        total_write = write_30m = write_12h = 0
    elif type(total_write) is int and total_write == 0:
        write_30m = 0 if write_30m is None else write_30m
        write_12h = 0 if write_12h is None else write_12h
    if (any(type(count) is not int or count < 0 for count in (total_write, write_30m, write_12h))
            or total_write != write_30m + write_12h or cached + total_write > supplied):
        return None
    write_cost = Decimal(0)
    for count, field in zip((write_30m, write_12h), _WRITE_RATES):
        if count:
            if field not in prices:
                return None
            write_cost += count * prices[field]
    uncached = supplied - cached - total_write
    return (uncached * input_rate + cached * cached_rate + write_cost + output * output_rate) / _MILLION


def score_costs(run_record: dict, price_table: object) -> dict:
    pricing_id, rates = _rates(price_table)
    if not run_record.get("samples"):
        raise ValueError("cost estimates require per-sample usage records")
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
