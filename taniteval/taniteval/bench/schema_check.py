"""A small, dependency-free JSON-Schema (draft 2020-12 SUBSET) validator for the suite's contract.

WHY IT EXISTS: the TanitAD venv has no ``jsonschema`` (MEASURED 2026-09-19), and the contract
files (``schema/bench_run.schema.json``, ``schema/summary.schema.json``) must be ENFORCED by the
suite on write, not merely documented. The schema files stay standard draft 2020-12, so a
consumer that has ``jsonschema`` can validate with it and get the same verdict on this subset.

Supported keywords (anything else in a schema is REFUSED, never silently ignored — an unknown
keyword is exactly how a rule stops being checked):
``$schema $id $defs $ref title description type enum const pattern minLength minimum maximum
required properties additionalProperties minProperties items minItems contains allOf anyOf oneOf
not if then else``.

``validate(instance, schema) -> list[str]`` returns every violation as ``"<json-pointer>: why"``;
an empty list means valid.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

_ANNOTATIONS = {"$schema", "$id", "$defs", "title", "description", "$comment", "examples", "default"}
_SUPPORTED = _ANNOTATIONS | {
    "$ref", "type", "enum", "const", "pattern", "minLength", "minimum", "maximum", "required",
    "properties", "additionalProperties", "minProperties", "items", "minItems", "contains",
    "allOf", "anyOf", "oneOf", "not", "if", "then", "else"}


class SchemaError(ValueError):
    """The SCHEMA itself uses something this validator does not implement."""


def _is_type(x, t: str) -> bool:
    if t == "null":
        return x is None
    if t == "boolean":
        return isinstance(x, bool)
    if t == "integer":
        return (isinstance(x, int) and not isinstance(x, bool)) or (
            isinstance(x, float) and x.is_integer())
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "string":
        return isinstance(x, str)
    if t == "array":
        return isinstance(x, list)
    if t == "object":
        return isinstance(x, dict)
    raise SchemaError(f"unknown type {t!r}")


def _json_equal(a, b) -> bool:
    """JSON equality: bool is NOT a number (Python says True == 1; JSON does not)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local refs are supported, got {ref!r}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _v(x, s, root, path: str, out: list) -> None:
    if s is True or s == {}:
        return
    if s is False:
        out.append(f"{path or '/'}: schema false")
        return
    unknown = set(s) - _SUPPORTED
    if unknown:
        raise SchemaError(f"unsupported keyword(s) {sorted(unknown)} at schema for {path or '/'}")
    p = path or "/"
    if "$ref" in s:
        _v(x, _resolve(s["$ref"], root), root, path, out)
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_is_type(x, t) for t in ts):
            out.append(f"{p}: type {type(x).__name__} is not {ts}")
            return
    if "const" in s and not _json_equal(x, s["const"]):
        out.append(f"{p}: {x!r} != const {s['const']!r}")
    if "enum" in s and not any(_json_equal(x, e) for e in s["enum"]):
        out.append(f"{p}: {x!r} not in enum {s['enum']}")
    if isinstance(x, str):
        if "pattern" in s and not re.search(s["pattern"], x):
            out.append(f"{p}: {x!r} does not match /{s['pattern']}/")
        if "minLength" in s and len(x) < s["minLength"]:
            out.append(f"{p}: string shorter than {s['minLength']}")
    if _is_type(x, "number"):
        if "minimum" in s and x < s["minimum"]:
            out.append(f"{p}: {x} < minimum {s['minimum']}")
        if "maximum" in s and x > s["maximum"]:
            out.append(f"{p}: {x} > maximum {s['maximum']}")
    if isinstance(x, dict):
        for k in s.get("required", []):
            if k not in x:
                out.append(f"{p}: missing required key {k!r}")
        if "minProperties" in s and len(x) < s["minProperties"]:
            out.append(f"{p}: fewer than {s['minProperties']} properties")
        props = s.get("properties", {})
        for k, sub in props.items():
            if k in x:
                _v(x[k], sub, root, f"{path}/{k}", out)
        if "additionalProperties" in s:
            ap = s["additionalProperties"]
            for k in x:
                if k in props:
                    continue
                if ap is False:
                    out.append(f"{p}: additional property {k!r} not allowed")
                elif isinstance(ap, dict):
                    _v(x[k], ap, root, f"{path}/{k}", out)
    if isinstance(x, list):
        if "minItems" in s and len(x) < s["minItems"]:
            out.append(f"{p}: fewer than {s['minItems']} items")
        if "items" in s:
            for i, e in enumerate(x):
                _v(e, s["items"], root, f"{path}/{i}", out)
        if "contains" in s:
            if not any(not _errs(e, s["contains"], root) for e in x):
                out.append(f"{p}: no item matches contains {json.dumps(s['contains'])}")
    for sub in s.get("allOf", []):
        _v(x, sub, root, path, out)
    if "anyOf" in s and not any(not _errs(x, sub, root) for sub in s["anyOf"]):
        out.append(f"{p}: matches none of anyOf")
    if "oneOf" in s:
        n_ok = sum(1 for sub in s["oneOf"] if not _errs(x, sub, root))
        if n_ok != 1:
            detail = [e for sub in s["oneOf"] for e in _errs(x, sub, root)][:4]
            out.append(f"{p}: matches {n_ok} of oneOf (need exactly 1); e.g. {detail}")
    if "not" in s and not _errs(x, s["not"], root):
        out.append(f"{p}: must NOT match {json.dumps(s['not'])}")
    if "if" in s:
        if not _errs(x, s["if"], root):
            if "then" in s:
                _v(x, s["then"], root, path, out)
        elif "else" in s:
            _v(x, s["else"], root, path, out)


def _errs(x, s, root) -> list:
    out: list = []
    _v(x, s, root, "", out)
    return out


def validate(instance, schema: dict) -> list:
    """Every violation of ``schema`` by ``instance`` (empty list = valid)."""
    out: list = []
    _v(instance, schema, schema, "", out)
    return out


def load_schema(name: str) -> dict:
    """``name`` in {"bench_run", "summary"} -> the parsed schema file."""
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
