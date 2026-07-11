"""SCENARIO_DATABASE.md -> structured ``Scenario`` dicts.

The Opponent-Weakness Scenario Database is authored by hand in Markdown
(``TanitAD Research Hub/Opponent Analyzer/SCENARIO_DATABASE.md``): one ``##
SC-xx — <title> [W-xx] ★★★`` heading per scenario, then a set of
``- **Field:** value`` bullets (Opponent evidence, Description, TanitAD
mechanism, Data sources, Metric hooks, Status).

This parser is deliberately **section-heading / regex driven and fail-soft**:
every field is extracted independently and a per-scenario ``parse_warnings``
list records anything missing or malformed, so one broken entry never crashes
the whole parse (repo fail-loud-but-not-fragile doctrine — we surface the
problem, we do not swallow the file).

Public surface:

    parse_scenarios(md_text)  -> list[Scenario dict]
    parse_file(path)          -> list[Scenario dict]
    to_index(scenarios, src)  -> {"scenarios": [...], "source": ..., ...}
    write_scenarios_json(...)  -> writes a portable scenarios.json (rel paths)

Run standalone to emit + report coverage::

    python -m tanitad.scena.parse --db-md <SCENARIO_DATABASE.md> --out scenarios.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Canonical lifecycle ladder (SCENARIO_DATABASE.md preamble). The front-end
# stepper and the server both key off this exact order.
STAGES: tuple[str, ...] = (
    "catalogued",
    "spec-drafted",
    "data-sourced",
    "oracle-tested",
    "live-measured",
    "excellence-proven",
)

# Evidence labels (P8): FACT (record/primary footage) / CLAIM (press) /
# INFER (our inference). Compound labels like "FACT-family" or "FACT/CLAIM"
# collapse to their primary token.
_LABELS: tuple[str, ...] = ("FACT", "CLAIM", "INFER")

# Data-source kind detection — first pattern that matches a clause wins, so the
# order is significant (specific dataset names before the generic CARLA/HF nets).
_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("nuScenes", r"nuscenes"),
    ("Cosmos", r"cosmos"),
    ("NuRec", r"nurec"),
    ("comma2k19", r"comma2k19|comma2k|\bcomma\b"),
    ("HF", r"physicalai|hugging\s*face|worldmodel-synthetic|\bhf\b"),
    ("CARLA", r"carla|town\s*\d|metadrive|blocked_route|walker\.child"),
    ("dashcam", r"dashcam"),
)

_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_FIELD_RE = re.compile(r"^\s*[-*]\s+\*\*(.+?):\*\*\s*(.*)$")
_HEADING_RE = re.compile(r"^##\s+(SC-\d+)\s*[—–-]\s*(.*)$")


# --------------------------------------------------------------------------- #
# small text helpers
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    """Strip Markdown emphasis/code fences for display; collapse whitespace."""
    if not text:
        return ""
    text = text.replace("**", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)      # inline code -> plain
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canon_label(raw: str) -> str | None:
    """First of FACT/CLAIM/INFER appearing in a raw parenthetical label."""
    up = (raw or "").upper()
    for lab in _LABELS:
        if lab in up:
            return lab
    return None


def _lifecycle_stage(status_text: str) -> str | None:
    """The first canonical ladder stage named in the (cleaned) Status text.

    "catalogued → spec next" -> catalogued; "live-measured (partial)" ->
    live-measured. Returns ``None`` if the Status names no known stage.
    """
    low = (status_text or "").lower()
    best_stage, best_pos = None, len(low) + 1
    for st in STAGES:
        i = low.find(st)
        if 0 <= i < best_pos:
            best_stage, best_pos = st, i
    return best_stage


def _dataset_link(kind: str, ref: str) -> str | None:
    """Resolve a clickable link for a data-source clause.

    An explicit URL in the clause always wins. Otherwise we resolve a *real,
    non-fabricated* landing page per kind (dataset homepage, or a HuggingFace
    dataset *search* for HF/Cosmos corpora) so every dataset card has a working
    "get the link" affordance without inventing an exact repo id.
    """
    m = _URL_RE.search(ref or "")
    if m:
        return m.group(0).rstrip(".,);")
    k = (kind or "").lower()
    if k == "nuscenes":
        return "https://www.nuscenes.org/nuscenes"
    if k == "comma2k19":
        return "https://github.com/commaai/comma2k19"
    if k == "carla":
        return "https://carla.org/"
    if k in ("cosmos", "hf"):
        name = re.search(r"([A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)", ref or "")
        token = name.group(1) if name else ("Cosmos" if k == "cosmos" else "")
        return "https://huggingface.co/datasets?search=" + quote(token)
    return None


# --------------------------------------------------------------------------- #
# heading + field parsing
# --------------------------------------------------------------------------- #

def _split_heading(rest: str) -> dict[str, Any]:
    """Parse the text after ``## SC-xx —`` into title + tag metadata."""
    w = re.search(r"\[(W-\d+)(\s+family)?\]", rest)
    stars = rest.count("★")
    headline = "headline" in rest.lower()
    is_new = bool(re.search(r"\(new\b", rest))

    cut = len(rest)
    for marker in ("[", "★", "(new"):
        i = rest.find(marker)
        if i != -1:
            cut = min(cut, i)
    title = rest[:cut].strip().strip("—–-").strip()

    tags: list[str] = []
    if w:
        tags.append(w.group(1) + (" family" if w.group(2) else ""))
    if stars:
        tags.append("★" * stars)
    if headline:
        tags.append("headline")
    if is_new:
        tags.append("new")

    return {
        "title": title,
        "w_code": w.group(1) if w else None,
        "family": bool(w and w.group(2)),
        "stars": stars,
        "headline": headline,
        "is_new": is_new,
        "tags": tags,
    }


def _field_map(body_lines: list[str]) -> dict[str, str]:
    """Group a scenario body's bullet lines into ``{field_name: raw_value}``.

    Continuation lines (indented / non-bullet) fold into the current field, so
    multi-line Status/Data-source blocks survive intact. Field names keep their
    trailing parenthetical (e.g. ``Opponent evidence (FACT)``) for the caller.
    """
    fields: dict[str, str] = {}
    order: list[str] = []
    cur: str | None = None
    for line in body_lines:
        m = _FIELD_RE.match(line)
        if m:
            cur = m.group(1).strip()
            fields[cur] = m.group(2).strip()
            order.append(cur)
        elif cur is not None:
            stripped = line.strip()
            if stripped:
                fields[cur] = (fields[cur] + " " + stripped).strip()
    fields["__order__"] = "\n".join(order)
    return fields


def _get(fields: dict[str, str], *names: str) -> tuple[str, str] | None:
    """Return ``(matched_field_name, value)`` for the first field whose name
    starts with any of ``names`` (case-insensitive), else ``None``."""
    for key in fields:
        if key == "__order__":
            continue
        kl = key.lower()
        for n in names:
            if kl.startswith(n.lower()):
                return key, fields[key]
    return None


def _parse_data_sources(text: str) -> list[dict[str, Any]]:
    """Split a Data-sources value into ``{kind, ref, status, link, replay}``."""
    out: list[dict[str, Any]] = []
    # Drop a leading emphasis lead-in like "**verified available** —".
    text = re.sub(r"^\s*\*\*[^*]+\*\*\s*[—–-]\s*", "", text)
    for clause in re.split(r";\s*", text):
        ref = _clean(clause)
        if not ref:
            continue
        kind = "other"
        low = ref.lower()
        for name, pat in _KIND_PATTERNS:
            if re.search(pat, low):
                kind = name
                break
        parens = re.findall(r"\(([^)]*)\)", ref)
        status = _clean(parens[-1]) if parens else ""
        replay = bool(re.search(r"\.rrd\b|resim bundle|tanitresim", low))
        out.append({
            "kind": kind,
            "ref": ref,
            "status": status,
            "link": _dataset_link(kind, ref),
            "replay": replay,
        })
    return out


def _parse_metric_hooks(text: str) -> tuple[list[str], str]:
    """Best-effort chip tokens + a cleaned full-text of the Metric-hooks value.

    Trailing hand-off / next-step prose ("Handoff to …", "Next: …") is dropped
    from the chip list but kept in the full text.
    """
    body = re.split(r"\*\*(?:Handoff|Next)\b", text)[0]
    body = _clean(body)
    tokens: list[str] = []
    for tok in re.split(r",\s*", body):
        tok = re.split(r"\s+via\s+|\s+toward\s+|\s*\(", tok)[0].strip(" .")
        if tok and len(tok) <= 60 and not tok.lower().startswith("handoff"):
            tokens.append(tok)
    return tokens, _clean(text)


# --------------------------------------------------------------------------- #
# per-scenario + whole-file
# --------------------------------------------------------------------------- #

def _parse_block(sc_id: str, heading_rest: str, body: str) -> dict[str, Any]:
    """Parse one scenario (heading remainder + body text). Never raises for a
    field-level problem — it records a ``parse_warnings`` entry instead."""
    warns: list[str] = []
    head = _split_heading(heading_rest)
    if not head["title"]:
        warns.append("empty title")

    fields = _field_map(body.splitlines())

    ev = _get(fields, "Opponent evidence", "Evidence")
    evidence_label = evidence_label_raw = None
    opponent_evidence = ""
    if ev:
        name, val = ev
        pm = re.search(r"\(([^)]*)\)", name)
        evidence_label_raw = pm.group(1) if pm else None
        evidence_label = _canon_label(evidence_label_raw or "")
        opponent_evidence = _clean(val)
        if evidence_label is None:
            warns.append(f"unrecognised evidence label {evidence_label_raw!r}")
    else:
        warns.append("no Opponent evidence field")

    desc = _get(fields, "Description")
    description = _clean(desc[1]) if desc else ""
    if not desc:
        warns.append("no Description field")

    correct_behavior = ""
    # Primary: "correct behavior is/bounds/…"; fallback: "the correct <noun> …".
    cb = re.search(r"correct behaviou?r\s+(?:is\s+)?(.*)", description, re.I) \
        or re.search(r"\bcorrect\s+\w+\s+(.*)", description, re.I)
    if cb:
        correct_behavior = cb.group(1).strip(" .;:") + "."
    else:
        warns.append("no 'correct behavior' clause in Description")

    mech = _get(fields, "TanitAD mechanism", "Mechanism")
    mechanism = _clean(mech[1]) if mech else ""

    ds = _get(fields, "Data sources", "Data source")
    data_sources = _parse_data_sources(ds[1]) if ds else []
    if not data_sources:
        warns.append("no Data sources parsed")

    mh = _get(fields, "Metric hooks", "Metric hook")
    metric_hooks, metric_hooks_text = ([], "")
    if mh:
        metric_hooks, metric_hooks_text = _parse_metric_hooks(mh[1])
    else:
        warns.append("no Metric hooks field")

    stt = _get(fields, "Status")
    status_text = _clean(stt[1]) if stt else ""
    lifecycle_stage = _lifecycle_stage(status_text) if stt else None
    if not stt:
        warns.append("no Status field")
    elif lifecycle_stage is None:
        warns.append("Status names no known lifecycle stage")
        lifecycle_stage = "catalogued"

    evidence_links = [u.rstrip(".,);") for u in _URL_RE.findall(
        (opponent_evidence + " " + status_text))]

    return {
        "id": sc_id,
        "title": head["title"],
        "w_code": head["w_code"],
        "family": head["family"],
        "stars": head["stars"],
        "headline": head["headline"],
        "is_new": head["is_new"],
        "tags": head["tags"],
        "evidence_label": evidence_label,
        "evidence_label_raw": evidence_label_raw,
        "opponent_evidence": opponent_evidence,
        "description": description,
        "correct_behavior": correct_behavior,
        "mechanism": mechanism,
        "data_sources": data_sources,
        "metric_hooks": metric_hooks,
        "metric_hooks_text": metric_hooks_text,
        "lifecycle_stage": lifecycle_stage,
        "status_text": status_text,
        "evidence_links": evidence_links,
        "parse_warnings": warns,
    }


def parse_scenarios(md_text: str) -> list[dict[str, Any]]:
    """Parse every ``## SC-xx`` entry in ``md_text`` into a Scenario dict.

    Robust to Markdown variation and ordering; a hard failure on one block is
    caught and rendered as a stub entry with a ``parse_warnings`` note rather
    than aborting the whole file.
    """
    lines = md_text.splitlines()
    # All level-2 headings, so a scenario block ends at the next ## / EOF.
    heads: list[tuple[int, str, str]] = []      # (line_idx, sc_id, rest)
    h2: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            h2.append(i)
            m = _HEADING_RE.match(line)
            if m:
                heads.append((i, m.group(1), m.group(2)))

    scenarios: list[dict[str, Any]] = []
    for idx, sc_id, rest in heads:
        end = next((h for h in h2 if h > idx), len(lines))
        body = "\n".join(lines[idx + 1:end])
        try:
            scenarios.append(_parse_block(sc_id, rest, body))
        except Exception as exc:                # never crash on one entry
            scenarios.append({
                "id": sc_id, "title": _split_heading(rest).get("title", ""),
                "parse_warnings": [f"hard parse error: {exc!r}"],
                "data_sources": [], "metric_hooks": [], "tags": [],
                "lifecycle_stage": "catalogued",
            })
    return scenarios


def parse_file(path: str | Path) -> list[dict[str, Any]]:
    """Read a SCENARIO_DATABASE.md file and parse it."""
    return parse_scenarios(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# scenarios.json emission (portable — relative source name only)
# --------------------------------------------------------------------------- #

def to_index(scenarios: list[dict[str, Any]],
             source_name: str = "SCENARIO_DATABASE.md") -> dict[str, Any]:
    """Wrap parsed scenarios into the portable ``scenarios.json`` payload.

    ``source`` is stored **basename-only** so the emitted JSON carries no
    machine-specific absolute path (portability invariant, mirrors TanitResim).
    """
    warns = {s["id"]: s.get("parse_warnings", [])
             for s in scenarios if s.get("parse_warnings")}
    return {
        "source": Path(source_name).name,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": list(STAGES),
        "n": len(scenarios),
        "n_warnings": sum(len(v) for v in warns.values()),
        "warnings": warns,
        "scenarios": scenarios,
    }


def write_scenarios_json(scenarios: list[dict[str, Any]], out_path: str | Path,
                         source_name: str = "SCENARIO_DATABASE.md") -> dict:
    """Emit a portable ``scenarios.json`` next to the index. Returns the dict."""
    payload = to_index(scenarios, source_name)
    Path(out_path).write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse SCENARIO_DATABASE.md")
    ap.add_argument("--db-md", required=True, help="path to SCENARIO_DATABASE.md")
    ap.add_argument("--out", default="scenarios.json", help="output JSON path")
    args = ap.parse_args(argv)

    scenarios = parse_file(args.db_md)
    payload = write_scenarios_json(scenarios, args.out, Path(args.db_md).name)
    print(f"[scena.parse] {payload['n']} scenarios from "
          f"{payload['source']} -> {args.out}")
    if payload["warnings"]:
        print(f"[scena.parse] {payload['n_warnings']} parse warning(s):")
        for sid, ws in payload["warnings"].items():
            for w in ws:
                print(f"    {sid}: {w}")
    else:
        print("[scena.parse] no parse warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
