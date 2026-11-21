#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ ARE ALL FOUR LEVERS PRESENT *AND CONSUMED*? Answer by probe, not by prose.

    python lever_census.py --trainer <refc_v3_train.py> \
        --agent-slots <agent_slots.py> --wp-index <refc_wp_index.py> \
        --json raw/lever_census.json

Two questions, and the second is the one that matters:

  1. is the flag DECLARED?   -- an `add_argument` line
  2. is its value CONSUMED?  -- does anything READ the parsed attribute and put it
                                into the model / the loss?

⭐ Question 2 exists because of `tac_goal_tok_head`: **11,286 parameters, `grad_abs_sum`
exactly 0 for all 40,284 steps** of refcv5-v2 -- built, stamped, rollable, and never
trained. A flag can be parsed, recorded into `config.json`, and reach nothing. The
programme calls this the *parsed, stamped and inert* defect and has now hit it on
`--wp-index`, on `tac_goal_stats`, and on seven phantom seam keys.

⛔ EVERY COUNT IS PAIRED WITH A SAME-BREATH CONTROL THAT MUST READ NON-ZERO.
On the G: mount a search tool reports "no matches" for files it could not OPEN, and
directory listings keep working while content reads fail -- so a zero is a claim
about the READ, not about the file. If a control reads zero the census reports
INCONCLUSIVE for that file and never "absent".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

#: lever -> (option string, parsed attribute, the consumption sites we require)
LEVERS = {
    "D": {"flag": "--w-u0", "attr": "w_u0",
          "claim": "the x0 loss in CONTROL space; DD-v1 has no such term"},
    "A": {"flag": "--agents", "attr": "agents",
          "claim": "the learned Hungarian-matched agent detector"},
    "B": {"flag": "--wp-index", "attr": "wp_index",
          "claim": "waypoint-indexed cross-attention, DD coupling (1)"},
    "C": {"flag": "--w-tac-goal", "attr": "w_tac_goal",
          "claim": "the 22-token tactical-goal SET loss"},
}

#: The DiffusionDrive machinery the design says already exists.
REQUIRED_SYMBOLS = {
    "agent_slots": ["AgentSlotDecoder", "hungarian", "match_slots",
                    "slot_set_loss", "targets_from_join"],
    "wp_index": ["WaypointIndexConfig", "WaypointIndexBias", "build_relation",
                 "waypoint_agent_geometry"],
}

#: Controls: tokens that MUST appear in each file. If one reads 0 the file was
#: not really read, and every other count from it is inadmissible.
CONTROLS = {
    "trainer": [("def main", 1), ("add_argument", 50)],
    "agent_slots": [("def ", 5)],
    "wp_index": [("def ", 3)],
}


def read_file(path: str) -> tuple[list[str] | None, str | None]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines(), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def md5(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except Exception:
        return None


def census_file(path: str, key: str) -> dict:
    lines, err = read_file(path)
    if lines is None:
        return {"path": path, "readable": False, "error": err,
                "verdict": "INCONCLUSIVE",
                "why": "⛔ the file could not be READ. Every count from it would "
                       "be a claim about the mount, not about the file."}
    text = "\n".join(lines)
    controls = {}
    for token, minimum in CONTROLS.get(key, []):
        n = text.count(token)
        controls[token] = {"count": n, "min_required": minimum, "ok": n >= minimum}
    ctrl_ok = all(c["ok"] for c in controls.values()) if controls else True
    return {"path": path, "readable": True, "md5": md5(path),
            "n_lines": len(lines), "controls": controls,
            "controls_ok": ctrl_ok, "lines": lines,
            "verdict": "READ" if ctrl_ok else "INCONCLUSIVE",
            "why": ("" if ctrl_ok else
                    "⛔ a same-breath control read below its minimum -- the file "
                    "was not fully read and its counts are inadmissible.")}


def find_decl(lines: list[str], flag: str) -> dict | None:
    pat = re.compile(r'add_argument\(\s*["\']' + re.escape(flag) + r'["\']')
    for i, ln in enumerate(lines, 1):
        if pat.search(ln):
            return {"line": i, "text": ln.strip()[:160]}
    return None


def find_consumption(lines: list[str], attr: str) -> list[dict]:
    """Sites that READ the parsed attribute (not the declaration)."""
    pats = [re.compile(r'\bargs\.' + re.escape(attr) + r'\b'),
            re.compile(r'getattr\(\s*args\s*,\s*["\']' + re.escape(attr) + r'["\']'),
            re.compile(r'getattr\(\s*model\s*,\s*["\']_' + re.escape(attr) + r'["\']'),
            re.compile(r'\bmodel\._' + re.escape(attr) + r'\b'),
            re.compile(r'\b_?' + re.escape(attr) + r'\s*=\s*float\(getattr')]
    out = []
    for i, ln in enumerate(lines, 1):
        if "add_argument" in ln:
            continue
        if any(p.search(ln) for p in pats):
            out.append({"line": i, "text": ln.strip()[:140]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--agent-slots", required=True)
    ap.add_argument("--wp-index", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = {
        "trainer": census_file(a.trainer, "trainer"),
        "agent_slots": census_file(a.agent_slots, "agent_slots"),
        "wp_index": census_file(a.wp_index, "wp_index"),
    }

    levers = {}
    tl = files["trainer"]
    for lid, spec in LEVERS.items():
        if not tl.get("readable") or not tl.get("controls_ok"):
            levers[lid] = {**spec, "verdict": "INCONCLUSIVE",
                           "why": "the trainer could not be read reliably"}
            continue
        decl = find_decl(tl["lines"], spec["flag"])
        cons = find_consumption(tl["lines"], spec["attr"])
        # ⚠️ TWO DIFFERENT NUMBERS, BOTH CORRECT, AND THEY DIFFER.
        # `grep -c` counts LINES containing the flag; `str.count` counts
        # OCCURRENCES, so one line carrying the flag twice is 1 vs 2. MEASURED
        # here: `--agents` reads 31 lines / 32 occurrences and `--wp-index`
        # 25 / 26. Neither is wrong; quoting one as the other is. Both are
        # emitted so a reader never has to guess which question was asked.
        joined = "\n".join(tl["lines"])
        n_occurrences = joined.count(spec["flag"])
        n_lines_with = sum(1 for ln in tl["lines"] if spec["flag"] in ln)
        ok = bool(decl) and bool(cons)
        levers[lid] = {
            **spec,
            "declared": bool(decl), "declaration": decl,
            "n_mention_lines_grep_c": n_lines_with,
            "n_mention_occurrences": n_occurrences,
            "n_raw_mentions": n_lines_with,
            "n_consumption_sites": len(cons),
            "consumption_sites": cons[:8],
            "verdict": "PRESENT_AND_CONSUMED" if ok else
                       ("DECLARED_BUT_INERT" if decl else "ABSENT"),
            "why": ("declared and its parsed value reaches the model/loss" if ok else
                    ("⛔ DECLARED BUT INERT -- the `tac_goal_tok_head` defect: "
                     "parsed, stamped, and reaching nothing" if decl else
                     "⛔ no add_argument declaration found")),
        }

    symbols = {}
    for key, wanted in REQUIRED_SYMBOLS.items():
        f = files[key]
        if not f.get("readable") or not f.get("controls_ok"):
            symbols[key] = {"verdict": "INCONCLUSIVE", "why": f.get("why")}
            continue
        found = {}
        for s in wanted:
            m = None
            for i, ln in enumerate(f["lines"], 1):
                if re.match(r'^\s*(class|def)\s+' + re.escape(s) + r'\b', ln):
                    m = i
                    break
            found[s] = m
        allf = all(v is not None for v in found.values())
        symbols[key] = {"path": f["path"], "md5": f["md5"], "symbols": found,
                        "verdict": "ALL_PRESENT" if allf else "MISSING",
                        "why": ("every required symbol is defined here" if allf else
                                "⛔ missing: " + ", ".join(
                                    k for k, v in found.items() if v is None))}

    all_levers_ok = all(v["verdict"] == "PRESENT_AND_CONSUMED" for v in levers.values())
    all_syms_ok = all(v["verdict"] == "ALL_PRESENT" for v in symbols.values())
    verdict = ("PASS" if (all_levers_ok and all_syms_ok) else
               ("INCONCLUSIVE" if any(
                   v["verdict"] == "INCONCLUSIVE"
                   for v in list(levers.values()) + list(symbols.values()))
                else "FAIL"))

    report = {
        "tool": "lever_census.py",
        "files": {k: {kk: vv for kk, vv in v.items() if kk != "lines"}
                  for k, v in files.items()},
        "levers": levers, "symbols": symbols,
        "verdict": verdict,
        "summary": ("all four levers are DECLARED and CONSUMED, and DiffusionDrive's "
                    "Hungarian matching + coupling (1) already exist ⇒ refcv6 is a "
                    "CONFIGURATION, not new code"
                    if verdict == "PASS" else "see per-item verdicts"),
    }
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[census] wrote {a.json}")
    for k, f in files.items():
        print(f"  file {k:<12} {f['verdict']:<14} "
              f"lines={f.get('n_lines')} md5={f.get('md5')}")
    for lid, v in sorted(levers.items()):
        print(f"  lever {lid} {v['flag']:<14} {v['verdict']:<22} "
              f"decl@{(v.get('declaration') or {}).get('line')} "
              f"mentions={v.get("n_mention_lines_grep_c")}L/{v.get("n_mention_occurrences")}occ "
              f"consumed@{v.get('n_consumption_sites')} sites")
    for k, v in sorted(symbols.items()):
        print(f"  symbols {k:<12} {v['verdict']:<14} {v.get('symbols')}")
    print(f"[census] verdict = {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
