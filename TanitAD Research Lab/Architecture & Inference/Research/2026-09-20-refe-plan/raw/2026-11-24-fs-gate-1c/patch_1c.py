#!/usr/bin/env python3
"""Fs launch gate 1c (MM 2026-11-24): G-LIVE's field_reach evidence form keeps 1b's non-probe keys.

The Fs gate (pod, 2026-11-24T07:29Z, package code_86d174bb) crashed after G-HYG and G-DVB PASS:
  gate_checks.py check_live: det["field_reach"] = {k: (v if k in ("ran", "labelled") else {"loss": v["loss"], "groups": v["groups"]}) for k, v in fr.items()}
  TypeError: list indices must be integers or slices, not str
because 1b added `losses` (a list), `n_active` and `active_index` to the field_reach result and the comprehension indexes every key but ran / labelled as a probe dict.
1b's tests called field_reach and judge_field_reach directly and never this caller.

usage: patch_1c.py <gate_checks.py> <test file holding chk_field_reach_active_sample>   (edits both in place, LF; each anchor must occur exactly once)
"""
import sys

GC, TF = sys.argv[1], sys.argv[2]
TEST_NAME = TF.replace("\\", "/").rsplit("/", 1)[-1]


def sub(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"REFUSING: anchor '{what}' found {n} times (need 1)")
    return src.replace(old, new)


# ---------------------------------------------------------------- gate_checks.py
g = open(GC, encoding="utf-8", newline="").read()
if "\r\n" in g:
    raise SystemExit("REFUSING: gate_checks.py has CRLF")
HELPER = (
    "def field_reach_detail(fr: dict) -> dict:\n"
    "    \"\"\"G-LIVE's evidence form of a `field_reach` result: each probe arm (`declared`, `zero`) is cut to {loss, groups}; every other key (ran, labelled and, since 1b,\n"
    "    losses / n_active / active_index) is kept as it is.  Fs launch gate 1c (MM 2026-11-24): the inline form indexed every key but ran / labelled as a probe dict and raised\n"
    "    TypeError on 1b's `losses` list, so the Fs gate crashed after G-DVB (pod, 2026-11-24T07:29Z).\"\"\"\n"
    "    return {k: ({\"loss\": v[\"loss\"], \"groups\": v[\"groups\"]} if isinstance(v, dict) else v) for k, v in fr.items()}\n"
    "\n"
    "\n"
)
g = sub(g, "def judge_field_reach(", HELPER + "def judge_field_reach(", "def judge_field_reach(")
g = sub(g,
        "        det[\"field_reach\"] = {k: (v if k in (\"ran\", \"labelled\") else {\"loss\": v[\"loss\"], \"groups\": v[\"groups\"]}) for k, v in fr.items()}\n",
        "        det[\"field_reach\"] = field_reach_detail(fr)\n",
        "check_live field_reach comprehension")
open(GC, "w", encoding="utf-8", newline="\n").write(g)

# ---------------------------------------------------------------- the test file
t = open(TF, encoding="utf-8", newline="").read()
if "\r\n" in t:
    raise SystemExit("REFUSING: test file has CRLF")
if "\nimport inspect\n" not in t:
    t = sub(t, "\nimport hashlib\n", "\nimport hashlib\nimport inspect\n", "import hashlib")
ANCHOR = "        ok(GC.judge_field_reach({\"ran\": True, \"labelled\": 0}, 2.0) != [] and GC.judge_field_reach({\"ran\": False}, 2.0) != [], \"unchanged: unlabelled / not run\")\n"
ADD = (
    "        # 1c (MM 2026-11-24): G-LIVE's evidence form of EVERY result shape -- the Fs gate crashed in check_live on 1b's `losses` list, a caller these lines never reached\n"
    "        for f_ in (fr, fr0, fr_n, frv, {\"ran\": False}, {\"ran\": True, \"labelled\": 0}):\n"
    "            d_ = GC.field_reach_detail(f_)\n"
    "            ok(set(d_) == set(f_) and all(d_[k] == f_[k] for k in f_ if not isinstance(f_[k], dict)), d_)\n"
    "        d_ = GC.field_reach_detail(fr)\n"
    "        ok(set(d_[\"declared\"]) == {\"loss\", \"groups\"} and set(d_[\"zero\"]) == {\"loss\", \"groups\"} and d_[\"losses\"] == fr[\"losses\"] and d_[\"n_active\"] == 2 and d_[\"active_index\"] == 2, d_)\n"
    "        ok(\"det[\\\"field_reach\\\"] = field_reach_detail(fr)\" in inspect.getsource(GC.check_live), \"check_live builds its field_reach evidence through field_reach_detail\")\n"
)
t = sub(t, ANCHOR, ANCHOR + ADD, "unchanged: unlabelled / not run")
MANCHOR = "\"field_reach_active\"),   # always probe idx[0] (the Fs defect)\n"
MADD = (
    "    (\"gate_checks.py\", \"    return {k: ({\\\"loss\\\": v[\\\"loss\\\"], \\\"groups\\\": v[\\\"groups\\\"]} if isinstance(v, dict) else v) for k, v in fr.items()}\\n\",\n"
    "     \"    return {k: (v if k in (\\\"ran\\\", \\\"labelled\\\") else {\\\"loss\\\": v[\\\"loss\\\"], \\\"groups\\\": v[\\\"groups\\\"]}) for k, v in fr.items()}\\n\", \"field_reach_active\"),   # 1c: the 1b-era form (TypeError on `losses`)\n"
    "    (\"gate_checks.py\", \"        det[\\\"field_reach\\\"] = field_reach_detail(fr)\\n\",\n"
    "     \"        det[\\\"field_reach\\\"] = {k: (v if k in (\\\"ran\\\", \\\"labelled\\\") else {\\\"loss\\\": v[\\\"loss\\\"], \\\"groups\\\": v[\\\"groups\\\"]}) for k, v in fr.items()}\\n\", \"field_reach_active\"),   # 1c: check_live bypasses the helper (the Fs crash)\n"
)
i = t.index(MANCHOR) + len(MANCHOR)
if t.count(MANCHOR) != 1:
    raise SystemExit("REFUSING: mutation anchor not unique")
t = t[:i] + MADD + t[i:]
open(TF, "w", encoding="utf-8", newline="\n").write(t)
print(f"ZZPATCH1C_OK {GC} {TF}")
