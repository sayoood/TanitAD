"""Mutation proof for the doc-vs-artifact checker (`test_p3_prebuild_doc_matches_artifacts.py`).

⛔ Unlike every other harness here, this one mutates a **DOCUMENT**, not code. The checker's whole
job is to notice when a figure in `P3_PREBUILD.md` stops agreeing with the JSON it came from, so
the mutations are exactly the mistakes a human makes while editing prose: a mistyped cell, a stale
value left behind, a retracted number quietly promoted back into the live text.

M1 and M2 reproduce the REAL error that motivated the checker — I typed `p95 15 / max 27` where
the banked census said `13 / 20`, inside the edit that was correcting an artifact-reading error.

Usage: python stack/scripts/mutate_p3_doc_check.py [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

REPO = Path(__file__).resolve().parents[2]
REL_PKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate"
DOC = REL_PKG + "/P3_PREBUILD.md"
TEST = "stack/tests/test_p3_prebuild_doc_matches_artifacts.py"

MUTATIONS = [
    # ⛔ THE REAL ERROR: the two cells I typed by hand instead of reading.
    ("M1_p95_cell_mistyped_as_15", DOC,
     "(trainer's order) | **3.989** | 3 | 13 | 20 |",
     "(trainer's order) | **3.989** | 3 | 15 | 20 |",
     ["test_the_delivered_gate_ROW_matches_halfA_s_census"]),
    ("M2_max_cell_mistyped_as_27", DOC,
     "(trainer's order) | **3.989** | 3 | 13 | 20 |",
     "(trainer's order) | **3.989** | 3 | 13 | 27 |",
     ["test_the_delivered_gate_ROW_matches_halfA_s_census"]),
    ("M3_halfB_line_left_at_the_OLD_delivered_value", DOC,
     "halfB, same order: **37.00 / 6.639 / 19.79 / 4.624**",
     "halfB, same order: **37.00 / 6.639 / 19.79 / 5.600**",
     ["test_the_halfB_summary_line_matches_halfB_s_census"]),
    ("M4_drop_pct_reverted_to_the_retracted_value", DOC,
     "| halfA | 4.6342 | **3.9894** | **13.91 %**",
     "| halfA | 4.6342 | **3.9894** | **0.05 %**",
     ["test_the_pad_DROP_ROW_is_recomputable_from_the_census"]),
    # ⛔ a retracted figure promoted back into live prose, where a reader takes it as current
    ("M5_retracted_figure_escapes_into_live_text", DOC,
     "## 3. ⚠️ CORRECTION",
     "The pad drops only 0.05 % of gate boxes on halfA.\n\n## 3. ⚠️ CORRECTION",
     ["test_the_WRONG_figures_survive_only_INSIDE_the_retraction"]),
]


def _copy(dst: Path) -> None:
    (dst / REL_PKG / "raw").mkdir(parents=True)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(REPO / DOC, dst / DOC)
    for h in ("halfA", "halfB"):
        shutil.copy2(REPO / REL_PKG / "raw" / ("p3_census_%s.json" % h),
                     dst / REL_PKG / "raw" / ("p3_census_%s.json" % h))
    shutil.copy2(REPO / TEST, dst / TEST)


def _apply(dst: Path, rel: str, old: str, new: str) -> None:
    p = dst / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if s.count(old) < 1:
        raise SystemExit("anchor absent in %s: %r" % (rel, old[:70]))
    s = s.replace(old, new, 1)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def _run(dst: Path) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST,
                        "-p", "no:cacheprovider"], cwd=str(dst), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "NO_TREE" in out:
        raise SystemExit("the scratch copy is missing the package - no verdict:\n" + out[-800:])
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))), "tail": out[-400:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="p3doc_") as tmp:
        c = Path(tmp) / "M0"
        _copy(c)
        res["M0_control"] = _run(c)
        for name, rel, old, new, red in MUTATIONS:
            d = Path(tmp) / name
            _copy(d)
            _apply(d, rel, old, new)
            r = _run(d)
            r["must_go_red"], r["caught"] = red, all(t in r["failed"] for t in red)
            res[name] = r
    ok = (res["M0_control"]["rc"] == 0 and res["M0_control"]["n_failed"] == 0
          and all(v["caught"] for k, v in res.items() if k != "M0_control"))
    for k, v in res.items():
        tag = ("GREEN (control)" if k == "M0_control" and v["n_failed"] == 0 else
               "CAUGHT" if v.get("caught") else "NOT CAUGHT")
        print("%-46s %s (%d passed / %d failed)" % (k, tag, v["n_passed"], v["n_failed"]))
        if k == "M0_control" and v["n_failed"]:
            print("    CONTROL IS RED - the proof is VOID:\n", v["tail"])
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], "\n    went red:", v["failed"])
    print("\nMUTATION VERDICT:", "ALL CAUGHT, control green" if ok else "FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
