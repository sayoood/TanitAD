"""Mutation proof for the per-arm gate (`stack/scripts/p_check_arm.py`).

This checker stands between a finished ~11 h arm and `p_runner`'s `move_aside`. If it passes an
arm it should refuse, a contaminated result enters the panel; if it refuses one it should pass,
11 h is thrown away. It had no tests at all until 2026-09-20.

⛔ M1 IS THE ONE THAT MOTIVATED THE HARNESS: restore the SILENT skip of an unparseable metrics
line. A bare ``except: continue`` cannot tell "a different kind of row" from "this file is
truncated". MEASURED in the pinned tree: every metrics write is a COMPLETE line followed immediately by `log.flush()` (`refc_v3_train.py` :7180-7181, and four sibling sites 7093/7098, 7261/7263, 7273/7274, 7324/7328; the file is opened once in append mode at :6958). ⇒ A hard kill therefore loses the final ROW, not half of one: killed between rows, or between `write()` and `flush()`, the file PARSES PERFECTLY and is simply SHORT. The only partial-line window is inside the flush syscall itself. ⛔ SO FOR A SIGKILL OR A cgroup-OOM — this box's documented trainer death — `n_unparseable_metrics_lines` is STRUCTURALLY BLIND, and the STEP-COUNT check is what fires. The unparseable counter's real triggers are power loss, a filesystem fault, or a different writer that does not flush per row.

⚠️ A7-IN-s0, the arm paused on 2026-09-20, is the CLEAN counter-example and was VERIFIED rather than assumed: 107 lines, **0 unparseable**, file ends with a newline, last row step 1,070 = 107 x 10. ⚠️ I had written that this arm's kill DID truncate it; that was FALSE — a plausible mechanism asserted about a specific file without opening it. Why it stayed clean is NOT established: the ordered kill (trainer last, by explicit PID) and a per-row flush are both candidates and I have separated neither.

⭐ THE CLASS, named with the Master Mind (2026-09-20) — the swallow living in the CHECKER's own error
handling rather than in the code under test, which is how a checker goes blind while looking
hardened.

Usage: python stack/scripts/mutate_p_check_arm.py [--out <json>]
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
MOD = "stack/scripts/p_check_arm.py"
TEST = "stack/tests/test_p_check_arm.py"

NB = "                n_bad += 1" + chr(10) + "                continue" + chr(10)
HEADROW = ("            if HEAD in d:" + chr(10) + "                best = d" + chr(10))

MUTATIONS = [
    ("M1_unparseable_line_is_SILENTLY_skipped", MOD, NB,
     "                continue  # MUTATION M1: swallowed again" + chr(10),
     ["test_an_UNPARSEABLE_line_is_COUNTED_and_REFUSES"]),
    ("M2_counted_but_does_NOT_refuse", MOD,
     "        if n_bad:" + chr(10),
     "        if False:  # MUTATION M2" + chr(10),
     ["test_an_UNPARSEABLE_line_is_COUNTED_and_REFUSES"]),
    # ⛔ THE GUARD THAT ACTUALLY CATCHES AN OOM-KILLED RUN. The unparseable counter cannot see a
    # missing final row; this check can, and it had no test until the writer was read.
    ("M7_the_final_step_is_not_checked", MOD,
     '        if str(row.get("step")) != str(steps):' + chr(10),
     "        if False:  # MUTATION M7" + chr(10),
     ["test_a_TRUNCATED_RUN_missing_its_final_row_is_INVALID"]),
    ("M3_the_seed_is_not_checked", MOD,
     "        if argv and str(got_seed) != str(seed):" + chr(10),
     "        if False:  # MUTATION M3" + chr(10),
     ["test_a_WRONG_SEED_is_INVALID"]),
    ("M4_label_density_identity_not_checked", MOD,
     '        if not rep["identity_matched_equals_target"]:' + chr(10),
     "        if False:  # MUTATION M4" + chr(10),
     ["test_a_BROKEN_label_density_identity_is_INVALID"]),
    ("M5_takes_the_last_LINE_not_the_last_headline_row", MOD, HEADROW,
     "            best = d  # MUTATION M5: any row wins" + chr(10),
     ["test_the_LAST_row_carrying_the_headline_wins_not_the_last_line"]),
    ("M6_a_non_finite_headline_passes", MOD,
     "        if h is None or not math.isfinite(float(h)):" + chr(10),
     "        if False:  # MUTATION M6" + chr(10),
     ["test_a_NON_FINITE_headline_is_INVALID"]),
]

CONFTEST = (
    "import sys" + chr(10) +
    "from pathlib import Path" + chr(10) +
    "_R = Path(__file__).resolve().parents[2]" + chr(10) +
    'for _p in ("stack", "stack/scripts"):' + chr(10) +
    "    sys.path.insert(0, str(_R / _p))" + chr(10) +
    "import p_check_arm as _m" + chr(10) +
    '_f = getattr(_m, "__file__", None)' + chr(10) +
    "if not _f or not str(Path(_f).resolve()).startswith(str(_R)):" + chr(10) +
    '    raise SystemExit("WRONG-DISK IMPORT: p_check_arm -> %s not under %s" % (_f, _R))' + chr(10)
)


def _copy(dst: Path) -> None:
    (dst / "stack" / "scripts").mkdir(parents=True)
    (dst / "stack" / "tests").mkdir(parents=True)
    shutil.copy2(REPO / MOD, dst / MOD)
    shutil.copy2(REPO / TEST, dst / TEST)
    (dst / "stack" / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")


def _apply(dst: Path, rel: str, old: str, new: str) -> None:
    p = dst / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if s.count(old) != 1:
        raise SystemExit("anchor count %d in %s: %r" % (s.count(old), rel, old[:70]))
    s = s.replace(old, new, 1)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def _run(dst: Path) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join(str(dst / p) for p in ("stack", "stack/scripts")))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", TEST,
                        "-p", "no:cacheprovider"], cwd=str(dst), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    if "WRONG-DISK IMPORT" in out:
        raise SystemExit("wrong-disk import - no verdict:" + chr(10) + out[-1200:])
    failed = sorted(set(re.findall(r"(?:FAILED|ERROR) \S+::(\w+)", out)))
    return {"rc": r.returncode, "failed": failed, "n_failed": len(failed),
            "n_passed": len(set(re.findall(r"PASSED \S+::(\w+)", out))), "tail": out[-400:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    with tempfile.TemporaryDirectory(prefix="pc_mut_") as tmp:
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
            print("    CONTROL IS RED - the proof is VOID:" + chr(10), v["tail"])
        if "caught" in v and not v["caught"]:
            print("    must go red:", v["must_go_red"], chr(10), "   went red:", v["failed"])
    print(chr(10) + "MUTATION VERDICT:",
          "ALL CAUGHT, control green" if ok else "FAILED")
    if a.out:
        Path(a.out).write_text(json.dumps({"ok": ok, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
