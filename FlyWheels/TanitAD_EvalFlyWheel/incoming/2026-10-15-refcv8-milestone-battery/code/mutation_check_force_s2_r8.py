"""mutation_check_force_s2_r8.py -- proves test_force_s2_r8.py can go RED: each mutation reintroduces ONE plausible defect into a COPY of
force_s2_reroll_r8.py (the original is never touched) and the test file is run against that copy.  A mutation whose target text is not found exactly
once is reported NOT APPLIED (a check that silently mutates nothing is green forever).

    python mutation_check_force_s2_r8.py [--only N] > raw/force_s2/MUTATION_CHECK_test_force_s2_r8.txt

Needs PYTHONPATH = <PKG>/code (+ the launch tree for the stack).  CPU only; each RED run stops at the first failing test."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Why a runner: force_lon_reroll_r8 prepends the REAL code dir to sys.path when it is imported, and the test imports it BEFORE force_s2_reroll_r8 -- so a
#: plain `pytest copy/test.py` silently imports the ORIGINAL module and every mutation "survives".  The runner imports the module under test FIRST from the
#: directory it is given and asserts its __file__ is there (a harness whose mutated copy is never loaded is green forever).
RUNNER = ("import sys, importlib; d = sys.argv[1].replace(chr(92), '/'); sys.path.insert(0, sys.argv[1]); m = importlib.import_module('force_s2_reroll_r8'); "
          "assert m.__file__.replace(chr(92), '/').startswith(d), 'module under test is NOT the one in ' + d + ': ' + m.__file__; "
          "import pytest; sys.exit(pytest.main(sys.argv[2:]))")
SRC = HERE / "force_s2_reroll_r8.py"
TEST = HERE / "test_force_s2_r8.py"
SRC_SH = HERE / "force_s2_launcher_r8.sh"
TEST_SH = HERE / "test_force_s2_launcher_r8.py"

MUTATIONS = [
    ("closure point threshold 0.50 -> 0.49", "CLOSE_POINT, CLOSE_LO_GT, DEC_HI_LT = 0.50, 0.0, 0.25", "CLOSE_POINT, CLOSE_LO_GT, DEC_HI_LT = 0.49, 0.0, 0.25"),
    ("decoder bound 0.25 -> 0.30", "CLOSE_POINT, CLOSE_LO_GT, DEC_HI_LT = 0.50, 0.0, 0.25", "CLOSE_POINT, CLOSE_LO_GT, DEC_HI_LT = 0.50, 0.0, 0.30"),
    ("F_slow lower bound: lo > 0 -> lo >= 0", 'fslow["lo"] > CLOSE_LO_GT and f1["point"] < CLOSE_POINT', 'fslow["lo"] >= CLOSE_LO_GT and f1["point"] < CLOSE_POINT'),
    ("TIMING needs F1 < 0.50: -> <= 0.50", 'and f1["point"] < CLOSE_POINT:', 'and f1["point"] <= CLOSE_POINT:'),
    ("IDENTITY lower bound: lo > 0 -> lo >= 0", 'f1["point"] >= CLOSE_POINT and f1["lo"] > CLOSE_LO_GT', 'f1["point"] >= CLOSE_POINT and f1["lo"] >= CLOSE_LO_GT'),
    ("DECODER: both uppers < 0.25 -> F1 upper <= 0.25", 'f1["hi"] < DEC_HI_LT and fslow["hi"] < DEC_HI_LT', 'f1["hi"] <= DEC_HI_LT and fslow["hi"] < DEC_HI_LT'),
    ("DECODER: both -> only F_slow", 'f1["hi"] < DEC_HI_LT and fslow["hi"] < DEC_HI_LT', 'fslow["hi"] < DEC_HI_LT'),
    ("undefined closure no longer refuses", 'if not c or c.get("point") is None or c.get("lo") is None or c.get("hi") is None:', "if False:"),
    ("slow classes swapped (HOLD <-> BRAKE_TO)", 'SLOW_CLASS6 = {"S2A": 4, "S2B": 2}', 'SLOW_CLASS6 = {"S2A": 2, "S2B": 4}'),
    ("S2A loses 'stopping'", 'SET_DEFS = {"S2A": ("slow_both", "stopping"), "S2B": ("decelerating",)}', 'SET_DEFS = {"S2A": ("slow_both",), "S2B": ("decelerating",)}'),
    ("S2B gains 'cruising'", 'SET_DEFS = {"S2A": ("slow_both", "stopping"), "S2B": ("decelerating",)}',
     'SET_DEFS = {"S2A": ("slow_both", "stopping"), "S2B": ("decelerating", "cruising")}'),
    ("chunks no longer single-set", 'e["chunk"] = int(si * chunks_per_set + order[k] % chunks_per_set)', 'e["chunk"] = int(order[k] % chunks_per_set)'),
    ("rebinding is a no-op", "FL.CRUISE6 = SLOW_CLASS6[set_name]", "FL.CRUISE6 = old"),
    ("driver not pointed at this file", "FL.__file__ = str(Path(__file__).resolve())", "pass"),
    ("content check expects CRUISE for F_slow", '"F2": e["slow_class6"]', '"F2": 1'),
    ("internals guard needle drifts", "'\"F2\": (0, CRUISE6)' not in src", "'\"F2\": (0, NOPE6)' not in src"),
    ("closure denominator ignores the echo", "ade[base][m] - ade_echo[m], eid[m], SD)\n        r[\"metric\"]", "ade[base][m], eid[m], SD)\n        r[\"metric\"]"),
    ("KB passes in the wrong direction", 'bool(kb1["hi"] < 0)', 'bool(kb1["lo"] > 0)'),
    ("E1 arithmetic shift sign flipped", "dprime[sets_full[sn]] += shift[sn]", "dprime[sets_full[sn]] -= shift[sn]"),
    ("F1 closure on all windows (NaN-fill)", "f1_has = union & np.isfinite(ade[\"F1\"])", "f1_has = union"),
    ("S2 slot count 4 -> 8", "S2_NSLOT = 4 ", "S2_NSLOT = 8 "),
    ("F1 class map identity", "c, why = FL.f1_class(int(ref[\"lon_v7\"][i]))", "c, why = (1, None)"),
    ("control subset never gets N2/Fown", 'e["arms"] = e["arms"] + ["N2", "Fown"]', 'e["arms"] = e["arms"] + []'),
]

#: mutations of the LAUNCHER (--target launcher): each reintroduces one wait-gate / sequencing defect
MUTATIONS_SH = [
    ("fp64 gate: terminal markers counted ANYWHERE (the stale-marker trap)",
     "c=$(awk '/ZZF64WAITZZ/{c=0; next} /ZZF64(DONE|ARMFAIL|COMBINEFAIL|TIMEOUT)ZZ/{c++} END{print c+0}' \"$F\" 2>/dev/null)",
     "c=$(grep -c -E 'ZZF64(DONE|ARMFAIL|COMBINEFAIL|TIMEOUT)ZZ' \"$F\" 2>/dev/null)"),
    ("fp64 gate: TIMEOUT is no longer terminal", "(DONE|ARMFAIL|COMBINEFAIL|TIMEOUT)ZZ/{c++}", "(DONE|ARMFAIL|COMBINEFAIL)ZZ/{c++}"),
    ("free commit: a single good check is enough", "h2=$(hr)", "h2=$h1"),
    ("free commit: exactly the threshold no longer counts", 'if [ "$h1" -ge "$NEED" ]; then', 'if [ "$h1" -gt "$NEED" ]; then'),
    ("free commit: an unreadable probe counts as plenty", "case \"$v\" in ''|*[!0-9]*) v=0;; esac", "case \"$v\" in ''|*[!0-9]*) v=99;; esac"),
    ("force_lon staleness window removed", "find \"$FLL\" -mmin -15", "find \"$FLL\" -mmin -999999"),
    ("queue-end gate skipped", "q_done && g1=1", "g1=1"),
    ("single-launcher guard never sees a live holder", 'ps -p "$op" >/dev/null 2>&1; then', 'false; then'),
    ("roll gets three passes", "while [ $pass -lt 2 ]; do", "while [ $pass -lt 3 ]; do"),
    ("analyze judged by the json alone", ' && grep -q ZZFS2ANALYZEDONEZZ "$SCRD/analyze_full.log"', ""),
    ("select judged without its own end marker", ' && grep -q ZZFS2SELECTOKZZ "$SCRD/select.log"', ""),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--target", choices=("module", "launcher"), default="module")
    a = ap.parse_args()
    global SRC, TEST, MUTATIONS
    if a.target == "launcher":
        SRC, TEST, MUTATIONS = SRC_SH, TEST_SH, MUTATIONS_SH
    src = SRC.read_text(encoding="utf-8")
    py = sys.executable
    print(f"mutation check of {TEST.name} ({time.strftime('%F %T%z')}); source sha-free copy per mutation, original untouched")
    t0 = time.time()
    def cmd_for(d: Path, test: Path, extra: list) -> list:
        if a.target == "module":
            return [py, "-X", "utf8", "-B", "-c", RUNNER, str(d), "-q", "-p", "no:cacheprovider", *extra, str(test)]
        return [py, "-X", "utf8", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra, str(test)]
    base = subprocess.run(cmd_for(HERE, TEST, []), capture_output=True, text=True, cwd=str(HERE))
    last = [ln for ln in base.stdout.strip().splitlines() if ln.strip()][-1] if base.stdout.strip() else "(no output)"
    print(f"unmutated: {last}")
    n_red = n_green = n_na = 0
    for i, (name, old, new) in enumerate(MUTATIONS):
        if a.only >= 0 and i != a.only:
            continue
        if src.count(old) != 1:
            print(f"[{i:2d}] NOT APPLIED ({src.count(old)} occurrences of the target): {name}")
            n_na += 1
            continue
        with tempfile.TemporaryDirectory(prefix="fs2mut_") as td:
            td = Path(td)
            (td / SRC.name).write_bytes(src.replace(old, new).encode("utf-8"))      # bytes: write_text would CRLF-translate a .sh on Windows
            shutil.copy(TEST, td / TEST.name)
            r = subprocess.run(cmd_for(td, td / TEST.name, ["-x"]), capture_output=True, text=True, cwd=str(td))
        out = r.stdout + r.stderr
        m = re.search(r"FAILED [^\s]*::(\S+)", out)
        red = ("failed" in out.lower() and bool(m)) or " error" in out.lower()
        if red:
            n_red += 1
            print(f"[{i:2d}] RED   ({m.group(1) if m else 'collection error'}): {name}")
        else:
            n_green += 1
            print(f"[{i:2d}] GREEN (SURVIVED -- the test does not guard this): {name}")
    print(f"summary: {n_red} RED, {n_green} SURVIVED, {n_na} NOT APPLIED of {len(MUTATIONS) if a.only < 0 else 1}; wall {time.time() - t0:.0f}s")
    return 0 if (n_green == 0 and n_na == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
