#!/usr/bin/env python3
"""make_chain_r2.py -- generates the r2 files of the DAG-OACC re-run into r2/box2_ship_r2/:

    chain_v8f_dagoacc_r2.sh         the DAG-OACC chain, REGENERATED the way make_chains.py generated it (every pinned sha256 COMPUTED from the banked artifacts, never retyped), asserted BYTE-IDENTICAL to the first attempt's
                                    chain (box2_ship/chain_v8f_dagoacc.sh), then exact-count edits (EDITS below; a count that is not the declared one is an error)
    run_chain_v8f_dagoacc_r2.cmd    the launcher (CRLF): AFTER_OUT = the FLAT-DAG chain's log (FLAT-DAG holds box 2 now; DAG-OACC re-runs after it)
    wake_v8fdagoacc_r2.sh           the waker, own log / guard / marker names

Not generated here (hand-written, beside them): gate_oaf_assert_r2.py (gate_oaf_assert.py + the one wacc_problems change), oaf_r2_lib.sh (the move-aside functions).
Nothing in this script touches box 2, git, or any installed file: it reads the banked artifacts and writes r2/box2_ship_r2/ only.

    python make_chain_r2.py [--check]       --check: regenerate in memory and compare with what is on disk (rc 1 on any difference)
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import chain_dagoacc_gen as G  # noqa: E402
import make_chains as MC  # noqa: E402

OUT = HERE / "box2_ship_r2"
BASE_SHIP = PARENT / "box2_ship"
NL = chr(10)
R2_HEADER = NL.join([
    "# ============================================================================================================================================================================================================",
    "# r2 -- THE RE-RUN of PREREG_V8F_DAG_OACC (2026-10-27). The first attempt (chain_v8f_dagoacc.sh) passed C-O3 (box w_acc 49.6969 == offline) and the box-2 gate for M2b_s0_dagoacc (rc 0), then STOPPED on the dev-box asserter:",
    "#   acc_probe control read_o_grad_diff = 864.79 (must be exactly 0.0). read_o_grad_diff is not a registered control: it read 0.0 in OACC's gate only because that gate was FROM SCRATCH with O's residual head zero-initialised",
    "#   (residual_zero_init True); DAG-OACC fine-tunes the TRAINED M2b (residual_zero_init False, init_ckpt applied), where read_o MUST receive the term's gradient. The chain below is that chain with ONLY these differences:",
    "#   1. ASSERT = gate_oaf_assert_r2.py (residual_zero_init True -> read_o_grad_diff exactly 0.0; False -> finite and > 0; the four registered controls unchanged; everything else byte-identical)",
    "#   2. its own names, so the first attempt's evidence is never overwritten: chain_v8f_dagoacc_r2.out, v8fdagoacc_r2_guard.out, v8fdagoacc_r2_current.txt, logs/v8f_dagoacc_r2/, repo_v8fdagoacc_r2_tip.tar, TIP_V8FDAGOACC_R2.txt",
    "#   3. r2_aside_all (oaf_r2_lib.sh) is the FIRST action after BOX2_BUSY is taken and the boot probe answered, before anything else touches box 2: the first attempt's gate artifacts are RENAMED to <name>_r1_assert_stop (never deleted,",
    "#      never overwritten): the dev box's pull dir $ART, then (one cmd job) box 2's gate_M2b-dagoacc_s<S> dir + .log and dagpin\\M2b_s<S>_dagoacc.train.argv.txt. A relaunch of r2 keeps r2's own progress (an aside marker on each side).",
    "#      <runs>\\dagoacc (the w_acc record that passed C-O3) is left in place: the queue skips that stage and C-O3 re-checks the re-pulled record.",
    "#   AFTER_OUT = the FLAT-DAG chain's log (run_chain_v8f_dagoacc_r2.cmd). The code-refresh, pins, gates, runs, controls and finish are the first attempt's, line for line.",
    "# ============================================================================================================================================================================================================",
    "",
])
# (what, old, new, count)  -- applied in order to the regenerated first-attempt chain
EDITS = [
    ("header", "# PREREG_V8F_DAG_OACC on box 2: OACC's acceleration-matching term", R2_HEADER + "# PREREG_V8F_DAG_OACC on box 2: OACC's acceleration-matching term", 1),
    ("asserter", "gate_oaf_assert.py", "gate_oaf_assert_r2.py", 2),
    ("logs dir", "L=$B/logs; E=$L/v8f_dagoacc", "L=$B/logs; E=$L/v8f_dagoacc_r2", 1),
    ("guard out", "GUARD_OUT=$L/v8fdagoacc_guard.out", "GUARD_OUT=$L/v8fdagoacc_r2_guard.out", 1),
    ("current", "CUR=$L/v8fdagoacc_current.txt", "CUR=$L/v8fdagoacc_r2_current.txt", 1),
    ("chain log", "chain_v8f_dagoacc.out", "chain_v8f_dagoacc_r2.out", 2),
    ("tar", "repo_v8fdagoacc_tip.tar", "repo_v8fdagoacc_r2_tip.tar", 7),
    ("tip file", "TIP_V8FDAGOACC.txt", "TIP_V8FDAGOACC_R2.txt", 1),
    ("r2 lib", 'source "$HERE_SHIP/oaf_lib.sh"', 'source "$HERE_SHIP/oaf_lib.sh"' + NL + 'source "$HERE_SHIP/oaf_r2_lib.sh"      # r2: r2_aside_all / r2_aside_job_text / r2_aside_local (a NEW file: oaf_lib.sh is the FLAT-DAG chain\'s and is not touched)', 1),
    ("installed", '[ -d "$(dirname "$ART")" ] || { echo "ZZV8FMINI-STOP $(dirname "$ART") does not exist',
     '[ -f "$ASSERT" ] && [ -f "$HERE_SHIP/oaf_r2_lib.sh" ] && type r2_aside_all >/dev/null 2>&1 || { echo "ZZV8FMINI-STOP the r2 files are not installed beside the chain ($ASSERT, $HERE_SHIP/oaf_r2_lib.sh): nothing was touched"; exit 1; }'
     + NL + '[ -d "$(dirname "$ART")" ] || { echo "ZZV8FMINI-STOP $(dirname "$ART") does not exist', 1),
    ("busy text", "v8F-mini PREREG_V8F_DAG_OACC (w_acc rule", "v8F-mini PREREG_V8F_DAG_OACC r2 (the re-run after the first attempt stopped at the gate asserter; w_acc rule", 1),
    ("finish", 'finish "PREREG_V8F_DAG_OACC:', 'finish "PREREG_V8F_DAG_OACC (r2):', 1),
    ("aside", "mkdir -p $E/runs $E/lines      # only now:",
     "# ---- r2: the FIRST action on box 2 after BOX2_BUSY and the boot probe: the first attempt's gate artifacts are moved ASIDE (renamed, never deleted), box 2's and the dev box's (oaf_r2_lib.sh)" + NL
     + "r2_aside_all" + NL + "mkdir -p $E/runs $E/lines      # only now:", 1),
]
WAKER_EDITS = [
    ("chain log", "chain_v8f_dagoacc.out", "chain_v8f_dagoacc_r2.out", 2),
    ("guard out", "v8fdagoacc_guard.out", "v8fdagoacc_r2_guard.out", 2),
    ("gates seen", ".dagoacc_gates_seen", ".dagoacc_r2_gates_seen", 2),
    ("title", "Waker for PREREG_V8F_DAG_OACC's box-2 chain", "Waker for PREREG_V8F_DAG_OACC's box-2 chain, r2 (the re-run)", 1),
]


def edit(text: str, edits) -> str:
    for what, old, new, n in edits:
        c = text.count(old)
        if c != n:
            raise SystemExit(f"EDIT ERROR [{what}]: anchor found {c} times, declared {n}: {old[:90]!r}")
        text = text.replace(old, new)
    return text


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build() -> dict:
    """{filename: bytes} of the r2 chain / launcher / waker, plus the regenerated first-attempt base (asserted equal to the installed first-attempt files)."""
    tmp = Path(tempfile.mkdtemp(prefix="r2gen_", dir="F:/TanitAD-scratch/oaf_r2_tmp"))
    try:
        shutil.copy(BASE_SHIP / "chain_v8f_dagoacc.sh.in", tmp / "chain_v8f_dagoacc.sh.in")
        G.write(tmp, MC.dag_pins(), MC.export_block, MC.sha_file, MC.OFFLINE, MC.FITPLANS)
        base = {n: (tmp / n).read_bytes() for n in ("chain_v8f_dagoacc.sh", "wake_v8fdagoacc.sh", "v8f_guard_dagoacc.sh", "run_chain_v8f_dagoacc.cmd")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for n, b in base.items():
        inst = (BASE_SHIP / n).read_bytes()
        assert b == inst, f"the regenerated first-attempt {n} is not byte-identical to the staged one ({sha(b)[:12]} vs {sha(inst)[:12]}): the pins or the generator changed"
        live = Path("C:/Users/Admin/box2_ship") / n
        if live.exists():
            assert live.read_bytes() == inst, f"the INSTALLED {n} differs from the staged first-attempt file"
    chain = edit(base["chain_v8f_dagoacc.sh"].decode("utf-8"), EDITS)
    waker = edit(base["wake_v8fdagoacc.sh"].decode("utf-8"), WAKER_EDITS)
    run = G.launcher("chain_v8f_dagoacc_r2", "chain_v8f_flatdag",
                     "rem PREREG_V8F_DAG_OACC r2 (the re-run after the first attempt stopped at the gate asserter's read_o control) on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the FLAT-DAG chain's log (FLAT-DAG holds box 2 now); "
                     "a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does.")
    return {"chain_v8f_dagoacc_r2.sh": chain.encode("utf-8"), "wake_v8fdagoacc_r2.sh": waker.encode("utf-8"), "run_chain_v8f_dagoacc_r2.cmd": run, "_base_chain": base["chain_v8f_dagoacc.sh"], "_base_waker": base["wake_v8fdagoacc.sh"]}


def main() -> int:
    files = build()
    if "--check" in sys.argv:
        bad = [n for n, b in files.items() if not n.startswith("_") and (OUT / n).read_bytes() != b]
        print("r2 files match the regeneration" if not bad else f"DIFFERENT: {bad}")
        return 1 if bad else 0
    OUT.mkdir(exist_ok=True)
    for n, b in files.items():
        if n.startswith("_"):
            continue
        (OUT / n).write_bytes(b)
        print("wrote", OUT / n, len(b), "bytes sha256", sha(b)[:16])
    print("first-attempt chain regenerated byte-identical:", sha(files["_base_chain"])[:16], "| waker:", sha(files["_base_waker"])[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
