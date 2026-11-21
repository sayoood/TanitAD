#!/usr/bin/env python3
"""fp_chain_v2.py -- the free-point chain with a STARTUP GATE on its gated controls.

An IMPROVED COPY of `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-17-free-point-navtest/code/fp_chain.py`.
That file is NOT edited: it is the banked record of what ran.  This copy adopts `taniteval/tools/chain_controls.py`
(INSTFIX, 2026-10-18) -- the answer to RESULT s5 of the free-point read:

    "A gate that reads a control file no code writes reports the control missing ... no stage asserted that every
     control it gates on has a producer.  Fix for any successor chain: the `checks` stage must refuse at STARTUP,
     not at the read, when a gated control has no producer stage."

WHAT CHANGED against the banked chain (the stage logic itself is unchanged):
  1. `build_plan(w)` declares, per stage, what it PRODUCES and what it GATES ON.  The `checks` gate list is NOT typed
     here: it is read off `fp_checks.py` by `chain_controls.scan_reads`, so it cannot drift from the script.
  2. `require_producers()` runs FIRST -- before `--wait-for-pull` sleeps -- and the chain exits 2 with
     `ZZCHAIN_DONE REFUSED_AT_STARTUP` (no stage run) if any gated control has no producer.
  3. Three stages the banked chain lacked: `ic1` (fp_capture.py check-a9 -> IC1.json), `d3_smoke` (fp_smoke.py --swap-cams)
     and `d3_link` (smoke_swap/smoke_result.json -> D3.json, the `cp` that was done by hand in raw/run_d3.sh).
  4. Before each stage the gates it needs are verified present (a pull that stopped is refused AT that stage); after each
     stage its declared products are verified to exist and be non-empty.

    python fp_chain_v2.py --work W [--wait-for-pull] [--parallel 2] [--check-only]

`--check-only` runs the startup gate and exits (0 = every gated control has a producer) without running anything.
The log protocol is the banked one: `ZZCHAIN <stage> <OK|FAIL ...> <s>` lines and a last `ZZCHAIN_DONE <status>`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.environ.get("TANITAD_REPO") or next((p for p in ("D:/Projects/TanitAD", "E:/Projects/TanitAD") if os.path.isdir(p)), "D:/Projects/TanitAD")
sys.path.insert(0, f"{REPO}/taniteval/tools")
import chain_controls as CC  # noqa: E402

FPCODE = (f"{REPO}/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/"
          "2026-10-17-free-point-navtest/code")
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
D3_LOG = "2021.06.28.15.02.02_veh-38_02398_02848"          # raw/run_d3.sh: the log the swap smoke ran on
READ_GATES = ("score_logs/*.status.json", "IC1.json", "IC3_IC4.json", "IC5.json", "IC6.json", "D1_D3.json")


def _scan(script):
    with open(f"{FPCODE}/{script}", encoding="utf-8") as fh:
        return fh.read()


def build_plan(w: str | None = None) -> CC.ChainPlan:
    """The declared plan.  `w` is unused by the declarations (all paths are work-relative); it is kept for symmetry."""
    checks_gates = tuple(sorted(CC.scan_reads(_scan("fp_checks.py"))["reads"]))     # picks_report.json, points.npz, seam_*.npz, ic5_forward.json, IC1.json, D3.json
    stages = (
        CC.Stage("ic5_prelude", gates_on=("out/chunk_000.npz",), produces=("ic5_forward.json",)),            # in-process: meta copy
        CC.Stage("picks", script="fp_picks.py", produces=("picks_report.json", "points.npz", "seam_*.npz")),
        CC.Stage("score", script="fp_score.py", produces=("score_logs/*.status.json", "score_status.json")),
        CC.Stage("subset", script="fp_score.py", produces=("seam_*_sub*.npz",)),
        CC.Stage("ic1", script="fp_capture.py", gates_on=("inputs_full.json",), produces=("IC1.json",)),
        CC.Stage("d3_smoke", script="fp_smoke.py", gates_on=(f"cap/inputs_{D3_LOG}.json",), produces=("smoke_swap/smoke_result.json",)),
        CC.Stage("d3_link", gates_on=("smoke_swap/smoke_result.json",), produces=("D3.json",)),             # in-process: the former hand `cp`
        CC.Stage("checks", script="fp_checks.py", gates_on=checks_gates,
                 produces=("IC3_IC4.json", "IC5.json", "IC6.json", "D1_D3.json", "gates_summary.json")),
        CC.Stage("read", script="fp_read.py", gates_on=READ_GATES, produces=("result/freepoint_result.json", "result/RESULT_TABLES.md")),
    )
    return CC.ChainPlan(
        stages=stages,
        external={"inputs_full.json": "fp_capture.py merge (stage A, dev box, before the chain)",
                  f"cap/inputs_{D3_LOG}.json": "fp_capture.py one --log (stage A, dev box, before the chain)"},
        arriving={"out/chunk_000.npz": "fp_ship.py pull (the pod stage, while this chain waits)"})


def _argv(stage: str, w: str, parallel: int):
    return {
        "picks": [f"{FPCODE}/fp_picks.py", "build", "--out-dir", w, "--chunks", f"{w}/out"],
        "score": [f"{FPCODE}/fp_score.py", "score", "--work", w, "--points", "BASE,C0A,R0B,M25", "--parallel", str(parallel)],
        "subset": [f"{FPCODE}/fp_score.py", "subset", "--work", w, "--point", "C0A", "--n", "1000"],
        "ic1": [f"{FPCODE}/fp_capture.py", "check-a9", "--out-dir", w],
        "d3_smoke": [f"{FPCODE}/fp_smoke.py", "--work", w, "--log", D3_LOG, "--k", "8", "--device", "cpu", "--swap-cams"],
        "checks": [f"{FPCODE}/fp_checks.py", "all", "--work", w],
        "read": [f"{FPCODE}/fp_read.py", "read", "--work", w, "--out-dir", f"{w}/result"],
    }[stage]


def _run(stage, argv, w, log):
    t0 = time.time()
    rc = subprocess.call([PY, "-I", "-X", "utf8", *argv], stdout=open(f"{w}/chain_{stage}.out", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    line = f"ZZCHAIN {stage} {'OK' if rc == 0 else 'FAIL rc=' + str(rc)} {time.time() - t0:.0f}s"
    print(line, flush=True)
    open(log, "a").write(line + "\n")
    return rc


def _ic5_prelude(w):
    import numpy as np
    z = np.load(f"{w}/out/chunk_000.npz", allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    json.dump(meta.get("ic5_forward", {"missing": True}), open(f"{w}/ic5_forward.json", "w"), indent=1)
    return 0


def _d3_link(w):
    shutil.copyfile(f"{w}/smoke_swap/smoke_result.json", f"{w}/D3.json")        # unchanged copy, as raw/run_d3.sh did by hand
    return 0


#: stages that are skipped when their products already exist (resumable); `checks` and `read` ALWAYS re-run.
_SKIP_IF_DONE = {"picks": "picks_report.json", "ic1": "IC1.json", "d3_smoke": "smoke_swap/smoke_result.json", "d3_link": "D3.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--wait-for-pull", action="store_true")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--check-only", action="store_true", help="run the startup gate and exit")
    a = ap.parse_args()
    w = a.work
    log = f"{w}/chain.log"
    plan = build_plan(w)
    # ---- THE STARTUP GATE: before --wait-for-pull sleeps, before any stage -----------------------------------------
    try:
        ledger = CC.require_producers(plan, work=w, script_root=FPCODE)
        for st in plan.stages:                                                      # the declaration may not drift from the script
            if st.name in ("checks", "read"):
                CC.raise_on(CC.check_declared_reads(st, _scan(st.script)), f"stage {st.name}: declared gates drifted from {st.script}")
    except CC.ChainPlanError as ex:
        msg = "ZZCHAIN_DONE REFUSED_AT_STARTUP"
        print(str(ex), file=sys.stderr)
        print(msg, flush=True)
        if os.path.isdir(w):
            open(log, "a").write(msg + "\n" + str(ex) + "\n")
        return 2
    print("ZZCHAIN_PLAN_OK " + json.dumps(ledger), flush=True)
    if a.check_only:
        return 0
    if a.wait_for_pull:
        while not (os.path.exists(f"{w}/ship.log") and "ZZSHIP_ALL_PULLED" in open(f"{w}/ship.log", errors="replace").read()):
            time.sleep(60)
    for st in plan.stages:
        done = _SKIP_IF_DONE.get(st.name)
        if done and os.path.exists(f"{w}/{done}"):
            continue
        miss = CC.verify_gates_present(st, w)                                       # a control that never arrived is refused HERE
        if miss:
            open(log, "a").write(f"ZZCHAIN_DONE REFUSED_AT_{st.name}\n" + "\n".join(miss) + "\n")
            print("\n".join(miss), file=sys.stderr)
            return 1
        if st.name == "ic5_prelude":
            rc = _ic5_prelude(w)
        elif st.name == "d3_link":
            rc = _d3_link(w)
        else:
            rc = _run(st.name, _argv(st.name, w, a.parallel), w, log)
        if rc != 0:
            open(log, "a").write(f"ZZCHAIN_DONE FAILED_AT_{st.name}\n")
            return 1
        short = CC.verify_stage_outputs(st, w)                                      # "stage OK" may not mean "stage wrote nothing"
        if short:
            open(log, "a").write(f"ZZCHAIN_DONE FAILED_AT_{st.name}\n" + "\n".join(short) + "\n")
            print("\n".join(short), file=sys.stderr)
            return 1
    status = json.load(open(f"{w}/gates_summary.json"))["status"]
    open(log, "a").write(f"ZZCHAIN_DONE {status}\n")
    print(f"ZZCHAIN_DONE {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
