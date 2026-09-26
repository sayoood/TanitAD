"""The suite CLI — ``python -m taniteval.bench <benchmark> --ckpt <path|none> --split <split> [--arms …] [--device …]``.

Exit codes: 0 COMPLETE · 1 FAILED / PARTIAL / contract-invalid · 2 REFUSED (precondition, plugin not
landed, bad arguments) · 3 submission refused · 4 submission approval verified but upload not built.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import contract as C

BENCHMARK_CMDS = C.BENCHMARKS
SUBCOMMANDS = BENCHMARK_CMDS + ("submit", "validate", "gpu-gap", "reaggregate", "legacy")
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m taniteval.bench",
        description=("TanitEval one-command benchmark suite (BUILD_PLAN.md §1). The legacy diagnostic panel "
                     "is `python -m taniteval.bench --model <key>` / `… legacy --help`."))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for b in BENCHMARK_CMDS:
        p = sub.add_parser(b, help=f"run the {b} benchmark")
        p.add_argument("--ckpt", required=True, help="checkpoint path, or 'none' (floors / references only)")
        p.add_argument("--split", required=True)
        p.add_argument("--arms", default="", help="comma list, e.g. A1,STOP,CV (NavSim floors STOP+CV are always added)")
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                       help="auto = CPU unless the GPU-gap launcher finds a gap; cuda = WAIT for a gap")
        p.add_argument("--registry-key", default=None, help="Project Steering/MODEL_REGISTRY.md key of --ckpt")
        p.add_argument("--results-root", default=None, help=f"default {C.RESULTS_ROOT}")
        p.add_argument("--run-id", default=None, help="override (tests); default <utc>-<benchmark>-<ckpt tag>-<hex6>")
        p.add_argument("--no-report", action="store_true", help="skip the taniteval.benchreport hook")
        p.add_argument("--dry-run", action="store_true", help="preflight only; the run dir goes to a scratch root")
        p.add_argument("--scratch", action="store_true",
                       help="write this run under results/bench/_scratch/… — consumers (W4, W5) skip that subtree. "
                            "The results tree is APPEND-ONLY: a run is never deleted, so a run you do not want on "
                            "the leaderboard goes here (or is tombstoned afterwards).")
        p.add_argument("--ram-floor-mb", type=float, default=3000.0)
        p.add_argument("--accept-training-box-load", action="store_true",
                       help="run OUR-MODEL inference on CPU although a training process is alive "
                            "(orchestrator arbitration 2026-09-20). The flag AND the accepted training PIDs are "
                            "recorded in bench_run.json — an exception that is not in the manifest is invisible.")
        if b == "navsim_v2":
            p.add_argument("--frame-bank", default=None, help="frame bank dir for model arms (E2 build_frames.py)")
            p.add_argument("--ckpt-md5", default=None, help="refuse unless the checkpoint md5 equals this")
            # ⭐ W7 2026-09-20: `run_model_arms` has always taken `threads`, and NOTHING reached it —
            # the caller used the default 6 on a 24-core box. MEASURED on navhard's bank (12 scenes,
            # refcv4b, CPU): 6 threads 4.34 s/scene vs 16 threads 2.49 s/scene (1.74x), i.e. 6.6 h vs
            # 3.8 h for one 5,462-scene arm. The value is RECORDED in bench_run.json's argv, so a run's
            # pace can be read against the threads it actually had.
            p.add_argument("--cpu-threads", type=int, default=6,
                           help="torch CPU threads for model-arm inference when the device is CPU (default 6)")
            # ⛔ W7 2026-09-20: navhard's A1 inference is 3.12 h and it COMPLETED; the run then died
            # in SCORING when the wrapper's RAM guard fired (available memory 1035 MB, floor 3000 —
            # the scorer's own RSS was 768 MB, so it was not the consumer). Re-running inference to
            # redo a step that never depended on it is the `t1_eval.py` waste in CLAUDE.md: check for
            # the banked dump BEFORE re-running anything. The adoption is guarded by an IDENTITY
            # check (ckpt md5, arm spec, frame tag, seam row/source counts) and REFUSES on any
            # mismatch — it never silently falls back to recomputing.
            p.add_argument("--reuse-seams", default=None, metavar="DIR",
                           help="adopt model-arm seams from a previous run's raw/model dir instead of "
                                "recomputing them; REFUSES unless the banked seam's identity matches")
            # ⛔ W7 2026-09-21: the navhard scoring died TWICE at ~80 % of its FIRST scored arm on
            # E1_RAM_GUARD_ABORT (948-1,750 MB available, other sessions' jobs). CV + STOP were already
            # scored COMPLETELY on the identical tokens in 06e257; re-scoring them was ~3 h of exposure
            # to that killer. The adoption REFUSES unless the source arm completed and the devkit sha,
            # patch set, metric cache, agent inputs and the token set (by value, with stage labels)
            # all match — see navsim/floor_reuse.py. The artifact records it; it is never a silent skip.
            p.add_argument("--reuse-floors", default=None, metavar="RUN_DIR",
                           help="adopt the CV and STOP floors from a previous COMPLETE run on the same "
                                "split instead of re-scoring them; REFUSES on any identity mismatch")
            # ⭐ W7 2026-09-21: a PASSED ECHO from an attempt that later failed on another arm is as
            # reusable as CV/STOP (its plan is a deterministic, checkpoint-independent function of the
            # export's t0 ego block) — same identity check PLUS the ECHO seam byte-equal to this run's.
            # ⭐ W7 2026-09-21: RE-DERIVE a completed run's artifacts/summary/criteria after an artifact-
            # builder fix, WITHOUT re-scoring: the model arm's scored rows are adopted from the run that
            # scored them, with the plan (seam) byte-equal as the mandatory identity check.
            p.add_argument("--reuse-model-scores", default=None, metavar="RUN_DIR",
                           help="adopt the SCORED rows of model arms from a completed run (plan byte-equal) "
                                "to re-derive artifacts/summary without re-scoring; REFUSES on any mismatch")
            p.add_argument("--reuse-echo", default=None, metavar="RUN_DIR",
                           help="adopt a PASSED ECHO floor from a previous run on the same split (the run "
                                "may have failed on OTHER arms); REFUSES on any identity mismatch")
        if b == "nuscenes_ol":
            # W6 (2026-09-20): ONE convention per run and NO DEFAULT — re-scoring the SAME
            # checkpoints under the two legacy harnesses FLIPS the UniAD/VAD ranking (W6, MEASURED),
            # so a convention the operator did not choose is how two protocols end up in one column.
            p.add_argument("--protocol", choices=("nuScenes_OL_L2_uniad", "nuScenes_OL_L2_stp3"), default=None,
                           help="REQUIRED (no default): the open-loop convention, one run directory per convention")
            p.add_argument("--nuscenes-root", default=None, help="nuScenes data root (else TANITAD_NUSCENES_ROOT)")
            p.add_argument("--construction", choices=("ST", "NT", "EXACT"), default=None,
                           help="history construction for model arms (else TANITAD_NUSCENES_HISTORY, default ST)")
        if b == "internal_t1":
            p.add_argument("--episodes", default=None, help="v2 episode cache dir (<clip>.v2ep.pt)")
            p.add_argument("--labels", default=None, help="v7.2 EVAL labels blob")
            p.add_argument("--config", default=None)
            p.add_argument("--grid", default="2s")
            p.add_argument("--episodes-n", type=int, default=0)
            p.add_argument("--window-stride", type=int, default=0)
            p.add_argument("--analyze-only", default=None, metavar="DUMP_DIR", help="re-analyse a banked dump (0 GPU)")
            p.add_argument("--n-boot", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--t1-args", default="", help="extra flags passed verbatim to refcv3_arm.py")
    s = sub.add_parser("submit", help="REFUSED unless --pi-approval names a recorded PI decision")
    s.add_argument("run_dir")
    s.add_argument("--target", required=True)
    s.add_argument("--benchmark", default=None)
    s.add_argument("--pi-approval", default=None)
    v = sub.add_parser("validate", help="validate a run dir's bench_run.json + summary.json")
    v.add_argument("run_dir")
    r = sub.add_parser("reaggregate", help="re-aggregate a banked PRE-AGGREGATION dump with the devkit's own "
                                           "functions (a scoring run whose aggregation died is NOT a failed run)")
    r.add_argument("--preagg", required=True, help="<arm>_preaggregation.pkl (or .csv)")
    r.add_argument("--split", required=True)
    r.add_argument("--out", required=True, help="output directory")
    sub.add_parser("gpu-gap", help="print the GPU-gap launcher's current verdict (probe only)")
    t = sub.add_parser("tombstone", help="withdraw a run WITHOUT deleting it (the results tree is append-only)")
    t.add_argument("run_dir")
    t.add_argument("--superseded-by", required=True, help="the run_id that replaces it")
    t.add_argument("--why", required=True)
    t.add_argument("--evidence-kept", default="")
    return ap


class _Log:
    def __init__(self):
        self.lines, self.path = [], None

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.lines.append(line)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _module_for(benchmark: str):
    if benchmark == "navsim_v2":
        from .navsim import run_benchmark
        return run_benchmark
    if benchmark == "internal_t1":
        from .internal_t1 import run_benchmark
        return run_benchmark
    from . import plugins
    return plugins.load(benchmark)          # raises PluginNotLanded


def run_benchmark_cmd(a, log=None) -> int:
    log = log or _Log()
    from . import plugins
    from .navsim.profiles import Refusal
    a.benchmark = a.cmd
    a.ckpt = None if str(a.ckpt).lower() == "none" else a.ckpt
    a.arms = [x for x in str(a.arms).split(",") if x.strip()]
    try:
        fn = _module_for(a.benchmark)       # a plugin that has not landed refuses BEFORE any run dir exists
    except plugins.PluginNotLanded as e:
        log(f"⛔ REFUSED: {e}")
        return EXIT_REFUSED
    ck = {"path": None, "sha256": None, "registry_key": a.registry_key}
    if a.ckpt is not None:
        if not Path(a.ckpt).is_file():
            log(f"⛔ REFUSED: --ckpt {a.ckpt} is not a file")
            return EXIT_REFUSED
        ck = {"path": str(Path(a.ckpt).resolve()).replace(os.sep, "/"), "sha256": C.sha256_file(a.ckpt),
              "registry_key": a.registry_key}
        if a.registry_key is None:
            ck["registry_key_status"] = "NOT NAMED by the operator (--registry-key) — a leaderboard row needs it"
    else:
        ck["note"] = "--ckpt none: floors / references only"
    # never a blank in the key's place (orchestrator ruling 2026-09-20) -- computed ONCE, here
    ck["registry_key_display"] = C.ckpt_display(ck)
    root = a.results_root
    if a.dry_run and root is None:
        import tempfile
        root = tempfile.mkdtemp(prefix="tanitad_bench_dryrun_")
    run_id = a.run_id or C.make_run_id(a.benchmark, C.ckpt_tag(a.ckpt, a.registry_key))
    run = C.RunDir(a.benchmark, a.split, run_id, root=Path(root) if root else None,
                   scratch=bool(getattr(a, "scratch", False))).create()
    log.path = run.p("raw/bench.log")
    ctx = C.BenchContext(a, run, log=log)
    gh = C.git_head()
    ctx.rec["git_head"] = gh["sha"]
    bench_dir = Path(__file__).resolve().parent
    ctx.rec["git"] = {**gh, "suite_code_blobs": C.code_blobs(sorted(p for p in bench_dir.rglob("*.py")
                                                                    if "__pycache__" not in p.parts))}
    ctx.rec["ckpt"] = ck
    from . import SUITE_VERSION
    ctx.rec["suite_version"] = SUITE_VERSION
    log(f"[bench] run {run_id} -> {run.path}")
    code = EXIT_OK
    try:
        fn(ctx)
        if ctx.rec.get("status") == "FAILED":
            code = EXIT_FAILED
        elif ctx.rec.get("status") in ("PARTIAL",):
            code = EXIT_FAILED
        elif ctx.rec.get("status") == "REFUSED":
            # ⚠ VOCABULARY (E1, 2026-09-20): a PASSING dry run used to print BENCH_STATUS=REFUSED — a
            # FAILURE WORD ON A SUCCESS, and exactly the kind of thing that gets copied into a report as a
            # failure. Normalised HERE, at the single seam every benchmark returns through, so the plugins
            # (incl. W3's navsim_v1) are fixed without editing files W1 does not own. REFUSED now means a
            # real refusal only; DRY_RUN_PASSED is a PASS and exits 0.
            if ctx.rec.get("dry_run"):
                ctx.rec["status"], code = "DRY_RUN_PASSED", EXIT_OK
            else:
                code = EXIT_REFUSED
    except (Refusal, C.DeviceRefused) as e:
        ctx.rec["status"], ctx.rec["refusal"] = "REFUSED", str(e)
        log(f"⛔ REFUSED: {e}")
        code = EXIT_REFUSED
    except C.ContractError as e:
        ctx.rec["status"], ctx.rec["contract_errors"] = "FAILED", e.errors
        log(f"⛔ CONTRACT: {e}")
        code = EXIT_FAILED
    except Exception as e:                                               # noqa: BLE001
        ctx.rec["status"] = "FAILED"
        ctx.rec["exception"] = f"{type(e).__name__}: {e}"[:2000]
        run.p("raw/traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"⛔ FAILED: {type(e).__name__}: {e}")
        code = EXIT_FAILED
    if ctx.rec.get("status") in ("COMPLETE", "PARTIAL") and not a.no_report:
        from .report_hook import render
        ctx.rec["report"] = render(run.path)
    else:
        ctx.rec["report"] = {"status": "SKIPPED", "reason": "--no-report" if a.no_report else f"run {ctx.rec.get('status')}"}
    log(f"[bench] report: {ctx.rec['report']['status']}")
    used = ctx.rec.pop("device_used_override", None)
    detail = ctx.rec.pop("device_detail", None)
    if ctx.gpu is not None:
        detail = ctx.gpu.record()
        used = detail["used"]
    ctx.rec["device"] = {"requested": a.device, "used": used or ("none" if a.ckpt is None else "cpu"),
                         "scoring": "cpu", "gpu_gap": detail or {"status": "NOT_INVOKED",
                                                                 "reason": "no model inference in this run"}}
    ctx.rec.setdefault("protocol", None)
    ctx.rec.setdefault("devkit", None)
    ctx.rec.setdefault("split", {"name": a.split, "n_scenes": None, "n_logs": None})
    ctx.rec.setdefault("stamps", None)
    ctx.rec.setdefault("claim_bearing", None)
    ctx.rec["utc_end"] = C.utc_now()
    ctx.rec["wall_s"] = round(time.time() - ctx.t0, 1)
    ctx.rec["files"] = run.finalize_files()
    errs = C.validate_bench_run(ctx.rec) if ctx.rec.get("status") in ("COMPLETE", "PARTIAL") else []
    if errs:
        ctx.rec["contract_errors"] = errs
        ctx.rec["status"] = "FAILED"
        code = EXIT_FAILED
        log(f"⛔ bench_run.json violates the contract: {errs[:6]}")
    run.write_json("bench_run.json", ctx.rec)
    log(f"[bench] {ctx.rec['status']} in {ctx.rec['wall_s']} s -> {run.path}")
    print(f"BENCH_RUN_DIR={str(run.path).replace(os.sep, '/')}", flush=True)
    print(f"BENCH_STATUS={ctx.rec['status']}", flush=True)
    print(f"BENCH_RETRYABLE={1 if ctx.rec.get('retryable') else 0}", flush=True)
    return code


def validate_cmd(run_dir: str) -> int:
    rd = Path(run_dir)
    out = {}
    for name, fn in (("bench_run.json", C.validate_bench_run), ("summary.json", C.validate_summary)):
        p = rd / name
        if not p.exists():
            out[name] = ["MISSING"]
            continue
        out[name] = fn(json.loads(p.read_text(encoding="utf-8")))
    print(json.dumps(out, indent=1))
    return EXIT_OK if all(not v for v in out.values()) else EXIT_FAILED


def reaggregate_cmd(preagg: str, split: str, out: str) -> int:
    """Run the devkit-side re-aggregation in the NavSim venv (BUILD_PLAN §1: a completed scoring run
    whose AGGREGATION died must be recoverable, never re-scored)."""
    import subprocess
    from .navsim import profiles as P
    if not Path(preagg).exists():
        print(f"⛔ no pre-aggregation dump at {preagg}")
        return EXIT_REFUSED
    script = P.DEVKIT_SIDE / "reaggregate.py"
    cmd = [str(P.PY), str(script), "--preagg", str(preagg), "--split", split, "--out", str(out)]
    print("CMD: " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=P.scorer_env(P.EXP_ROOT)).returncode


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_REFUSED if (e.code not in (0, None)) else EXIT_OK
    if a.cmd in BENCHMARK_CMDS:
        return run_benchmark_cmd(a)
    if a.cmd == "submit":
        from .submission import submit
        return submit(a.run_dir, target=a.target, benchmark=a.benchmark, decision_id=a.pi_approval)
    if a.cmd == "validate":
        return validate_cmd(a.run_dir)
    if a.cmd == "gpu-gap":
        from .gpu_gap import main as gmain
        return gmain()
    if a.cmd == "tombstone":
        p = C.write_tombstone(a.run_dir, superseded_by=a.superseded_by, why=a.why, evidence_kept=a.evidence_kept)
        print(f"TOMBSTONE={str(p).replace(os.sep, '/')} (the directory is kept; nothing was deleted)")
        return EXIT_OK
    if a.cmd == "reaggregate":
        return reaggregate_cmd(a.preagg, a.split, a.out)
    ap.error(f"unknown command {a.cmd}")
    return EXIT_REFUSED
