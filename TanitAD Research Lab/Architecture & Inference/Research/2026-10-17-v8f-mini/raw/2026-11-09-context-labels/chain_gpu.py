"""Detached GPU chain: smoke (20 clips, all tiers) -> sanity gate -> full run for the train4512 + eval clips in 30-minute lock-held jobs -> merge.
Every GPU job is one `extract_context.py` process that takes the shared dev-box GPU lock (gpu_lock.py, wait_idle pattern) and releases it when its
30 minutes end, so other sessions' chains can queue between jobs. Markers: F:/strat_out/ZZ_CHAIN_DONE, ZZ_CHAIN_FAILED, STOP (create it to stop).

Launch (detached, see launch_detached.ps1):
  chain_gpu.py                                  PILOT only (default): about 200 clips, ends at F:/strat_out/pilot/ZZ_PILOT_DONE
  chain_gpu.py --full-run-approved-by-pi [--skip-smoke]   the full run; only the Master Mind passes it, after the PI validated the pilot.
A pilot cannot escalate: without the flag the code path that reads splits/clips_all.json is never entered (tests/test_pilot_chain.py).
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT, SMOKE = "F:/strat_out", "F:/strat_smoke"
ENV = dict(os.environ, PYTHONPATH="F:/strat_models/pylibs", HF_HUB_OFFLINE="1")


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def job(stage, clips, out, budget=30, limit=0):
    logf = f"{out}/logs/job_{stage}_{int(time.time())}.log"
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    cmd = [PY, HERE + "/extract_context.py", "--stage", stage, "--clips", clips, "--out", out, "--budget-min", str(budget)]
    if limit:
        cmd += ["--limit", str(limit)]
    with open(logf, "w") as f:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    return r.returncode, logf


def job_retry(stage, clips, out, budget=30, limit=0, max_wait_s=20 * 3600):
    """A GpuBusy exit (lock or idle check not granted within the wait window) is retried; any other failure is returned."""
    t0 = time.time()
    while True:
        rc, lf = job(stage, clips, out, budget, limit)
        busy = rc != 0 and "GpuBusy" in open(lf, errors="ignore").read()
        if not busy or time.time() - t0 > max_wait_s or os.path.exists(OUT + "/STOP"):
            return rc, lf
        log(f"GPU lock not granted ({lf}); retrying in 120 s")
        time.sleep(120)


def parts(out, stage, clips):
    return sum(os.path.exists(f"{out}/ctx{stage}/{c}.npz") for c in clips)


def smoke_report():
    import numpy as np
    clips = json.load(open(HERE + "/smoke/smoke20.json"))["clips"]
    rep = {"n_clips": len(clips), "A_parts": parts(SMOKE, "A", clips), "B_parts": parts(SMOKE, "B", clips)}
    ta, tb, td = [], [], []
    for lf in glob.glob(f"{SMOKE}/logs/job_*.log"):
        for l in open(lf, errors="ignore"):
            if l.startswith("{") and '"clip"' in l:
                d = json.loads(l)
                if "siglip_s" in d:
                    ta.append((d["siglip_s"], d["stats_s"], d["decode_wait_s"]))
                if "qwen_s" in d:
                    tb.append((d["qwen_s"], d["decode_wait_s"]))
    ok = rep["A_parts"] == len(clips) and rep["B_parts"] == len(clips)
    if ta:
        a = np.array(ta)
        rep["siglip_frames_per_s"] = round(101 / a[:, 0].mean(), 1)
        rep["tier1_stats_s_per_clip"] = round(a[:, 1].mean(), 2)
        rep["decode_wait_s_per_clip_A"] = round(a[:, 2].mean(), 2)
    if tb:
        b = np.array(tb)
        rep["qwen_s_per_clip_21_frames_4_fields"] = round(b[:, 0].mean(), 2)
        rep["qwen_frames_per_s"] = round(21 / b[:, 0].mean(), 2)
    for c in clips:
        pa_, pb_ = f"{SMOKE}/ctxA/{c}.npz", f"{SMOKE}/ctxB/{c}.npz"
        if os.path.exists(pa_) and os.path.exists(pb_):
            A, B = np.load(pa_), np.load(pb_)
            fin = all(np.isfinite(A[k]).all() for k in A.files if k.startswith(("t1_", "z_"))) and all(np.isfinite(B[k]).all() for k in B.files if k.startswith("v_"))
            sums = all(abs(A[k].sum(1) - 1).max() < 1e-2 for k in A.files if k.startswith("z_")) and all(abs(B[k].sum(1) - 1).max() < 1e-2 for k in B.files if k.startswith("v_"))
            ok = ok and fin and sums and len(A["emb"]) == 101 and len(B["v_weather"]) == 21
    rep["gate_ok"] = bool(ok)
    json.dump(rep, open(HERE + "/smoke/SMOKE_REPORT.json", "w"), indent=1)
    return rep


PILOT = HERE + "/splits/pilot_clips.json"
POUT = OUT + "/pilot"
FULL_FLAG = "--full-run-approved-by-pi"


def run_pilot():
    """A then B on the pilot clips (resumable), model sha, merge with the geometry + schema pin, stats, audit page, then ZZ_PILOT_DONE. Nothing else."""
    ids = json.load(open(PILOT))["clips"]
    os.makedirs(POUT + "/logs", exist_ok=True)
    for stage in ("A", "B"):
        stalls = 0
        while parts(POUT, stage, ids) < len(ids):
            if os.path.exists(OUT + "/STOP"):
                log("STOP marker seen")
                open(POUT + "/ZZ_PILOT_STOPPED", "w").write("stop\n")
                return 2
            before = parts(POUT, stage, ids)
            rc, lf = job_retry(stage, PILOT, POUT)
            after = parts(POUT, stage, ids)
            log(f"pilot {stage} rc={rc} {before}->{after}/{len(ids)} {lf}")
            stalls = stalls + 1 if after == before else 0
            if stalls >= 3:
                open(POUT + "/ZZ_PILOT_FAILED", "w").write(f"stage {stage} made no progress in 3 jobs, last log {lf}\n")
                return 1
    for name, cmd in (("model_sha", [PY, HERE + "/model_sha.py"]),
                      ("merge", [PY, HERE + "/merge_context.py", "--geom", OUT + "/geom", "--out", POUT]),
                      ("stats", [PY, HERE + "/pilot_stats.py"]),
                      ("page", [PY, HERE + "/build_pilot_audit.py"])):
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, env=ENV, capture_output=True, text=True)
        open(f"{POUT}/logs/{name}.out", "w").write(r.stdout + r.stderr)
        log(f"{name} rc={r.returncode}")
        if r.returncode != 0:
            open(POUT + "/ZZ_PILOT_FAILED", "w").write(f"{name} rc={r.returncode}, see {POUT}/logs/{name}.out\n")
            return 1
    open(POUT + "/ZZ_PILOT_DONE", "w").write("pilot finished; the full run needs " + FULL_FLAG + " from the Master Mind\n")
    return 0


def run_full(skip):
    """The full 4,659-clip run. Reachable only through main() with FULL_FLAG and a finished pilot."""
    os.makedirs(OUT + "/logs", exist_ok=True)
    sm = HERE + "/smoke/smoke20.json"
    if not skip:
        for stage in ("A", "B"):
            rc, lf = job_retry(stage, sm, SMOKE)
            log(f"smoke {stage} rc={rc} {lf}")
            if rc != 0 or parts(SMOKE, stage, json.load(open(sm))["clips"]) < 20:
                open(OUT + "/ZZ_CHAIN_FAILED", "w").write(f"smoke stage {stage} rc={rc} log={lf}\n")
                return 1
        subprocess.run([PY, HERE + "/model_sha.py"], stdin=subprocess.DEVNULL, env=ENV)
        rep = smoke_report()
        mr = subprocess.run([PY, HERE + "/merge_context.py", "--geom", OUT + "/geom", "--out", SMOKE], stdin=subprocess.DEVNULL, env=ENV, capture_output=True, text=True)
        rep["smoke_merge_rc"] = mr.returncode
        rep["smoke_merge_out"] = (mr.stdout + mr.stderr)[-600:]
        rep["gate_ok"] = bool(rep["gate_ok"] and mr.returncode == 0)
        json.dump(rep, open(HERE + "/smoke/SMOKE_REPORT.json", "w"), indent=1)
        log("smoke report " + json.dumps(rep))
        if not rep["gate_ok"]:
            open(OUT + "/ZZ_CHAIN_FAILED", "w").write("smoke gate failed: " + json.dumps(rep) + "\n")
            return 1
    allc = HERE + "/splits/clips_all.json"
    ids = json.load(open(allc))["clips"]
    stalls = 0
    for stage in ("A", "B"):
        while parts(OUT, stage, ids) < len(ids):
            if os.path.exists(OUT + "/STOP"):
                log("STOP marker seen")
                open(OUT + "/ZZ_CHAIN_STOPPED", "w").write("stop\n")
                return 2
            before = parts(OUT, stage, ids)
            rc, lf = job_retry(stage, allc, OUT)
            after = parts(OUT, stage, ids)
            log(f"full {stage} rc={rc} {before}->{after}/{len(ids)} {lf}")
            stalls = stalls + 1 if after == before else 0
            if stalls >= 3:
                open(OUT + "/ZZ_CHAIN_FAILED", "w").write(f"stage {stage} made no progress in 3 jobs, last log {lf}\n")
                return 1
    subprocess.run([PY, HERE + "/model_sha.py"], stdin=subprocess.DEVNULL, env=ENV)
    rc = subprocess.run([PY, HERE + "/merge_context.py", "--geom", OUT + "/geom", "--out", OUT], stdin=subprocess.DEVNULL, env=ENV).returncode
    log(f"merge rc={rc}")
    open(OUT + ("/ZZ_CHAIN_DONE" if rc == 0 else "/ZZ_CHAIN_FAILED"), "w").write(f"merge rc={rc}\n")
    return rc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if FULL_FLAG not in argv:
        return run_pilot()                    # default: a pilot ends at ZZ_PILOT_DONE and never reads clips_all.json
    if not os.path.exists(POUT + "/ZZ_PILOT_DONE"):
        raise SystemExit("refusing the full run: the pilot has not finished (no ZZ_PILOT_DONE)")
    return run_full("--skip-smoke" in argv)


if __name__ == "__main__":
    sys.exit(main())
