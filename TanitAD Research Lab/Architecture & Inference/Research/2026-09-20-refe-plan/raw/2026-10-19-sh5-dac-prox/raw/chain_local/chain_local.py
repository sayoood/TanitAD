"""SH5 LOCAL chain (Master Mind, 2026-10-19 ~16:05; written BEFORE any SH5 fit on real ctx was run).

Runs READY.md Q3 -> Q4 -> Q5 on LOCAL, each GPU stage behind sh5_gpulock (the fit takes the lock itself),
then STOPS. Q6 / Q7 (conditional on the A1 raw call) are decided by the Master Mind from analyze's output.

Pre-stated continue rule for the C-REACH pre-flight (READY.md: "near 1 and r2_reg >= 0.5"), operationalised here
before the run: EVERY group in preflight_PRE_s0.json has auc_all_strata >= 0.95 AND r2_reg >= 0.5.
Otherwise ZZCHAIN_STOP (no 20-learner stage). C-ID is read by sh5_gates.py from the evidence files, never typed.
"""
import json, os, subprocess, sys, time

ROOT = "C:/Users/Admin/par18/sh5"
DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
PY = ["C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe", "-B", "-W", "ignore"]
FIN = f"{ROOT}/work/FINISHED_A3.txt"
CTX = ["--data", f"{DD}/data", "--ctx", f"{DD}/ctx_local/ctx.npy", "--ctx-manifest", f"{DD}/ctx_local/ctx_manifest.json"]


def log(m):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {m}", flush=True)


def run(args, gpu):
    env = dict(os.environ)
    if gpu:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"; env["OMP_NUM_THREADS"] = "2"
    log("RUN " + " ".join(args))
    p = subprocess.run(PY + args, cwd=f"{ROOT}/code", env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    sys.stdout.write(p.stdout); sys.stdout.flush()
    log(f"RC {p.returncode}")
    return p.returncode, p.stdout


def stop(why):
    log(f"ZZCHAIN_STOP {why}")
    sys.exit(2)


def main():
    t0 = time.time()
    while True:
        s = open(FIN, encoding="utf-8", errors="replace").read() if os.path.exists(FIN) else ""
        if "FAILED" in s or "TIMEOUT" in s:
            stop("finisher reported failure: " + s.strip().splitlines()[-1])
        if "DONE" in s:
            break
        if time.time() - t0 > 7200:
            stop("finisher not DONE after 2 h")
        time.sleep(30)
    log("finisher DONE; starting Q3")
    for seed in (0, 1):
        rc, _ = run(["sh5_train_eval.py", "fit", *CTX, "--seed", str(seed), "--stage", "A1", "--epochs", "0", "--out-dir", f"{DD}/fits_local/cid"], gpu=True)
        if rc != 0:
            stop(f"C-ID s{seed} rc={rc}")
    rc, _ = run(["sh5_train_eval.py", "fit", *CTX, "--seed", "0", "--stage", "PRE", "--folds", "0", "--out-dir", f"{DD}/fits_local/pre"], gpu=True)
    if rc != 0:
        stop(f"PRE rc={rc}")
    rep = json.load(open(f"{DD}/fits_local/pre/preflight_PRE_s0.json"))
    ok = len(rep) > 0 and all(v["auc_all_strata"] is not None and v["auc_all_strata"] >= 0.95 and v["r2_reg"] >= 0.5 for v in rep.values())
    log(f"ZZCHAIN_PREFLIGHT ok={ok} {json.dumps(rep)}")
    if not ok:
        stop("C-REACH pre-flight below the pre-stated rule (auc >= 0.95 and r2_reg >= 0.5 on every group)")
    for seed in (0, 1):
        rc, _ = run(["sh5_train_eval.py", "fit", *CTX, "--seed", str(seed), "--stage", "A1", "--out-dir", f"{DD}/fits_local/main"], gpu=True)
        if rc != 0:
            stop(f"A1 s{seed} rc={rc}")
    rc, _ = run(["sh5_gates.py", "--cid-dir", f"{DD}/fits_local/cid", "--glive", f"{DD}/fits_local/main/g_live_A1_s0.json", f"{DD}/fits_local/main/g_live_A1_s1.json",
                 "--ctx-manifest", f"{DD}/ctx_local/ctx_manifest.json", "--tests", f"{ROOT}/work/tests_summary_amend3.json", "--out", f"{DD}/fits_local/gates.json"], gpu=False)
    if rc != 0:
        stop(f"gates rc={rc}")
    rc, out = run(["sh5_train_eval.py", "analyze", "--data", f"{DD}/data", "--fits", f"{DD}/fits_local/main", "--gates", f"{DD}/fits_local/gates.json",
                   "--keep-tokens", f"{DD}/ctx_local/ctx_manifest.json", "--out-dir", f"{DD}/results_local"], gpu=False)
    if rc != 0:
        stop(f"analyze rc={rc}")
    last = [l for l in out.strip().splitlines() if l.strip()][-1] if out.strip() else ""
    log(f"ZZCHAIN_DONE A1 analyze: {last}")


if __name__ == "__main__":
    main()
