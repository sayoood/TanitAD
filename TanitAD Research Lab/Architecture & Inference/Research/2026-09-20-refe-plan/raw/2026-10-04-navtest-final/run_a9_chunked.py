"""Amendment 9, one arm in LOG CHUNKS (the unchanged seam per chunk), then merged into the single seam / report / inputs
files a9_analyze.py reads. Why: the pdm_route arm's single 3,069-token seam died after 25 of 84 logs with a NATIVE access
violation (rc 0xC0000005, no Python traceback) and the seam only writes at the end. A chunk that crashes is re-run log by
log, so a crash costs at most one log and is localised; a log that still crashes is recorded, never silently dropped.
    python run_a9_chunked.py <arm> [--chunks 6]
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_a9_arms as R  # noqa: E402  (the lock, the env, the command)

A9 = R.OUT


def run_chunk(arm, tag, toks, tl, env):
    # ⛔ ".chunk.json", NOT "tokens_{arm}_{tag}.json": for arm "adopted" / tag "rest" that name IS the registered token file
    # (rule / rest / controls), and writing {tokens, token_log} over it made compose_full_seam.py fail on KeyError 'rest'
    # (MEASURED 2026-10-04 13:05; restored from git, blob 461cd84c).
    tj = f"{A9}/tokens_{arm}_{tag}.chunk.json"
    json.dump({"tokens": toks, "token_log": {t: tl[t] for t in toks}}, open(tj, "w", encoding="utf-8"), indent=0)
    cmd = [R.EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", R.CKPT, "--frames", f"{R.D}/frames",
           "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
           "--export", "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
           "--tokens", tj, "--out", f"{A9}/seam_{arm}_{tag}.npz", "--arm", f"REFe_a9_{arm}",
           "--record-inputs", f"{A9}/inputs_{arm}_{tag}.json"] + R.FLAGS[arm]
    t0 = time.time()
    with open(f"{A9}/seam_{arm}_{tag}.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=f"{R.PKG}/eval", env=env, stdout=f, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(f"{A9}/seam_{arm}_{tag}.npz")
    print(f"  {tag}: {len(toks)} tokens rc {rc} {time.time() - t0:.0f} s {'OK' if ok else 'FAILED'}", flush=True)
    return ok


def main():
    arm = sys.argv[1]
    n_chunks = int(sys.argv[sys.argv.index("--chunks") + 1]) if "--chunks" in sys.argv else 6
    ts = json.load(open(f"{A9}/tokens_{arm}.json", encoding="utf-8"))
    tl = ts["token_log"]
    logs = sorted({tl[t] for t in ts["tokens"]})
    groups = [logs[i::n_chunks] for i in range(n_chunks)]
    me = json.dumps({"job": f"refe-a9-{arm}-chunked", "pid": os.getpid(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    R.take_lock(me)
    print(f"lock taken {time.strftime('%F %T')}; {arm}: {len(ts['tokens'])} tokens, {len(logs)} logs, {n_chunks} chunks", flush=True)
    env = {k: R.fix(v) for k, v in R.EC.env_driverl().items()}
    done, crashed = [], []
    try:
        for ci, g in enumerate(groups):
            toks = [t for t in ts["tokens"] if tl[t] in set(g)]
            if run_chunk(arm, f"c{ci}", toks, tl, env):
                done.append(f"c{ci}"); continue
            for li, lg in enumerate(g):                       # localise: one log at a time
                lt = [t for t in ts["tokens"] if tl[t] == lg]
                if run_chunk(arm, f"c{ci}l{li}", lt, tl, env):
                    done.append(f"c{ci}l{li}")
                else:
                    crashed.append({"log": lg, "tokens": lt})
    finally:
        R.release(me)
    # merge
    T, F, P = [], [], []
    rep_all, inputs = {"rows": 0, "misses": 0, "miss_examples": [], "fc": [], "ic": []}, {}
    for tag in done:
        z = np.load(f"{A9}/seam_{arm}_{tag}.npz")
        T += list(z["token"]); F += list(z["fingerprint"]); P.append(z["poses"]); samp, armname = z["sampling"], z["arm"]
        r = json.load(open(f"{A9}/seam_{arm}_{tag}.report.json", encoding="utf-8"))
        rep_all["rows"] += r["rows"]; rep_all["misses"] += r["misses"]; rep_all["miss_examples"] += r["miss_examples"]
        rep_all["fc"].append(r["frame_control"]["max_m"]); rep_all["ic"].append((r.get("interpolation_residual") or {}).get("max_m") or 0)
        base = r
        inputs.update(json.load(open(f"{A9}/inputs_{arm}_{tag}.json", encoding="utf-8")))
    np.savez(f"{A9}/seam_{arm}.npz", token=np.array(T), fingerprint=np.array(F), poses=np.concatenate(P), sampling=samp, arm=armname)
    rep = {k: base[k] for k in ("ckpt", "select", "rule", "repair_last_heading", "sanitize_goal", "goal_fix") if k in base}
    rep.update(tokens_asked=len(ts["tokens"]), rows=rep_all["rows"],
               misses=rep_all["misses"] + sum(len(c["tokens"]) for c in crashed),
               miss_examples=rep_all["miss_examples"] + [[c["log"], "seam process crashed on this log"] for c in crashed],
               frame_control={"max_m": max(rep_all["fc"]), "bar_m": base["frame_control"]["bar_m"],
                              "what": "max over the chunks' own frame controls"},
               interpolation_residual={"max_m": max(rep_all["ic"])},
               chunked={"chunks": done, "crashed_logs": crashed, "why": __doc__.split("\n\n")[0]})
    json.dump(rep, open(f"{A9}/seam_{arm}.report.json", "w", encoding="utf-8"), indent=1)
    json.dump(inputs, open(f"{A9}/inputs_{arm}.json", "w", encoding="utf-8"))
    print(f"ZZA9_CHUNKED {arm} rows {rep['rows']}/{rep['tokens_asked']} misses {rep['misses']} crashed_logs {len(crashed)} "
          f"frame_control_max {rep['frame_control']['max_m']:.4g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
