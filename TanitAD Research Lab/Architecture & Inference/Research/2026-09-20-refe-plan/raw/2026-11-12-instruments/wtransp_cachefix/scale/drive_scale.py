"""W-TRANSP --metric-cache-root at SCALE: the registered teacher T seam (12,146 navtest tokens) through transp_score.py's
cache route, tracked sub-scores vs the banked pinned-harness CSV refe_p4_teacher_navtest.csv, |d| <= 1e-6 on all 7 metrics.
Non-selection arm (the T seam is the C-ID-H identity reference, read many times). REGRESS (DB route) at scale is already
banked: TF-N1 void C-ID-H, 3,346 / 12,146 tokens over 1e-6. MUTATION was proven at 60 tokens (check_cli_cachefix.json).
Usage: python drive_scale.py [--procs 2] [--limit 0]   (MM, 2026-11-12)"""
import argparse, csv, glob, json, os, subprocess, sys, time
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
CACHE = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
TOKLOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEAM = "F:/tanitad_scratch/tfn2_nt/run_a1/inputs/refe_p4_teacher_navtest.npz"
BANKED = "F:/tanitad_scratch/tfn2_nt/run_a1/inputs/refe_p4_teacher_navtest.csv"
COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance", "ep": "ego_progress",
       "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = os.path.join(W, a.tag)
    os.makedirs(out, exist_ok=True)
    z = np.load(SEAM, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    plans = os.path.join(out, "t_plans8.npz")
    np.savez(plans, tokens=np.array(toks), plans8=z["poses"].astype(np.float64))
    refe = os.path.join(W, "w", "pk", "refe")
    ts = os.path.join(W, "w", "pk", "eval", "instruments", "transp_score.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", PYTHONPATH=f"{NAVSIM};{refe}")
    t0 = time.time()
    ps = []
    for i in range(a.procs):
        cmd = [sys.executable, ts, "--refe", refe, "--split", "navtest", "--token-log", TOKLOG, "--plans", f"T={plans}",
               "--metric-cache-root", CACHE, "--no-fam", "--out", os.path.join(out, f"rows_s{i}of{a.procs}.jsonl"), "--shard", f"{i}/{a.procs}"]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        ps.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=open(os.path.join(out, f"transp_s{i}.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in ps]
    banked = {r["token"]: r for r in csv.DictReader(open(BANKED, encoding="utf-8")) if r["token"] != "average"}
    mx = {m: 0.0 for m in COL}
    over, errs, seen = [], [], set()
    for fp in sorted(glob.glob(os.path.join(out, f"rows_s*of{a.procs}.jsonl"))):
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if "error" in r:
                errs.append({"token": r.get("token"), "error": str(r["error"])[:200]})
                continue
            tr = r["arms"]["T"]["tracked"]
            seen.add(r["token"])
            d = {m: abs(float(tr[m]) - float(banked[r["token"]][COL[m]])) for m in COL}
            for m in COL:
                mx[m] = max(mx[m], d[m])
            if max(d.values()) > 1e-6:
                over.append({"token": r["token"], **{m: d[m] for m in COL if d[m] > 1e-6}})
    want = a.limit * a.procs if a.limit else len(toks)
    res = {"what": "W-TRANSP --metric-cache-root, T seam vs banked harness CSV", "tag": a.tag, "rcs": rcs, "tokens_scored": len(seen),
           "tokens_wanted": want, "errors": len(errs), "error_examples": errs[:10], "tokens_over_1e-6": len(over), "over_examples": over[:20],
           "max_abs_diff": mx, "complete": len(seen) >= want and not errs and not any(rcs), "wall_s": round(time.time() - t0, 1),
           "seam": SEAM, "banked": BANKED, "cache": CACHE}
    res["PASS"] = bool(res["complete"] and not over)
    json.dump(res, open(os.path.join(out, "scale_result.json"), "w", encoding="utf-8"), indent=1)
    print("ZZSCALE", "PASS" if res["PASS"] else "FAIL", json.dumps({k: res[k] for k in ("tokens_scored", "errors", "tokens_over_1e-6", "wall_s")}), flush=True)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
