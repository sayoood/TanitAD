#!/usr/bin/env python3
"""Reachability check of the W-TRANSP harness-identity fix FROM THE ACTUAL CALLER (the transp_score.py command line).

Scores the HUMAN reference arm only (never an N1/N2 or model arm) on the first --n tokens of the banked navtest human sub-seam by running
    transp_score.py --split navtest --human-arm [--metric-cache-root <cache>] --plans H8=<plans of that seam's human poses> --limit n
as a real subprocess, and compares the `human` arm's TRACKED sub-scores (nc dac ddc ep ttc c pdms) per token with the banked harness CSV
refe_e0b_human_sub200.csv. Three arms, rc 0 only if all three hold:
  FIX       with --metric-cache-root          : 0 tokens over TOL                       (PASS bar)
  REGRESS   without it (the DB route)         : >= 1 token over TOL                     (else the check cannot tell the routes apart)
  MUTATION  temp copy of transp_score.py with the one line that routes `parts` to load_cache_parts deleted, run WITH the flag
            : >= 1 token over TOL                                                       (the flag must actually reach the scorer)
CPU only. Usage:
  PYTHONPATH=<navsim devkit>;<refe dir>  python check_cli_cachefix.py --refe <refe dir> --seam human_sub200.npz --banked refe_e0b_human_sub200.csv \
      --work <scratch dir> --out result.json [--n 60] [--ts <transp_score.py under test>]
"""
import argparse, csv, gzip, json, os, subprocess, sys

COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance", "ep": "ego_progress",
       "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}
CACHE = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
TOKLOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
ROUTE_LINE = "parts = load_cache_parts(a.metric_cache_root, lg, tok) if a.metric_cache_root else PT.metric_cache_parts(sc)"


def mutate(src: str, dst: str) -> None:
    s = open(src, encoding="utf-8", newline="").read()
    if s.count(ROUTE_LINE) != 1:
        raise SystemExit(f"mutation: routing line not found exactly once in {src}")
    open(dst, "w", encoding="utf-8", newline="\n").write(s.replace(ROUTE_LINE, "parts = PT.metric_cache_parts(sc)"))


def run(ts, refe, plans, out, cache_root, n, env):
    if os.path.exists(out):
        os.remove(out)
    cmd = [sys.executable, ts, "--refe", refe, "--split", "navtest", "--token-log", TOKLOG, "--plans", f"H8={plans}", "--out", out,
           "--human-arm", "--no-fam", "--limit", str(n)]
    if cache_root:
        cmd += ["--metric-cache-root", cache_root]
    return subprocess.Popen(cmd, env=env, stdout=open(out + ".stdout", "w"), stderr=subprocess.STDOUT)


def compare(rows_path, banked, n):
    mx = {m: 0.0 for m in COL}; over = []; errs = []; seen = 0
    for ln in open(rows_path, encoding="utf-8"):
        r = json.loads(ln)
        if "error" in r:
            errs.append({"token": r.get("token"), "error": r["error"]}); continue
        tr = r["arms"]["human"]["tracked"]; seen += 1
        d = {m: abs(float(tr[m]) - float(banked[r["token"]][COL[m]])) for m in COL}
        for m in COL:
            mx[m] = max(mx[m], d[m])
        if max(d.values()) > 1e-6:
            over.append(r["token"])
    return {"tokens_scored": seen, "errors": errs, "tokens_over_1e-6": len(over), "over_tokens": over[:20], "max_abs_diff": mx, "complete": seen >= n and not errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True); ap.add_argument("--seam", required=True); ap.add_argument("--banked", required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "transp_score.py"))
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    import numpy as np
    os.makedirs(a.work, exist_ok=True)
    z = np.load(a.seam, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    sel = sorted(toks)[: a.n]
    plans = os.path.join(a.work, "human_plans8.npz")
    np.savez(plans, tokens=np.array(sel), plans8=np.stack([z["poses"][toks.index(t)] for t in sel]).astype(np.float64))
    banked = {r["token"]: r for r in csv.DictReader(open(a.banked, encoding="utf-8")) if r["token"] != "average"}
    mut = os.path.join(a.work, "transp_score_mutant.py")
    mutate(a.ts, mut)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")
    out = {k: os.path.join(a.work, f"rows_{k}.jsonl") for k in ("fix", "regress", "mutation")}
    # at most 2 processes at once: fix + regress together, then the mutation
    p1 = run(a.ts, a.refe, plans, out["fix"], a.cache, a.n, env); p2 = run(a.ts, a.refe, plans, out["regress"], "", a.n, env)
    rc = {"fix": p1.wait(), "regress": p2.wait()}
    rc["mutation"] = run(mut, a.refe, plans, out["mutation"], a.cache, a.n, env).wait()
    res = {k: dict(compare(out[k], banked, a.n), subprocess_rc=rc[k]) for k in out}
    res["n_tokens"] = a.n; res["tol"] = 1e-6; res["ts_under_test"] = a.ts
    res["FIX_PASS"] = res["fix"]["complete"] and res["fix"]["tokens_over_1e-6"] == 0
    res["REGRESS_DISCRIMINATES"] = res["regress"]["complete"] and res["regress"]["tokens_over_1e-6"] > 0
    res["MUTATION_RED"] = res["mutation"]["complete"] and res["mutation"]["tokens_over_1e-6"] > 0
    res["PASS"] = bool(res["FIX_PASS"] and res["REGRESS_DISCRIMINATES"] and res["MUTATION_RED"])
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("FIX_PASS", "REGRESS_DISCRIMINATES", "MUTATION_RED", "PASS")}))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
