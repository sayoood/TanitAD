#!/usr/bin/env python3
"""RL-1 step 4c: the pre-registered navtest read of the composite (PREREG_RL1_B section 5).  CPU only.  Reuses fp_read / stage3_full (the FREEPOINT estimator and rows).
    python nt_read.py --variant V1 --work C:/Users/Admin/par18/rl/nav --out C:/Users/Admin/par18/rl/nav/result
Needs <work>/DONE_<V> = OK, <work>/seam_<V>.npz, <work>/points_<V>.npz, <work>/score/refe_rl1_<V>/refe_rl1_<V>.csv."""
import argparse, gzip, json, os, pathlib, sys
import numpy as np

sys.path.insert(0, "C:/Users/Admin/par18/rl/fpkg/code")
import fp_read as FR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="V1")
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
V, work, out = a.variant, a.work, a.out
label = f"refe_rl1_{V}"
os.makedirs(out, exist_ok=True)
done = open(f"{work}/DONE_{V}").read().split("\n")[0]
if done != "OK":
    print("REFUSED: DONE marker is", done)
    sys.exit(2)
S3 = FR.load_s3()
exp = json.load(gzip.open(FR.EXPORT, "rt", encoding="utf-8"))["tokens"]
full = json.load(open(f"{FR.DATA}/navtest_full_tokens.json", encoding="utf-8"))
toks = sorted(full["tokens"] if isinstance(full, dict) else full)
tl = {t: exp[t]["log_name"] for t in toks}
tabs = {p: S3.load_table(f"{FR.DATA}/score/refe_fp_{p}/refe_fp_{p}.csv") for p in ("BASE", "C0A", "R0B")}
tabs[V] = S3.load_table(f"{work}/score/{label}/{label}.csv")
st = json.load(open(f"{work}/score_logs/{label}.status.json"))
gates = {"harness_PASS": st.get("status") == "PASS", "rows_12146": len([t for t in toks if t in tabs[V]]) == len(toks)}
# identity gate (a): tokens whose pick equals a banked point's pick must read that point's row exactly
pz = np.load(f"{work}/points_{V}.npz", allow_pickle=False)
ptok = [str(t) for t in pz["token"]]
same = {}
for p in ("BASE", "C0A", "R0B"):
    eq = np.asarray(pz[f"pick_{V}"]) == np.asarray(pz[f"pick_{p}"])
    ts = [t for t, e in zip(ptok, eq) if e]
    mx = max((abs(tabs[V][t][k] - tabs[p][t][k]) for t in ts for k in ("PDMS", "NC", "DAC", "EP", "TTC", "C", "DDC")), default=0.0)
    same[p] = {"n_tokens_same_pick": len(ts), "max_abs_diff_rows": float(mx)}
gates["identity_a_same_pick_rows_equal"] = all(v["max_abs_diff_rows"] <= 1e-9 for v in same.values())
pairs = {"V_vs_BASE": S3.read_pair(tabs[V], tabs["BASE"], toks, tl), "R0B_vs_BASE": S3.read_pair(tabs["R0B"], tabs["BASE"], toks, tl),
         "C0A_vs_BASE": S3.read_pair(tabs["C0A"], tabs["BASE"], toks, tl)}
lever = {"V-R0B": S3.read_pair(tabs[V], tabs["R0B"], toks, tl), "V-C0A": S3.read_pair(tabs[V], tabs["C0A"], toks, tl)}
safe = {k: FR.is_safe(v) for k, v in pairs.items()}
p = pairs["V_vs_BASE"]["primary"]["ci95"]
s = pairs["V_vs_BASE"]["subscores"]
adv = [c for c in ("NC", "DAC", "TTC") if s[c]["ci95"][1] < 0]
outcome = "N-PASS" if (p[0] > 0 and not adv) else ("N-GAIN-LOST" if not adv else ("N-SAFETY-LOST" if p[0] > 0 else "N-BOTH"))
famdir = pathlib.Path(out) / "families"
seams = {"BASE": f"{FR.DATA}/../../../../../../Users/Admin/freept/work/seam_BASE.npz"}
sb = "C:/Users/Admin/freept/work"
fams = {pp: S3.compute_families(pathlib.Path(f"{sb}/seam_{pp}.npz"), f"fp_{pp}", famdir, set(toks)) for pp in ("BASE", "R0B")}
fams[V] = S3.compute_families(pathlib.Path(f"{work}/seam_{V}.npz"), f"rl1_{V}", famdir, set(toks))
fam_vs = {pp: S3.families_vs(fams["BASE"], fams[pp]) for pp in ("R0B", V)}
res = {"tool": "nt_read/1", "variant": V, "tier": S3.TIER, "estimator": S3.ESTIMATOR, "n_tokens": len(toks), "gates": gates, "same_pick_identity": same,
       "pairs_vs_BASE": pairs, "lever_view": lever, "deployable_safe": safe, "adverse_components_V_vs_BASE": adv, "OUTCOME": outcome,
       "levels_x100": {k: {m: float(np.mean([tabs[k][t][m] for t in toks])) for m in ("PDMS",) + FR.SUBS} for k in tabs},
       "families_adverse_vs_BASE": fam_vs, "strategic": "PAUSED (PI 2026-09-19, item 25); UNAVAILABLE in NAVSIM"}
json.dump(res, open(f"{out}/navtest_{V}_result.json", "w"), indent=1, default=str)
md = ["## Levels (mean x100, all 12,146 tokens)", "", FR.level_table(tabs, toks), "", FR.pair_table("Each point minus BASE (paired, x100)", pairs), "",
      FR.pair_table("Lever view (paired, x100)", lever), "", "## DEPLOYABLE-SAFE (vs BASE)", "", *[f"* {k}: **{v['safe']}** {v['conditions']}" for k, v in safe.items()], "",
      f"## OUTCOME: **{outcome}** (adverse: {adv}); gates {gates}", "", "## Families", "", FR.fam_summary({**fams})]
open(f"{out}/RESULT_TABLES_{V}.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md[:40]))
print("ZZNTREAD_OK", outcome, gates)
