#!/usr/bin/env python3
"""RL-1 step 4a: build the navtest seam of the pre-registered composite (PREREG_RL1_B section 5).  CPU only, no label touched.
    python nt_build.py --variant V1 --out-dir C:/Users/Admin/par18/rl/nav
V1 = RL{EP, C} from R0B + base0 {NC, DAC, TTC, DDC, lane} from BASE.  Uses the FREEPOINT pipeline's own functions (fp_picks) unchanged."""
import argparse, gzip, json, os, sys
import numpy as np

sys.path.insert(0, "C:/Users/Admin/par18/rl/fpkg/code")
import fp_picks as FP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="V1")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--chunks", default="C:/Users/Admin/freept/work/out")
ap.add_argument("--points", default="C:/Users/Admin/freept/work/points.npz")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

AT = FP.load_attrib()
exp = json.load(gzip.open(FP.EXPORT, "rt", encoding="utf-8"))["tokens"]
zb = np.load(FP.BANKED_U400_SEAM, allow_pickle=False)
order = [str(t) for t in zb["token"]]
tok, traj, lg, ic2, meta = FP.merge_chunks(a.chunks, order)
assert ic2.max() == 0.0, "IC2"
L = {k: np.asarray(lg[k], np.float64) for k in ("BASE", "C0A", "R0B")}


def compose(rl, base, heads):
    x = base.copy()
    for h in heads:
        x[..., h] = rl[..., h]
    return x


VAR = {"V1": (L["R0B"], L["BASE"], (2, 4)), "V2": (L["R0B"], L["C0A"], (2, 4))}
rl, src, heads = VAR[a.variant]
X = compose(rl, src, heads)
pk = np.asarray(AT.picks(X), np.int64)

# identity gates computed from picks only (no labels)
pz = np.load(a.points, allow_pickle=False)
ptok = [str(t) for t in pz["token"]]
assert ptok == tok, "token order differs from points.npz"
banked = {p: np.asarray(pz[f"pick_{p}"], np.int64) for p in ("BASE", "C0A", "R0B", "M25")}
gates = {
    "BASE_picks_reproduced": bool(np.array_equal(np.asarray(AT.picks(L["BASE"]), np.int64), banked["BASE"])),
    "R0B_picks_reproduced": bool(np.array_equal(np.asarray(AT.picks(L["R0B"]), np.int64), banked["R0B"])),
    "compose_none_equals_SRC": bool(np.array_equal(np.asarray(AT.picks(compose(rl, src, ())), np.int64), np.asarray(AT.picks(src), np.int64))),
    "compose_all_equals_RL": bool(np.array_equal(np.asarray(AT.picks(compose(rl, src, range(7))), np.int64), np.asarray(AT.picks(rl), np.int64))),
}
rep = {"variant": a.variant, "n_tokens": len(tok), "gates": gates,
       "pick_changes": {p: int((pk != banked[p]).sum()) for p in banked},
       "same_pick_as": {p: int((pk == banked[p]).sum()) for p in banked},
       "tokens_with_no_banked_pose_equal": int((~np.any([pk == banked[p] for p in banked], axis=0)).sum())}
poses = FP.executed_poses(traj, pk)
FP.write_seam(f"{a.out_dir}/seam_{a.variant}.npz", tok, poses, exp, f"RL1_{a.variant}")
np.savez(f"{a.out_dir}/points_{a.variant}.npz", token=np.array(tok), **{f"pick_{a.variant}": pk}, **{f"pick_{p}": banked[p] for p in banked})
json.dump(rep, open(f"{a.out_dir}/build_{a.variant}.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("ZZNTBUILD_OK" if all(gates.values()) else "ZZNTBUILD_GATE_FAIL")
