"""S1 known-value control: refit P10 on the refav1 grid and reproduce 6 s ADE 2.9257 +- 0.02 m (PREREG H-V8F-FLOOR-1).

The P10 package's TRAIN features (feat_train_s8) no longer exist on the dev box, so they are REBUILT from the
refcv8 train-view manifest (read-only pull from Thor) with the SAME window rule / feature definitions as
``extract.py`` -- and that rebuild is itself validated against the package's banked EVAL arrays (feat_s4) before use.
Scoring side is the package's own banked feat_s4 (arrays untouched).
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_lib as L                                                                   # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                      # noqa: E402

import torch                                                                         # noqa: E402
torch.set_num_threads(6)

OUT = os.path.join(HERE, "..", "raw")
PKG = "C:/Users/Admin/refav1_probe"
sys.path.insert(0, PKG)
import probe as P                                                                    # noqa: E402  (original, read-only)
import probe_te as PT                                                                # noqa: E402  (original fit_mlp_kin)

res = {"tool": "run_control_refav1.py", "grid": "refav1 operative 0.2 s; 2 s = 10 x 0.2 s, 6 s = 30 x 0.2 s",
       "tier": "T1-equivalent (t0 state only; no future input)"}

# ------------------------------------------------------------------ 1. validate the rebuild on EVAL --------
t = time.time()
eman = L.load_manifest(L.EVAL_MAN)
eps141 = sorted(os.path.basename(p)[:-len(".v2ep.pt")]
                for p in glob.glob("C:/Users/Admin/tanitad-data/refav1-eval141/eps/*.v2ep.pt"))
cid = {c: i for i, c in enumerate(eman["clip_id"])}
worst = {k: 0.0 for k in ("kin", "g", "g6", "ha0", "ha0_ext", "ha0_6", "ha0_ext_6")}
n_cmp, n_ws_ok, n_win = 0, 0, 0
for fi, c in enumerate(eps141):
    if c not in cid:
        continue
    z = np.load(f"{PKG}/feat_s4/ep{fi:03d}.npz")
    r = L.refav1_episode(eman["poses"][cid[c]], eman["actions"][cid[c]], stride=4)
    n_cmp += 1
    n_win += len(z["ws"])
    if not np.array_equal(r["ws"], z["ws"]):
        continue
    n_ws_ok += 1
    ok = z["g6_ok"]
    assert np.array_equal(r["g6_ok"], ok)
    for k in worst:
        a, b = r[k], z[k]
        if k == "g6":
            a, b = a[ok], b[ok]
        worst[k] = max(worst[k], float(np.abs(a - b).max()))
res["rebuild_vs_banked_eval"] = {"episodes_compared": n_cmp, "episodes_with_identical_ws": n_ws_ok,
                                 "windows": n_win, "max_abs_diff": worst,
                                 "note": "rebuilt from the refcv eval139 manifest (raw = manifest + 2 frames) vs feat_s4 (float32)"}
print("[control] rebuild vs banked EVAL:", json.dumps(res["rebuild_vs_banked_eval"]), f"{time.time()-t:.0f}s", flush=True)
assert n_ws_ok == n_cmp and max(worst.values()) < 2e-3, "TRAIN rebuild is not validated -- refusing to use it"

# ------------------------------------------------------------------ 2. rebuild TRAIN (600 clips, stride 8) --
tman = L.load_manifest(L.TRAIN_MAN)
ids = L.pick_train_ids(tman)
tc = {c: i for i, c in enumerate(tman["clip_id"])}
rows = {k: [] for k in ("kin", "g", "g6", "g6_ok", "ha0", "ha0_ext", "ha0_6", "ha0_ext_6", "ep")}
for fi, c in enumerate(ids):                                   # sorted by clip id == the package's episode order
    r = L.refav1_episode(tman["poses"][tc[c]], tman["actions"][tc[c]], stride=8)
    if r is None:
        continue
    for k in rows:
        if k != "ep":
            rows[k].append(r[k])
    rows["ep"].append(np.full(len(r["ws"]), fi))
T = {k: np.concatenate(v) for k, v in rows.items()}
res["train"] = {"episodes": len(ids), "windows": int(len(T["kin"])), "windows_6s_valid": int(T["g6_ok"].sum()),
                "expected_windows_from_extract_train_log": 5398}
print("[control] TRAIN rebuilt:", res["train"], flush=True)

# ------------------------------------------------------------------ 3. the package's EVAL arrays -----------
E = P.load()
n = len(E["kin"])
bE2, bE6 = P.kdx(E["ha0"], E["ha0_ext"]), P.kdx(E["ha0_6"], E["ha0_ext_6"])
gE2, gE6 = E["g"].astype(np.float64), E["g6"].astype(np.float64)
okE = E["g6_ok"].astype(bool)
# kd_x: my vectorised retime vs the original on the banked arrays (known-value control)
res["kdx_vs_original_P_kdx"] = {
    "max_abs_6s": float(np.abs(fk.kd_x(E["ha0_6"], E["ha0_ext_6"]).astype(np.float64) - bE6).max()),
    "max_abs_2s": float(np.abs(fk.kd_x(E["ha0"], E["ha0_ext"]).astype(np.float64) - bE2).max())}
print("[control] kd_x vs P.kdx:", res["kdx_vs_original_P_kdx"], flush=True)

kfT = fk.kin_features(T["kin"])
kfE = fk.kin_features(E["kin"].astype(np.float64))
bT2, bT6 = P.kdx(T["ha0"], T["ha0_ext"]), P.kdx(T["ha0_6"], T["ha0_ext_6"])
gT2, gT6 = T["g"].astype(np.float64), T["g6"].astype(np.float64)
okT = T["g6_ok"].astype(bool)
yT2 = (gT2 - bT2).reshape(len(gT2), -1)
yT6 = np.where(okT[:, None, None], gT6 - bT6, 0.0).reshape(len(gT6), -1)

pred = {}
cmp_orig = {}
for hz, y, b, g, ok, bE in (("2s", yT2, bT2, gT2, np.ones(len(gT2), bool), bE2), ("6s", yT6, bT6, gT6, okT, bE6)):
    t = time.time()
    net, st, best = fk.fit_p10(kfT[ok], y[ok], b[ok], g[ok], T["ep"][ok])
    mine = fk.predict_p10((net, st, best), kfE, bE)
    net_o, st_o, best_o = PT.fit_mlp_kin(kfT[ok], y[ok], b[ok], g[ok], T["ep"][ok])
    with torch.no_grad():
        yh = net_o(torch.tensor((kfE - st_o[0]) / st_o[1], dtype=torch.float32)).numpy() + st_o[2]
    orig = bE + yh.reshape(-1, *bE.shape[1:])
    cmp_orig[hz] = {"max_abs_pred_diff_vs_original_fit_mlp_kin": float(np.abs(mine - orig).max()),
                    "sel_mine": [0, float(best[1]), int(best[2])], "sel_original": [0, float(best_o[1]), int(best_o[2])]}
    pred[hz] = mine
    print(f"[control] {hz} fit {time.time()-t:.0f}s sel={best} vs original {best_o} maxdiff="
          f"{cmp_orig[hz]['max_abs_pred_diff_vs_original_fit_mlp_kin']:.3g}", flush=True)
res["fit_p10_vs_original"] = cmp_orig

ade2 = fk.ade(pred["2s"], gE2)
ade6 = fk.ade(pred["6s"][okE], gE6[okE])
fde6 = fk.fde(pred["6s"][okE], gE6[okE])
banked = np.load(f"{PKG}/out_te/arm_P10_kin_mlp.npz", allow_pickle=False)
ade2_b = fk.ade(banked["pred2"], gE2)
ade6_b = fk.ade(banked["pred6"][okE], gE6[okE])
res["known_value"] = {
    "P10_6s_ade_m": float(ade6.mean()), "target_6s_ade_m": 2.9257, "tolerance_m": 0.02,
    "abs_error_m": float(abs(ade6.mean() - 2.9257)), "PASS": bool(abs(ade6.mean() - 2.9257) <= 0.02),
    "P10_2s_ade_m": float(ade2.mean()), "banked_2s_ade_m": float(ade2_b.mean()),
    "banked_6s_ade_m_recomputed": float(ade6_b.mean()),
    "P10_6s_fde_m": float(fde6.mean()), "n_eval_windows": int(n), "n_eval_6s": int(okE.sum()),
    "n_eval_episodes": int(len(np.unique(E["ep"]))),
    "paired_ours_minus_banked_P10_6s_ade": pb(ade6, ade6_b, E["ep"][okE]),
    "paired_ours_minus_banked_P10_2s_ade": pb(ade2, ade2_b, E["ep"]),
    "paired_P10_minus_P10_6s_ade": pb(ade6, ade6, E["ep"][okE]),
}
print("[control] KNOWN VALUE:", json.dumps({k: v for k, v in res["known_value"].items() if not k.startswith("paired")}), flush=True)
for k in ("paired_P10_minus_P10_6s_ade",):
    print("[control]", k, res["known_value"][k]["delta"], res["known_value"][k]["lo"], res["known_value"][k]["hi"])
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "control_refav1.json"), "w"), indent=1)
np.savez(f"{L.SCRATCH}/control_refav1_pred.npz", pred2=pred["2s"], pred6=pred["6s"])
print("ZZCONTROL-DONEZZ")
