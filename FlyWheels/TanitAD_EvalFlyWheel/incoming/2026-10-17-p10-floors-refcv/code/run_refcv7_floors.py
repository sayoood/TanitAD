"""S1 main: build the floor arms for ONE (horizon, inference seed), write a derived dump, run the battery's own
four-family analysis on it, and compute the paired ADE/FDE deltas + controls + acts + (EXPLORATORY) late fusion.

usage: run_refcv7_floors.py S2|S6 0|1 [--no-analysis]

Inputs  : banked refcv7-50,400 T1 dumps (E:/refcv7_eval_kit/battery/step50400_a6text/{dump_s*,s6_s*}),
          eval139 + train4572 v2 manifests (read-only copies pulled from Thor), p10_<H>.pt (fit_p10_refcv.py).
Outputs : raw/arms_<H>_s<seed>.npz, raw/floors_<H>_s<seed>.json, raw/analysis_<H>_s<seed>.json
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_lib as L                                                                   # noqa: E402
from taniteval import floors_kin as fk                                               # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb                      # noqa: E402

import torch                                                                         # noqa: E402
torch.set_num_threads(int(os.environ.get("S1_THREADS", "3")))

H, SEED = sys.argv[1], int(sys.argv[2])
DO_ANALYSIS = "--no-analysis" not in sys.argv
RAW = os.path.join(HERE, "..", "raw")
CFG = {"S2": dict(sel=L.H_S2, n_fine=20, dt=0.5, tag=("dump_s0", "dump_s1"), man=("panel_s0", "panel_s1"),
                  banked=("analysis_s0.json", "analysis_s1.json")),
       "S6": dict(sel=L.H_S6, n_fine=60, dt=1.0, tag=("s6_s0", "s6_s1"), man=("s6_s0", "s6_s1"),
                  banked=("analysis_s6_s0.json", "analysis_s6_s1.json"))}[H]
sel, n_fine, dt = CFG["sel"], CFG["n_fine"], CFG["dt"]
N_BOOT = 2000

t_all = time.time()
eman = L.load_manifest(L.EVAL_MAN)
W = L.refcv_eval_windows(eman, L.BATT, CFG["tag"][SEED], sel)
n = len(W["kin"])
fl = L.refcv_floors(W["kin"], n_fine, sel)
g = W["d_g"].astype(np.float64)
ep = W["ep"]

# ---- known-value controls: the floors rebuilt from POSES reproduce the banked arms ------------------------------
ctrl = {"v0_max_abs_vs_dump": W["check"]["v0"], "g_max_abs_vs_dump": W["check"]["g"],
        "ha0_max_abs_vs_dump_arm": float(np.abs(fl["ha0"] - W["d_ha0"]).max()),
        "ha0_ext_max_abs_vs_dump_arm": float(np.abs(fl["ha0_ext"] - W["d_ha0_ext"]).max()),
        "damp50_max_abs_vs_half_half_of_dump_arms": float(np.abs(
            fl["damp50"] - (0.5 * W["d_ha0"].astype(np.float64) + 0.5 * W["d_ha0_ext"].astype(np.float64))).max()),
        "_tolerance": "float32 dump vs float64 rebuild: < 2e-4 m"}
assert max(v for k, v in ctrl.items() if k.endswith(("dump", "arm", "arms"))) < 2e-4, ctrl

# ---- P10 (TRAIN-fit) -------------------------------------------------------------------------------------------
ck = torch.load(f"{L.SCRATCH}/p10_{H}.pt", weights_only=False)
net = fk._make_mlp(8, len(sel) * 2)
net.load_state_dict(ck["state"])
kf = fk.kin_features(W["kin"])
P10 = fk.predict_p10((net, (ck["xm"], ck["xs"], ck["ym"]), ck["best"]), kf, fl["kd_x"].astype(np.float32))
arms = {"os": W["d_os"], "ha": W["d_ha"], "ha0": W["d_ha0"], "ha0_ext": W["d_ha0_ext"],
        "damp50": fl["damp50"].astype(np.float32), "kd_x": fl["kd_x"].astype(np.float32), "P10": P10.astype(np.float32)}

# ---- sensitivity floors (NOT in the heavy analysis) ---------------------------------------------------------------
sens = {"kd_x_coarse": fk.kd_x(W["d_ha0"], W["d_ha0_ext"])}                  # retime on the dump-grid polyline
if H == "S6":                                                                   # the 30 x 0.2 s definition (refav1's)
    h0 = fk.ha0_paths(W["kin"][:, 0], 0.2, 30)
    he = fk.ha0_ext_paths(W["kin"], 0.2, 30, action_units="steer")
    sens["kd_x30_6s"] = fk.kd_x(h0, he)[:, 4::5].astype(np.float32)             # steps 5,10,..,30 == 1..6 s

np.savez_compressed(os.path.join(RAW, f"arms_{H}_s{SEED}.npz"), ep=ep, ws=W["t0"], clip_index=W["clip"], g=W["d_g"],
                    v0=W["d_v0"], **arms, **{k: v for k, v in sens.items()})

# ---- direct paired deltas (ADE, FDE), the standard estimator --------------------------------------------------------
A = {k: v.astype(np.float64) for k, v in {**arms, **sens}.items()}
ade_w = {k: fk.ade(v, g) for k, v in A.items()}
fde_w = {k: fk.fde(v, g) for k, v in A.items()}
PAIRS = [("os", "P10"), ("os", "damp50"), ("os", "kd_x"), ("os", "ha0_ext"), ("os", "ha0"), ("P10", "kd_x"),
         ("P10", "ha0_ext"), ("damp50", "ha0_ext"), ("kd_x", "ha0_ext"), ("P10", "P10"), ("os", "kd_x_coarse")]
if H == "S6":
    PAIRS += [("os", "kd_x30_6s"), ("kd_x", "kd_x30_6s")]
paired = {}
for a_, b_ in PAIRS:
    paired[f"{a_}-{b_}"] = {"ade": pb(ade_w[a_], ade_w[b_], ep, n_boot=N_BOOT, seed=0),
                            "fde": pb(fde_w[a_], fde_w[b_], ep, n_boot=N_BOOT, seed=0)}
means = {k: {"ade": float(ade_w[k].mean()), "fde": float(fde_w[k].mean())} for k in A}
means["GT_zero_check"] = {"ade": float(fk.ade(g, g).mean())}

# ---- acts check (every arm and floor, and the GT as the reference) ----------------------------------------------------
acts = {k: fk.acts_check(v, dt) for k, v in A.items() if k in ("os", "ha0", "ha0_ext", "damp50", "kd_x", "P10", "ha")}
acts["GT"] = fk.acts_check(g, dt)

# ---- EXPLORATORY late fusion, alpha by grouped 5-fold CV -----------------------------------------------------------------
fold = fk.grouped_folds(ep, 5)
os_ = A["os"]
p10_ = A["P10"]
fused = np.empty_like(os_)
alphas = []
for f in range(5):
    tr, te = fold != f, fold == f
    a_f = fk.fit_alpha(os_[tr], p10_[tr], g[tr])
    alphas.append(a_f)
    fused[te] = fk.late_fusion(os_[te], p10_[te], a_f)
ade_f, fde_f = fk.ade(fused, g), fk.fde(fused, g)
fusion = {"LABEL": "EXPLORATORY -- never a lever claim (PREREG H-V8F-FLOOR-1)",
          "alpha_per_fold": alphas, "alpha_all_windows_for_information": fk.fit_alpha(os_, p10_, g),
          "folds": "grouped 5-fold by episode (rank mod 5); alpha fit on the 4 held-in folds only",
          "ade_mean": float(ade_f.mean()), "fde_mean": float(fde_f.mean()),
          "fused_minus_P10": {"ade": pb(ade_f, ade_w["P10"], ep, n_boot=N_BOOT, seed=0),
                              "fde": pb(fde_f, fde_w["P10"], ep, n_boot=N_BOOT, seed=0)},
          "fused_minus_os": {"ade": pb(ade_f, ade_w["os"], ep, n_boot=N_BOOT, seed=0),
                             "fde": pb(fde_f, fde_w["os"], ep, n_boot=N_BOOT, seed=0)}}

out = {"horizon": H, "inference_seed": SEED, "tier": "T1 (self-action OPEN loop; the model does not control the vehicle)",
       "n_windows": int(n), "n_episodes": int(len(np.unique(ep))), "instants_s": [round(h * 0.1, 2) for h in sel],
       "dump": os.path.join(L.BATT, CFG["tag"][SEED]), "estimator": "paired episode-cluster bootstrap, B=2000, seed 0 (taniteval/ci.py)",
       "variance_answered": ("episodes (resampled with the models held fixed); inference seed via the s0/s1 pair; "
                             "P10 is ONE training run (training-seed variance of the floor NOT measured)"),
       "p10": ck["meta"], "controls": ctrl, "means": means, "paired_x_minus_y": paired, "acts": acts, "fusion": fusion}
json.dump(out, open(os.path.join(RAW, f"floors_{H}_s{SEED}.json"), "w"), indent=1)
print(f"[{H} s{SEED}] direct stats done {time.time()-t_all:.0f}s  os-P10 ADE:", paired["os-P10"]["ade"]["delta"],
      paired["os-P10"]["ade"]["lo"], paired["os-P10"]["ade"]["hi"], flush=True)

# ---- the battery's own four-family analysis on a derived dump ----------------------------------------------------------------
if DO_ANALYSIS:
    dd = os.path.join(L.SCRATCH, "dumps", f"{H}_s{SEED}")
    shutil.rmtree(dd, ignore_errors=True)
    os.makedirs(dd, exist_ok=True)
    files = sorted(glob.glob(f"{L.BATT}/{CFG['tag'][SEED]}/ep*.npz"))
    off = 0
    for fi, f in enumerate(files):
        z = np.load(f)
        m = len(z["ws"])
        sl = slice(off, off + m)
        off += m
        assert np.array_equal(W["t0"][sl], z["ws"])
        np.savez(os.path.join(dd, os.path.basename(f)), g=z["g"], v0=z["v0"], ws=z["ws"], eid=z["eid"],
                 clip_index=z["clip_index"], **{k: v[sl] for k, v in arms.items()})
    man = json.load(open(f"{L.BATT}/{CFG['man'][SEED]}/manifest.json", encoding="utf-8"))
    man["s1_derived"] = {"note": ("DERIVED dump: arms os/ha/ha0/ha0_ext are the banked refcv7-50,400 T1 arrays "
                                  "(unchanged); damp50/kd_x/P10 were ADDED by S1 (H-V8F-FLOOR-1). The model fields "
                                  "describe the banked checkpoint, not the floors."),
                         "source_dump": os.path.join(L.BATT, CFG["tag"][SEED])}
    json.dump(man, open(os.path.join(dd, "manifest.json"), "w"), indent=1)
    sys.path.insert(0, "E:/Projects/TanitAD/taniteval/tools")
    import refcv3_arm as rc3                                                          # noqa: E402
    EXTRA = [("P10", "os", "paired_os_minus_P10"), ("damp50", "os", "paired_os_minus_damp50"),
             ("kd_x", "os", "paired_os_minus_kd_x"), ("kd_x", "P10", "paired_P10_minus_kd_x"),
             ("ha0_ext", "P10", "paired_P10_minus_ha0ext"), ("ha0_ext", "damp50", "paired_damp50_minus_ha0ext"),
             ("P10", "P10", "paired_P10_minus_P10")]
    _orig = rc3.t1.analyze

    def _wrap(files_, **kw):                      # `paired` IS analyze_refcv3's own `pairs` list: extending it in place
        kw["paired"].extend(EXTRA)                # adds the pairs to families_paired and distance_keeping too
        return _orig(files_, **kw)
    rc3.t1.analyze = _wrap
    tiers = {"P10": "T1", "damp50": "T1", "kd_x": "T1"}
    t = time.time()
    rec = rc3.analyze_refcv3(dd, n_boot=N_BOOT, seed=0, tiers=tiers, lead_block=rc3.ra.LEAD_BLOCK_DEFAULT)
    rec["s1_note"] = ("analyze_refcv3 (the battery's own tool) over the derived dump; extra pairs injected: "
                      + json.dumps([e[2] for e in EXTRA]))
    json.dump(rec, open(os.path.join(RAW, f"analysis_{H}_s{SEED}.json"), "w"), default=str)
    print(f"[{H} s{SEED}] four-family analysis {time.time()-t:.0f}s", flush=True)
print(f"ZZFLOORS-{H}-s{SEED}-DONEZZ")
