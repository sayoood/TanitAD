"""SH5 C-REACH diagnosis, probe 4 (CPU, synthetic only): did the sealed synthetic C-REACH pass (test_sh5_pipeline.TestLearning.test_creach_planted_clearance_is_recovered, AUC > 0.9, R2 > 0.5)
read the PLANT, or the synthetic ctx's info channels (wl, wr written into channels 0, 1 of EVERY slot, sh5_synth.make_synth info=True)?
Same world (T=60, seed=21, zl_noise=0.3), same CPU recipe (lr 2e-3, 40 epochs, d=32, fourier_k=3, p_sub 8, 4 tokens/step), the plant arm only, with
  (a) info=True  (the sealed test's ctx), (b) info=False (pure-noise ctx: the plant is the ONLY source of the clearance).  Also reports the proposal spacing of the synthetic world."""
import json, sys, numpy as np
sys.path.insert(0, "C:/Users/Admin/par18/sh5/code")
import sh5_lib as L, sh5_synth as SY, sh5_train_eval as TE, sh5_branch as B
CFG = B.SH5Cfg(d=32, heads=2, layers=2, ffn=64, d_ctx=40, fourier_k=3)
L.LR = 2e-3
R = {}
for info in (True, False):
    D, ctx = SY.make_synth(T=60, seed=21, info=info, zl_noise=0.3)
    specs = [TE.Spec("plant", k, 0) for k in range(L.FOLDS)]
    rc = TE.RunCfg(device="cpu", autocast=False, seed=0, p_sub=8, tokens_per_step=4, epochs_override=40, shard_gib=10.0)
    tr = TE.Trainer(D, specs, rc, TE.MemCtx(ctx), cfg=CFG, log=lambda m: None)
    tr.train()
    g = tr.predict()["plant_s0"]
    FL, _ = L.refs()
    lab = D["lab"][..., L.DAC_COL]
    st = FL.pair_stats(g["z"], lab == 0.0, lab == 1.0, D["code"], names=("ALLQ", "Q123"))
    a = {k: round(float(n.sum() / d.sum()), 4) for k, (n, d) in st.items()}
    R[f"info={info}"] = {"auc": a, "r2_reg": round(TE.r2_reg(g["reg"], D["tgt"]), 4)}
    print(R[f"info={info}"], flush=True)
pts = D["pts"]
nn = []
for k in range(9):
    v = []
    for i in range(min(len(pts), 100)):
        p = pts[i, :, k, :2].astype(np.float64); dm = np.linalg.norm(p[:, None] - p[None], axis=-1); np.fill_diagonal(dm, np.inf); v.append(np.median(dm.min(1)))
    nn.append(round(float(np.median(v)), 3))
R["synthetic_nearest_other_proposal_m_median_per_keytime"] = nn
print(json.dumps(R, indent=1))
json.dump(R, open("C:/Users/Admin/par18/sh5diag/raw/probe4_synth_plant_only.json", "w"), indent=1)
