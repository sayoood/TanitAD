"""SH5 C-REACH diagnosis, probe 3 (CPU): REACHABILITY of the planted channel from the sealed trainer path (hypothesis c).
Real inputs + labels, 6 real ctx rows read from the LOCAL ctx file (bf16 bits), the SEALED Trainer._batch (which writes the plant) and the sealed model under the CPU bf16 autocast.
Measures d(loss)/d(ctx) on (i) the planted CLEARANCE channel of the planted slots, (ii) the planted code channels, (iii) the real-ctx slots:
  * ZERO CONTROL: at step 0 both heads are zero-initialised, so EVERY ctx gradient must be EXACTLY 0.0 (proves the probe can read a zero);
  * after 3 Adam steps on the same batch the planted-clearance gradient must be > 0 (the plant reaches the loss);
  * attention mass on the planted slots at init and after the steps (how much of the read the plant gets among 7,680 slots)."""
import json, sys, numpy as np, torch
sys.path.insert(0, "C:/Users/Admin/par18/sh5/code")
import sh5_lib as L, sh5_branch as B, sh5_train_eval as TE
import torch.nn.functional as F
DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
toks = [t for t in range(D["T"]) if D["tok_fold"][t] != 0][:6]
prov = TE.NpyCtx(f"{DD}/ctx_local/ctx.npy", [row_of[str(t)] for t in D["token"]], "cpu")
shard = prov.load(toks)                                    # bf16 [6, 7680, 256], the real cached ctx
ctx_pos = {t: i for i, t in enumerate(toks)}
rc = TE.RunCfg(seed=0, device="cpu", autocast=True)
tr = TE.Trainer(D, [TE.Spec("plant", 0, 0)], rc, TE.MemCtx(np.zeros((1, 7680, 256), np.float32)), log=lambda m: None)
ln = tr.learners[0]
pts, s2t, ctx, y, tgt, sets = tr._batch(ln, toks, ctx_pos, shard, True)
n_pl = [len(D["tok_sets"][t]) * 576 for t in toks]
P = ctx.shape[1]


def probe(model):
    c = ctx.float().clone().requires_grad_(True)
    attn_mass = []
    with tr._ac():
        kv = model.kv(c)
        logit, reg = model(pts, kv, s2t)
        # attention mass of layer 0 on the planted slots (recomputed explicitly)
        lay = model.layers[0]
        h = model.q_out(F.gelu(model.q_in(B.point_features(pts, model.c.fourier_k)))) + model.t_emb
        S, Pp = pts.shape[:2]
        h = h.reshape(S, Pp * 9, model.c.d)
        q = lay.q(lay.ln_q(h)).view(S, Pp * 9, model.c.heads, -1).transpose(1, 2)
        K = kv[0][0][s2t]
        a = torch.softmax((q.float() @ K.float().transpose(-1, -2)) / q.shape[-1] ** 0.5, -1)
        for si in range(S):
            npl = n_pl[int(s2t[si])]
            attn_mass.append(float(a[si, :, :, P - npl:].sum(-1).mean()))
    lt = B.loss_terms(logit, reg, y, tgt, ln.w_reg)
    lt["total"].backward()
    g = c.grad.abs()
    pl_clear = sum(float(g[i, P - n_pl[i]:, 32].sum()) for i in range(len(toks)))
    pl_code = sum(float(g[i, P - n_pl[i]:, :32].sum()) for i in range(len(toks)))
    real = sum(float(g[i, :P - n_pl[i], :].sum()) for i in range(len(toks)))
    return {"grad_planted_clearance": pl_clear, "grad_planted_code": pl_code, "grad_real_slots": real, "attn_mass_on_plant_layer0_mean": float(np.mean(attn_mass)),
            "plant_slot_fraction": float(np.mean([n / P for n in n_pl])), "loss": float(lt["total"])}


R = {"tokens": [str(D["token"][t]) for t in toks], "n_planted": n_pl}
R["step0_ZERO_CONTROL"] = probe(ln.model)
ln.model.zero_grad(set_to_none=True)
for _ in range(3):
    tr._train_step(ln, toks, ctx_pos, shard)
R["after_3_steps"] = probe(ln.model)
R["zero_control_ok"] = R["step0_ZERO_CONTROL"]["grad_planted_clearance"] == 0.0 and R["step0_ZERO_CONTROL"]["grad_real_slots"] == 0.0
R["reach_ok"] = R["after_3_steps"]["grad_planted_clearance"] > 0.0
print(json.dumps(R, indent=1))
json.dump(R, open("C:/Users/Admin/par18/sh5diag/raw/probe3_reach_grad.json", "w"), indent=1)
