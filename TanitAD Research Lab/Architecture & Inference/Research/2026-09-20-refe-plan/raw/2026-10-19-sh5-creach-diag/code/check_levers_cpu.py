"""SH5 C-REACH diagnosis, grid 3 pre-launch checks (CPU, real inputs, REAL ctx rows of the LOCAL ctx file).  Writes raw/levers_cpu_check.json.
 1. IDENTITY: the "temp" lever at tau = 1 is the sealed layer BIT FOR BIT (fp32 and the CPU bf16 autocast), same init (same Learner seed).
 2. LOGITS: for base / temp (tau perturbed) / cos / qkinit, softmax(diagnostic logits) @ V equals the layer's SDPA output (fp32): the diagnostic reads the logits the model USES.
 3. ESTIMATOR controls: beta_fit on synthetic logits 16 x kernel + per-query offset + noise must read 16; on random logits ~0; on a constant exactly 0.
 4. INIT attention on the plant, per cell (2 fold-0 tokens): beta_eff, logit range over the plant entries / all slots, plant mass, own-entry mass -- the numbers the ideal reader needs
    are beta >= 16 and a plant logit range of ~30.  Also sets the q/k init scale by the rule pre-stated in PREREG_GRID3.md (smallest c in 2,3,4,6,8 with layer-0 plant range >= 30).
 5. REACHABILITY of the temperature parameter s: at step 0 (zero-init heads) d loss / d s is EXACTLY 0.0 (the control); after 3 Adam steps on real fold-1..4 tokens it is > 0 and s moved."""
import json, math, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre3 as D3, diag_attn as DA
DP, L, B, TE = D3.DP, D3.L, D3.B, D3.TE
DD = D3.DD
torch.set_num_threads(8)
R = {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, row_of = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
ev_toks = [t for t in range(D["T"]) if D["tok_fold"][t] == 0][:2]
tr_toks = [t for t in range(D["T"]) if D["tok_fold"][t] != 0][:4]
prov = TE.NpyCtx(f"{DD}/ctx_local/ctx.npy", [row_of[str(t)] for t in D["token"]], "cpu")
shard = prov.load(ev_toks + tr_toks)
ctx_pos = {t: i for i, t in enumerate(ev_toks + tr_toks)}
rc = TE.RunCfg(seed=0, device="cpu", autocast=True)
tr = D3.DiagTrainer3(D, [TE.Spec("plant", 0, 0)], rc, TE.MemCtx(np.zeros((1, 7680, 256), np.float32)), log=lambda m: None)
ac_off = lambda: torch.autocast(device_type="cpu", enabled=False)          # noqa: E731


def cells(qk_scale):
    c = []
    for K in (6, 8):
        base = {"lr": 3e-4, "fourier_k": K, "plant_oct": 8}
        c += [dict(base, tag=f"base_k{K}o8", lever=None), dict(base, tag=f"temp_k{K}o8", lever={"mode": "temp", "tau0": 1.0, "mult": 10.0}),
              dict(base, tag=f"cos16_k{K}o8", lever={"mode": "cos", "tau0": 16.0, "mult": 10.0}), dict(base, tag=f"qk{qk_scale:g}_k{K}o8", lever={"mode": "qkinit", "scale": qk_scale})]
    return c


def build(lcs, seed=0):
    run = {"seed": seed, "learners": lcs}
    _, lns = D3.make_learners(D, run, prov, "cpu")
    return lns


def init_stats(ln, ac):
    acc = None
    for t in ev_toks:
        pts, s2t, ctx, _, _, _ = tr._batch(ln, [t], ctx_pos, shard, False)
        acc = DA.acc_add(acc, DA.attn_token_stats(ln.model, pts, ctx, min(ln.model.c.fourier_k, ln.plant_oct), ac))
    return DA.acc_report(acc, len(ev_toks))


# ---- 3. estimator controls ---------------------------------------------------------------------------------------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
xq, xe = torch.rand(300, 2, generator=g) * 40, torch.rand(500, 2, generator=g) * 40
ker = DA.kernel(xq, xe, 6)
syn = 16.0 * ker + torch.randn(300, 1, generator=g) * 5 + torch.randn(300, 500, generator=g) * 0.1
n, d, s = DA.beta_fit(syn[None], ker)
rnd = DA.beta_fit(torch.randn(1, 300, 500, generator=g) * 3, ker)
cst = DA.beta_fit(torch.full((1, 300, 500), 7.0), ker)
R["estimator"] = {"synthetic_beta16": float(n / d), "random": float(rnd[0] / rnd[1]), "constant": float(cst[0] / cst[1])}
assert abs(R["estimator"]["synthetic_beta16"] - 16) < 0.05 and abs(R["estimator"]["random"]) < 0.5 and R["estimator"]["constant"] == 0.0, R["estimator"]

# ---- 1. identity + 2. logits ---------------------------------------------------------------------------------------------------------------------------------------------------------
lns = build(cells(4.0)[:4])
base, temp, cos, qk = lns
t = ev_toks[0]
pts, s2t, ctx, _, _, _ = tr._batch(base, [t], ctx_pos, shard, False)
idn = {}
hb, ht = base.model.state_dict(), temp.model.state_dict()
idn["shared_params_equal"] = all(torch.equal(hb[k], ht[k]) for k in hb)
idn["extra_params_in_temp"] = sorted(set(ht) - set(hb))
# the heads are zero-init (both logits would be 0.0 trivially), so give BOTH models the same random DAC head: the logit then reads the pooled features
torch.manual_seed(5)
wd, bd = torch.randn(1, 256), torch.randn(1)
for m in (base.model, temp.model):
    with torch.no_grad():
        m.dac_head.weight.copy_(wd); m.dac_head.bias.copy_(bd)
for name, ac in (("fp32", ac_off), ("bf16_autocast", tr._ac)):
    with torch.no_grad(), ac():
        cx = ctx.float() if name == "fp32" else ctx
        zb, _ = base.model(pts, base.model.kv(cx), s2t)
        zt, _ = temp.model(pts, temp.model.kv(cx), s2t)
    idn[name + "_logit_bit_equal"] = bool(torch.equal(zb, zt))
    idn[name + "_max_abs"] = float((zb.float() - zt.float()).abs().max())
for m in (base.model, temp.model):
    with torch.no_grad():
        m.dac_head.weight.zero_(); m.dac_head.bias.zero_()
R["identity_temp_tau1_vs_sealed"] = idn
assert idn["fp32_logit_bit_equal"] and idn["bf16_autocast_logit_bit_equal"], idn

with torch.no_grad():
    for lay in temp.model.layers:
        lay.s.add_(torch.tensor([0.05, -0.03, 0.1, 0.0]))
rec = {}
for name, ln in (("base", base), ("temp_perturbed", temp), ("cos16", cos), ("qkinit4", qk)):
    m = ln.model
    with torch.no_grad(), ac_off():
        kv = m.kv(ctx.float())
        h = (m.q_out(F.gelu(m.q_in(B.point_features(pts, m.c.fourier_k)))) + m.t_emb).reshape(pts.shape[0], -1, 256)
        lay, (K, V) = m.layers[0], kv[0]
        q, Kq = DA.layer_qk(lay, h, K[s2t])
        lg = (q @ Kq.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mine = torch.softmax(lg, -1) @ V[s2t]
        sdpa = F.scaled_dot_product_attention(q, Kq, V[s2t])
    rec[name] = float((mine - sdpa).abs().max())
R["logit_reconstruction_max_abs"] = rec
assert max(rec.values()) < 1e-4, rec
with torch.no_grad():
    for lay in temp.model.layers:
        lay.s.zero_()

# ---- 4. init stats; choose the q/k init scale --------------------------------------------------------------------------------------------------------------------------------------
R["qkinit_scale_scan_layer0_range_plant"] = {}
chosen = None
for c in (2.0, 3.0, 4.0, 6.0, 8.0):
    ln = build([{"tag": "q", "lr": 3e-4, "fourier_k": 6, "plant_oct": 8, "lever": {"mode": "qkinit", "scale": c}}])[0]
    st = init_stats(ln, tr._ac)
    rp = float(np.mean(st[0]["range_plant_mean"]))
    R["qkinit_scale_scan_layer0_range_plant"][str(c)] = {"range_plant_mean": rp, "range_all_mean": float(np.mean(st[0]["range_all_mean"])), "plant_mass": float(np.mean(st[0]["plant_mass"]))}
    if chosen is None and rp >= 30.0:
        chosen = c
R["qkinit_scale_chosen"] = chosen if chosen is not None else 8.0
R["init"] = {}
for lc in cells(R["qkinit_scale_chosen"]):
    ln = build([lc])[0]
    R["init"][lc["tag"]] = init_stats(ln, tr._ac)
    a0 = R["init"][lc["tag"]][0]
    print(lc["tag"], "L0 beta_max", a0["beta_eff_max"], "range_plant", a0["range_plant_mean"], "plant_mass", a0["plant_mass"], "self", a0["self_mass"], flush=True)

# ---- 5. reachability of s ---------------------------------------------------------------------------------------------------------------------------------------------------------
reach = {}
for lc in cells(R["qkinit_scale_chosen"]):
    if not lc["lever"] or lc["lever"]["mode"] == "qkinit" or lc["fourier_k"] != 6:
        continue
    ln = build([lc])[0]
    pts_, s2t_, ctx_, y_, tgt_, _ = tr._batch(ln, tr_toks, ctx_pos, shard, True)
    with tr._ac():
        lg_, rg_ = ln.model(pts_, ln.model.kv(ctx_), s2t_)
    B.loss_terms(lg_, rg_, y_, tgt_, ln.w_reg)["total"].backward()
    g0 = [float(lay.s.grad.abs().sum()) for lay in ln.model.layers]
    s_before = [lay.s.detach().clone() for lay in ln.model.layers]
    ln.opt.zero_grad(set_to_none=True)
    for _ in range(3):
        tr._train_step(ln, tr_toks, ctx_pos, shard)
    ln.opt.zero_grad(set_to_none=True)
    pts_, s2t_, ctx_, y_, tgt_, _ = tr._batch(ln, tr_toks, ctx_pos, shard, True)
    with tr._ac():
        lg_, rg_ = ln.model(pts_, ln.model.kv(ctx_), s2t_)
    B.loss_terms(lg_, rg_, y_, tgt_, ln.w_reg)["total"].backward()
    g3 = [float(lay.s.grad.abs().sum()) for lay in ln.model.layers]
    moved = [float((lay.s.detach() - s0).abs().sum()) for lay, s0 in zip(ln.model.layers, s_before)]
    in_opt = sum(any(p is lay.s for p in ln.opt.param_groups[0]["params"]) for lay in ln.model.layers)
    reach[lc["tag"]] = {"grad_s_step0_ZERO_CONTROL": g0, "grad_s_after3": g3, "s_moved_after3": moved, "s_in_optimizer_layers": in_opt}
    assert all(v == 0.0 for v in g0) and all(v > 0 for v in g3) and all(v > 0 for v in moved) and in_opt == 2, reach[lc["tag"]]
R["reach_tau"] = reach
R["ALL_OK"] = True
json.dump(R, open("C:/Users/Admin/par18/sh5diag/raw/levers_cpu_check.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ("estimator", "identity_temp_tau1_vs_sealed", "logit_reconstruction_max_abs", "qkinit_scale_scan_layer0_range_plant", "qkinit_scale_chosen", "reach_tau")}, indent=1))
print("CHECK_OK")
