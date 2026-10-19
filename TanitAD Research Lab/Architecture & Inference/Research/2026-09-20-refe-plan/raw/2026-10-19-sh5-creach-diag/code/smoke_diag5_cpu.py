"""CPU smoke of diag_pre5 (15 real tokens, random bf16 ctx, CPU bf16 autocast), run BEFORE grid 5 launches:
(1) fp32 lever: without "fp32" `diag_attn5.apply_lever` == `diag_attn.apply_lever` (same class, bit-equal output); with it the layer is Fp32LeverLayer and
    (a) CONTROL: with autocast OFF it equals the cos LeverLayer BIT FOR BIT (diff exactly 0.0),
    (b) LIVE: under bf16 autocast it differs from the bf16 cos layer (diff > 0) and is CLOSER to the all-fp32 reference than the bf16 layer is,
    (c) its temperature gets a non-zero gradient;
(2) AdamW: "adamw_wd" -> AdamW, the decayed group holds exactly the weight matrices except t_emb (s, t_emb, biases, LayerNorms in the 0 group); CONTROL: a learner without it keeps the
    sealed Adam(weight_decay = 1e-4) object type and value;
(3) every grid5.json learner builds, takes one real-data train step with a finite loss, predict + the attention diagnostic run, and the PASS rule is diag_pre3's (RED: one seed is no pass)."""
import copy, json, sys, numpy as np, torch
sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre5 as D5, diag_pre3 as D3, diag_attn as DA, diag_attn5 as DA5
D5.install()
DP, L, B, TE = D3.DP, D3.L, D3.B, D3.TE
DD = D3.DD
COS = {"mode": "cos", "tau0": 16.0, "mult": 10.0}
torch.manual_seed(0)
# ---- (1) the fp32 lever on a bare model
cfg = B.SH5Cfg(d_ctx=256, fourier_k=8)
base = B.SH5Branch(cfg)
with torch.no_grad():                     # the heads are ZERO-INIT (z == 0 for any input): give them weights, or every comparison below reads a trivial 0
    for hd in (base.dac_head, base.reg_head):
        hd.weight.normal_(0, 0.05); hd.bias.normal_(0, 0.05)
m_cos, m_f32, m_none = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
DA.apply_lever(m_cos, COS); DA5.apply_lever(m_f32, dict(COS, fp32=True))
m_cos2 = DA5.apply_lever(copy.deepcopy(base), COS)
assert type(m_cos2.layers[0]) is DA.LeverLayer and type(m_f32.layers[0]) is DA5.Fp32LeverLayer and DA5.apply_lever(m_none, None) is m_none
pts = torch.randn(2, 64, 9, 3) * 20
ctx = torch.randn(1, 7680, 256).to(torch.bfloat16).float()   # bf16-quantised values held in fp32 (the autocast-off reference needs fp32 inputs)
s2t = torch.zeros(2, dtype=torch.long)
def run(m, ac):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=ac):
        lo_, rg_ = m(pts, m.kv(ctx), s2t)
        return torch.cat([lo_.float().reshape(-1), rg_.float().reshape(-1)])
ref = run(m_cos, False)
d_ctrl = (run(m_f32, False) - ref).abs().max().item()
assert d_ctrl == 0.0, f"CONTROL: autocast off, fp32 lever must equal the cos layer bit for bit, got {d_ctrl}"
assert torch.equal(run(m_cos2, True), run(m_cos, True)), "no fp32 flag -> diag_attn.apply_lever bit for bit"
d_bf16, d_f32, d_live = (run(m_cos, True) - ref).abs().mean().item(), (run(m_f32, True) - ref).abs().mean().item(), (run(m_f32, True) - run(m_cos, True)).abs().max().item()
assert d_live > 0.0, "LIVE: under autocast the fp32 lever must change the output"
assert d_f32 < d_bf16, f"fp32 attention must be closer to the all-fp32 reference: {d_f32} vs {d_bf16}"
with torch.autocast("cpu", dtype=torch.bfloat16):
    lo, rg = m_f32(pts, m_f32.kv(ctx), s2t)
(lo.float().pow(2).mean() + rg.float().pow(2).mean()).backward()
g_s = [lay.s.grad.abs().sum().item() for lay in m_f32.layers]
assert all(g > 0 for g in g_s), g_s
print(f"(1) fp32: control diff {d_ctrl}, live diff {d_live:.3e}, mean|.-ref| bf16 {d_bf16:.3e} fp32 {d_f32:.3e}, tau grad {g_s}")
# ---- (3) grid5.json learners on real tokens, random ctx
grid = {"runs": [r for g in ("grid5L", "grid5F") for r in json.load(open(f"C:/Users/Admin/par18/sh5diag/work/{g}.json"))["runs"]]}
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, _ = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
idx = np.flatnonzero(keep)
D0 = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", keep)
pick = []
for f in range(5):
    pick += [t for t in range(D0["T"]) if D0["tok_fold"][t] == f][:3]
k2 = np.zeros(len(allt), bool); k2[idx[pick]] = True
D = TE.load_data(f"{DD}/data/sh5_inputs.npz", f"{DD}/data/sh5_labels.npz", k2)
rng = np.random.default_rng(0)
cx = torch.as_tensor(rng.normal(0, 1, (D["T"], 7680, 256)).astype(np.float32)).to(torch.bfloat16)
prov = TE.MemCtx(cx)
rc = TE.RunCfg(seed=0, cycles=1, inner=1, device="cpu", shard_gib=0.03, autocast=True)
seen = set()
for r in grid["runs"]:
    assert set(r.get("eval_cycles", [])) <= set(range(1, r["cycles"] + 1)) and r["cycles"] in r["eval_cycles"], r["name"]
    assert r["ablate"] == ["zero_clear", "shuffle_clear", "no_plant"], r["name"]
    key = json.dumps(r["learners"], sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    specs, lns = D3.make_learners(D, dict(r, seed=0), prov, "cpu")
    tr = D3.DiagTrainer3(D, specs, rc, prov, log=lambda m: None)
    tr.learners, tr.n_attn = lns, 2
    toks = [t for t in range(D["T"]) if D["tok_fold"][t] != 0][:3]
    ctx_pos = {t: i for i, t in enumerate(range(D["T"]))}
    for ln, lc in zip(lns, r["learners"]):
        # (2) optimizer
        if lc.get("adamw_wd") is not None:
            assert type(ln.opt) is torch.optim.AdamW and ln.opt.param_groups[0]["weight_decay"] == lc["adamw_wd"] and ln.opt.param_groups[1]["weight_decay"] == 0.0
            dec = {id(p) for p in ln.opt.param_groups[0]["params"]}
            names_dec = {n for n, p in ln.model.named_parameters() if id(p) in dec}
            want = {n for n, p in ln.model.named_parameters() if p.ndim >= 2 and not n.endswith("t_emb")}
            assert names_dec == want, names_dec ^ want
            assert not any(n.endswith(".s") or n == "t_emb" or n.endswith("bias") for n in names_dec), names_dec
            assert len(dec) + len(ln.opt.param_groups[1]["params"]) == len(list(ln.model.parameters()))
        else:
            assert type(ln.opt) is torch.optim.Adam and ln.opt.param_groups[0]["weight_decay"] == L.WD == 1e-4, "CONTROL: the sealed optimizer"
        if lc.get("lever") and lc["lever"].get("fp32"):
            assert all(type(l_) is DA5.Fp32LeverLayer for l_ in ln.model.layers)
        tr._train_step(ln, toks, ctx_pos, cx)
        assert np.isfinite(ln.last_loss), (lc["tag"], ln.last_loss)
    _ = tr.predict()
    assert all(ln.attn_n == 2 for ln in lns)
    print(f"(3) {r['name']}: " + ", ".join(f"{ln.tag} {type(ln.opt).__name__} loss {ln.last_loss:.3f} tau {DA.taus(ln.model)[0]}" for ln in lns))
good = {"auc_F": 0.98, "auc_all_strata": 0.96, "r2_reg": 0.6, "ablate_zero_clear": {"auc_all_strata": 0.6, "auc_F": 0.6}, "ablate_shuffle_clear": {"auc_all_strata": 0.6, "auc_F": 0.55}}
assert D3.passing_cells({"G5_s0": {"a": good}, "G5_s1": {"a": good}}) == ["a"] and D3.passing_cells({"G5_s0": {"a": good}}) == []
print("SMOKE5_OK")
