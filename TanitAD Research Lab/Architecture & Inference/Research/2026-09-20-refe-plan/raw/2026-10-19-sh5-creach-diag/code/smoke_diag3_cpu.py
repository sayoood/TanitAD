"""CPU smoke of diag_pre3 (12 real tokens, random bf16 ctx, CPU bf16 autocast): (1) DiagTrainer3.predict == the sealed Trainer.predict for a base learner (z, reg bit-equal);
(2) a lever learner trains (temp / cos / qkinit), its tau is reported, the attention diagnostic runs and is skipped under an ablation; (3) per-learner p_sub = 64 reaches the batch, p_sub = 32
keeps the sealed path byte for byte; (4) passing_cells / best_lever / resolve_best on hand-built summaries, incl. RED cases (a cell whose zero_clear ablation does NOT collapse must NOT pass;
a cell passing on ONE seed must NOT pass)."""
import json, sys, numpy as np, torch
sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre3 as D3, diag_attn as DA
DP, L, B, TE = D3.DP, D3.L, D3.B, D3.TE
DD = D3.DD
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
ctx = torch.as_tensor(rng.normal(0, 1, (D["T"], 7680, 256)).astype(np.float32)).to(torch.bfloat16)
prov = TE.MemCtx(ctx)
rc = TE.RunCfg(seed=0, cycles=1, inner=1, device="cpu", shard_gib=0.03, autocast=True)
run = {"seed": 0, "learners": [
    {"tag": "base_k6o8", "lr": 3e-4, "fourier_k": 6, "plant_oct": 8, "lever": None},
    {"tag": "temp_k6o8", "lr": 3e-4, "fourier_k": 6, "plant_oct": 8, "lever": {"mode": "temp", "tau0": 1.0, "mult": 10.0}},
    {"tag": "cos16_k8o8", "lr": 3e-4, "fourier_k": 8, "plant_oct": 8, "lever": {"mode": "cos", "tau0": 16.0, "mult": 10.0}, "p_sub": 64},
    {"tag": "qk6_k6o8", "lr": 3e-3, "fourier_k": 6, "plant_oct": 8, "lever": {"mode": "qkinit", "scale": 6.0}}]}
specs, lns = D3.make_learners(D, run, prov, "cpu")
tr = D3.DiagTrainer3(D, specs, rc, prov, log=lambda m: None)
tr.learners = lns
tr.n_attn = 2
assert tr.n_shards > 1
toks = [t for t in range(D["T"]) if D["tok_fold"][t] != 0][:3]
ctx_pos = {t: i for i, t in enumerate(range(D["T"]))}
# (3) p_sub: 32 = the sealed batch byte for byte; 64 reaches the batch
st0 = lns[0].rng.bit_generator.state
b_sealed = TE.Trainer._batch(tr, lns[0], toks, ctx_pos, ctx, True)
lns[0].rng.bit_generator.state = st0
b_mine = tr._batch(lns[0], toks, ctx_pos, ctx, True)
assert b_mine[0].shape[1] == 32 and all(torch.equal(a, b) for a, b in zip(b_mine[:5], b_sealed[:5])), "p_sub 32 = the sealed batch"
lns[2].rng.bit_generator.state = st0
b64 = tr._batch(lns[2], toks, ctx_pos, ctx, True)
assert b64[0].shape[1] == 64 and b64[3].shape[1] == 64, b64[0].shape
assert torch.equal(b64[2], b_sealed[2]), "p_sub 64 path writes the same plant (8 octaves, un-ablated)"
assert tr.rc.p_sub == 32, "rc restored"
# (2) train one step each, tau moves only on lever layers
for ln in lns:
    tr._train_step(ln, toks, ctx_pos, ctx)
assert DA.taus(lns[0].model) == [None, None] and DA.taus(lns[1].model)[0] is not None
assert abs(lns[3].opt.param_groups[0]["lr"] - 3e-3 / 50) < 1e-12
# (1) predict == sealed predict for every learner (z, reg)
for ln in lns:
    ln.ablate = None
mine = tr.predict()
sealed = TE.Trainer.predict(tr)
for g in mine:
    assert np.array_equal(mine[g]["z"], sealed[g]["z"], equal_nan=True) and np.array_equal(mine[g]["reg"], sealed[g]["reg"]), g
assert all(ln.attn_n == 2 and ln.attn_acc is not None for ln in lns)
rep = DA.acc_report(lns[2].attn_acc, lns[2].attn_n)
assert len(rep) == 2 and len(rep[0]["beta_eff"]) == 4
for ln in lns:
    ln.ablate = "zero_clear"
_ = tr.predict()
assert all(ln.attn_n == 0 for ln in lns), "no attention diagnostic under an ablation"
for ln in lns:
    ln.ablate = None
ev, _ = DP.evaluate(tr, D, D["fold"] == 0)
print({k: round(v["auc_all_strata"], 3) for k, v in ev.items()})
# (4) the pass / BEST logic
good = {"auc_F": 0.98, "auc_all_strata": 0.96, "r2_reg": 0.6, "ablate_zero_clear": {"auc_all_strata": 0.6, "auc_F": 0.6}, "ablate_shuffle_clear": {"auc_all_strata": 0.6, "auc_F": 0.55}}
nocollapse = dict(good, ablate_zero_clear={"auc_all_strata": 0.95, "auc_F": 0.96})
lowF = dict(good, auc_F=0.969)
summ = {"R6_s0": {"a": good, "b": nocollapse, "c": good, "d": lowF}, "R7_s1": {"a": good, "b": nocollapse, "c": lowF, "d": good}}
assert D3.passing_cells(summ) == ["a"], D3.passing_cells(summ)
assert D3.passing_cells({"R6_s0": {"a": good}}) == [], "one seed is not a pass"
grid = {"runs": [{"name": "R6_s0", "learners": [{"tag": "a", "lever": None, "fourier_k": 6, "plant_oct": 8}, {"tag": "b", "lever": {"mode": "temp", "tau0": 1, "mult": 10}, "fourier_k": 6, "plant_oct": 8},
                                                {"tag": "c", "lever": {"mode": "cos", "tau0": 16, "mult": 10}, "fourier_k": 8, "plant_oct": 8}, {"tag": "d", "lever": {"mode": "qkinit", "scale": 6}, "fourier_k": 6, "plant_oct": 8}]}]}
best = D3.best_lever(summ, grid)
assert best["from_tag"] == "b", best          # b: min F 0.98 (its ablation does not matter for BEST); c, d: min F 0.969; a is not a lever
g4 = D3.resolve_best({"runs": [{"name": "R8_s0", "learners": [{"tag": "BEST_lr3e-3", "lever": "BEST", "lr": 3e-3}]}]}, best)
assert g4["runs"][0]["learners"][0]["lever"] == {"mode": "temp", "tau0": 1, "mult": 10} and g4["runs"][0]["learners"][0]["fourier_k"] == 6
print("SMOKE3_OK")
