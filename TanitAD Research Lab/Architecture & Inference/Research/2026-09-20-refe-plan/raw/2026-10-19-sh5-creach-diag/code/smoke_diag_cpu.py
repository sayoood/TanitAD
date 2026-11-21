"""CPU smoke of diag_pre.DiagTrainer on the REAL inputs (12 tokens) with a random bf16 ctx under the CPU bf16 autocast: (1) plant_oct=8 reproduces the sealed _batch byte for byte,
(2) plant_oct=12 writes a 12-octave code, (3) per-learner lr reaches the optimizer, (4) per-learner fourier_k reaches the model, (5) unique groups, evaluate runs."""
import sys, numpy as np, torch
sys.path.insert(0, "C:/Users/Admin/par18/sh5diag/code")
import diag_pre as DP
L, B, TE = DP.L, DP.B, DP.TE
DD = DP.DD
allt = np.load(f"{DD}/data/sh5_inputs.npz", allow_pickle=True)["token"]
keep, _ = TE.ctx_keep(allt, f"{DD}/ctx_local/ctx_manifest.json")
idx = np.flatnonzero(keep)
# keep 12 tokens spread over folds
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
specs = [TE.Spec("plant", 0, 0) for _ in range(2)]
tr = DP.DiagTrainer(D, specs, rc, prov, log=print)
assert tr.n_shards > 1, tr.n_shards
lns = []
for i, lc in enumerate([{"tag": "k6o8", "lr": 3e-4, "fourier_k": 6, "plant_oct": 8}, {"tag": "k12o12", "lr": 1e-3, "fourier_k": 12, "plant_oct": 12}]):
    cfg = B.SH5Cfg(d_ctx=256, fourier_k=lc["fourier_k"])
    ln = TE.Learner(specs[i], cfg, D, "cpu")
    ln.diag_lr, ln.diag_warmup, ln.plant_oct, ln.tag = lc["lr"], 50, lc["plant_oct"], lc["tag"]
    ln.spec = DP._TagSpec("plant", 0, 0, i, lc["tag"])
    lns.append(ln)
tr.learners = lns
assert lns[1].model.q_in.in_features == 4 + 4 * 12 and lns[0].model.q_in.in_features == 28
toks = [t for t in range(D["T"]) if D["tok_fold"][t] != 0][:3]
ctx_pos = {t: i for i, t in enumerate(range(D["T"]))}
sealed = TE.Trainer._batch(tr, lns[0], toks, ctx_pos, ctx, False)
mine = tr._batch(lns[0], toks, ctx_pos, ctx, False)
assert torch.equal(sealed[2], mine[2]), "plant_oct=8 must be the sealed path"
m12 = tr._batch(lns[1], toks, ctx_pos, ctx, False)[2]
n = len(D["tok_sets"][toks[0]]) * 576
blk = m12[0, -n:].float()
assert torch.all(blk[:, 49] == 1.0) and torch.all(blk[:, 50:] == 0), "12-octave layout: 48 code + clearance (48) + flag (49)"
assert not torch.equal(m12, sealed[2])
tr._train_step(lns[1], toks, ctx_pos, ctx)
assert abs(lns[1].opt.param_groups[0]["lr"] - 1e-3 / 50) < 1e-12, lns[1].opt.param_groups[0]["lr"]
assert L.LR == 3e-4 and L.WARMUP == 50, "globals restored"
tr._train_step(lns[0], toks, ctx_pos, ctx)
assert abs(lns[0].opt.param_groups[0]["lr"] - 3e-4 / 50) < 1e-12
ev, groups = DP.evaluate(tr, D, D["fold"] == 0)
print(sorted(groups), {k: (v["n_sets"], round(v["auc_all_strata"], 3)) for k, v in ev.items()})
assert len(groups) == 2
print("SMOKE_OK")
# the diag path with 8 octaves (forced by an inert ablate tag) equals the sealed path; the ablations change only what they claim
lns[0].ablate = "inert"
mine8 = tr._batch(lns[0], toks, ctx_pos, ctx, False)[2]
assert torch.equal(mine8, sealed[2]), "diag 8-octave plant == sealed plant_block"
lns[0].ablate = "zero_clear"
z = tr._batch(lns[0], toks, ctx_pos, ctx, False)[2]
assert torch.all(z[0, -n:, 32] == 0) and torch.equal(z[0, -n:, :32], sealed[2][0, -n:, :32]) and torch.equal(z[..., 33], sealed[2][..., 33])
lns[0].ablate = "no_plant"
npl = tr._batch(lns[0], toks, ctx_pos, ctx, False)[2]
assert torch.equal(npl, ctx[torch.as_tensor(toks)]), "no_plant = the real ctx"
lns[0].ablate = None
print("SMOKE2_OK")
