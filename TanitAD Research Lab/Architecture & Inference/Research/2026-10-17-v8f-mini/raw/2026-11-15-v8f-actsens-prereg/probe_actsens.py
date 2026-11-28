"""ACTSENS probes (dev-box CPU, read-only on checkpoints; writes only under F:/TanitAD-scratch/actsens/).

Part A  MEASURED dose input on R35_s0 / R35_s1: the level-1 IDM head's per-channel fit (R^2 of acceleration and of yaw rate)
        and the level-1 predictor's per-channel forward sensitivity, on the 2,397 TRAIN windows of 141 TRAIN clips already
        collected by hwm_plan/collect_latents_train_R35.py (latents_train_R35_s{0,1}.pt).
Part B  Reachability of the IDM-weight lever at MODEL level (cfg.wm_gamma -> LevelConfig.gamma -> gamma * L_idm -> backward):
        IDM-head gradient at gamma 300 = 3.0 x gamma 100 (analytic identity), gamma 0 (wm a) reads EXACTLY 0.0 (control),
        the predictor gradient is bit-identical across gamma (identity control), and a builder mutation
        (cfg says 300, the loss's LevelConfig keeps 100) is invisible to the trainer's present wm_idm_gamma check
        and RED on the stack's own built_levers() reader and on the gradient ratio.
"""
import dataclasses, json, os, sys
import torch

torch.set_num_threads(2)
OUT = "F:/TanitAD-scratch/actsens/probe_result.json"
PULL = "C:/Users/Admin/t2v8f_pulls"
LAT = "F:/TanitAD-scratch/hwm_plan"

import tanitad
from tanitad.models import v8f_mini as M

res = {"tanitad_file": tanitad.__file__}


def cfg_from_ckpt(ck, **over):
    names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names}
    kw.update(over)
    return M.V8FMiniConfig(**kw)


def build_wm(cfg, seed=0):
    torch.manual_seed(seed)
    return M.WMAux(cfg)


# ---------------- Part A
partA = {}
for s in ("s0", "s1"):
    tag = f"R35_{s}"
    ck = torch.load(f"{PULL}/ladder_{tag}/ckpt_last.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(ck)
    wm = build_wm(cfg)
    sd = {k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}
    missing, unexpected = wm.load_state_dict(sd, strict=True), None
    wm.eval()
    L = torch.load(f"{LAT}/latents_train_{tag}.pt", map_location="cpu", weights_only=False)
    z = L["Z1"][tag].float()                       # [N, 16, 128]
    a = wm.scale_actions(L["A_raw"].float())[:, :-1]  # [N, 15, 2]
    ok = L["ok"]
    v = (ok[:, :-1] & ok[:, 1:])                     # valid transitions
    st = wm.stack
    with torch.no_grad():
        ip = st.idm_heads[0](z[:, :-1], z[:, 1:])
        e = (ip - a)[v]                                # [n, 2]
        tv = a[v]
        mse = e.pow(2).mean(0)
        var = tv.var(0, unbiased=False)
        # level-1 forward sensitivity, planning form (context 1): latent change from a +1 sd step of one action channel,
        # relative to the real one-step latent change. sd in the WM's own action units over the same transitions.
        z0 = z[:, :-1][v]                              # [n, 128]
        a0 = a[v]                                      # [n, 2]
        z1t = z[:, 1:][v]
        sd_a = a0.std(0, unbiased=False)
        base = st.rollout(1, z0, a0[:, None, :], context=1)[:, 0]
        real = (z1t - z0).norm(dim=-1).mean()
        err = (base - z1t).norm(dim=-1).mean()
        sens = {}
        for c, nm in ((0, "acc"), (1, "yaw_rate")):
            ap = a0.clone()
            ap[:, c] += sd_a[c]
            d = (st.rollout(1, z0, ap[:, None, :], context=1)[:, 0] - base).norm(dim=-1).mean()
            sens[nm] = {"dz_per_sd": float(d), "rel_to_real_step": float(d / real), "rel_to_pred_error": float(d / err)}
    partA[tag] = {
        "n_transitions": int(v.sum()), "n_windows": int(z.shape[0]),
        "idm_L1_mse_mean": float(mse.mean()),
        "idm_L1": {nm: {"mse": float(mse[c]), "target_var": float(var[c]), "r2": float(1 - mse[c] / var[c])} for c, nm in ((0, "acc"), (1, "yaw_rate"))},
        "action_sd_wm_units": {"acc": float(sd_a[0]), "yaw_rate": float(sd_a[1])},
        "real_step_norm": float(real), "pred_one_step_err_norm": float(err),
        "fwd_sens_L1_ctx1": sens,
        "levels_built": st.built_levers(),
    }
res["partA"] = partA

# ---------------- Part B
ck = torch.load(f"{PULL}/ladder_R35_s0/ckpt_last.pt", map_location="cpu", weights_only=False)
g = torch.Generator().manual_seed(7)
B, F = 2, int(ck["cfg"]["hmax"])
zb = torch.randn(B, F, int(ck["cfg"]["wm_dim"]), generator=g)
ab = torch.randn(B, F - 1, 2, generator=g)


def grads(wm, seed=11):
    wm.zero_grad(set_to_none=True)
    out = wm.stack.loss(zb, ab, generator=torch.Generator().manual_seed(seed))
    out["total"].backward()
    gi = [float(sum(p.grad.abs().sum() for p in h.parameters() if p.grad is not None)) for h in wm.stack.idm_heads]
    gp = [torch.cat([p.grad.flatten() for p in pr.parameters() if p.grad is not None]) for pr in wm.stack.predictors]
    return gi, gp, {k: float(v) for k, v in out.items() if k.startswith("idm/")}


rows = {}
cfg_b = cfg_from_ckpt(ck)                                   # wm b, gamma 100 (the record)
cfg_3 = cfg_from_ckpt(ck, wm_gamma=300.0)                   # the lever at the model level (no trainer flag exists)
cfg_a = cfg_from_ckpt(ck, wm="a")                           # gamma 0: the control
gi_b, gp_b, idm_b = grads(build_wm(cfg_b))
gi_3, gp_3, idm_3 = grads(build_wm(cfg_3))
gi_a, gp_a, idm_a = grads(build_wm(cfg_a))
rows["gamma100_idm_head_grad_abs"] = gi_b
rows["gamma300_idm_head_grad_abs"] = gi_3
rows["gamma0_idm_head_grad_abs"] = gi_a
rows["ratio_300_over_100"] = [x / y for x, y in zip(gi_3, gi_b)]
rows["gamma0_exact_zero"] = all(x == 0.0 for x in gi_a)
rows["predictor_grad_bit_identical_100_vs_300"] = all(torch.equal(x, y) for x, y in zip(gp_b, gp_3))
rows["predictor_grad_bit_identical_100_vs_0"] = all(torch.equal(x, y) for x, y in zip(gp_b, gp_a))
rows["idm_terms_equal_100_vs_300"] = idm_b == idm_3
rows["stack_gamma_built"] = {"b": [c.gamma for c in build_wm(cfg_b).stack.cfg], "300": [c.gamma for c in build_wm(cfg_3).stack.cfg], "a": [c.gamma for c in build_wm(cfg_a).stack.cfg]}

# mutation: cfg declares 300, the builder hands the loss LevelConfigs with gamma 100 (a builder that ignores the field)
wm_m = build_wm(cfg_3)
wm_m.stack.cfg = [dataclasses.replace(c, gamma=100.0) for c in wm_m.stack.cfg] if isinstance(wm_m.stack.cfg, list) else type(wm_m.stack.cfg)(dataclasses.replace(c, gamma=100.0) for c in wm_m.stack.cfg)
gi_m, _, _ = grads(wm_m)
present_check_green = float(wm_m.gamma) == float(cfg_3.wm_gamma)          # what v8f_mini_train.py:1500 compares (built_levers()['wm_idm_gamma'] = WMAux.gamma)
proposed_reader = [r["gamma"] for r in wm_m.stack.built_levers().values()] if isinstance(wm_m.stack.built_levers(), dict) else None
rows["mutation_builder_ignores_gamma"] = {
    "present_trainer_check_passes": present_check_green,
    "stack_built_levers_gamma": proposed_reader,
    "proposed_reader_red": proposed_reader != [300.0, 300.0, 300.0],
    "grad_ratio_vs_gamma100": [x / y for x, y in zip(gi_m, gi_b)],
    "grad_probe_red": not all(abs(x / y - 3.0) < 1e-4 for x, y in zip(gi_m, gi_b)),
}
res["partB"] = rows
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:6000])
