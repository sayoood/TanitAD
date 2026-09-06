"""P1 forensics: did the O5 EMA teacher move by GRADIENT, by EMA only, or is it inconclusive?

D0 (post-hoc-enabled, most direct): the checkpoint's own optimizer state dict. torch AdamW
    creates state[p] ONLY for params it collects, and it collects exactly those with
    p.grad is not None. state[p]['step'] counts steps actually taken.
D1 (pre-registered, PRIMARY): decoupled-weight-decay norm signature, r = ||teacher||/||student||.
D2 (pre-registered, DECISIVE): frozen-tensor identity test.
D3 (pre-registered, CORROBORATING): teacher/student geometry.
"""
import hashlib, json, os, sys, torch

ASSETS = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
EMA_ARMS = ["v7tiny_emao14_30k", "v7tiny_emao14_30k_tauramp"]
ALL_ARMS = ["v7tiny_emao14_30k", "v7tiny_emao14_30k_tauramp", "v7tiny_o14fut30k",
            "v7tiny_postrain30k", "v7tiny_rdw8p30k", "v7tiny_splitp30k",
            "v7tiny_k8clip05p30k", "v7tiny_k60clip05p30k", "v7tiny_postrain30k_freeze"]

def md5t(t):
    return hashlib.md5(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

res = {"D0_optimizer_state": {}, "D1_norm_ratio": {}, "D2_frozen_identity": {},
       "D3_geometry": {}, "C1_C2_controls": {}, "C4_arm_record": {}}

# ---------- C1/C2: controls that must read a known value ----------
ctrl = {}
for arm in ALL_ARMS:
    ck = os.path.join(ASSETS, arm, "ckpt.pt")
    if not os.path.exists(ck):
        continue
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    sd = obj["stack"]
    row = {}
    for name in ("predictor_op.heads.1.weight", "predictor_op.out_proj.weight",
                 "predictor_op.heads.2.weight", "predictor_op.heads.4.weight",
                 "step_readout_op.net.1.weight"):
        if name in sd:
            row[name] = md5t(sd[name])
    ctrl[arm] = row
    del obj, sd
res["C1_C2_controls"]["per_arm_md5"] = ctrl
for name in ("predictor_op.heads.1.weight", "predictor_op.out_proj.weight",
             "predictor_op.heads.2.weight", "step_readout_op.net.1.weight"):
    vals = [ctrl[a][name] for a in ctrl if name in ctrl[a]]
    res["C1_C2_controls"][name] = {"n_arms": len(vals), "n_distinct": len(set(vals))}

# ---------- the EMA arms ----------
for arm in EMA_ARMS:
    ck = os.path.join(ASSETS, arm, "ckpt.pt")
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    sd, opt, cfg = obj["stack"], obj["opt"], obj.get("config", {})
    step = obj.get("step")

    # ---- C4: the arm's own record ----
    arec = {"ckpt_step": step}
    for f in ("run", "stage", "o5_target", "ema_decay", "lr", "wd", "weight_decay", "steps",
              "horizons", "o5_k", "o5_form", "seed", "batch"):
        if isinstance(cfg, dict) and f in cfg:
            arec["cfg." + f] = cfg[f]
    if isinstance(cfg, dict):
        argv = cfg.get("argv")
        if argv is not None:
            s = " ".join(argv) if isinstance(argv, list) else str(argv)
            arec["argv"] = s
            arec["argv_has_o5_target_ema"] = ("--o5-target ema" in s) or ("--o5-target=ema" in s)
        if "weights" in cfg and isinstance(cfg["weights"], dict):
            arec["weights"] = cfg["weights"]
        if "freeze" in cfg and isinstance(cfg["freeze"], dict):
            arec["freeze"] = {k: v for k, v in cfg["freeze"].items()
                              if not isinstance(v, (list, dict))}
    res["C4_arm_record"][arm] = arec

    # ---- D0: optimizer state ----
    d0 = {}
    pgs = opt.get("param_groups", [])
    st = opt.get("state", {})
    idx_in_groups = []
    for gi, g in enumerate(pgs):
        idx_in_groups.extend(list(g.get("params", [])))
    d0["n_param_groups"] = len(pgs)
    d0["n_params_in_groups"] = len(idx_in_groups)
    d0["n_state_entries"] = len(st)
    d0["n_in_groups_WITHOUT_state"] = len([i for i in idx_in_groups if i not in st])
    steps_seen = {}
    for i, s in st.items():
        v = s.get("step")
        try:
            v = int(v.item()) if hasattr(v, "item") else int(v)
        except Exception:
            v = str(v)
        steps_seen[v] = steps_seen.get(v, 0) + 1
    d0["step_histogram_over_state_entries"] = steps_seen
    d0["group_hparams"] = [{k: g[k] for k in ("lr", "weight_decay", "betas")
                            if k in g} for g in pgs]
    res["D0_optimizer_state"][arm] = d0

    # ---- D1/D2/D3: teacher vs student ----
    ema_keys = [k for k in sd if k.startswith("ema_o5_enc.")]
    pairs = []
    for k in ema_keys:
        sub = k[len("ema_o5_enc.module."):]
        stu = "encoder." + sub
        if stu in sd and sd[k].shape == sd[stu].shape:
            pairs.append((k, stu))
    ro_keys = [k for k in sd if k.startswith("ema_o5_ro.")]
    for k in ro_keys:
        sub = k[len("ema_o5_ro.module."):]
        stu = "readout." + sub
        if stu in sd and sd[k].shape == sd[stu].shape:
            pairs.append((k, stu))

    rows = []
    for tk, sk in pairs:
        T = sd[tk].detach().float().flatten()
        S = sd[sk].detach().float().flatten()
        nT, nS = float(T.norm()), float(S.norm())
        diff = (T - S)
        maxabs = float(diff.abs().max())
        sinf = float(S.abs().max())
        rows.append({
            "teacher": tk, "student": sk, "numel": int(T.numel()),
            "norm_teacher": nT, "norm_student": nS,
            "r_norm_ratio": (nT / nS) if nS > 0 else None,
            "bit_identical": md5t(sd[tk]) == md5t(sd[sk]),
            "max_abs_diff": maxabs,
            "rel_dev_inf": (maxabs / sinf) if sinf > 0 else None,
            "rel_l2_dist": (float(diff.norm()) / nS) if nS > 0 else None,
            "cos": float(torch.nn.functional.cosine_similarity(T, S, dim=0)) if nT > 0 and nS > 0 else None,
        })
    rows.sort(key=lambda r: (r["r_norm_ratio"] if r["r_norm_ratio"] is not None else 9e9))
    ratios = sorted([r["r_norm_ratio"] for r in rows if r["r_norm_ratio"] is not None])
    med = ratios[len(ratios) // 2] if ratios else None
    res["D1_norm_ratio"][arm] = {
        "n_pairs": len(rows), "median_r": med,
        "min_r": ratios[0] if ratios else None, "max_r": ratios[-1] if ratios else None,
        "n_r_below_0p95": len([x for x in ratios if x < 0.95]),
        "n_r_above_0p99": len([x for x in ratios if x > 0.99]),
        "PREREG_predicted_r_if_H_GRAD": 0.8607,
    }
    bit = [r for r in rows if r["bit_identical"]]
    res["D2_frozen_identity"][arm] = {
        "n_bit_identical_teacher_eq_student": len(bit),
        "bit_identical_keys": [r["teacher"] for r in bit][:60],
        "max_rel_dev_inf_over_bit_identical": 0.0 if bit else None,
        "min_rel_dev_inf_overall": min([r["rel_dev_inf"] for r in rows if r["rel_dev_inf"] is not None], default=None),
        "max_rel_dev_inf_overall": max([r["rel_dev_inf"] for r in rows if r["rel_dev_inf"] is not None], default=None),
    }
    res["D3_geometry"][arm] = {
        "median_cos": sorted([r["cos"] for r in rows if r["cos"] is not None])[len(rows)//2] if rows else None,
        "median_rel_l2_dist": sorted([r["rel_l2_dist"] for r in rows if r["rel_l2_dist"] is not None])[len(rows)//2] if rows else None,
    }
    res.setdefault("per_tensor", {})[arm] = rows
    del obj, sd, opt

print(json.dumps(res, indent=1, default=str))
