"""D-GSTR-1 P2 downstream -- does a REPAIRED g_str change the PLAN?

The frozen-trunk refit proves the SIGNAL can turn left. This asks the next
question with the ONLY instrument available without a retrain: patch
`str_goal_head`'s OUTPUT with the certified E arm's head (a forward hook, no
file edited) and roll the same windows twice -- baseline and patched -- with
everything else bit-identical.

⛔ WHAT THIS IS AND IS NOT. It is a CAUSAL intervention on the deployed model's
strategic goal at inference. It is NOT a retrained arm: the FiLM that consumes
`g_str`, and every weight downstream, were trained against the BROKEN goal, so
a small plan movement here is a lower bound on what a retrain would give, not
an estimate of it.

⛔ THE CONTROL THAT MAKES IT ADMISSIBLE: a NULL-PATCH arm whose hook replaces
`g_str` with ITSELF. It must move the plan by EXACTLY 0.0. A patch harness that
perturbs the plan by existing would make every number below meaningless.

ASCII-only.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/nvidia/refcv4b/ckpt_40284_FINAL.pt")
    ap.add_argument("--config", default="/home/nvidia/refcv4b/config.json")
    ap.add_argument("--episodes", default="/home/nvidia/navpred/data_eval")
    ap.add_argument("--labels",
                    default="/home/nvidia/data/v72/labels/"
                            "s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--nav-source", default="v72")
    ap.add_argument("--episodes-n", type=int, default=45)
    ap.add_argument("--window-stride", type=int, default=5)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head", default="/home/nvidia/navroute/gstr_head_E.pt")
    ap.add_argument("--out", default="/home/nvidia/navroute/GSTR_DOWNSTREAM.json")
    a = ap.parse_args()

    sys.path.insert(0, "/home/nvidia/navpred/taniteval/tools")
    import refcv3_arm as A
    A._bootstrap_paths()
    tr = A.trainer()
    model, cfg, targs, prov = A.load_model(a.ckpt, a.config, device=a.device)
    model.eval()
    eps, files, clip_ids, ds, lman, join, src, raw_off = A.build_corpus(
        a, cfg, prov)
    W, steps, dev = int(cfg.core.window), int(prov["decoder_steps"]), a.device

    hd = torch.load(a.head, map_location=dev, weights_only=False)
    Wr = hd["weight"].to(dev).float()
    br = hd["bias"].to(dev).float()

    MODE = {"m": "off"}

    def hook(_mod, inp, out):
        if MODE["m"] == "off":
            return None
        ctx = inp[0]
        if MODE["m"] == "null":
            return out                      # the CONTROL: patch with itself
        return torch.nn.functional.linear(ctx, Wr, br)
    h = model.str_goal_head.register_forward_hook(hook)

    by_ep = {}
    for i, (e_i, t) in enumerate(ds.index):
        if (t % max(1, a.window_stride)) == 0:
            by_ep.setdefault(e_i, []).append((i, t))

    rows = []
    for e_i in sorted(by_ep):
        ep = eps[e_i]
        for (wi, t) in by_ep[e_i]:
            item = ds[wi]
            fv = item.get("future_valid_ext")
            if fv is None or len(fv) < 60 or not bool(fv[59]):
                continue
            fr = tr.frames_to_device(item["frames"][None], dev)
            v0_t = item["pose_last"].float()[3].reshape(1).to(dev)
            nv = item.get("nav_cmd")
            nav_t = nv.reshape(1).long().to(dev) if nv is not None else None
            ego = None
            if bool(cfg.ego_state_inject):
                ego = tr.v3.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
            outs = {}
            with torch.no_grad():
                for m in ("off", "null", "patch"):
                    MODE["m"] = m
                    o = model(fr, nav_cmd=nav_t, v0=v0_t, steps=steps,
                              ego_state=ego)
                    outs[m] = (o["traj"][0].float().cpu().numpy(),
                               o["g_str"][0].float().cpu().numpy())
            MODE["m"] = "off"
            gt = tr.refb_labels.waypoint_targets(
                item["pose_last"].float()[None],
                item["future_poses_ext"].float()[None], [60])
            rows.append({
                "ep": int(e_i), "t": int(t + W - 1),
                "gt_y6": float(gt[0, -1, 1]),
                "base_y": float(outs["off"][0][-1, 1]),
                "null_y": float(outs["null"][0][-1, 1]),
                "patch_y": float(outs["patch"][0][-1, 1]),
                "base_lat": float(outs["off"][1][1]),
                "patch_lat": float(outs["patch"][1][1]),
                "d_null": float(np.linalg.norm(outs["null"][0][-1]
                                               - outs["off"][0][-1])),
                "d_patch": float(np.linalg.norm(outs["patch"][0][-1]
                                                - outs["off"][0][-1])),
            })
        print(f"# ep {e_i} rows={len(rows)}", flush=True)
    h.remove()

    def arr(k):
        return np.array([r[k] for r in rows], dtype=np.float64)
    epv = arr("ep").astype(np.int64)

    def boot(x, nboot=2000, seed=0):
        rng = np.random.default_rng(seed)
        ue = np.unique(epv)
        idx = {e: np.where(epv == e)[0] for e in ue}
        d = np.empty(nboot)
        for i in range(nboot):
            pick = rng.choice(ue, size=ue.size, replace=True)
            d[i] = x[np.concatenate([idx[e] for e in pick])].mean()
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"mean": round(float(x.mean()), 4), "lo": round(float(lo), 4),
                "hi": round(float(hi), 4), "n": int(x.size),
                "n_ep": int(ue.size)}

    gl = arr("gt_y6") > 1.0
    res = {
        "tool": "D-GSTR-1 downstream counterfactual (g_str patched at inference)",
        "evidence_class": "MEASURED (ours)",
        "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling)",
        "estimator": "paired episode-cluster bootstrap, 2000 resamples, seed 0",
        "n_windows": len(rows), "n_episodes": int(np.unique(epv).size),
        "head": a.head,
        "_scope": ("the FiLM that consumes g_str and every downstream weight "
                   "were trained against the BROKEN goal, so this is a LOWER "
                   "BOUND on a retrain, not an estimate of it"),
        "CONTROL_null_patch": {
            "rule": "patching g_str with ITSELF must move the plan by EXACTLY 0",
            "plan_terminal_displacement_m": boot(arr("d_null")),
            "max": round(float(arr("d_null").max()), 8),
            "PASS": bool(float(arr("d_null").max()) < 1e-6)},
        "gstr_lateral": {
            "baseline_frac_positive": round(float((arr("base_lat") > 0).mean()), 4),
            "patched_frac_positive": round(float((arr("patch_lat") > 0).mean()), 4),
            "baseline_frac_positive_on_GT_left":
                round(float((arr("base_lat")[gl] > 0).mean()), 4),
            "patched_frac_positive_on_GT_left":
                round(float((arr("patch_lat")[gl] > 0).mean()), 4)},
        "PLAN": {
            "terminal_displacement_m_patch_vs_base": boot(arr("d_patch")),
            "plan_terminal_y_base": boot(arr("base_y")),
            "plan_terminal_y_patch": boot(arr("patch_y")),
            "delta_y_on_GT_LEFT_windows": boot(
                np.where(gl, arr("patch_y") - arr("base_y"), 0.0)[gl]
                if gl.any() else np.zeros(1)),
            "n_gt_left": int(gl.sum()),
            "frac_plans_LEFT_base": round(float((arr("base_y") > 1.0).mean()), 4),
            "frac_plans_LEFT_patch": round(float((arr("patch_y") > 1.0).mean()), 4),
        },
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
