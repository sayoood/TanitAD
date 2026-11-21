#!/usr/bin/env python3
"""H-ADAM section 2 (+ amendment-1 diagnostics D1-D3): is the substituted sctx admissible?

  V = median over the 7 heads of the mean-over-scenes within-scene Pearson r between (this script's base-scorer logits on the CACHED visual_ctx)
      and (the banked scores_step0_base.npz logits, computed under the real encoder), on the RL-val frames present in the trunk cache.

    python face_validity.py --out face_validity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adam_common as AC  # noqa: E402

HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "LANE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args(argv)
    t0 = time.time()
    SF = AC.load_trainer()
    a, argv_r0 = AC.r0_args(SF)
    holder, cfg = AC.build_scorer(SF, a)
    res = {"tool": "face_validity/1", "trainer_blob_sha256": AC.sha256_file(SF.__file__), "base_ckpt_sha256_expected": "8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d",
           "C6_strict_load": "PASS (load_state_dict(strict=True) of all three scorer modules did not raise)"}
    # ---- RL-val overlap: logits vs banked
    pv = AC.Pool(AC.prepare_pool(SF, a, cfg, "val")).load_visual_all()
    res["val_pool"] = pv.info
    tr = AC.Trainer(SF, a, holder, pv)
    z = np.load("C:/Users/Admin/rls1b_dl/r0/final/scores_step0_base.npz", allow_pickle=False)
    idx = {(str(z["log"][i]), str(z["token"][i]), int(z["step"][i]), int(z["rank"][i])): i for i in range(len(z["token"]))}
    banked = np.stack([z["scores"][idx[k]] for k in pv.keys]).astype(np.float64)          # [n, 64, 7]
    mine = []
    for lo in range(0, len(pv.keys), 8):
        ids = list(range(lo, min(lo + 8, len(pv.keys))))
        cxy, ctg, sctx, _ = pv.micro(ids)
        with torch.no_grad():
            mine.append(tr.base(cxy, sctx).float().numpy())
    mine = np.concatenate(mine).astype(np.float64)
    per_head = {}
    for j, h in enumerate(HEADS):
        rs = []
        for n in range(mine.shape[0]):
            x, y = mine[n, :, j], banked[n, :, j]
            if x.std() > 0 and y.std() > 0:
                rs.append(np.corrcoef(x, y)[0, 1])
        per_head[h] = {"mean_within_scene_r": float(np.mean(rs)), "median_within_scene_r": float(np.median(rs)), "scenes": len(rs),
                       "mean_abs_logit_diff": float(np.abs(mine[..., j] - banked[..., j]).mean()),
                       "mean_logit_mine": float(mine[..., j].mean()), "mean_logit_banked": float(banked[..., j].mean()),
                       "std_logit_mine": float(mine[..., j].std()), "std_logit_banked": float(banked[..., j].std())}
    V = float(np.median([per_head[h]["mean_within_scene_r"] for h in HEADS]))
    res["V_median_over_heads_of_mean_within_scene_r"] = V
    res["status"] = "ADMISSIBLE" if V >= 0.90 else ("DEGRADED" if V >= 0.60 else "INADMISSIBLE")
    res["per_head"] = per_head
    # the selection rule's own score: does the pick agree?  (diagnostic)
    def sel(l):
        p = 1 / (1 + np.exp(-l))
        return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0 * p[..., 6]
    res["pick_agreement_with_banked"] = float((sel(mine).argmax(-1) == sel(banked).argmax(-1)).mean())
    res["n_val_scenes"] = int(mine.shape[0])
    # ---- RL-train pool: D1 D2 D3
    pt = AC.Pool(AC.prepare_pool(SF, a, cfg, "train")).load_visual_all()
    res["train_pool"] = pt.info
    trt = AC.Trainer(SF, a, holder, pt)
    bces, Js, kls = [], [], []
    rng = np.random.default_rng(0)
    for lo in range(0, len(pt.keys) - 7, 8):
        ids = list(range(lo, lo + 8))
        cxy, ctg, sctx, _ = pt.micro(ids)
        gB, gE, info = trt.micro_grads(ids, cxy, ctg, sctx)
        bces.append(info["bce"]); Js.append(info["J"]); kls.append(info["kl"])
    res["D1_mean_BCE_base_on_pool"] = float(np.mean(bces)); res["D1_logged_R0_bceA_u25"] = 0.5741515401005745
    res["D2_mean_J_base_on_pool"] = float(np.mean(Js)); res["D2_logged_R0_epoJ_u25"] = 0.8171236744523048
    res["D2_KL_at_init"] = float(np.mean(kls))
    tg = pt.ctg
    ttc_fail = tg[..., 3] < 1.0 - 1e-6
    res["D3_pool"] = {"TTC_fail_share_of_proposals": float(ttc_fail.mean()), "scenes_with_TTC_fail": float(ttc_fail.any(-1).mean()), "n": int(len(tg))}
    try:
        att = json.load(open(os.path.join(HERE, "..", "2026-10-14-rls1b-attribution", "attribution.json")))
        res["D3_RL_train_partition"] = att["rl_train_base_rates"]
    except Exception as e:  # noqa: BLE001
        res["D3_RL_train_partition"] = f"unavailable: {type(e).__name__}"
    res["closure_hashes"] = trt.closure_hashes
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a_.out, "w", encoding="utf-8"), indent=1)
    print("V", round(V, 4), res["status"], "| D1", round(res["D1_mean_BCE_base_on_pool"], 4), "vs", res["D1_logged_R0_bceA_u25"], "| D2", round(res["D2_mean_J_base_on_pool"], 4), "vs", res["D2_logged_R0_epoJ_u25"])
    print({h: round(v["mean_within_scene_r"], 3) for h, v in per_head.items()}, "pick agree", res["pick_agreement_with_banked"], "n_val", res["n_val_scenes"], "n_train", len(pt.keys))
    print("D3", res["D3_pool"], res["D3_RL_train_partition"])
    print("ZZFACE_OK", a_.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
