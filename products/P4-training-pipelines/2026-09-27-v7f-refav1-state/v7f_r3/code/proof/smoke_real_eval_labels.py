"""CPU smoke on the REAL v7.2 EVAL labels: real blob -> real join -> R3 term -> S-T optimiser steps.

What is REAL: the label blob (canonical eval, md5 aa12c948f062181c3297265b51526ec5, 147 records),
`load_v72_labels_for_trainer`, the v7.2 window join + the R3 targets, `tac_label_policy`,
`v6_loss_step`, the S-T freeze, AdamW. What is NOT: pixels and the model geometry -- a tiny V6Stack on
random frames (no corpus, no GPU on this box). ⇒ the loss can only fall towards the label MARGINALS;
this proves plumbing + optimisation on real label records, NOTHING about driving. No number here is
quotable as a capability.

usage: PYTHONPATH=<tree>/stack python smoke_real_eval_labels.py <eval blob> <out.json>
"""
from __future__ import annotations

import json
import os
import random
import sys
import types

TREE = os.environ.get("PROOF_TREE", "C:/Users/Admin/v7f_r3")
sys.path.insert(0, os.path.join(TREE, "stack"))
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))

import torch  # noqa: E402

torch.set_num_threads(2)

import train_v6_staged as T  # noqa: E402
from s2_labels import stable_episode_id  # noqa: E402
from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.data import v7_labels as V7L  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack, apply_stage_freeze  # noqa: E402

W, DT, N_STACK, N_T = 4, 0.1, 3, 190     # 190 provider windows per clip (~199-frame episodes)
STEPS, BATCH = 30, 16


def stack() -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=W, horizons=(1,),
                                  action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, d_plan_feat=16, emission_hidden=16, n_candidates=3, aux_hidden=16,
        sigreg_slices=8, goal_multilabel=True))


def join(s, blob, negatives):
    ls = T.load_v72_labels_for_trainer(blob, stack=s)          # canonical eval: md5-pinned
    labels = list(ls.v7_by_stable.values())
    eps = [types.SimpleNamespace(episode_id=stable_episode_id(x.clip_id),
                                 frames=torch.zeros(1, 3 * N_STACK, 2, 2)) for x in labels]
    index = [(e_i, t) for e_i in range(len(eps)) for t in range(N_T)]
    sup = ls.supervision(eps, window=W, dt=DT, index=index)
    rep = T.build_tac_label_targets(types.SimpleNamespace(tac_goal_negatives=negatives,
                                                          cot_negative_sidecar=None),
                                    ls, sup, s)
    return ls, sup, index, rep


def train(blob, negatives, w_on: float, seed=0) -> dict:
    s = stack()
    ls, sup, index, rep = join(s, blob, negatives)
    in_band = [i for i in range(len(index))
               if bool(sup.batch([i])["tac_valid"][0])]
    apply_stage_freeze(s, "S-T")
    params = [p for p in s.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    rng = random.Random(seed)
    rows = []
    for step in range(1, STEPS + 1):
        idx = rng.sample(in_band, BATCH)
        b = T.synthetic_train_batch(s, batch=BATCH, k=12, seed=1000 + step)
        b["gt_wp"] = torch.randn(BATCH, 10, 2, generator=torch.Generator().manual_seed(step))
        b |= sup.batch(idx)
        torch.manual_seed(step)
        L = T.v6_loss_step(s, b, stage="S-T",
                           weights=T.V6LossWeights(w_tac_label_all=w_on), o1_k=10, o5_k=12,
                           generator=torch.Generator().manual_seed(step))
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        lg = L["log"]
        rows.append({"step": step, "loss": lg["loss"], "terms": lg["terms"], "gnorm": round(gn, 4),
                     **{k: lg.get(k) for k in ("tac_label_loss", "tac_lat_ce", "tac_lon_ce",
                                               "tac_goal_bce", "tac_speedband_l1_ms",
                                               "tac_goal_n_supervised", "tac_goal_n_pos",
                                               "t1_latent", "seam_op")}})
    return {"negatives": negatives, "w_tac_label_all": w_on, "rows": rows,
            "n_in_band_windows": len(in_band), "n_windows": len(index), "policy": rep}


def reach_on_real_records(blob, negatives) -> dict:
    """One batch holding ONE in-band window of EVERY clip: which goal rows get a gradient?"""
    s = stack()
    ls, sup, index, rep = join(s, blob, negatives)
    first = {}
    for i, (e_i, _t) in enumerate(index):
        if e_i not in first and bool(sup.batch([i])["tac_valid"][0]):
            first[e_i] = i
    idx = sorted(first.values())
    b = T.synthetic_train_batch(s, batch=len(idx), k=12, seed=7)
    b["gt_wp"] = torch.randn(len(idx), 10, 2, generator=torch.Generator().manual_seed(7))
    keys = sup.batch(idx)
    b |= keys
    torch.manual_seed(3)
    L = T.v6_loss_step(s, b, stage="S-T", weights=T.V6LossWeights(w_tac_label_all=1.0),
                       o1_k=10, o5_k=12, generator=torch.Generator().manual_seed(3))
    g = torch.autograd.grad(L["tac_label_all"], s.goal_head_tac.type_head.weight,
                            retain_graph=True)[0]
    ga = torch.autograd.grad(L["tac_label_all"], s.goal_head_tac.arg_head.weight)[0]
    toks = V7L.TAC_GOAL_TOKENS
    y, w, cm = keys["tac_goal_y"], keys["tac_goal_w"], keys["tac_goal_class_mask"]
    per = {}
    for i, t in enumerate(toks):
        sup_cells = (w[:, i] > 0) & (cm[i] > 0)
        per[t] = {"n_pos_supervised": int((sup_cells & (y[:, i] > 0.5)).sum()),
                  "n_neg_supervised": int((sup_cells & (y[:, i] <= 0.5)).sum()),
                  "class_mask": int(cm[i]),
                  "grad_row_abs_sum": float(g[i].abs().sum()),
                  "reached": bool(float(g[i].abs().sum()) > 0.0)}
    return {"negatives": negatives, "n_clips_in_batch": len(idx),
            "tac_label_all": float(L["tac_label_all"].detach()),
            "log": {k: v for k, v in L["log"].items() if k.startswith("tac_")},
            "speed_band_arg_rows_reached": [i for i in range(ga.shape[0])
                                            if float(ga[i].abs().sum()) > 0],
            "per_token": per,
            "n_tokens_reached": sum(1 for d in per.values() if d["reached"]),
            "masked": rep["masked_why"]}


def main() -> None:
    blob, out = sys.argv[1], sys.argv[2]
    res = {"_read": __doc__.split("usage")[0].strip(), "blob": blob,
           "reach_measured": reach_on_real_records(blob, "measured"),
           "reach_all": reach_on_real_records(blob, "all"),
           "train_on_measured": train(blob, "measured", 1.0),
           "train_off": train(blob, "measured", 0.0)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    on, off = res["train_on_measured"]["rows"], res["train_off"]["rows"]
    print(f"in-band tactical windows: {res['train_on_measured']['n_in_band_windows']}/"
          f"{res['train_on_measured']['n_windows']}")
    for nm in ("reach_measured", "reach_all"):
        r = res[nm]
        print(f"[{nm}] clips {r['n_clips_in_batch']} tokens reached {r['n_tokens_reached']}/22 "
              f"masked {sorted(r['masked'])} speed-band arg rows {r['speed_band_arg_rows_reached']}")
        for t in ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN",
                  "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT"):
            print(f"    {t:28s} {r['per_token'][t]}")
    print("ON  step 1 :", {k: on[0][k] for k in ("terms", "tac_lat_ce", "tac_lon_ce",
                                                  "tac_goal_bce", "tac_speedband_l1_ms")})
    print(f"ON  step {STEPS}:", {k: on[-1][k] for k in ("tac_lat_ce", "tac_lon_ce",
                                                         "tac_goal_bce", "tac_speedband_l1_ms")})
    print("OFF step 1 :", {k: off[0][k] for k in ("terms", "loss", "tac_label_loss")})
    print(f"OFF step {STEPS}:", {k: off[-1][k] for k in ("terms", "loss")})


if __name__ == "__main__":
    main()
