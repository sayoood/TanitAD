#!/usr/bin/env python3
"""A FULL checkpoint (what planner.py loads) from a base model and one arm of a scorer fine-tune's `scorer_states_*.pt`
(score_q_mlp / score_dec / score_head, written at every held-out read). The head's row count sets
meta["n_score_components"] (ckpt_io.config_for_checkpoint sizes the planner's head from it).
Used for an EXPLORATORY navtest look at an intermediate update; a registered stage 2 uses the run's own save_full.
    python eval/build_sft_ckpt.py --base <model_final.pt> --states <scorer_states.pt> --arm B --out <model.pt>
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refe"))
import ckpt_io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--states", required=True)
ap.add_argument("--arm", required=True, choices=("A", "B"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
sd = torch.load(a.base, map_location="cpu", weights_only=False)
st = sd["model"]
S = torch.load(a.states, map_location="cpu", weights_only=False)
n_set = 0
for mod, msd in S[a.arm].items():
    for k, v in msd.items():
        full = f"{mod}.{k}"
        assert full in st, full
        if not (mod == "score_head" and v.shape[0] != st[full].shape[0]):
            assert st[full].shape == v.shape, (full, st[full].shape, v.shape)
        st[full] = v.to(st[full].dtype)
        n_set += 1
n_out = int(st["score_head.weight"].shape[0])
meta = dict(sd.get("meta", {}) or {})
meta["n_score_components"] = n_out
meta["sft_intermediate"] = {"states": a.states, "arm": a.arm, "update": S.get("update"), "base": a.base}
ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": st, "meta": meta}, a.out)
print(f"ZZCKPT {a.out} arm {a.arm} update {S.get('update')} tensors_replaced {n_set} n_out {n_out}")
