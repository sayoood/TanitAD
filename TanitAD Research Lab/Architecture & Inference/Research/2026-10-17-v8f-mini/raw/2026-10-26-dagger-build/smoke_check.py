#!/usr/bin/env python3
"""End-to-end plumbing check of the DAgger swap on a generated bank with a REAL checkpoint (CPU): load the bank by its manifest, draw a batch of eligible windows from the bank's own join,
swap at the probability given on the command line (never a default), and show that the swapped tensors differ from the logged ones, that every control reads what it must, and that the real model's loss
moves.  Usage: smoke_check.py --bank DIR --ckpt CKPT --cache-dir D --tracks F --join-train F --p P [--n 32]"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 6))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    sys.path.insert(0, p)
import numpy as np
import torch

from tanitad.data import v8f_dagger as G
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache-dir", required=True)
ap.add_argument("--tracks", required=True)
ap.add_argument("--join-train", required=True)
ap.add_argument("--p", type=float, required=True)
ap.add_argument("--n", type=int, default=32)
a = ap.parse_args()
torch.set_num_threads(4)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = M.V8FMiniConfig(**ck["cfg"])
model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
model.load_state_dict(ck["model"])
model.train()
J = D.load_join(a.join_train)
bank = G.DaggerBank.load(a.bank, clip_names=J["clip"])
bank.check_on_policy(G.sha256_file(a.ckpt))                                            # K-B2 against the very checkpoint
store = D.ClipStore(a.cache_dir, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(a.tracks))
wins = D.train_windows(store, J, D.select_train_clips(list(bank.header["clips"].values()), 1.0), hist=cfg.hist, start=cfg.window_start)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
el = [w for w in wins if bank.candidates([w[0]], [w[1]])[0][0] > 0][:a.n]
b = D.collate([ds.item(w[0], w[1], 1, 3) for w in el])
sw = G.DaggerSwap(bank, p=a.p, seed=0)
out = sw.apply(b, 0)
res = {"hmax": cfg.hmax, "wm": cfg.wm, "bank_sha256": bank.sha256, "bank_ckpt_sha256": bank.header["ckpt_sha256"], "coverage": bank.coverage(wins), "reasons": bank.reason_counts(),
       "n_rows": sw.last["n_rows"], "n_eligible": sw.last["n_eligible"], "n_selected": sw.last["n_selected"], "p": a.p}
sel = sw.last["sel"]
res["changed_keys"] = sorted(k for k in G.REBUILT if bool((out[k] != b[k]).reshape(len(el), -1).any(1)[sel].any()))
res["unselected_rows_changed"] = sorted(k for k in G.REBUILT if bool((out[k] != b[k]).reshape(len(el), -1).any(1)[~sel].any()))
res["kept_same_object"] = all(out[k] is b[k] for k in G.KEPT + G.RECORD + ("gt", "lab", "lab_t"))


def loss(batch):
    torch.manual_seed(1)
    return float(model.loss(batch, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]["total"].detach())


l0, l0b, l1 = loss(b), loss(b), loss(out)
res["loss"] = {"logged": l0, "logged_again": l0b, "same_loss_twice_diff": l0 - l0b, "swapped": l1, "swapped_minus_logged": l1 - l0}
res["p0_is_same_object"] = G.DaggerSwap(bank, p=a.p).apply(b, 0, p=0.0) is b
print(json.dumps(res, indent=1, default=str))
