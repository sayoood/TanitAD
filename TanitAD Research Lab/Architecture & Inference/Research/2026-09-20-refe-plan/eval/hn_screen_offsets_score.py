#!/usr/bin/env python3
"""P6-HN-S read M3, the scoring half: score the RL-val bank's offsets with each arm's FINAL scorer, from the context cache, as a 65th candidate in the real fan.
(PREREG_P6HN_SCREEN.md section 9 M3 and section 2.4; SCREEN_NOTES.md checklist item 5.)  No training, no trunk forward, no navtest.

For every labelled scene of the RL-val bank the nine labelled trajectories (slot 0 = the anchor, slots 1-8 = the offsets) are each scored AGAINST THE 64 FAN ROWS AND THEMSELVES, by the SAME masked second
call the training term uses (`hn_bank.hn_scores`: fan rows see the fan only; a bank row sees the fan and itself and nothing else), so nine slots are nine independent '65th candidate' scores of the one real set.
The context is read from the `ctx_cache` (refused if it belongs to another trunk); the fan is the BANKED frozen fan of RL-S1 (the sets the checkpoints' lookup scored), supplied by the caller's
`fan_provider` -- the cache's own re-emitted fan is only within the registered tolerance of it, so it is a fallback the caller must name, never the default.

OUTPUT (`write_scores9`): an npz {log, token, step, rank, ckpt_step (str / int [N9]), scores9 float64 [N9, 9, C] LOGITS, status int8 [N9], dac9 float64 [N9, 9] (the bank's DAC labels), space = 'logit'}.  The read
(`hn_screen_read.py --bank9 ... --scores9 name=...`) takes `dac9` and `status` from any one of them (they must agree: it checks) and `scores9` per arm.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if os.path.join(PKG, "refe") not in sys.path:
    sys.path.insert(0, os.path.join(PKG, "refe"))
import hn_bank as HB  # noqa: E402  -- the term's own masked call
import ctx_cache as CCH  # noqa: E402

TOOL = "hn_screen_offsets_score/1"


def load_scorer_state(scorer, state):
    """strict load of an arm's saved scorer tensors (score_q_mlp / score_dec / score_head only); a missing or extra key is an error, never a silently half-loaded scorer"""
    keep = {k: v for k, v in state.items() if k.split(".")[0] in CCH.SCORER}
    res = scorer.load_state_dict(keep, strict=True)
    return res


@torch.no_grad()
def score_bank_offsets(scorer, bank, cache, fan_provider, *, batch=8, device="cpu", expect_trunk_fingerprint=None, autocast=True):
    """-> (keys [N9], scores9 float32 [N9, 9, C]) for the bank's LABELLED scenes (status 0), in bank order.  `fan_provider(keys) -> float tensor [B, 64, T, 3]`."""
    if expect_trunk_fingerprint is not None and cache.man["trunk_fingerprint"] != expect_trunk_fingerprint:
        raise CCH.StaleCache("the context cache belongs to another trunk than the checkpoint being scored")
    scorer = scorer.to(device).eval()
    idx = [i for i in range(len(bank)) if bank.status[i] == HB.STATUS_LABELLED]
    keys = [tuple(bank.keys[i][:4]) for i in idx]
    miss = [k for k in keys if k not in cache.index]
    if miss:
        raise KeyError(f"{len(miss)} bank scenes are not in the context cache (first {miss[0]})")
    out = []
    for s in range(0, len(idx), batch):
        ks = keys[s:s + batch]
        ii = idx[s:s + batch]
        sctx = cache.get_ctx(ks).to(device)
        fan = fan_provider(ks).to(device).float()
        t9 = torch.from_numpy(bank.traj9[ii]).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):          # the trainer's own dtype
            lg = HB.hn_scores(scorer, fan, t9, torch.ones(t9.shape[:2], dtype=torch.bool, device=device), sctx.to(torch.bfloat16 if autocast else torch.float32))
        out.append(lg.float().cpu())
    return keys, torch.cat(out).numpy() if out else np.zeros((0, 9, 7), np.float32)


def write_scores9(path, bank, keys, scores9):
    kk = {k: i for i, k in enumerate(tuple(x[:4]) for x in bank.keys)}
    rows = [kk[k] for k in keys]
    np.savez(path, log=np.asarray([k[0] for k in keys]), token=np.asarray([k[1] for k in keys]), step=np.asarray([k[2] for k in keys], np.int64),
             rank=np.asarray([k[3] for k in keys], np.int64), ckpt_step=np.asarray([bank.keys[i][4] for i in rows], np.int64), scores9=np.asarray(scores9, np.float64),
             status=bank.status[rows].astype(np.int8), dac9=bank.lab9[rows, :, 1].astype(np.float64), space=np.asarray("logit"))


def bank9_from(path):
    z = np.load(path, allow_pickle=False)
    return {"dac9": z["dac9"], "status": z["status"], "logs": [str(x) for x in z["log"]]}, z["scores9"]
