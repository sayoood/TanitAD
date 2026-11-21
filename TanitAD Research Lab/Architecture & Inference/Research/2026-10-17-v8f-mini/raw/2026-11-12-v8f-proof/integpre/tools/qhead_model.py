#!/usr/bin/env python
"""qhead_model.py -- card PREREG_V8F_PROOF_ADD42 (QHEAD): the quantile head, shared by the fit (qhead_fit.py), the rule tool (v8f_t1_eval_rule_qhead.py) and the controls.

QHead has the shape of the gap branch of the banked LEADSUP head (LeadSupHead of the tree, stack/tanitad/models/v8f_mini.py lines 814-830): a per-frame linear D -> 64, the `hist`
embeddings concatenated, Linear(hist * 64, 128), GELU, Linear(128, 1). Its one output is the gap divided by 50 m (the banked scale). Tier T0 for the fit, T1-kin when it is read in a rollout.
"""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

GAP_SCALE_M = 50.0
FRAME_DIM = 64
HIDDEN = 128
HIST = 5
Q_LEVEL = 0.20
TENSOR_KEYS = ("frame_w", "frame_b", "mlp0_w", "mlp0_b", "mlp2_w", "mlp2_b")


class QHead(nn.Module):
    def __init__(self, state_dim, hist=HIST, frame_dim=FRAME_DIM, hidden=HIDDEN):
        super().__init__()
        self.hist = int(hist)
        self.frame = nn.Linear(state_dim, frame_dim)
        self.mlp = nn.Sequential(nn.Linear(hist * frame_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z):
        """z [B, F, D] -> [B, F - hist + 1, 1] (gap / 50 m at the end frames), as the banked head's windows."""
        F_ = z.shape[1]
        if F_ < self.hist:
            raise ValueError("quantile head needs at least %d frames, got %d" % (self.hist, F_))
        e = self.frame(z)
        w = e.unfold(1, self.hist, 1).permute(0, 1, 3, 2)
        return self.mlp(w.flatten(2)).float()

    def tensors(self):
        sd = self.state_dict()
        m = {"frame_w": sd["frame.weight"], "frame_b": sd["frame.bias"], "mlp0_w": sd["mlp.0.weight"], "mlp0_b": sd["mlp.0.bias"], "mlp2_w": sd["mlp.2.weight"], "mlp2_b": sd["mlp.2.bias"]}
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.items()}


def sha256_file(path, blk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def save_head(path, head, extra=None):
    t = head.tensors()
    with open(path, "wb") as fh:
        np.savez(fh, **t, hist=np.int64(head.hist), q_level=np.float64(Q_LEVEL), **(extra or {}))


def load_head(path, device="cpu"):
    z = np.load(path, allow_pickle=False)
    for k in TENSOR_KEYS + ("hist",):
        if k not in z.files:
            raise SystemExit("REFUSED: head file %s lacks %s" % (path, k))
    hist = int(z["hist"])
    d = z["frame_w"].shape[1]
    if z["mlp2_w"].shape[0] != 1:
        raise SystemExit("REFUSED: head file %s has %d outputs, a quantile head has 1" % (path, z["mlp2_w"].shape[0]))
    h = QHead(d, hist, frame_dim=z["frame_w"].shape[0], hidden=z["mlp0_w"].shape[0])
    sd = {"frame.weight": z["frame_w"], "frame.bias": z["frame_b"], "mlp.0.weight": z["mlp0_w"], "mlp.0.bias": z["mlp0_b"], "mlp.2.weight": z["mlp2_w"], "mlp.2.bias": z["mlp2_b"]}
    h.load_state_dict({k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in sd.items()})
    return h.to(device).eval()


def functional_forward(tensors, z):
    """An independent forward of a head given as six tensors (numpy), z [B, hist, D] -> [B, n_out] (end frame only); used to recompute the banked head (3 outputs) and the quantile head (1 output)."""
    t = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in tensors.items() if k in TENSOR_KEYS}
    x = torch.from_numpy(np.asarray(z, np.float32))
    e = x @ t["frame_w"].T + t["frame_b"]
    e = e.reshape(e.shape[0], -1)
    h = e @ t["mlp0_w"].T + t["mlp0_b"]
    h = 0.5 * h * (1.0 + torch.special.erf(h / 2.0 ** 0.5))
    return (h @ t["mlp2_w"].T + t["mlp2_b"]).numpy()


def used_gap(g_head, q_head):
    """g_used = min(g_head, q_head); a gap that is not finite stays as it is; a quantile that is not finite leaves the head gap."""
    g = np.asarray(g_head)
    q = np.asarray(q_head)
    out = np.where(np.isfinite(q), np.minimum(g, q), g)
    return out.astype(g.dtype)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def is_eval_path(p):
    """True for any path of the evaluation split: a basename or a parent folder name that carries `eval`."""
    parts = os.path.normpath(str(p)).replace("\\", "/").split("/")
    return any("eval" in s.lower() for s in parts[-2:])
