"""PREREG_V8F_VOE1: the real-model side (needs WMACT_CODE_ROOT = the pinned tip export; see HANDOFF_VOE1.md).
Models, the random-init controls, and the per-clip frame bank. No surprise statistic is computed here."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wmact"))
import wmact_lib as W  # noqa: E402  (puts the code export on sys.path, asserts tanitad.__file__ lies under it)
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

TAGS = ("M2b_s0", "M2b_s1", "RI_s0", "RI_s1", "RIs_s0", "RIs_s1")
DECIDING = ("M2b_s0", "M2b_s1")
CONTROL = ("RI_s0", "RI_s1")                    # control 2: fresh WMAux (every wm.* parameter re-initialised)
BESIDE = ("RIs_s0", "RIs_s1")                   # beside, not gating: fresh stack, trained wm.proj
CHASSIS = {"M2b_s0": 0, "M2b_s1": 1, "RI_s0": 0, "RI_s1": 1, "RIs_s0": 0, "RIs_s1": 1}


def state_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_trained():
    """{0: (model, cfg), 1: (model, cfg)}: the M2b HACC checkpoints (trained chassis, trained wm.*)."""
    out = {}
    for s, (tag, path) in enumerate(W.CKPT.items()):
        m, cfg, _ = W.load_model(path)
        out[s] = (m, cfg)
    return out


def make_wm(cfg, seed: int, trained_proj_from=None):
    """A fresh ``WMAux(cfg)`` built under ``torch.manual_seed(seed)`` (PyTorch default init of every layer; the HJEPAStack is built before ``proj``).
    ``trained_proj_from`` = a trained WMAux: its ``proj`` is copied in (the BESIDE variant: only the stack is random)."""
    torch.manual_seed(seed)
    wm = M.WMAux(cfg)
    if trained_proj_from is not None:
        wm.proj.load_state_dict(trained_proj_from.proj.state_dict())
    return wm.eval()


def build_wms(trained: dict) -> dict:
    """{tag: WMAux} for the six tags, deterministic."""
    out = {}
    for tag in TAGS:
        s = CHASSIS[tag]
        m, cfg = trained[s]
        if tag.startswith("M2b"):
            out[tag] = m.wm
        elif tag.startswith("RIs"):
            out[tag] = make_wm(cfg, s, trained_proj_from=m.wm)
        else:
            out[tag] = make_wm(cfg, s)
    return out


def reinit_report(trained: dict, wms: dict) -> dict:
    """Which parameters were re-initialised, by name and count, and the hashes that let the run re-build the same weights."""
    rep = {}
    for tag in CONTROL + BESIDE:
        s = CHASSIS[tag]
        tr = dict(trained[s][0].wm.named_parameters())
        new = dict(wms[tag].named_parameters())
        names = sorted(new)
        same = [n for n in names if torch.equal(new[n], tr[n])]
        rep[tag] = {"seed": s, "n_params_total": int(sum(p.numel() for p in new.values())), "n_tensors": len(names),
                    "n_params_reinitialised": int(sum(new[n].numel() for n in names if n not in same)),
                    "tensors_equal_to_trained": same, "groups_reinitialised": sorted({n.split(".")[0] + "." + n.split(".")[1] for n in names if n not in same}),
                    "wm_state_sha256": state_sha(wms[tag].state_dict())}
    rep["chassis_kept"] = "adapter.* and pos of M2b_s{0,1} (V8FMini.adapter, V8FMini.pos); the readers and every non-wm head are not used by VOE1"
    return rep


@torch.no_grad()
def clip_z1(models_for_z1: dict, chassis: dict, feats: np.ndarray, chunk: int = 64) -> dict:
    """z1 [T, 128] per tag for one clip: ``wm.z1(adapter(feats) + pos)`` with the tag's chassis (adapter, pos) and the tag's WMAux."""
    out = {tag: [] for tag in models_for_z1}
    for c0 in range(0, feats.shape[0], chunk):
        f = torch.from_numpy(feats[c0:c0 + chunk]).float()
        ad = {}
        for s, m in chassis.items():
            ad[s] = m.adapter(f)[None] + m.pos[None, None]          # [1, c, n_tok, d]: the shape collect_latents.py gives WMAux.z1
        for tag, wm in models_for_z1.items():
            out[tag].append(wm.z1(ad[CHASSIS[tag]])[0])
    return {t: torch.cat(v) for t, v in out.items()}
