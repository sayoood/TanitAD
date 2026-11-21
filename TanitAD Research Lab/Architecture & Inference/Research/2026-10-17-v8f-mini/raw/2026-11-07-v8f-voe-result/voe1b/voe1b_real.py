"""PREREG_V8F_VOE1B: the real-model side (VOE1's voe1_real.py, copied; needs WMACT_CODE_ROOT = the pinned tip export; see HANDOFF_VOE1B.md).
Models, the 16 untrained references, and the per-clip frame bank. No surprise statistic is computed here.
Delta from VOE1: the six VOE1 tags are replaced by M2b_s0 / M2b_s1 and the untrained family UT{chassis}_{seed}: for each of the two trained chassis (adapter, pos) and each
torch seed 100..115, a fresh ``WMAux(cfg)`` (every wm.* parameter re-initialised, as VOE1's RI)."""
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

DECIDING = ("M2b_s0", "M2b_s1")
UT_SEEDS = tuple(range(100, 116))                 # torch.manual_seed of the 16 untrained world models (card: 100 to 115)
BANK_PATH = os.environ.get("VOE1B_BANK", "F:/TanitAD-scratch/voe1b_bank/bank_voe1b.pt")


def ut_tag(chassis: int, seed: int) -> str:
    return f"UT{chassis}_{seed}"


def tags_for(n_ut: int = len(UT_SEEDS)) -> tuple:
    """Deciding tags, then the untrained tags of chassis 0 then chassis 1 (seeds in order)."""
    return DECIDING + tuple(ut_tag(c, s) for c in (0, 1) for s in UT_SEEDS[:n_ut])


def chassis_of(tag: str) -> int:
    return int(tag[-1]) if tag.startswith("M2b") else int(tag[2])


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


def build_wms(trained: dict, n_ut: int = len(UT_SEEDS)) -> dict:
    """{tag: WMAux}, deterministic: the two trained wm and the untrained family (a fresh WMAux per chassis-config and seed; the chassis enters only through the tokens)."""
    out = {}
    for s in (0, 1):
        out[f"M2b_s{s}"] = trained[s][0].wm
    for c in (0, 1):
        cfg = trained[c][1]
        for sd in UT_SEEDS[:n_ut]:
            out[ut_tag(c, sd)] = make_wm(cfg, sd)
    return out


def reinit_report(trained: dict, wms: dict) -> dict:
    """Which parameters of the untrained family were re-initialised (counts, per chassis check on one seed) and the wm state hashes that let the run rebuild the same weights."""
    rep = {"ut": {}}
    for tag in wms:
        if tag.startswith("M2b"):
            continue
        c = chassis_of(tag)
        tr = dict(trained[c][0].wm.named_parameters())
        new = dict(wms[tag].named_parameters())
        names = sorted(new)
        same = [n for n in names if torch.equal(new[n], tr[n])]
        rep["ut"][tag] = {"chassis": c, "seed": int(tag.split("_")[1]), "n_params_total": int(sum(p.numel() for p in new.values())), "n_tensors": len(names),
                          "n_params_reinitialised": int(sum(new[n].numel() for n in names if n not in same)), "n_tensors_equal_to_trained": len(same),
                          "wm_state_sha256": state_sha(wms[tag].state_dict())}
    rep["chassis_kept"] = "adapter.* and pos of M2b_s{0,1} (V8FMini.adapter, V8FMini.pos): untrained family member UT{c}_{seed} reads the tokens of chassis c; the readers and every non-wm head are not used by VOE1B"
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
            out[tag].append(wm.z1(ad[chassis_of(tag)])[0])
    return {t: torch.cat(v) for t, v in out.items()}
