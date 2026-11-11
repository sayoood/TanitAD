"""SLOWWM READ-SIDE HOOK (PREREG_V8F_SLOWWM): where the ladder readers and the planner harness build z1 from frames, a `--wm-basis slow|pca` arm must use ITS basis. A basis model RAISES on `wm.z1_from_mean` and on `wm.z1(tokens)`
(by design: a token mean handed to it would silently run the adapter arm), so an unchanged reader fails loudly; this module is the small set of functions the read package calls instead. Nothing in `theory/ladder_read/` or
`theory/ladder_plantest/` is edited.

What the readers do today (MEASURED, source): `ladder_read/lr_lib.py` `build_eval_frames` stores `zmean` per eval frame; `encode_clip(model, zmean, act, contrast)` -> `wm.z1_from_mean(x)` -> `wm.stack.encode`; `ladder_plantest/lp_lib.py`
`clip_z1(wm, fr)` -> `wm.z1_from_mean(LR.z_in(wm, fr, ci))` (whole-clip latents; the G-TRUE goal latent is that latent at the future knot, so "the arm's own basis applied to the future frame" follows from using `u` there).
The WMPOOL read side dispatches on `LR.is_pattn`; the SLOWWM dispatch is `is_basis` below, ONE more `if` at the same two call sites:

    lr_lib.encode_clip      :  if SH.is_basis(wm): z1 = wm.z1_from_basis(x)        # x = fr["ubasis"][ci]  (SH.z_in_basis)
    lp_lib.clip_z1          :  if SH.is_basis(wm): out = SH.clip_z1_basis(wm, fr)  # or call it from prepare_arm

`prepare_arm` for a basis arm: `fr = LR.load_eval_frames(frames_s0.npz)`; `SH.attach_basis_frames(fr, basis_npz, ubasis_npz)`; `SH.require_arm_basis(selfcheck_json, basis_npz)`.
The frames carry the arm's OWN basis (`ubasis`) for the 141 eval clips; the adapter seed of the frames file is irrelevant to u (the basis is adapter-independent), but the frames file still supplies `act`, `ok`, `clips`, `T`.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import slow_basis as SB  # noqa: E402


class BasisMismatch(RuntimeError):
    pass


def is_basis(model_or_wm) -> bool:
    """True for a SLOWWM world model (``wm_basis`` slow / pca); every earlier arm has ``wm_basis == "adapter"`` or no such attribute."""
    wm = getattr(model_or_wm, "wm", model_or_wm)
    return getattr(wm, "wm_basis", "adapter") != "adapter"


def basis_kind(model_or_wm) -> str:
    wm = getattr(model_or_wm, "wm", model_or_wm)
    return getattr(wm, "wm_basis", "adapter")


def require_arm_basis(selfcheck_json: str, basis_npz: str) -> dict:
    """The basis a reader applies must be the basis the arm TRAINED on: ``levers.wm_basis_readback`` of the arm's ``selfcheck.json`` (written by the trainer from the BUILT dataset's manifest) carries kind and sha256; both must equal
    the file's. Raises :class:`BasisMismatch`. Returns the loaded basis."""
    B = SB.load_basis(basis_npz)
    sc = json.load(open(selfcheck_json, encoding="utf-8"))
    rb = (sc.get("levers") or {}).get("wm_basis_readback")
    if not rb:
        raise BasisMismatch(f"{selfcheck_json} has no levers.wm_basis_readback: not a --wm-basis arm")
    if rb.get("kind") != B["kind"] or rb.get("basis_sha256") != B["sha"]:
        raise BasisMismatch(f"the arm trained on {rb.get('kind')} {str(rb.get('basis_sha256'))[:12]}, the reader's basis is {B['kind']} {B['sha'][:12]}")
    if rb.get("cache_problems"):
        raise BasisMismatch(f"the arm's readback lists cache problems: {rb['cache_problems']}")
    return B


def ubasis_of_feats(B: dict, feats16) -> np.ndarray:
    """feats16 [T, 40, 1024] (the pool4x10 grid of one clip) -> u [T, 256] float32: the arm's basis on ``nm`` of the features (``slow_basis.nm_of`` = ``norm_mean_tokens``, the ladder's own path)."""
    return SB.apply_basis(SB.nm_of(feats16), B["mean"], B["proj"])


def build_eval_basis_frames(B: dict, fr: dict, root: str, tracks_path: str, out_path: str, split: str = "eval") -> str:
    """``ubasis`` [sum T, 256] for every clip of the frames dict ``fr`` (``lr_lib.load_eval_frames``), in its own file with the clip list, T, kind and basis sha (written whole, then renamed)."""
    src = SB.Source(root, split, tracks_path)
    us = []
    for c, T in zip(fr["clips"], fr["T"]):
        u = ubasis_of_feats(B, src.store.get(c)["feats"])
        if len(u) != T:
            raise BasisMismatch(f"{c}: {len(u)} frames, the frames file says {T}")
        us.append(u)
    part = out_path[:-4] + ".part.npz"
    np.savez(part, clips=np.array(fr["clips"]), T=np.array(fr["T"]), ubasis=np.concatenate(us), kind=np.array(B["kind"]), basis_sha=np.array(B["sha"]))
    os.replace(part, out_path)
    return out_path


def attach_basis_frames(fr: dict, B: dict, ubasis_npz: str) -> dict:
    """Adds ``fr["ubasis"]`` (a list of [T, 256] per clip) from a ``build_eval_basis_frames`` file; clips, T, kind and basis sha must equal the frames dict's and the basis'."""
    z = np.load(ubasis_npz, allow_pickle=False)
    if [str(c) for c in z["clips"]] != fr["clips"] or [int(t) for t in z["T"]] != fr["T"]:
        raise BasisMismatch(f"{ubasis_npz} does not match the frames (clips or T)")
    if str(z["kind"]) != B["kind"] or str(z["basis_sha"]) != B["sha"]:
        raise BasisMismatch(f"{ubasis_npz} was built with {str(z['kind'])} {str(z['basis_sha'])[:12]}, the basis is {B['kind']} {B['sha'][:12]}")
    cut = np.cumsum(fr["T"])[:-1]
    fr["ubasis"] = np.split(z["ubasis"], cut)
    fr["basis_kind"], fr["basis_sha"] = B["kind"], B["sha"]
    return fr


def z_in_basis(model, fr, ci):
    """The per-frame input z1 is built from for a basis model (chosen by the MODEL): ``fr["ubasis"][ci]``; frames loaded without it raise."""
    if "ubasis" not in fr:
        raise BasisMismatch("a --wm-basis model reads the basis frames: attach_basis_frames(fr, B, ubasis_npz)")
    if fr.get("basis_kind") != basis_kind(model):
        raise BasisMismatch(f"the frames carry basis {fr.get('basis_kind')!r}, the model is {basis_kind(model)!r}")
    return fr["ubasis"][ci]


def clip_z1_basis(wm, fr) -> dict:
    """``lp_lib.clip_z1`` for a basis model: {clip: [T, D]} level-1 latents of the WHOLE clip, ``wm.z1_from_basis`` (proj, enc1 when K = 3) of the clip's u."""
    import torch
    out = {}
    with torch.no_grad():
        for ci, c in enumerate(fr["clips"]):
            out[c] = wm.z1_from_basis(torch.as_tensor(z_in_basis(wm, fr, ci))[None])[0].float()
    return out


def encode_clip_basis(model, fr, ci, act, contrast=None):
    """``lr_lib.encode_clip`` for a basis model: one clip (u [T, 256], act [T, 2] raw) -> (z list per level [1, T_l, D], planning actions per level)."""
    import torch
    wm = model.wm
    a = torch.as_tensor(act).clone()
    if contrast is not None:
        ch, dv = contrast
        a[..., ch] += dv
    with torch.no_grad():
        z1 = wm.z1_from_basis(torch.as_tensor(z_in_basis(model, fr, ci))[None])
        a_sc = wm.scale_actions(a[None])[:, :-1]
        enc = wm.stack.encode(z1, a_sc)
    return enc.z, [a_sc] + [enc.act[l] for l in (1, 2)]


def top_mu_columns(B: dict, n: int = 64) -> np.ndarray:
    """Column indices of SLOW's top-``n`` mu directions (the H-PRH-PLAN beside read: the planner's latent cost restricted to them). Only for kind ``slow``; the basis columns are stored in mu order, so this is ``arange(n)``."""
    if B["kind"] != "slow":
        raise BasisMismatch("top-mu directions are defined for the slow basis only")
    return np.arange(n)
