"""refcv9 C10 -- photometric augmentation for the refcv9 full run (``--r9-photometric``; REFCV9_DESIGN 2.5).

WHAT: brightness / contrast / saturation +-0.2, hue +-0.02, a light 3x3 blur, clamp [0, 1] -- ONE parameter draw PER SAMPLE, shared by every window row, every
stacked frame and the sample's own ``future_frames`` ("applied to the whole sample"), and by NOTHING else: no label, pose, nav, goal or mask key is touched, so
every loss keeps its target (label-invariant).  The horizontal flip is NOT built: E0a (e4b028d) found the train-vs-held-out AP gap inside the bootstrap
half-width, so A1 is not run and only photometric enters E3 (disclosed in the stamp).

FRAME CHANNEL LAYOUT (measured, not assumed).  A batch holds ``frames [B, rows, C, H, W]`` and ``future_frames [B, F, C, H, W]`` with ``C = 3 * K``: the D-015
STACK of K consecutive frames of ONE camera (``camera_front_wide_120fov``), each an RGB triplet, OLDEST frame first, the current frame in the LAST three channels
-- ``tanitad/data/comma2k19.py:633-641`` (``stack_frames``: ``torch.cat(parts, dim=1)`` of ``[T,3,S,S]`` uint8 parts), called by ``tanitad/data/physicalai.py:737``;
K = ``encoder.in_channels // 3`` (``refc_v3_train.py`` ``trunk_frames`` stamp, ``--trunk-in-channels`` help: 9 = 3-frame stack, 3 = one frame).  Channel ``3*k + c`` is
colour ``c`` (R, G, B) of stacked frame ``k``.  Saturation and hue are therefore applied per RGB TRIPLET (the ``K`` groups of 3 channels), never across a triplet
boundary; the SAME draw covers all K groups of a sample (a stack is one moment of one camera; per-group draws would fake motion between its frames).
There is no multi-camera channel packing in this corpus; ``C % 3 != 0`` (e.g. the 1-channel toy BEV) REFUSES.

OPERATIONS (per sample, per stacked frame, in this order, each followed by a clamp to [0, 1]):
  brightness  x * fb,            fb = 1 + U(-0.2, 0.2)                 (torchvision's multiplicative convention)
  contrast    (x - m) * fc + m,  fc = 1 + U(-0.2, 0.2), m = mean BT.601 luma of THAT frame
  saturation + hue, fused into ONE 3x3 colour matrix applied in YIQ space (a LINEAR rotation, NOT the HSV hue shift of torchvision): [Y, I, Q] -> [Y, s*R(t)[I, Q]],
              s = 1 + U(-0.2, 0.2), t = 2*pi*h, h = U(-0.02, 0.02) of a full turn (max |t| = 7.2 deg).  s = 1, h = 0 is the identity to float rounding.
  blur        with probability BLUR_P a 3x3 separable Gaussian (replicate edges), sigma ~ U(BLUR_SIGMA_RANGE)
APPLY probability is P_APPLY = 1.0 (the design states no p; every sample is jittered, each factor drawn from its range).

MEMORY (Thor is unified memory; ``future_frames`` is ~2.36 GB fp32 at the live shape): IN PLACE on the tensors passed in (their ``data_ptr`` is unchanged),
chunked per SAMPLE and per stacked-frame group, so no temporary larger than ~0.7 of one group (a sample's rows x 3 channels) is ever allocated
(``tests/test_r9_c10_photometric.py`` measures the largest fresh allocation of every op).  ALIASING: ``frames_to_device`` returns the batch's own tensor when it is
already float on the target device (the CPU smoke path), so there the batch's frames are modified in place -- a second ``compute_losses_*`` call on the SAME batch
object would augment twice.  On the live path (uint8 batches ``.float()``, or a CPU -> CUDA copy) the batch is untouched.

DETERMINISM: the generator is seeded from ``(seed, step)`` through ``numpy.random.SeedSequence`` -- NO RNG state exists to checkpoint (the trainer does not restore RNG
on resume, the C6 finding), so a resumed run draws exactly what the uninterrupted run would have drawn at the same step.

REACHABILITY, one hop per line:
  argv ``--r9-photometric`` (``refc_v3_train.py`` build_parser, g8 group, store_true) -> ``args.r9_photometric`` -> ``attach`` (train() and the eval loader) ->
  ``model._r9_photometric`` (:class:`PhotometricState`) -> ``set_step`` (the training loop, every step) -> ``maybe_apply_`` (``compute_losses_v3`` and
  ``compute_losses_perc_rig``, right after ``frames_to_device``, ``model.training`` only) -> ``apply_`` -> the trunk input and the LAW target -> the loss -> ``.backward()``.
  Never reached by: the in-run eval / dump / BN recalibration / ``r9_e1_dump`` (``model.eval()``, or ``frames_to_device`` called without ``maybe_apply_``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["FLAG", "DEST", "MODEL_ATTR", "STAMP_KEY", "BRIGHTNESS", "CONTRAST", "SATURATION", "HUE", "BLUR_P", "BLUR_KERNEL", "BLUR_SIGMA_RANGE", "P_APPLY",
           "PhotometricState", "is_on", "attach", "set_step", "maybe_apply_", "apply_", "draw_params", "color_matrix", "metrics", "stamp",
           "dvb_check", "register_dvb_levers"]

FLAG = "--r9-photometric"
DEST = "r9_photometric"
MODEL_ATTR = "_r9_photometric"
STAMP_KEY = "r9_photometric"

#: +- half-widths of the uniform draws (REFCV9_DESIGN 2.5)
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
#: hue, as a fraction of a full turn of the YIQ chroma plane (0.02 -> 7.2 degrees)
HUE = 0.02
#: probability a sample is blurred, the (odd) kernel size and the sigma range (pixels) of the separable Gaussian
BLUR_P = 0.2
BLUR_KERNEL = 3
BLUR_SIGMA_RANGE = (0.3, 0.8)
#: probability a sample is jittered at all (1.0: the design states none)
P_APPLY = 1.0

#: BT.601 luma, and the YIQ matrix the colour rotation is built in
_LUMA = (0.299, 0.587, 0.114)
_YIQ = ((0.299, 0.587, 0.114), (0.595716, -0.274453, -0.321263), (0.211456, -0.522591, 0.311135))
#: columns of the per-sample draw
_N_DRAW = 7


@dataclass
class PhotometricState:
    """What ``model._r9_photometric`` holds: the seed (argv ``--seed``), the step the loop last set, and the last call's fraction of jittered samples."""
    seed: int
    step: int = 0
    frac: float = 0.0


def is_on(args) -> bool:
    return bool(getattr(args, DEST, False))


def attach(model, args) -> None:
    """``model._r9_photometric`` <- a fresh state when the flag is on (train() AND the eval loader, so G-DVB reads the same build in both; the eval forward never augments).
    OFF: nothing is set (the attribute is absent, ``maybe_apply_`` returns at once, the tip's behaviour bit for bit)."""
    if is_on(args):
        if int(getattr(args, "seed", 0)) < 0:
            raise SystemExit("[r9-photometric] --seed must be >= 0 (the draw is seeded from (seed, step))")
        setattr(model, MODEL_ATTR, PhotometricState(seed=int(getattr(args, "seed", 0))))


def set_step(model, step: int) -> None:
    """The training loop calls this once per step BEFORE the loss; a no-op when the flag is off."""
    st = getattr(model, MODEL_ATTR, None)
    if st is not None:
        st.step = int(step)


def maybe_apply_(model, frames: torch.Tensor, fut_frames: torch.Tensor | None = None) -> None:
    """THE call site helper: augment ``frames`` (and ``fut_frames``) IN PLACE iff the flag is attached AND the model is in training mode.  Eval / dump / recalibration:
    a no-op (``model.eval()``).  Records the fraction of jittered samples for the ``aug/photometric_frac`` log key."""
    st = getattr(model, MODEL_ATTR, None)
    if st is None or not model.training:
        return
    st.frac = apply_(frames, fut_frames, seed=st.seed, step=st.step)


def metrics(model) -> dict:
    """``{"aug/photometric_frac": ...}`` -- the fraction of the LAST training batch's samples that were jittered; ``{}`` when the flag is off."""
    st = getattr(model, MODEL_ATTR, None)
    return {} if st is None else {"aug/photometric_frac": float(st.frac)}


def _gen_seed(seed: int, step: int) -> int:
    if int(seed) < 0 or int(step) < 0:
        raise ValueError(f"[r9-photometric] (seed, step) must be non-negative, got ({seed}, {step})")
    ss = np.random.SeedSequence([int(seed), int(step)])
    return int(ss.generate_state(1, dtype=np.uint64)[0]) & 0x7FFFFFFFFFFFFFFF


def draw_params(batch_size: int, seed: int, step: int) -> dict:
    """The per-SAMPLE parameter draw: a pure function of ``(seed, step, batch_size)`` -- CPU float64, no global RNG, nothing to checkpoint.  -> lists of length B."""
    g = torch.Generator(device="cpu")
    g.manual_seed(_gen_seed(seed, step))
    u = torch.rand(int(batch_size), _N_DRAW, generator=g, dtype=torch.float64)
    sym = lambda col, half: (half * (2.0 * u[:, col] - 1.0)).tolist()      # noqa: E731  U(-half, half)
    lo, hi = BLUR_SIGMA_RANGE
    return {"fb": [1.0 + v for v in sym(0, BRIGHTNESS)],
            "fc": [1.0 + v for v in sym(1, CONTRAST)],
            "fs": [1.0 + v for v in sym(2, SATURATION)],
            "hue": sym(3, HUE),
            "blur": (u[:, 4] < BLUR_P).tolist(),
            "sigma": (lo + (hi - lo) * u[:, 5]).tolist(),
            "apply": (u[:, 6] < P_APPLY).tolist()}


def color_matrix(sat: float, hue_turns: float) -> torch.Tensor:
    """The 3x3 RGB -> RGB matrix of saturation ``sat`` and a YIQ chroma rotation by ``2*pi*hue_turns``: ``T^-1 diag(1, sat*R(t)) T`` (float64)."""
    t = torch.tensor(_YIQ, dtype=torch.float64)
    ang = 2.0 * math.pi * float(hue_turns)
    c, s = math.cos(ang) * float(sat), math.sin(ang) * float(sat)
    mid = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    return torch.linalg.inv(t) @ mid @ t


def _gauss_kernel(sigma: float) -> torch.Tensor:
    """The 3x3 (BLUR_KERNEL^2) outer-product Gaussian, normalised to sum 1 (float64)."""
    half = BLUR_KERNEL // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float64)
    w1 = torch.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    w1 = w1 / w1.sum()
    return w1[:, None] * w1[None, :]


def _jitter_group_(x: torch.Tensor, fb: float, fc: float, m3: torch.Tensor, kern: torch.Tensor | None) -> None:
    """One stacked frame's RGB triplet of ONE sample: ``x [rows, 3, H, W]`` (a VIEW; every op below is in place or allocates at most two planes)."""
    x.mul_(fb).clamp_(0.0, 1.0)
    m = (x[:, 0].mean(dim=(-2, -1)) * _LUMA[0] + x[:, 1].mean(dim=(-2, -1)) * _LUMA[1] + x[:, 2].mean(dim=(-2, -1)) * _LUMA[2]).view(-1, 1, 1, 1)
    x.sub_(m).mul_(fc).add_(m).clamp_(0.0, 1.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    v = [[float(m3[i, j]) for j in range(3)] for i in range(3)]
    nr = torch.mul(r, v[0][0]).add_(g, alpha=v[0][1]).add_(b, alpha=v[0][2])
    ng = torch.mul(r, v[1][0]).add_(g, alpha=v[1][1]).add_(b, alpha=v[1][2])
    b.mul_(v[2][2]).add_(r, alpha=v[2][0]).add_(g, alpha=v[2][1])          # r, g still hold the OLD values here
    r.copy_(nr)
    g.copy_(ng)
    del nr, ng
    x.clamp_(0.0, 1.0)
    if kern is not None:
        k = kern.to(device=x.device, dtype=x.dtype).view(1, 1, BLUR_KERNEL, BLUR_KERNEL)
        pad = BLUR_KERNEL // 2
        for c in range(3):                                                  # one plane at a time: pad + conv output = 2 planes
            p = x[:, c:c + 1]
            p.copy_(F.conv2d(F.pad(p, (pad, pad, pad, pad), mode="replicate"), k))
        x.clamp_(0.0, 1.0)


def _check(t: torch.Tensor, name: str) -> None:
    if t.ndim != 5 or not t.is_floating_point():
        raise ValueError(f"[r9-photometric] {name} must be a float [B, rows, C, H, W] tensor, got {tuple(t.shape)} {t.dtype}")
    if t.shape[2] % 3 != 0:
        raise ValueError(f"[r9-photometric] {name} has C={t.shape[2]}: not a stack of RGB triplets (C % 3 != 0)")


@torch.no_grad()
def apply_(frames: torch.Tensor, fut_frames: torch.Tensor | None = None, *, seed: int, step: int) -> float:
    """Photometrically jitter ``frames [B, rows, 3K, H, W]`` and ``fut_frames [B, F, 3K, H, W]`` (float, [0, 1]) IN PLACE, one draw per sample (see the module docstring).

    -> the fraction of the B samples jittered.  The same ``(seed, step)`` always gives the same output; step + 1 gives another draw."""
    _check(frames, "frames")
    B = int(frames.shape[0])
    if fut_frames is not None:
        _check(fut_frames, "fut_frames")
        if fut_frames.shape[0] != B or fut_frames.shape[2] != frames.shape[2]:
            raise ValueError(f"[r9-photometric] fut_frames {tuple(fut_frames.shape)} does not match frames {tuple(frames.shape)} in (B, C)")
    p = draw_params(B, seed, step)
    K = int(frames.shape[2]) // 3
    n_on = 0
    for i in range(B):
        if not p["apply"][i]:
            continue
        n_on += 1
        m3 = color_matrix(p["fs"][i], p["hue"][i])
        kern = _gauss_kernel(p["sigma"][i]) if p["blur"][i] else None
        for t in (frames, fut_frames):
            if t is None:
                continue
            for k in range(K):
                _jitter_group_(t[i, :, 3 * k:3 * k + 3], p["fb"][i], p["fc"][i], m3, kern)
    return n_on / max(B, 1)


# ---- record + G-DVB ---------------------------------------------------------------------------------------------------------------- #
def stamp(model, args) -> dict | None:
    """``config.json["r9_photometric"]`` -- read OFF the attached state; ``None`` (key absent) when the flag is off."""
    st = getattr(model, MODEL_ATTR, None)
    if st is None:
        return None
    return {"flag": FLAG, "seed": int(st.seed), "draw": "per sample, from numpy SeedSequence([seed, step]) -> torch CPU float64; no RNG state to checkpoint",
            "brightness": BRIGHTNESS, "contrast": CONTRAST, "saturation": SATURATION, "hue_turns": HUE, "hue_rule": "linear YIQ chroma rotation (not HSV)",
            "blur": {"p": BLUR_P, "kernel": BLUR_KERNEL, "sigma_range": list(BLUR_SIGMA_RANGE)}, "p_apply": P_APPLY,
            "scope": "frames AND future_frames of the same sample share the draw; every label / pose / nav / goal key untouched",
            "channel_layout": "C = 3K: K stacked frames, oldest first, RGB per frame (comma2k19.py:633 stack_frames; physicalai.py:737)",
            "training_only": "compute_losses_v3 / compute_losses_perc_rig under model.training; eval, dump and BN recalibration never augment",
            "flip": "NOT built (E0a e4b028d: the train-vs-held-out AP gap is inside the bootstrap half-width) -- photometric only, disclosed"}


def dvb_check(model, args) -> list:
    """``built`` lever: declared (argv) <=> ``model._r9_photometric`` is attached, and its seed is the argv's."""
    from tanitad.train.declared_vs_built import Mismatch
    declared, st = is_on(args), getattr(model, MODEL_ATTR, None)
    out = []
    if declared != (st is not None):
        out.append(Mismatch(DEST, declared, st is not None, "model." + MODEL_ATTR,
                            "argv and the build disagree about the photometric augmentation (a declared augmentation that is never attached is not applied)"))
    elif st is not None and int(st.seed) != int(getattr(args, "seed", 0)):
        out.append(Mismatch(DEST, int(getattr(args, "seed", 0)), int(st.seed), "model." + MODEL_ATTR + ".seed",
                            "the augmentation draws from another seed than argv declares"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "built", dvb_check)
