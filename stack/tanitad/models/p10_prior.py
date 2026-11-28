"""v8F-alpha arm alpha-1 -- THE FROZEN P10 PRIOR, INSIDE THE REFCV DECODER PATH.

SPEC: ``Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`` section 6, rung alpha, arm alpha-1 (*"the
refcv8 recipe with ONE lever changed: the residual prior -> P10"*); design record
``TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha/DESIGN.md``.

WHAT P10 IS (read at source, not recalled)
------------------------------------------
``taniteval/taniteval/floors_kin.py::fit_p10`` -> ``tanitad.models.v8f_mini.P10Prior``: a ONE-hidden-layer MLP
(8 -> 256 -> GELU -> 60) on the eight kinematic features ``(v, a, k, v^2, v*a, v*k, v^2*k, a*k)`` of
``kin3 = (v0, a0, steer0)``, whose output is a RESIDUAL on ``kd_x`` (the damp50 path re-timed by ``ha0_ext``'s
travelled distance), all on the 0.2 s grid, 30 knots over 6 s, in the ego frame at t0, ``ha0_ext`` integrated in
the CORRECT unit (``action_units="steer"``: ``kappa = tan(steer) / 2.9``).

THIS MODULE IS A PORT, NOT A SECOND DEFINITION. ``kin3``, ``const_control_paths``, ``damp50``, ``retime``,
``kd_x`` and the standardise->MLP->de-standardise step are re-expressed in torch (the training tree must not import
``taniteval``: ``tanitad`` never depends on it) and PINNED to the numpy reference by known-value tests on real
windows (``stack/tests/test_p10_prior.py``: <= 1e-4 m against ``floors_kin`` AND against the banked
``p10_eval_pred6.npy`` that the fit script wrote). A port that agrees with the reference only on the data its
author looked at is how two "independent" checks agree on a wrong answer, so the test reads the BANKED artifact.

THE CLOCK (the thing G-CLOCK asks about)
----------------------------------------
``kin3`` is read from the OBSERVED window ``poses[:, n_past-1]`` (v0), ``poses[:, n_past-3]`` (the speed two
frames = 0.2 s earlier, the fit's stride-2 backward difference ``a0 = (v[t]-v[t-2])/0.2``) and
``actions[:, n_past-1, 0]`` (the RECORDED STEER at t0). Those are the SAME rows the ego-history encoder and the
``ha0_ext`` prior read, and ``RefCModel._residual_prior_inputs`` additionally asserts the window's last speed equals
the ``v0`` the plan is rolled from (``V0_WINDOW_TOL``). The 30 knots are then ``0.2 * j`` seconds AFTER that frame.
A window that ends one frame early/late changes v0, a0 and steer0 and therefore the whole path: the
one-tick-shifted-clock mutation in the tests moves the prediction by far more than the 1e-4 m tolerance.

TWO VARIANTS, ONE MODULE (``variant``: ``"steer"`` | ``"pose"``)
-----------------------------------------------------------------
``steer`` is the v8F-mini artifact: its third feature ``k`` is the RECORDED STEER at t0 and its ``kd_x`` base integrates
it as ``tan(k) / L``. ``pose`` (mode ``p10_pose``, the arm alpha-1 as launched) is refit with ``k`` := the POSE-TRACK
curvature ``kappa0`` of ``ha0_ext_pose`` (``kinematic_prior.pose_kin3``: ONE-frame ``a0``, ``omega0 / max(v0, 2.0)``
clamped to +-0.3) and its ``kd_x`` base integrates it as a curvature (``ha0_ext_pose`` instead of ``ha0_ext``). The variant
is declared by the ARTIFACT (``meta['units']``: ``steer`` | ``pose_kappa``) and a caller that names the other one is
REFUSED at load -- a steer P10 can never be loaded under ``p10_pose``, nor the reverse.

ADMISSIBILITY OF THE STEER VARIANT (stated, and an OPEN PI DECISION -- see DESIGN.md D1)
-------------------------------------------------------------------------------------
P10 reads the recorded steer at t0, exactly the channel ``ha0_ext`` reads, which SPEC_REFCV7 section 10 (A5)
excluded for refcv7 ("unruled at inference"). It is therefore in ``kinematic_prior.ACTION_MODES``, and a build
whose caller does not supply ``ego_actions`` REFUSES (``prior_controls``), it never degrades to a pose curvature.

FROZEN, BY CONSTRUCTION
-----------------------
Every P10 tensor is a BUFFER (``register_buffer(..., persistent=False)``): not a ``Parameter``, so it can never
enter an optimizer; not in ``state_dict``, so the warm start (``--init-from``: every key the source lacks must be a
declared refcv8 seam) and the strict checkpoint load are untouched. The weights' identity is carried by SHA-256
stamps instead -- the FILE's bytes and the live buffers' CONTENT -- both recorded in ``config.json`` and re-checked
at eval and at resume. The forward runs under ``torch.no_grad()`` so even a caller that flips ``requires_grad``
cannot route a gradient into it through this module.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import torch
from torch import Tensor, nn

from tanitad.models.kinematic import STEER_WHEELBASE_M

__all__ = [
    "P10_DT", "P10_K", "P10_STRIDE_FRAMES", "P10_STATE_KEYS", "P10Error", "kin3_from_window", "kin_features",
    "const_control_paths", "damp50", "retime", "kd_x_base", "FrozenP10", "load_frozen_p10", "content_sha256",
    "file_sha256", "p10_fit_clip_ids", "assert_fit_disjoint", "P10_UNITS_TO_VARIANT",
]

#: the artifact's declared feature unit (``meta['units']``) -> the variant it is
P10_UNITS_TO_VARIANT: dict[str, str] = {"steer": "steer", "pose_kappa": "pose"}

#: the fit's grid: 30 knots at 0.2 s (``v8f_mini_data.DT`` / ``HORIZON``)
P10_DT: float = 0.2
P10_K: int = 30
#: the fit's backward-difference stride in 10 Hz frames: ``a0 = (v[t] - v[t-2]) / 0.2``
#: (``v8f_mini_data.window_kin``: ``(poses[k, 3] - poses[k - 2, 3]) / DT``)
P10_STRIDE_FRAMES: int = 2
#: the artifact's state keys, in the order the content hash walks them
P10_STATE_KEYS: tuple[str, ...] = ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "xm", "xs", "ym")
_EPS_SEG = 1e-9


class P10Error(ValueError):
    """The P10 prior cannot be built / loaded as declared -- refused, never defaulted."""


# --------------------------------------------------------------------------------------------- identity
def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def content_sha256(tensors: dict[str, Tensor]) -> str:
    """sha256 over the canonical serialisation of the P10 tensors: for each key in :data:`P10_STATE_KEYS` the key,
    the shape and the float32 little-endian bytes. Independent of how the tensors are stored (file, buffers)."""
    h = hashlib.sha256()
    for k in P10_STATE_KEYS:
        if k not in tensors:
            raise P10Error(f"P10 tensors lack {k!r} (have {sorted(tensors)})")
        t = tensors[k].detach().to("cpu", torch.float32).contiguous()
        h.update(k.encode("utf-8"))
        h.update(repr(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- kinematics
def kin3_from_window(poses: Tensor, actions: Tensor, n_past: int | None = None, *, dt_frame: float = 0.1,
                     stride: int = P10_STRIDE_FRAMES) -> Tensor:
    """``[B, 3]`` float64 = ``(v0, a0, steer0)`` at the window's LAST observed frame, the fit's convention.

    ``poses`` ``[B, T, >=4]`` (x, y, yaw, v) and ``actions`` ``[B, T, 2]`` (steer, accel) are the OBSERVED window
    (``pose_hist`` / ``actions``); ``n_past`` how many leading steps are past (``None`` = all). Every read is at an
    index ``<= n_past - 1`` -- the slice is taken first, so a future read is impossible by construction.
    """
    if poses.dim() != 3 or poses.shape[-1] < 4:
        raise P10Error(f"poses must be [B, T, >=4], got {tuple(poses.shape)}")
    if actions.dim() != 3 or actions.shape[-1] != 2 or actions.shape[0] != poses.shape[0]:
        raise P10Error(f"actions must be [B, T, 2] with the poses' batch, got {tuple(actions.shape)}")
    n = int(poses.shape[1] if n_past is None else n_past)
    if n <= stride:
        raise P10Error(f"n_past {n} <= stride {stride}: P10's a0 is a {stride}-frame backward difference and needs "
                       f"frame t0-{stride} inside the window")
    if actions.shape[1] < n:
        raise P10Error(f"actions cover {actions.shape[1]} steps, the window has {n} past steps")
    p = poses[:, :n].to(torch.float64)
    a = actions[:, :n].to(torch.float64)
    v0 = p[:, n - 1, 3]
    a0 = (v0 - p[:, n - 1 - stride, 3]) / (float(stride) * float(dt_frame))
    return torch.stack([v0, a0, a[:, n - 1, 0]], dim=-1)


def kin_features(kin3: Tensor) -> Tensor:
    """``[n, 8]`` = (v, a, k, v^2, v*a, v*k, v^2*k, a*k) -- ``floors_kin.kin_features`` verbatim."""
    v, a, k = kin3[:, 0], kin3[:, 1], kin3[:, 2]
    return torch.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], dim=1)


def const_control_paths(v0: Tensor, a: Tensor, k: Tensor, dt: float, n: int, *, steer_units: bool = True) -> Tensor:
    """``[B, n, 2]`` ego-frame paths under CONSTANT (accel ``a``, channel-1 ``k``) -- ``floors_kin.
    const_control_paths`` in torch float64. ``steer_units``: ``kappa = tan(k) / L`` first (the correct reading of a
    RECORDED action). Forward Euler, position and heading on the speed at the START of the step, ``v`` updated LAST
    and clamped at 0 -- ``kinematic.rollout_unicycle``."""
    v = v0.to(torch.float64).clone()
    a = a.to(torch.float64).expand_as(v)
    kap = k.to(torch.float64).expand_as(v)
    if steer_units:
        kap = torch.tan(kap) / float(STEER_WHEELBASE_M)
    x = torch.zeros_like(v)
    y = torch.zeros_like(v)
    yaw = torch.zeros_like(v)
    out = v.new_empty(v.shape + (int(n), 2))
    for j in range(int(n)):
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + v * kap * dt
        v = (v + a * dt).clamp_min(0.0)
        out[..., j, 0] = x
        out[..., j, 1] = y
    return out


def damp50(ha0: Tensor, ha0_ext: Tensor) -> Tensor:
    return 0.5 * ha0 + 0.5 * ha0_ext


def _arclen(p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n = p.shape[0]
    P = torch.cat([p.new_zeros(n, 1, 2), p], dim=1)
    seg = P[:, 1:] - P[:, :-1]
    L = torch.linalg.norm(seg, dim=-1)
    S = torch.cat([L.new_zeros(n, 1), torch.cumsum(L, dim=1)], dim=1)
    return P, seg, L, S


def retime(path: Tensor, lon_src: Tensor) -> Tensor:
    """``floors_kin.retime`` in torch float64: re-time ``path`` so its along-track distance at step k equals
    ``lon_src``'s own travelled distance. ``[B, K, 2]`` positions at steps 1..K (origin implicit).

    Case by case as the reference: a degenerate damped path (total length < 1e-6 or no segment longer than 1e-9)
    -> straight ahead ``(s, 0)``; ``s`` inside the path -> linear interpolation in the segment that holds it (a
    zero-length segment is skipped forward); ``s`` beyond the end -> extrapolate along the LAST non-zero segment."""
    path = path.to(torch.float64)
    lon_src = lon_src.to(torch.float64)
    n, K, _ = path.shape
    P, seg, L, S = _arclen(path)
    s_src = _arclen(lon_src)[3][:, 1:]                                          # [n, Ks]
    nz = L > _EPS_SEG
    deg = (S[:, -1] < 1e-6) | ~nz.any(1)
    jj = (K - 1) - torch.argmax(nz.flip(1).to(torch.int64), dim=1)             # last non-zero segment
    nxt = torch.empty((n, K), dtype=torch.int64, device=path.device)
    nxt[:, K - 1] = K - 1
    for i in range(K - 2, -1, -1):
        nxt[:, i] = torch.where(nz[:, i], torch.full_like(nxt[:, i], i), nxt[:, i + 1])
    j = (S[:, None, :] <= s_src[:, :, None]).sum(-1) - 1                       # searchsorted(S, s, 'right') - 1
    j = j.clamp(0, K - 1)
    j = torch.gather(nxt, 1, j)
    Lj = torch.gather(L, 1, j)
    Pj = torch.gather(P, 1, j[..., None].expand(-1, -1, 2))
    segj = torch.gather(seg, 1, j[..., None].expand(-1, -1, 2))
    Sj = torch.gather(S, 1, j)
    inside = Pj + ((s_src - Sj) / Lj.clamp_min(1e-12))[..., None] * segj
    res = torch.where((Lj > _EPS_SEG)[..., None], inside, Pj)
    jj_c = jj.clamp(0, K - 1)[:, None]
    Ljj = torch.gather(L, 1, jj_c)
    segjj = torch.gather(seg, 1, jj_c[..., None].expand(-1, -1, 2))
    beyond = P[:, -1:, :] + ((s_src - S[:, -1:]) / Ljj.clamp_min(1e-12))[..., None] * segjj
    res = torch.where((s_src <= S[:, -1:])[..., None], res, beyond)
    straight = torch.stack([s_src, torch.zeros_like(s_src)], dim=-1)
    return torch.where(deg[:, None, None], straight, res)


def kd_x_base(kin3: Tensor, *, dt: float = P10_DT, n: int = P10_K, steer_units: bool = True) -> Tensor:
    """``v8f_mini_data.kdx_base`` in torch: ``kd_x`` on the 30 x 0.2 s grid from ``kin3`` ``[B, 3]``. ``steer_units``:
    the third column is a STEER angle (``tan(k)/L`` first; the ``steer`` variant) or a CURVATURE (the ``pose`` variant:
    ``ha0_ext_pose``). Float64, rounded to float32 exactly where the reference casts (``kd_x(...).astype(float32)``)."""
    kin3 = kin3.to(torch.float64)
    z = torch.zeros_like(kin3[:, 0])
    ha0 = const_control_paths(kin3[:, 0], z, z, dt, n, steer_units=False)
    ext = const_control_paths(kin3[:, 0], kin3[:, 1], kin3[:, 2], dt, n, steer_units=steer_units)
    return retime(damp50(ha0, ext), ext).to(torch.float32)


# --------------------------------------------------------------------------------------------- the module
class FrozenP10(nn.Module):
    """The frozen, fitted P10 floor as a BUFFER-ONLY module. ``forward(kin3) -> [B, 30, 2]`` (float32, metres, the
    ego frame at the window's last frame, knots at ``0.2 * j`` s). No parameters; the forward is a no_grad island."""

    def __init__(self, tensors: dict[str, Tensor], *, source: str = "", file_sha: str = "", meta: dict | None = None,
                 variant: str = "steer"):
        super().__init__()
        if variant not in ("steer", "pose"):
            raise P10Error(f"P10 variant {variant!r} is not one of ('steer', 'pose')")
        self.variant = str(variant)
        w1, b1, w2, b2 = (tensors[k].detach().to(torch.float32).clone() for k in P10_STATE_KEYS[:4])
        if w1.dim() != 2 or w1.shape[1] != 8:
            raise P10Error(f"P10 first layer must be [hidden, 8], got {tuple(w1.shape)}")
        if w2.shape != (2 * P10_K, w1.shape[0]):
            raise P10Error(f"P10 output layer must be [{2 * P10_K}, {w1.shape[0]}] (30 knots x 2), got "
                           f"{tuple(w2.shape)}")
        # persistent=False: NOT in state_dict (see the module docstring) -- identity travels by sha256 stamps
        for name, t in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
            self.register_buffer(name, t, persistent=False)
        for name, key in (("xm", "xm"), ("xs", "xs"), ("ym", "ym")):
            self.register_buffer(name, tensors[key].detach().to(torch.float32).clone(), persistent=False)
        self.source = str(source)
        self.file_sha = str(file_sha)
        self.meta = dict(meta or {})
        self.content_sha_at_load = self.content_sha()

    # the tensors in the artifact's own key space
    def tensors(self) -> dict[str, Tensor]:
        return {"net.0.weight": self.w1, "net.0.bias": self.b1, "net.2.weight": self.w2, "net.2.bias": self.b2,
                "xm": self.xm, "xs": self.xs, "ym": self.ym}

    def content_sha(self) -> str:
        return content_sha256(self.tensors())

    @torch.no_grad()
    def forward(self, kin3: Tensor) -> Tensor:
        """``kin3`` ``[B, 3]`` = (v0, a0, steer0) for the ``steer`` variant, (v0, a0, kappa0) for the ``pose`` variant
        -> the P10 path ``[B, 30, 2]`` float32."""
        with torch.autocast(device_type=kin3.device.type, enabled=False):
            kin3 = kin3.detach().to(torch.float64)
            base = kd_x_base(kin3, steer_units=(self.variant == "steer"))      # [B, 30, 2] float32
            f8 = kin_features(kin3)                                            # float64
            x = ((f8 - self.xm.to(torch.float64)) / self.xs.to(torch.float64)).to(torch.float32)
            h = torch.nn.functional.gelu(x @ self.w1.t() + self.b1)
            y = (h @ self.w2.t() + self.b2).to(torch.float64) + self.ym.to(torch.float64)
            return (base.to(torch.float64) + y.reshape(-1, P10_K, 2)).to(torch.float32)


def load_frozen_p10(path: str | os.PathLike, *, expect_file_sha256: str | None = None,
                    variant: str | None = None) -> FrozenP10:
    """Load a banked P10 artifact (``v8f_mini.save_p10``'s format) into a :class:`FrozenP10`.

    ``expect_file_sha256`` (if given) is compared to the file's bytes BEFORE anything is built: a wrong-sha file is
    refused here, so a run can never train on a P10 other than the one its record names. The artifact DECLARES its
    feature unit in ``meta['units']`` (``steer`` | ``pose_kappa``); a missing or unknown unit is refused, and so is a
    ``variant`` that is not the one the artifact declares."""
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise P10Error(f"P10 file {p!r} does not exist")
    fsha = file_sha256(p)
    if expect_file_sha256 and fsha != str(expect_file_sha256).lower():
        raise P10Error(f"P10 file {p!r} has sha256 {fsha}, the record names {expect_file_sha256}")
    o = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(o, dict) or "state" not in o:
        raise P10Error(f"P10 file {p!r} is not a v8f_mini.save_p10 artifact (no 'state')")
    st = o["state"]
    missing = [k for k in P10_STATE_KEYS if k not in st]
    if missing:
        raise P10Error(f"P10 file {p!r} lacks {missing}")
    if int(o.get("k", P10_K)) != P10_K:
        raise P10Error(f"P10 file {p!r} has k={o.get('k')}, the decoder path needs {P10_K} knots")
    meta = o.get("meta") or {}
    units = meta.get("units")
    if units not in P10_UNITS_TO_VARIANT:
        raise P10Error(f"P10 file {p!r} does not declare a known feature unit (meta['units']={units!r}, expected one of "
                       f"{sorted(P10_UNITS_TO_VARIANT)}): which input channel it was fit on is unknown")
    declared = P10_UNITS_TO_VARIANT[units]
    if variant is not None and variant != declared:
        raise P10Error(f"P10 file {p!r} is the {declared!r} variant (meta['units']={units!r}); the caller asked for the "
                       f"{variant!r} one -- a P10 fit on another input channel is ANOTHER prior")
    return FrozenP10({k: st[k] for k in P10_STATE_KEYS}, source=p, file_sha=fsha, meta=meta, variant=declared)


# --------------------------------------------------------------------------------------------- provenance
def p10_fit_clip_ids(manifest_clip_ids, n_fit: int = 600) -> list[str]:
    """The fit's clip set: the first ``n_fit`` train clips ordered by ``sha12`` (``v8f_mini_fit_p10.py``:
    ``sorted(byc, key=D.sha12)[:a.clips]``)."""
    key = lambda c: hashlib.sha256(str(c).encode("utf-8")).hexdigest()[:12]      # noqa: E731
    return sorted((str(c) for c in manifest_clip_ids), key=key)[: int(n_fit)]


def assert_fit_disjoint(fit_clips, eval_clips) -> dict[str, Any]:
    """The P10 fit clips and the eval clips are disjoint by clip id AND by stable episode id
    (``blake2b-8(clip) >> 1``, ``v8f_mini_data.stable_episode_id``)."""
    sid = lambda c: int.from_bytes(hashlib.blake2b(str(c).encode("utf-8"), digest_size=8).digest(), "big") >> 1  # noqa
    f, e = set(map(str, fit_clips)), set(map(str, eval_clips))
    if f & e:
        raise P10Error(f"P10 fit clips overlap the eval clips: {sorted(f & e)[:3]} ({len(f & e)})")
    if {sid(c) for c in f} & {sid(c) for c in e}:
        raise P10Error("P10 fit clips and eval clips share a stable episode id")
    return {"n_fit": len(f), "n_eval": len(e), "overlap": 0, "sid_overlap": 0}
