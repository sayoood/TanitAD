"""P9 — probe-gradient saliency: WHERE IN THE IMAGE each physical belief
comes from (DIAGRAM_CONFORMANCE.md F-19; the battery's last unbuilt probe).

WHAT IT MEASURES. For a FROZEN checkpoint and a chosen readout of the latent
(a "physical belief": kinematic speed, yaw rate, lead gap when computable),
the gradient of the readout's output w.r.t. the INPUT PIXELS:

    S[t, h, w] = sum_c | d y / d frames[t, c, h, w] |

aggregated per readout target over a few windows. A belief whose saliency
lives on the road surface ahead is grounded; one whose saliency lives on the
sky or the hood is reading a spurious cue — this is the instrument that can
SEE the difference, which no scalar probe R^2 can.

TARGETS (each stamps its provenance and its structural support):
  * ``speed`` / ``yaw``  — the checkpoint's OWN ``step_readout_op``
    (:class:`tanitad.models.metric_dynamics.StepDisplacementReadout`) applied
    to the ENCODED pair (z[-2], z[-1]); speed = |dxy| / dt, yaw = dyaw / dt.
    These heads train from step 0 of S-W (O1), so they are meaningful at any
    pause checkpoint.
  * ridge targets from ``--probe-arrays`` (``probe_latent_state.py``'s P9
    reuse dump: full-set ridge directions w, b per driving target, including
    ``lead_gap``) — ``y = w . z[-1] + b``. ⛔ REFUSED unless the dump's
    recorded checkpoint matches ``--ckpt`` (or ``--accept-probe-family``
    states the operator checked): a probe direction fit on another model's
    latent space produces a colourful, meaningless map.
  * ``lead-state`` is therefore computable ONLY via ``--probe-arrays``; when
    absent the summary SAYS SO with the reason (a missing metric is declared,
    never silently dropped — PI 2026-08-02).

⛔ STRUCTURAL SUPPORT IS STAMPED, SO A VACUOUS READING IS UNQUOTABLE (the
``rank_ceiling`` lesson applied here). ``V6Stack.encode_window`` encodes the
W frames INDEPENDENTLY — there is no cross-frame attention — so a readout
consuming z[j] can reach ONLY frames[j], and zero saliency outside the
declared support frames is a GRAPH FACT, not a temporal-attention finding.
The record carries ``support_frames`` + this note, and the energy outside the
support is VERIFIED to be exactly zero: nonzero there means the target's
declared support is wrong or the graph has an undeclared path, and the record
flips ``support_violation`` (the guard is proved able to fire in
``tests/test_p9_saliency.py`` by mis-declaring a support on purpose).

FOV SPLIT (``bev_raster.fov_mask`` reused as a PREDICATE, P4 discipline).
The saliency map lives in the image plane, so the mask's cell test
``|azimuth| <= half_angle`` is applied to pixel COLUMNS through the encoder's
own :class:`~tanitad.data.calib.CanonicalFrame` (cylindrical: azimuth linear
in column; pinhole: atan). Two consequences, both stamped:
  * when the mask half-angle >= the frame's own half-angle, EVERY column is
    inside and ``in_fov_energy_share = 1.0`` BY CONSTRUCTION —
    ``fov_split.vacuous_by_construction`` is then True and the number must
    not be quoted as a finding;
  * for the LEAD-STATE target the lead's BEV position is graded by the SAME
    predicate at the agent centre (= ``visibility_occ``, MEASURED identical:
    P4_FOV_PREDICATE.md, 0/7680 cells disagree) and windows are grouped by
    it. ⚠️ THE OUT-OF-FOV GROUP IS MEANINGFUL, NOT NOISE (the P4 lesson): a
    belief about an out-of-field lead can only come from earlier frames /
    context, which is exactly what its saliency shows. It is reported as a
    group, never dropped or masked away.

TIER STAMP: T0-DIAGNOSTIC BY DESIGN — this interrogates a REPRESENTATION.
No number here is driving performance; no ADE/four-family table is produced
(this is one probe row of the WM-physics battery).

RUNNABLE-AT-A-CHECKPOINT (the S-W -> S-T pause list): the full path needs a
v6 staged checkpoint (``{"stack": state_dict, "config": {...}}`` or the
fp16 snapshot ``{"model": ..., "_fp16_weights_only": true}``) plus a windows
tensor (``--windows-pt``: ``{"frames": [N, W, C, H, W'], "v0": [N]?,
"lead_x_m"/"lead_y_m": [N]?}``). GPU execution is NOT this stream's; the
CPU-smoke path is ``--synthetic`` (tiny stack, random frames) and is what the
tests run. Usage at the pause (Thor-side, PYTHONPATH=stack):

  python scripts/probe_saliency_p9.py \
      --ckpt /home/nvidia/experiments/v6F-SW-30k/ckpt.pt \
      --windows-pt /home/nvidia/experiments/p9_windows.pt \
      --out /home/nvidia/experiments/p9-saliency \
      --frame-hfov 117 --projection cylindrical

Evidence class of any REAL run: MEASURED (ours; artifact = the out dir).
The ``--synthetic`` smoke stamps quotable: NONE in its own summary.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))

from tanitad.config import (EncoderConfig, PredictorConfig,  # noqa: E402
                            ReadoutConfig)
from tanitad.data.calib import CanonicalFrame  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402

#: The default split half-angle IS ``bev_raster.fov_mask``'s default (60 deg;
#: bit-identical to ``visibility_occ``'s — P4_FOV_PREDICATE.md §1.1). Passing
#: the encoder frame's own half-angle instead makes the split vacuous, which
#: the record then says.
FOV_MASK_DEFAULT_HALF_ANGLE_DEG = 60.0

#: Relative tolerance for "energy outside the declared support is zero".
#: The comparison is against an exact-zero graph fact, so this only guards
#: float accumulation noise.
SUPPORT_RTOL = 1e-9


# ============================================================================
# checkpoint -> V6Stack
# ============================================================================
def v6config_from_dict(d: dict) -> V6Config:
    """Rebuild :class:`V6Config` from ``stack.cfg.to_dict()`` output.

    Defensive by design: only fields the CURRENT dataclasses declare are
    passed (a checkpoint written by a newer v6.py must not crash the probe on
    an unknown key — it is reported instead), and list-vs-tuple drift from
    JSON round-tripping is normalised.
    """
    d = dict(d)
    d.pop("_derived", None)

    def _build(cls, sub: dict):
        names = {f.name for f in dataclasses.fields(cls)}
        kept = {k: v for k, v in sub.items() if k in names}
        for k, v in kept.items():
            if isinstance(v, list):
                kept[k] = tuple(v)
        dropped = sorted(set(sub) - names)
        return cls(**kept), dropped

    enc, drop_e = _build(EncoderConfig, d.pop("encoder"))
    ro, drop_r = _build(ReadoutConfig, d.pop("readout"))
    pr, drop_p = _build(PredictorConfig, d.pop("predictor"))
    cfg, drop_c = _build(V6Config, {**d, "encoder": 0, "readout": 0,
                                    "predictor": 0})
    cfg = dataclasses.replace(cfg, encoder=enc, readout=ro, predictor=pr)
    dropped = {"encoder": drop_e, "readout": drop_r, "predictor": drop_p,
               "v6": [k for k in drop_c
                      if k not in ("encoder", "readout", "predictor")]}
    if any(dropped.values()):
        print(f"[p9] WARNING: config keys unknown to this v6.py were "
              f"DROPPED: {dropped} — a newer trainer wrote this checkpoint",
              flush=True)
    return cfg


def load_v6_stack(ckpt_path: str) -> tuple[V6Stack, dict]:
    """``ckpt.pt`` (trainer format) or the fp16 snapshot -> a frozen stack.

    The fp16-snapshot unwrap mirrors ``load_stage_init``'s 2026-08-16 fix:
    looking only for ``"stack"`` blamed a geometry mismatch on an unopened
    container.
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise SystemExit(f"[p9] ⛔ --ckpt {p} does not exist")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "stack" in ck:
        sd, cfg_json = ck["stack"], ck.get("config", {})
    elif isinstance(ck, dict) and "model" in ck:
        sd, cfg_json = ck["model"], ck.get("_meta", {}).get("config",
                                                            ck.get("config",
                                                                   {}))
    else:
        raise SystemExit(
            f"[p9] ⛔ {p} is neither a v6 trainer checkpoint ('stack') nor an "
            f"fp16 snapshot ('model'); keys: {sorted(ck)[:8] if isinstance(ck, dict) else type(ck)}")
    vcfg = (cfg_json or {}).get("v6_config")
    if not vcfg:
        raise SystemExit(
            f"[p9] ⛔ {p} carries no v6_config — cannot rebuild the stack "
            f"geometry. (A guessed geometry would load garbage silently.)")
    stack = V6Stack(v6config_from_dict(vcfg))
    sd = {k: v for k, v in sd.items()}
    missing, unexpected = stack.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"[p9] ⛔ state_dict mismatch (missing {list(missing)[:4]}, "
            f"unexpected {list(unexpected)[:4]}) — refusing to probe a "
            f"partially-loaded model: its saliency would be noise wearing "
            f"the checkpoint's name.")
    return stack, {"ckpt": str(p),
                   "step": ck.get("step"),
                   "run": (cfg_json or {}).get("run")}


def freeze(stack: V6Stack) -> V6Stack:
    stack.eval()
    for prm in stack.parameters():
        prm.requires_grad_(False)
    return stack


# ============================================================================
# targets — each carries fn, structural support, provenance
# ============================================================================
class TargetSpec:
    """``fn(z_win) -> [B]`` plus the frames it can structurally reach."""

    def __init__(self, name: str, fn, support_frames: list[int],
                 provenance: str, family: str):
        self.name, self.fn = name, fn
        self.support_frames = list(support_frames)
        self.provenance, self.family = provenance, family


def builtin_targets(stack: V6Stack) -> list[TargetSpec]:
    """speed / yaw from the checkpoint's own step readout (per-step Δpose on
    the encoded last transition). Support: the two frames the pair reads."""
    w = stack.cfg.predictor.window
    dt = stack.cfg.dt

    def _dpose(z):
        return stack.step_readout_op(z[:, -2], z[:, -1])   # [B, 3] dx dy dyaw

    return [
        TargetSpec("speed",
                   lambda z: _dpose(z)[:, :2].norm(dim=-1) / dt,
                   [w - 2, w - 1],
                   "checkpoint step_readout_op: |Δxy(z[-2], z[-1])| / dt "
                   "(m/s over the last encoded transition)",
                   "kinematic"),
        TargetSpec("yaw",
                   lambda z: _dpose(z)[:, 2] / dt,
                   [w - 2, w - 1],
                   "checkpoint step_readout_op: Δyaw(z[-2], z[-1]) / dt "
                   "(rad/s over the last encoded transition)",
                   "kinematic"),
    ]


def ridge_targets(probe_arrays: str, ckpt_path: str | None, d_op: int,
                  accept_family: bool, window: int) -> tuple[list, list[dict]]:
    """Load ``probe_latent_state.py``'s dumped full-set ridge directions.

    Returns ``(targets, refusals)`` — every direction that cannot be used is
    a REFUSAL RECORD with its reason, never a silent drop.
    """
    p = Path(probe_arrays)
    if not p.exists():
        raise SystemExit(f"[p9] ⛔ --probe-arrays {p} does not exist")
    dump = torch.load(p, map_location="cpu", weights_only=False)
    probes = dump.get("probes", dump)
    refusals: list[dict] = []
    fam_src = str(dump.get("ckpt", ""))
    if not accept_family:
        if not fam_src or not ckpt_path or \
                Path(fam_src).name != Path(ckpt_path).name:
            refusals.append({
                "targets": sorted(probes) if isinstance(probes, dict) else [],
                "reason": f"probe family unverified: dump records ckpt="
                          f"{fam_src or 'NONE'} vs --ckpt "
                          f"{Path(ckpt_path).name if ckpt_path else 'NONE'}. "
                          f"A ridge direction fit on another model's latents "
                          f"is meaningless here. Pass --accept-probe-family "
                          f"only after checking the provenance yourself.",
            })
            return [], refusals
    out: list[TargetSpec] = []
    for name, rec in (probes.items() if isinstance(probes, dict) else []):
        w_vec = rec.get("w") if isinstance(rec, dict) else None
        if w_vec is None:
            refusals.append({"targets": [name],
                             "reason": "no 'w' in the probe record"})
            continue
        w_vec = torch.as_tensor(w_vec, dtype=torch.float32).reshape(-1)
        b = float(rec.get("b", 0.0)) if isinstance(rec, dict) else 0.0
        if w_vec.numel() != d_op:
            refusals.append({
                "targets": [name],
                "reason": f"dim mismatch: probe w has {w_vec.numel()} dims, "
                          f"the checkpoint's d_op is {d_op} — different "
                          f"latent space, refused"})
            continue
        if float(w_vec.abs().max()) == 0.0:
            refusals.append({"targets": [name],
                             "reason": "degenerate all-zero direction"})
            continue
        fam = "lead-state" if "lead" in name else "ridge"
        out.append(TargetSpec(
            f"ridge_{name}",
            (lambda z, _w=w_vec, _b=b: z[:, -1] @ _w + _b),
            [window - 1],
            f"probe_arrays.pt full-set ridge direction {name!r} on the "
            f"ENCODED last-frame latent (w.z[-1] + b)",
            fam))
    return out, refusals


# ============================================================================
# the saliency pass
# ============================================================================
def saliency(stack: V6Stack, frames: Tensor, target: TargetSpec
             ) -> tuple[Tensor, Tensor]:
    """``frames [N, W, C, H, W']`` -> per-pixel saliency ``[N, W, H, W']``
    and the readout values ``[N]``. Params are frozen; only the input gets a
    gradient."""
    x = frames.detach().clone().requires_grad_(True)
    z = stack.encode_window(x)
    y = target.fn(z)
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"target {target.name} must map to [B], got "
                         f"{tuple(y.shape)}")
    y.sum().backward()
    if x.grad is None:                                     # pragma: no cover
        raise RuntimeError(f"no input gradient for target {target.name}")
    return x.grad.abs().sum(dim=2), y.detach()


def frame_energy_report(sal: Tensor, support: list[int]) -> dict:
    """Per-frame energy + the STRUCTURAL-SUPPORT verification.

    Zero outside the support is a graph fact (independent per-frame encode);
    nonzero there means the declared support is wrong or an undeclared path
    exists — the record flips ``support_violation`` and the number is
    unquotable until the wiring is explained.
    """
    e = sal.sum(dim=(0, 2, 3)).double()                    # [W]
    total = float(e.sum())
    w = e.numel()
    outside = [j for j in range(w) if j not in support]
    e_out = float(e[outside].sum()) if outside else 0.0
    return {
        "energy_by_frame": [float(v) for v in e],
        "energy_total": total,
        "support_frames": list(support),
        "support_note": "encode_window encodes frames INDEPENDENTLY (no "
                        "cross-frame attention): zero energy outside "
                        "support_frames is a GRAPH FACT, not a "
                        "temporal-attention finding",
        "energy_outside_support": e_out,
        "support_violation": bool(e_out > SUPPORT_RTOL * max(total, 1e-30)),
        "last_frame_share": (float(e[-1] / total) if total > 0 else None),
    }


def column_azimuth_rad(frame: CanonicalFrame, n_cols: int) -> Tensor:
    """Azimuth of each pixel-column centre — the image-plane form of
    ``bev_raster.cell_azimuth_rad``. Cylindrical: linear in column;
    pinhole: atan. Columns are the SALIENCY map's columns (== frame.width)."""
    cx = frame.width / 2.0
    scale = frame.width / float(n_cols)
    px = (torch.arange(n_cols, dtype=torch.float64) + 0.5) * scale - cx
    if frame.projection == "cylindrical":
        return px / float(frame.f_ref)
    return torch.atan2(px, torch.tensor(float(frame.f_ref),
                                        dtype=torch.float64))


def fov_split(sal: Tensor, frame: CanonicalFrame,
              half_angle_rad: float) -> dict:
    """Energy inside/outside the fov_mask predicate applied to columns.

    ``vacuous_by_construction`` is stamped whenever the frame's own field is
    narrower than or equal to the mask — in_share is then 1.0 no matter what
    the model attends to, and quoting it as a finding is the rank_ceiling
    error in image clothes."""
    az = column_azimuth_rad(frame, sal.shape[-1])
    inside = az.abs() <= float(half_angle_rad)
    e_cols = sal.sum(dim=(0, 1, 2)).double()               # [W']
    total = float(e_cols.sum())
    e_in = float(e_cols[inside].sum())
    vac = frame.half_angle_x_rad() <= float(half_angle_rad) + 1e-12
    return {
        "predicate": "|column azimuth| <= half_angle — the SAME test "
                     "bev_raster.fov_mask applies to BEV cell centres "
                     "(identical to visibility_occ, P4_FOV_PREDICATE.md)",
        "half_angle_deg": math.degrees(float(half_angle_rad)),
        "frame_half_angle_deg": math.degrees(frame.half_angle_x_rad()),
        "in_fov_energy_share": (e_in / total) if total > 0 else None,
        "out_fov_energy_share": ((total - e_in) / total) if total > 0
        else None,
        "n_cols_in": int(inside.sum()), "n_cols": int(az.numel()),
        "vacuous_by_construction": bool(vac),
        "vacuous_note": ("frame field <= mask angle: every column is inside "
                         "and in_share=1.0 BY CONSTRUCTION — not a finding"
                         if vac else
                         "mask is narrower than the frame: the split is "
                         "informative"),
    }


def lead_groups(per_window_energy: Tensor, last_frame_share: Tensor,
                lead_x: Tensor, lead_y: Tensor,
                half_angle_rad: float) -> dict:
    """Group windows by the LEAD's own in/out-of-field status (agent-centre
    azimuth — the ``visibility_occ`` == ``fov_mask`` predicate, P4).

    ⚠️ P4 lesson, applied: the out-of-FOV group is MEANINGFUL, not noise —
    those beliefs can only be fed by earlier frames / context, and their
    saliency is exactly the evidence of that. Both groups are reported with
    their n; neither is dropped."""
    az = torch.atan2(lead_y.double(), lead_x.double())
    inside = az.abs() <= float(half_angle_rad)
    out = {"predicate": "|atan2(lead_y, lead_x)| <= half_angle at the AGENT "
                        "CENTRE — visibility_occ == fov_mask "
                        "(P4_FOV_PREDICATE.md: 0/7680 cells disagree)",
           "half_angle_deg": math.degrees(float(half_angle_rad)),
           "p4_note": "the out_of_fov group is MEANINGFUL, not noise (the "
                      "P4 lesson): its beliefs can only come from earlier "
                      "frames/context, which is what its saliency shows. "
                      "Never drop or mask it."}
    for label, m in (("in_fov", inside), ("out_of_fov", ~inside)):
        n = int(m.sum())
        grp = {"n": n}
        if n:
            grp |= {"mean_energy": float(per_window_energy[m].mean()),
                    "mean_last_frame_share":
                        float(last_frame_share[m].mean())}
        else:
            grp["note"] = "no windows in this group (n=0) — reported, not " \
                          "imputed"
        out[label] = grp
    return out


# ============================================================================
# artifacts
# ============================================================================
def write_png_gray(path: Path, arr: Tensor) -> None:
    """Minimal 8-bit grayscale PNG (stdlib only — pods/Thor need no plotting
    stack for a heatmap)."""
    a = arr.detach().double()
    rng = float(a.max() - a.min())
    a = (a - a.min()) / (rng if rng > 0 else 1.0)
    img = (a * 255).round().to(torch.uint8).cpu().numpy()
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)


# ============================================================================
# the pass
# ============================================================================
def run(a) -> dict:
    t0 = time.time()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---- model -------------------------------------------------------------
    if a.synthetic:
        cfg = V6Config(
            encoder=EncoderConfig(in_channels=3, image_size=64, patch_size=16,
                                  d_model=32, depth=1, n_heads=2),
            readout=ReadoutConfig(grid=2, d_readout=16),
            predictor=PredictorConfig(d_model=32, depth=1, n_heads=2,
                                      window=4, horizons=(1,)),
            d_tac=32, d_str=16, d_goal_embed=8, adapter_hidden=32,
            f_hidden_tac=32, f_hidden_str=16, f_blocks=1, n_candidates=2)
        torch.manual_seed(a.seed)
        stack, meta = V6Stack(cfg), {"ckpt": "SYNTHETIC (random init)"}
    else:
        if not a.ckpt:
            raise SystemExit("[p9] ⛔ need --ckpt (or --synthetic for the "
                             "CPU smoke)")
        stack, meta = load_v6_stack(a.ckpt)
    stack = freeze(stack)
    cfg = stack.cfg
    W = cfg.predictor.window
    c, h, w_px = cfg.encoder.in_channels, *cfg.encoder.image_hw()
    # ---- windows -----------------------------------------------------------
    lead_x = lead_y = None
    if a.windows_pt:
        blob = torch.load(a.windows_pt, map_location="cpu",
                          weights_only=False)
        frames = blob["frames"].float()
        if frames.dtype == torch.uint8 or frames.max() > 8.0:
            frames = frames.float() / 255.0
        if frames.ndim != 5 or frames.shape[1:] != (W, c, h, w_px):
            raise SystemExit(
                f"[p9] ⛔ --windows-pt frames {tuple(frames.shape)} do not "
                f"match the checkpoint's [N, {W}, {c}, {h}, {w_px}]")
        lead_x, lead_y = blob.get("lead_x_m"), blob.get("lead_y_m")
        source = str(a.windows_pt)
    elif a.synthetic:
        g = torch.Generator().manual_seed(a.seed + 1)
        frames = torch.rand(a.n_windows, W, c, h, w_px, generator=g)
        # synthetic lead meta so the grouping path is exercised end-to-end:
        lead_x = torch.rand(a.n_windows, generator=g) * 40 + 2.0
        lead_y = (torch.rand(a.n_windows, generator=g) - 0.5) * 30
        source = "SYNTHETIC random frames + lead meta"
    else:
        raise SystemExit("[p9] ⛔ need --windows-pt (or --synthetic)")
    frames = frames[: a.n_windows]
    # ---- geometry ----------------------------------------------------------
    frame_geom = CanonicalFrame.from_hfov(a.frame_hfov, h, w_px, a.projection)
    half = math.radians(a.mask_half_angle_deg)
    # ---- targets -----------------------------------------------------------
    targets = builtin_targets(stack)
    refusals: list[dict] = []
    if a.probe_arrays:
        extra, refusals = ridge_targets(a.probe_arrays, a.ckpt, cfg.d_op,
                                        a.accept_probe_family, W)
        targets += extra
    if not any(t.family == "lead-state" for t in targets):
        refusals.append({
            "targets": ["lead-state"],
            "reason": "not computable at this checkpoint: no lead readout "
                      "lives in the v6 stack and no admissible lead_gap "
                      "ridge direction was supplied (--probe-arrays with a "
                      "family-matched dump provides one)"})
    # ---- the pass ----------------------------------------------------------
    per_target: dict = {}
    for t in targets:
        sal, y = saliency(stack, frames, t)                # [N, W, H, W']
        rep = {"provenance": t.provenance, "family": t.family,
               "n_windows": int(sal.shape[0]),
               "readout_values": [round(float(v), 5) for v in y],
               **frame_energy_report(sal, t.support_frames),
               "fov_split": fov_split(sal, frame_geom, half)}
        if float(sal.abs().max()) == 0.0:
            rep["degenerate"] = True
            rep["degenerate_note"] = ("zero saliency everywhere — the "
                                      "readout does not depend on the input "
                                      "(dead head or zero direction); the "
                                      "map is NOT normalised into fake "
                                      "uniformity and no PNG is written")
        else:
            rep["degenerate"] = False
            mean_map = sal.sum(dim=1).mean(dim=0)          # [H, W']
            torch.save({"mean_map_hw": mean_map,
                        "per_frame_map": sal.mean(dim=0)},
                       out_dir / f"saliency_{t.name}.pt")
            if a.png:
                write_png_gray(out_dir / f"saliency_{t.name}.png", mean_map)
            rep["artifacts"] = [f"saliency_{t.name}.pt"] + (
                [f"saliency_{t.name}.png"] if a.png else [])
        if t.family == "lead-state" and lead_x is not None:
            rep["lead_fov_groups"] = lead_groups(
                sal.sum(dim=(1, 2, 3)),
                sal.sum(dim=(2, 3)).double()[:, -1]
                / sal.sum(dim=(1, 2, 3)).double().clamp_min(1e-30),
                torch.as_tensor(lead_x).float()[: sal.shape[0]],
                torch.as_tensor(lead_y).float()[: sal.shape[0]], half)
        elif t.family == "lead-state":
            rep["lead_fov_groups"] = {
                "n": 0, "reason": "no lead meta in the windows blob "
                                  "(lead_x_m/lead_y_m) — grouping not "
                                  "computable, stated"}
        per_target[t.name] = rep
    # ---- summary -----------------------------------------------------------
    summary = {
        "instrument": "P9 probe-gradient saliency (F-19)",
        "tier": "T0-DIAGNOSTIC BY DESIGN — representation interrogation; "
                "NEVER driving performance (EVAL_DOCTRINE.md)",
        "quotable": ("NONE — synthetic smoke (random weights, random frames)"
                     if a.synthetic else
                     "maps + energy splits, at this checkpoint only"),
        "_evidence_class": ("MEASURED (ours; synthetic smoke — machinery "
                            "only)" if a.synthetic
                            else "MEASURED (ours; this checkpoint + windows)"),
        "checkpoint": meta,
        "windows": {"source": source, "n": int(frames.shape[0]),
                    "shape": list(frames.shape)},
        "frame_geometry": {"height": h, "width": w_px,
                           "hfov_deg": a.frame_hfov,
                           "projection": a.projection,
                           "f_ref": float(frame_geom.f_ref)},
        "fov_mask_half_angle_deg": a.mask_half_angle_deg,
        "targets": per_target,
        "not_computable": refusals,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "p9_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[p9] {len(per_target)} targets, {len(refusals)} declared "
          f"not-computable -> {out_dir / 'p9_summary.json'} "
          f"({summary['elapsed_s']} s)", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None,
                    help="v6 staged ckpt.pt or fp16 snapshot")
    ap.add_argument("--windows-pt", default=None,
                    help="{'frames': [N,W,C,H,W'], 'v0'?, 'lead_x_m'?, "
                         "'lead_y_m'?}")
    ap.add_argument("--synthetic", action="store_true",
                    help="CPU smoke: tiny random stack + random frames. "
                         "Stamps quotable: NONE.")
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-hfov", type=float, default=120.0,
                    help="hfov of the frames THE ENCODER WAS FED (the "
                         "P4 encoder_frame_rule: the sub-frame's field, not "
                         "the sensor's)")
    ap.add_argument("--projection", choices=("pinhole", "cylindrical"),
                    default="cylindrical")
    ap.add_argument("--mask-half-angle-deg", type=float,
                    default=FOV_MASK_DEFAULT_HALF_ANGLE_DEG,
                    help="fov_mask predicate half-angle (default = "
                         "bev_raster.fov_mask's own 60 deg)")
    ap.add_argument("--probe-arrays", default=None,
                    help="probe_latent_state.py dump with full-set ridge "
                         "directions (adds ridge_* targets incl. lead_gap)")
    ap.add_argument("--accept-probe-family", action="store_true",
                    help="override the probe-family refusal AFTER checking "
                         "the dump was fit on THIS checkpoint's latents")
    ap.add_argument("--png", action="store_true", default=True)
    ap.add_argument("--no-png", dest="png", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> dict:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
