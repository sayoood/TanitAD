"""W3 = P3+P6 — the counterfactual action-response probe pack on the FROZEN
v5f trunk (WM_PHYSICS_PROOF.md P3 + P6; V58F_FUSION.md wedge W3: "is the
co-trained trunk more action-controllable than v1arch's?").

WHAT IT DOES. For every window of the canonical eval grid (episodes < 40,
t % 8 == 0 — the 881-window grid every W4-family eval uses), the predictor is
rolled ``--k`` = 10 steps (1 s @ 10 Hz) under SIX action sequences:

  (a) FACTUAL   — the recorded actions (the same roll the canary scores);
  (b) LEFT      — +``--dkappa`` curvature on the steer channel;
  (c) RIGHT     — −``--dkappa``;
  (d) BRAKE     — −``--daccel`` m/s² on the accel channel;
  (e) THROTTLE  — +``--daccel``;
  (f) HOLD      — repeat the last recorded action (zero-order-hold; identical
                  by construction to ``rollout_transitions(..., None, k)`` —
                  pinned in tests/test_stage_a.py).

Each rolled latent sequence is decoded to waypoints via the grounding readout,
and P3 (sign correctness + response gain vs the unicycle-analytic prediction)
and P6 (latent-delta subspace concentration) are scored. No training happens
here; nothing is updated (trunk md5-checksummed before/after).

⭐ ACTION ENCODING (resolved from code, not assumed — the counterfactuals are
meaningless without it):
  * ``actions[..., 0]`` = **steer_road_rad** = ``atan(WHEELBASE * curvature)``
    with the LEGACY constant wheelbase **2.9 m** (physicalai.py:143
    ``CORPUS_META``; the mint is physicalai.py:621 ``steer =
    np.arctan(float(wheelbase) * curv)``; ``DEFAULT_WHEELBASE_MODE =
    "const2p9"`` physicalai.py:69/82 — every published cache is the legacy
    regime). A "+Δκ" counterfactual therefore maps through the encoding:
    ``steer_cf = atan(WB * (tan(steer)/WB + Δκ))`` — applied per step, NOT a
    raw additive offset on the steer channel.
  * ``actions[..., 1]`` = **accel m/s²** (the dataset's own longitudinal
    ``ax`` — physicalai.py:604-606); brake/throttle are plain ∓/± offsets.
  * The predictor consumes THREE channels: the 2 recorded channels plus
    ``v0/SPEED_SCALE`` appended constant along the horizon — the exact
    ``canary_rollout`` lift (train_flagship_v4.py:578-580), imported here as
    ``train_p8_occupancy.lift_actions3`` (not re-implemented). ⚠️ Stated
    limitation: the speed channel stays at the OBSERVED v0 under every
    counterfactual, because that is the trunk's own training/rollout contract
    (the canary holds it constant along the factual roll too); the accel
    counterfactuals therefore probe the accel CHANNEL, not a re-simulated
    speed input.
  * Only the LAST window action (the one acting from the present frame) and
    the future actions are perturbed. Earlier window actions condition on the
    OBSERVED past — perturbing them would ask the model to re-explain history,
    not respond to a counterfactual from now.

⭐ THE ROLL AND THE DECODE (cited, byte-reused):
  * roll: ``tanitad.models.metric_dynamics.rollout_transitions``
    (metric_dynamics.py:247-266) — the SAME latents-out roll P8 reuses;
    ``trans[k-1][1]`` is ẑ_{t+k}.
  * decode: ``decode_transitions(grounding.step["op"], trans, k)``
    (metric_dynamics.py:269-280) — pinned in its own docstring to reproduce
    ``rollout_decode`` exactly, which is the decode call ``canary_rollout``
    makes (train_flagship_v4.py:584-586: ``rollout_decode(world.predictor,
    states, aw, fa, step_readout, k_max)`` with ``step_readout =
    grounding.step["op"]``). Same latents, same readout, same SE(2)
    accumulation (``accumulate_se2``, metric_dynamics.py:114-139). Ego frame
    +x forward, +y LEFT (refb_labels.ego_frame convention).

P3 METRICS (per channel, at k = ``--k`` i.e. 1 s):
  * SIGN-CORRECTNESS — LEFT: decoded endpoint displaces to +y of the factual
    endpoint; RIGHT: −y; BRAKE: endpoint forward distance x SHORTER than
    factual (the ego travels less — the gap to the factual position grows);
    THROTTLE: longer. Scored over ADMISSIBLE windows only: a window whose
    unicycle-ANALYTIC response to the same Δ is < ``EPS_RESP`` (standstill,
    or a Δ the kinematics cannot express) has no defined sign and is
    excluded+counted, never scored.
  * RESPONSE GAIN — |Δ decoded displacement| / |Δ unicycle-analytic
    displacement| for the SAME action delta, lateral (y) for steer channels,
    longitudinal (x) for accel channels. The analytic reference is
    ``train_v58f_unicycle_head.unicycle_rollout`` (IMPORTED — the W4
    discretisation convention verbatim: dx_k = v_k·dt pre-update speed,
    dyaw_k = κ_k·v_k·dt, translate-then-turn via accumulate_se2, v updated
    after the row, clamped ≥ 0), fed a_k = accel channel, κ_k =
    tan(steer)/WHEELBASE per rolled step.
  * HOLD carries no sign/gain gate (there is no signed expectation for
    "repeat the last action"); its endpoint delta vs factual is reported as a
    smoothness diagnostic only.

⛔ PRE-REGISTERED GATES (WM_PHYSICS_PROOF.md P3/P6, committed before any run;
written verbatim to ``<out>/w3_gate.json``):
  * P3-sign: sign-correctness ≥ 95 % of admissible windows, per channel
    (left/right/brake/throttle);
  * P3-gain: median response gain ∈ [0.5, 2.0] per channel;
  * P6-dim:  stacking ALL counterfactual latent deltas (ẑ_cf − ẑ_fact at
    k = ``--k``, all five counterfactuals), the PCA dim count capturing 80 %
    of variance is ≤ 32; the energy fraction each channel's deltas place
    inside that subspace is reported alongside.
  * P6 scene-invariance (P1-style lead-gap probe under ego perturbation) is
    NOT computable here — it lands with the P1/P2 probe harness
    (WM_PHYSICS_PROOF.md build order) — and is emitted as n/a with the
    reason, never faked.

P6 PCA is UNCENTERED (stated choice): a latent delta has a meaningful zero —
"no action change ⇒ no latent change" — so uncentered principal axes measure
action-induced energy directly; centering would subtract the mean response
and misstate the subspace a controller would actually steer through.

TIER STAMP: **T1-diagnostic** — the actions are COUNTERFACTUAL, so this
probes the WM roll under hypothetical actions (the world model's
controllability), NOT driving performance; it is one row of the WM-physics
battery, never quotable as a driving number (EVAL_DOCTRINE.md: T0 = teacher-
forced diagnostic, T1 = action-closed loop; this is the diagnostic face of
T1 — the model conditioned on non-recorded actions — without a task metric).

ESTIMATOR: point estimates over the corpus grid; per-window deltas + episode
uids are BANKED to ``<out>/w3_windows.npz`` so the registry-grade
episode-cluster bootstrap (taniteval/ci.py, over the 40 val episodes) can be
run pod-side before any registry claim. This JSON alone is not decision-grade
for close calls.

⚠️ POD-SIDE ONLY for the full path (GPU + v5f checkpoint + v2 val corpus).
Runnable here: ``python -m py_compile`` and the CPU tests
(``stack/tests/test_stage_a.py``).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or trainers die
with ModuleNotFound: tanitad):

  python3 scripts/stage_a_probes.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --out /workspace/experiments/w3-stage-a-probes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

DT = 0.1                     # 10 Hz tick — the dense-horizon contract
K_DEFAULT = 10               # 1 s @ 10 Hz — the P3 probe horizon
WHEELBASE = 2.9              # LEGACY const2p9 — the regime every published
                             # cache was minted under (physicalai.py:63/69/82)
DKAPPA_DEFAULT = 0.02        # 1/m — ~0.2 rad/s yaw rate at 10 m/s; well inside
                             # the corpus curvature range (KAPPA_MAX 0.2)
DACCEL_DEFAULT = 2.0         # m/s² — half the W4 A_MAX bound; a firm but
                             # in-distribution brake/throttle step
EPS_RESP = 1e-2              # m — admissibility floor on the ANALYTIC response
GATE_SIGN = 0.95             # P3: sign-correctness ≥ 95 % per channel
GATE_GAIN = (0.5, 2.0)       # P3: median gain band vs unicycle-analytic
GATE_P6_DIMS = 32            # P6: ≤ 32 dims capture 80 % of delta variance
P6_VAR_TARGET = 0.80

CF_CHANNELS = ("left", "right", "brake", "throttle", "hold")
GATED_CHANNELS = ("left", "right", "brake", "throttle")
# (expected endpoint-delta sign, axis) — axis 0 = x fwd, 1 = y left; ego frame
# +x fwd / +y LEFT (refb_labels.ego_frame). Brake ⇒ shorter forward travel.
CHANNEL_SIGN_AXIS = {"left": (+1.0, 1), "right": (-1.0, 1),
                     "brake": (-1.0, 0), "throttle": (+1.0, 0)}


# ============================================================================
# the steer encoding — steer_road_rad = atan(WHEELBASE * kappa)
# ============================================================================
def kappa_of_steer(steer: Tensor, wheelbase: float = WHEELBASE) -> Tensor:
    """Invert the corpus steer encoding (physicalai.py:621):
    ``kappa = tan(steer) / wheelbase`` [1/m]."""
    return torch.tan(steer) / wheelbase


def steer_of_kappa(kappa: Tensor, wheelbase: float = WHEELBASE) -> Tensor:
    """The corpus steer encoding (physicalai.py:621):
    ``steer = atan(wheelbase * kappa)`` [rad]."""
    return torch.atan(wheelbase * kappa)


# ============================================================================
# counterfactual construction (pure, CPU-tested)
# ============================================================================
def apply_counterfactual(aw2: Tensor, fa2: Tensor, channel: str, *,
                         dkappa: float = DKAPPA_DEFAULT,
                         daccel: float = DACCEL_DEFAULT,
                         wheelbase: float = WHEELBASE
                         ) -> tuple[Tensor, Tensor]:
    """Recorded 2-channel actions -> the counterfactual pair (cloned).

    ``aw2`` [B, W, 2] window actions (steer_road_rad, accel_mps2), ``fa2``
    [B, H, 2] future actions. Perturbs ONLY ``aw2[:, -1]`` (the action acting
    from the present frame — earlier window actions condition on observed
    history) and every future step. Steer deltas are applied IN KAPPA SPACE
    through the encoding (module docstring): ``steer_cf = atan(WB *
    (tan(steer)/WB + Δκ))``. ``hold`` repeats the last recorded action —
    identical to ``rollout_transitions``' zero-order-hold branch.
    """
    if channel not in CF_CHANNELS:
        raise ValueError(f"unknown counterfactual channel {channel!r}; "
                         f"expected one of {CF_CHANNELS}")
    aw = aw2.clone()
    fa = fa2.clone()
    if channel == "hold":
        fa = aw2[:, -1:, :].expand_as(fa2).clone()
        return aw, fa
    if channel in ("left", "right"):
        dk = dkappa if channel == "left" else -dkappa
        aw[:, -1, 0] = steer_of_kappa(
            kappa_of_steer(aw2[:, -1, 0], wheelbase) + dk, wheelbase)
        fa[:, :, 0] = steer_of_kappa(
            kappa_of_steer(fa2[:, :, 0], wheelbase) + dk, wheelbase)
    else:
        da = daccel if channel == "throttle" else -daccel
        aw[:, -1, 1] = aw2[:, -1, 1] + da
        fa[:, :, 1] = fa2[:, :, 1] + da
    return aw, fa


def rolled_action_sequence(aw2: Tensor, fa2: Tensor, k: int
                           ) -> tuple[Tensor, Tensor]:
    """The (a, kappa) control sequences the roll actually consumes, [B, k].

    Step 0 is ``aw2[:, -1]`` (rollout_transitions predicts z_{t+1} from the
    window whose last action is the present one); steps 1..k-1 are
    ``fa2[:, :k-1]`` (metric_dynamics.py:258-261). Returns
    ``(a_seq, kappa_seq)`` with kappa through the steer encoding."""
    if fa2.shape[1] < k - 1:
        raise ValueError(f"future_actions horizon {fa2.shape[1]} < k-1="
                         f"{k - 1} — the roll would run out of actions")
    steer = torch.cat([aw2[:, -1:, 0], fa2[:, :k - 1, 0]], dim=1)
    accel = torch.cat([aw2[:, -1:, 1], fa2[:, :k - 1, 1]], dim=1)
    return accel, kappa_of_steer(steer)


def analytic_endpoints(aw2: Tensor, fa2: Tensor, v0: Tensor, k: int) -> Tensor:
    """Unicycle-analytic waypoints [B, k, 2] for the rolled action sequence —
    ``train_v58f_unicycle_head.unicycle_rollout`` (IMPORTED: the W4
    discretisation, accumulate_se2 composition) at N=1 candidate."""
    from train_v58f_unicycle_head import unicycle_rollout
    a_seq, kappa_seq = rolled_action_sequence(aw2, fa2, k)
    wp, _v = unicycle_rollout(a_seq[:, None, :], kappa_seq[:, None, :],
                              v0, dt=DT)
    return wp[:, 0]                                       # [B, k, 2]


# ============================================================================
# P3 statistics (pure, CPU-tested)
# ============================================================================
def channel_stats(d_wm: np.ndarray, d_an: np.ndarray, expected_sign: float,
                  eps: float = EPS_RESP) -> dict:
    """Sign rate + gain stats for one channel.

    ``d_wm``/``d_an`` [n] — decoded / analytic endpoint deltas on the
    channel's axis (signed). Windows with ``|d_an| < eps`` are inadmissible
    (no analytic response ⇒ no defined sign, gain undefined): excluded and
    counted. Gain = |d_wm| / |d_an| over admissible windows."""
    d_wm = np.asarray(d_wm, dtype=np.float64)
    d_an = np.asarray(d_an, dtype=np.float64)
    if d_wm.shape != d_an.shape:
        raise ValueError(f"shape mismatch {d_wm.shape} vs {d_an.shape}")
    adm = np.abs(d_an) >= eps
    n_adm = int(adm.sum())
    out = {"n_grid": int(d_wm.shape[0]), "n_admissible": n_adm,
           "n_excluded_no_analytic_response": int(d_wm.shape[0]) - n_adm,
           "admissibility_eps_m": eps}
    if n_adm == 0:
        out.update(sign_rate=None, gain_median=None, gain_p25=None,
                   gain_p75=None)
        return out
    dw, da = d_wm[adm], d_an[adm]
    out["sign_rate"] = float(np.mean(dw * expected_sign > 0.0))
    gain = np.abs(dw) / np.abs(da)
    out["gain_median"] = float(np.median(gain))
    out["gain_p25"] = float(np.percentile(gain, 25))
    out["gain_p75"] = float(np.percentile(gain, 75))
    return out


# ============================================================================
# P6 statistics (pure, CPU-tested)
# ============================================================================
def pca_subspace_stats(deltas: dict[str, np.ndarray],
                       var_target: float = P6_VAR_TARGET) -> dict:
    """Uncentered PCA over ALL stacked counterfactual latent deltas.

    ``deltas``: channel -> [n, S] (ẑ_cf − ẑ_fact at the probe horizon).
    Returns dims capturing ``var_target`` of total energy, the per-channel
    energy fraction inside that subspace, and the normalised head of the
    spectrum. UNCENTERED by design (module docstring: a delta has a
    meaningful zero)."""
    mats = [np.asarray(deltas[c], dtype=np.float64) for c in deltas]
    if not mats or sum(m.shape[0] for m in mats) == 0:
        return {"computable": False, "reason": "no latent deltas collected"}
    stack = np.concatenate(mats, axis=0)                  # [N_rows, S]
    total = float((stack ** 2).sum())
    if total <= 0.0:
        return {"computable": False,
                "reason": "all latent deltas are exactly zero — the "
                          "predictor ignores the action channel entirely "
                          "(that is a P3 catastrophic fail, not a subspace)"}
    _u, s, vt = np.linalg.svd(stack, full_matrices=False)
    energy = s ** 2
    cum = np.cumsum(energy) / energy.sum()
    dims = int(np.searchsorted(cum, var_target) + 1)
    basis = vt[:dims]                                     # [dims, S]
    frac = {}
    for c, m in zip(deltas, mats):
        e = float((m ** 2).sum())
        frac[c] = (float(((m @ basis.T) ** 2).sum() / e) if e > 0 else None)
    return {"computable": True, "n_rows": int(stack.shape[0]),
            "latent_dim": int(stack.shape[1]),
            "var_target": var_target, "dims_for_var_target": dims,
            "energy_fraction_in_subspace_per_channel": frac,
            "centering": "uncentered (deltas have a meaningful zero — "
                         "stated design choice, module docstring)",
            "spectrum_head_normalised":
                [round(float(x), 6) for x in (energy / energy.sum())[:40]]}


# ============================================================================
# the gate JSON (pure, both/all branches CPU-tested)
# ============================================================================
def _agg(passes: list) -> bool | None:
    if any(p is False for p in passes):
        return False
    if any(p is None for p in passes):
        return None
    return True


def w3_gate_dict(per_channel: dict[str, dict], p6: dict, *,
                 hold: dict | None = None,
                 sign_gate: float = GATE_SIGN,
                 gain_band: tuple[float, float] = GATE_GAIN,
                 p6_dim_gate: int = GATE_P6_DIMS) -> dict:
    """Assemble the pre-registered verdicts. Missing/empty channels yield
    ``pass: None`` (not computable) — never a fake verdict."""
    channels = {}
    passes = []
    for c in GATED_CHANNELS:
        st = dict(per_channel.get(c) or {})
        sr = st.get("sign_rate")
        gm = st.get("gain_median")
        sp = None if sr is None else bool(sr >= sign_gate)
        gp = None if gm is None else bool(gain_band[0] <= gm <= gain_band[1])
        st["sign_gate"] = {"rule": f"sign-correctness >= {sign_gate} of "
                                   f"admissible windows", "pass": sp}
        st["gain_gate"] = {"rule": f"median response gain in "
                                   f"[{gain_band[0]}, {gain_band[1]}] vs "
                                   f"unicycle-analytic at 1 s", "pass": gp}
        channels[c] = st
        passes += [sp, gp]
    if hold is not None:
        channels["hold"] = {**hold,
                            "note": "no sign/gain gate — 'repeat last "
                                    "action' has no signed expectation; "
                                    "diagnostic only"}
    if p6.get("computable"):
        p6p = bool(p6["dims_for_var_target"] <= p6_dim_gate)
    else:
        p6p = None
    p6_out = {**p6, "gate": {"rule": f"dims capturing "
                                     f"{int(P6_VAR_TARGET * 100)} % of "
                                     f"stacked counterfactual latent-delta "
                                     f"variance <= {p6_dim_gate}",
                             "pass": p6p}}
    passes.append(p6p)
    return {"channels": channels, "p6": p6_out, "PASS": _agg(passes)}


# ============================================================================
# frozen forward — encode once, roll six action arms, decode with the canary's
# readout (POD-SIDE; needs GPU + checkpoint + corpus)
# ============================================================================
@torch.no_grad()
def probe_batch(world, grounding, batch: dict, k: int, *, amp_on: bool,
                dkappa: float, daccel: float) -> dict:
    """One batch -> per-arm decoded endpoints + latent deltas.

    Returns ``{"wp": {arm: [B, k, 2] cpu}, "z_delta": {cf: [B, S] cpu},
    "v0": [B]}`` where arm ranges over ("factual",) + CF_CHANNELS.
    Roll = ``rollout_transitions`` (metric_dynamics.py:247-266); decode =
    ``decode_transitions(grounding.step["op"], ...)`` (metric_dynamics.py:
    269-280) — the canary decode (module docstring)."""
    from train_p8_occupancy import lift_actions3            # canary lift seam
    from tanitad.models.metric_dynamics import (decode_transitions,
                                                rollout_transitions)
    frames = batch["frames"]
    aw2 = batch["actions"].float()
    fa2 = batch["future_actions"].float()
    v0 = batch["pose_last"][:, 3].float()
    step_readout = grounding.step["op"]
    dev_type = frames.device.type
    out_wp: dict[str, Tensor] = {}
    out_zd: dict[str, Tensor] = {}
    with torch.autocast(dev_type, dtype=torch.bfloat16,
                        enabled=amp_on and dev_type == "cuda"):
        states = world.encode_window(frames)                # [B, W, S]
        z_fact = None
        for arm in ("factual",) + CF_CHANNELS:
            if arm == "factual":
                a2, f2 = aw2, fa2
            else:
                a2, f2 = apply_counterfactual(aw2, fa2, arm, dkappa=dkappa,
                                              daccel=daccel)
            aw3, fa3 = lift_actions3(a2, f2, v0)
            trans = rollout_transitions(world.predictor, states, aw3, fa3, k)
            wp, _dpose = decode_transitions(step_readout, trans, k)
            out_wp[arm] = wp.float().cpu()
            z_k = trans[k - 1][1].float()
            if arm == "factual":
                z_fact = z_k
            else:
                out_zd[arm] = (z_k - z_fact).cpu()
    return {"wp": out_wp, "z_delta": out_zd, "v0": v0.cpu()}


# ============================================================================
# main (POD-SIDE)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("stage_a_probes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint with model+grounding keys (v1/v4/v5f all "
                         "qualify — MODE A load, planner head unused)")
    # corpus seams — the W4-family eval arg surface (val split only; the probe
    # trains nothing)
    ap.add_argument("--v2-val-cache", required=True, nargs="+")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # probe knobs (defaults are the pre-registered ones)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K_DEFAULT,
                    help="roll length in 0.1 s ticks (10 = the 1 s P3 horizon)")
    ap.add_argument("--dkappa", type=float, default=DKAPPA_DEFAULT,
                    help="counterfactual curvature delta, 1/m (applied through "
                         "the steer encoding — module docstring)")
    ap.add_argument("--daccel", type=float, default=DACCEL_DEFAULT,
                    help="counterfactual accel delta, m/s²")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    # the canonical eval grid (881 windows at the defaults)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[w3] WARNING: cuda unavailable, falling back to cpu", flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v1_from_ck, resolve_eval_frames)
    from train_flagship4b import FlagshipWindowDataset
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import module_md5

    # ---- geometry FIRST (the W4/eval seam, not re-resolved here) ------------
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg,
                                                   label="stage_a_probes")
    plan = _plan(cfg)
    if plan.max_horizon < a.k:
        raise SystemExit(f"[w3] --k {a.k} > plan.max_horizon "
                         f"{plan.max_horizon} — future_actions cannot cover it")

    # ---- frozen trunk: MODE A (model + grounding; no planner head) ----------
    print(f"[w3] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # v6 ({"stack": …}) rebuilds a V6Stack behind the v5 trunk interface; v5
    # takes the byte-identical old path. See tanitad/eval/v6_probe_trunk.
    from tanitad.eval.v6_probe_trunk import is_v6_checkpoint, load_trunk_auto
    is_v6 = is_v6_checkpoint(ck)
    world, grounding, base_step = load_trunk_auto(
        ck, device, ckpt_path=a.ckpt, frame=model_frame)
    del ck
    assert not any(p.requires_grad for p in world.parameters())
    # ⚠️ v6 has no separate grounding module (its metric decode lives inside the
    # stack as step_readout_op), so the md5 covers whatever modules exist —
    # never `module_md5(world, None)`, which would crash on named_parameters().
    md5_before = module_md5(*[m for m in (world, grounding) if m is not None])
    print(f"[w3] trunk frozen ({'v6' if is_v6 else 'v5'}) · base step "
          f"{base_step} · state_dim {world.state_dim} · "
          f"md5 {md5_before[:12]}", flush=True)

    # ---- val data (the W4-family loader seam, imported) ---------------------
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    # ⛔ The window is the TRUNK's, not the v5 eval default. MEASURED
    # 2026-08-14: v6's predictor is configured for window 6 while
    # `_eval_cfg()` says 8, and the mismatch surfaced as a ValueError deep
    # inside the predictor. Same class as the geometry seam above: a
    # property of the checkpoint must be read from the checkpoint.
    ds_val = FlagshipWindowDataset(val_eps,
                                   window=getattr(world, "window",
                                                  cfg.predictor.window),
                                   max_horizon=plan.max_horizon,
                                   maneuver_h=plan.maneuver_h,
                                   channels=cfg.encoder.in_channels)
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < a.episodes and t % a.stride == 0]
    if not sel:
        raise SystemExit("[w3] eval grid selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    print(f"[w3] val {len(val_eps)} eps / {len(ds_val)} windows; probe grid "
          f"{len(sel)} windows (episodes<{a.episodes}, stride {a.stride})",
          flush=True)

    # ---- the six-arm roll over the grid -------------------------------------
    t0 = time.time()
    d_wm = {c: [] for c in GATED_CHANNELS}
    d_an = {c: [] for c in GATED_CHANNELS}
    hold_norm: list[float] = []
    z_deltas = {c: [] for c in CF_CHANNELS}
    ep_uids: list[int] = []
    v0_all: list[float] = []
    for b0 in range(0, len(sel), a.batch):
        idx = sel[b0:b0 + a.batch]
        for i in idx:
            e_i, _t = ds_val.index[i]
            ep_uids.append(int(ds_val.episodes[e_i].episode_id))
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        res = probe_batch(world, grounding, b, a.k, amp_on=amp_on,
                          dkappa=a.dkappa, daccel=a.daccel)
        v0_all.extend(res["v0"].tolist())
        aw2 = b["actions"].float().cpu()
        fa2 = b["future_actions"].float().cpu()
        v0 = b["pose_last"][:, 3].float().cpu()
        an_fact = analytic_endpoints(aw2, fa2, v0, a.k)
        wp_fact = res["wp"]["factual"]
        for c in GATED_CHANNELS:
            _sign, axis = CHANNEL_SIGN_AXIS[c]
            a2c, f2c = apply_counterfactual(aw2, fa2, c, dkappa=a.dkappa,
                                            daccel=a.daccel)
            an_cf = analytic_endpoints(a2c, f2c, v0, a.k)
            d_an[c].append((an_cf[:, a.k - 1, axis]
                            - an_fact[:, a.k - 1, axis]).numpy())
            d_wm[c].append((res["wp"][c][:, a.k - 1, axis]
                            - wp_fact[:, a.k - 1, axis]).numpy())
        hold_norm.extend((res["wp"]["hold"][:, a.k - 1]
                          - wp_fact[:, a.k - 1]).norm(dim=-1).tolist())
        for c in CF_CHANNELS:
            z_deltas[c].append(res["z_delta"][c].numpy())
        done = min(b0 + a.batch, len(sel))
        if (b0 // a.batch) % 8 == 0:
            print(f"[w3] {done}/{len(sel)} windows "
                  f"({time.time() - t0:.0f} s)", flush=True)

    md5_after = module_md5(*[m for m in (world, grounding) if m is not None])

    # ---- P3 + P6 ------------------------------------------------------------
    per_channel = {}
    cf_delta = {"left": f"+{a.dkappa} 1/m kappa (through the steer encoding)",
                "right": f"-{a.dkappa} 1/m kappa (through the steer encoding)",
                "brake": f"-{a.daccel} m/s^2 accel",
                "throttle": f"+{a.daccel} m/s^2 accel"}
    for c in GATED_CHANNELS:
        sign, _axis = CHANNEL_SIGN_AXIS[c]
        per_channel[c] = channel_stats(np.concatenate(d_wm[c]),
                                       np.concatenate(d_an[c]), sign)
        per_channel[c]["kind"] = ("steer" if c in ("left", "right")
                                  else "accel")
        per_channel[c]["delta"] = cf_delta[c]
    hold = {"n_grid": len(hold_norm),
            "endpoint_delta_median_m": float(np.median(hold_norm)),
            "endpoint_delta_p90_m": float(np.percentile(hold_norm, 90))}
    zd = {c: np.concatenate(z_deltas[c]) for c in CF_CHANNELS}
    p6 = pca_subspace_stats(zd)
    p6["k"] = a.k
    p6["scene_invariance"] = {
        "available": False,
        "reason": "the P1-style lead-gap probe is not built yet — it lands "
                  "with the P1/P2 probe harness (WM_PHYSICS_PROOF.md build "
                  "order: P7 -> P1/P2 -> P3/P6). Emitting n/a rather than a "
                  "fake invariance number."}
    gate = w3_gate_dict(per_channel, p6, hold=hold)

    # ---- bank per-window arrays (for the pod-side cluster bootstrap) --------
    np.savez_compressed(
        os.path.join(a.out, "w3_windows.npz"),
        episode_uid=np.asarray(ep_uids, dtype=np.int64),
        v0=np.asarray(v0_all, dtype=np.float32),
        hold_endpoint_delta=np.asarray(hold_norm, dtype=np.float32),
        **{f"d_wm_{c}": np.concatenate(d_wm[c]).astype(np.float32)
           for c in GATED_CHANNELS},
        **{f"d_an_{c}": np.concatenate(d_an[c]).astype(np.float32)
           for c in GATED_CHANNELS})

    summary = {
        "probe": "W3 stage-A counterfactual action-response pack "
                 "(WM_PHYSICS_PROOF.md P3+P6; V58F_FUSION.md W3)",
        "tier": "T1-diagnostic — actions are COUNTERFACTUAL: this probes the "
                "WM roll under hypothetical actions (controllability), NOT "
                "driving performance; never quotable as a driving number "
                "(EVAL_DOCTRINE.md)",
        **gate,
        "roll": {"k": a.k, "seconds": a.k * DT,
                 "fn": "tanitad.models.metric_dynamics.rollout_transitions "
                       "(metric_dynamics.py:247-266)",
                 "decode": "decode_transitions(grounding.step['op'], ...) "
                           "(metric_dynamics.py:269-280) — pinned twin of "
                           "the canary's rollout_decode "
                           "(train_flagship_v4.py:584-586)",
                 "action_lift": "train_p8_occupancy.lift_actions3 — the "
                                "canary 3-channel speed-append "
                                "(train_flagship_v4.py:578-580); speed "
                                "channel held at observed v0 under every "
                                "arm (the trunk's rollout contract)"},
        "action_encoding": {
            "channels": ["steer_road_rad = atan(2.9 * kappa) "
                         "(physicalai.py:621, legacy const2p9 regime "
                         "physicalai.py:69/82)",
                         "accel m/s^2 (dataset ax, physicalai.py:604-606)"],
            "wheelbase_m": WHEELBASE,
            "dkappa_1_per_m": a.dkappa, "daccel_mps2": a.daccel,
            "perturbed": "last window action + all future actions "
                         "(history untouched — module docstring)"},
        "analytic_reference": "train_v58f_unicycle_head.unicycle_rollout "
                              "(IMPORTED — the W4 discretisation via "
                              "accumulate_se2), controls a=accel channel, "
                              "kappa=tan(steer)/2.9 per rolled step",
        "grid": {"episodes": a.episodes, "stride": a.stride,
                 "n_windows": len(sel), "n_val_episodes": len(val_eps)},
        "trunk_frozen_proof": {"md5_before": md5_before,
                               "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "base_ckpt": a.ckpt, "base_step": base_step,
        "wall_s": round(time.time() - t0, 1),
        "_estimator_note": ("point estimates over the corpus grid; "
                            "per-window deltas + episode uids banked to "
                            "w3_windows.npz — the DECISION-grade interval "
                            "for any registry claim is the episode-cluster "
                            "bootstrap (taniteval/ci.py) over the 40 val "
                            "episodes; run it pod-side before publishing"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON + "
                           "w3_windows.npz)",
    }
    with open(os.path.join(a.out, "w3_gate.json"), "w") as gf:
        json.dump(summary, gf, indent=1)
    print(f"\n[W3 SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    if not summary["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK/GROUNDING CHANGED DURING THE PROBE — "
                         "run invalid")
    verdict = ("PASS" if gate["PASS"] else
               "FAIL" if gate["PASS"] is not None else "NOT COMPUTABLE")
    print(f"[W3 GATE] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
