"""ACTCONS (`--w-act-cons W`, `--act-cons-max M`; card PREREG_V8F_PROOF_ADD37 section 2): a kinematic consistency term on the level-1 imagination.

A TRAINING-ONLY term. It adds no module, no parameter, no state-dict key and no config field: the model and the checkpoint are exactly the base tree's; the trainer calls :func:`act_cons_term` after
``model.loss`` and adds ``W * term`` to the total (the same place and style as the RULEOWN term). Nothing runs at inference.

What the term teaches. With g the lead gap, T a horizon and d an offset of the ego acceleration, for a lead that keeps its motion  g_T(do(a + d)) = g_T(do(a)) - 0.5 d T^2.
For each valid (sample, start frame s) the level-1 predictor is rolled out 10 steps from the DETACHED state of frame s twice, with the same function the rollout loss uses (``HJEPAStack.roll_pairs``):
under the logged actions (no gradient: the reference) and under the logged actions with ONE offset d [m/s2] per sample added to the acceleration channel at every step (with gradient). The
supervised lead head (its parameters detached, so this term reaches none of them) reads the gap (output x 50 m) from steps 1..5 (T = 1.0 s) and 6..10 (T = 2.0 s) of both rollouts;
term = mean over the valid (sample, start) pairs and the two times of | gap(offset) - gap(reference) + 0.5 d T^2 |, in metres. The gradient reaches the predictor and its action embedding only.
This is the training-side twin of the read ``imaghead`` (``imag_dump.py``: the same rollout core, the same head reads, the same offset units)."""
from __future__ import annotations

import torch
from torch import Tensor

from tanitad.models import v8f_mini as M

ACT_CONS_MAX_DEFAULT = 1.5                 # m/s2: the half-width of the offset range (card section 2)
ROLL_K = 10                                 # rollout length in steps of 0.2 s (card: 10 steps)
SPEED_MULT = 2.0                            # valid needs ego speed at frame s >= SPEED_MULT * max [m/s] (no standstill inside the 2.0 s rollout)
GAP_LO_M, GAP_HI_M = 0.0, 60.0              # valid needs a label gap in this range [m] (the label range of v8f_mini_data.lead_sup_label_arrays)
STEP_S = 0.2                                # one rollout step (the cache grid)
#: (first step, end step, horizon T [s]) as 0-based slices of the 10 rolled-out states: the head reads 5 consecutive states ending at step 5 (T = 1.0) and at step 10 (T = 2.0)
T_READS = ((0, 5, 1.0), (5, 10, 2.0))
LEAD_HIST_REQUIRED = 5                      # the head reads exactly the 5 states of one window; any other width has no 1.0 s / 2.0 s read
GEN_SALT = 0x41435443                       # "ACTC": domain separation of the dedicated offset generator from every other stream


def valid_starts(n_frames: int, k: int = ROLL_K) -> list:
    """The start set of the existing rollout loss for ``k`` steps (``HJEPAStack.roll_pairs``: ``range(0, T - k, 2)``): the same rule even in an arm without a rollout loss."""
    return list(range(0, int(n_frames) - int(k), 2))


def draw_offsets(seed: int, step: int, n: int, max_off: float) -> Tensor:
    """One offset d ~ U(-max, +max) [m/s2] per sample, float32 [n], from a DEDICATED generator seeded from the run seed and the step (a fresh generator per step: nothing is carried, a resumed run draws the
    same offsets, and the global RNG and every other draw of the trainer are untouched)."""
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + int(step)) ^ GEN_SALT)
    return (torch.rand(int(n), generator=g, dtype=torch.float32) * 2.0 - 1.0) * float(max_off)


def check_buildable(cfg, model) -> list:
    """The reasons this term cannot run on the BUILT model (empty = fine): it needs the supervised lead head (5-frame read), a grid state, level-1 action width 2 (acc, yaw), a rollout that fits the window, no dropout."""
    bad = []
    wm = getattr(model, "wm", None)
    if wm is None or getattr(wm, "lead_head", None) is None:
        bad.append("act_cons needs the supervised lead head (--lead-sup on): the term reads the gap through it, so with no head it is a lever that is declared and does nothing")
        return bad
    if cfg.wm_state != "grid":
        bad.append("act_cons needs --wm-state grid")
    if wm.l1_act != "acc_yaw":
        bad.append(f"act_cons needs the level-1 action width 2 (--wm-l1-act acc_yaw), the model has {wm.l1_act!r}: the offset is added to the acceleration channel")
    hist = int(wm.lead_head.mlp[0].in_features // M.LEAD_SUP_FRAME_DIM)
    if hist != LEAD_HIST_REQUIRED:
        bad.append(f"act_cons needs --lead-hist {LEAD_HIST_REQUIRED} (reads of 5 steps at 1.0 s and 2.0 s), the head was built with {hist}")
    if not valid_starts(cfg.hmax):
        bad.append(f"act_cons needs a window of at least {ROLL_K + 1} frames (hmax {cfg.hmax}): no start frame lets {ROLL_K} steps fit")
    pd = float(getattr(wm.stack.cfg[0], "pred_dropout", 0.0))
    if pd != 0.0:
        bad.append(f"act_cons needs a predictor without dropout (pred_dropout {pd}): the reference and the offset rollout would use different masks")
    return bad


def build_record(w: float, max_off: float, cfg, model) -> dict:
    """What the term was BUILT as, read from the model and the config (the G-DVB 'built' side; ``info["act_cons"]``)."""
    wm = model.wm
    return {"w_act_cons": float(w), "act_cons_max": float(max_off), "k": ROLL_K, "starts": valid_starts(cfg.hmax), "reads": [list(r) for r in T_READS],
            "speed_bound_mps": float(max_off) * SPEED_MULT, "gap_range_m": [GAP_LO_M, GAP_HI_M], "acc_scale": float(M.WM_ACC_SCALE),
            "lead_hist": int(wm.lead_head.mlp[0].in_features // M.LEAD_SUP_FRAME_DIM), "l1_act": wm.l1_act,
            "gap_scale_m": float(M.LEAD_SUP_GAP_SCALE_M), "head_params_detached": True, "start_state_detached": True}


def valid_mask(batch: dict, starts: list, max_off: float) -> Tensor:
    """[B, S] bool: a label lead at frame s (presence, gap in [0, 60] m), a REAL frame s, and ego speed at frame s >= max * 2.0 m/s. NaN labels are never valid."""
    cols = []
    for s in starts:
        pres, gap = batch["lead_pres"][:, s].float(), batch["lead_gapm"][:, s].float()
        sp = batch["wm_speed"][:, s].float()
        cols.append((pres > 0.5) & torch.isfinite(gap) & (gap >= GAP_LO_M) & (gap <= GAP_HI_M) & batch["frame_ok"][:, s].bool() & (sp >= float(max_off) * SPEED_MULT))
    return torch.stack(cols, 1)


def _gap_m(head, params: dict, x: Tensor) -> Tensor:
    """x [S, B, 5, D] rolled-out (or reference) states -> [S, B] gap in metres at the end frame of the five, the head's PARAMETERS detached (gradient flows into x only)."""
    s, b = x.shape[:2]
    o = torch.func.functional_call(head, params, (x.reshape(s * b, x.shape[2], x.shape[3]),))
    return (o[:, 0, 1].float() * M.LEAD_SUP_GAP_SCALE_M).reshape(s, b)


def act_cons_term(model, adapt: Tensor, batch: dict, d: Tensor, max_off: float) -> dict:
    """The ACTCONS term on one batch. ``adapt`` [B, F, n_tok, d_model] (the loss's own adapter tokens), ``batch`` with act / wm_speed / frame_ok / lead_pres / lead_gapm, ``d`` [B] offsets [m/s2].
    Returns ``term`` (a scalar tensor in metres: exactly 0.0 with NO gradient when no sample is valid), ``n_valid`` (valid (sample, start) pairs, float tensor), ``n_rows`` (samples with at least one valid start).
    Everything is computed in float32 (autocast off) on the device of ``adapt``."""
    wm = model.wm
    st, head = wm.stack, wm.lead_head
    dev = adapt.device
    with torch.autocast(device_type=dev.type, enabled=False):
        with torch.no_grad():
            z = wm.z1(adapt.detach().float()).float()                   # [B, F, D] the START STATE without gradient: nothing before the predictor receives any from this term
        starts = valid_starts(z.shape[1])
        valid = valid_mask(batch, starts, max_off)                       # [B, S]
        rows = valid.any(1)
        n_valid = valid.sum().to(torch.float32)
        if not bool(rows.any()):
            return {"term": torch.zeros((), dtype=torch.float32, device=dev), "n_valid": n_valid, "n_rows": torch.zeros((), dtype=torch.float32, device=dev)}
        zr = z[rows]
        a = wm.scale_actions(batch["act"][rows].float())[:, :-1]         # [Br, F-1, 2] the WM's action units: acc / WM_ACC_SCALE (v8f_mini.py scale_actions), yaw / ch1_scale
        dr = d.to(dev).float()[rows]
        off = torch.stack([dr / float(M.WM_ACC_SCALE), torch.zeros_like(dr)], -1)[:, None, :]      # d / WM_ACC_SCALE on the acceleration channel, yaw unchanged
        with torch.no_grad():
            ref, _, _ = st.roll_pairs(zr, st.act_embed(a), ROLL_K)       # reference: logged actions, no gradient
        rolled, _, _ = st.roll_pairs(zr, st.act_embed(a + off), ROLL_K)  # offset actions, with gradient (predictor, action embedding)
        params = {n: p.detach() for n, p in head.named_parameters()}     # the head's parameters carry NO gradient from this term (its own supervised loss in the same step keeps its gradient)
        vr = valid[rows].T.float()                                       # [S, Br]
        tot = rolled.new_zeros(())
        for lo, hi, T in T_READS:
            with torch.no_grad():
                g_ref = _gap_m(head, params, ref[:, :, lo:hi])
            g_off = _gap_m(head, params, rolled[:, :, lo:hi])
            resid = g_off - g_ref + 0.5 * dr[None, :] * (T ** 2)
            tot = tot + (resid.abs() * vr).sum()
        term = tot / (2.0 * n_valid.clamp_min(1.0))
    return {"term": term, "n_valid": n_valid, "n_rows": rows.sum().to(torch.float32)}
