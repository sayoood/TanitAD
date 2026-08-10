"""E4.4 — the stage-0 TACTICAL-layer trainer on the FROZEN v5f trunk.

WHAT THIS IS (V18_BACKLOG.md E4.4; HIERARCHICAL_WM_REDESIGN.md §3.1–§3.5)
-------------------------------------------------------------------------
Trains :class:`tanitad.models.tactical.TacticalStage0` (PhiTac + goal fan +
FTac + ranking selector, E4.2+E4.3) as the ONLY trainable module on top of the
frozen v5f operative trunk — redesign §3.5 stage 0: a pure addition, zero risk
to the operative stack. The trunk is loaded EXACTLY the way the W4 trainer
(``train_v58f_unicycle_head.py``) loads it — ``eval_flagship_v4.load_v4_from_ck``
with ``frame=train_frame`` — and every loaded module is frozen, md5-checksummed
before/after, and excluded from the optimiser (three independent locks, the
train_unicycle_readout.py contract). The v4 planner head is loaded for loader
fidelity but NEVER forwarded (this trainer only calls ``world.encode_window``),
so no probe vocab is needed.

DATA / LABELS
-------------
Same v2 corpus args + parity/geometry seams as the W4 trainer (``--v2-cache`` /
``--v2-val-cache`` + ``tanitad.geometry`` flags + ``--v2-subframe``). Windowing
is the SAME grid as W4/eval (``FlagshipV4Dataset`` at
``cfg.predictor.window`` / ``plan.max_horizon``), so eval-grid numbers are
grid-comparable. Frames are fetched through the dataset's own slice path
(``to_float_frames(ep.frames[t:t+W])`` — byte-identical to
``EpisodeWindowDataset.__getitem__``); the E4.1 labels are minted ON THE FLY
from FULL-EPISODE poses at ``t_last`` via ``refb_labels.goal_tac_labels`` /
``maneuver3_labels`` (memoised per episode), because the batch's own
``future_poses`` stop at ``max_horizon``=20 (2 s) while the tactical taus reach
6 s. Taus beyond the episode end are clipped by the E4.1 valid mask —
``TacticalStage0.losses`` zeroes them exactly (pinned by tests/test_tactical.py).

Admissibility (binding rules 2026-08-03): labels are ego/future-derived —
LABELS MAY USE EGO. The model's only inference input is ``z_op`` from the
VISION encoder — vision-only at inference; no ego channel, no situation-
classifier output anywhere in the input path (the fan's signature is pinned).

⚠️ THE "1 Hz" z_op WINDOW IS DEGRADED AT THE v5f GEOMETRY — STATED, NOT HIDDEN
------------------------------------------------------------------------------
Redesign §3.1 wants PhiTac pooling z_op(t−3..t) at 1 Hz — 4 latents 10 steps
apart. The v5f dataset window is ``cfg.predictor.window`` = 8 frames (0.8 s),
which cannot span 3 s. Per the E4.4 brief, :func:`zop_window_indices` then uses
the AVAILABLE STRIDE TAIL: effective stride ``(W-1)//(slots-1)`` = 2 at W=8 →
indices [1, 3, 5, 7], i.e. 4 latents 0.2 s apart. Evidence class: DERIVED from
config (window=8 is ``tanitad.config`` base250cam), recorded in the log,
metrics.json and e44_gate.json under ``zop_window``. The 1 s TACTICAL step for
the f_tac latent loss is real, however: ``z_op_next`` is the window 10 dataset
steps (1 s) later, so f_tac learns 1 Hz dynamics regardless of the pool's span.
When ``--w-ftac`` > 0, training samples only windows whose +1 s partner window
exists (the excluded tail count is logged).

PRE-REGISTERED E4.4 GATES (V18_BACKLOG.md 4.4) — written to <out>/e44_gate.json
-------------------------------------------------------------------------------
  * goal FDE@4 s (selected candidate) < CV-extrapolated goal FDE@4 s.
    HERE this is a POINT ESTIMATE over the eval grid; the decision-grade
    interval is the PAIRED episode-cluster bootstrap (taniteval/ci.py) and is a
    POD-SIDE RESCORE item — the JSON says so.
  * manoeuvre 3-axis macro-F1 > the 5-way head's remapped F1 — macro_f1 field
    is NULL when the aux heads are disabled (the default); the 5-way remap
    reference is likewise a pod-side rescore item.
  * sel_gap_tac REPORTED, no threshold (baseline row).

Four-families note (binding 2026-08-02): this instrument measures the TACTICAL
family (goal-setting quality, selector decision quality, manoeuvre axes). The
goal-point heading/speed errors are reported as LATERAL/LONGITUDINAL adjuncts
AT THE GOAL; the full four-family T1 eval of any arm remains the pod-side
taniteval harness. STRATEGIC is n/a at stage 0 (E7) — stated per the rule.

⚠️ POD-SIDE ONLY for the full path: this box has no GPU, no v5f checkpoint, no
v2 corpus. Runnable (and run) here: ``python -m py_compile`` + the pure-helper
smoke ``stack/tests/test_stage0_trainer.py`` (CV baseline, z_op window builder,
gate-JSON schema).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or trainers die with
ModuleNotFound: tanitad):

  python3 train_tactical_stage0.py \
      --ckpt /workspace/experiments/flagship-v5f-.../ckpt_step30000.pt \
      --v2-cache  /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 --out /workspace/experiments/e44-tactical-stage0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refb_labels  # noqa: E402  (light: torch + math only)

DT = 0.1                                   # 10 Hz dataset tick
TAC_STEP_STEPS = 10                        # 1 tactical second @ 10 Hz
GOAL_TAUS = tuple(refb_labels.GOAL_TAC_TAUS_STEPS)   # (20, 40, 60)


def tau_name(steps: int) -> str:
    """20 -> '2s' — the key the mini-eval / gate JSON index taus by."""
    return f"{steps * DT:g}s"


# ============================================================================
# pure helpers (CPU-testable — tests/test_stage0_trainer.py)
# ============================================================================
def zop_window_indices(n_frames: int, slots: int = 4,
                       stride: int = 10) -> list[int]:
    """Indices (into a length-``n_frames`` encoded window, oldest first) of the
    ``slots`` z_op samples PhiTac pools, newest = ``n_frames - 1``.

    Ideal (redesign §3.1): ``stride`` = 10 → 1 Hz samples t-3s..t. When the
    window cannot span ``(slots-1)*stride`` the effective stride degrades to
    ``max(1, (n_frames-1)//(slots-1))`` — the AVAILABLE STRIDE TAIL of the
    E4.4 brief (at the v5f window=8: [1, 3, 5, 7], 0.2 s spacing — documented
    in the module docstring, the log and the gate JSON). Windows shorter than
    ``slots`` repeat the earliest frame (clamp at 0), never raise: the pool is
    causal and a repeated oldest latent is a defined, stated degradation."""
    if n_frames < 1 or slots < 1 or stride < 1:
        raise ValueError(f"bad zop window spec: n_frames={n_frames}, "
                         f"slots={slots}, stride={stride}")
    if slots == 1:
        return [n_frames - 1]
    span = (slots - 1) * stride
    eff = stride if n_frames > span else max(1, (n_frames - 1) // (slots - 1))
    last = n_frames - 1
    return [max(0, last - eff * (slots - 1 - j)) for j in range(slots)]


def cv_goal_baseline(v0: Tensor, taus_steps: tuple[int, ...] = GOAL_TAUS,
                     dt: float = DT) -> Tensor:
    """The E4.4 gate's explicit CV-extrapolated goal baseline: ``v0 [B] ->
    [B, K, 4]`` in the E4.1 layout (x, y, heading, speed).

    Constant velocity + heading hold from v0: x = v0·tau·dt straight ahead,
    y = 0, heading = 0, speed = v0 — i.e. the ego keeps doing exactly what it
    is doing now. On a constant-velocity straight (any fixed heading) this
    matches ``refb_labels.goal_tac_targets`` to float precision (FDE 0 —
    pinned by tests/test_stage0_trainer.py)."""
    v0 = torch.as_tensor(v0, dtype=torch.float32)
    if v0.ndim != 1:
        raise ValueError(f"v0 must be [B], got {tuple(v0.shape)}")
    taus = torch.as_tensor([int(t) for t in taus_steps], dtype=torch.float32,
                           device=v0.device)
    g = v0.new_zeros(v0.shape[0], taus.shape[0], 4)
    g[..., 0] = v0[:, None] * taus[None, :] * dt
    g[..., 3] = v0[:, None]
    return g


def macro_f1_from_confusion(conf) -> float:
    """Macro-F1 from a [C, C] confusion (rows = true, cols = predicted).

    Averaged over classes WITH SUPPORT only (a class absent from the labels
    cannot contribute an F1); NaN when no class has support."""
    c = torch.as_tensor(conf, dtype=torch.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"conf must be square, got {tuple(c.shape)}")
    tp = c.diagonal()
    support = c.sum(dim=1)
    predn = c.sum(dim=0)
    prec = tp / predn.clamp(min=1e-12)
    rec = tp / support.clamp(min=1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    keep = support > 0
    if not bool(keep.any()):
        return float("nan")
    return float(f1[keep].mean())


def build_e44_gate(mini: dict, aux_enabled: bool) -> dict:
    """The V18 E4.4 pre-registered gate record from a mini-eval dict.

    Pure (JSON-in, JSON-out; schema pinned by tests/test_stage0_trainer.py).
    Gate 1 is a POINT-ESTIMATE comparison here — the decision-grade interval
    is the paired episode-cluster bootstrap (taniteval/ci.py), a pod-side
    rescore item, and the record says so. Gate 2's macro_f1 is null when the
    aux heads are disabled; its 5-way-remap reference is pod-side too. Gate 3
    (sel_gap_tac) is a baseline row with NO threshold, per the backlog."""
    k4 = tau_name(40)
    sel4 = mini["goal_fde_m"]["selected"].get(k4)
    cv4 = mini["goal_fde_m"]["cv_baseline"].get(k4)
    n4 = int(mini["n_per_tau"].get(k4, 0) or 0)
    if n4 > 0 and sel4 is not None and cv4 is not None:
        fde_pass, fde_reason = bool(sel4 < cv4), None
    else:
        fde_pass = None
        fde_reason = (f"gate not judgeable: n windows with 4 s of valid "
                      f"future on the eval grid = {n4}")
    gap = mini["sel_gap_tac"]
    return {
        "item": "V18 backlog E4.4 — stage-0 tactical layer, pre-registered gates",
        "gate_goal_fde": {
            "rule": "goal FDE@4s of the SELECTED candidate < CV-extrapolated "
                    "goal FDE@4s (constant velocity + heading hold from v0)",
            "selected_fde_4s_m": sel4,
            "cv_fde_4s_m": cv4,
            "n_4s": n4,
            "pass": fde_pass,
            "reason_if_null": fde_reason,
            "estimator": "POINT ESTIMATE over the eval grid (episodes<40, "
                         "stride 8). The decision-grade interval is the PAIRED "
                         "episode-cluster bootstrap (taniteval/ci.py) — "
                         "POD-SIDE RESCORE REQUIRED before any registry or "
                         "report claim.",
        },
        "gate_maneuver_f1": {
            "rule": "manoeuvre 3-axis macro-F1 > the 5-way head's remapped F1",
            "macro_f1_3axis": mini.get("macro_f1") if aux_enabled else None,
            "reference_5way_remapped_f1": None,
            "pass": None,
            "note": ("aux heads DISABLED (weights 0, the E4.2 default) — "
                     "macro_f1 null per the E4.4 brief" if not aux_enabled else
                     "3-axis macro-F1 measured; the 5-way remapped reference "
                     "is a pod-side rescore item — gate not judged here"),
        },
        "sel_gap_tac": {
            "rule": "reported, no threshold — baseline row (V18 E4.4)",
            "gap": gap.get("gap"),
            "selected_err": gap.get("selected_err"),
            "oracle_err": gap.get("oracle_err"),
            "units_note": "mixed-unit fan_goal_error (m + rad + m/s) — an "
                          "ORDERING quantity, never a quotable metric "
                          "(tanitad/models/tactical.py fan_goal_error)",
        },
        "families": {
            "TACTICAL": "goal FDE per tau + sel_gap_tac + (aux) 3-axis "
                        "manoeuvre macro-F1 — this instrument's family",
            "LATERAL": "goal-point heading |err| per tau (adjunct; full "
                       "curvature/yaw-rate family is the pod-side taniteval "
                       "harness)",
            "LONGITUDINAL": "goal-point speed |err| per tau (adjunct; "
                            "headway/TTC need a lead agent — pod-side)",
            "STRATEGIC": "n/a at stage 0 — the strategic layer is E7 (v1.9); "
                         "stated per the 2026-08-02 rule",
        },
        "mini_eval": mini,
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }


# ============================================================================
# on-the-fly E4.1 label mint (full-episode poses; memoised per episode)
# ============================================================================
class TacticalLabelMint:
    """Mints E4.1 labels at ``t_last`` from FULL-EPISODE poses.

    Why not the batch's ``future_poses``: those stop at max_horizon=20 (2 s)
    while the tactical taus reach 60 (6 s). ``goal_tac_labels`` /
    ``maneuver3_labels`` are computed once per episode and memoised (a [T,3,4]
    float + [T,3] long table per episode — KBs), so the per-sample cost is a
    row lookup. Taus beyond the episode end carry valid=False (clip semantics
    live in refb_labels; ``TacticalStage0.losses`` zeroes them)."""

    def __init__(self, taus_steps: tuple[int, ...] = GOAL_TAUS):
        self.taus = tuple(int(t) for t in taus_steps)
        self._poses: dict[int, Tensor] = {}
        self._goal: dict[int, tuple[Tensor, Tensor]] = {}
        self._man3: dict[int, Tensor] = {}

    def _ep_poses(self, ds, e_i: int) -> Tensor:
        p = self._poses.get(e_i)
        if p is None:
            p = self._poses[e_i] = torch.as_tensor(
                ds.episodes[e_i].poses).float()
        return p

    def labels_for(self, ds, idxs: list[int],
                   need_man3: bool = False) -> dict[str, Tensor]:
        """dataset indices -> {goal [B,K,4], goal_valid [B,K], v0 [B]
        (+ man3 [B,3] when requested)} — CPU tensors, caller moves them."""
        goals, valids, man3s, v0s = [], [], [], []
        for i in idxs:
            e_i, t = ds.index[i]
            t_last = t + ds.window - 1
            if e_i not in self._goal:
                self._goal[e_i] = refb_labels.goal_tac_labels(
                    self._ep_poses(ds, e_i), self.taus)
            g, v = self._goal[e_i]
            goals.append(g[t_last])
            valids.append(v[t_last])
            v0s.append(self._ep_poses(ds, e_i)[t_last, 3])
            if need_man3:
                if e_i not in self._man3:
                    self._man3[e_i] = refb_labels.maneuver3_labels(
                        self._ep_poses(ds, e_i))
                man3s.append(self._man3[e_i][t_last])
        out = {"goal": torch.stack(goals).float(),
               "goal_valid": torch.stack(valids),
               "v0": torch.stack(v0s).float()}
        if need_man3:
            out["man3"] = torch.stack(man3s).long()
        return out


# ============================================================================
# data plumbing (pod-side)
# ============================================================================
def _frames_at(ds, e_i: int, t: int) -> Tensor:
    """The window's frames [W, C, H, W'], float — byte-identical to the
    ``EpisodeWindowDataset.__getitem__`` frames path (same slice, same
    ``to_float_frames``); the rest of the item dict (v4 label mint) is skipped
    because this trainer mints its own labels from episode poses."""
    from tanitad.data._contract import to_float_frames
    ep = ds.episodes[e_i]
    return to_float_frames(torch.as_tensor(ep.frames[t:t + ds.window]))


def fetch_frames(ds, idxs: list[int]) -> Tensor:
    return torch.stack([_frames_at(ds, *ds.index[i]) for i in idxs])


def make_sampler(ds, eps_per_batch: int, rng: random.Random,
                 allowed: list[int] | None = None):
    """Episode-grouped batch sampler — the W4 trainer's MooseFS I/O shape
    (FEW episodes x MANY windows per batch), optionally restricted to
    ``allowed`` dataset indices (the +1 s-partner constraint of the f_tac
    loss). Same accepted trade: mild within-batch correlation for ~8x fewer
    cold payload loads."""
    ep2idx: dict[int, list[int]] = {}
    it = allowed if allowed is not None else range(len(ds.index))
    for i in it:
        e, _t = ds.index[i]
        ep2idx.setdefault(e, []).append(i)
    if not ep2idx:
        raise SystemExit("[e44] sampler has no eligible windows")
    ep_ids = list(ep2idx)

    def sample(bs: int) -> list[int]:
        chosen = [ep_ids[rng.randrange(len(ep_ids))]
                  for _ in range(min(eps_per_batch, len(ep_ids)))]
        out: list[int] = []
        gi = 0
        while len(out) < bs:
            pool = ep2idx[chosen[gi % len(chosen)]]
            out.append(pool[rng.randrange(len(pool))])
            gi += 1
        return out

    return sample


def encode_zop(world, frames: Tensor, zop_idx: list[int],
               amp_on: bool) -> Tensor:
    """Frozen-trunk encode + the 1 Hz(-degraded) slice: frames
    [B, W, C, H, W'] -> z_op [B, slots, d_op] float32, detached (no_grad —
    frozen means the tactical stack trains on constants)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=amp_on):
        st = world.encode_window(frames)
    return st.float()[:, zop_idx, :]


# ============================================================================
# end-of-training mini-eval — SAME grid rule as eval defaults (eps<40, stride 8)
# ============================================================================
@torch.no_grad()
def mini_eval(world, tac, ds_val, mint: TacticalLabelMint, device, *,
              amp_on: bool, zop_idx: list[int],
              taus: tuple[int, ...] = GOAL_TAUS, episodes: int = 40,
              stride: int = 8, batch: int = 16,
              aux_enabled: bool = False) -> dict:
    """Goal FDE@{2,4,6}s (selected / CV baseline / fan oracle / fan min),
    sel_gap_tac, per-tau heading/speed adjuncts, and (aux) 3-axis macro-F1,
    over the eval-default window grid ``e < episodes and t % stride == 0`` —
    grid-comparable with the W4 / eval_flagship_v4 numbers.

    Per-tau means are masked by the E4.1 valid mask and each tau reports its
    own n (a tau with n=0 reports None, never a silent drop — the 2026-08-02
    rule). ``fan_oracle`` = the hindsight-oracle candidate (argmin mixed
    fan_goal_error — the sel_gap oracle definition); ``fan_min`` = per-tau
    minimum positional error over the fan (the fan's positional ceiling)."""
    from tanitad.models.tactical import (N_LANE3, N_LAT3, N_LON3,
                                         fan_goal_error, sel_gap_tac)
    tac.eval()
    names = [tau_name(t) for t in taus]
    K = len(taus)
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < episodes and t % stride == 0]
    if not grid:
        raise SystemExit("[e44] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    z = torch.zeros(K, dtype=torch.float64)
    sums = {"sel": z.clone(), "cv": z.clone(), "oracle": z.clone(),
            "fmin": z.clone(), "head": z.clone(), "speed": z.clone()}
    n_tau = torch.zeros(K, dtype=torch.float64)
    gsum = {"sel_err": 0.0, "oracle_err": 0.0, "gap": 0.0, "n": 0}
    confs = ([torch.zeros(c, c, dtype=torch.long)
              for c in (N_LAT3, N_LON3, N_LANE3)] if aux_enabled else None)
    n = 0
    t0 = time.time()
    for b0 in range(0, len(grid), batch):
        idxs = grid[b0:b0 + batch]
        lab = mint.labels_for(ds_val, idxs, need_man3=aux_enabled)
        frames = fetch_frames(ds_val, idxs).to(device)
        z_op = encode_zop(world, frames, zop_idx, amp_on)
        out = tac(z_op)
        goals = out["goals"].float()
        sel = out["sel_idx"]
        labels = lab["goal"].to(device)
        valid = lab["goal_valid"].to(device)
        err = fan_goal_error(goals, labels, valid)              # [B, N] mixed
        s, o, g = sel_gap_tac(err, sel)
        anyv = valid.any(-1)
        gsum["sel_err"] += float(s[anyv].sum())
        gsum["oracle_err"] += float(o[anyv].sum())
        gsum["gap"] += float(g[anyv].sum())
        gsum["n"] += int(anyv.sum())
        bs = goals.shape[0]
        ar = torch.arange(bs, device=device)
        g_sel = goals[ar, sel]                                  # [B, K, 4]
        g_or = goals[ar, err.argmin(dim=1)]
        cv = cv_goal_baseline(lab["v0"], taus).to(device)
        m = valid.double()
        d_sel = (g_sel[..., :2] - labels[..., :2]).norm(dim=-1).double()
        d_cv = (cv[..., :2] - labels[..., :2]).norm(dim=-1).double()
        d_or = (g_or[..., :2] - labels[..., :2]).norm(dim=-1).double()
        d_min = (goals[..., :2] - labels[:, None, :, :2]).norm(
            dim=-1).min(dim=1).values.double()
        d_head = refb_labels.wrap_to_pi(
            g_sel[..., 2] - labels[..., 2]).abs().double()
        d_speed = (g_sel[..., 3] - labels[..., 3]).abs().double()
        for key, d in (("sel", d_sel), ("cv", d_cv), ("oracle", d_or),
                       ("fmin", d_min), ("head", d_head), ("speed", d_speed)):
            sums[key] += (d * m).sum(dim=0).cpu()
        n_tau += m.sum(dim=0).cpu()
        if aux_enabled:
            man3 = lab["man3"]
            z_tac = out["z_tac"]
            for ax, head in enumerate((tac.aux_lat, tac.aux_lon,
                                       tac.aux_lane)):
                pred = head(z_tac).argmax(dim=-1).cpu()
                for tcls, pcls in zip(man3[:, ax].tolist(), pred.tolist()):
                    confs[ax][tcls, pcls] += 1
        n += bs
    tac.train()

    def per_tau(key: str) -> dict:
        return {names[k]: (round(float(sums[key][k] / n_tau[k]), 4)
                           if float(n_tau[k]) > 0 else None)
                for k in range(K)}

    macro = None
    if aux_enabled:
        f1 = {ax: macro_f1_from_confusion(confs[i])
              for i, ax in enumerate(("lat", "lon", "lane"))}
        vals = [v for v in f1.values() if not math.isnan(v)]
        f1 = {k: (None if math.isnan(v) else round(v, 4))
              for k, v in f1.items()}
        f1["mean3"] = round(sum(vals) / len(vals), 4) if vals else None
        macro = f1
    ng = max(gsum["n"], 1)
    return {
        "n_windows": n,
        "grid": {"episodes": episodes, "stride": stride, "batch": batch},
        "n_per_tau": {names[k]: int(n_tau[k]) for k in range(K)},
        "goal_fde_m": {"selected": per_tau("sel"),
                       "cv_baseline": per_tau("cv"),
                       "fan_oracle": per_tau("oracle"),
                       "fan_min": per_tau("fmin")},
        "goal_heading_err_rad": {"selected": per_tau("head")},
        "goal_speed_err_ms": {"selected": per_tau("speed")},
        "sel_gap_tac": {k: round(gsum[k] / ng, 4)
                        for k in ("selected_err", "oracle_err", "gap")}
        | {"n": gsum["n"]},
        "macro_f1": macro,
        "wallclock_s": round(time.time() - t0, 1),
    }


# ============================================================================
# main (POD-SIDE: needs GPU + the v5f checkpoint + the v2 corpora)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_tactical_stage0", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v5f checkpoint "
                    "(keys: model, grounding, head[, goal_head])")
    ap.add_argument("--head-config", default=None,
                    help="run config.json (default: sibling of --ckpt); the "
                         "head is loaded frozen for loader fidelity, never "
                         "forwarded")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer — pass-through to "
                         "load_v4_from_ck (irrelevant to encode_window)")
    # corpus (v2 compressed only — the ONLY format v5 has); same seams as W4
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000,
                    help="head-scale budget (E4.4: ~3 GPU-h)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4,
                    help="episode-grouped sampling (the MooseFS I/O shape)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint + metrics.json cadence")
    ap.add_argument("--log-every", type=int, default=50)
    # tactical geometry + loss weights (TacticalStage0Config knobs)
    ap.add_argument("--d-tac", type=int, default=512,
                    help="z_tac width (redesign §3.1: d_t ~ 512)")
    ap.add_argument("--n-goals", type=int, default=8, help="fan size N (E4.2)")
    ap.add_argument("--zop-slots", type=int, default=4,
                    help="latents PhiTac pools (§3.1: z_op(t-3..t) = 4)")
    ap.add_argument("--zop-stride", type=int, default=10,
                    help="requested inter-latent stride in dataset steps (10 = "
                         "1 Hz); degrades to the available stride tail when "
                         "the window is shorter — see module docstring")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--w-wta", type=float, default=1.0)
    ap.add_argument("--w-rank-fan", type=float, default=1.0)
    ap.add_argument("--w-rank-sel", type=float, default=1.0)
    ap.add_argument("--w-ftac", type=float, default=1.0,
                    help="f_tac 1 s latent loss; > 0 restricts sampling to "
                         "windows whose +1 s partner window exists")
    ap.add_argument("--aux-maneuver", action="store_true",
                    help="enable the E4.1 3-axis aux CE heads (weights below; "
                         "OFF by default per E4.2)")
    ap.add_argument("--w-lat", type=float, default=0.2,
                    help="aux LAT CE weight (applied only with --aux-maneuver)")
    ap.add_argument("--w-lon", type=float, default=0.2)
    ap.add_argument("--w-lane", type=float, default=0.2)
    # mini-eval grid (eval defaults)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[e44] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v4_from_ck, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity
    from tanitad.models.tactical import TacticalStage0, TacticalStage0Config
    from train_v58f_unicycle_head import build_train_episodes, module_md5

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(
        a, cfg, label="train_tactical_stage0")
    plan = _plan(cfg)
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[e44] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs E4.4 train frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[e44] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    # ---- frozen v5f: EXACTLY the W4 trainer's loader ------------------------
    print(f"[e44] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "model" in ck):
        raise SystemExit("[e44] --ckpt has no 'model' key — not a v4/v5 "
                         "flagship checkpoint")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=a.anchors_dense, frame=model_frame)
    del ck

    # ---- frozen means PROVED frozen (three locks, W4 contract) --------------
    frozen_mods = [world, grounding, head] + (
        [goal_head] if goal_head is not None else [])
    assert not any(p.requires_grad for m in frozen_mods
                   for p in m.parameters())
    md5_before = module_md5(world)
    # d_op DISCOVERED from the loaded trunk, asserted against its own config
    # composition (readout.out_dim IS state_dim by construction) and re-checked
    # against the first encoded batch below.
    d_op = int(world.state_dim)
    assert d_op == int(world.readout.out_dim), (d_op, world.readout.out_dim)
    print(f"[e44] trunk frozen · base step {base_step} · d_op={d_op} · "
          f"md5 {md5_before[:12]}", flush=True)

    # ---- data (same grid as W4/eval: FlagshipV4Dataset at the plan sizes) ---
    train_eps, train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_train = FlagshipV4Dataset(train_eps, window=cfg.predictor.window,
                                 max_horizon=plan.max_horizon,
                                 maneuver_h=plan.maneuver_h,
                                 channels=cfg.encoder.in_channels)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[e44] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)

    # ---- the z_op window (documented degradation — module docstring) --------
    W = int(cfg.predictor.window)
    zop_idx = zop_window_indices(W, a.zop_slots, a.zop_stride)
    eff = (zop_idx[-1] - zop_idx[-2]) if len(zop_idx) >= 2 else 0
    zop_doc = {"window_frames": W, "slots": a.zop_slots,
               "stride_requested": a.zop_stride, "stride_effective": eff,
               "indices": zop_idx,
               "note": ("1 Hz as specified" if eff == a.zop_stride else
                        f"DEGRADED: window {W} cannot span "
                        f"{(a.zop_slots - 1) * a.zop_stride} steps — using the "
                        f"available stride tail ({eff * DT:g} s spacing); the "
                        f"f_tac step stays 1 s via z_op_next")}
    print(f"[e44] z_op window: {zop_doc}", flush=True)

    # ---- the ONLY trainable module ------------------------------------------
    aux_enabled = bool(a.aux_maneuver)
    tcfg = TacticalStage0Config(
        d_op=d_op, d_tac=a.d_tac, window=a.zop_slots, n_goals=a.n_goals,
        horizon_taus=GOAL_TAUS, margin=a.margin, w_wta=a.w_wta,
        w_rank_fan=a.w_rank_fan, w_rank_sel=a.w_rank_sel, w_ftac=a.w_ftac,
        w_lat=(a.w_lat if aux_enabled else 0.0),
        w_lon=(a.w_lon if aux_enabled else 0.0),
        w_lane=(a.w_lane if aux_enabled else 0.0))
    tac = TacticalStage0(tcfg).to(device)
    assert tac.cfg.d_op == d_op
    assert all(p.requires_grad for p in tac.parameters())
    n_par = sum(p.numel() for p in tac.parameters() if p.requires_grad)
    print(f"[e44] TacticalStage0 d_op={d_op} d_tac={a.d_tac} N={a.n_goals} "
          f"aux={aux_enabled} ({n_par / 1e6:.3f} M trainable; frozen "
          f"everything else)", flush=True)
    opt = torch.optim.AdamW(tac.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    # ---- label mint + partner-constrained sampler ---------------------------
    mint = TacticalLabelMint(GOAL_TAUS)
    partner: dict[int, int] = {}
    allowed = None
    if a.w_ftac > 0:
        pos = {et: i for i, et in enumerate(ds_train.index)}
        partner = {i: pos[(e, t + TAC_STEP_STEPS)]
                   for i, (e, t) in enumerate(ds_train.index)
                   if (e, t + TAC_STEP_STEPS) in pos}
        allowed = sorted(partner)
        print(f"[e44] f_tac partner constraint: {len(allowed)}/"
              f"{len(ds_train.index)} windows have a +1 s partner "
              f"({len(ds_train.index) - len(allowed)} tail windows excluded "
              f"from TRAINING only)", flush=True)
    sample = make_sampler(ds_train, a.eps_per_batch, rng, allowed=allowed)

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "e44-tactical-stage0", "args": vars(a),
        "base_ckpt": a.ckpt, "base_step": base_step,
        "trunk_md5": md5_before, "d_op": d_op, "n_trainable": n_par,
        "zop_window": zop_doc, "aux_enabled": aux_enabled,
        "train_parity": {"n_dirs": len(a.v2_cache)},
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    acc: dict[str, float] = {}
    acc_n = 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        lab = mint.labels_for(ds_train, idx, need_man3=aux_enabled)
        frames = fetch_frames(ds_train, idx).to(device)
        z_op = encode_zop(world, frames, zop_idx, amp_on)
        if step == 1:
            assert z_op.shape[-1] == d_op, (z_op.shape, d_op)  # runtime lock
        batch = {"z_op": z_op,
                 "goal": lab["goal"].to(device),
                 "goal_valid": lab["goal_valid"].to(device)}
        if a.w_ftac > 0:
            frames_next = fetch_frames(
                ds_train, [partner[i] for i in idx]).to(device)
            batch["z_op_next"] = encode_zop(world, frames_next, zop_idx,
                                            amp_on)
        if aux_enabled:
            batch["man3"] = lab["man3"].to(device)

        L = tac.losses(batch)
        opt.zero_grad(set_to_none=True)
        L["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(tac.parameters(), 5.0)
        opt.step()
        sched.step()

        bs = z_op.shape[0]
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach()) * bs
        acc_n += bs

        if step % a.log_every == 0:
            rec = {"step": step,
                   **{k: round(float(v.detach()), 5) for k, v in L.items()},
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.save_every == 0:
            n_ = max(acc_n, 1)
            row = {"step": step,
                   **{k: round(v / n_, 5) for k, v in acc.items()},
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            acc, acc_n = {}, 0
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "args": vars(a),
                           "base_step": base_step, "d_op": d_op,
                           "zop_window": zop_doc,
                           "_read": "rows are TRAIN-batch running means over "
                                    "the last save window (sel_gap/sel_err/"
                                    "oracle_err are the mixed-unit train "
                                    "monitors); the gate numbers are the "
                                    "held-out mini-eval in e44_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf, indent=1)
            torch.save({"tactical": tac.state_dict(),
                        "cfg": dataclasses.asdict(tcfg),
                        "zop_indices": zop_idx, "step": step,
                        "args": vars(a), "base_ckpt": a.ckpt,
                        "base_step": base_step, "d_op": d_op},
                       os.path.join(a.out, "tactical_stage0.pt"))
            fh.write(json.dumps({"per500": row}) + "\n")
            fh.flush()
            print(f"[e44 @{step}] {row}", flush=True)

    # ---- frozen proof + the pre-registered gates ----------------------------
    md5_after = module_md5(world)
    ev = mini_eval(world, tac, ds_val, mint, device, amp_on=amp_on,
                   zop_idx=zop_idx, taus=GOAL_TAUS, episodes=a.episodes,
                   stride=a.stride, batch=a.eval_batch,
                   aux_enabled=aux_enabled)
    gate = build_e44_gate(ev, aux_enabled)
    gate.update({
        "steps": a.steps, "base_ckpt": a.ckpt, "base_step": base_step,
        "d_op": d_op, "n_trainable": n_par, "zop_window": zop_doc,
        "aux_enabled": aux_enabled,
        "trunk_frozen_proof": {"md5_before": md5_before,
                               "md5_after": md5_after,
                               "identical": md5_before == md5_after},
        "wall_s": round(time.time() - t0, 1),
    })
    with open(os.path.join(a.out, "e44_gate.json"), "w") as gf:
        json.dump(gate, gf, indent=1)
    fh.write(json.dumps({"summary": gate}) + "\n")
    fh.close()
    print(f"\n[E4.4 SUMMARY] {json.dumps(gate, indent=1)}", flush=True)
    if not gate["trunk_frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK CHANGED DURING TRAINING — run invalid")
    fde = gate["gate_goal_fde"]
    verdict = {True: "PASS", False: "FAIL", None: "NOT JUDGEABLE"}[fde["pass"]]
    print(f"[E4.4 GATE goal-FDE@4s] {verdict} (selected "
          f"{fde['selected_fde_4s_m']} vs CV {fde['cv_fde_4s_m']}, "
          f"n={fde['n_4s']}; point estimate — CI is a pod-side paired "
          f"episode-cluster bootstrap rescore) · sel_gap_tac "
          f"{gate['sel_gap_tac']['gap']} (baseline row)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
