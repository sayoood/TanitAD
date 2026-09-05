#!/usr/bin/env python3
"""FAN SAFETY on the EMITTED fan — the PRIMARY endpoint of the REF-C RL re-scope.

Added 2026-09-05 (Deploy FlyWheel) after the PI's correction of `H-RL-MIN-1`:
*"the paper is talking about improving/eliminating trajectories leading to
[collisions / infeasible outcomes]"*. DiffusionDriveV2's RL stage exists to push
probability mass AWAY from collision-prone and infeasible candidates (the
collision-truncated advantage, PUBLISHED-CODE `_model_rl.py:891-902`), so the
question an RL-on/off experiment must answer FIRST is not "is the selected path
closer to the human" but "is the FAN safer, and does the confidence head put less
mass on the unsafe part of it". The four families stay as the SECONDARY endpoint.

WHAT IS SCORED (every path is the 2 s PREFIX on the 0.5 s grid — origin + 4 slots,
the same index-select the `2s` eval grid uses):

  (a) CONTACT      `rewards._collision`, MOVING-LEAD mode: time-aligned centre
                   distance < 2 m (ego 1 m + obstacle 1 m) against the lead agent's
                   own `obstacle.offline` track (`build_lead_block_b1.py`, ego t0
                   frame, resampled to the grid). Population: windows WITH a lead.
  (b) TTC-BELOW    `rewards.ttc_violation` at 2.93 s — the H-RL-THRESH-1 class:
                   2.93 s is the human driver's OWN 5th-percentile time gap on the
                   fit8 lead windows (`raw/humanflag_fit8.json`), so a candidate
                   below it is closer than the human ever follows. The 1.5 s VETO
                   threshold is reported beside it. Population: lead windows.
  (c) INFEASIBLE   three instruments, each reported and then OR-ed:
                   `kamm_over`  — friction-circle load sqrt(a_lon^2 + a_lat^2)/g
                                  from `rewards.kinematics` finite differences on
                                  the 0.5 s prefix, > mu = 0.7. ⚠️ STATED LIMIT:
                                  the emitted fan is FREE WAYPOINTS (bank + offset),
                                  so `instruments/flyability.friction_load` — the
                                  exact, control-rolled instrument — cannot be
                                  applied; the finite difference UNDER-reports by
                                  1.21-1.85x on control-rolled paths (flyability.py
                                  docstring). A rate here is a LOWER bound.
                   `envelope`   — the reward's own feasibility envelope: |a| > 4
                                  m/s^2 or |kappa| > 0.2 1/m at any step.
                   `off_reach`  — OUTSIDE the S2 reachability band
                                  (`refc_select.reachability_mask`: mean speed over
                                  the 2 s prefix within v0 ± 2.5 m/s^2 · 2 s), the
                                  band refcv3 @ 40,284 does NOT apply at inference
                                  (`sel_reach_clamp` False in its config).
  (d) MASS         the probability the model's ranking softmax and the raw
                   confidence head put on (a)-(c): `softmax(sel_score_v3)` (the
                   score the argmax actually uses) and `softmax(anchor_logits)`
                   (`conf_head`), summed over flagged candidates. Reported by the
                   DRIVER's readout (it needs the live model); this module supplies
                   `summarise_fan` for it.

TWO MODES
  --dump   score the SELECTED paths banked by `refcv3_arm.py` (g = the human,
           os, ha, ha0, os_navshuf, os_navzero, oracle_sel) plus a `frozen`
           zero-path control that must read kamm 0 / envelope 0 EXACTLY. This is
           the human REFERENCE and the trivial FLOOR the fan numbers are read
           against. 0 GPU.
  --pair   paired episode-cluster bootstrap (`taniteval.ci`) of an arm's fan
           readout against the base's, on the same windows, per population.

Evidence class of everything computed here: MEASURED (ours). Tier: T1-adjacent —
the fan is what the deployed model EMITS on the eval clips (self-action open loop,
no re-perception); never a driving claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(_TE)
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
from torch import Tensor                                             # noqa: E402

from tanitad.instruments.flyability import G as G_MS2                # noqa: E402
from tanitad.refs.refc_select import reachability_mask               # noqa: E402
from tanitad.rl import rewards as RW                                 # noqa: E402
import taniteval.ci as CI                                            # noqa: E402

TOOL = "taniteval/tools/fan_safety.py"
DT_S = 0.5
GRID_S = (0.0, 0.5, 1.0, 1.5, 2.0)
TTC_THRESH_S = 2.93          # H-RL-THRESH-1 class: the human's 5th-pct time gap (fit8)
TTC_VETO_S = 1.5             # the RL veto threshold (THRESHOLD_CALIBRATION: OK)
MU_KAMM = 0.7                # friction circle
A_MAX_REACH_MPS2 = 2.5       # S2 band (refc_select.reachability_mask default)
HORIZON_REACH_S = 2.0
LEAD_LEN_DEFAULT_M = 4.5
FAR_LEAD_X_M = 1.0e6
TOPK = (8, 32)               # 32 = DDv2's selector top-k (`_model_sel.py:1372`)
FLAGS = ("contact", "ttc_below", "ttc_veto", "kamm_over", "envelope", "off_reach",
         "infeasible", "unsafe", "flagged")
LEAD_ONLY = ("contact", "ttc_below", "ttc_veto", "unsafe")
DUMP_ARMS = ("g", "os", "ha", "ha0", "os_navshuf", "os_navzero", "oracle_sel")


# --------------------------------------------------------------------------- #
# geometry helpers — ONE definition each, shared with the driver                #
# --------------------------------------------------------------------------- #
def with_origin(x: Tensor) -> Tensor:
    """[..., 4, 2] -> [..., 5, 2] with the ego origin prepended (the 2 s prefix)."""
    z = torch.zeros(*x.shape[:-2], 1, 2, dtype=x.dtype, device=x.device)
    return torch.cat([z, x], dim=-2)


def resample_track(track: np.ndarray, ts_rel: np.ndarray) -> np.ndarray:
    """[K, 2] lead samples at ``ts_rel`` (0.2..2.0 s) -> [5, 2] at GRID_S. t = 0 is a
    linear extrapolation from the first two samples (one 0.2 s step back)."""
    out = np.zeros((len(GRID_S), 2))
    for i, t in enumerate(GRID_S):
        if t < ts_rel[0]:
            slope = (track[1] - track[0]) / (ts_rel[1] - ts_rel[0])
            out[i] = track[0] + slope * (t - ts_rel[0])
        else:
            out[i, 0] = np.interp(t, ts_rel, track[:, 0])
            out[i, 1] = np.interp(t, ts_rel, track[:, 1])
    return out


def hold_v0_path(v0: Tensor) -> Tensor:
    """The `ha0` floor: constant-velocity straight path, [..., 5, 2] from v0 [...]."""
    ts = torch.tensor(GRID_S, dtype=v0.dtype, device=v0.device)
    xs = v0.unsqueeze(-1) * ts
    return torch.stack([xs, torch.zeros_like(xs)], dim=-1)


def reach_ok(paths5: Tensor, v0: Tensor, *, accel_max: float = A_MAX_REACH_MPS2,
             horizon_s: float = HORIZON_REACH_S) -> Tensor:
    """The S2 band on ``[..., 5, 2]`` prefixes with ``v0`` broadcastable to ``[...]``.

    Same rule as ``refc_select.reachability_mask`` (mean speed = |wp_last| / T within
    [max(0, v0 - aT), v0 + aT]); ``test_fan_safety.py`` asserts equality against it
    on a [B, N, S, 2] fan so this stays ONE definition, not a second implementation.
    """
    v_mean = paths5[..., -1, :].norm(dim=-1) / horizon_s
    reach = accel_max * horizon_s
    v0b = v0
    while v0b.dim() < v_mean.dim():
        v0b = v0b.unsqueeze(-1)
    lo = (v0b - reach).clamp_min(0.0)
    return (v_mean >= lo) & (v_mean <= v0b + reach)


def min_ttc_s(paths5: Tensor, lead5: Tensor, *, lead_len_m: float = LEAD_LEN_DEFAULT_M,
              dt: float = DT_S) -> Tensor:
    """Minimum TTC over the closing steps, +inf when never closing. Mirrors
    ``rewards.ttc_violation`` exactly (surface gap, closing speed) — kept beside it so
    a threshold sweep and the veto agree on the geometry."""
    gap = (lead5[..., 1:, :] - paths5[..., 1:, :]).norm(dim=-1)
    gap = (gap - lead_len_m).clamp_min(0.0)
    closing = (gap[..., :-1] - gap[..., 1:]) / dt
    ttc = gap[..., :-1] / closing.clamp_min(1e-3)
    ttc = torch.where(closing > 0, ttc, torch.full_like(ttc, float("inf")))
    return ttc.amin(dim=-1)


# --------------------------------------------------------------------------- #
# the scorer                                                                    #
# --------------------------------------------------------------------------- #
def score_paths(paths5: Tensor, v0: Tensor, lead5: Tensor | None, *,
                lead_len_m: float = LEAD_LEN_DEFAULT_M,
                a_max: float = RW.A_MAX_MPS2, kappa_max: float = RW.KAPPA_MAX_1PM,
                mu: float = MU_KAMM) -> dict[str, Tensor]:
    """``paths5 [..., 5, 2]`` (metres, ego t0 frame, 0.5 s grid), ``v0`` broadcastable
    to ``[...]``, ``lead5`` ``[..., 5, 2]`` broadcastable (or None = no lead: contact and
    TTC read False and are NOT in the population). -> dict of ``[...]`` tensors."""
    if paths5.shape[-2] != len(GRID_S) or paths5.shape[-1] != 2:
        raise ValueError(f"expected [..., 5, 2] prefixes on the 0.5 s grid, got {tuple(paths5.shape)}")
    lead = paths5.shape[:-2]
    dev, dt_ = paths5.device, paths5.dtype
    v0b = v0.to(dev, dt_)
    while v0b.dim() < len(lead):
        v0b = v0b.unsqueeze(-1)
    ctx = {"dt": DT_S, "lead_len_m": lead_len_m}
    if lead5 is not None:
        ctx["lead_path"] = lead5.to(dev, dt_)
        contact = RW.COMPONENTS["collision"](paths5, ctx) < 0
        ttc_below = RW.ttc_violation(paths5, {**ctx, "ttc_min_s": TTC_THRESH_S})
        ttc_veto = RW.ttc_violation(paths5, {**ctx, "ttc_min_s": TTC_VETO_S})
        ttc_min = min_ttc_s(paths5, ctx["lead_path"], lead_len_m=lead_len_m)
        gap = (ctx["lead_path"][..., 1:, :] - paths5[..., 1:, :]).norm(dim=-1)
        gap_min = (gap - lead_len_m).clamp_min(0.0).amin(dim=-1)
    else:
        contact = torch.zeros(lead, dtype=torch.bool, device=dev)
        ttc_below = torch.zeros_like(contact)
        ttc_veto = torch.zeros_like(contact)
        ttc_min = torch.full(lead, float("inf"), dtype=dt_, device=dev)
        gap_min = torch.full(lead, float("inf"), dtype=dt_, device=dev)
    kin = RW.kinematics(paths5, DT_S)
    a_tot = torch.sqrt(kin.accel.pow(2) + kin.lat_acc.pow(2))          # [..., 3]
    peak_g = a_tot.amax(dim=-1) / G_MS2
    kamm_over = peak_g > mu
    envelope = (kin.accel.abs().amax(dim=-1) > a_max) | (kin.kappa.abs().amax(dim=-1) > kappa_max)
    off_reach = ~reach_ok(paths5, v0b)
    infeasible = kamm_over | envelope | off_reach
    unsafe = contact | ttc_below
    return {"contact": contact, "ttc_below": ttc_below, "ttc_veto": ttc_veto,
            "kamm_over": kamm_over, "envelope": envelope, "off_reach": off_reach,
            "infeasible": infeasible, "unsafe": unsafe, "flagged": unsafe | infeasible,
            "peak_g": peak_g, "min_ttc_s": ttc_min, "min_gap_m": gap_min,
            "v_mean_2s": paths5[..., -1, :].norm(dim=-1) / HORIZON_REACH_S}


def summarise_fan(sc: dict[str, Tensor], rank: Tensor, conf: Tensor, sel_idx: Tensor,
                  *, topk: tuple[int, ...] = TOPK) -> dict[str, Tensor]:
    """Per-window fan summaries from per-candidate flags ``sc[f] [B, N]``.

    ``rank`` [B, N] is the score the model's argmax uses (``sel_score_v3`` masked by
    ``reach_keep`` when present — -inf rows carry zero mass); ``conf`` [B, N] the raw
    ``anchor_logits`` of `conf_head`; ``sel_idx`` [B] the emitted candidate.
    Returns ``[B]`` tensors: ``fan_<f>`` (fraction of candidates), ``top<k>_<f>``,
    ``sel_<f>`` (0/1), ``mass_rank_<f>``, ``mass_conf_<f>`` (probability mass)."""
    b, n = rank.shape
    order = rank.argsort(dim=1, descending=True)
    p_rank = torch.softmax(rank.float(), dim=1)
    p_conf = torch.softmax(conf.float(), dim=1)
    out: dict[str, Tensor] = {}
    for f in FLAGS:
        x = sc[f].to(torch.float32)                                   # [B, N]
        out[f"fan_{f}"] = x.mean(dim=1)
        for k in topk:
            k = min(int(k), n)
            out[f"top{k}_{f}"] = x.gather(1, order[:, :k]).mean(dim=1)
        out[f"sel_{f}"] = x.gather(1, sel_idx.reshape(b, 1))[:, 0]
        out[f"mass_rank_{f}"] = (p_rank * x).sum(dim=1)
        out[f"mass_conf_{f}"] = (p_conf * x).sum(dim=1)
    out["fan_peak_g_mean"] = sc["peak_g"].float().mean(dim=1)
    out["sel_peak_g"] = sc["peak_g"].float().gather(1, sel_idx.reshape(b, 1))[:, 0]
    out["fan_v_mean_2s_spread"] = sc["v_mean_2s"].float().std(dim=1)
    return out


# --------------------------------------------------------------------------- #
# the lead block                                                                #
# --------------------------------------------------------------------------- #
def load_lead(path: str):
    """The per-frame B1 lead block -> a namespace: leads [R, K, 2], has [R], lens [R],
    idx {(clip_id, RAW frame): row}, ts_rel [K]. Same keys `refav1_arm.load_lead_block_rows`
    requires; read directly so this tool has no dependency on the arm modules."""
    from types import SimpleNamespace
    z = np.load(path, allow_pickle=True)
    for k in ("clip_id", "frame", "leads", "has_lead", "lead_lens", "ts_rel_s"):
        if k not in z.files:
            raise SystemExit(f"[fan_safety] lead block {path} carries no {k!r}")
    cid = np.asarray(z["clip_id"]).astype(str).reshape(-1)
    fr = np.asarray(z["frame"]).astype(np.int64).reshape(-1)
    idx = {}
    for i, (c, f) in enumerate(zip(cid.tolist(), fr.tolist())):
        if (c, f) in idx:
            raise SystemExit(f"[fan_safety] duplicate lead row for ({c}, {f})")
        idx[(c, f)] = i
    return SimpleNamespace(path=path, leads=np.asarray(z["leads"], dtype=np.float64),
                           has=np.asarray(z["has_lead"]).astype(bool).reshape(-1),
                           lens=np.asarray(z["lead_lens"], dtype=np.float64).reshape(-1),
                           idx=idx, ts_rel=np.asarray(z["ts_rel_s"], dtype=np.float64).reshape(-1),
                           n_rows=int(cid.shape[0]))


def lead5_for(lead, clip_id: str, raw_frame: int):
    """-> (has_lead, lead5 [5, 2] float32 tensor, lead_len_m). No lead -> the far
    sentinel held over the horizon (constant per window)."""
    far = torch.tensor([[FAR_LEAD_X_M, 0.0]] * len(GRID_S), dtype=torch.float32)
    if lead is None:
        return False, far, LEAD_LEN_DEFAULT_M
    row = lead.idx.get((clip_id, int(raw_frame)))
    if row is None or not bool(lead.has[row]):
        return False, far, LEAD_LEN_DEFAULT_M
    track = np.asarray(lead.leads[row], dtype=np.float64)
    if not np.all(np.isfinite(track)):
        return False, far, LEAD_LEN_DEFAULT_M
    return True, torch.tensor(resample_track(track, lead.ts_rel), dtype=torch.float32), float(lead.lens[row])


# --------------------------------------------------------------------------- #
# --dump: the human reference and the trivial floor from a banked suite dump    #
# --------------------------------------------------------------------------- #
def _boot(v, eid, n_boot, seed):
    r = CI.episode_cluster_bootstrap(np.asarray(v, dtype=np.float64), np.asarray(eid), n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}


def _pboot(a, b, eid, n_boot, seed):
    r = CI.paired_episode_cluster_bootstrap(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                                            np.asarray(eid), n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator")}


def score_dump(dump_dir: str, lead_path: str | None, *, arms=DUMP_ARMS, n_boot: int = 2000,
               seed: int = 0) -> dict:
    mf = os.path.join(dump_dir, "manifest.json")
    with open(mf, encoding="utf-8") as fh:
        man = json.load(fh)
    frames = (man.get("corpus", {}) or {}).get("frames", {}) or {}
    if "provider_to_raw_frame_offset" not in frames:
        raise SystemExit(f"[fan_safety] {mf} carries no corpus.frames.provider_to_raw_frame_offset — "
                         "the lead join needs it (refcv3-shaped dumps only)")
    raw_off = int(frames["provider_to_raw_frame_offset"])
    grid = man.get("grid", {})
    if list(grid.get("horizons_steps", [])) != [5, 10, 15, 20]:
        raise SystemExit(f"[fan_safety] dump grid {grid.get('horizons_steps')} is not the 2 s / 0.5 s grid")
    lead = load_lead(lead_path) if lead_path else None
    rows: list[dict] = []
    present_arms = None
    for e in man.get("episodes", []):
        fi = int(e["file_index"])
        f = os.path.join(dump_dir, f"ep{fi:03d}.npz")
        if not os.path.isfile(f):
            continue
        z = np.load(f)
        have = [a for a in arms if a in z.files]
        present_arms = have if present_arms is None else [a for a in present_arms if a in have]
        ws = np.asarray(z["ws"]).astype(int).ravel()
        v0 = np.asarray(z["v0"], dtype=np.float64).ravel()
        eid = int(np.asarray(z["eid"]).ravel()[0])
        clip = str(e.get("clip_id") or e.get("name"))
        for i, w in enumerate(ws):
            has, l5, ln = lead5_for(lead, clip, int(w) + raw_off)
            paths = {a: with_origin(torch.tensor(np.asarray(z[a][i]), dtype=torch.float32)) for a in have}
            v = torch.tensor(float(v0[i]))
            paths["frozen"] = torch.zeros(len(GRID_S), 2)
            paths["ha0_derived"] = hold_v0_path(v)           # must equal the dump's ha0 (identity check)
            rec = {"clip": clip, "ws": int(w), "eid": eid, "v0": float(v0[i]), "has_lead": bool(has),
                   "lead_len_m": float(ln)}
            for a, p in paths.items():
                sc = score_paths(p, v, l5 if has else None, lead_len_m=ln)
                for k in FLAGS:
                    rec[f"{a}_{k}"] = float(sc[k])
                rec[f"{a}_peak_g"] = float(sc["peak_g"])
                rec[f"{a}_min_ttc_s"] = float(sc["min_ttc_s"])
            rows.append(rec)
    if not rows:
        raise SystemExit(f"[fan_safety] no windows read under {dump_dir}")
    arms_out = list(present_arms or []) + ["frozen", "ha0_derived"]
    eid_all = [r["eid"] for r in rows]
    lead_rows = [r for r in rows if r["has_lead"]]
    eid_lead = [r["eid"] for r in lead_rows]
    rates = {}
    for a in arms_out:
        rates[a] = {}
        for k in FLAGS:
            pop = lead_rows if k in LEAD_ONLY else rows
            ev = eid_lead if k in LEAD_ONLY else eid_all
            rates[a][k] = _boot([r[f"{a}_{k}"] for r in pop], ev, n_boot, seed) if pop else None
        rates[a]["peak_g_mean"] = float(np.mean([r[f"{a}_peak_g"] for r in rows]))
    # the identity control on the floor construction, and the known-value control
    ha0_id = (float(np.max(np.abs([r["ha0_derived_peak_g"] - r["ha0_peak_g"] for r in rows])))
              if "ha0" in arms_out else None)
    frozen_known = {"kamm_over": rates["frozen"]["kamm_over"]["mean"],
                    "envelope": rates["frozen"]["envelope"]["mean"],
                    "PASS": rates["frozen"]["kamm_over"]["mean"] == 0.0 and rates["frozen"]["envelope"]["mean"] == 0.0}
    paired = {}
    if "os" in arms_out:
        for ref in ("g", "ha0", "ha"):
            if ref not in arms_out:
                continue
            paired[f"os_minus_{ref}"] = {}
            for k in FLAGS:
                pop = lead_rows if k in LEAD_ONLY else rows
                ev = eid_lead if k in LEAD_ONLY else eid_all
                if pop:
                    paired[f"os_minus_{ref}"][k] = _pboot([r[f"os_{k}"] for r in pop], [r[f"{ref}_{k}"] for r in pop],
                                                          ev, n_boot, seed)
    return {"_what": "fan-safety scores of the SELECTED paths banked in a refcv3_arm dump: the human "
                     "reference (g), the deployed selection (os), the trivial floors (ha, ha0), the nav "
                     "controls, and two known-value controls (frozen: kamm/envelope must read 0.0 exactly; "
                     "ha0_derived: must reproduce the dump's own ha0)",
            "_tool": TOOL, "_evidence_class": "MEASURED (ours)",
            "_tier": "T1-adjacent (the emitted plan on the eval clips, self-action OPEN loop); never a driving claim",
            "dump": os.path.abspath(dump_dir), "lead_block": lead_path, "raw_frame_rule": f"ws + {raw_off}",
            "thresholds": {"contact_radius_m": 2.0, "ttc_below_s": TTC_THRESH_S, "ttc_veto_s": TTC_VETO_S,
                           "mu_kamm": MU_KAMM, "a_max_mps2": RW.A_MAX_MPS2, "kappa_max_1pm": RW.KAPPA_MAX_1PM,
                           "reach_accel_max_mps2": A_MAX_REACH_MPS2, "reach_horizon_s": HORIZON_REACH_S},
            "populations": {"n_windows": len(rows), "n_windows_with_lead": len(lead_rows),
                            "n_episodes": len(set(eid_all)), "n_episodes_with_lead": len(set(eid_lead)),
                            "lead_only_flags": list(LEAD_ONLY), "d_points_per_path": len(GRID_S)},
            "arms": arms_out, "rates": rates, "paired_os_vs_reference": paired,
            "controls": {"frozen_known_value": frozen_known, "ha0_derived_vs_dump_ha0_max_abs_peak_g_diff": ha0_id,
                         "PASS": bool(frozen_known["PASS"] and (ha0_id is None or ha0_id < 1e-4))},
            "n_boot": n_boot, "seed": seed, "per_window": rows}


# --------------------------------------------------------------------------- #
# --pair: an arm's fan readout against the base's on the same windows           #
# --------------------------------------------------------------------------- #
def pair_readouts(base: dict, arm: dict, *, n_boot: int = 2000, seed: int = 0,
                  keys: tuple[str, ...] | None = None) -> dict:
    """Both records carry ``per_window`` rows with ``wi``, ``eid``, ``has_lead`` and the
    ``summarise_fan`` fields plus ``human_*`` / ``ha0_*``. Direction: arm - base; a
    NEGATIVE delta on a flag rate or a mass means the fan got SAFER."""
    b = {r["wi"]: r for r in base["per_window"]}
    a = {r["wi"]: r for r in arm["per_window"]}
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("[fan_safety] the two readouts share no windows")
    if keys is None:
        keys = tuple(k for k in b[common[0]] if isinstance(b[common[0]][k], (int, float))
                     and k not in ("wi", "eid", "v0", "sel_idx") and not isinstance(b[common[0]][k], bool))
    out = {"_what": "paired fan-safety deltas (arm - base) on the SAME eval windows",
           "_estimator": "paired episode-cluster bootstrap (taniteval.ci)", "_tool": TOOL,
           "n_windows_common": len(common), "n_windows_base": len(b), "n_windows_arm": len(a),
           "n_boot": n_boot, "seed": seed, "metrics": {}}
    for k in keys:
        lead_only = any(k.endswith("_" + f) for f in LEAD_ONLY)
        wis = [w for w in common if (b[w]["has_lead"] if lead_only else True)]
        if not wis:
            out["metrics"][k] = None
            continue
        av = [float(a[w][k]) for w in wis]
        bv = [float(b[w][k]) for w in wis]
        ev = [int(b[w]["eid"]) for w in wis]
        r = _pboot(av, bv, ev, n_boot, seed)
        r.update({"base_mean": float(np.mean(bv)), "arm_mean": float(np.mean(av)),
                  "population": "lead windows" if lead_only else "all windows",
                  "exactly_zero": bool(np.all(np.asarray(av) == np.asarray(bv)))})
        out["metrics"][k] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=None, help="score the selected paths of a refcv3_arm dump")
    ap.add_argument("--lead-block", default=None)
    ap.add_argument("--pair", nargs=2, metavar=("BASE_READOUT", "ARM_READOUT"), default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.dump) == bool(a.pair):
        raise SystemExit("exactly one of --dump / --pair")
    if a.dump:
        rec = score_dump(a.dump, a.lead_block, n_boot=a.n_boot, seed=a.seed)
        c = rec["controls"]
        print(f"[fan_safety] dump {rec['populations']} · controls PASS={c['PASS']} · "
              f"os contact {rec['rates']['os']['contact']} · g contact {rec['rates']['g']['contact']}"
              if "os" in rec["rates"] and "g" in rec["rates"] else f"[fan_safety] dump {rec['populations']}")
    else:
        with open(a.pair[0], encoding="utf-8") as fh:
            base = json.load(fh)
        with open(a.pair[1], encoding="utf-8") as fh:
            arm = json.load(fh)
        rec = pair_readouts(base, arm, n_boot=a.n_boot, seed=a.seed)
        sep = [k for k, v in rec["metrics"].items() if v and v["separated"]]
        print(f"[fan_safety] paired on {rec['n_windows_common']} windows · separated: {sep[:12]}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[fan_safety] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
