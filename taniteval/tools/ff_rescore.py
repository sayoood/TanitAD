#!/usr/bin/env python3
"""Rescore an ALREADY-BANKED per-window eval dump into the FOUR BINDING FAMILIES.

⛔ WHY THIS EXISTS. ``MODEL_REGISTRY.md`` §1.14 (v5.8f) is an ADE-dominated row,
and the binding rule (Sayed, 2026-08-02) is that *"any future eval must
include"* LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC **in addition to** ADE,
per-family, never pooled, each with the paired episode-cluster bootstrap. The
row itself says the gap out loud: *"NOT yet a release row: four families +
episode-cluster CIs on the banked windows, then T1 (E1.4), complete it."* The
windows are already banked, so closing it must not cost another GPU pass — this
tool consumes the dump.

WHAT IT REFUSES, AND WHY EACH REFUSAL IS LOAD-BEARING
-----------------------------------------------------
1. **A CROSS-GRID JOIN.** Two arms are comparable only on the SAME windows. A
   length-equal but differently-built grid produces a plausible number that
   scores one arm's trajectory against another episode's road — an error that
   never looks like one. The fingerprint therefore hashes the **GT tensor and
   the eid list**, not just the row count, and a mismatch REFUSES the
   comparison (the per-arm files are still written, and the refusal is BANKED
   as a record rather than only printed).
2. **AN UNSTAMPED TIER.** ``EVAL_DOCTRINE.md`` is binding: **T0 is
   teacher-forced and is a WM diagnostic, NEVER "driving performance"**; **T1 is
   the action-closed loop and the PRIMARY tier for a capability claim.** An arm
   this tool cannot resolve a tier for is a hard error, because an un-tiered
   number is exactly what the doctrine forbids.
3. **THE WRONG ESTIMATOR.** ``overlapping_holdout_se`` is refused by name.
   ⚠️ It is not only an interval problem: its central value is a
   mean-of-split-means, so it **BIASES THE POINT ESTIMATE** — MEASURED across 27
   arms, headline ``ade_0_2s`` shifted **−6.67 % to +11.69 %**, bidirectional,
   and paired deltas moved up to **×−4.15 INCLUDING A SIGN FLIP**. Every point
   estimate here is the **full_set** mean; the bootstrap supplies the interval
   and never moves the mean.

WHAT IT READS
-------------
* a **T1 dump directory** — ``ep*.npz`` per episode with ``g`` (GT) and one key
  per arm, the schema ``taniteval/tools/t1_eval.py`` writes. Each arm key
  becomes an arm named ``<label>:<key>``.
* a **``rollout.collect`` windows dump** — a ``.pt`` with ``pred``/``gt``/``eid``
  (``pred_dense``/``gt_dense`` used when present). One arm, named ``<label>``.

⚠️ **THE TACTICAL FAMILY IS TRAJECTORY-DERIVED HERE** — see
``four_families.tactical_from_trajectory``. It compares the arm's EXECUTED
manoeuvre against the human's, factored into the lateral and longitudinal axes
that the shipped 5-way softmax mixes. It is **NOT** "selected vs executed",
which needs a tactical head, and at **T0 it is substantially an action echo**
(§1.12) — the block carries both warnings itself.

⚠️ **THE STRATEGIC FAMILY IS n/a ON PhysicalAI-AV, with its reason and its n.**
Pass ``--strategic-no-label`` to declare it (it is never inferred). The corpus
carries no map, no lane graph, no junction label, no traffic-light feature and
no route signal — the card says verbatim *"we do not include open maps data"* —
and ``egomotion`` has no lat/lon, so map-matching is impossible. The instrument
that would close it is the VLM pipeline PH0→PH1→PH2.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

# path bootstrap — same convention as the other taniteval tools, so the CLI runs
# from any cwd with no preset PYTHONPATH (the CPU tests invoke it as a subprocess)
_HERE = os.path.dirname(os.path.abspath(__file__))     # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                     # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                     # <repo>
for _pth in (os.path.join(_REPO, "stack"), _TE_PARENT, _HERE):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

TOOL = "taniteval/tools/ff_rescore.py"

#: ⛔ REFUSED BY NAME. See the module docstring — it biases the POINT ESTIMATE.
BANNED_ESTIMATORS = ("overlapping_holdout_se", "jackknife", "holdout_se",
                     "8-split", "split_mean", "heldout")
ESTIMATOR = "episode_cluster_bootstrap"

#: ⛔ The per-window components whose direction is INVERTED relative to the rest.
#: Everything else in the paired table is an error (lower is better); these two
#: are decision ACCURACIES. Getting this wrong flips the reading of the TACTICAL
#: family, which is the family this rescore exists to add.
HIGHER_IS_BETTER = frozenset({"TAC_lat_decision_correct",
                              "TAC_lon_decision_correct"})

_TIER_NOTE = {
    "T0": ("TEACHER-FORCED — the predictor consumes the RECORDED future "
           "actions. A WM/prediction diagnostic ONLY; ⛔ NEVER quotable as "
           "driving performance (EVAL_DOCTRINE.md)."),
    "T1": ("ACTION-CLOSED LOOP — the predictor consumes the planner's OWN "
           "actions. The PRIMARY offline tier; a capability claim requires it."),
}


def _p(*a):
    print(*a, flush=True)


def _die(msg: str, code: int = 2):
    print(f"⛔ REFUSED: {msg}", flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# Tier resolution — an unstamped arm is a hard error, never a default          #
# --------------------------------------------------------------------------- #
def known_tiers() -> dict:
    """The arm-key -> tier table, IMPORTED from ``t1_eval`` so there is ONE.

    ⛔ Deliberately not mirrored. A second copy of a tier table is a second
    thing that can drift, and a drifted tier stamp mislabels a T0 diagnostic as
    driving performance — the precise failure the doctrine exists to prevent.
    """
    try:
        from t1_eval import DEFAULT_TIERS
        return dict(DEFAULT_TIERS)
    except Exception:
        return {}


def resolve_tier(arm_name: str, arm_key: str, overrides: dict) -> str:
    """-> "T0"/"T1". Hard error when it cannot be resolved."""
    for k in (arm_name, arm_key):
        if k in overrides:
            t = str(overrides[k]).upper()
            if t not in _TIER_NOTE:
                _die(f"arm {arm_name!r}: tier {t!r} is not T0 or T1")
            return t
    t = known_tiers().get(arm_key)
    if t:
        return str(t).upper()
    _die(f"arm {arm_name!r} (key {arm_key!r}) has NO TIER. EVAL_DOCTRINE.md is "
         f"binding: T0 (teacher-forced) is a WM diagnostic and NEVER driving "
         f"performance; T1 (action-closed) is the primary tier. An un-tiered "
         f"number is exactly what the doctrine forbids. Pass "
         f"--tier {arm_name}=T0|T1.")


# --------------------------------------------------------------------------- #
# Loading — a T1 npz dump dir, or a rollout windows .pt                        #
# --------------------------------------------------------------------------- #
_FAN_SUFFIXES = ("_fan_err", "_sel_idx", "_fan_scores")


def load_dump(label: str, path: str, gt_key: str = "g", dt: float = 0.1) -> dict:
    """-> ``{"kind":…, "gt":[N,K,2], "eid":[N], "arms": {name: (key, [N,K,2])},
    "dt_s":…, "wp_steps":…, "source":…}``. Fails loud on an empty/odd dump."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "ep*.npz")))
        if not files:
            _die(f"{label}: no ep*.npz under {path}")
        with np.load(files[0]) as d0:
            keys = list(d0.files)
        if gt_key not in keys:
            _die(f"{label}: dump has no GT key {gt_key!r}; keys={keys}")
        arm_keys = [k for k in keys if k not in (gt_key, "ws")
                    and not k.endswith(_FAN_SUFFIXES)]
        if not arm_keys:
            _die(f"{label}: dump has no arm keys beside {gt_key!r}/'ws'")
        G, eid = [], []
        A = {k: [] for k in arm_keys}
        for f in files:
            with np.load(f) as d:
                g = d[gt_key][..., :2].astype(np.float64)
                G.append(g)
                eid += [os.path.splitext(os.path.basename(f))[0]] * g.shape[0]
                for k in arm_keys:
                    if k not in d.files:
                        _die(f"{label}: {f} is missing arm key {k!r} that "
                             f"{files[0]} has — a ragged dump cannot be joined")
                    a = d[k][..., :2].astype(np.float64)
                    if a.shape != g.shape:
                        _die(f"{label}: {f}: arm {k!r} shape {a.shape} != GT "
                             f"{g.shape}")
                    A[k].append(a)
        gt = np.concatenate(G)
        return {"kind": "t1_npz_dump", "gt": gt, "eid": eid, "dt_s": dt,
                "wp_steps": None, "source": path, "n_episodes": len(files),
                "arms": {f"{label}:{k}": (k, np.concatenate(A[k]))
                         for k in arm_keys}}

    if not os.path.exists(path):
        _die(f"{label}: no such dump {path}")
    import torch
    from taniteval import rollout
    win = rollout.load_windows(path)
    if not isinstance(win, dict) or "pred" not in win or "gt" not in win:
        _die(f"{label}: {path} is not a rollout windows dump "
             f"(keys={list(win) if isinstance(win, dict) else type(win)})")
    dense = win.get("pred_dense") is not None and win.get("gt_dense") is not None
    pred = torch.as_tensor(win["pred_dense"] if dense else win["pred"]).float()
    gt = torch.as_tensor(win["gt_dense"] if dense else win["gt"]).float()
    eid = list(win.get("eid") or [])
    if len(eid) != int(pred.shape[0]):
        _die(f"{label}: {path} has {len(eid)} eids for {int(pred.shape[0])} "
             f"windows — no episode-cluster bootstrap can be formed, and a bare "
             f"point estimate is not decision-grade")
    # ⛔ dt IS DERIVED, NEVER ASSUMED. A `rollout.collect` sparse dump carries
    # wp_steps=(5,10,15,20) — a **0.5 s** grid, not the 0.1 s the module
    # constant names. Assuming 0.1 here would reinstate exactly the defect
    # four_families documents: every speed x5, every accel x25, every yaw-rate
    # x5, on a NEGATIVE CONTROL that measured GT ego speed 12.4565 m/s reported
    # as 62.9789 m/s. `infer_dt` reads the window's own sampling contract and
    # returns the provenance string, which is carried into the output.
    from taniteval import four_families as _ff
    if dense:
        dt_s, dt_prov = float(win.get("dt_s", 0.1) or 0.1), "DENSE path dt_s"
    else:
        dt_s, dt_prov = _ff.infer_dt(win)
    return {"kind": "rollout_windows" + ("_dense" if dense else "_sparse"),
            "gt": gt.numpy().astype(np.float64), "eid": eid,
            "dt_s": dt_s, "dt_provenance": dt_prov,
            "wp_steps": list(win.get("wp_steps") or []) if not dense else None,
            "source": path, "n_episodes": len(set(eid)),
            # ⭐ v0 CARRIED THROUGH (2026-08-16). `rollout.collect` publishes
            # `speed` = ego speed at t0 (ep.poses[last, 3]) and this loader used
            # to drop it. Without it the PI's anti-echo controls report
            # UNAVAILABLE on every rollout dump we have already banked — a WORK
            # ITEM manufactured by a discarded column, not by missing data.
            # ⛔ NEVER derived from gt/pred: the first GT step is a FUTURE
            # displacement (see v0_antiecho.resolve_v0).
            "v0": (np.asarray(win["speed"], dtype=np.float64).reshape(-1)
                   if win.get("speed") is not None else None),
            "arms": {label: (label, pred.numpy().astype(np.float64))}}


# --------------------------------------------------------------------------- #
# The grid fingerprint — what makes a cross-grid join detectable               #
# --------------------------------------------------------------------------- #
def fingerprint(gt: np.ndarray, eid, dt: float, wp_steps) -> dict:
    """A hash of what a comparison must AGREE on: the windows themselves.

    ⛔ The row COUNT is not the grid. Two dumps can have the same N and score
    different windows — the failure mode is a plausible number, not an
    exception. Hashing the GT tensor bytes and the eid sequence is what makes
    that detectable: identical windows imply an identical ground truth.
    ``gt`` is rounded to 1e-6 m before hashing so a float32/float64 round-trip
    through two dump formats does not read as a different corpus.
    """
    g = np.ascontiguousarray(np.round(np.asarray(gt, dtype=np.float64), 6))
    return {
        "n_windows": int(g.shape[0]),
        "horizon_steps": int(g.shape[1]),
        "dt_s": float(dt),
        "wp_steps": list(wp_steps) if wp_steps else None,
        "n_episodes": int(len(set(eid))),
        "eid_sha1": hashlib.sha1("|".join(map(str, eid)).encode()).hexdigest()[:16],
        "gt_sha1": hashlib.sha1(g.tobytes()).hexdigest()[:16],
    }


def same_grid(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in
               ("n_windows", "horizon_steps", "dt_s", "eid_sha1", "gt_sha1"))


# --------------------------------------------------------------------------- #
# The lead block — the distance-keeping half of LONGITUDINAL                   #
# --------------------------------------------------------------------------- #
def canon_eid(eid) -> list:
    """Episode ids reduced to a comparable form: the trailing integer when every
    entry has one, else the raw strings.

    ``rollout.collect`` writes integer indices (``0``) and
    ``tools/build_lead_block.py`` writes ``ep_00000`` — the same episodes under
    two conventions. Comparing the raw text would refuse a valid join.
    """
    import re
    out = []
    for x in eid:
        m = re.search(r"(\d+)\s*$", str(x))
        if not m:
            return [str(x) for x in eid]
        out.append(int(m.group(1)))
    return out


def _runs(seq) -> list:
    """Run-length structure ``[(rank_of_first_appearance, length), …]`` — the
    episode PARTITION a positional join actually depends on, independent of what
    the episodes are called."""
    out, order = [], {}
    for x in map(str, seq):
        if x not in order:
            order[x] = len(order)
        r = order[x]
        if out and out[-1][0] == r:
            out[-1][1] += 1
        else:
            out.append([r, 1])
    return [tuple(v) for v in out]


def load_lead(path: str, n_windows: int, horizon: int, dt: float,
              eid) -> tuple[dict, list | None]:
    """-> (lead block for ``four_families``, step indices to score it on).

    ⛔ TWO joins can go wrong here and both are silent. (a) ROW count: a lead
    block built on another stride scores this arm against another episode's
    traffic. (b) TIME GRID: the banked val40 block is on the SPARSE
    ``ts_rel_s`` = (0.5, 1.0, 1.5, 2.0) grid while a T1 dump is dense at 0.1 s,
    so the lead's K and the path's K differ. Rather than truncate — which would
    silently compare a 2 s path against a 0.4 s lead track — the block's own
    ``ts_rel_s`` is mapped to step indices on THIS grid, and anything that does
    not map exactly is refused.
    """
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        lead = {k: z[k] for k in z.files}
    else:
        import torch
        lead = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("leads", "lead_lens", "speeds"):
        if k not in lead:
            _die(f"lead block {path} has no {k!r}; keys={sorted(lead)}")
    leads = np.asarray(lead["leads"], dtype=np.float64)
    if leads.shape[0] != n_windows:
        _die(f"lead block has {leads.shape[0]} rows for {n_windows} scored "
             f"windows — different window grids. Rebuild it against this "
             f"corpus/stride (taniteval/tools/build_lead_block.py); do NOT "
             f"truncate: every row after the first divergence would score this "
             f"arm against another episode's traffic.")
    eid_note = None
    if "eid" in lead:
        le = [str(x) for x in np.asarray(lead["eid"]).reshape(-1)]
        de = [str(x) for x in eid]
        if canon_eid(le) != canon_eid(de):
            # ⚠️ A NAME difference is not a GRID difference. `rollout.collect`
            # writes integer episode indices (0, 1, …) while the lead builder
            # writes `ep_00000` — the SAME 40 episodes under two conventions, and
            # refusing on the label text alone would block a perfectly valid
            # join. The fallback test is therefore structural: the run-length
            # PARTITION (how many consecutive windows each episode owns, in
            # order) must match exactly. That is what a positional join actually
            # depends on. If even that differs, the rows are genuinely not the
            # same windows and the join is refused.
            if _runs(le) != _runs(de):
                _die("lead block's episode PARTITION differs from the dump's — "
                     "same row count, DIFFERENT windows. That is the cross-grid "
                     "join this tool exists to refuse. Rebuild the lead block "
                     "against this corpus/stride; do NOT truncate.")
            eid_note = (f"lead eid labels differ from the dump's "
                        f"({le[0]!r} vs {de[0]!r}) but the run-length episode "
                        f"partition is IDENTICAL, so the positional join is "
                        f"valid. Naming convention only.")
    kl = int(leads.shape[1])
    steps = None
    if kl != horizon:
        ts = lead.get("ts_rel_s")
        if ts is None:
            _die(f"lead block has K={kl} but this grid has K={horizon} and the "
                 f"block carries no 'ts_rel_s' to map it — refusing to guess.")
        ts = np.asarray(ts, dtype=np.float64).reshape(-1)
        if ts.size != kl:
            _die(f"lead block ts_rel_s has {ts.size} entries for K={kl}")
        idx = np.rint(ts / dt).astype(int) - 1
        if idx.min() < 0 or idx.max() >= horizon or \
                not np.allclose(ts, (idx + 1) * dt, atol=1e-6):
            _die(f"lead block ts_rel_s {ts.tolist()} does not land on this "
                 f"grid's steps (dt={dt}, K={horizon}) — refusing an "
                 f"approximate time join.")
        steps = idx.tolist()
    blk = {"leads": leads,
           "lead_lens": np.asarray(lead["lead_lens"], dtype=np.float64),
           "speeds": np.asarray(lead["speeds"], dtype=np.float64),
           "state": (np.asarray(lead["state"]) if "state" in lead else None),
           "eid": [str(x) for x in eid]}
    if eid_note:
        blk["_eid_note"] = eid_note
    return blk, steps


# --------------------------------------------------------------------------- #
# Per-arm scoring                                                              #
# --------------------------------------------------------------------------- #
def score_arm(name: str, arm_key: str, pred: np.ndarray, gt: np.ndarray,
              eid, dt: float, tier: str, *, wp_steps=None, lead=None,
              lead_steps=None, strategic_no_label: bool = False,
              v0=None, n_boot: int = 2000, seed: int = 0) -> tuple[dict, dict]:
    """-> (the arm's record, its per-window components for the paired test)."""
    import torch

    from taniteval import ci as _ci
    from taniteval import four_families as ff

    pred_t = torch.as_tensor(pred).float()
    gt_t = torch.as_tensor(gt).float()
    K = int(gt_t.shape[1])
    # ⛔ `wp_steps` is the MODEL-TICK contract of the supplied columns, not an
    # index list. When it already has one entry per column the array IS the
    # sparse view (a `rollout.collect` dump: 4 columns at ticks 5/10/15/20) and
    # there is nothing to sub-select — indexing by tick would run off the end.
    # Only a DENSE array needs quartile picks to form the sparse companion view.
    if wp_steps and len(wp_steps) == K:
        idx, contract = list(range(K)), list(wp_steps)
    else:
        idx = sorted({max(0, int(round(K * q)) - 1) for q in (.25, .5, .75, 1.0)})
        contract = [i + 1 for i in idx]
    win = {"pred_dense": pred_t, "gt_dense": gt_t,
           "pred": pred_t[:, idx], "gt": gt_t[:, idx],
           "wp_steps": contract, "dt_s": dt, "eid": list(eid)}
    # ⭐ v0 for the PI's anti-echo controls (2026-08-16). Absent => they report
    # UNAVAILABLE with reason + n; a lead block also supplies it via
    # lead['speeds'], which four_families._anti_echo folds in automatically.
    if v0 is not None:
        win["v0"] = np.asarray(v0, dtype=np.float64).reshape(-1)
    if lead is not None:
        lb = dict(lead)
        lb["n_boot"] = n_boot
        lb["seed"] = seed
        if lead_steps is not None:
            # ⛔ the lead track is on a COARSER time grid than the path. Declare
            # the steps it is defined on and let four_families do the TIME join;
            # a truncation would score a 2 s lead track against 0.4 s of path.
            lb["path_steps"] = list(lead_steps)
            lb["dt_s"] = (dt * (lead_steps[1] - lead_steps[0])
                          if len(lead_steps) > 1 else dt)
        win["lead"] = lb

    fam = ff.all_families(win, tactical_from_traj=True,
                          strategic_no_label=strategic_no_label,
                          tier=tier, n_boot=n_boot, seed=seed)

    # -- per-window components -> ONE episode-cluster resampling -------------- #
    # ⛔ computed with four_families' OWN _seq_geometry, never a re-derivation:
    # a second implementation lets the interval and the point estimate drift
    # apart silently, which is the failure the estimator rule exists to prevent.
    P, G = ff._seq_geometry(pred_t, dt), ff._seq_geometry(gt_t, dt)
    comps = {
        "ade_dense_m": torch.linalg.norm(pred_t - gt_t, dim=-1).mean(1).numpy(),
        "fde_last_m": torch.linalg.norm(pred_t[:, -1] - gt_t[:, -1], dim=-1).numpy(),
        "LON_speed_mae_mps": (P["speed"] - G["speed"]).abs().mean(1).numpy(),
        "LON_along_mae_m": (P["along"] - G["along"]).abs().mean(1).numpy(),
        "LAT_cross_mae_m": (P["cross"] - G["cross"]).abs().mean(1).numpy(),
        "TAC_goal_point_error_m": torch.linalg.norm(
            pred_t[:, -1] - gt_t[:, -1], dim=-1).numpy(),
    }
    # the TACTICAL decision indicators, so the paired test can move on a
    # DECISION and not only on a displacement
    tac = fam.get("tactical", {})
    if tac.get("status") == "OK" and "lateral_decision" in tac:
        dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(pred_t, dt)
        dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(gt_t, dt)
        from tanitad.refs.refc_tactical import factor_from_kinematics
        lp, np_ = factor_from_kinematics(dyp, dvp, v0p, v1p)
        lg, ng = factor_from_kinematics(dyg, dvg, v0g, v1g)
        comps["TAC_lat_decision_correct"] = (lp == lg).float().numpy()
        comps["TAC_lon_decision_correct"] = (np_ == ng).float().numpy()

    boots = _ci.bootstrap_metrics({k: (v, "mean") for k, v in comps.items()},
                                  eid, n_boot=n_boot, seed=seed)
    both = P["valid"] & G["valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    nv = both.sum(1)
    head_w = (torch.where(nv > 0, (dh.abs() * both).sum(1) / nv.clamp_min(1),
                          torch.nan).numpy() * 180.0 / np.pi)
    keep = ~np.isnan(head_w)
    if keep.any():
        boots["LAT_heading_mae_deg"] = _ci.episode_cluster_bootstrap(
            head_w[keep], [e for e, k in zip(eid, keep) if k],
            n_boot=n_boot, seed=seed)
        boots["LAT_heading_mae_deg"]["n_windows_dropped_no_valid_step"] = \
            int((~keep).sum())
    for b in boots.values():
        b["tier"] = tier

    rec = {
        "tool": TOOL,
        "arm": name,
        "arm_key": arm_key,
        "tier": tier,
        "tier_note": _TIER_NOTE[tier],
        "n_windows": int(gt_t.shape[0]),
        "n_episodes": int(len(set(eid))),
        "horizon_steps": K,
        "dt_s": dt,
        "four_families": fam,
        "intervals": {"tier": tier, "estimator": ESTIMATOR,
                      "n_windows": int(gt_t.shape[0]),
                      "point_estimate": "full_set pooled mean over windows",
                      "metrics": boots},
        "_estimator": (
            "point estimates are FULL_SET pooled means; intervals are the "
            "episode-cluster bootstrap (taniteval.ci). ⛔ overlapping_holdout_se "
            "is NOT used anywhere: it biases the POINT ESTIMATE, not only the "
            "interval (MEASURED −6.67 % to +11.69 % on headline ade_0_2s across "
            "27 arms, bidirectional; up to ×−4.15 with a SIGN FLIP on paired "
            "deltas)."),
        "_tier_doctrine": (
            "EVAL_DOCTRINE.md — T1 (action-closed) is the PRIMARY tier for any "
            "capability claim; T0 (teacher-forced) is a WM diagnostic and is "
            "⛔ NEVER quotable as driving performance. Comparisons across tiers "
            "are invalid."),
        "_binding": (
            "Sayed 2026-08-02 — LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC "
            "in ADDITION to ADE, per-family, never pooled. A family reported "
            "UNAVAILABLE is a WORK ITEM, not a pass; one that genuinely cannot "
            "be computed states its REASON and its n."),
    }
    return rec, comps


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rescore banked per-window dumps into the four binding "
                    "families with episode-cluster bootstrap CIs.")
    ap.add_argument("--dump", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable. PATH is a T1 dump DIR (ep*.npz) or a "
                         "rollout windows .pt")
    ap.add_argument("--tier", action="append", default=[], metavar="NAME=T0|T1",
                    help="repeatable. NAME is the full arm name (LABEL:key) or "
                         "the bare arm key. ⛔ An arm with no resolvable tier is "
                         "a hard error.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lead", default=None,
                    help="lead block (.npz/.pt) for the distance-keeping half "
                         "of LONGITUDINAL. Without it that half is UNAVAILABLE "
                         "— a WORK ITEM, not a pass.")
    ap.add_argument("--strategic-no-label", action="store_true",
                    help="declare that this corpus carries no admissible "
                         "strategic label (true of PhysicalAI-AV). Never "
                         "inferred: the fact is a property of the CORPUS.")
    ap.add_argument("--paired", action="append", default=[], metavar="A,B",
                    help="repeatable. Paired delta B - A on the SAME windows. "
                         "Default: every ordered pair on a shared grid.")
    ap.add_argument("--gt-key", default="g")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimator", default=ESTIMATOR,
                    help="the ONLY accepted value is episode_cluster_bootstrap")
    ap.add_argument("--include-excluded", action="store_true",
                    help="permit scoring BOTH members of a known same-model "
                         "dump pair (results/dump_exclusions.json) side by "
                         "side — e.g. to verify the pair is equal. Without "
                         "this flag such a pair is REFUSED: two names, one "
                         "model, double-counted (C126).")
    a = ap.parse_args(argv)

    if a.estimator != ESTIMATOR:
        _die(f"estimator {a.estimator!r} is not accepted. The decision-grade "
             f"interval is the {ESTIMATOR} (taniteval/ci.py). "
             + ("⚠️ overlapping_holdout_se is NOT merely a wider interval — it "
                "BIASES THE POINT ESTIMATE (mean-of-split-means instead of the "
                "full_set mean): MEASURED −6.67 % to +11.69 % on headline "
                "ade_0_2s across 27 arms, bidirectional, and up to ×−4.15 on "
                "paired deltas INCLUDING A SIGN FLIP."
                if any(b in a.estimator.lower() for b in BANNED_ESTIMATORS)
                else ""))

    overrides = {}
    for t in a.tier:
        if "=" not in t:
            _die(f"--tier expects NAME=T0|T1, got {t!r}")
        k, v = t.split("=", 1)
        overrides[k.strip()] = v.strip()

    os.makedirs(a.out_dir, exist_ok=True)

    # ⛔ Duplicate-VALUE guard (C126). The name-uniqueness check below cannot
    # see that two DIFFERENT names are one model — exactly how "27 arms" was
    # 27 dumps over 25 distinct arms. `dump_exclusions.json` beside the dumps
    # is the machine-readable equality knowledge; consult it BEFORE the
    # expensive part, and fail loud rather than score one model twice.
    from taniteval import dump_census as _census  # noqa: E402 — after bootstrap
    _pt_paths = [s.split("=", 1)[1].strip() for s in a.dump
                 if "=" in s and s.split("=", 1)[1].strip().endswith(".pt")]
    if _pt_paths:
        _findings, _consulted = _census.check_explicit(_pt_paths)
        if not _consulted:
            _p("[census] no dump_exclusions.json beside the passed .pt dumps "
               "-- duplicate-VALUE check unavailable for them")
        _pairs = [f for f in _findings if f["kind"] == "pair_present"]
        for f0 in _findings:
            _p(f"[census] NOTE: {f0['excluded']} is a recorded same-model "
               f"duplicate of {f0['canonical']} ({f0['exclusions_file']})")
        if _pairs and not a.include_excluded:
            _die("the passed dumps include BOTH members of "
                 + (f"{len(_pairs)} known same-model pair(s): " if len(_pairs) > 1
                    else "a known same-model pair: ")
                 + "; ".join(f"{f['excluded']} == {f['canonical']}"
                             for f in _pairs)
                 + ". Scoring them side by side double-counts one model "
                   "(dump_exclusions.json; DUPLICATES.md has the evidence). "
                   "Drop one, or pass --include-excluded if the point IS the "
                   "pair (e.g. verifying their equality).")

    dumps, arms = [], {}
    for spec in a.dump:
        if "=" not in spec:
            _die(f"--dump expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        d = load_dump(label.strip(), path.strip(), a.gt_key, a.dt)
        d["label"] = label.strip()
        d["fingerprint"] = fingerprint(d["gt"], d["eid"], d["dt_s"], d["wp_steps"])
        dumps.append(d)
        _p(f"[dump] {label.strip():24s} {d['kind']:22s} "
           f"n={d['fingerprint']['n_windows']} K={d['fingerprint']['horizon_steps']} "
           f"eps={d['fingerprint']['n_episodes']} arms={list(d['arms'])}")
        for name, (key, pred) in d["arms"].items():
            if name in arms:
                _die(f"duplicate arm name {name!r}")
            arms[name] = (key, pred, d)

    lead, lead_steps = None, None
    if a.lead:
        d0 = dumps[0]
        lead, lead_steps = load_lead(a.lead, d0["fingerprint"]["n_windows"],
                                     d0["fingerprint"]["horizon_steps"],
                                     d0["dt_s"], d0["eid"])
        _p(f"[lead] attached from {a.lead}"
           + (f"  scored on steps {lead_steps}" if lead_steps else ""))

    written, comps_by_arm, tiers = {}, {}, {}
    for name, (key, pred, d) in arms.items():
        tier = resolve_tier(name, key, overrides)
        tiers[name] = tier
        rec, comps = score_arm(
            name, key, pred, d["gt"], d["eid"], d["dt_s"], tier,
            wp_steps=d["wp_steps"], lead=lead, lead_steps=lead_steps,
            strategic_no_label=a.strategic_no_label, v0=d.get("v0"),
            n_boot=a.n_boot, seed=a.seed)
        rec["source"] = d["source"]
        rec["dump_kind"] = d["kind"]
        rec["dt_provenance"] = d.get("dt_provenance", "npz dump; --dt flag")
        rec["grid_fingerprint"] = d["fingerprint"]
        rec["lead_block"] = a.lead
        out = os.path.join(a.out_dir,
                           f"ff_{name.replace(':', '_').replace('/', '_')}.json")
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        written[name] = out
        comps_by_arm[name] = comps
        fam = rec["four_families"]
        _p(f"[arm] {name:24s} tier={tier}  ade={rec['intervals']['metrics']['ade_dense_m']['mean']}"
           f"  unavailable={fam['_families_unavailable']}"
           f"  rule_satisfied={fam['_rule_satisfied']}  -> {out}")

    # ---- the combined comparison, or a BANKED REFUSAL --------------------- #
    from taniteval import ci as _ci
    names = list(arms)
    groups: dict[str, list] = {}
    for n in names:
        groups.setdefault(arms[n][2]["fingerprint"]["gt_sha1"], []).append(n)
    comp = {
        "tool": TOOL,
        "arms": {n: {"tier": tiers[n], "file": written[n],
                     "grid": arms[n][2]["fingerprint"]} for n in names},
        "_estimator": f"paired_{ESTIMATOR} on the SAME resampled episodes",
        "_binding": ("per-family, never pooled; every number carries its tier "
                     "and its n"),
    }
    if len(groups) > 1:
        comp["status"] = "REFUSED"
        comp["reason"] = (
            "⛔ CROSS-GRID JOIN REFUSED. The arms are not scored on the same "
            "windows (distinct GT/eid fingerprints), so no paired comparison "
            "between them is valid. A length-equal but differently-built grid "
            "produces a PLAUSIBLE number, not an error — which is why this is "
            "refused rather than warned. The per-arm records above are each "
            "valid on their OWN grid.")
        comp["grid_groups"] = {k: v for k, v in groups.items()}
        path = os.path.join(a.out_dir, "ff_comparison.json")
        with open(path, "w") as f:
            json.dump(comp, f, indent=1, default=str)
        _p(f"[comparison] {path}")
        _die(f"arms span {len(groups)} distinct window grids: "
             f"{ {k: v for k, v in groups.items()} }. Refusal BANKED at {path}.",
             code=3)

    pairs = []
    for spec in a.paired:
        if "," not in spec:
            _die(f"--paired expects A,B got {spec!r}")
        x, y = (s.strip() for s in spec.split(",", 1))
        for s in (x, y):
            if s not in arms:
                _die(f"--paired names unknown arm {s!r}; known: {names}")
        pairs.append((x, y))
    if not pairs:
        pairs = [(names[i], names[j])
                 for i in range(len(names)) for j in range(len(names)) if i < j]

    comp["status"] = "OK"
    eid0 = arms[names[0]][2]["eid"] if names else []
    comp["paired"] = {}
    for x, y in pairs:
        key = f"{y}_minus_{x}"
        blk = {"direction": f"{y} - {x}",
               "tier": (tiers[x] if tiers[x] == tiers[y]
                        else f"{tiers[y]} minus {tiers[x]}"),
               "estimator": f"paired_{ESTIMATOR}"}
        # ⛔ PER-METRIC, never blanket. Most rows here are ERRORS (lower is
        # better) but the TACTICAL decision rows are ACCURACIES (higher is
        # better), so a single "negative = better" sentence would invert the
        # reading of exactly the family this rescore exists to add.
        blk["sign_convention"] = {
            m: ("ACCURACY — higher is better, so POSITIVE delta = the second "
                "arm is BETTER" if m in HIGHER_IS_BETTER else
                "ERROR — lower is better, so NEGATIVE delta = the second arm "
                "is BETTER")
            for m in sorted(set(comps_by_arm[x]) & set(comps_by_arm[y]))}
        if tiers[x] != tiers[y]:
            blk["⛔_cross_tier_warning"] = (
                f"{tiers[y]} vs {tiers[x]} — this delta measures the TIER as "
                f"much as the arm. T0 is teacher-forced and T1 is action-closed; "
                f"EVAL_DOCTRINE.md forbids reading a cross-tier delta as a "
                f"capability comparison. It is a diagnostic (the §1.12 "
                f"action-echo contrast), never a leaderboard row.")
        shared = set(comps_by_arm[x]) & set(comps_by_arm[y])
        for m in sorted(shared):
            blk[m] = _ci.paired_episode_cluster_bootstrap(
                comps_by_arm[y][m], comps_by_arm[x][m], eid0,
                n_boot=a.n_boot, seed=a.seed)
        comp["paired"][key] = blk
        _p(f"[paired] {key:44s} ade Δ={blk['ade_dense_m']['delta']} "
           f"[{blk['ade_dense_m']['lo']}, {blk['ade_dense_m']['hi']}] "
           f"separated={blk['ade_dense_m']['separated']}")

    path = os.path.join(a.out_dir, "ff_comparison.json")
    with open(path, "w") as f:
        json.dump(comp, f, indent=1, default=str)
    _p(f"[comparison] {path}")
    _p(f"FF_EXIT=0  arms={len(names)}  pairs={len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
