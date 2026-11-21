#!/usr/bin/env python3
"""X2 — THE BAND-SEAM PROBE. **Verify, never repair.** (fix F-16)

⛔ WHAT THIS IS FOR. ``DIAGRAM_CONFORMANCE.md``:131 records that v6's 0–2 s and
2–6 s bands are seam-free **by construction** (they are SLICES of ONE 60-step
``(a, κ)`` @ 10 Hz rollout; ``V6Config.split_bands`` returns views) and that the
claim was ⬜ **verified by no instrument at all** — the trainer's
``plan_ade_0_2s`` / ``plan_ade_2_6s`` are two band ADEs, and two band ADEs cannot
see a discontinuity. This tool is the instrument. It is built to FALSIFY the
claim: read ``taniteval/taniteval/seam.py``'s docstring for the null, the
statistic, the two nulls, the three-valued verdict and the power statement.

⛔ **IT MEASURES AND NOTHING ELSE.** No loss term, no gradient, no repair. If it
finds a seam the finding is REPORTED and the architecture is re-examined; the
one thing that must never happen is a seam-repair term appearing because this
tool went off.

TIER
----
The continuity blocks consume the EMITTED plan alone — no recorded future
actions, no future frames, no ground truth — so they are tier-invariant and are
stamped ``T1`` (the primary tier). The **band-error** block compares the plan
against the GT future, so it inherits the dump's declared tier; **an undeclared
tier is a hard refusal**, because an un-tiered number is exactly what
``EVAL_DOCTRINE.md`` forbids.

THE DUMP SCHEMA (``.pt`` via ``torch.load`` **or** ``.npz`` via ``np.load``)
----------------------------------------------------------------------------
    eid        [N]            episode id per window   — REQUIRED (the CI's
                              resampling unit; windows inside one episode are
                              strongly dependent, so the episode is the
                              independent unit)
    controls   [N, T, 2] or [N, C, T, 2]   the emitted (a, κ) sequence
    waypoints  [N, T, 2] or [N, C, T, 2]   the integrated ego waypoints
                              (at least ONE of controls/waypoints is required;
                              supply both — they fail differently, see
                              ``seam.control_channels``)
    sel        [N]            the selector's pick — REQUIRED when a fan (the
                              [N, C, T, 2] shape) is present and
                              ``--candidate winner``
    gt         [N, T, 2]      GT ego-frame future waypoints — optional; enables
                              the per-band ADE block and the ``err`` channel
    tier       str            "T0" | "T1"  (or pass ``--tier``)
    arm        str            label       (or pass ``--arm``)
    plan_steps / dt / op_band_s / tac_band_s   optional geometry declaration;
                              when present it is CHECKED against the probed
                              boundary and a mismatch is a refusal

HOW TO PRODUCE A DUMP FROM v6 (the producer side, six lines)
------------------------------------------------------------
``V6Stack.emit`` already returns everything the schema needs::

    out = stack.emit(z_op, e_g_tac, v0)            # no GT, no future actions
    torch.save({"controls": out["controls"].cpu(),   # [B, N, 60, 2]
                "waypoints": out["waypoints"].cpu(), # [B, N, 60, 2]
                "sel": out["sel_score"].argmax(-1).cpu() if "sel_score" in out
                       else torch.zeros(len(v0), dtype=torch.long),
                "gt": plan_target.cpu(),             # [B, 60, 2] if available
                "eid": eids, "tier": "T1", "arm": "<run>@<step>"}, path)

⚠️ ``emit`` is the ONLY place the 60-step plan exists; nothing else in the
programme banks it, so there is no pre-existing artifact this tool can be
pointed at. That is stated plainly rather than demonstrated on something else.

SELF-TEST (no GPU, no data, no checkpoint)
------------------------------------------
``--self-test`` runs the instrument against three synthetic arms and ASSERTS the
outcome, exiting non-zero if the validation fails:

  ``genuine``           ONE control sequence, ONE ``unicycle_rollout``
  ``stitch_controls``   two INDEPENDENT control sequences concatenated, then
                        integrated by ONE rollout — position stays continuous,
                        the defect lives in control space
  ``stitch_rollout``    two INDEPENDENT rollouts concatenated, the second
                        re-based at its own origin — the classic position jump

An instrument never shown to detect the defect it hunts is not validated, so the
self-test is part of the tool rather than only part of the test suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# path bootstrap, the taniteval-tools convention: derive the package parent and
# the stack from THIS file's location so the CLI runs from any cwd with no
# preset PYTHONPATH (the tests invoke it as a subprocess).
_HERE = os.path.dirname(os.path.abspath(__file__))     # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                    # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                    # <repo>
for _pth in (os.path.join(_REPO, "stack"),
             os.path.join(_REPO, "stack", "scripts"), _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

from taniteval import seam as _seam            # noqa: E402
from taniteval import ci as _ci                # noqa: E402

TIERS = ("T0", "T1")
#: the self-test runs THREE arms, so it defaults to a cheaper bootstrap than a
#: decision run. Stamped into its record; ``--n-boot`` overrides it.
SELFTEST_N_BOOT = 400


def _p(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def load_dump(path: str) -> dict:
    """Load a seam dump from EITHER container the programme actually writes.

    ⛔ WHY THIS IS NOT JUST ``torch.load``. The identical lesson was MEASURED
    2026-08-16 on the lead block (``eval_four_families.load_lead_block``): one
    builder ends in ``torch.save`` (a ``.pt`` zip) and another in ``np.savez``,
    and ``torch.load`` on an ``.npz`` does not degrade — it raises
    ``RuntimeError: file in archive is not in a subdirectory``, so a perfectly
    good artifact reads as "cannot be scored". Returns a plain dict so no
    caller can depend on the container (``NpzFile`` is lazy and its ``.get``
    semantics differ from ``dict``'s, which is how an optional key goes missing
    without an error).
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    import torch                                        # noqa: PLC0415
    return dict(torch.load(path, map_location="cpu", weights_only=False))


def _as_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _scalar(v, default=None):
    """Unwrap a 0-d numpy array / bytes / str the two containers disagree on."""
    if v is None:
        return default
    a = np.asarray(v)
    if a.ndim == 0:
        v = a.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return v


def pick_candidate(x, sel, mode: str, name: str):
    """[N, T, 2] straight through; [N, C, T, 2] reduced per ``--candidate``.

    ``winner`` needs ``sel`` — the EMITTED WINNER is what F-16 asks for ("a 2 s
    boundary continuity verifier on the emitted winner"), and guessing candidate
    0 when a selector exists would silently probe a trajectory the planner never
    proposed.
    """
    a = _as_np(x).astype(np.float64)
    if a.ndim == 3:
        return a, "single"
    if a.ndim != 4 or a.shape[-1] != 2:
        raise SystemExit(f"{name} must be [N, T, 2] or [N, C, T, 2], got "
                         f"{a.shape}")
    n, c = a.shape[0], a.shape[1]
    if mode == "all":
        return a.reshape(n * c, a.shape[2], 2), f"all({c})"
    if mode == "winner":
        if sel is None:
            raise SystemExit(
                f"{name} is a fan [N, {c}, T, 2] but the dump carries no "
                f"'sel'. --candidate winner probes the EMITTED WINNER; without "
                f"a selector index there is no winner. Pass --candidate 0 to "
                f"probe a named candidate, or --candidate all, and say so in "
                f"the report.")
        s = _as_np(sel).astype(np.int64).reshape(-1)
        if s.shape[0] != n:
            raise SystemExit(f"sel must be [N={n}], got {s.shape}")
        if s.min() < 0 or s.max() >= c:
            raise SystemExit(f"sel out of range [0, {c}): {s.min()}..{s.max()}")
        return a[np.arange(n), s], "winner"
    idx = int(mode)
    if not 0 <= idx < c:
        raise SystemExit(f"--candidate {idx} out of range [0, {c})")
    return a[:, idx], f"index{idx}"


def expand_eid(eid, factor: int):
    """Repeat the eid when ``--candidate all`` multiplies the row count.

    ⚠️ Candidates of one window are NOT independent, so the episode stays the
    resampling unit and every candidate of a window keeps that window's episode
    id. The interval therefore does not shrink by ``sqrt(C)`` — which it would
    if the candidates were (wrongly) treated as new episodes.
    """
    e = [str(x) for x in _as_np(eid).reshape(-1).tolist()]
    if factor == 1:
        return e
    return [x for x in e for _ in range(factor)]


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #
def run_probe(dump: dict, *, arm: str, tier: str, candidate: str,
              seam: int | None, orders, local_halfwidth: int,
              materiality_k: float, n_boot: int, seed: int, alpha: float,
              scan: bool, scan_channel: str | None, scan_order: int,
              scan_n_boot: int, scan_stride: int,
              source: str | None = None) -> dict:
    """The whole probe on one dump -> the record a registry row quotes."""
    if "eid" not in dump:
        raise SystemExit(
            "the dump carries no 'eid'. The interval is the EPISODE-cluster "
            "bootstrap and the episode is the independent unit — without "
            "episode ids there is no admissible interval, and a per-window "
            "bootstrap would report a confidently wrong one.")
    ctl_raw = dump.get("controls")
    wp_raw = dump.get("waypoints")
    if ctl_raw is None and wp_raw is None:
        raise SystemExit("the dump carries neither 'controls' nor 'waypoints' "
                         "— there is no plan to probe.")
    sel = dump.get("sel")

    ctl = wp = None
    mode = None
    factor = 1
    if ctl_raw is not None:
        ctl, mode = pick_candidate(ctl_raw, sel, candidate, "controls")
    if wp_raw is not None:
        wp, m2 = pick_candidate(wp_raw, sel, candidate, "waypoints")
        mode = mode or m2
    ref = ctl if ctl is not None else wp
    n_rows, t_steps = ref.shape[0], ref.shape[1]
    if candidate == "all":
        base_n = _as_np(dump["eid"]).reshape(-1).shape[0]
        factor = n_rows // max(1, base_n)
    eid = expand_eid(dump["eid"], factor)
    if len(eid) != n_rows:
        raise SystemExit(f"eid length {len(eid)} != {n_rows} rows after "
                         f"candidate selection ({mode})")

    # --- geometry: DERIVE the seam, then check the dump's declaration --------
    dt = float(_scalar(dump.get("dt"), _seam.DT))
    op_b = dump.get("op_band_s")
    tac_b = dump.get("tac_band_s")
    if op_b is not None and tac_b is not None:
        derived = _seam.seam_boundary_of(
            [float(v) for v in _as_np(op_b).reshape(-1)],
            [float(v) for v in _as_np(tac_b).reshape(-1)], dt)
    else:
        derived = _seam.seam_boundary_of(dt=dt)
    seam_i = int(seam) if seam is not None else derived
    if seam is not None and seam_i != derived:
        _p(f"⚠️ --seam {seam_i} OVERRIDES the geometry-derived boundary "
           f"{derived}. Recorded as an override; a boundary that is not the "
           f"band edge does not answer the X2 question.")
    declared_steps = _scalar(dump.get("plan_steps"))
    if declared_steps is not None and int(declared_steps) != int(t_steps):
        raise SystemExit(f"the dump declares plan_steps={int(declared_steps)} "
                         f"but the arrays carry {t_steps} steps — refusing "
                         f"rather than probing a geometry nobody declared.")
    if not 0 < seam_i < t_steps:
        raise SystemExit(f"seam boundary {seam_i} is outside the {t_steps}-step "
                         f"plan")

    # --- channels ------------------------------------------------------------
    gt = dump.get("gt")
    extra = {}
    gt_np = None
    if gt is not None and wp is not None:
        gt_np = _as_np(gt).astype(np.float64)
        if gt_np.ndim == 3 and factor > 1:
            gt_np = np.repeat(gt_np, factor, axis=0)
        if gt_np.shape == wp.shape:
            # the ERROR series is a channel too: does the error JUMP at the
            # boundary? Same machinery, no new statistics.
            extra["err"] = np.linalg.norm(wp - gt_np, axis=-1)
        else:
            _p(f"⚠️ gt shape {gt_np.shape} != waypoints {wp.shape} — the band "
               f"block and the 'err' channel are UNAVAILABLE (reason recorded)")
            gt_np = None
    chans = _seam.control_channels(controls=ctl, waypoints=wp, extra=extra)
    units = chans["units"]
    units.setdefault("err", "m")

    panel = _seam.continuity_panel(
        chans["channels"], eid, units=units, seam=seam_i, orders=orders,
        halfwidths=(None, local_halfwidth), materiality_k=materiality_k,
        n_boot=n_boot, seed=seed, alpha=alpha)

    scan_block = None
    if scan:
        sc = scan_channel or ("a" if "a" in chans["channels"]
                              else sorted(chans["channels"])[0])
        if sc not in chans["channels"]:
            raise SystemExit(f"--scan-channel {sc!r} not among "
                             f"{sorted(chans['channels'])}")
        scan_block = _seam.boundary_scan(
            chans["channels"][sc], eid, seam=seam_i, order=scan_order,
            channel=sc, materiality_k=materiality_k, n_boot=scan_n_boot,
            seed=seed, alpha=alpha, stride=scan_stride)

    bands = None
    band_reason = None
    if gt_np is not None and wp is not None:
        bands = _seam.band_errors(wp, gt_np, eid, seam=seam_i, n_boot=n_boot,
                                  seed=seed, alpha=alpha, tier=tier, arm=arm)
    elif wp is None:
        band_reason = ("no 'waypoints' in the dump — the band ADE needs the "
                       "integrated plan, not the controls")
    else:
        band_reason = ("no usable 'gt' in the dump — the per-band ADE block is "
                       "UNAVAILABLE. This is a WORK ITEM (bank the plan target "
                       "beside the emission), not a pass.")

    return {
        "block": "taniteval.seam.probe", "version": _seam.VERSION,
        "arm": arm, "tier": tier,
        "tier_note": "the continuity blocks are tier-INVARIANT (emitted plan "
                     "only); the band block inherits this declared tier",
        "source": source,
        "candidate_mode": mode, "candidate_arg": candidate,
        "n_rows": int(n_rows), "plan_steps": int(t_steps), "dt": dt,
        "seam_boundary": int(seam_i),
        "seam_boundary_derived": int(derived),
        "seam_boundary_overridden": bool(seam is not None
                                         and seam_i != derived),
        "seam_time_s": round(seam_i * dt, 4),
        "materiality_k": float(materiality_k),
        "local_halfwidth": int(local_halfwidth),
        "n_boot": int(n_boot), "seed": int(seed), "alpha": float(alpha),
        "estimator": "paired_episode_cluster_bootstrap + "
                     "episode_cluster_bootstrap (taniteval/ci.py)",
        "estimator_refusal": "overlapping_holdout_se is NEVER used here: it "
                             "narrows 1.107-3.100x AND biases the point "
                             "estimate bidirectionally, up to a SIGN FLIP on "
                             "paired deltas — and the seam contrast IS a "
                             "paired delta",
        "continuity": panel,
        "boundary_scan": scan_block,
        "bands": bands,
        "bands_unavailable_reason": band_reason,
        "headline": panel["headline"],
        "mandate": "X2 — VERIFY, NEVER REPAIR. A seam finding is REPORTED; no "
                   "loss term may be added in response to it.",
    }


# --------------------------------------------------------------------------- #
# the self-test — synthetic arms with a KNOWN answer                           #
# --------------------------------------------------------------------------- #
def _ou(rng, n, t, rho, sigma, scale):
    """An OU (temporally correlated) control series [n, t].

    ⚠️ DECLARED SYNTHETIC. A trained planner emits temporally COHERENT controls
    — that is precisely what ``DiffusionProposalGenerator``'s correlated noise
    exists to produce — and the instrument's power depends on that coherence:
    against a white-noise control sequence the within-band null is already as
    large as any stitch could make it, and NO instrument can see a control-space
    seam. That is a real property, reported by the power block, not hidden.
    """
    e = rng.normal(0.0, sigma, size=(n, t))
    x = np.empty((n, t))
    x[:, 0] = e[:, 0]
    for j in range(1, t):
        x[:, j] = rho * x[:, j - 1] + e[:, j]
    return scale * x


def build_selftest_arms(n_episodes=40, per_episode=22, t=_seam.PLAN_STEPS,
                        seam=_seam.SEAM_BOUNDARY, seed=0, dt=_seam.DT):
    """Three arms with a KNOWN ground truth about their seams."""
    import torch                                          # noqa: PLC0415
    from train_v58f_unicycle_head import unicycle_rollout  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n = n_episodes * per_episode
    eid = [f"ep{e:03d}" for e in range(n_episodes) for _ in range(per_episode)]
    v0 = np.clip(rng.normal(12.0, 4.0, size=n), 0.5, None)

    a1 = _ou(rng, n, t, 0.92, 0.30, 1.0)          # m/s^2
    k1 = _ou(rng, n, t, 0.95, 0.02, 1.0)          # 1/m
    a2 = _ou(rng, n, t, 0.92, 0.30, 1.0)          # an INDEPENDENT draw
    k2 = _ou(rng, n, t, 0.95, 0.02, 1.0)

    def roll(a, k, v):
        wp, _ = unicycle_rollout(torch.tensor(a)[:, None, :],
                                 torch.tensor(k)[:, None, :],
                                 torch.tensor(v), dt=dt)
        return wp[:, 0].numpy()

    # ARM 1 — genuine: ONE control sequence, ONE rollout.
    ctl_g = np.stack([a1, k1], axis=-1)
    wp_g = roll(a1, k1, v0)

    # ARM 2 — stitched CONTROLS: the tactical band comes from an independent
    # sequence; ONE rollout integrates the concatenation, so POSITION stays
    # continuous and only the controls jump.
    a_s = np.concatenate([a1[:, :seam], a2[:, seam:]], axis=1)
    k_s = np.concatenate([k1[:, :seam], k2[:, seam:]], axis=1)
    ctl_s = np.stack([a_s, k_s], axis=-1)
    wp_s = roll(a_s, k_s, v0)

    # ARM 3 — stitched ROLLOUT: two INDEPENDENT rollouts, the second re-based at
    # its own origin and concatenated. The controls are the genuine ones, so
    # only POSITION jumps — the defect the control channels cannot see.
    wp_b = roll(a2, k2, v0)
    wp_r = np.concatenate([wp_g[:, :seam], wp_b[:, :t - seam]], axis=1)

    # a plausible "true future": the genuine plan perturbed by an independent
    # smooth control error, rolled by the SAME integrator. Present only so the
    # per-band ADE block and the ``err`` channel are exercised by the self-test
    # — a block that is never run is a block that is never validated.
    gt = roll(a1 + _ou(rng, n, t, 0.9, 0.25, 1.0),
              k1 + _ou(rng, n, t, 0.9, 0.015, 1.0), v0)

    common = {"eid": eid, "tier": "T1", "plan_steps": t, "dt": dt, "gt": gt,
              "op_band_s": list(_seam.OP_BAND_S),
              "tac_band_s": list(_seam.TAC_BAND_S)}
    return {
        "genuine": {"controls": ctl_g, "waypoints": wp_g,
                    "arm": "selftest_genuine", **common},
        "stitch_controls": {"controls": ctl_s, "waypoints": wp_s,
                            "arm": "selftest_stitch_controls", **common},
        "stitch_rollout": {"controls": ctl_g, "waypoints": wp_r,
                           "arm": "selftest_stitch_rollout", **common},
    }


def _fired_on(rec, prefixes) -> bool:
    """Did any CONFIRMED seam row (both nulls) belong to one of ``prefixes``?"""
    rows = rec["continuity"]["seam_rows_confirmed_both_nulls"]
    return any(r.split("/")[0] in prefixes for r in rows)


def self_test(*, n_boot=400, seed=0, n_episodes=40, per_episode=22,
              materiality_k=_seam.MATERIALITY_K, verbose=True) -> dict:
    """Run the three synthetic arms and ASSERT the instrument's behaviour."""
    arms = build_selftest_arms(n_episodes=n_episodes, per_episode=per_episode,
                               seed=seed)
    recs, checks = {}, []
    for name, dump in arms.items():
        rec = run_probe(dump, arm=dump["arm"], tier="T1", candidate="winner",
                        seam=None, orders=_seam.ORDERS,
                        local_halfwidth=_seam.LOCAL_HALFWIDTH,
                        materiality_k=materiality_k, n_boot=n_boot, seed=seed,
                        alpha=0.05, scan=False, scan_channel=None,
                        scan_order=1, scan_n_boot=_seam.SCAN_N_BOOT,
                        scan_stride=1, source="--self-test (synthetic)")
        recs[name] = rec
        if verbose:
            _p("")
            _p(_seam.seam_report(rec["continuity"]))

    checks.append({
        "name": "genuine_single_rollout_does_NOT_fire",
        "pass": not recs["genuine"]["continuity"][
            "seam_rows_confirmed_both_nulls"],
        "detail": recs["genuine"]["continuity"][
            "seam_rows_confirmed_both_nulls"]})
    checks.append({
        "name": "injected_control_seam_FIRES_on_a_or_kappa",
        "pass": _fired_on(recs["stitch_controls"], {"a", "kappa"}),
        "detail": recs["stitch_controls"]["continuity"][
            "seam_rows_confirmed_both_nulls"]})
    checks.append({
        "name": "injected_rollout_seam_FIRES_on_wp",
        "pass": _fired_on(recs["stitch_rollout"], {"wp_x", "wp_y"}),
        "detail": recs["stitch_rollout"]["continuity"][
            "seam_rows_confirmed_both_nulls"]})
    checks.append({
        "name": "genuine_arm_is_WELL_POWERED_not_merely_quiet",
        "pass": recs["genuine"]["continuity"]["headline"]
        == "NO_MATERIAL_SEAM",
        "detail": recs["genuine"]["continuity"]["headline"]})

    ok = all(c["pass"] for c in checks)
    out = {"block": "taniteval.seam.selftest", "version": _seam.VERSION,
           "validated": bool(ok), "checks": checks,
           "n_windows": recs["genuine"]["n_rows"],
           "n_episodes": int(n_episodes), "n_boot": int(n_boot),
           "arms": {k: {"headline": v["continuity"]["headline"],
                        "confirmed": v["continuity"][
                            "seam_rows_confirmed_both_nulls"],
                        "global_only": v["continuity"][
                            "seam_rows_global_null_only"]}
                    for k, v in recs.items()},
           "records": recs,
           "_read": "an instrument never shown to detect the defect it hunts "
                    "is not validated; these four checks are that "
                    "demonstration, on DECLARED SYNTHETIC data"}
    if verbose:
        _p("")
        for c in checks:
            _p(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}  "
               f"-> {c['detail']}")
        _p(f"\nself-test: {'VALIDATED' if ok else '⛔ FAILED'}")
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(
        prog="seam_probe.py",
        description="X2 band-seam instrument — VERIFY, NEVER REPAIR (F-16)")
    ap.add_argument("--dump", default=None,
                    help="emission dump (.pt or .npz) — see the module "
                         "docstring for the schema")
    ap.add_argument("--dump-b", default=None,
                    help="a SECOND arm on the SAME windows -> per-band PAIRED "
                         "deltas (never a quadrature combination)")
    ap.add_argument("--arm", default=None,
                    help="label for the record (else the dump's 'arm')")
    ap.add_argument("--arm-b", default=None)
    ap.add_argument("--tier", default=None, choices=list(TIERS),
                    help="T0 (teacher-forced, WM diagnostic) | T1 (action-"
                         "closed-loop, the primary tier). REQUIRED unless the "
                         "dump declares it — an un-tiered number is what the "
                         "doctrine forbids.")
    ap.add_argument("--out", default=None, help="JSON output FILE (not a dir)")
    ap.add_argument("--candidate", default="winner",
                    help="winner (default, needs 'sel') | all | <int>")
    ap.add_argument("--seam", type=int, default=None,
                    help="override the geometry-derived boundary index "
                         "(default 20 = the 2.0 s band edge). An override does "
                         "NOT answer the X2 question and is recorded as such.")
    ap.add_argument("--orders", default="1,2,3",
                    help="finite-difference orders (1 level, 2 slope, "
                         "3 curvature)")
    ap.add_argument("--local-halfwidth", type=int,
                    default=_seam.LOCAL_HALFWIDTH,
                    help="the LOCAL null's half-width in boundaries — it is "
                         "what separates an index TREND from a seam")
    ap.add_argument("--materiality-k", type=float, default=_seam.MATERIALITY_K,
                    help="excess > k * within-band scale == material "
                         "(k=1 means 'the boundary step is more than twice a "
                         "typical within-band step')")
    ap.add_argument("--n-boot", type=int, default=None,
                    help=f"bootstrap draws. Unset -> {_ci.DEFAULT_N_BOOT} for a "
                         f"dump (the programme default) and {SELFTEST_N_BOOT} "
                         f"for --self-test (which runs three arms). An "
                         f"explicit value is ALWAYS honoured — a flag that is "
                         f"silently overridden is how a run reports a "
                         f"resolution it did not use.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-scan", action="store_true",
                    help="skip the boundary scan. ⚠️ the scan is the "
                         "instrument's own false-positive calibration; without "
                         "it a SEAM verdict has no reference rate.")
    ap.add_argument("--scan-channel", default=None)
    ap.add_argument("--scan-order", type=int, default=1)
    ap.add_argument("--scan-n-boot", type=int, default=_seam.SCAN_N_BOOT)
    ap.add_argument("--scan-stride", type=int, default=1)
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic seam-injection validation "
                         "(no GPU, no data, no checkpoint) and exit non-zero "
                         "if the instrument fails to detect an injected seam")
    ap.add_argument("--self-test-episodes", type=int, default=40)
    ap.add_argument("--self-test-per-episode", type=int, default=22)
    ap.add_argument("--self-test-dump-dir", default=None,
                    help="also write the three synthetic arms as .npz dumps "
                         "(documents the schema by example)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    orders = tuple(int(o) for o in str(args.orders).split(",") if o.strip())
    n_boot = args.n_boot if args.n_boot is not None else (
        SELFTEST_N_BOOT if args.self_test else _ci.DEFAULT_N_BOOT)

    if args.self_test:
        if args.self_test_dump_dir:
            os.makedirs(args.self_test_dump_dir, exist_ok=True)
            for name, d in build_selftest_arms(
                    n_episodes=args.self_test_episodes,
                    per_episode=args.self_test_per_episode).items():
                np.savez(os.path.join(args.self_test_dump_dir, f"{name}.npz"),
                         **{k: np.asarray(v) for k, v in d.items()})
        rec = self_test(n_boot=n_boot, seed=args.seed,
                        n_episodes=args.self_test_episodes,
                        per_episode=args.self_test_per_episode,
                        materiality_k=args.materiality_k,
                        verbose=not args.quiet)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, default=str)
            if not args.quiet:
                _p(f"wrote {args.out}")
        return 0 if rec["validated"] else 1

    if not args.dump:
        raise SystemExit("--dump is required (or --self-test). "
                         "See the module docstring for the dump schema.")
    dump = load_dump(args.dump)
    tier = args.tier or _scalar(dump.get("tier"))
    if tier not in TIERS:
        raise SystemExit(
            f"REFUSING: no admissible tier. Got {tier!r}; expected one of "
            f"{TIERS}. T0 = teacher-forced (a WM diagnostic, ⛔ never 'driving "
            f"performance'); T1 = action-closed-loop, the primary tier for any "
            f"capability claim. Pass --tier or stamp the dump. An un-tiered "
            f"number is exactly what EVAL_DOCTRINE.md forbids.")
    arm = args.arm or _scalar(dump.get("arm")) or os.path.basename(args.dump)

    rec = run_probe(dump, arm=str(arm), tier=str(tier),
                    candidate=args.candidate, seam=args.seam, orders=orders,
                    local_halfwidth=args.local_halfwidth,
                    materiality_k=args.materiality_k, n_boot=n_boot,
                    seed=args.seed, alpha=args.alpha, scan=not args.no_scan,
                    scan_channel=args.scan_channel, scan_order=args.scan_order,
                    scan_n_boot=args.scan_n_boot, scan_stride=args.scan_stride,
                    source=args.dump)

    if args.dump_b:
        db = load_dump(args.dump_b)
        if "waypoints" not in dump or "waypoints" not in db:
            raise SystemExit(
                "a paired per-band delta needs 'waypoints' in BOTH dumps — the "
                "band ADE is measured on the integrated plan, not on the "
                "controls. Refusing rather than comparing different objects.")
        wp_a = pick_candidate(dump["waypoints"], dump.get("sel"),
                              args.candidate, "waypoints")[0]
        wp_b = pick_candidate(db["waypoints"], db.get("sel"),
                              args.candidate, "waypoints")[0]
        gt = dump.get("gt")
        if gt is None:
            rec["bands_paired_unavailable_reason"] = (
                "arm A carries no 'gt' — a paired per-band delta needs the "
                "shared target. WORK ITEM, not a pass.")
        else:
            rec["bands_paired"] = _seam.band_errors_paired(
                wp_a, wp_b, _as_np(gt).astype(np.float64),
                expand_eid(dump["eid"], 1), seam=rec["seam_boundary"],
                n_boot=n_boot, seed=args.seed, alpha=args.alpha,
                arm_a=str(arm),
                arm_b=str(args.arm_b or _scalar(db.get("arm")) or args.dump_b))

    if not args.quiet:
        _p(_seam.seam_report(rec["continuity"], bands=rec["bands"],
                             scan=rec["boundary_scan"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=str)
        if not args.quiet:
            _p(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
