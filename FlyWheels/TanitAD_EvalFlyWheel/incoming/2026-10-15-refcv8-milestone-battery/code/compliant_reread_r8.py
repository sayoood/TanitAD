"""compliant_reread_r8.py -- AMENDMENT A-R8-CLAMP-2: the refcv8 bars re-read on CEILING-COMPLIANT windows.  CPU only, NEW file (nothing existing is edited).

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 compliant_reread_r8.py read   --tree <tree> [--force]   # one banked tree, BOTH inference seeds -> raw/compliant_reread/trees/<tree>.json
      python -X utf8 compliant_reread_r8.py strict --tree step30000ema_unc   # strict-cap (brake 0) sensitivity for E1/E3 on the compliant set
      python -X utf8 compliant_reread_r8.py assemble                         # COMPLIANT_REREAD.json / .md from every trees/*.json written so far

THE RULE (Project Steering/AMENDMENT_A_R8_CLAMP_2_COMPLIANT_WINDOWS.md, commit b6f7fac; parent AMENDMENT_A_R8_CLAMP_1_PLAN_VMAX_CLAMP.md):
  section 2.1  a window is COMPLIANT iff the logged ego's maximum realised speed over the window's valid 6-s future (`gt_max_speed_6s`) is <= the
               model's FED CEILING (MaxSpeedOneHotEncoder -> limit_ms_of_bin, an all-zero row -> +inf; refcv8's label rule).  STRICT, no tolerance.
               One window set for every surface and family; it depends on the logged data and the fed label only, never on a model.
  section 2.2  BAR-R8-E1 / E3 / NR on the compliant set, capped refcv8 primary, all-window values printed beside.  Same estimator: paired
               episode-cluster bootstrap B 2000 over the episodes the set contains.
  section 2.3  COMPLIANCE row per arm, on ALL windows and on the compliant set: rate of windows whose emitted plan exceeds the fed ceiling on the
               waypoint grid (`planned_max_speed`), the residual excess in m/s (median, p90, max), the rate of rows whose v0 is already over it.
  section 2.4  braking allowance stays as landed (2.5 m/s^2 = sel.accel_max); the strict cap (brake 0) is reported beside as a sensitivity.
  section 2.5  banked milestones may be re-read under the rule as EARLY READS; they never replace a banked verdict.
  section 3    CONTROLS, run BEFORE any compliant number is written:
      C1  all-True mask reproduces every banked cell exactly (delta, lo, hi, n_windows, n_episodes, separated, n_dropped_nonfinite):
          (a) uncapped refcv8 vs the banked PAIRED_READ, (b) capped refcv8 vs CLAMP_POSTHOC `read_ON`.  If C1 fails, NO compliant number is written.
      C2  the mask rolled by 35 windows (one episode block) changes the window set AND >= 1 headline cell.
      C3  the mask is computed once per tree/seed from the logged data + the fed label, identical for every arm; its sha is asserted equal across
          the two inference seeds (different model draws) and re-derived from a SEPARATE artifact (the panels' decisions/gt_future_ext).

REUSE, no re-implementation: `clamp_posthoc_r8` (fed_ceiling, load_seed_inputs, apply_clamp, cluster_rate, q3, the vendored landed
`cap_plan_to_limit` / `planned_max_speed` of commit de4e15b) and `pair_read_r8` (components, paired_cell, bar_E, bar_NR_metric, rollup_NR, METRICS,
load_panel_keyed) through `nr_matched_instrument` (panel_layout, rows_of, load_extras, compare_to_banked).  The ONLY new statistics are the window
mask and the in-memory panel substitution (the same substitution `nr_matched_instrument.write_variant` writes to disk, kept in memory).

D-R8-E1-REF (PI ruling 2026-10-19 ~05:40 Berlin, commit 7525bd7): *"Switch to speed, acceleration and yaw rate."*  From step 40,000 on, BAR-R8-E1 (S2) and
BAR-R8-E3 (S6) score refcv8 against `ha0_ext_pose` = refcv8's OWN causal prior (kinematic_prior.py: measured v0, backward-difference acceleration,
kappa0 = omega0 / max(v0, floor) from the backward yaw difference at t0), NOT against the echo `ha0_ext` whose kappa0 is read from the STEER channel
(curvature: inadmissible).  The pose arm is built by `a0k0_headroom_r8.load_decisions` + `references` (REUSED, no second derivation) from the panel's
decisions/ep_poses.  Output schema (per bars block `bars.<all_window|compliant>.<capped|uncapped>`):
    BAR-R8-E1 / BAR-R8-E3                 scored against ha0_ext_pose, field "reference": "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)"
    BAR-R8-E1_ha0_ext / BAR-R8-E3_ha0_ext the registered read (os - ha0_ext), same mechanics, labelled as the PRE-RULING reference -- kept BESIDE, never dropped
and per-cell `rows.ade_m.os_minus_ha0_ext_pose` next to `os_minus_ha0_ext`; the COMPLIANCE row gains the arm `echo_ha0_ext_pose`.  Control C4 gates the pose arm:
construction (decisions check + the echo rebuilt from the same decisions), model-independence across inference seeds, the 2026-09-26 model-free known value
(S2, all windows, ha0_ext_pose - ha0_ext) on EVERY tree, and the raw/a0k0_headroom/A0K0_HEADROOM.json known value on the 30k trees.  The switch FOLLOWED the 30k
reads (it was motivated after seeing them); the ha0_ext read stays beside so the series is auditable.  `read --mutate-pose-to-ha0-ext` is the deliberate-regression arm
(C4 must go RED; writes under raw/compliant_reread/mutation/, never trees/).  CLI of `read --tree T [--force]` is unchanged (chain_40k_r8.sh calls it).

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR / QUESTION: paired episode-cluster bootstrap over the
episodes the set contains (B 2000, seed 0, 95 % percentile), models FIXED -- "another draw of EPISODES" only; NOT another training run, NOT another
inference run.  Single training seed.  These are EARLY READS of banked milestones, never a bar verdict (the bars are decided at step 48,852).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clamp_posthoc_r8 as CP  # noqa: E402

PKG = HERE.parent
OUT = PKG / "raw" / "compliant_reread"
ARCHIVE = CP.ARCHIVE
PRIMARY = CP.PRIMARY                      # step30000ema_unc
SEEDS = CP.SEEDS
N_BOOT = CP.N_BOOT
ROLL = CP.MIS_ROLL                        # 35 windows ~ one episode block
TIER = CP.TIER
QUESTION = CP.QUESTION
SURFS = (("S2", "panel_s{s}"), ("S6", "s6_s{s}"))
DEFAULT_TREES = ["step5000_unc", "step5000ema_unc", "step10500_unc", "step10500ema_unc", "step15000_unc", "step15000ema_unc",
                 "step20000_unc", "step20000ema_unc", "step30000ema_unc"]
ARMS = ("refcv8_capped", "refcv8_uncapped", "refcv7_50400", "echo_ha0_ext", "echo_ha0_ext_pose")
VARIANTS = ("capped", "uncapped")
HEADLINE_NOTE = "headline = BAR-R8-E1 (S2 ade_m os-ha0_ext), BAR-R8-E3 (S6 ade_m os-ha0_ext) and the BAR-R8-NR cells of pair_read_r8.METRICS"
jload, jdump = CP.jload, CP.jdump
E1REF_ID = "D-R8-E1-REF"
E1REF_LABEL = "ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)"
E1REF_PRE_LABEL = "ha0_ext (PRE-RULING reference, E1 as registered; steer-channel kappa0; kept beside by D-R8-E1-REF)"
E1REF_RULING = {"id": E1REF_ID, "date": "2026-10-19", "commit": "7525bd7", "pi_words": "Switch to speed, acceleration and yaw rate.",
                "reference": "ha0_ext_pose = kinematic_prior.prior_controls('ha0_ext_pose'): measured v0, backward-difference a0, kappa0 = omega0 / max(v0, floor)",
                "switch_followed_the_30k_reads": True,
                "disclosure": "the reference switch was ruled AFTER the step-30,000 reads were seen; every tree before 40,000 is re-read under it as an EARLY READ, and the ha0_ext read stays beside"}
A0K0_JSON = PKG / "raw" / "a0k0_headroom" / "A0K0_HEADROOM.json"
A0K0_TREES = ("step30000ema_unc", "step30000_unc")             # the trees the banked A0K0_HEADROOM.json covers
REF0926 = "C:/lgs/Lfb47a58376/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior/raw/measure_prior_vs_echo.json"
KNOWN_FIELDS = ("delta", "lo", "hi", "n_windows", "n_episodes", "separated")


# ------------------------------------------------------------------------------------------ the mask (section 2.1)
def compliant_mask(gt_max: np.ndarray | None, lim) -> np.ndarray:
    """True iff gt_max_speed_6s <= the fed ceiling, STRICT (an unknown ceiling is +inf, so always compliant).  Refuses non-finite human speeds."""
    if gt_max is None:
        raise SystemExit("[compliant_reread] the dump's extras carry no gt_max_speed_6s: REFUSING (the mask needs the logged human speed)")
    g = np.asarray(gt_max, dtype=np.float64)
    if not np.isfinite(g).all():
        raise SystemExit(f"[compliant_reread] {int((~np.isfinite(g)).sum())} non-finite gt_max_speed_6s: REFUSING (a window without a valid future has no verdict)")
    l = np.asarray(lim.numpy() if hasattr(lim, "numpy") else lim, dtype=np.float64)
    return ~(g > l)


def mask_sha(clip: np.ndarray, ws: np.ndarray, comp: np.ndarray) -> str:
    """sha256 over the (clip_index, ws, compliant) triples sorted by (clip_index, ws): order-free, so two dumps with the same windows agree."""
    o = np.lexsort((np.asarray(ws, np.int64), np.asarray(clip, np.int64)))
    rec = np.stack([np.asarray(clip, np.int64)[o], np.asarray(ws, np.int64)[o], np.asarray(comp, np.int64)[o]], 1)
    return hashlib.sha256(np.ascontiguousarray(rec).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------ the masked read (the battery's own functions)
def read_masked(PR, comps: dict, eid: np.ndarray, mask: np.ndarray, metrics=None) -> dict:
    """pair_read_r8.read_surface on the windows of `mask` only: per-window components are computed ONCE on the full panel (as the banked read does)
    and then subset, so an all-True mask is the banked computation byte for byte.  comps = {'os','ha0_ext','ref'} each a components() dict."""
    m = np.asarray(mask, bool)
    e = np.asarray(eid)[m]
    out = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e))), "means": {}, "rows": {}}
    out["means"]["os"] = {k: (float(np.nanmean(v[m])) if np.isfinite(v[m]).any() else None) for k, v in comps["os"].items()}
    for fam, key, label, unit, lower, _nr in (metrics or PR.METRICS):
        row = {"family": fam, "label": label, "unit": unit, "lower_is_better": lower}
        row["os_minus_ha0_ext"] = PR.paired_cell(comps["os"][key][m], comps["ha0_ext"][key][m], e, N_BOOT)
        row["os_minus_refcv7_50400"] = PR.paired_cell(comps["os"][key][m], comps["ref"][key][m], e, N_BOOT)
        if key == "ade_m" and "ha0_ext_pose" in comps:        # D-R8-E1-REF: the bar reference (E1/E3 read ADE only)
            row["os_minus_ha0_ext_pose"] = PR.paired_cell(comps["os"][key][m], comps["ha0_ext_pose"][key][m], e, N_BOOT)
        out["rows"][key] = row
    return out


def headline_metrics(PR, surf: str) -> list:
    """The cells a bar reads on this surface: E1/E3 (ade_m vs echo) + the NR headline metrics of the surface."""
    return [t for t in PR.METRICS if surf in t[5] or t[1] == "ade_m"]


def headline_deltas(PR, read: dict, surf: str) -> dict:
    out = {}
    for fam, key, *_r, surfs in PR.METRICS:
        if surf in surfs:
            out[f"{key}|ref7"] = read["rows"][key]["os_minus_refcv7_50400"].get("delta")
    out["ade_m|echo"] = read["rows"]["ade_m"]["os_minus_ha0_ext"].get("delta")
    return out


def bars_of(PR, reads: dict) -> dict:
    """BAR-R8-E1 / E3 / NR exactly as pair_read_r8.build() states them, on the supplied reads[(seed, surf)]; the inference floor is |mean(os ADE, s0) -
    mean(os ADE, s1)| ON THE SAME SET the cells were read on.  D-R8-E1-REF: E1/E3 are scored against ha0_ext_pose (cell `os_minus_ha0_ext_pose`);
    the registered read against ha0_ext is kept as BAR-R8-E1_ha0_ext / BAR-R8-E3_ha0_ext (same `PR.bar_E`, same floor), labelled as the pre-ruling reference.
    A read with no pose cell yields NOT EVALUABLE for E1/E3 (never a silent fall-back to the echo)."""
    floors = {}
    for surf in ("S2", "S6"):
        a, b = reads[(0, surf)]["means"]["os"]["ade_m"], reads[(1, surf)]["means"]["os"]["ade_m"]
        floors[surf] = abs(a - b)
    cell = lambda surf, sd, key, which: reads[(sd, surf)]["rows"][key].get(which)  # noqa: E731

    def bar(surf, which, label):
        b = PR.bar_E({sd: cell(surf, sd, "ade_m", which) for sd in SEEDS}, floors[surf])
        b["reference"] = label
        b["reference_cell"] = which
        return b
    out = {"inference_floor_ADE_m": floors,
           "BAR-R8-E1": bar("S2", "os_minus_ha0_ext_pose", E1REF_LABEL),
           "BAR-R8-E3": bar("S6", "os_minus_ha0_ext_pose", E1REF_LABEL),
           "BAR-R8-E1_ha0_ext": bar("S2", "os_minus_ha0_ext", E1REF_PRE_LABEL),
           "BAR-R8-E3_ha0_ext": bar("S6", "os_minus_ha0_ext", E1REF_PRE_LABEL)}
    nr = {}
    for fam, key, label, unit, lower, surfs in PR.METRICS:
        for surf in surfs:
            nr[f"{surf}:{key}"] = {"family": fam, **PR.bar_NR_metric({sd: cell(surf, sd, key, "os_minus_refcv7_50400") for sd in SEEDS})}
    out["BAR-R8-NR"] = {"metrics": nr, "per_family": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()})}
    return out


# ------------------------------------------------------------------------------------------ the COMPLIANCE row (section 2.3)
def pmax(V, plan: np.ndarray, horizons, tick_s: float) -> np.ndarray:
    """The module's own planned_max_speed of ONE plan per row on the given waypoint grid -> float32 [n] (the dtype clamp_posthoc's r1_gap uses)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(plan, dtype=np.float32))
    return V.planned_max_speed(t[:, None], horizons=[int(h) for h in horizons], tick_s=tick_s)[:, 0].numpy()


def compliance_sets(pm: np.ndarray, lim: np.ndarray, v0: np.ndarray, eid: np.ndarray, sels: dict) -> dict:
    """One COMPLIANCE row per window set.  exceed = the plan's maximum waypoint-interval speed > the fed ceiling (STRICT, as SpeedCeilingFilter's
    default; +inf never exceeded); the >0.01 m/s count beside excludes fp32 dust of a plan capped AT the ceiling."""
    fin = np.isfinite(lim)
    exc = (pm - lim).astype(np.float32)
    ex_s, ex_c, v0o = fin & (exc > 0), fin & (exc > CP.EXEMPT_TOL_COARSE), fin & (v0 > lim)
    out = {}
    for name, sel in sels.items():
        s = np.ones(len(pm), bool) if sel is None else np.asarray(sel, bool)
        if not s.any():
            out[name] = {"n_windows": 0}
            continue
        e = eid[s]
        out[name] = {"n_windows": int(s.sum()), "n_episodes": int(len(np.unique(e))), "n_finite_ceiling": int(fin[s].sum()),
                     "exceed_rate_strict": CP.cluster_rate(ex_s[s].astype(float), e), "n_exceed_strict": int(ex_s[s].sum()),
                     "exceed_rate_gt_0p01_ms": CP.cluster_rate(ex_c[s].astype(float), e), "n_exceed_gt_0p01_ms": int(ex_c[s].sum()),
                     "residual_excess_ms_on_exceeding_windows": CP.q3(exc[s][ex_s[s]]),
                     "residual_excess_gt_0p01_ms_on_those": CP.q3(exc[s][ex_c[s]]),
                     "v0_over_ceiling_rate": CP.cluster_rate(v0o[s].astype(float), e), "n_v0_over": int(v0o[s].sum())}
    return out


# ------------------------------------------------------------------------------------------ per (tree, seed) context
def load_panel(NR, PR, sha12, ex: dict, pdir: str) -> dict:
    """The panel arms a bar reads, the window -> dump-row map, and the grid (slots / horizons / dt).  Everything else in the panel is dropped."""
    man, files, ep_of = NR.panel_layout(pdir)
    rows = np.concatenate([np.asarray(NR.rows_of(ex, ep_of, f, np.load(f)["ws"]), dtype=np.int64) for f in files])
    man2, A, eid, keys = PR.load_panel_keyed(pdir, sha12)
    if len(rows) != len(eid):
        raise SystemExit(f"[compliant_reread] {pdir}: row map {len(rows)} != panel windows {len(eid)}")
    if len(set(rows.tolist())) != len(rows):
        raise SystemExit(f"[compliant_reread] {pdir}: duplicate dump rows in the panel")
    return {"dir": pdir, "man": man, "slots": [int(x) for x in man["grid"]["slots"]], "dt": float(man["grid"]["dt_s"]), "rows": rows,
            "eid": np.asarray(eid), "A": {k: A[k] for k in A if k in ("g", "os", "ha0_ext") or k.startswith("b_refcv7_50400")}}


def decisions_gt_max(pdir: str, rows: np.ndarray, gt_extras: np.ndarray) -> dict:
    """C3 independent derivation: the human's max speed re-derived from the panel's OWN decisions/gt_future_ext (a separate artifact from the extras)."""
    import glob
    fs = sorted(glob.glob(os.path.join(pdir, "decisions", "ep*.npz")))
    if not fs:
        return {"status": "ABSENT (no decisions/ in the panel)"}
    mx = []
    for f in fs:
        z = np.load(f)
        if "gt_future_ext" not in z.files or "gt_future_valid_ext" not in z.files:
            return {"status": "ABSENT (decisions lack gt_future_ext / gt_future_valid_ext)"}
        sp = z["gt_future_ext"][:, :, 3].astype(np.float64)
        va = z["gt_future_valid_ext"].astype(bool)
        mx.append(np.where(va, sp, -np.inf).max(1))
    mx = np.concatenate(mx)
    if len(mx) != len(rows):
        return {"status": f"LENGTH MISMATCH {len(mx)} vs {len(rows)}"}
    d = np.abs(mx - gt_extras[rows].astype(np.float64))
    return {"status": "OK", "n_windows": int(len(mx)), "max_abs_diff_ms": float(d.max()), "n_diff_gt_1e-4": int((d > 1e-4).sum())}


def prepare_seed(tree: str, seed: int, tools: dict, rc: dict) -> dict:
    NR, PR, sha12, V = tools["NR"], tools["PR"], tools["sha12"], tools["V"]
    BP = tools["BP"]
    raw, lim, linfo, dump = CP.load_seed_inputs(tree, seed, rc)
    ex = NR.load_extras(os.path.dirname(dump))
    limf = lim.numpy().astype(np.float32)
    comp = compliant_mask(raw["gt_max_speed_6s"], lim)                # computed ONCE from logged data + the fed label, no model call
    clip = raw["clip_index"].astype(np.int64)
    ws = ex["ws"].astype(np.int64)
    if len(clip) != len(ws) or not np.array_equal(clip, ex["clip_index"].astype(np.int64)):
        raise SystemExit("[compliant_reread] raw / extras row order disagree")
    ctx = {"tree": tree, "seed": seed, "dump": dump, "raw": raw, "lim_t": lim, "lim": limf, "linfo": linfo, "ex": ex, "comp": comp,
           "clip": clip, "ws": ws, "sha": mask_sha(clip, ws, comp), "v0": raw["v0"].astype(np.float32), "plan": raw["nav_true.traj"],
           "refarm": BP.R7_BASE_LABEL.format(s=seed), "panels": {}}
    for surf, pat in SURFS:
        ctx["panels"][surf] = load_panel(NR, PR, sha12, ex, f"{ARCHIVE}/{tree}/{pat.format(s=seed)}")
    return ctx


def build_pose_reference(tools: dict, ctx: dict, surf: str) -> tuple:
    """The ha0_ext_pose arm of ONE (tree, seed, surface), built EXACTLY as a0k0_headroom_r8.surface_inputs builds it (that module's load_decisions +
    references are REUSED -- there is no second derivation here).  Returns (HP float32 [n, S, 2] in panel row order, meta)."""
    import a0k0_headroom_r8 as AK
    P = ctx["panels"][surf]
    rows = P["rows"]
    ws, v0 = ctx["ws"][rows], ctx["v0"][rows]
    hz = [int(h) for h in P["man"]["grid"]["horizons_steps"]]
    dec = AK.load_decisions(tools["NR"], P["dir"], ws, v0)
    REF, V = AK.references(dec, P["A"]["ha0_ext"], hz)
    ok = bool(dec["chk"]["PASS"] and V["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"] <= AK.V1_TOL_M)
    return REF["HP"], {"decisions_check": dec["chk"], "V1_ha0_ext_rebuilt_vs_panel_max_abs_m": V["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"],
                       "V1_frac_windows_bit_equal": V["V1_frac_windows_bit_equal"], "V1_tol_m": AK.V1_TOL_M,
                       "ha0_ext_pose_vs_ha0_ext_max_abs_m": V["ha0_ext_pose_vs_ha0_ext_max_abs_m"],
                       "builder": "a0k0_headroom_r8.load_decisions + references (kinematic_prior.prior_controls('ha0_ext_pose') -> prior_path)", "PASS": ok}


def build_pose_reference_MUTATED_TO_HA0_EXT(tools: dict, ctx: dict, surf: str) -> tuple:
    """DELIBERATE REGRESSION (control c): the pose prior swapped back to the echo.  C4 must read RED against the banked A0K0 / 2026-09-26 values."""
    P = ctx["panels"][surf]
    return np.array(P["A"]["ha0_ext"], dtype=np.float32, copy=True), {"builder": "MUTATION: ha0_ext in place of ha0_ext_pose", "PASS": True,
                                                                       "decisions_check": {"PASS": True}, "MUTATED": True}


POSE_BUILDER = build_pose_reference            # tests / the mutation arm replace this; production never does


def attach_pose_reference(tools: dict, ctx: dict, builder=None) -> dict:
    """Adds A['ha0_ext_pose'] to every panel of the context (model-free: identical for both inference seeds) and returns the per-surface meta."""
    b = builder or POSE_BUILDER
    meta = {}
    for surf, _pat in SURFS:
        hp, m = b(tools, ctx, surf)
        P = ctx["panels"][surf]
        if hp.shape != P["A"]["ha0_ext"].shape:
            raise SystemExit(f"[compliant_reread] {ctx['tree']} s{ctx['seed']} {surf}: ha0_ext_pose {hp.shape} != ha0_ext {P['A']['ha0_ext'].shape}")
        P["A"]["ha0_ext_pose"] = hp
        meta[surf] = m
    return meta


def known_value_check(cell: dict | None, lit: dict | None) -> dict:
    """A cell against a banked literal on KNOWN_FIELDS, exactly.  `exact` is False when either side is absent (a control that could not run never passes)."""
    if not isinstance(cell, dict) or not isinstance(lit, dict) or "delta" not in cell or "delta" not in lit:
        return {"exact": False, "status": "ABSENT", "got": cell and {k: cell.get(k) for k in KNOWN_FIELDS}, "banked": lit and {k: lit.get(k) for k in KNOWN_FIELDS}}
    diff = {k: {"got": cell.get(k), "banked": lit.get(k)} for k in KNOWN_FIELDS if k in lit and cell.get(k) != lit.get(k)}
    return {"exact": not diff, "diff": diff, "got": {k: cell.get(k) for k in KNOWN_FIELDS}, "banked": {k: lit.get(k) for k in KNOWN_FIELDS if k in lit}}


def load_known_values() -> dict:
    """The two banked artifacts the pose arm is checked against: A0K0_HEADROOM.json (30k trees) and the 2026-09-26 measure_prior_vs_echo.json (model-free)."""
    out = {"a0k0": None, "r0926": None, "errors": []}
    try:
        out["a0k0"] = jload(str(A0K0_JSON))
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"A0K0_HEADROOM.json unreadable: {e!r}")
    try:
        out["r0926"] = jload(REF0926)["B_ade_0_2s"]["P_ha0_ext_pose_minus_echo"]
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"measure_prior_vs_echo.json unreadable: {e!r}")
    return out


def a0k0_literal(known: dict, tree: str, seed: int, surf: str, set_name: str) -> dict | None:
    try:
        return known["a0k0"]["trees"][tree]["seeds"][str(seed)]["surfaces"][surf]["sets"][set_name]["refs"]["os_minus_ha0_ext_pose"]["ade_m"]
    except (KeyError, TypeError):
        return None


def plan_variants(V, ctx: dict, rc: dict, brake=None) -> dict:
    """The capped plan [N, 8, 2] (the landed function, as CLAMP_POSTHOC applies it) next to the emitted (uncapped) one."""
    out_c, diag = CP.apply_clamp(V, ctx["plan"], ctx["lim_t"], ctx["v0"], rc, brake=brake)
    return {"capped": out_c.numpy().astype(np.float32), "uncapped": ctx["plan"].astype(np.float32), "diag": diag}


def surface_reads(tools: dict, ctx: dict, rc: dict, banked: dict, clamp_banked: dict | None, pv: dict, mask_dump: np.ndarray, roll_dump: np.ndarray,
                  known: dict | None = None) -> dict:
    """C1 / C2 / C4 / the compliant reads for BOTH variants on both surfaces of one (tree, seed).  `known` = load_known_values() (the pose arm's known-value controls)."""
    NR, PR, ra = tools["NR"], tools["PR"], tools["ra"]
    seed = ctx["seed"]
    res = {}
    for surf, _pat in SURFS:
        P = ctx["panels"][surf]
        A, dt, slots, rows, eid = P["A"], P["dt"], P["slots"], P["rows"], P["eid"]
        n = len(eid)
        os_slots = {v: pv[v][rows][:, slots] for v in VARIANTS}                # capped / uncapped plan on the panel's grid, panel row order
        s_rec = {"n_panel_windows": int(n), "grid_slots": slots,
                 "panel_os_vs_extras_nav_true_slots_max_abs": float(np.abs(A["os"].astype(np.float64) - os_slots["uncapped"].astype(np.float64)).max())}
        comps = {"uncapped": PR.components(ra, A["os"], A["g"], dt),            # the panel's OWN os, exactly the banked path
                 "capped": PR.components(ra, os_slots["capped"], A["g"], dt)}
        fixed = {"ha0_ext": PR.components(ra, A["ha0_ext"], A["g"], dt), "ref": PR.components(ra, A[ctx["refarm"]], A["g"], dt)}
        if "ha0_ext_pose" in A:                                                  # D-R8-E1-REF: the bar reference
            fixed["ha0_ext_pose"] = PR.components(ra, A["ha0_ext_pose"], A["g"], dt)
        mk = mask_dump[rows]
        mk_roll = roll_dump[rows]
        ones = np.ones(n, bool)
        # ---- C1: all-True mask through the SAME pipeline reproduces every banked cell exactly
        r_all = {v: read_masked(PR, {"os": comps[v], **fixed}, eid, ones) for v in VARIANTS}
        c1a = NR.compare_to_banked(r_all["uncapped"], banked["surfaces"][surf]["seeds"][str(seed)])
        bm = (banked["surfaces"][surf]["seeds"][str(seed)].get("means") or {}).get("os") or {}
        c1a["os_means_equal_banked"] = bool(bm) and all(bm.get(k) == r_all["uncapped"]["means"]["os"].get(k) for k in bm)
        c1a["n_windows_equal_banked"] = bool(banked["surfaces"][surf]["seeds"][str(seed)]["n_windows"] == r_all["uncapped"]["n_windows"])
        c1a["exact"] = bool(c1a["exact"] and c1a["os_means_equal_banked"] and c1a["n_windows_equal_banked"])
        if clamp_banked is not None:
            ro = clamp_banked["seeds"][str(seed)]["surfaces"][surf].get("read_ON")
            c1b = NR.compare_to_banked(r_all["capped"], ro) if ro else {"exact": False, "status": "UNAVAILABLE (no read_ON in CLAMP_POSTHOC for this tree/seed/surface)"}
            if ro:
                c1b["os_means_equal_banked"] = bool(ro.get("os_means") and all(ro["os_means"].get(k) == r_all["capped"]["means"]["os"].get(k) for k in ro["os_means"]))
                c1b["exact"] = bool(c1b["exact"] and c1b["os_means_equal_banked"])
        else:
            c1b = {"exact": False, "status": "UNAVAILABLE (no CLAMP_POSTHOC tree json for this tree)"}
        s_rec["C1a_uncapped_vs_banked_PAIRED_READ"] = c1a
        s_rec["C1b_capped_vs_CLAMP_POSTHOC_read_ON"] = c1b
        s_rec["C1_pass"] = bool(c1a["exact"] and c1b["exact"])
        s_rec["all_window"] = {v: r_all[v] for v in VARIANTS}
        # ---- C4 (D-R8-E1-REF): the pose arm against banked KNOWN VALUES (uncapped os, the cell the bar reads)
        c4 = {"applicable_a0k0": bool(ctx["tree"] in A0K0_TREES)}
        if "ha0_ext_pose" in A and known is not None:
            if surf == "S2":     # model-free, tree-free: ha0_ext_pose - ha0_ext on all 4,754 S2 windows must equal the 2026-09-26 measurement
                c4["R0926_S2_all_ha0_ext_pose_minus_ha0_ext"] = known_value_check(
                    PR.paired_cell(fixed["ha0_ext_pose"]["ade_m"], fixed["ha0_ext"]["ade_m"], eid, N_BOOT), known.get("r0926"))
            if c4["applicable_a0k0"]:
                c4["A0K0_all_window_uncapped"] = known_value_check(r_all["uncapped"]["rows"]["ade_m"]["os_minus_ha0_ext_pose"],
                                                                   a0k0_literal(known, ctx["tree"], seed, surf, "all"))
        s_rec["C4_pose_known_values"] = c4
        # ---- the mask on this surface
        s_rec["mask"] = {"n_windows": int(mk.sum()), "n_total": int(n), "n_noncompliant": int((~mk).sum()),
                         "n_episodes_in_set": int(len(np.unique(eid[mk]))), "n_episodes_total": int(len(np.unique(eid)))}
        if s_rec["C1_pass"]:
            r_comp = {v: read_masked(PR, {"os": comps[v], **fixed}, eid, mk) for v in VARIANTS}
            s_rec["compliant"] = {v: r_comp[v] for v in VARIANTS}
            if c4["applicable_a0k0"] and "ha0_ext_pose" in A and known is not None:
                c4["A0K0_compliant_uncapped"] = known_value_check(r_comp["uncapped"]["rows"]["ade_m"]["os_minus_ha0_ext_pose"],
                                                                  a0k0_literal(known, ctx["tree"], seed, surf, "compliant"))
            # ---- C2: the rolled mask changes the window set and >= 1 headline cell (capped, the primary arm)
            hm = headline_metrics(PR, surf)
            r_roll = read_masked(PR, {"os": comps["capped"], **fixed}, eid, mk_roll, metrics=hm)
            h_true, h_roll = headline_deltas(PR, r_comp["capped"], surf), headline_deltas(PR, r_roll, surf)
            changed = {k: bool(h_true[k] != h_roll[k]) for k in h_true}
            s_rec["C2_rolled_mask"] = {"roll_windows": ROLL, "n_rows_whose_membership_changed": int((mk_roll != mk).sum()),
                                       "set_changed": bool((mk_roll != mk).any()), "n_windows_rolled": int(mk_roll.sum()),
                                       "n_headline_cells": len(changed), "n_headline_cells_changed": int(sum(changed.values())), "changed": changed,
                                       "PASS": bool((mk_roll != mk).any() and sum(changed.values()) >= 1)}
            if "os_minus_ha0_ext_pose" in r_comp["capped"]["rows"]["ade_m"]:     # informational (not part of PASS): the pose cell moves with the mask too
                s_rec["C2_rolled_mask"]["pose_ade_cell_changed"] = bool(r_comp["capped"]["rows"]["ade_m"]["os_minus_ha0_ext_pose"].get("delta")
                                                                        != r_roll["rows"]["ade_m"]["os_minus_ha0_ext_pose"].get("delta"))
        else:
            s_rec["compliant"] = "WITHHELD: C1 did not pass for this (seed, surface) -- no compliant number is written"
        res[surf] = s_rec
        del comps, fixed
    return res


def compliance_all(tools: dict, ctx: dict, rc: dict, pv: dict, mask_dump: np.ndarray, withhold: bool) -> dict:
    """The COMPLIANCE row for the four arms.  Common grids so the arms are comparable: the S2 panel grid (0.5-s waypoints, 0-2 s) on the S2 windows
    and the S6 panel grid (1-s waypoints, 0-6 s) on the S6 windows -- refcv7-50,400 and the echo exist ONLY on those panel grids (their embedded
    arms).  The two refcv8 arms ALSO get the full emitted plan (8 slots, 0.5-s then 1-s waypoints, 0-6 s) from the dump's extras."""
    V = tools["V"]
    rc_h = rc["horizons"]
    tick = rc["tick_s"]
    out = {"note": "exceed = planned_max_speed(arm plan on the stated grid) > the FED CEILING OF refcv8 (one ceiling, one window set, for every arm; "
                   "refcv7-50,400 consumed a different, per-clip speed label -- its own ceiling is not the bar). Strict > 0; the >0.01 m/s count beside excludes fp32 dust.",
           "grids": {}}
    limf, v0f = ctx["lim"], ctx["v0"]
    for surf, _pat in SURFS:
        P = ctx["panels"][surf]
        rows, eid, slots, A = P["rows"], P["eid"], P["slots"], P["A"]
        hz = [rc_h[i] for i in slots]
        lim_r, v0_r = limf[rows], v0f[rows]
        mk = mask_dump[rows]
        arms = {"refcv8_capped": pv["capped"][rows][:, slots], "refcv8_uncapped": A["os"], "refcv7_50400": A[ctx["refarm"]], "echo_ha0_ext": A["ha0_ext"]}
        if "ha0_ext_pose" in A:                                                  # D-R8-E1-REF: the new echo arm
            arms["echo_ha0_ext_pose"] = A["ha0_ext_pose"]
        g = {"horizons_ticks": hz, "n_windows": int(len(rows)), "arms": {}}
        for arm, plan in arms.items():
            pm = pmax(V, plan, hz, tick)
            sets = compliance_sets(pm, lim_r, v0_r, eid, {"all_windows": None, **({} if withhold else {"compliant_set": mk})})
            if withhold:
                sets["compliant_set"] = "WITHHELD (C1)"
            g["arms"][arm] = sets
        out["grids"][f"{surf}_panel_grid"] = g
    # full emitted plan, refcv8 arms, S2 window set (dump order)
    eid_d = ctx["clip"]
    g = {"horizons_ticks": rc_h, "n_windows": int(len(eid_d)), "arms": {}}
    for arm, key in (("refcv8_capped", "capped"), ("refcv8_uncapped", "uncapped")):
        pm = pmax(V, pv[key], rc_h, tick)
        sets = compliance_sets(pm, limf, v0f, eid_d, {"all_windows": None, **({} if withhold else {"compliant_set": mask_dump})})
        if withhold:
            sets["compliant_set"] = "WITHHELD (C1)"
        g["arms"][arm] = sets
    out["grids"]["FULL_plan_0_6s_S2_windows"] = g
    return out


# ------------------------------------------------------------------------------------------ one tree
def tree_complete(tree: str) -> tuple[bool, str, dict | None]:
    """A tree is readable only when its PAIRED_READ json exists AND is complete (finished, both seeds, both surfaces with rows, bars)."""
    p = f"{ARCHIVE}/{tree}/PAIRED_READ_{tree}.json"
    if not os.path.exists(p):
        return False, f"{p} does not exist yet", None
    try:
        b = jload(p)
    except Exception as e:  # noqa: BLE001
        return False, f"{p} unreadable: {e!r}", None
    why = []
    if not b.get("finished"):
        why.append("no 'finished' stamp")
    for surf in ("S2", "S6"):
        for s in ("0", "1"):
            sd = b.get("surfaces", {}).get(surf, {}).get("seeds", {}).get(s, {})
            if "rows" not in sd:
                why.append(f"{surf} seed {s} has no rows")
    if not (b.get("bars") or {}).get("BAR-R8-E1"):
        why.append("no bars")
    for s in (0, 1):
        for sub in (f"dump_s{s}/refcv6_extras.npz", f"panel_s{s}/manifest.json", f"s6_s{s}/manifest.json"):
            if not os.path.exists(f"{ARCHIVE}/{tree}/{sub}"):
                why.append(f"{sub} absent")
    return (not why), ("; ".join(why) if why else "complete"), b


def tools_load() -> dict:
    V, vrec = CP.load_vendored()
    NR, BP, PR, ra, sha12 = CP.get_readers()
    return {"V": V, "vrec": vrec, "NR": NR, "BP": BP, "PR": PR, "ra": ra, "sha12": sha12}


def c4_pose_control(tree: str, res: dict, hp_by_seed: dict, known: dict) -> dict:
    """C4 (D-R8-E1-REF): the ha0_ext_pose arm is valid iff (i) its construction checks pass on every (seed, surface) [decisions check + the echo rebuilt from the
    SAME decisions within 1e-3 m], (ii) it is model-free -- bit-identical for the two inference seeds -- (iii) S2 all-window ha0_ext_pose - ha0_ext reproduces
    the 2026-09-26 measurement exactly (every tree), (iv) on the trees A0K0_HEADROOM.json covers the uncapped os - ha0_ext_pose cell reproduces it exactly
    (all-window and compliant, both seeds, both surfaces).  A known-value check that could not run (artifact unreadable) FAILS the control."""
    cons, ident, r0926, a0k0 = {}, {}, {}, {}
    for surf, _ in SURFS:
        for s_ in SEEDS:
            m = res["seeds"][str(s_)]["pose_reference"][surf]
            cons[f"{surf}|s{s_}"] = bool(m.get("PASS"))
        k0, hp0 = hp_by_seed[0][surf]
        k1, hp1 = hp_by_seed[1][surf]
        ident[surf] = bool(np.array_equal(k0, k1) and np.array_equal(hp0, hp1))
    for s_ in SEEDS:
        for surf, _ in SURFS:
            c4 = res["seeds"][str(s_)]["surfaces"][surf].get("C4_pose_known_values", {})
            if surf == "S2":
                r0926[f"s{s_}"] = c4.get("R0926_S2_all_ha0_ext_pose_minus_ha0_ext", {"exact": False, "status": "ABSENT"})
            for k in ("A0K0_all_window_uncapped", "A0K0_compliant_uncapped"):
                if k in c4:
                    a0k0[f"{k}|s{s_}|{surf}"] = c4[k]
    applicable = tree in A0K0_TREES
    n_expected_a0k0 = len(SEEDS) * len(SURFS) * 2
    a0k0_ok = (len(a0k0) == n_expected_a0k0 and all(v.get("exact") for v in a0k0.values())) if applicable else True
    r0926_ok = bool(len(r0926) == len(SEEDS) and all(v.get("exact") for v in r0926.values()))
    out = {"construction_per_seed_surface": cons, "model_free_identical_across_inference_seeds": ident,
           "R0926_known_value_S2_all_ha0_ext_pose_minus_ha0_ext": {"exact_per_seed": {k: v.get("exact") for k, v in r0926.items()}, "detail": r0926,
                                                                  "source": REF0926 + " :: B_ade_0_2s.P_ha0_ext_pose_minus_echo"},
           "A0K0_known_value": {"applicable": applicable, "source": str(A0K0_JSON), "n_cells_expected": n_expected_a0k0 if applicable else 0,
                                "n_cells_checked": len(a0k0), "all_exact": a0k0_ok if applicable else None,
                                "detail": a0k0 if applicable else "NOT APPLICABLE (the banked A0K0_HEADROOM.json covers only " + ", ".join(A0K0_TREES) + ")"},
           "known_value_artifact_errors": known.get("errors", [])}
    out["PASS"] = bool(all(cons.values()) and all(ident.values()) and r0926_ok and a0k0_ok and not known.get("errors"))
    return out


def read_tree(tree: str, force: bool, mutate_pose: bool = False) -> int:
    """mutate_pose=True is the DELIBERATE-REGRESSION arm of D-R8-E1-REF (control c): the pose prior is swapped back to ha0_ext, the output goes to
    raw/compliant_reread/mutation/<tree>.json (NEVER trees/), and C4 must read RED."""
    out_json = (OUT / "mutation" if mutate_pose else OUT / "trees") / f"{tree}.json"
    if out_json.exists() and not force:
        print(f"[{tree}] already read ({out_json}); --force to redo")
        return 0
    ok, why, banked = tree_complete(tree)
    if not ok:
        print(f"[{tree}] NOT READABLE: {why}")
        return 3
    t0 = time.time()
    rc = CP.run_cfg()
    if rc["horizons"] != [5, 10, 15, 20, 30, 40, 50, 60] or rc["argv_has_refcv7"]:
        raise SystemExit(f"[compliant_reread] unexpected run config: {rc}")
    tools = tools_load()
    PR, V = tools["PR"], tools["V"]
    known = load_known_values()
    pose_builder = build_pose_reference_MUTATED_TO_HA0_EXT if mutate_pose else POSE_BUILDER
    cb_path = CP.OUT / "trees" / f"{tree}.json"
    clamp_banked = jload(str(cb_path)) if cb_path.exists() else None
    res = {"tool": "compliant_reread_r8.py", "tree": tree, "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT, "generated": time.strftime("%FT%T"),
           "rule": "Project Steering/AMENDMENT_A_R8_CLAMP_2_COMPLIANT_WINDOWS.md (b6f7fac) over AMENDMENT_A_R8_CLAMP_1_PLAN_VMAX_CLAMP.md",
           "status": "EARLY READ of a banked milestone under the compliant-window rule; never a bar verdict (decided at step 48,852)",
           "ruling": {**E1REF_RULING, "MUTATION_ARM_NOT_A_RESULT": bool(mutate_pose)},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "vendored": {k: tools["vrec"][k] for k in ("source_commit", "git_rev_parse_source_blob", "vendored_file_git_blob_sha", "vendored_file_sha256")},
           "run_cfg": {k: rc[k] for k in ("horizons", "tick_s", "brake_ms2")},
           "inputs": {"paired_read": f"{ARCHIVE}/{tree}/PAIRED_READ_{tree}.json", "paired_read_finished": banked.get("finished"),
                      "clamp_posthoc": str(cb_path) if clamp_banked else None},
           "seeds": {}, "controls": {}}
    sha_by_seed, reads = {}, {"compliant": {v: {} for v in VARIANTS}, "all_window": {v: {} for v in VARIANTS}}
    c1_all, c2_all = [], []
    masks_for_c3 = {}
    hp_by_seed = {}
    for seed in SEEDS:
        ctx = prepare_seed(tree, seed, tools, rc)
        pose_meta = attach_pose_reference(tools, ctx, pose_builder)       # D-R8-E1-REF: the bar reference, model-free
        hp_by_seed[seed] = {surf: (np.stack([ctx["clip"][ctx["panels"][surf]["rows"]], ctx["ws"][ctx["panels"][surf]["rows"]]]).copy(),
                                   ctx["panels"][surf]["A"]["ha0_ext_pose"].copy()) for surf, _ in SURFS}
        pv = plan_variants(V, ctx, rc)
        mask_dump = ctx["comp"]
        roll_dump = np.roll(mask_dump, -ROLL)
        sha_by_seed[seed] = ctx["sha"]
        masks_for_c3[seed] = (ctx["clip"].copy(), ctx["ws"].copy(), mask_dump.copy())
        sd = {"extras": ctx["dump"], "n_windows": int(len(mask_dump)), "ceiling": ctx["linfo"], "mask_sha256": ctx["sha"],
              "n_compliant": int(mask_dump.sum()), "n_noncompliant": int((~mask_dump).sum()),
              "noncompliant_rate_all_windows": CP.cluster_rate((~mask_dump).astype(float), ctx["clip"]),
              "binding_rate_capped_all_windows": CP.cluster_rate(pv["diag"]["plan_cap_binding"].numpy().astype(float), ctx["clip"]),
              "n_binding_capped": int(pv["diag"]["plan_cap_binding"].sum()),
              "n_binding_on_compliant_windows": int((pv["diag"]["plan_cap_binding"].numpy() & mask_dump).sum()),
              "n_changed_plans_on_compliant_windows": int(((np.abs(pv["capped"].astype(np.float64) - pv["uncapped"].astype(np.float64)).reshape(len(mask_dump), -1).max(1) > 0) & mask_dump).sum())}
        sd["c3_independent_derivation"] = {surf: decisions_gt_max(ctx["panels"][surf]["dir"], ctx["panels"][surf]["rows"], ctx["raw"]["gt_max_speed_6s"]) for surf, _ in SURFS}
        sd["pose_reference"] = pose_meta
        sd["surfaces"] = surface_reads(tools, ctx, rc, banked, clamp_banked, pv, mask_dump, roll_dump, known=known)
        c1_seed = all(sd["surfaces"][s]["C1_pass"] for s, _ in SURFS)
        c1_all.append(c1_seed)
        c2_all += [sd["surfaces"][s].get("C2_rolled_mask", {}).get("PASS") is True for s, _ in SURFS]
        sd["compliance"] = compliance_all(tools, ctx, rc, pv, mask_dump, withhold=not c1_seed)
        for surf, _ in SURFS:
            for v in VARIANTS:
                reads["all_window"][v][(seed, surf)] = sd["surfaces"][surf]["all_window"][v]
                if c1_seed:
                    reads["compliant"][v][(seed, surf)] = sd["surfaces"][surf]["compliant"][v]
        res["seeds"][str(seed)] = sd
        print(f"[{tree}] seed {seed} done {time.time() - t0:.0f}s  mask n={sd['n_compliant']}/{sd['n_windows']} C1={c1_seed}", flush=True)
        del ctx, pv
    # ---- C3: one mask per tree, identical across arms (the same array fed every read) and across the two inference seeds
    c0, w0, m0 = masks_for_c3[0]
    c1_, w1, m1 = masks_for_c3[1]
    same_sha = sha_by_seed[0] == sha_by_seed[1]
    ind = [res["seeds"][str(s)]["c3_independent_derivation"][surf] for s in SEEDS for surf, _ in SURFS]
    ind_ok = all((i.get("status") == "OK" and i.get("n_diff_gt_1e-4", 1) == 0) or str(i.get("status", "")).startswith("ABSENT") for i in ind)
    res["controls"]["C3_model_independence"] = {"mask_sha_seed0": sha_by_seed[0], "mask_sha_seed1": sha_by_seed[1], "sha_equal_across_inference_seeds": bool(same_sha),
                                               "windows_equal_across_seeds": bool(np.array_equal(np.stack([c0, w0]), np.stack([c1_, w1]))),
                                               "independent_derivation_from_panel_decisions_gt_future_ext": ind, "independent_derivation_OK": bool(ind_ok),
                                               "inputs_of_the_mask": "gt_max_speed_6s, v_max_ms, v_max_valid (logged data + the fed label); no model output",
                                               "independent_derivation_n_OK": int(sum(i.get("status") == "OK" for i in ind)), "independent_derivation_n_total": len(ind),
                                               "PASS": bool(same_sha and ind_ok)}
    res["controls"]["C1_all_true_mask_reproduces_banked_cells"] = {"PASS": bool(all(c1_all)), "per_seed": c1_all,
                                                                   "detail": "per seed/surface in seeds.*.surfaces.*.C1a_/C1b_ (delta, lo, hi, n_windows, n_episodes, separated, n_dropped_nonfinite, os means)"}
    res["controls"]["C2_rolled_mask_changes_window_set_and_a_headline_cell"] = {"PASS": bool(all(c2_all)) and bool(all(c1_all)), "per_seed_surface": c2_all, "roll_windows": ROLL}
    res["controls"]["C4_pose_reference_valid"] = c4_pose_control(tree, res, hp_by_seed, known)
    c_all = bool(all(c1_all) and res["controls"]["C3_model_independence"]["PASS"] and res["controls"]["C2_rolled_mask_changes_window_set_and_a_headline_cell"]["PASS"]
                 and res["controls"]["C4_pose_reference_valid"]["PASS"])
    res["controls"]["ALL_PASS"] = c_all
    # ---- bars: all-window (recomputed; must equal the banked verdicts) and compliant; capped primary, uncapped beside
    res["bars"] = {"status": None, "E1_E3_reference": E1REF_LABEL, "E1_E3_reference_beside": E1REF_PRE_LABEL}
    allw = {v: bars_of(PR, reads["all_window"][v]) for v in VARIANTS}
    res["bars"]["all_window"] = allw
    # the banked verdicts are the REGISTERED read (os - ha0_ext): the control compares them with BAR-R8-E1_ha0_ext, the pose-scored BAR-R8-E1 has no banked twin
    verdict_check = {"uncapped_E1": allw["uncapped"]["BAR-R8-E1_ha0_ext"]["verdict"] == banked["bars"]["BAR-R8-E1"]["verdict"],
                     "uncapped_E3": allw["uncapped"]["BAR-R8-E3_ha0_ext"]["verdict"] == banked["bars"]["BAR-R8-E3"]["verdict"],
                     "uncapped_NR_per_family": allw["uncapped"]["BAR-R8-NR"]["per_family"] == banked["bars"]["BAR-R8-NR"]["per_family"]}
    if clamp_banked and isinstance(clamp_banked.get("bars"), dict) and "BAR-R8-E1_ON" in clamp_banked["bars"]:
        cbb = clamp_banked["bars"]
        verdict_check.update({"capped_E1": allw["capped"]["BAR-R8-E1_ha0_ext"]["verdict"] == cbb["BAR-R8-E1_ON"]["verdict"],
                              "capped_E3": allw["capped"]["BAR-R8-E3_ha0_ext"]["verdict"] == cbb["BAR-R8-E3_ON"]["verdict"],
                              "capped_NR_per_family": allw["capped"]["BAR-R8-NR"]["per_family"] == cbb["BAR-R8-NR_ON"]["per_family"]})
    res["controls"]["C1v_all_window_verdicts_equal_banked"] = {"PASS": bool(all(verdict_check.values())), "checks": verdict_check}
    c_all = bool(c_all and res["controls"]["C1v_all_window_verdicts_equal_banked"]["PASS"])
    res["controls"]["ALL_PASS"] = c_all
    if c_all and all(len(reads["compliant"][v]) == 4 for v in VARIANTS):
        res["bars"]["compliant"] = {v: bars_of(PR, reads["compliant"][v]) for v in VARIANTS}
        res["bars"]["status"] = "WRITTEN (all controls passed)"
    else:
        res["bars"]["compliant"] = "WITHHELD: a control did not pass (C1/C1v/C2/C3) -- no compliant number is written"
        res["bars"]["status"] = "WITHHELD"
        for sd in res["seeds"].values():       # nothing compliant leaves the file when a control failed
            for srec in sd["surfaces"].values():
                if "compliant" in srec:
                    srec["compliant"] = "WITHHELD: a control did not pass"
            c = sd["compliance"]["grids"]
            for g in c.values():
                for arm in g["arms"].values():
                    arm["compliant_set"] = "WITHHELD (control failed)"
    res["wall_s"] = round(time.time() - t0, 1)
    if mutate_pose:
        res["MUTATION_ARM"] = ("DELIBERATE REGRESSION (D-R8-E1-REF control c): the pose prior is ha0_ext. NOT A RESULT. C4 must be RED: "
                               + str(not res["controls"]["C4_pose_reference_valid"]["PASS"]))
        os.makedirs(out_json.parent, exist_ok=True)
    jdump(res, str(out_json))
    print(f"[{tree}] WROTE {out_json} ALL_PASS={c_all} {res['wall_s']}s", flush=True)
    return 0 if c_all else 4


# ------------------------------------------------------------------------------------------ strict-cap (brake 0) sensitivity (section 2.4)
def strict_tree(tree: str) -> int:
    """E1 / E3 on the compliant set with the STRICT cap (brake 0), beside the landed 2.5 m/s^2.  The all-True strict read is cross-checked against
    CLAMP_POSTHOC's SENSITIVITY_brake.json (brake 0.0) cell by cell before the compliant strict number is used."""
    main_p = OUT / "trees" / f"{tree}.json"
    if not main_p.exists():
        print(f"[strict] run `read --tree {tree}` first")
        return 3
    main = jload(str(main_p))
    if not main["controls"].get("ALL_PASS"):
        print(f"[strict] {tree}: main read controls did not pass -- no strict number")
        return 4
    ok, why, banked = tree_complete(tree)
    rc = CP.run_cfg()
    tools = tools_load()
    PR, V, NR, ra = tools["PR"], tools["V"], tools["NR"], tools["ra"]
    sens_p = CP.OUT / "SENSITIVITY_brake.json"
    sens = jload(str(sens_p)) if sens_p.exists() and tree == PRIMARY else None
    out = {"tool": "compliant_reread_r8.py strict", "tree": tree, "tier": TIER, "estimator": QUESTION, "generated": time.strftime("%FT%T"),
           "status": "SENSITIVITY (AMENDMENT A-R8-CLAMP-2 section 2.4): the strict cap, brake 0 = the literal R1, beside the landed 2.5 m/s^2; no verdict", "seeds": {}}
    reads = {"strict_comp": {}, "strict_all": {}}
    xchk = []
    t0 = time.time()
    for seed in SEEDS:
        ctx = prepare_seed(tree, seed, tools, rc)
        attach_pose_reference(tools, ctx)                           # D-R8-E1-REF: the bar reference (same builder as `read`)
        mask = ctx["comp"]
        pv = plan_variants(V, ctx, rc, brake=0.0)                   # "capped" key = the strict-capped plan
        resid = pv["diag"]["plan_cap_excess_ms"].numpy()
        sd = {"mask_sha256": ctx["sha"], "binding_rate_all_windows": CP.cluster_rate(pv["diag"]["plan_cap_binding"].numpy().astype(float), ctx["clip"]),
              "n_residual_gt_0p01_ms_all_windows": int((resid > CP.EXEMPT_TOL_COARSE).sum()), "n_residual_strict_gt0_incl_fp_dust": int((resid > 0).sum()),
              "n_residual_gt_0p01_ms_on_compliant": int(((resid > CP.EXEMPT_TOL_COARSE) & mask).sum()), "surfaces": {}}
        for surf, _pat in SURFS:
            P = ctx["panels"][surf]
            A, dt, slots, rows, eid = P["A"], P["dt"], P["slots"], P["rows"], P["eid"]
            os_s = pv["capped"][rows][:, slots]
            comp_os = PR.components(ra, os_s, A["g"], dt)
            fixed = {"ha0_ext": PR.components(ra, A["ha0_ext"], A["g"], dt), "ref": PR.components(ra, A[ctx["refarm"]], A["g"], dt),
                     "ha0_ext_pose": PR.components(ra, A["ha0_ext_pose"], A["g"], dt)}
            hm = [t for t in PR.METRICS if t[1] == "ade_m"]
            ones = np.ones(len(eid), bool)
            r_all = read_masked(PR, {"os": comp_os, **fixed}, eid, ones, metrics=hm)
            r_cmp = read_masked(PR, {"os": comp_os, **fixed}, eid, mask[rows], metrics=hm)
            rec = {"all_window": r_all, "compliant": r_cmp}
            if sens is not None:
                sc = sens["cells"][f"s{seed}_brake0.0"][surf]["ade_ON_minus_ha0_ext"]
                got = r_all["rows"]["ade_m"]["os_minus_ha0_ext"]
                eq = all(sc.get(f) == got.get(f) for f in ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite"))
                rec["crosscheck_all_true_vs_SENSITIVITY_brake_json"] = {"equal": bool(eq), "banked": [sc["delta"], sc["lo"], sc["hi"]], "reread": [got["delta"], got["lo"], got["hi"]]}
                xchk.append(eq)
            reads["strict_comp"][(seed, surf)] = {"means": r_cmp["means"], "rows": {"ade_m": r_cmp["rows"]["ade_m"]}}
            reads["strict_all"][(seed, surf)] = {"means": r_all["means"], "rows": {"ade_m": r_all["rows"]["ade_m"]}}
            sd["surfaces"][surf] = rec
        # compliance of the strict-capped plan on the S2 panel grid, all windows and compliant set
        P = ctx["panels"]["S2"]
        hz = [rc["horizons"][i] for i in P["slots"]]
        pm = pmax(V, pv["capped"][P["rows"]][:, P["slots"]], hz, rc["tick_s"])
        sd["compliance_S2_panel_grid_strict_cap"] = compliance_sets(pm, ctx["lim"][P["rows"]], ctx["v0"][P["rows"]], P["eid"], {"all_windows": None, "compliant_set": mask[P["rows"]]})
        out["seeds"][str(seed)] = sd
        print(f"[strict] seed {seed} done {time.time() - t0:.0f}s", flush=True)
        del ctx, pv
    fl = {}
    for kind in ("strict_comp", "strict_all"):
        fl[kind] = {surf: abs(reads[kind][(0, surf)]["means"]["os"]["ade_m"] - reads[kind][(1, surf)]["means"]["os"]["ade_m"]) for surf in ("S2", "S6")}
    out["bars_strict"] = {}
    out["ruling"] = E1REF_RULING
    for kind in ("strict_comp", "strict_all"):
        def sbar(surf, which, label, kind=kind):
            b = PR.bar_E({sd_: reads[kind][(sd_, surf)]["rows"]["ade_m"][which] for sd_ in SEEDS}, fl[kind][surf])
            b["reference"], b["reference_cell"] = label, which
            return b
        out["bars_strict"][kind] = {"inference_floor_ADE_m": fl[kind],
                                    "BAR-R8-E1": sbar("S2", "os_minus_ha0_ext_pose", E1REF_LABEL), "BAR-R8-E3": sbar("S6", "os_minus_ha0_ext_pose", E1REF_LABEL),
                                    "BAR-R8-E1_ha0_ext": sbar("S2", "os_minus_ha0_ext", E1REF_PRE_LABEL), "BAR-R8-E3_ha0_ext": sbar("S6", "os_minus_ha0_ext", E1REF_PRE_LABEL)}
    out["crosscheck_all_true_strict_equals_SENSITIVITY_brake_json"] = {"all_equal": bool(xchk) and all(xchk), "n_cells": len(xchk)}
    mc = main["bars"]["compliant"]["capped"] if isinstance(main["bars"]["compliant"], dict) else {}
    out["landed_brake_2p5_compliant_capped_beside"] = {"E1": (mc.get("BAR-R8-E1") or {}).get("verdict"), "E3": (mc.get("BAR-R8-E3") or {}).get("verdict"),
                                                      "E1_ha0_ext": (mc.get("BAR-R8-E1_ha0_ext") or {}).get("verdict"), "E3_ha0_ext": (mc.get("BAR-R8-E3_ha0_ext") or {}).get("verdict"),
                                                      "reference_of_E1_E3": E1REF_LABEL}
    jdump(out, str(OUT / f"STRICT_CAP_{tree}.json"))
    print("WROTE", OUT / f"STRICT_CAP_{tree}.json", "crosscheck", out["crosscheck_all_true_strict_equals_SENSITIVITY_brake_json"])
    return 0


# ------------------------------------------------------------------------------------------ assemble
def _c(c, kind="worse") -> str:
    return CP._c(c, kind)


def _pct(r) -> str:
    return CP._pct(r) if isinstance(r, dict) and "mean" in r else "n/a"


def _ord(t: str) -> tuple:
    import re
    m = re.match(r"step(\d+)(ema)?", t)
    return (int(m.group(1)), 1 if m.group(2) else 0, t) if m else (10 ** 9, 0, t)


def assemble() -> int:
    from pair_read_r8 import METRICS as M
    files = sorted((OUT / "trees").glob("*.json"), key=lambda p: _ord(p.stem))
    trees = {p.stem: jload(str(p)) for p in files}
    if not trees:
        print("no trees read yet")
        return 3
    strict = {p.stem.replace("STRICT_CAP_", ""): jload(str(p)) for p in OUT.glob("STRICT_CAP_*.json")}
    S = {"tool": "compliant_reread_r8.py assemble", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION,
         "rule": "AMENDMENT_A_R8_CLAMP_2_COMPLIANT_WINDOWS.md (b6f7fac)",
         "status": "EARLY READS of banked milestones under the compliant-window rule; never a bar verdict (decided at step 48,852)",
         "ruling": E1REF_RULING, "E1_E3_reference": E1REF_LABEL, "E1_E3_reference_beside": E1REF_PRE_LABEL,
         "trees_read": list(trees), "per_tree_files": {t: str(OUT / "trees" / f"{t}.json") for t in trees}, "summary": {}, "strict_cap": strict or None}
    for t, r in trees.items():
        comp = r["bars"]["compliant"]
        sm = {"controls_ALL_PASS": r["controls"]["ALL_PASS"], "C1": r["controls"]["C1_all_true_mask_reproduces_banked_cells"]["PASS"],
              "C1v": r["controls"]["C1v_all_window_verdicts_equal_banked"]["PASS"],
              "C2": r["controls"]["C2_rolled_mask_changes_window_set_and_a_headline_cell"]["PASS"], "C3": r["controls"]["C3_model_independence"]["PASS"],
              "mask": {s: {"sha256": r["seeds"][s]["mask_sha256"], "n_compliant_S2": r["seeds"][s]["surfaces"]["S2"]["mask"]["n_windows"],
                           "n_total_S2": r["seeds"][s]["surfaces"]["S2"]["mask"]["n_total"],
                           "n_episodes_S2": r["seeds"][s]["surfaces"]["S2"]["mask"]["n_episodes_in_set"],
                           "n_compliant_S6": r["seeds"][s]["surfaces"]["S6"]["mask"]["n_windows"], "n_total_S6": r["seeds"][s]["surfaces"]["S6"]["mask"]["n_total"],
                           "n_episodes_S6": r["seeds"][s]["surfaces"]["S6"]["mask"]["n_episodes_in_set"]} for s in ("0", "1")}}
        vd = lambda b, k: (b.get(k) or {}).get("verdict")  # noqa: E731
        if isinstance(comp, dict):
            for v in VARIANTS:
                sm[f"compliant_{v}"] = {"E1": vd(comp[v], "BAR-R8-E1"), "E3": vd(comp[v], "BAR-R8-E3"), "E1_ha0_ext": vd(comp[v], "BAR-R8-E1_ha0_ext"),
                                        "E3_ha0_ext": vd(comp[v], "BAR-R8-E3_ha0_ext"), "NR": comp[v]["BAR-R8-NR"]["per_family"]}
        for v in VARIANTS:
            aw = r["bars"]["all_window"][v]
            sm[f"all_window_{v}"] = {"E1": vd(aw, "BAR-R8-E1"), "E3": vd(aw, "BAR-R8-E3"), "E1_ha0_ext": vd(aw, "BAR-R8-E1_ha0_ext"),
                                     "E3_ha0_ext": vd(aw, "BAR-R8-E3_ha0_ext"), "NR": aw["BAR-R8-NR"]["per_family"]}
        sm["E1_E3_reference"] = (r.get("bars") or {}).get("E1_E3_reference", "ha0_ext (pre-D-R8-E1-REF file: re-read this tree)")
        S["summary"][t] = sm
    jdump(S, str(OUT / "COMPLIANT_REREAD.json"))
    # ------------------------------------------------------------ markdown
    L = ["# COMPLIANT_REREAD -- the refcv8 bars on CEILING-COMPLIANT windows (AMENDMENT A-R8-CLAMP-2), early reads of the banked milestones", "",
         f"**TIER:** {TIER}. **ESTIMATOR / QUESTION:** {QUESTION} -- on the compliant set it is the episodes the set contains. Single training seed, two inference seeds (0, 1).",
         "**STATUS:** EARLY READS under the PI's compliant-window rule (A-R8-CLAMP-2 section 2.5). They never replace a banked verdict; the bars are decided at the final checkpoint (step 48,852).",
         "**Rule:** a window is compliant iff the logged ego's max realised speed over the window's valid 6-s future (`gt_max_speed_6s`) <= the fed ceiling (strict; unknown ceiling = +inf). "
         "One mask per tree, from logged data + the fed label only. **Capped** = the landed `cap_plan_to_limit` (de4e15b, brake 2.5 m/s^2) applied post-hoc to the banked emitted plan; "
         "**uncapped** = the banked emitted plan. Source: `code/compliant_reread_r8.py`, trees `raw/compliant_reread/trees/<tree>.json`.", ""]
    L += ["## 0. Controls (rule section 3) -- run BEFORE any compliant number was written", "",
          "| tree | C1 all-True mask = banked cells (uncapped vs PAIRED_READ, capped vs CLAMP_POSTHOC) | C1v all-window ha0_ext verdicts = banked | C2 rolled-35 mask changes the set + a headline cell | C3 mask sha equal across inference seeds + re-derived from panel decisions | C4 ha0_ext_pose arm valid (construction, model-free, 2026-09-26 + A0K0 known values) | ALL |",
          "|---|---|---|---|---|---|---|"]
    for t, r in trees.items():
        c = r["controls"]
        ind = c["C3_model_independence"]
        L.append(f"| {t} | {c['C1_all_true_mask_reproduces_banked_cells']['PASS']} | {c['C1v_all_window_verdicts_equal_banked']['PASS']} | {c['C2_rolled_mask_changes_window_set_and_a_headline_cell']['PASS']} | "
                 f"{ind['PASS']} (independent derivation OK: {ind['independent_derivation_OK']}) | {(c.get('C4_pose_reference_valid') or {}).get('PASS', 'n/a (pre-ruling file)')} | **{c['ALL_PASS']}** |")
    L.append("")
    L += ["## 1. The compliant set per tree", "",
          "| tree | seed | S2 compliant / total windows (episodes) | S6 compliant / total windows (episodes) | non-compliant rate, all S2 windows [episode-cluster CI] | capped plans that change on compliant windows | mask sha256 (first 16) |", "|---|---|---|---|---|---|---|"]
    for t, r in trees.items():
        for s in ("0", "1"):
            sd = r["seeds"][s]
            m2, m6 = sd["surfaces"]["S2"]["mask"], sd["surfaces"]["S6"]["mask"]
            L.append(f"| {t} | {s} | {m2['n_windows']} / {m2['n_total']} ({m2['n_episodes_in_set']}) | {m6['n_windows']} / {m6['n_total']} ({m6['n_episodes_in_set']}) | "
                     f"{_pct(sd['noncompliant_rate_all_windows'])} | {sd['n_changed_plans_on_compliant_windows']} | `{sd['mask_sha256'][:16]}` |")
    shas = {r["seeds"][s]["mask_sha256"] for r in trees.values() for s in ("0", "1")}
    L += ["", f"Distinct mask shas over every tree and seed read: **{len(shas)}** ({'one mask for every arm, tree and seed (C3)' if len(shas) == 1 else 'NOT one mask -- the trees differ in their window set; see per-tree sha'}).", ""]
    # ---- the bars
    L += ["## 2. The bars: compliant set vs all windows, capped (primary) vs uncapped (beside)", "",
          f"**D-R8-E1-REF (PI 2026-10-19, commit 7525bd7): BAR-R8-E1 / E3 are scored against {E1REF_LABEL}** -- refcv8's own causal prior (measured v0, backward-difference acceleration, "
          "yaw rate -> kappa0). The registered read against `ha0_ext` (steer-channel kappa0) is kept BESIDE as `BAR-R8-E1_ha0_ext` / `BAR-R8-E3_ha0_ext`. "
          "The switch followed the step-30,000 reads; the pre-40k trees below are EARLY READS under it.", "",
          "Verdicts exactly as `pair_read_r8` words them: E1/E3 = `os - reference` ADE < 0 and separated at BOTH inference seeds, NOT PROVEN inside 2x the inference floor (the floor is read on the same set); "
          "NR = per family, refcv8 - refcv7-50,400 not separated worse at BOTH seeds. (NR reference side unchanged: `D-R8-NR-INSTRUMENT` stays open.)", "",
          "| tree | arm | E1 compliant (pose) | E1 all-window (pose) | E3 compliant (pose) | E3 all-window (pose) | E1 compliant (ha0_ext, beside) | E3 compliant (ha0_ext, beside) |", "|---|---|---|---|---|---|---|---|"]
    vdt = lambda b, k: (b.get(k) or {}).get("verdict", "n/a (pre-ruling file)")  # noqa: E731
    for t, r in trees.items():
        comp = r["bars"]["compliant"]
        for v in VARIANTS:
            aw = r["bars"]["all_window"][v]
            if isinstance(comp, dict):
                L.append(f"| {t} | {v} | {vdt(comp[v], 'BAR-R8-E1')} | {vdt(aw, 'BAR-R8-E1')} | {vdt(comp[v], 'BAR-R8-E3')} | {vdt(aw, 'BAR-R8-E3')} | "
                         f"{vdt(comp[v], 'BAR-R8-E1_ha0_ext')} | {vdt(comp[v], 'BAR-R8-E3_ha0_ext')} |")
            else:
                L.append(f"| {t} | {v} | WITHHELD | {vdt(aw, 'BAR-R8-E1')} | WITHHELD | {vdt(aw, 'BAR-R8-E3')} | WITHHELD | WITHHELD |")
    L += ["", "NR per family (PASS = not separated worse at both seeds; `FAIL: ...` lists the failing headline cells):", "",
          "| tree | arm | LONGITUDINAL (compliant / all) | LATERAL (compliant / all) | ADE (compliant / all) |", "|---|---|---|---|---|"]
    for t, r in trees.items():
        comp = r["bars"]["compliant"]
        for v in VARIANTS:
            aw = r["bars"]["all_window"][v]["BAR-R8-NR"]["per_family"]
            cc = comp[v]["BAR-R8-NR"]["per_family"] if isinstance(comp, dict) else {}
            f = lambda fam: f"{cc.get(fam, 'WITHHELD')} / {aw.get(fam)}"  # noqa: E731
            L.append(f"| {t} | {v} | {f('LONGITUDINAL')} | {f('LATERAL')} | {f('ADE')} |")
    # ---- E1/E3 cells
    L += ["", "## 3. The E1 / E3 cells (`os - ha0_ext_pose` = the bar; `os - ha0_ext` beside; ADE; negative = better than the reference, B = CI upper < 0)", "",
          "| tree | arm | seed | E1 compliant, pose (n win / ep) | E1 all-window, pose | E3 compliant, pose (n win / ep) | E3 all-window, pose | E1 compliant, ha0_ext | E3 compliant, ha0_ext |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, r in trees.items():
        for v in VARIANTS:
            for s in ("0", "1"):
                sd = r["seeds"][s]["surfaces"]
                def cell(surf, which, ref="os_minus_ha0_ext_pose"):
                    x = sd[surf][which]
                    if isinstance(x, str):
                        return "WITHHELD", ""
                    c = x[v]["rows"]["ade_m"].get(ref)
                    if c is None:
                        return "n/a (pre-ruling file)", ""
                    return _c(c, "better"), f" ({c['n_windows']} / {c['n_episodes']})"
                e1c, e1n = cell("S2", "compliant")
                e3c, e3n = cell("S6", "compliant")
                e1a, _ = cell("S2", "all_window")
                e3a, _ = cell("S6", "all_window")
                e1x, _ = cell("S2", "compliant", "os_minus_ha0_ext")
                e3x, _ = cell("S6", "compliant", "os_minus_ha0_ext")
                L.append(f"| {t} | {v} | {s} | {e1c}{e1n} | {e1a} | {e3c}{e3n} | {e3a} | {e1x} | {e3x} |")
    # ---- NR cells on the primary tree
    prim = trees.get(PRIMARY)
    if prim is not None and isinstance(prim["bars"]["compliant"], dict):
        L += ["", f"## 4. The NR headline cells on the primary tree `{PRIMARY}` (refcv8 - refcv7-50,400, error metrics: W = CI lower > 0 = separated WORSE)", "",
              "| surface | metric | seed | CAPPED compliant (n win / ep) | CAPPED all-window | UNCAPPED compliant | UNCAPPED all-window |", "|---|---|---|---|---|---|---|"]
        for fam, key, label, unit, lower, surfs in M:
            for surf in surfs:
                for s in ("0", "1"):
                    sd = prim["seeds"][s]["surfaces"][surf]
                    cells = []
                    for v in ("capped", "uncapped"):
                        cc = sd["compliant"][v]["rows"][key]["os_minus_refcv7_50400"]
                        ca = sd["all_window"][v]["rows"][key]["os_minus_refcv7_50400"]
                        cells += [_c(cc, "worse") + (f" ({cc['n_windows']} / {cc['n_episodes']})" if v == "capped" else ""), _c(ca, "worse")]
                    L.append(f"| {surf} | {fam}: {label} ({unit}) | {s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    # ---- compliance rows
    L += ["", "## 5. The COMPLIANCE row (LONGITUDINAL family) -- rate of windows whose emitted plan exceeds the fed ceiling (`planned_max_speed`, strict), residual excess, v0 already over", "",
          "One refcv8 fed ceiling and one window set for every arm; refcv7-50,400 and the echo exist only as the panel's embedded arms, so the cross-arm table is on the panel grids "
          "(S2: 0.5-s waypoints 0-2 s; S6: 1-s waypoints 0-6 s). On the compliant set the human did not exceed the ceiling, so a plan above it is an R1 violation he did not commit: "
          "**that rate is the R1 metric**. `resid` = excess over the ceiling on the exceeding windows (median / p90 / max, m/s).", ""]
    for t, r in trees.items():
        if t != PRIMARY and not (len(trees) <= 2):
            continue
        for s in ("0", "1"):
            L += [f"### {t}, inference seed {s}", "", "| grid | arm | all windows: exceed rate [CI] (n) | resid med/p90/max | compliant set: exceed rate [CI] (n) | resid med/p90/max | v0 already over, all / compliant |", "|---|---|---|---|---|---|---|"]
            for gname, g in r["seeds"][s]["compliance"]["grids"].items():
                for arm in ARMS:
                    a = g["arms"].get(arm)
                    if a is None:
                        continue
                    aw = a["all_windows"]
                    cs = a["compliant_set"]
                    q = lambda x: "-" if not x or x.get("n", 0) == 0 else f"{x['median']}/{x['p90']}/{x['max']}"  # noqa: E731
                    cs_txt = ("WITHHELD", "-", "-") if isinstance(cs, str) else (f"{_pct(cs['exceed_rate_strict'])} ({cs['n_exceed_strict']}/{cs['n_windows']})", q(cs["residual_excess_ms_on_exceeding_windows"]),
                                                                           f"{_pct(cs['v0_over_ceiling_rate'])} ({cs['n_v0_over']})")
                    L.append(f"| {gname} | {arm} | {_pct(aw['exceed_rate_strict'])} ({aw['n_exceed_strict']}/{aw['n_windows']}) | {q(aw['residual_excess_ms_on_exceeding_windows'])} | "
                             f"{cs_txt[0]} | {cs_txt[1]} | {_pct(aw['v0_over_ceiling_rate'])} ({aw['n_v0_over']}) / {cs_txt[2]} |")
            L.append("")
    L += ["Compliance rate of the CAPPED refcv8 arm (the R1 metric) on the compliant set, every tree, S2 panel grid, strict > 0 / > 0.01 m/s (the second excludes fp32 dust of a plan capped AT the ceiling):", "",
          "| tree | seed | capped exceed (strict) | capped exceed (>0.01 m/s) | uncapped exceed (strict) | refcv7-50,400 exceed | echo ha0_ext exceed | echo ha0_ext_pose exceed |", "|---|---|---|---|---|---|---|---|"]
    for t, r in trees.items():
        for s in ("0", "1"):
            g = r["seeds"][s]["compliance"]["grids"]["S2_panel_grid"]["arms"]

            def cr(arm, k="exceed_rate_strict", n="n_exceed_strict"):
                if arm not in g:
                    return "n/a (pre-ruling file)"
                cs = g[arm]["compliant_set"]
                return "WITHHELD" if isinstance(cs, str) else f"{_pct(cs[k])} ({cs[n]}/{cs['n_windows']})"
            L.append(f"| {t} | {s} | {cr('refcv8_capped')} | {cr('refcv8_capped', 'exceed_rate_gt_0p01_ms', 'n_exceed_gt_0p01_ms')} | {cr('refcv8_uncapped')} | {cr('refcv7_50400')} | {cr('echo_ha0_ext')} | {cr('echo_ha0_ext_pose')} |")
    # ---- strict
    for t, st in strict.items():
        ref_note = (st["bars_strict"]["strict_comp"]["BAR-R8-E1"].get("reference") or "ha0_ext (PRE-RULING file: re-run `strict`)")
        L += ["", f"## 6. Strict-cap (brake 0) sensitivity, `{t}` (rule section 2.4; no verdict); E1/E3 reference: {ref_note}", "",
              f"Cross-check (the all-True strict read equals CLAMP_POSTHOC's SENSITIVITY_brake.json cell by cell, on the ha0_ext cell): **{st['crosscheck_all_true_strict_equals_SENSITIVITY_brake_json']}**.", "",
              "| set | E1 verdict | E1 seed 0 | E1 seed 1 | E3 verdict | landed 2.5 brake, compliant (E1 / E3) |", "|---|---|---|---|---|---|"]
        for kind, nm in (("strict_comp", "compliant"), ("strict_all", "all windows")):
            b = st["bars_strict"][kind]
            e1 = b["BAR-R8-E1"]["per_inference_seed"]
            L.append(f"| {nm} | {b['BAR-R8-E1']['verdict']} | {_c({'delta': e1['0']['delta'], 'lo': e1['0']['lo'], 'hi': e1['0']['hi']}, 'better') if '0' in e1 else _c({'delta': e1[0]['delta'], 'lo': e1[0]['lo'], 'hi': e1[0]['hi']}, 'better')} | "
                     f"{_c({'delta': e1['1']['delta'], 'lo': e1['1']['lo'], 'hi': e1['1']['hi']}, 'better') if '1' in e1 else _c({'delta': e1[1]['delta'], 'lo': e1[1]['lo'], 'hi': e1[1]['hi']}, 'better')} | "
                     f"{b['BAR-R8-E3']['verdict']} | {st['landed_brake_2p5_compliant_capped_beside']['E1']} / {st['landed_brake_2p5_compliant_capped_beside']['E3']} |")
        for s in ("0", "1"):
            c = st["seeds"][s]["compliance_S2_panel_grid_strict_cap"]["compliant_set"]
            L.append(f"\nStrict-capped plan, seed {s}, compliant set, S2 panel grid: exceed (strict) {_pct(c['exceed_rate_strict'])} ({c['n_exceed_strict']}/{c['n_windows']}); >0.01 m/s: {c['n_exceed_gt_0p01_ms']}.")
    L += ["", "## 7. What this is and is not", "",
          "* TIER T1 self-action OPEN LOOP on logged eval139 frames: a plan scored against the logged human, **not driving performance**.",
          "* The estimator answers 'another draw of EPISODES' (models fixed). It does NOT answer 'another training run' (`H-ESTIM-SEED-1`) or 'another inference run'; the inference floor is printed in the JSON beside every E1/E3 verdict.",
          "* The compliant set removes the windows where the logged human exceeds the fed ceiling (the cap changes only those: `n_changed_plans_on_compliant_windows` above). It is a property of the labels, not a model-dependent filter (C3).",
          "* Banked milestones are early reads; the final battery (step 48,852) decides. `step30000_unc` is read only when its PAIRED_READ is complete (`read --tree step30000_unc`).",
          "* `step5000_corr` (corrected pose clock) is not read: its PAIRED_READ is E1/E3-only (`--no-ref`), the GT differs by the pose shift, and CLAMP_POSTHOC's own tree list excludes it."]
    (OUT / "COMPLIANT_REREAD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("WROTE", OUT / "COMPLIANT_REREAD.json", OUT / "COMPLIANT_REREAD.md")
    return 0


def _cell_txt(c) -> str:
    if not isinstance(c, dict) or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def _ver(b) -> str:
    return (b.get("verdict") or "n/a").split(" (")[0] if isinstance(b, dict) else "n/a"


def series() -> int:
    """raw/compliant_reread/E1REF_SERIES.{md,json}: per tree and inference seed, the CAPPED, COMPLIANT-window E1 (S2) / E3 (S6) cell and bar verdict under
    ha0_ext_pose (D-R8-E1-REF) with the ha0_ext read beside.  Reads only the per-tree JSONs (no recomputation); a tree whose file predates the ruling is listed as NOT RE-READ."""
    files = sorted((OUT / "trees").glob("*.json"), key=lambda p: _ord(p.stem))
    rows, J = [], {"tool": "compliant_reread_r8.py series", "generated": time.strftime("%FT%T"), "ruling": E1REF_RULING, "reference": E1REF_LABEL,
                   "reference_beside": E1REF_PRE_LABEL, "tier": TIER, "estimator": QUESTION, "trees": {}}
    for p in files:
        t = p.stem
        r = jload(str(p))
        comp = (r.get("bars") or {}).get("compliant")
        if not isinstance(comp, dict) or "BAR-R8-E1_ha0_ext" not in comp.get("capped", {}):
            J["trees"][t] = {"status": "NOT RE-READ under D-R8-E1-REF (file predates the ruling or its controls withheld the numbers)",
                             "controls_ALL_PASS": (r.get("controls") or {}).get("ALL_PASS")}
            rows.append(f"| {t} | - | NOT RE-READ / WITHHELD | | | | | | | |")
            continue
        cap = comp["capped"]
        J["trees"][t] = {"controls_ALL_PASS": r["controls"]["ALL_PASS"], "C4": r["controls"].get("C4_pose_reference_valid", {}).get("PASS"),
                         "verdicts": {k: cap[k].get("verdict") for k in ("BAR-R8-E1", "BAR-R8-E3", "BAR-R8-E1_ha0_ext", "BAR-R8-E3_ha0_ext")},
                         "inference_floor_ADE_m": cap["inference_floor_ADE_m"], "seeds": {}}
        for s in ("0", "1"):
            sf = r["seeds"][s]["surfaces"]
            e1 = sf["S2"]["compliant"]["capped"]["rows"]["ade_m"]
            e3 = sf["S6"]["compliant"]["capped"]["rows"]["ade_m"]
            J["trees"][t]["seeds"][s] = {"E1_pose": e1["os_minus_ha0_ext_pose"], "E1_ha0_ext": e1["os_minus_ha0_ext"], "E3_pose": e3["os_minus_ha0_ext_pose"], "E3_ha0_ext": e3["os_minus_ha0_ext"],
                                         "n_windows_S2": e1["os_minus_ha0_ext_pose"]["n_windows"], "n_windows_S6": e3["os_minus_ha0_ext_pose"]["n_windows"]}
            rows.append(f"| {t} | {s} | {_cell_txt(e1['os_minus_ha0_ext_pose'])} | {_cell_txt(e1['os_minus_ha0_ext'])} | {_cell_txt(e3['os_minus_ha0_ext_pose'])} | "
                        f"{_cell_txt(e3['os_minus_ha0_ext'])} | {_ver(cap['BAR-R8-E1'])} / {_ver(cap['BAR-R8-E3'])} | {_ver(cap['BAR-R8-E1_ha0_ext'])} / {_ver(cap['BAR-R8-E3_ha0_ext'])} | "
                        f"{cap['inference_floor_ADE_m']['S2']:.4f} / {cap['inference_floor_ADE_m']['S6']:.4f} | {e1['os_minus_ha0_ext_pose']['n_windows']} / {e3['os_minus_ha0_ext_pose']['n_windows']} |")
    L = ["# E1REF_SERIES -- BAR-R8-E1 / E3 under the ha0_ext_pose reference (D-R8-E1-REF), capped plans on ceiling-compliant windows", "",
         f"**Reference:** {E1REF_LABEL} -- refcv8's own causal prior (measured v0, backward-difference acceleration, kappa0 = omega0 / max(v0, floor) from the backward yaw difference at t0). "
         "The registered `ha0_ext` read (steer-channel kappa0, curvature: inadmissible) stays BESIDE and is never dropped.", "",
         "**DISCLOSURE: this reference switch FOLLOWED the step-30,000 reads.** It was ruled by the PI on 2026-10-19 (~05:40 Berlin) after the 30k cells below had been seen; every row before step 40,000 is an "
         "EARLY READ re-scored under it, not a pre-registered test. Steps >= 40,000 are the first reads scored under it by construction.", "",
         f"**TIER:** {TIER}. **ESTIMATOR / QUESTION:** {QUESTION}. Single training seed, two inference seeds; cell = `os - reference` ADE, negative = better than the reference, "
         "verdict = bar_E (separated at BOTH inference seeds, NOT PROVEN inside 2x the inference floor). Capped plans (A-R8-CLAMP-1/2), compliant windows. "
         "The paired bootstrap answers 'another draw of EPISODES' only; `H-ESTIM-SEED-1` and the inference floor are NOT covered by the interval.", "",
         "| tree | inference seed | E1 S2 `os - pose` | E1 S2 `os - ha0_ext` (beside) | E3 S6 `os - pose` | E3 S6 `os - ha0_ext` (beside) | verdict E1 / E3 (pose) | verdict E1 / E3 (ha0_ext, beside) | inference floor S2 / S6 (m) | n windows S2 / S6 |",
         "|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    (OUT / "E1REF_SERIES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    jdump(J, str(OUT / "E1REF_SERIES.json"))
    print("WROTE", OUT / "E1REF_SERIES.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "strict", "assemble", "series"])
    ap.add_argument("--tree")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mutate-pose-to-ha0-ext", action="store_true",
                    help="D-R8-E1-REF deliberate-regression arm: swap the pose prior back to ha0_ext; writes raw/compliant_reread/mutation/<tree>.json; C4 must go RED")
    a = ap.parse_args()
    os.makedirs(OUT / "trees", exist_ok=True)
    if a.stage == "read":
        if not a.tree:
            ap.error("--tree is required for read")
        return read_tree(a.tree, a.force, mutate_pose=a.mutate_pose_to_ha0_ext)
    if a.stage == "strict":
        return strict_tree(a.tree or PRIMARY)
    if a.stage == "series":
        return series()
    rc_ = assemble()
    return rc_


if __name__ == "__main__":
    sys.exit(main())
