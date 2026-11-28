"""r9_e2_dose_verdict.py -- the PRE-REGISTERED DOSE-HYPOTHESIS reader of the refcv9 E2 K2DC arm (K2DC vs K2 = K2M1N20, with K0 as the floor).

Encodes PREREG_K2DC_2026-11-12.md "Read -> Dose hypothesis" EXACTLY (frozen; PI-approved "Approve K2DC"); nothing here may move after any K2DC step:

  DIAGNOSTIC, one training run per arm (H-ESTIM-SEED-1).  Per inference seed s in {0, 1}, DEV windows, paired episode-cluster bootstrap (taniteval.ci), delta = A - B:
  * SUPPORTED    at BOTH seeds: (i) K2DC - K0 LAT_cross_mae is NOT separated worse  AND  (ii) K2DC - K2 LAT_cross_mae <= -2.5 mm (half of K2 - K0 at seed 1)
                 AND  (iii) the frame-to-frame |d e| (lat_jitter.py `dlag`) K2DC - K2 is NEGATIVE.
  * REFUTED      at BOTH seeds: the K2DC - K2 LATERAL DISTANCE rows (LAT_cross_mae_m and LAT_cross_abs_final_m, the two lateral rows in metres) are within +-1 mm.
  * INCONCLUSIVE anything between; the next step is a replicate arm (same flags, seed + 1), NOT a reinterpretation.
  * DOSE-VOID    the run is valid for the dose question only if the trainer's logged trunk share (metrics.jsonl `dc_share_read`) lies in [0.05, 0.15] at >= 2 probes
                 at steps >= 3,000.  Outside => reported DOSE-VOID (the controller did not hold), never silently re-run.  The three outcomes above are still computed and
                 printed under `outcome_if_valid`; `outcome` reads DOSE-VOID.
  All four metric families are reported per seed (K2DC - K0 and K2DC - K2), per family and never pooled; STRATEGIC = REPORT-ONLY (PI 2026-11-16: shown beside, outside the decision; NOT-SUPPLIED when no rows).  The map primary / box AP /
  S6c rate K2DC - K2 are reported with NO committed direction (the PREREG says so), from the two reads' EVAL / S6c files when supplied.

The E2 GO rule itself (REFCV9_DESIGN 4.4) is NOT re-implemented here: `r9_e2_verdict.py` reads K2DC vs K0 exactly as it read K2.

Ambiguities in the PREREG text, resolved here and STATED in the output (`resolutions`): (1) "at both seeds" governs the whole SUPPORTED sentence; (2) "lateral rows" in the REFUTED
clause = the two lateral rows that are lengths (a heading in degrees or a curvature in 1/m has no millimetre band); (3) "separated worse" = CI excludes zero with delta > 0 (lower is
better); (4) the jitter clause is the POINT delta < 0 (the PREREG sets no CI requirement on it; its interval is reported).

CPU only.  Usage:
  python r9_e2_dose_verdict.py --k2dc-pf <E2_READ_K2DC/E2_K2_pf> --k2-pf <E2_READ_K2M1N20/E2_K2_pf> --k0-pf <E2_READ_K2M1N20/E2_K0_pf> --dev-split DEV_HOLD_SPLIT.json \
      --metrics <runs/r9e2-K2DC/metrics.jsonl> --out-dir <dir> [--arm K2DC02 (reads target/bounds from runs.d/r9e2_K2DC02.argv.json; default K2DC = unchanged)] [--k2dc-read <E2_READ_K2DC/read> --k2-read <E2_READ_K2M1N20/read>]
  (each --*-pf dir holds dump_s0 / dump_s1; PYTHONPATH = <tree>/stack:<tree>/stack/scripts:<tree>/taniteval, `;` on Windows.)
EXIT: 0 SUPPORTED | 1 REFUTED | 2 INCONCLUSIVE | 3 DOSE-VOID (or OP-VOID, K2SG) | 4 REFUSED.

ARM K2SG (the map->trunk stop-gradient arm; PREREG_K2SG_2026-11-19.md): `--arm K2SG` reads its argv json (default stack/ops/runs.d/r9e2_K2SG.argv.json), REFUSES unless
`--map-hires-trunk-grad-scale` appears exactly once with value exactly 0.0 and there is no `--map-hires-share-target`.  The decision constants are PREREG_K2SG section 3's (= K2DC's,
K2 in the comparator slot, the arm's dumps in the `--k2dc-pf` slot); the validity rule is section 2's exact identity (`void_check_sg`, needs --metrics): outcome OP-VOID when invalid.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "r9e2-dose-verdict-v1"
PREREG = "TanitAD-artifacts/dagland/e2/PREREG_K2DC_2026-11-12.md"
SEEDS = ("0", "1")
CROSS = "LAT_cross_mae_m"
LAT_DIST_ROWS = ("LAT_cross_mae_m", "LAT_cross_abs_final_m")
BAR_M = 0.0025              # SUPPORTED: K2DC - K2 LAT_cross_mae <= -2.5 mm (half of K2 - K0 at seed 1, which was +5.0 mm)
BAND_M = 0.0010             # REFUTED: K2DC - K2 lateral rows within +-1 mm
EPS = 1e-9                  # float slack on the two inclusive bounds (the CI rounds its point to 4+ digits)
SHARE_BAND = (0.05, 0.15)   # in-training validity: logged trunk share, design 2.2's dose band
SHARE_MIN_STEP = 3000
SHARE_MIN_PROBES = 2
ARM_DEFAULT = "K2DC"        # the default arm: every constant above is K2DC's (share target 0.10); `--arm K2DC` and no --arm read bit-for-bit as before
ARM_ARGV = {"K2DC02": "r9e2_K2DC02.argv.json", "K2DC02s1": "r9e2_K2DC02s1.argv.json", "K2DC02cl": "r9e2_K2DC02cl.argv.json"}      # arms whose dose settings are READ from stack/ops/runs.d/<file>, never typed here
# the pre-registration sentence each derived arm's document carries (MM 11-20: K2DC02s1 = the PREREG_K2DC02 section 4.4 replicate; K2DC02's text is unchanged byte for byte)
ARM_PREREG_NOTE = {
    "K2DC02": "Pre-registered for K2DC02 in e2/e2raw/PREREG_K2DC02_2026-11-18.md (landed 75d9ca52, before any K2DC02 training step): K2DC is the comparator "
              "in K2's slot (--k2-read = E2_READ_K2DC's read); `reason` text naming next arms is K2DC's, and that PREREG's section 4 replaces it.",
    "K2DC02s1": "Pre-registered for K2DC02s1 in e2/e2raw/PREREG_K2DC02S1_2026-11-20.md (landed 48c797f3, before any K2DC02s1 training step): K2DC (EMA) or K2DC-tail "
                "is the comparator in K2's slot; with K2DC02 (EMA) or K2DC02-tail in K2's slot the document is that PREREG's section 4 training-run FLOOR, "
                "a noise reading and never a lever verdict; `reason` text naming next arms is K2DC's, and that PREREG's section 6 replaces it.",
    "K2DC02cl": "Pre-registered for K2DC02cl in e2/e2raw/PREREG_K2DC02CL_2026-11-22.md (DRAFT FOR MM REVIEW until landed; to be landed before any K2DC02cl training step): the bars, comparators "
                "and estimator are PREREG_K2DC02's, unchanged, with K2DC (EMA) in K2's slot; validity adds the DOSE-MEAN check (mean late share inside the mean band, else DOSE-OFF-TARGET, "
                "which makes no claim about the 0.02 dose); `reason` text naming next arms is K2DC's, and that PREREG's section 5 replaces it.",
}
# K2DC02cl only (PREREG_K2DC02CL section 2 (iii)): the MEAN of the probe shares at step >= SHARE_MIN_STEP must lie inside this band, else DOSE-OFF-TARGET. Other arms: no mean check, documents byte-identical.
ARM_MEAN_BAND = {"K2DC02cl": (0.01, 0.035)}
ARGV_DIR = Path(__file__).resolve().parents[1] / "ops" / "runs.d"
BAND_REL = (0.5, 1.5)       # K2DC's band [0.05, 0.15] is 0.5x..1.5x of its target 0.10; a derived arm keeps the same RELATIVE band around its own target
EXIT = {"SUPPORTED": 0, "REFUTED": 1, "INCONCLUSIVE": 2, "DOSE-VOID": 3, "OP-VOID": 3, "DOSE-OFF-TARGET": 3}
ARM_SG = {"K2SG": "r9e2_K2SG.argv.json"}            # the stop-gradient arm: its lever is READ from the argv json, never typed here
PREREG_SG = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e2/e2raw/PREREG_K2SG_2026-11-19.md"
SG_FLAG = "--map-hires-trunk-grad-scale"
SG_SHARE_FLAG = "--map-hires-share-target"
SG_KEY = "gs_trunk_proj_map_hires"                  # must read EXACTLY 0.0 on every row at step >= 1
SG_CONTROL_KEY = "gs_trunk_proj_tac_v6"             # must read non-zero on those rows (the probe works)
SG_INFO_KEYS = ("gs_layer2_proj_map_hires", "gs_bev025_proj_map_hires", "gs_bev_pool_proj_map_hires")    # informational only, never in the rule
RESOLUTIONS = ("'at both seeds' governs the whole SUPPORTED sentence",
               "REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)",
               "'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)",
               "jitter clause = the POINT delta of the frame-to-frame |d e| (K2DC - K2) < 0; its interval is reported, no CI is required by the PREREG")
RESOLUTIONS_SG = ("K2SG: the arm's dumps / reads sit in the `--k2dc-*` slot and the JSON's internal keys keep the `dc_*` names (dc_vs_k0 = K2SG - K0, dc_vs_k2 = K2SG - K2); every label and text is K2SG",
                  "K2SG validity (PREREG_K2SG section 2, an exact identity, not a band): every telemetry row at step >= 1 that carries gs_trunk_proj_map_hires must read EXACTLY 0.0 (no tolerance); "
                  "zero such rows = OP-VOID (a missing check is a FAIL)",
                  "K2SG control: 'the tac_v6 row reads non-zero' = on EVERY such row gs_trunk_proj_tac_v6 is present, finite and != 0.0 (strict); otherwise OP-VOID ('the probe does not work')",
                  "K2SG informational only, never in the rule: the layer2 / bev025 / bev_pool map_hires proj readings per row and how many read exactly 0.0",
                  "'at both seeds' governs the whole SUPPORTED sentence",
                  "REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)",
                  "'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)",
                  "jitter clause = the POINT delta of the frame-to-frame |d e| (K2SG - K2) < 0; its interval is reported, no CI is required by the PREREG")
REASON_SG = {"SUPPORTED": "at both seeds K2SG-K0 LAT_cross_mae is not separated worse, K2SG-K2 LAT_cross_mae <= -2.5 mm and the frame-to-frame |d e| K2SG-K2 is negative",
             "REFUTED": "at both seeds the K2SG-K2 lateral distance rows are within +-1 mm: map gradient reaching the shared trunk is ELIMINATED as the lateral cause; next = K2 minus "
                        "--r9-hist-supervise-geom, or the planner L2-SP anchor (each its own pre-registration)",
             "INCONCLUSIVE": "between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation"}


def _fail(msg: str):
    raise SystemExit(f"[r9-e2-dose-verdict] REFUSED: {msg}")


def arm_dose(arm: str = ARM_DEFAULT, argv_json=None):
    """None for the default arm K2DC (nothing changes).  For a derived arm: its dose settings READ from its argv json (target and bounds are typed nowhere in this file):
    {"arm", "argv_json", "target", "bounds": [lo, hi], "share_band": [lo, hi]} with share_band = BAND_REL x target."""
    if arm == ARM_DEFAULT:
        if argv_json:
            _fail(f"--argv-json is only for a derived arm, not {ARM_DEFAULT}")
        return None
    if arm not in ARM_ARGV:
        _fail(f"unknown --arm {arm!r} (known: {ARM_DEFAULT}, {', '.join(sorted(ARM_ARGV))})")
    p = Path(argv_json) if argv_json else ARGV_DIR / ARM_ARGV[arm]
    try:
        toks = json.loads(p.read_text(encoding="utf-8"))["argv"]
        t = toks[toks.index("--map-hires-share-target") + 1]
        bnd = toks[toks.index("--map-hires-w-bounds") + 1]
        target = float(t)
        lo, hi = (float(x) for x in bnd.split(","))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        _fail(f"{p}: no usable --map-hires-share-target / --map-hires-w-bounds ({e!r})")
    if not (target > 0.0 and 0.0 < lo < hi):
        _fail(f"{p}: implausible dose settings target {target}, bounds {lo},{hi}")
    return {"arm": arm, "argv_json": str(p), "target": target, "bounds": [lo, hi],
            "share_band": [round(BAND_REL[0] * target, 6), round(BAND_REL[1] * target, 6)]}


def arm_sg(arm: str = "K2SG", argv_json=None) -> dict:
    """K2SG: read the lever from its argv json and REFUSE unless it is the stop-gradient at exactly 0.0 and has no share controller."""
    if arm not in ARM_SG:
        _fail(f"unknown stop-gradient arm {arm!r}")
    p = Path(argv_json) if argv_json else ARGV_DIR / ARM_SG[arm]
    try:
        toks = json.loads(p.read_text(encoding="utf-8"))["argv"]
        if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
            raise TypeError("argv is not a list of strings")
    except (OSError, ValueError, KeyError, TypeError) as e:
        _fail(f"{p}: unreadable argv ({e!r})")
    for t in toks:
        if t.startswith(SG_FLAG + "=") or t.startswith(SG_SHARE_FLAG + "="):
            _fail(f"{p}: `--flag=value` form is not accepted ({t!r})")
    if SG_SHARE_FLAG in toks:
        _fail(f"{p}: {SG_SHARE_FLAG} is present: K2SG has no share controller")
    n = toks.count(SG_FLAG)
    if n != 1:
        _fail(f"{p}: {SG_FLAG} appears {n} times, exactly once is required")
    i = toks.index(SG_FLAG)
    try:
        v = float(toks[i + 1])
    except (IndexError, ValueError) as e:
        _fail(f"{p}: {SG_FLAG} has no parsable value ({e!r})")
    if v != 0.0:
        _fail(f"{p}: {SG_FLAG} = {v!r}, exactly 0.0 is required")
    return {"arm": arm, "kind": "stop-gradient", "argv_json": str(p), "trunk_grad_scale": 0.0}


# ---------------------------------------------------------------------------------------------------------------- #
# the decision: pure functions of the per-seed numbers                                                              #
# ---------------------------------------------------------------------------------------------------------------- #
def _row(d: dict, key: str, where: str) -> dict:
    try:
        r = d[key]
        float(r["delta"])
    except (KeyError, TypeError, ValueError):
        _fail(f"{where}: no usable row {key!r} (got {sorted(d)[:6]})")
    return r


def seed_clauses(ps: dict, seed: str) -> dict:
    """ps = {"dc_vs_k0": {row_id: {delta, lo, hi, separated}}, "dc_vs_k2": {...}, "jitter_dlag_dc_vs_k2": {delta, lo, hi, separated}}."""
    w = _row(ps["dc_vs_k0"], CROSS, f"seed {seed} K2DC-K0")
    d = _row(ps["dc_vs_k2"], CROSS, f"seed {seed} K2DC-K2")
    j = ps["jitter_dlag_dc_vs_k2"]
    try:
        jd = float(j["delta"])
    except (KeyError, TypeError, ValueError):
        _fail(f"seed {seed}: no usable jitter row (got {j!r})")
    not_worse = not (bool(w["separated"]) and float(w["delta"]) > 0.0)
    bar = float(d["delta"]) <= -BAR_M + EPS
    jit = jd < 0.0
    band = {r: abs(float(_row(ps["dc_vs_k2"], r, f"seed {seed} K2DC-K2")["delta"])) <= BAND_M + EPS for r in LAT_DIST_ROWS}
    return {"i_dc_k0_cross_not_separated_worse": not_worse, "ii_dc_k2_cross_le_minus_2p5mm": bar, "iii_jitter_delta_negative": jit,
            "supported": bool(not_worse and bar and jit), "refuted_band": band, "refuted": bool(all(band.values()))}


def decide(per_seed: dict) -> dict:
    if set(per_seed) != set(SEEDS):
        _fail(f"inference seeds 0 AND 1 are both required, got {sorted(per_seed)}")
    cl = {s: seed_clauses(per_seed[s], s) for s in SEEDS}
    sup = all(cl[s]["supported"] for s in SEEDS)
    ref = all(cl[s]["refuted"] for s in SEEDS)
    if sup and ref:                                              # cannot happen (-2.5 mm vs +-1 mm); a contradiction in the constants is refused, not resolved
        _fail("SUPPORTED and REFUTED both hold: the bar and the band overlap")
    out = "SUPPORTED" if sup else "REFUTED" if ref else "INCONCLUSIVE"
    why = {"SUPPORTED": "at both seeds K2DC-K0 LAT_cross_mae is not separated worse, K2DC-K2 LAT_cross_mae <= -2.5 mm and the frame-to-frame |d e| K2DC-K2 is negative",
           "REFUTED": "at both seeds the K2DC-K2 lateral distance rows are within +-1 mm: the dose is ELIMINATED; next arm = K2 minus --r9-hist-supervise-geom, then K2 minus map-banded "
                      "(each its own pre-registration)",
           "INCONCLUSIVE": "between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation"}[out]
    return {"outcome_if_valid": out, "reason": why, "per_seed": cl}


# ---------------------------------------------------------------------------------------------------------------- #
# the in-training validity check                                                                                    #
# ---------------------------------------------------------------------------------------------------------------- #
def void_check(metrics_path, band=SHARE_BAND, bounds=None, mean_band=None) -> dict:
    """Reads the trainer's metrics.jsonl: rows carrying `dc_share_read` (map_dose_ctl.MapDoseController.fire, merged into the metrics row at the firing step)."""
    if metrics_path is None:
        return {"status": "NOT-CHECKED", "reason": "no --metrics given: the in-training validity condition was not read; the outcome is UNCHECKED-VOID", "valid": None}
    p = Path(metrics_path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        _fail(f"--metrics {p}: unreadable ({e})")
    probes, weffs = [], []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict) and "dc_share_read" in r and "step" in r:
            probes.append((int(r["step"]), float(r["dc_share_read"])))
            if "dc_w_eff" in r:
                weffs.append((int(r["step"]), float(r["dc_w_eff"])))
    late = [(s, v) for s, v in probes if s >= SHARE_MIN_STEP]
    inside = [(s, v) for s, v in late if band[0] <= v <= band[1]]
    out = {"metrics": str(p), "n_probes_total": len(probes), "n_probes_step_ge_3000": len(late), "n_inside_band": len(inside), "band": list(band),
           "rule": f">= {SHARE_MIN_PROBES} probes at step >= {SHARE_MIN_STEP} inside [{band[0]}, {band[1]}]",
           "probes_step_ge_3000": [{"step": s, "share": round(v, 5)} for s, v in late]}
    if bounds is not None:       # informational only (a derived arm): how often the controller sat ON its weight bounds at the late probes; never enters the validity rule
        lw = [w for s, w in weffs if s >= SHARE_MIN_STEP]
        out["w_bounds"] = list(bounds)
        out["n_late_w_eff_probes"] = len(lw)
        out["n_late_at_lower_bound"] = sum(1 for w in lw if w <= bounds[0] * (1 + 1e-6))
        out["n_late_at_upper_bound"] = sum(1 for w in lw if w >= bounds[1] * (1 - 1e-6))
    if not probes:
        out.update(status="VOID", valid=False, reason="metrics.jsonl holds no dc_share_read row: the controller never fired (or the log format changed): refusing to call the dose held")
    else:
        ok = len(inside) >= SHARE_MIN_PROBES
        out.update(status="HELD" if ok else "VOID", valid=ok, reason="" if ok else "the controller did not hold the share inside the band: DOSE-VOID")
    if mean_band is not None:    # K2DC02cl's DOSE-MEAN check (PREREG_K2DC02CL section 2 (iii)); a VOID by the >= 2-probe rule stays VOID (the mean is then only reported)
        vals = [v for s, v in late]
        mean = sum(vals) / len(vals) if vals else None
        mok = mean is not None and mean_band[0] <= mean <= mean_band[1]
        out.update(mean_band=list(mean_band), mean_share_step_ge_3000=None if mean is None else round(mean, 6), n_mean_probes=len(vals), mean_in_band=mok,
                   mean_rule=f"mean of the {SHARE_MIN_STEP}+ step probe shares inside [{mean_band[0]}, {mean_band[1]}]")
        if out.get("valid") is True and not mok:
            out.update(status="OFF-TARGET", valid=False, reason="the controller held the share in the band on >= 2 probes but the MEAN late share is outside the mean band: DOSE-OFF-TARGET (no claim about the 0.02 dose)")
    return out


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def void_check_sg(metrics_path) -> dict:
    """PREREG_K2SG section 2: S = 0 is an analytic identity, so the check is EXACT.  Rows with `gs_trunk_proj_map_hires` at step >= 1: none => OP-VOID; any value != 0.0 => OP-VOID
    (listing them); any such row whose `gs_trunk_proj_tac_v6` is missing / non-finite / 0.0 => OP-VOID (the probe does not work)."""
    if metrics_path is None:
        _fail("K2SG needs --metrics: the validity identity cannot be read without it")
    p = Path(metrics_path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        _fail(f"--metrics {p}: unreadable ({e})")
    rows, nonzero, bad_ctl, info = [], [], [], {k: [] for k in SG_INFO_KEYS}
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if not isinstance(r, dict) or SG_KEY not in r:
            continue
        st = r.get("step")
        if not isinstance(st, (int, float)) or isinstance(st, bool):
            nonzero.append({"step": None, "value": r[SG_KEY], "why": "row has no numeric step"})
            continue
        if st < 1:
            continue
        v, c = _num(r[SG_KEY]), (_num(r[SG_CONTROL_KEY]) if SG_CONTROL_KEY in r else None)
        rows.append((int(st), v, c))
        if v is None or v != 0.0:
            nonzero.append({"step": int(st), "value": r[SG_KEY]})
        if c is None or c == 0.0:
            bad_ctl.append({"step": int(st), "value": r.get(SG_CONTROL_KEY)})
        for k in SG_INFO_KEYS:
            info[k].append({"step": int(st), "value": r.get(k)})
    out = {"metrics": str(p), "rule": f"{SG_KEY} == 0.0 EXACTLY on every row at step >= 1 (>= 1 row), and {SG_CONTROL_KEY} present and != 0.0 on each of those rows",
           "n_rows": len(rows), "n_trunk_nonzero": len(nonzero), "trunk_nonzero": nonzero, "n_control_bad": len(bad_ctl), "control_bad": bad_ctl,
           "informational_not_in_rule": {k: {"n_rows": len(v), "n_exactly_zero": sum(1 for x in v if _num(x["value"]) == 0.0), "readings": v} for k, v in info.items()}}
    if not rows and not nonzero:
        out.update(status="OP-VOID", valid=False, reason=f"metrics.jsonl holds no {SG_KEY} row at step >= 1: a missing check is a FAIL")
    elif nonzero:
        out.update(status="OP-VOID", valid=False, reason=f"{SG_KEY} is not exactly 0.0 at {len(nonzero)} row(s): the stop-gradient did not hold; fix the wiring before any claim")
    elif bad_ctl:
        out.update(status="OP-VOID", valid=False, reason=f"{SG_CONTROL_KEY} is missing or 0.0 at {len(bad_ctl)} row(s): the probe does not work")
    else:
        out.update(status="VALID", valid=True, reason="")
    return out


# ---------------------------------------------------------------------------------------------------------------- #
# the numbers: families (r9_e2_families.build) and the jitter (lat_jitter.py's dlag)                                #
# ---------------------------------------------------------------------------------------------------------------- #
def _families():
    import r9_e2_families as F
    return F


def jitter_dlag(a_dir, b_dir, dev: set, n_boot: int = 2000, seed: int = 0) -> dict:
    """lat_jitter.py's `dlag`: signed final-waypoint lateral error e_w = P[w, -1, 1] - G[w, -1, 1]; per episode (windows ordered by ws) |e_{w+1} - e_w|; paired episode-cluster
    bootstrap, delta = A - B.  The arms must hold the SAME windows and the SAME GT (REFUSED otherwise)."""
    import numpy as np
    from taniteval import ci
    F = _families()

    def per_episode(d):
        dd, _ = F.restrict_dev(F.load_dump(d), dev)
        by = defaultdict(list)
        for i, (c, w) in enumerate(dd["keys"]):
            by[c].append((w, i))
        out = {}
        for c, lst in by.items():
            lst.sort()
            idx = [i for _, i in lst]
            out[c] = ([w for w, _ in lst], dd["P"][idx, -1, 1] - dd["G"][idx, -1, 1], dd["G"][idx])
        return out
    A, B = per_episode(a_dir), per_episode(b_dir)
    if set(A) != set(B):
        _fail(f"jitter: episode sets differ ({len(set(A) ^ set(B))} episodes only in one arm)")
    a, b, eid = [], [], []
    for c in sorted(A):
        if A[c][0] != B[c][0]:
            _fail(f"jitter: episode {c} holds different windows in the two arms")
        if float(np.abs(A[c][2] - B[c][2]).max()) > 1e-4:
            _fail(f"jitter: episode {c} GT differs between the arms (different label clocks)")
        da, db = np.abs(np.diff(A[c][1])), np.abs(np.diff(B[c][1]))
        a += list(da)
        b += list(db)
        eid += [c] * len(da)
    if not a:
        _fail("jitter: no consecutive-window pair in any episode")
    a, b = np.asarray(a), np.asarray(b)
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed)
    return {"row": "frame-to-frame |d e| of the final-waypoint lateral error (lat_jitter.py dlag), m", "n": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]),
            "a_mean": float(a.mean()), "b_mean": float(b.mean()), "delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "better": "lower"}


def _slim(r: dict) -> dict:
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "better", "n", "n_episodes", "a_mean", "b_mean") if k in r}


STRATEGIC_REASON = ("REPORT-ONLY: the strategic layer was switched back on by the PI on 2026-11-16 and is reported in full; the dose decision and the E2 rule were pre-registered "
                    "over LONGITUDINAL / LATERAL / TACTICAL, so STRATEGIC rows are shown BESIDE them and never enter the decision; never a pass, never omitted.")
STRATEGIC_PRODUCER = ("taniteval/tools/openloop_suite.py:943 (_strategic_block: route head vs the v2.1 route label under nav true / shuffled / zero; headline :1140) "
                      "and taniteval/tools/refcv3_arm.py:2646 (_navcomp_row)")
STRATEGIC_NOT_SUPPLIED = ("NOT-SUPPLIED: the paired dump read produced no STRATEGIC rows (never a pass, never omitted); the strategic numbers are computed by " + STRATEGIC_PRODUCER
                          + " and are not yet paired rows; reported beside the decision, outside it.")
STRATEGIC_WORSE = "separated worse (report-only, outside the decision)"


def strategic_entry(raw) -> dict:
    """One per-family STRATEGIC entry: REPORT-ONLY with its paired rows (only dict rows carrying `better` count; a stamp dict has none) or NOT-SUPPLIED."""
    got = {rid: r for rid, r in raw.items() if isinstance(r, dict) and "better" in r} if isinstance(raw, dict) else {}
    if not got:
        return {"status": "NOT-SUPPLIED", "reason": STRATEGIC_NOT_SUPPLIED, "rows": {}}
    rows = {}
    for rid, r in got.items():
        s = _slim(r)
        good = (s.get("delta", 0) > 0) if s.get("better") == "higher" else (s.get("delta", 0) < 0)
        s["separated_worse"] = bool(s.get("separated") and not good)
        if s["separated_worse"]:
            s["report_only_note"] = STRATEGIC_WORSE
        rows[rid] = s
    return {"status": "REPORT-ONLY", "reason": STRATEGIC_REASON, "rows": rows, "separated_worse_rows": [rid for rid, s in rows.items() if s["separated_worse"]]}


def strategic_top(numbers: dict) -> dict:
    """The document-level STRATEGIC block: per seed and per pairing, the entry's status and its separated-worse rows (no decision input)."""
    per = {s: {tag: {k: v for k, v in (numbers[s].get(tag + "_by_family") or {}).get("STRATEGIC", {}).items() if k in ("status", "separated_worse_rows")}
               for tag in ("dc_vs_k2", "dc_vs_k0")} for s in numbers}
    sup = any(e.get("status") == "REPORT-ONLY" for p in per.values() for e in p.values())
    return {"status": "REPORT-ONLY" if sup else "NOT-SUPPLIED", "reason": STRATEGIC_REASON if sup else STRATEGIC_NOT_SUPPLIED, "per_seed": per}


def seed_numbers(k2dc_pf, k2_pf, k0_pf, dev_split, seed: str, n_boot: int = 2000, label: str = "K2DC") -> dict:
    F = _families()
    dev = F.load_dev(dev_split)
    sub = lambda root: Path(root) / f"dump_s{seed}"                                  # noqa: E731
    out = {"dumps": {"k2dc": str(sub(k2dc_pf)), "k2": str(sub(k2_pf)), "k0": str(sub(k0_pf))}}
    docs = {}
    for tag, b, lab in (("dc_vs_k0", k0_pf, "K0"), ("dc_vs_k2", k2_pf, "K2")):
        doc = F.build(sub(k2dc_pf), sub(b), mode="families", dev=dev, n_boot=n_boot, labels=(label, lab))
        docs[tag] = doc
        out[tag] = {rid: _slim(r) for fam in F.FAMILIES for rid, r in doc["rows"][fam].items()}
        out[tag + "_ade_m"] = _slim(doc["ade_m"])
        out[tag + "_by_family"] = {fam: {rid: _slim(r) for rid, r in doc["rows"][fam].items()} for fam in F.FAMILIES}
        out[tag + "_by_family"]["STRATEGIC"] = strategic_entry(doc["rows"].get("STRATEGIC"))
        out[tag + "_window_set_sha256"] = doc["window_set_sha256"]
        out[tag + "_n_windows_paired"] = doc["n_windows_paired"]
    if out["dc_vs_k0_window_set_sha256"] != out["dc_vs_k2_window_set_sha256"]:
        _fail(f"seed {seed}: the K2DC-K0 and K2DC-K2 pairings use different window sets")
    out["jitter_dlag_dc_vs_k2"] = jitter_dlag(sub(k2dc_pf), sub(k2_pf), dev, n_boot)
    out["jitter_dlag_dc_vs_k0"] = jitter_dlag(sub(k2dc_pf), sub(k0_pf), dev, n_boot)
    return out


def _dig(o, *ks):
    for k in ks:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def beside(k2dc_read, k2_read) -> dict:
    """Map primary / box AP (T0 evals) and the S6c sidewalk rate, K2DC - K2, NO committed direction (PREREG).  Same digs as r9_e2_verdict.build."""
    if not k2dc_read or not k2_read:
        return {"status": "NOT-SUPPLIED", "reason": "--k2dc-read / --k2-read not given"}
    a, b = Path(k2dc_read), Path(k2_read)
    res = {"status": "REPORTED", "direction": "none committed (PREREG: 'map primary K2DC - K2 reported with no committed direction')", "tier": "T0 label level (map, box); T1 (S6c)"}

    def ev(p):
        try:
            return json.loads((p / "EVAL_K2.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    ea, eb = ev(a), ev(b)
    for name, path in (("map_primary_nf_swedge_prec_lv_5_20", ("nearfield", "ci", "nf_swedge_prec_lv_5_20", "point")), ("box_ap2m_all", ("boxes", "ap2m_all", "point"))):
        va, vb = _dig(ea, *path), _dig(eb, *path)
        res[name] = {"k2dc": va, "k2": vb, "delta": (None if va is None or vb is None else float(va) - float(vb))}
    for s in SEEDS:
        def rate(p):
            try:
                j = json.loads((p / "s6c" / f"S6C_K2_s{s}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return j.get("rate") if "rate" in j else _dig(j, "map_compliance", "S6c", "arms", "BASE", "off_drivable", "rate")
        ra, rb = rate(a), rate(b)
        res[f"s6c_sidewalk_rate_s{s}"] = {"k2dc": ra, "k2": rb, "delta": (None if ra is None or rb is None else float(ra) - float(rb)), "better": "lower"}
    return res


# ---------------------------------------------------------------------------------------------------------------- #
def build(numbers: dict, void: dict, extra: dict | None = None, arm: dict | None = None) -> dict:
    d = decide({s: {k: numbers[s][k] for k in ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")} for s in SEEDS})
    sg = arm is not None and arm.get("kind") == "stop-gradient"
    if sg:
        d["reason"] = REASON_SG[d["outcome_if_valid"]]
    if void.get("valid") is False:
        outcome = "OP-VOID" if sg else "DOSE-OFF-TARGET" if void.get("status") == "OFF-TARGET" else "DOSE-VOID"
    elif void.get("valid") is None:
        outcome = d["outcome_if_valid"] + " (UNCHECKED-VOID: the in-training share condition was not read)"
    else:
        outcome = d["outcome_if_valid"]
    doc = {"schema": SCHEMA, "class": "DIAGNOSTIC (one training run per arm, H-ESTIM-SEED-1)", "prereg": PREREG, "outcome": outcome, **d, "dose_validity": void,
           "thresholds": {"bar_m": -BAR_M, "band_m": BAND_M, "share_band": list(SHARE_BAND), "share_min_step": SHARE_MIN_STEP, "share_min_probes": SHARE_MIN_PROBES},
           "resolutions": list(RESOLUTIONS),
           "tier": "T1 (self-action open loop on logged eval139 frames; the `os` arm)", "windows": "DEV (DEV_HOLD_SPLIT.json); HOLD is never read",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers 'another draw of EPISODES', not 'another training run' "
                        "(H-ESTIM-SEED-1) and not 'another inference draw' (each seed is one call)",
           "families_note": "four families per seed, per family, never pooled: LONGITUDINAL / LATERAL / TACTICAL rows in `seeds.<s>.dc_vs_k2_by_family` and `dc_vs_k0_by_family`; ade_m beside; "
                            "STRATEGIC = REPORT-ONLY (PI 2026-11-16), shown beside the decision and never in it; NOT-SUPPLIED when no rows; never a pass, never omitted",
           "STRATEGIC": strategic_top(numbers),
           "seeds": numbers}
    if extra:
        doc["beside_no_direction"] = extra
    if sg:                       # K2SG: its own prereg, validity (`op_validity`), thresholds and labels; nothing below is shared with the dose arms
        doc.pop("dose_validity")
        doc["op_validity"] = void
        doc["prereg"] = PREREG_SG
        doc["thresholds"] = {"bar_m": -BAR_M, "band_m": BAND_M, "trunk_proj_map_hires": "== 0.0 exactly at every row, step >= 1", "control": f"{SG_CONTROL_KEY} != 0.0 at every such row"}
        doc["resolutions"] = list(RESOLUTIONS_SG)
        doc["families_note"] = doc["families_note"].replace("K2DC", "K2SG")
        doc["arm"] = {**arm, "note": "PREREG_K2SG_2026-11-19.md section 3: PREREG_K2DC_2026-11-12's constants applied UNCHANGED with K2 in the comparator slot; section 2's exact-identity validity; "
                                     "section 4's next steps replace K2DC's. Pre-registered before any K2SG training step. One training run per arm: a separated interval answers "
                                     "'another draw of EPISODES' only (H-ESTIM-SEED-1)."}
    elif arm is not None:          # a derived arm only; the K2DC document has no `arm` key and its thresholds are untouched
        doc["thresholds"]["share_band"] = list(arm["share_band"])
        doc["arm"] = {**arm, "note": "the pre-registered decision constants (bar, band, jitter clause, two seeds) are PREREG_K2DC_2026-11-12's, applied UNCHANGED to this arm; the in-training share band is "
                                     "BAND_REL x this arm's own target (read from its argv json). " + ARM_PREREG_NOTE[arm["arm"]]}
    return doc


def md(doc: dict) -> str:
    if doc.get("arm", {}).get("kind") == "stop-gradient":        # K2SG: same tables, K2SG labels, its own validity block
        ov = doc["op_validity"]
        return _md_body(doc, f"# K2SG mechanism (map->trunk stop-gradient): **{doc['outcome']}**",
                        f"In-training validity (`op_validity`): **{ov.get('status')}** ({ov.get('rule', '')}; {ov.get('n_rows')} telemetry rows, {ov.get('n_trunk_nonzero')} non-zero trunk readings, "
                        f"{ov.get('n_control_bad')} bad control readings). {ov.get('reason', '')}").replace("K2DC", "K2SG")
    return _md_body(doc, f"# {doc.get('arm', {}).get('arm', ARM_DEFAULT)} dose hypothesis: **{doc['outcome']}**",
                    f"In-training validity (`dose_validity`): **{doc['dose_validity'].get('status')}** ({doc['dose_validity'].get('rule', '')}; "
                    f"{doc['dose_validity'].get('n_inside_band', '-')} of {doc['dose_validity'].get('n_probes_step_ge_3000', '-')} late probes inside the band)."
                    + (f" DOSE-MEAN: mean late share {doc['dose_validity'].get('mean_share_step_ge_3000')} vs mean band {doc['dose_validity'].get('mean_band')}: "
                       f"{'inside' if doc['dose_validity'].get('mean_in_band') else 'OUTSIDE'}." if "mean_band" in doc["dose_validity"] else ""))


def _md_body(doc: dict, head: str, validity: str) -> str:
    L = [head, "", f"{doc['reason']}", "",
         f"*{doc['class']}.* Pre-registration: `{doc['prereg']}`. Tier: {doc['tier']}. Windows: {doc['windows']}.", "", f"Estimator: {doc['estimator']}.", "",
         validity, "",
         "## Pre-registered rows (delta = A - B, metres; lower is better)", "",
         "| seed | K2DC-K0 LAT_cross_mae | separated worse? | K2DC-K2 LAT_cross_mae | <= -2.5 mm | K2DC-K2 LAT_cross_abs_final | K2DC-K2 jitter abs(d e) | jitter < 0 | supported | refuted |",
         "|" + "---|" * 11]
    for s in SEEDS:
        n, c = doc["seeds"][s], doc["per_seed"][s]
        a, b, f, j = n["dc_vs_k0"][CROSS], n["dc_vs_k2"][CROSS], n["dc_vs_k2"]["LAT_cross_abs_final_m"], n["jitter_dlag_dc_vs_k2"]
        L.append(f"| {s} | {a['delta']:+.5f} [{a['lo']:+.5f}, {a['hi']:+.5f}] | {'no' if c['i_dc_k0_cross_not_separated_worse'] else 'YES'} | {b['delta']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}] "
                 f"| {'yes' if c['ii_dc_k2_cross_le_minus_2p5mm'] else 'no'} | {f['delta']:+.5f} [{f['lo']:+.5f}, {f['hi']:+.5f}] | {j['delta']:+.5f} [{j['lo']:+.5f}, {j['hi']:+.5f}] "
                 f"| {'yes' if c['iii_jitter_delta_negative'] else 'no'} | {'yes' if c['supported'] else 'no'} | {'yes' if c['refuted'] else 'no'} |")
    L += ["", "## All four families (K2DC - K2 | K2DC - K0), per seed, never pooled", ""]
    for s in SEEDS:
        L += [f"### seed {s}", "", "| family | row | K2DC - K2 | sep | K2DC - K0 | sep |", "|---|---|---|---|---|---|"]
        for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
            for rid, r in doc["seeds"][s]["dc_vs_k2_by_family"][fam].items():
                r0 = doc["seeds"][s]["dc_vs_k0_by_family"][fam].get(rid, {})
                L.append(f"| {fam} | {rid} | {r['delta']:+.5f} | {'*' if r['separated'] else ''} | {r0.get('delta', float('nan')):+.5f} | {'*' if r0.get('separated') else ''} |")
        e2 = (doc["seeds"][s].get("dc_vs_k2_by_family") or {}).get("STRATEGIC") or {}
        e0 = (doc["seeds"][s].get("dc_vs_k0_by_family") or {}).get("STRATEGIC") or {}
        if e2.get("status") == "REPORT-ONLY" or e0.get("status") == "REPORT-ONLY":
            for rid in sorted(set(e2.get("rows", {})) | set(e0.get("rows", {}))):
                a, b = e2.get("rows", {}).get(rid, {}), e0.get("rows", {}).get(rid, {})
                tag = lambda x: ("*" if x.get("separated") else "") + (" WORSE" if x.get("separated_worse") else "")        # noqa: E731
                L.append(f"| STRATEGIC (report-only) | {rid} | {a.get('delta', float('nan')):+.5f} | {tag(a)} | {b.get('delta', float('nan')):+.5f} | {tag(b)} |")
            L += ["", f"STRATEGIC rows are REPORT-ONLY: shown beside the decision, never in it ({STRATEGIC_WORSE} when marked WORSE).", ""]
        else:
            L += ["| STRATEGIC | - | NOT-SUPPLIED | | NOT-SUPPLIED | |", "", f"STRATEGIC: {STRATEGIC_NOT_SUPPLIED}", ""]
    b = doc.get("beside_no_direction")
    if b:
        L += ["## Beside (no committed direction)", "", "```", json.dumps(b, indent=1, default=float), "```", ""]
    L += ["## Resolutions of PREREG wording", ""] + [f"* {r}" for r in doc["resolutions"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k2dc-pf", required=True, help="E2_READ_K2DC/E2_K2_pf (holds dump_s0, dump_s1)")
    ap.add_argument("--k2-pf", required=True, help="E2_READ_K2M1N20/E2_K2_pf")
    ap.add_argument("--k0-pf", required=True, help="E2_READ_K2M1N20/E2_K0_pf (the K0 the K2DC read reuses)")
    ap.add_argument("--dev-split", required=True)
    ap.add_argument("--metrics", help="runs/r9e2-K2DC/metrics.jsonl (the in-training share check); without it the outcome reads UNCHECKED-VOID")
    ap.add_argument("--k2dc-read", help="E2_READ_K2DC/read (EVAL_K2.json, s6c/) for the no-direction map / box / S6c rows")
    ap.add_argument("--k2-read", help="E2_READ_K2M1N20/read")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--arm", default=ARM_DEFAULT, help=f"{ARM_DEFAULT} (default: unchanged behaviour) or {', '.join(sorted(ARM_ARGV))}: the arm's `--k2dc-*` inputs are that arm's; "
                    f"its share target and bounds are read from stack/ops/runs.d/<arm>.argv.json (or --argv-json); or {', '.join(sorted(ARM_SG))}: the stop-gradient arm "
                    "(its lever is checked in its argv json, validity is read from --metrics, which is then required)")
    ap.add_argument("--argv-json", help="a derived arm's argv json (default: stack/ops/runs.d/r9e2_<arm>.argv.json next to this script's stack/)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        sg = a.arm in ARM_SG
        arm = arm_sg(a.arm, a.argv_json) if sg else arm_dose(a.arm, a.argv_json)
        numbers = {s: seed_numbers(a.k2dc_pf, a.k2_pf, a.k0_pf, a.dev_split, s, a.boot, a.arm if sg else "K2DC") for s in SEEDS}
        void = void_check_sg(a.metrics) if sg else void_check(a.metrics) if arm is None else void_check(a.metrics, tuple(arm["share_band"]), tuple(arm["bounds"]), ARM_MEAN_BAND.get(a.arm))
        doc = build(numbers, void, beside(a.k2dc_read, a.k2_read)) if arm is None else build(numbers, void, beside(a.k2dc_read, a.k2_read), arm)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 4
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DOSE_VERDICT.json").write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "DOSE_VERDICT.md").write_text(md(doc), encoding="utf-8")
    print(f"[r9-e2-dose-verdict] {doc['outcome']} -> {out / 'DOSE_VERDICT.json'}")
    base = doc["outcome"].split(" ")[0]
    return EXIT.get(base, 4)


if __name__ == "__main__":
    raise SystemExit(main())
