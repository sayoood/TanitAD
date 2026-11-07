#!/usr/bin/env python3
"""r9_e1_verdict.py -- the refcv9 E1 PER-ARM VERDICT (REFCV9_DESIGN 4.3): one lever arm's dump -> paired eval against ITS reference
arm -> the pre-registered keep rule against the C0/C1 replicate FLOOR.  Tier T0 (label level), perception only.

    python r9_e1_verdict.py --floor FLOOR.json --paired EVAL_<ARM>_vs_<REF>.json --eval-arm EVAL_<ARM>.json --eval-ref EVAL_<REF>.json \\
                            --arm <ARM> [--ref <REF>] [--rast-arm <dir> --rast-ref <dir>] [--lift-valid <mask>] [--floor-eval-c0 EVAL_C0.json] --out-dir <dir>
                                                                   # -> VERDICT_<ARM>.json + VERDICT_<ARM>.md   (rc 0 OK, rc 2 REFUSED)
    python r9_e1_verdict.py mapping [--json]                       # the arm -> reference -> primary -> emitted-key table (from this file)
    python r9_e1_verdict.py ref --arm <ARM>                        # the arm's default reference (rc 3 when the design names none)
    python r9_e1_verdict.py state --arm <ARM> [--ref <REF>] --status CHAIN_STATUS.txt --root R --read-dir RD --out state.env
    python r9_e1_verdict.py dumpcheck --dump A/dump.json --dump B/dump.json [...]

NO SECOND KEEP RULE, NO SECOND ESTIMATOR.  Every judgement is ``r9_e1_floor.lever_read`` -> ``taniteval.ci.e1_floor_read``; the
paired intervals are ``r9_e1_floor.box_rows`` / ``r9_e1_floor.nearfield_rows`` (-> ``taniteval.ci.paired_stat_cluster_bootstrap``).  This
file adds only (1) the arm -> reference -> primary MAPPING, (2) the "no other perception primary separated worse beyond 2 x its floor"
CLAUSE swept over the table's perception primaries but the arm's own primary keys AND own proxies (other floor rows reported, never a veto; a scope key the floor
has NO row for is COVERED by the primary's named proxies or makes clause 3 incomplete -> never KEEP), (3) the overall KEEP / DROP / NOT-COMPUTABLE with the
failing clause named, and (4) the refusals (incl. a --lift-valid mask whose sha256 differs from the floor's: the `nf_swedge_prec_lv_*` rows would be read on
different cells).  The M1 MAP PRIMARY is `nf_swedge_prec_lv_5_20` (hold_r9_e1_mapprim) and exists only on a lift-valid mask.
BESIDE, NEVER DECIDING (hold_r9_pedgefit2): for M1 and M2a the edge-class P / R / F @ 0.3 m rows `nf_edge03_*` (PREREG_M1_PEDGEFIT; emitted only on a THRESHOLDED
raster read) are read from this verdict's own paired bootstrap into ``beside_pedgefit`` / the md section 'Beside (PREREG_M1_PEDGEFIT rows 2/3 selector)': not in
``reads``, clauses 1-3, ``table_keys`` or the clause-3 sweep (a floor without them is not 'uncovered'); NOT-COMPUTABLE with the reason on a read without them.

THE KEEP RULE (REFCV9_DESIGN 4.3, verbatim in ``KEEP_RULE``): keep a lever iff its primary is separated in the good direction (clause 1)
AND |delta| > 2 x |C1 - C0| on that metric (clause 2) AND no other perception primary is separated worse beyond 2 x its floor (clause 3).
KEEP needs all three; DROP names every clause that failed; a primary the evaluator does not emit is NOT-COMPUTABLE (never guessed) --
unless clause 3 already fails, in which case the arm cannot be kept whatever the primary reads, and the verdict is DROP.

SAME-WINDOWS is enforced exactly like ``r9_e1_floor.py`` (its own ``comparability`` is called) and against the FLOOR's recorded read: the arm,
the reference and the paired record must hold the identical window set (paired alignment drops none; equal n_windows / n_clips / window file /
split / P=R gate / B / seed), equal to the floor's; raster name sets equal; or the verdict is REFUSED (rc 2).

WHAT A VERDICT IS AND IS NOT (H-ESTIM-SEED-1).  The paired interval answers *another draw of EPISODES*.  The floor |C1 - C0| is ONE seed pair
(n = 1).  A lever arm here is ONE training run (seed 0): KEEP means "cleared the pre-registered E1 bar", not "proved".
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "r9e1-verdict-v1"
TIER = "T0 (label level)"
KEEP_FACTOR = 2.0
KEEP_RULE = ("Keep a lever iff its primary is separated in the good direction AND |delta| > 2 x |C1 - C0| on that metric AND no "
             "other perception primary is separated worse beyond 2 x its floor (REFCV9_DESIGN 4.3; taniteval.ci.e1_floor_read).")
FLOOR_ARMS = ("C0", "C1")


class Refused(Exception):
    """The inputs do not describe one read of the same windows / a known arm: no verdict is quoted."""


# ======================================================================================================================== lazy floor
_FL = None


def _fl():
    """``r9_e1_floor`` (numpy). Imported lazily so ``state`` / ``ref`` / ``mapping`` / ``dumpcheck`` stay stdlib-only."""
    global _FL
    if _FL is None:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import r9_e1_floor
        _FL = r9_e1_floor
    return _FL


# ======================================================================================================================== the MAPPING
#: key family -> (file, line, needle that must be on that line). ``tests/test_r9_e1_verdict.py`` opens each and checks the needle.
_EVAL = "stack/scripts/r9_e1_eval.py"
KEY_SRC = (
    (r"^n_pos$", _EVAL, 239, '"n_pos"'),
    (r"^ap(1|2)m_all$", _EVAL, 243, '"ap%sm_all"'),
    (r"^ap(1|2)m_(0_20|20_40|40_60)$", _EVAL, 245, '"ap%sm_%s"'),
    (r"^ap(1|2)m_20_60$", _EVAL, 249, '"ap%sm_20_60"'),
    (r"^dx_0_40_mean$", _EVAL, 251, '"dx_0_40_mean"'),
    (r"^dx_0_40_median$", _EVAL, 252, '"dx_0_40_median"'),
    (r"^tp_class_acc$", _EVAL, 254, '"tp_class_acc"'),
    (r"^prec_gate_(noNMS|NMS)$", _EVAL, 258, '"prec_gate_"'),
    (r"^rec_gate_(noNMS|NMS)$", _EVAL, 259, '"rec_gate_"'),
    (r"^boxes_per_object_(noNMS|NMS)$", _EVAL, 260, '"boxes_per_object_"'),
    (r"^frac_objects_ge2_(noNMS|NMS)$", _EVAL, 261, '"frac_objects_ge2_"'),
    (r"^boxes_per_object_noNMS_(0_20|20_40|40_60)$", _EVAL, 265, '"boxes_per_object_noNMS_"'),
    (r"^n_det_noNMS_(0_20|20_40|40_60)$", _EVAL, 266, '"n_det_noNMS_"'),
    (r"^ap2m_all_NMS$", _EVAL, 268, '"ap2m_all_NMS"'),
    (r"^nf_prec_(0_5|5_10|10_20|20_40)$", _EVAL, 588, '"nf_prec_%d_%d"'),
    (r"^nf_prec05_(0_5|5_10|10_20|20_40)$", _EVAL, 589, '"nf_prec05_%d_%d"'),
    (r"^nf_prec_0_10$", _EVAL, 594, '"nf_prec_0_10"'),
    # the M1 map primary (hold_r9_e1_mapprim): per band, then the 5-10 + 10-20 m pooled key; each exists ONLY when a --lift-valid mask was given
    (r"^nf_swedge_prec_(lv_)?(0_5|5_10|10_20|20_40)$", _EVAL, 605, '"nf_swedge_prec_%s%d_%d"'),
    (r"^nf_swedge_prec_(lv_)?5_20$", _EVAL, 606, '"nf_swedge_prec_%s5_20"'),
    # PREREG_M1_PEDGEFIT's BESIDE rows (hold_r9_pedgefit): edge-class P / R / F @ 0.3 m; one line emits all three kinds, per band and pooled 5-20 m,
    # and ONLY on a thresholded read (nearfield_stats adds the columns when the evaluator was given --map-thresholds)
    (r"^nf_edge03_prec_(0_5|5_10|10_20|20_40|5_20)$", _EVAL, 697, '"nf_edge03_prec_"'),
    (r"^nf_edge03_rec_(0_5|5_10|10_20|20_40|5_20)$", _EVAL, 697, '"nf_edge03_rec_"'),
    (r"^nf_edge03_f_(0_5|5_10|10_20|20_40|5_20)$", _EVAL, 697, '"nf_edge03_f_"'),
    # M1's two design BESIDE rows (hold_r9_d3beside; E0d MAP-3a / MAP-3b): lane tolerance-IoU and drivable-boundary recall @0.3 m, emitted ONLY on a
    # read with --map-beside-d3 (r9_e1_eval._d3_stats_keys). The pooled 0-20 m lines come FIRST (key_src returns the first matching pattern).
    (r"^lane_tiou05_0_20$", _EVAL, 875, '"lane_tiou05_0_20"'),
    (r"^lane_tiou05_(0_5|5_10|10_20|20_40)$", _EVAL, 872, '"lane_tiou05_'),
    (r"^drv_bnd_rec03_cells_0_20$", _EVAL, 881, '"drv_bnd_rec03_cells_0_20"'),
    (r"^drv_bnd_rec03_cells_(0_5|5_10|10_20|20_40)$", _EVAL, 879, '"drv_bnd_rec03_cells_'),
    (r"^drv_bnd_rec03_0_20$", _EVAL, 880, '"drv_bnd_rec03_0_20"'),
    (r"^drv_bnd_rec03_(0_5|5_10|10_20|20_40)$", _EVAL, 878, '"drv_bnd_rec03_'),
)


def key_src(key: str):
    """``(file, line, needle)`` of the line of r9_e1_eval.py that emits ``key``, or None when no line of it does."""
    for pat, f, ln, needle in KEY_SRC:
        if re.match(pat, key):
            return f, ln, needle
    return None


def _spec(mid, text, role, keys=(), better="higher", proxies=(), reason="", would_compute="", clause3=True):
    """One metric of an arm's sec. 4.3 row.  ``keys`` are the emitted keys that REALISE it (empty = NOT-COMPUTABLE); ``role`` is
    ``primary`` (clauses 1 and 2 decide on it), ``guard`` (a named non-regression: it must be evaluable, and is covered by clause 3) or
    ``beside`` (listed in the design 'beside', reported, never decides).  ``clause3`` False = a beside row OUTSIDE the keep rule altogether (PREREG_M1_PEDGEFIT's
    edge-class rows): its keys are not in ``table_keys()`` / clause 3's scope, the sweep skips them, and ``judge`` reads them into ``beside_pedgefit``, not ``reads``."""
    return {"id": mid, "text": text, "role": role, "keys": tuple(keys), "better": better, "proxies": tuple(proxies),
            "status": "COMPUTABLE" if keys else "NOT-COMPUTABLE", "reason": reason, "would_compute": would_compute, "clause3": bool(clause3)}


#: The M1 MAP PRIMARY (REFCV9_DESIGN 4.3: "corridor sidewalk+edge precision 5-20 m (lift-valid)") as r9_e1_eval emits it since hold_r9_e1_mapprim:
#: flag (psw >= 128) | (pedge >= 128) against GT {5 edge, 7 sidewalk}, GT-seen AND lift-valid cells, the 5-10 and 10-20 m bands pooled by COUNTS.
#: The key exists ONLY when a --lift-valid mask was given (floor read, evals and this verdict on the SAME mask).  ``proxies`` = the historical
#: SIDEWALK-ONLY, lift-UNMASKED per-band precisions: a different quantity, never the primary, but they stay in clause 3's scope so a lever of
#: ANOTHER family is still vetoed on the map family when no mask exists -- an arm's OWN primary excludes its own proxies (clause3_vetoes).
MAP_PRIMARY_KEY = "nf_swedge_prec_lv_5_20"
MAP_PROXIES = ("nf_prec_5_10", "nf_prec_10_20")
LV_FLOOR_HINT = (" -- the floor has no lift-valid map rows: it was read without --lift-valid (or by an evaluator that predates the map primary); "
                 "re-read the floor with the current eval and --lift-valid <mask>")


def _map_primary(mid="map_primary", role="primary"):
    return _spec(mid, "corridor sidewalk+edge precision 5-20 m (lift-valid)", role, (MAP_PRIMARY_KEY,), "higher", MAP_PROXIES)


#: PREREG_M1_PEDGEFIT's BESIDE rows (E0d thin-class protocol, r9_e1_eval.edge_tol_keys): edge-class P / R / F @ 0.3 m of {pedge >= the arm's own FIT level}
#: against GT class 5, per read band and pooled 5-20 m.  They exist ONLY on a thresholded raster read.  ``clause3=False``: NEVER in clauses 1-3 -- not a primary,
#: not a guard, not a veto, and a floor that lacks them is not ``missing_uncovered`` -- and they are read from the verdict's OWN paired bootstrap (``_read_beside``),
#: not from the floor.  Only the pooled F row (``EDGE03_SELECTOR_KEY``) selects between rows 2 and 3 of the PREREG's committed-readings table; nothing else.
EDGE03_BANDS = ("5_20", "0_5", "5_10", "10_20", "20_40")
EDGE03_KINDS = (("f", "F"), ("prec", "P"), ("rec", "R"))
EDGE03_SELECTOR_KEY = "nf_edge03_f_5_20"
BESIDE_TITLE = "Beside (PREREG_M1_PEDGEFIT rows 2/3 selector)"
BESIDE_NOTE = ("Edge-class precision / recall / F @ 0.3 m (E0d thin-class protocol: GT edge class 5 dilated by 3 cells; flag = pedge >= the arm's OWN FIT level; "
               "P over judged cells) per read band and pooled 5-20 m, paired (arm - reference) on this verdict's own episode-cluster bootstrap (the primary's B and "
               "seed). NEVER decisive: not in clauses 1-3 (not a primary, not a guard, not a veto; a floor without these rows is not 'uncovered'), and the KEEP / "
               "DROP above does not read them.")


#: M1's two design BESIDE rows (REFCV9_DESIGN 4.3 M1 row: "beside: lane tolerance-IoU 0-20 m, boundary recall @0.3 m"), from hold_r9_d3beside (E0d MAP-3a
#: L-DIL lane IoU of {code == 2} vs GT lane, both 0.5 m wide; MAP-3b recall of GT edge cells within 0.3 m of the inner boundary of {pdrv >= level}, pooled
#: window mean). ``clause3=False`` like the edge rows: read from this verdict's OWN paired bootstrap into the beside block, NEVER a clause, a guard or a veto.
#: The rows exist ONLY on a read with ``--map-beside-d3`` (the raster counts are taken when asked; absent -> NOT-COMPUTABLE with the reason).
D3_LANE_KEY = "lane_tiou05_0_20"
D3_BOUNDARY_KEY = "drv_bnd_rec03_0_20"
D3_KEY_PREFIXES = ("lane_tiou05_", "drv_bnd_rec03_")


def _edge03_specs(mid):
    return tuple(_spec("%s.beside.edge03_%s_%s" % (mid, kind, band), "edge-class %s@0.3 m %s m (beside)" % (lab, band.replace("_", "-")), "beside",
                       ("nf_edge03_%s_%s" % (kind, band),), "higher", clause3=False)
                 for band in EDGE03_BANDS for kind, lab in EDGE03_KINDS)


_AP2_ALL = ("ap2m_all", "AP@2 m, head output, no NMS (all ranges)")


def _m3_metrics(mid):
    """E1-M3N20 (PREREG_M3): M3 / M3N20 read M1's map primary, the design's box guard (`ap2m_all`, clause 3 beside `boxes_per_object_noNMS`), and M1 / M1N20's
    beside rows (the two D3 rows and the edge F@0.3 m rows, ``clause3=False``): the same rows for the decisive read (vs BASE) and the CARRY read (vs C0)."""
    return (_map_primary(mid + ".primary"), _spec(mid + ".guard.ap2", "box AP@2 m not worse (the ladder's cost)", "guard", ("ap2m_all",)),
            _spec(mid + ".beside.lane", "lane tolerance-IoU 0-20 m (beside)", "beside", (D3_LANE_KEY,), "higher", clause3=False),
            _spec(mid + ".beside.edge", "boundary recall @0.3 m (beside)", "beside", (D3_BOUNDARY_KEY,), "higher", clause3=False),
            *_edge03_specs(mid))


_M3_READINGS = ("PREREG_M3 committed readings (DECISIVE = vs BASE, CARRY = vs C0): KEEP/KEEP -> M3 is K2's map config -> FLIP -> K2; KEEP/DROP -> own encoder logged as a kept "
                "lever, K2 map config = C0's head, the Cfp fallback (8-way head + --w-map-fp, needs its 50-step smoke) goes to the PI; DROP clause 1-2 -> map capacity is not the "
                "limit on this rig at 4,000 steps, K2 map config = BASE iff BASE's own read was KEEP else C0's head, Cfp to the PI; DROP clause 3 -> the widened SHARED lift "
                "(64 -> 128) costs a box / other primary beyond 2 x its floor, named; next arm: the own encoder on the 64-channel lift = new code + its own prereg")

#: The 17 arms of sec. 4.3 (+ T1s, the shuffled control of D3; + T1h / T1hs, T1's follow-up lever and its control, REFCV9_E1_T1H_PREREG.md).  ``ref``: the default reference arm (None = the design names none);
#: ``allowed``: the references this verdict accepts; ``kind``: lever (KEEP/DROP) or control (a control outcome).
ARMS = {
    "C0": {"kind": "floor", "variable": "refcv8 recipe on the rig, seed 0 (the floor)", "ref": None, "allowed": (), "metrics": (),
           "next_lever": "n/a: C0 / C1 are the floor (r9_e1_floor.py), not a lever", "argv_flags": (), "ambiguities": ()},
    "C1": {"kind": "floor", "variable": "C0's flags, training seed 1 (the floor)", "ref": None, "allowed": (), "metrics": (),
           "next_lever": "n/a: C0 / C1 are the floor (r9_e1_floor.py), not a lever", "argv_flags": (), "ambiguities": ()},
    "M1": {"kind": "lever", "variable": "sec. 2.1 representation (banded map, near lift 20 -> 10 m) + R7 losses at w = 1", "ref": "C0", "allowed": ("C0",),
           "metrics": (_map_primary("M1.primary"),
                       _spec("M1.beside.lane", "lane tolerance-IoU 0-20 m (beside)", "beside", (D3_LANE_KEY,), "higher", clause3=False),
                       _spec("M1.beside.edge", "boundary recall @0.3 m (beside)", "beside", (D3_BOUNDARY_KEY,), "higher", clause3=False),
                       *_edge03_specs("M1")),
           "next_lever": "the `--w-map-fp` fallback E1 arm ('kept only as a fallback E1 arm if M1 misses at 5-20 m', sec. 2.1 line 111); Q1 branch (a): M2 "
                         "gradient share second (sec. 3 line 263); branch (b2): (1) M2, (2) F1, (3) B1, (4) T1 (sec. 3 line 265)",
           "argv_flags": ("--r9-map-banded",),
           "ambiguities": ("the primary `nf_swedge_prec_lv_5_20` exists only when the FLOOR read, both evals and this verdict ran with the SAME arm-independent "
                           "--lift-valid mask (sha256 cross-checked; a floor without the row reads NO-FLOOR -> NOT-COMPUTABLE, never a guess)",
                           "the design's inherited 'sidewalk+edge' 0.251 / 0.733 / 0.716 / 0.744 are SIDEWALK-ONLY, GT-seen, lift-UNMASKED numbers "
                           "(hold_r9_e1_mapprim HANDOFF): a different quantity from this primary, so they are no baseline for it",
                           "UNVERIFIED: exact-cell precision against a 1-cell GT edge penalises a banded arm's dilated (~0.5 m) `pedge` heatmap; flags use "
                           "probability planes >= 128, not the FIT-split per-channel thresholds",)},
    #: E1-NEAR20 (PREREG_M1_NEAR20, MM 2026-11-06 22:30:03Z): M1's argv with the banded head's near band 0-20 m @ 0.10 m and C0's 20 m near lift
    #: (`--r9-map-banded-near-m 20 --map-hires-near-lift-m 20`). DECISIVE: M1N20 vs C0 under M1's keep rule and primary, UNCHANGED. SECONDARY (never decisive):
    #: M1N20 vs M1 -- the attribution of the near-band end -- is the same arm read with `--ref M1`; its beside edge F@0.3 m 10-20 m row is the committed selector.
    "M1N20": {"kind": "lever", "variable": "M1 with the banded head's near band 0-20 m @ 0.10 m (C0's 20 m near lift; mid 20-40 m) -- the near-band end 10 -> 20 m, nothing else",
              "ref": "C0", "allowed": ("C0", "M1"),
              "metrics": (_map_primary("M1N20.primary"),
                          _spec("M1N20.beside.lane", "lane tolerance-IoU 0-20 m (beside)", "beside", (D3_LANE_KEY,), "higher", clause3=False),
                          _spec("M1N20.beside.edge", "boundary recall @0.3 m (beside)", "beside", (D3_BOUNDARY_KEY,), "higher", clause3=False),
                          *_edge03_specs("M1N20")),
              "next_lever": "PREREG_M1_NEAR20 committed readings: KEEP -> M1N20 is the map config for K2, M3 builds on it; DROP clause 1-2 with the beside edge F@0.3 m "
                            "10-20 m separated BETTER vs M1 -> M3 built on M1N20 (next pre-registration); NOT separated better -> M3 (map capacity) or the Cfp fallback (PI)",
              "argv_flags": ("--r9-map-banded", "--r9-map-banded-near-m"),
              "ambiguities": ("same primary / keep rule as M1 (PREREG_M1_NEAR20: UNCHANGED); the fitter's band layout stays 0-10 / 10-40 / 40-100 m for every arm, so M1N20's "
                              "10-20 m is cut with its 10-40 m level even though its head serves 10-20 m at 10 cm (like-for-like with M1 and C0, disclosed in the PREREG)",
                              "the attribution read M1N20 vs M1 is SECONDARY: C0 differs from M1N20 in the head, not only the lift; separated is necessary, not sufficient "
                              "(one training seed per arm, H-ESTIM-SEED-1)",)},
    "M2a": {"kind": "lever", "variable": "M1 + --w-map-hires 0.511 (the 0.05 layer2-share dose; argv lever block)", "ref": "M1", "allowed": ("M1",),
            "metrics": (_map_primary("M2a.primary"), _spec("M2a.guard.ap2", "box AP@2 m not worse (the ladder's cost)", "guard", ("ap2m_all",)),
                        *_edge03_specs("M2a")),
            "next_lever": "read M2b (the other dose; sec. 2.2 line 143: 'the weight falls back to the E1 dose arms'); if both doses fail M1 stands and "
                          "a dynamic (GradNorm-style) weight is 'held as a fallback arm only' (sec. 2.2 lines 148-149)",
            "argv_flags": ("--r9-map-banded",),
            "ambiguities": ("'box AP@2 m not worse' read as clause 3 on ap2m_all vs M1 (separated worse beyond 2 x floor); the stricter reading "
                            "'not separated worse at all' is reported in `named_guards`, not applied",)},
    "M2b": {"kind": "lever", "variable": "M1 + --w-map-hires 1.018 (the 0.15 layer2-share dose; argv lever block)", "ref": "M1", "allowed": ("M1",),
            "metrics": (_map_primary("M2b.primary"), _spec("M2b.guard.ap2", "box AP@2 m not worse (the ladder's cost)", "guard", ("ap2m_all",))),
            "next_lever": "read M2a (the other dose; sec. 2.2 line 143); if both doses fail M1 stands, dynamic weight held as a fallback arm only "
                          "(sec. 2.2 lines 148-149)",
            "argv_flags": ("--r9-map-banded",),
            "ambiguities": ("UNVERIFIED: M2b's --w-map-hires (1.018) is within 2 % of M1's (1.0) -- if the share-per-weight geometry is the "
                            "same, M2b vs M1 reads a replicate-sized difference; M2a (0.511) is the only real dose contrast",)},
    #: E1-M3N20 (PREREG_M3, MM 2026-11-07 01:31:44Z, md5 8e497d34): M3 = BASE + the map's OWN 128-channel BEV encoder (`--r9-map-own-enc`), `--w-map-hires 1.0`
    #: (M1's: no M2 winner exists). BASE is chosen by M1N20's COMMITTED reading: M3 on M1 (near 10), M3N20 on M1N20 (near 20, `--r9-map-banded-near-m 20`).
    #: DECISIVE: M3 vs M1 / M3N20 vs M1N20 (the design row: "M1's primary vs the M2 winner; box AP@2 m"). CARRY (decides whether the banded config enters K2): vs C0, the
    #: SAME rule, reported for every outcome and never re-deciding the lever -- read with `--ref C0`.
    "M3": {"kind": "lever", "variable": "M1 + the map's own 128-ch BEV encoder (lift 64 -> 128), --w-map-hires 1.0 (PREREG_M3: M1's weight; no M2 winner exists)",
           "ref": "M1", "allowed": ("M1", "C0"), "metrics": _m3_metrics("M3"), "next_lever": _M3_READINGS,
           "argv_flags": ("--r9-map-banded", "--r9-map-own-enc"),
           "ambiguities": ("DECISIVE = M3 vs M1 (its BASE when M1N20 reads DROP clause 3 or DROP 1-2 with the selector not separated better); CARRY = M3 vs C0 (--ref C0), "
                           "same rule, never re-decides the lever; the PREREG's ONE lever vs BASE is the own encoder",
                           "the fitter's band layout stays 0-10 / 10-40 / 40-100 m for every arm (like-for-like with M1, M1N20 and C0, disclosed in the PREREG)",
                           "one training seed per arm; the floor is ONE seed pair; separated is necessary, not sufficient (H-ESTIM-SEED-1)")},
    "M3N20": {"kind": "lever", "variable": "M1N20 + the map's own 128-ch BEV encoder (lift 64 -> 128), --w-map-hires 1.0; near band 0-20 m @ 0.10 m (PREREG_M3: M1's weight)",
              "ref": "M1N20", "allowed": ("M1N20", "C0"), "metrics": _m3_metrics("M3N20"), "next_lever": _M3_READINGS,
              "argv_flags": ("--r9-map-banded", "--r9-map-own-enc", "--r9-map-banded-near-m"),
              "ambiguities": ("DECISIVE = M3N20 vs M1N20 (its BASE when M1N20 reads KEEP or DROP 1-2 with the selector separated better); CARRY = M3N20 vs C0 (--ref C0), "
                              "same rule, never re-decides the lever; the PREREG's ONE lever vs BASE is the own encoder",
                              "the fitter's band layout stays 0-10 / 10-40 / 40-100 m for every arm, so M3N20's 10-20 m is cut with its 10-40 m level even though its head serves "
                              "10-20 m at 10 cm (like-for-like with M1N20, M1 and C0, disclosed in the PREREG)",
                              "one training seed per arm; the floor is ONE seed pair; separated is necessary, not sufficient (H-ESTIM-SEED-1)")},
    "B1": {"kind": "lever", "variable": "box-derived depth aux (--r9-box-depth-aux, --w-box-depth 0.1)", "ref": "C0", "allowed": ("C0",),
           "metrics": (_spec("B1.primary", "abs(dx) on matched TPs 0-40 m", "primary", ("dx_0_40_mean",), "lower"),
                       _spec("B1.beside.dxmed", "abs(dx) median (robust companion, same quantity)", "beside", ("dx_0_40_median",), "lower"),
                       _spec("B1.guard.ap1", "AP@1 m", "guard", ("ap1m_all",))),
           "next_lever": "sec. 3 branch (b2) order: (4) T1 after (3) B1 (line 265); the other box rows (B2, B3) are independent arms (sec. 4.3)",
           "argv_flags": ("--r9-box-depth-aux",),
           "ambiguities": ("'abs(dx) on matched TPs 0-40 m; AP@1 m': the first is the primary, AP@1 m a guard; the conjunction reading is reported in "
                           "`alternative_reading`", "abs(dx) mean vs median: dx_0_40_mean is the primary, the median is shown beside")},
    "B2": {"kind": "lever", "variable": "perspective 3D aux head (--r9-box-persp-aux, --w-box-persp 0.1), aux only", "ref": "C0", "allowed": ("C0",),
           "metrics": (_spec("B2.primary", "AP@2 m 20-60 m", "primary", ("ap2m_20_60",), "higher"),),
           "next_lever": "B2c and B4 are not run ('B2c iff B2 passes'; B4 = 'B2 winner + ...', sec. 4.3); the other box levers are B3 and B1 (sec. 4.3 rows)",
           "argv_flags": ("--r9-box-persp-aux",), "ambiguities": ()},
    "B2c": {"kind": "control", "variable": "B2's control: a second BEV head (--r9-box-bev2-aux, --w-box-bev2 0.1) instead of the perspective head",
            "ref": "C0", "allowed": ("C0", "B2"),
            "metrics": (_spec("B2c.primary", "B2's primary: AP@2 m 20-60 m", "primary", ("ap2m_20_60",), "higher"),),
            "next_lever": "n/a: a control; if B2c gains as much as B2 the gain was capacity, not view (sec. 4.3)",
            "argv_flags": ("--r9-box-bev2-aux",),
            "ambiguities": ("sec. 4.3 gives no numeric 'as much': read with --ref B2, B2c - B2 is 'less' iff separated worse beyond 2 x floor "
                            "(the keep rule's own words) -- an OPERATIONALISATION the MM must confirm",
                            "run only if B2 passes ('iff B2 passes'): the shell does not check this, the MM does")},
    "B3": {"kind": "lever", "variable": "quality-aware presence + DN + class-sqrt (--r9-box-quality, --r9-box-dn, --r9-box-class-sqrt)",
           "ref": "C0", "allowed": ("C0",),
           "metrics": (_spec("B3.primary", "boxes / detected object at P=R, no NMS", "primary", ("boxes_per_object_noNMS",), "lower"),
                       _spec("B3.guard.ap2", "AP@2 m no NMS", "guard", ("ap2m_all",))),
           "next_lever": "B3' = B3 + the in-loss BEV-NMS (sec. 4.3 B3 row, sec. 5.4 line 422), and 'more DN groups' (sec. 5.4 line 422)",
           "argv_flags": ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt"),
           "ambiguities": ("'at P=R': the evaluator reads the census at ONE fixed gate for both arms (the refcv8 TRAIN-fit gate), not at each arm's "
                           "own P=R point; B3's quality-aware presence changes the score scale, so the fixed gate is not neutral to it -- a "
                           "per-arm P=R read is a different operating point (UNVERIFIED which the design intends)",
                           "sec. 2.4 line 179: 'if B3's DN + quality alone reach <= 1.25 boxes/object, the in-loss NMS is dropped' -- shown in the .md as "
                           "information, NOT part of the keep rule")},
    "B3p": {"kind": "lever", "variable": "B3 + the in-loss BEV-NMS (--r9-box-inloss-nms)", "ref": "C0", "allowed": ("C0", "B3"),
            "metrics": (_spec("B3p.primary", "boxes / detected object at P=R, no NMS", "primary", ("boxes_per_object_noNMS",), "lower"),
                        _spec("B3p.guard.ap2", "AP@2 m no NMS", "guard", ("ap2m_all",))),
            "next_lever": "'more DN groups' (sec. 5.4 line 422)",
            "argv_flags": ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms"),
            "ambiguities": ("reference C0 = the keep rule; the question 'does the in-loss NMS add anything over B3' (sec. 2.4 line 179) is the --ref B3 read",
                            "same fixed-gate caveat as B3")},
    "B4": {"kind": "lever", "variable": "B2 winner + per-frame queries + 6 layers + s8 horizon tokens (needs D2)", "ref": "B2", "allowed": ("B2",),
           "metrics": (_spec("B4.primary", "AP@2 m no NMS, by range (primary: all ranges)", "primary", ("ap2m_all",)),
                       _spec("B4.guard.near", "AP@2 m 0-20 m", "guard", ("ap2m_0_20",)),
                       _spec("B4.guard.mid", "AP@2 m 20-40 m", "guard", ("ap2m_20_40",)),
                       _spec("B4.guard.far", "AP@2 m 40-60 m", "guard", ("ap2m_40_60",))),
           "next_lever": "none named in sec. 4.3 / sec. 3 / sec. 5.4 for a B4 miss -- UNVERIFIED; the B2 winner's head stays (B4 is defined as 'B2 winner + ...')",
           "argv_flags": (), "ambiguities": ("'by range': the primary is read on ap2m_all, the three range bands are guards; whether the design means "
                                             "a per-range decision is UNVERIFIED",
                                             "the reference is the 'B2 winner' = B2 (B2 must have passed)")},
    "A1": {"kind": "lever", "variable": "sec. 2.5 augmentation (iff E0a)", "ref": "C0", "allowed": ("C0",),
           "metrics": (_spec("A1.primary", "AP@2 m DEV", "primary", ("ap2m_all",)),
                       _map_primary("A1.guard.map", "guard")),
           "next_lever": "none named in sec. 4.3 / sec. 3 / sec. 5.4 for an augmentation miss -- UNVERIFIED; the MM decides",
           "argv_flags": (), "ambiguities": ("'AP@2 m DEV; map primary': the first is the primary, the map primary is a guard (not worse); A1 can "
                                             "therefore not be KEPT while the map primary is NOT-COMPUTABLE",)},
    "F1": {"kind": "lever", "variable": "FPN map tap + re-projected image-plane aux (iff Q1 = b2)", "ref": "C0", "allowed": ("C0",),
           "metrics": (_map_primary("F1.primary"),),
           "next_lever": "sec. 3 branch (b2) order: (3) B1 depth supervision, then (4) T1 (line 265)",
           "argv_flags": (), "ambiguities": ("'map primary' = M1's primary",)},
    "T1": {"kind": "lever", "variable": "2 past frames' BEV (t-0.5 s, t-1.0 s) warped by past ego-motion (--r9-bev-warp), iff D3 = YES", "ref": "C0",
           "allowed": ("C0",),
           "metrics": (_spec("T1.primary", "corridor precision 0-10 m incl. lift-invalid cells", "primary", ("nf_prec_0_10",), "higher"),),
           "next_lever": "the NO branch of D3: 0-5 m is 'no evidence by construction' and MAP-2's 0-5 m clause is read on lift-valid cells (sec. 6 line 434; sec. 5.4 line 419)",
           "argv_flags": ("--r9-bev-warp",), "ambiguities": ("the bev-warp is stamped 'not certified by the overfit records' in the argv lever block",)},
    "T1s": {"kind": "control", "variable": "T1 + --r9-bev-warp-shuffle: the history frames come from a DONOR episode (only the history CONTENT is wrong)",
            "ref": "T1", "allowed": ("T1", "C0"),
            "metrics": (_spec("T1s.primary", "T1's primary: corridor precision 0-10 m incl. lift-invalid cells", "primary", ("nf_prec_0_10",), "higher"),),
            "next_lever": "n/a: a control; 'a BEV-history-shuffled control must read worse' (sec. 6 D3, line 434)",
            "argv_flags": ("--r9-bev-warp", "--r9-bev-warp-shuffle"),
            "ambiguities": ("T1s is not a row of sec. 4.3's table; it is D3's control (sec. 6). Read with --ref T1 it must be separated worse than T1 beyond "
                            "2 x floor (an OPERATIONALISATION of 'must read worse' the MM must confirm)",)},
    "T1h": {"kind": "lever", "variable": "T1 made reachable: the map loss also supervises history-reachable cells (--r9-bev-warp-supervise-hist), the fuse "
                                         "starts as a copy of the warped past (--r9-bev-warp-fuse-init identity) and fills ONLY the cells the current lift cannot "
                                         "see (--r9-bev-warp-fill-only)", "ref": "C0", "allowed": ("C0",),
            "metrics": (_spec("T1h.primary", "corridor precision 0-10 m incl. lift-invalid cells", "primary", ("nf_prec_0_10",), "higher"),),
            "next_lever": "REFCV9_E1_T1H_PREREG.md: an OPEN gate (final |gate| mean > 0.05) => longer lags 1.5 / 2.0 s; a SHUT gate => a gate lr "
                          "multiplier / a non-zero gate init",
            "argv_flags": ("--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only"),
            "ambiguities": ("T1h is not a row of sec. 4.3's table: T1's follow-up lever, pre-registered in REFCV9_E1_T1H_PREREG.md (32eda41) under 4.3's "
                            "keep rule, unrelaxed",
                            "a KEEP here is a HISTORY lever only if T1hs, read with --ref T1h, is AS-REQUIRED (the prereg's clause 4); otherwise it "
                            "is a supervision / init lever")},
    "T1hs": {"kind": "control", "variable": "T1h + --r9-bev-warp-shuffle: the same supervision set and fuse, with DONOR-episode history (only the "
                                           "history CONTENT is wrong)",
             "ref": "T1h", "allowed": ("T1h", "C0"),
             "metrics": (_spec("T1hs.primary", "T1h's primary: corridor precision 0-10 m incl. lift-invalid cells", "primary", ("nf_prec_0_10",),
                               "higher"),),
             "next_lever": "n/a: a control; T1h's gain is a history gain only if T1hs reads worse (REFCV9_E1_T1H_PREREG.md clause 4 + its addendum)",
             "argv_flags": ("--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only",
                            "--r9-bev-warp-shuffle"),
             "ambiguities": ("read with --ref T1h it must be separated worse than T1h beyond 2 x floor: the prereg's addendum (2026-11-05, before any "
                             "T1h data) tightens clause 4's 'separated in T1h's favour' to the T1s convention",)},
    "M1-FROZEN": {"kind": "lever", "variable": "M1 on the MAPFT frozen-feature rig (iff Q1 = a or WITHHELD)", "ref": None, "allowed": (),
                  "metrics": (_map_primary("M1-FROZEN.primary"),),
                  "next_lever": "none named for an M1-FROZEN miss -- UNVERIFIED (sec. 3 row (a) names it as 'the cheapest possible read')",
                  "argv_flags": (),
                  "ambiguities": ("sec. 4.3 names NO reference arm: the MAPFT rig is not the E1 rig, and its own control (MAPFT with the refcv8 head) is "
                                  "not an arm in sec. 4.3 -- reading it against C0 crosses rigs; REFUSED until the MM rules",)},
}
DESIGN_ARMS = tuple(ARMS)


def table_keys() -> set:
    """Every emitted key that some sec. 4.3 arm names as primary / guard / beside (the 'perception primaries' of the table) -- but not the
    ``clause3=False`` beside rows (PREREG_M1_PEDGEFIT's edge-class P / R / F @ 0.3 m): those are outside the keep rule (:func:`beside_only_keys`)."""
    return {k for a in ARMS.values() for m in a["metrics"] if m["clause3"] for k in m["keys"]}


def beside_only_keys() -> set:
    """The emitted keys some arm reports ONLY beside (``clause3=False``): never in clause 3's scope, never swept by it (not even as informative)."""
    return {k for a in ARMS.values() for m in a["metrics"] if not m["clause3"] for k in m["keys"]}


def clause3_scope() -> set:
    """Clause 3's scope: ``table_keys()`` plus the named PROXIES of the table primaries the evaluator does not emit (the map primary's
    ``nf_prec_*`` bands), so a box lever is still vetoed on the map family.  Census rows and per-class cells are outside it (informative)."""
    return table_keys() | {p for a in ARMS.values() for m in a["metrics"] for p in m["proxies"]}


def primary_proxies() -> dict:
    """``{emitted primary key: its named proxies}`` for the sec. 4.3 primaries that HAVE proxies (today the map primary only). A missing
    floor row of such a key is covered when every proxy has a usable floor row (``clause3_vetoes``)."""
    return {k: tuple(m["proxies"]) for a in ARMS.values() for m in a["metrics"] if m["role"] == "primary" and m["proxies"] for k in m["keys"]}


def own_primary_keys(arm: str) -> set:
    """What an arm's own primary EXCLUDES from clause 3: its primary's emitted keys AND its named proxies (a map arm must not be vetoed by the
    sidewalk-only stand-in of its OWN primary).  The arm's guards and every other key stay in the sweep."""
    return {x for m in ARMS[arm]["metrics"] if m["role"] == "primary" for x in (*m["keys"], *m["proxies"])}


def default_ref(arm: str):
    return ARMS[arm]["ref"]


# ======================================================================================================================== helpers
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _num(x):
    return float(x) if _finite(x) else None


def _load_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused("%s: cannot read %s (%s)" % (what, path, e))


def _check(checks: list, name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


def _relabel(c: dict) -> dict:
    """r9_e1_floor.comparability names its arms C0 / C1; here they are the REFERENCE and the ARM."""
    for k in ("check",):
        c[k] = c[k].replace("C0 / C1 / paired", "REF / ARM / paired").replace("C1 dump", "ARM dump").replace("C0 and C1", "REF and ARM")
    return c


# ======================================================================================================================== comparability
def floor_comparability(floor: dict, e_ref: dict, e_arm: dict, ep: dict, boot: int, seed: int, floor_eval_c0: dict | None,
                        ref: str) -> list:
    """r9_e1_floor's own checks (ref plays C0, arm plays C1) + equality with the FLOOR's recorded read."""
    fl = _fl()
    cs = [_relabel(c) for c in fl.comparability(e_ref, e_arm, ep, boot, seed)]
    _check(cs, "floor status OK", floor.get("status") == "OK" and bool(floor.get("rows")), "FLOOR.json status %r, %d rows"
           % (floor.get("status"), len(floor.get("rows") or {})))
    fe = floor.get("estimator") or {}
    _check(cs, "B / seed equal the floor's", (fe.get("B"), fe.get("seed")) == (boot, seed), "floor B %s seed %s vs %s / %s"
           % (fe.get("B"), fe.get("seed"), boot, seed))
    for key in ("n_windows", "n_clips"):
        _check(cs, "%s equal the floor's" % key, e_arm.get(key) == e_ref.get(key) == floor.get(key), "arm %s / ref %s / floor %s"
               % (e_arm.get(key), e_ref.get(key), floor.get(key)))
    _check(cs, "P=R gate equals the floor's (the census rows are read at ONE operating point)", e_arm.get("gate") == floor.get("gate"),
           "arm %s vs floor %s" % (e_arm.get("gate"), floor.get("gate")))
    if floor_eval_c0 is None:
        _check(cs, "window file / split equal the floor's", False,
               "UNVERIFIED: the floor's C0 eval is not on disk at the path FLOOR.json records (or its sha256 differs) and --floor-eval-c0 was "
               "not given: the window file / split of the floor cannot be compared")
    else:
        _check(cs, "window file / split equal the floor's",
               (e_arm.get("windows_file"), e_arm.get("split")) == (floor_eval_c0.get("windows_file"), floor_eval_c0.get("split")),
               "arm %s vs floor C0 eval %s" % ((e_arm.get("windows_file"), e_arm.get("split")),
                                               (floor_eval_c0.get("windows_file"), floor_eval_c0.get("split"))))
    if ref == "C0":                              # the reference IS the floor's C0: its points must be the floor's c0 to the last digit
        bad = []
        for m, r in floor["rows"].items():
            if r.get("family") != "BOXES" or r.get("undefined"):
                continue
            pt = (e_ref.get("boxes", {}).get(m) or {}).get("point")
            if not (_finite(pt) and abs(float(pt) - float(r["c0"])) <= 1e-9):
                bad.append("%s: ref %r vs floor c0 %r" % (m, pt, r["c0"]))
        _check(cs, "reference C0 eval reproduces the floor's C0 values (boxes)", not bad, "; ".join(bad[:4]) or "all box points equal")
    return cs


# ======================================================================================================================== the judgement
def _read_key(floor: dict, pair_rows: dict, key: str) -> dict:
    """lever_read of ONE emitted key: state READ / UNREAD / UNDEFINED / NO-FLOOR."""
    fl = _fl()
    frow = floor["rows"].get(key)
    rec = {"key": key}
    if frow is None:
        rec.update(state="NO-FLOOR", why="%s is not a row of the floor%s" % (key, LV_FLOOR_HINT if key.startswith("nf_swedge_prec_lv_") else ""))
        return rec
    if frow.get("undefined") or frow.get("higher_is_better") is None:
        rec.update(state="NO-FLOOR", why="%s has no floor (undefined on the floor's windows / a count)" % key)
        return rec
    prow = pair_rows.get(key)
    if prow is None:
        rec.update(state="UNREAD", why="%s was not read for this arm (no raster dirs given?)" % key)
        return rec
    if prow.get("undefined"):
        rec.update(state="UNDEFINED", why="%s is undefined on the arm-vs-reference windows (no positive / no flag): never read as 0" % key)
        return rec
    lr = fl.lever_read(floor, key, prow["delta"], prow["separated"])
    margin = lr["margin_over_floor"]
    rec.update(state="READ", ref_point=prow["ref"], arm_point=prow["arm"], delta=prow["delta"], ci_lo=prow.get("ci_lo"), ci_hi=prow.get("ci_hi"),
               separated=bool(prow["separated"]), floor_c0=frow["c0"], floor_c1=frow["c1"], floor=lr["floor_abs_c1_minus_c0"],
               keep_threshold=KEEP_FACTOR * lr["floor_abs_c1_minus_c0"], higher_is_better=frow["higher_is_better"],
               lever_read=dict(lr, margin_over_floor=_num(margin), margin_is_inf=bool(not math.isfinite(margin))))
    return rec


def _usable_floor_row(floor: dict, key: str) -> bool:
    r = floor["rows"].get(key)
    return r is not None and not r.get("undefined") and r.get("higher_is_better") is not None


def clause3_vetoes(floor: dict, pair_rows: dict, exclude: set, in_table: set) -> dict:
    """Clause 3 over the sec. 4.3 table's PERCEPTION PRIMARIES (``clause3_scope()``: every key some arm names + the named proxies of the
    primaries that have them) but the arm's own primary KEYS and its own PROXIES (``own_primary_keys``): veto iff
    separated in the BAD direction AND |delta| > 2 x that row's floor.  Every OTHER floor row (census counts at the fixed gate, per-class
    cells, ...) is still swept, and one separated worse beyond 2 x its floor is listed under ``informative_worse`` -- reported, NEVER a veto.
    (MM 2026-11-04, before any arm was read: the design says "no other PERCEPTION PRIMARY is separated worse"; the first brief said "every
    floor row", which would let a census row at a fixed gate veto B3, whose quality-aware presence moves exactly those rows by design.)
    The sweep walks the FLOOR's rows; a scope key the floor has NO row for is reported as ``missing_from_floor`` (see the block below):
    ``missing_covered`` (a primary's key whose named proxies all have floor rows) vs ``missing_uncovered`` (clause 3 is then INCOMPLETE)."""
    fl = _fl()
    out = {"vetoes": [], "informative_worse": [], "unread": [], "undefined_in_pair": [], "no_floor": [], "checked": 0,
           "excluded": sorted(exclude), "scope": "sec. 4.3 table keys + the named proxies of the primaries that have them"}
    # COMPLETENESS (MM 2026-11-04): the loop below walks the FLOOR's rows, so a scope key the floor has NO row for would be silently unchecked.
    # It is listed, and it is COVERED only when it is a primary's key whose named proxies all have a usable floor row (the map primary on a
    # mask-less floor, covered by the sidewalk-only nf_prec_5_10 / nf_prec_10_20); any other missing key leaves clause 3 INCOMPLETE.
    cover = primary_proxies()
    beside = beside_only_keys()                              # PREREG_M1_PEDGEFIT's edge rows: reported in their own section, never swept here
    out["missing_from_floor"], out["missing_covered"], out["missing_uncovered"] = [], [], []
    for k in sorted(set(in_table) - set(exclude)):
        if k in floor["rows"]:
            continue
        out["missing_from_floor"].append(k)
        px = cover.get(k, ())
        if px and all(_usable_floor_row(floor, p) for p in px):
            out["missing_covered"].append({"key": k, "covered_by_proxies": list(px)})
        else:
            out["missing_uncovered"].append(k)
    for m in sorted(floor["rows"]):
        if m in exclude:
            continue
        if m in beside:                                      # a thresholded floor carries nf_edge03_*: not a veto, not 'no floor', not even informative
            continue
        frow = floor["rows"][m]
        if frow.get("undefined") or frow.get("higher_is_better") is None:
            out["no_floor"].append(m)
            continue
        if m not in in_table:
            prow = pair_rows.get(m)
            if prow is None or prow.get("undefined"):
                continue
            lr = fl.lever_read(floor, m, prow["delta"], prow["separated"])
            if lr["exceeds_2x_floor"] and lr["separated_bad_direction"]:          # (operand order differs from the veto line: its RED arm mutates that one)
                out["informative_worse"].append({"metric": m, "group": frow.get("group"), "delta": prow["delta"],
                                                 "keep_threshold": KEEP_FACTOR * lr["floor_abs_c1_minus_c0"]})
            continue
        prow = pair_rows.get(m)
        if prow is None:
            out["unread"].append(m)
            continue
        if prow.get("undefined"):
            out["undefined_in_pair"].append(m)
            continue
        lr = fl.lever_read(floor, m, prow["delta"], prow["separated"])
        out["checked"] += 1
        if lr["separated_bad_direction"] and lr["exceeds_2x_floor"]:
            out["vetoes"].append({"metric": m, "group": frow.get("group"), "family": frow.get("family"), "delta": prow["delta"],
                                  "ci_lo": prow.get("ci_lo"), "ci_hi": prow.get("ci_hi"), "floor": lr["floor_abs_c1_minus_c0"],
                                  "keep_threshold": KEEP_FACTOR * lr["floor_abs_c1_minus_c0"], "floor_degenerate": lr["floor_degenerate"],
                                  "in_table_primaries": m in in_table})
    return out


def control_outcome(arm: str, ref: str, primary: dict) -> str:
    lr = primary["lever_read"]
    if ref == "C0":
        return ("GAINS: separated better and beyond 2 x floor" if lr["keep_on_this_metric"] else "NO-GAIN over C0 (below the keep bar)")
    worse = lr["separated_bad_direction"] and lr["exceeds_2x_floor"]
    if arm == "T1s":
        return ("AS-REQUIRED: the shuffled-history control reads worse than T1 (separated worse beyond 2 x floor)" if worse else
                "NOT-WORSE: the shuffled-history control does NOT read worse than T1 beyond 2 x floor -- T1's gain is not attributable to the history content")
    if arm == "T1hs":
        if worse:
            return "AS-REQUIRED: the shuffled-history control reads worse than T1h (separated worse beyond 2 x floor) -- T1h's gain is a HISTORY gain"
        if lr["separated_bad_direction"]:
            return ("NOT-WORSE-BEYOND-FLOOR: T1hs is separated worse than T1h but inside 2 x floor -- clause 4 as first written would pass, its "
                    "addendum does not: T1h's gain is a supervision / init gain, not attributed to history")
        return "NOT-WORSE: the shuffled-history control does NOT read worse than T1h -- T1h's gain is a supervision / init gain, not attributed to history"
    if arm == "B2c":
        return ("GAINS-LESS-THAN-B2: B2c is separated worse than B2 beyond 2 x floor -- the view, not capacity" if worse else
                "GAINS-AS-MUCH-AS-B2: B2c is not separated worse than B2 beyond 2 x floor -- the gain was capacity, not view")
    return "n/a"


def _beside_absent(ctx, key: str = "") -> str:
    """Why a beside edge row has NO paired row in this read (state NOT-COMPUTABLE, with the reason; never a guess, never 0)."""
    if key.startswith(D3_KEY_PREFIXES):                       # hold_r9_d3beside's rows need --map-beside-d3, not thresholds
        if ctx is None or not ctx.get("rasters"):
            return "no raster dirs were given to this verdict (--rast-arm / --rast-ref): the lane / drivable-boundary rows are counted on the rasters"
        if not ctx.get("beside_d3"):
            return ("this read ran without --map-beside-d3: the lane tolerance-IoU / drivable-boundary recall counts are taken only when asked "
                    "(hold_r9_d3beside; an unset read is byte-identical to the read before them)")
        return "the --map-beside-d3 read emitted no such row (rasters without a code / pdrv plane)"
    if ctx is None:
        return ("no paired row in this read: the edge-class F@0.3 m rows are emitted only by a THRESHOLDED raster read (--rast-arm / --rast-ref with "
                "--map-thresholds-arm / -ref); a fixed-128 read, or an evaluator that predates hold_r9_pedgefit, has none")
    if not ctx.get("rasters"):
        return "no raster dirs were given to this verdict (--rast-arm / --rast-ref): the edge rows are counted on the rasters"
    if not ctx.get("thresholds"):
        return ("this read is at the fixed 128 (no --map-thresholds-arm / -ref): the edge-class P / R / F @ 0.3 m is defined at the arm's OWN FIT pedge level, "
                "so the evaluator emits it only on a thresholded read")
    return "the thresholded read emitted no such row (an evaluator that predates hold_r9_pedgefit, or rasters without a pedge plane)"


def _read_beside(floor: dict, pair_rows: dict, key: str, ctx) -> dict:
    """Read ONE never-deciding beside row from the verdict's own PAIRED row: delta, 95 % CI, separated, direction (higher is better). The floor is
    context only (``floor_context``, when it has a usable row for the key) and decides nothing. States: READ / UNDEFINED / NOT-COMPUTABLE."""
    prow = pair_rows.get(key)
    if prow is None:
        return {"key": key, "state": "NOT-COMPUTABLE", "why": _beside_absent(ctx, key)}
    if prow.get("undefined") or not _finite(prow.get("delta")):
        return {"key": key, "state": "UNDEFINED", "why": "undefined on the arm-vs-reference windows (no GT edge cell in the band / no judged flag): never read as 0"}
    d, sep = float(prow["delta"]), bool(prow["separated"])
    frow = floor["rows"].get(key)
    fc = None
    if frow is not None and not frow.get("undefined") and frow.get("higher_is_better") is not None and _finite(frow.get("c0")) and _finite(frow.get("c1")):
        fc = {"c0": _num(frow["c0"]), "c1": _num(frow["c1"]), "abs_c1_minus_c0": _num(abs(frow["c1"] - frow["c0"]))}
    return {"key": key, "state": "READ", "ref_point": _num(prow.get("ref")), "arm_point": _num(prow.get("arm")), "delta": d, "ci_lo": _num(prow.get("ci_lo")),
            "ci_hi": _num(prow.get("ci_hi")), "separated": sep, "direction": "better" if d > 0 else "worse" if d < 0 else "flat",
            "reading": "SEPARATED WORSE" if (sep and d < 0) else "SEPARATED BETTER" if (sep and d > 0) else "NOT SEPARATED", "floor_context": fc}


def _beside_block(arm: str, ref: str, rows: list) -> dict:
    """The ``beside_pedgefit`` block: every ``clause3=False`` row of the arm, and the SELECTOR reading of the pooled F@0.3 m 5-20 m row (PREREG_M1_PEDGEFIT:
    it selects between rows 2 and 3 of the committed-readings table when the decisive M1-vs-C0 verdict is a DROP on clause 1-2 -- and nothing else)."""
    sel = next((r for r in rows if r["key"] == EDGE03_SELECTOR_KEY), None)
    worse = None if sel is None or sel["state"] != "READ" else bool(sel["separated"] and sel["delta"] < 0)
    applies = ("PREREG_M1_PEDGEFIT: if M1-vs-C0 is a DROP on clause 1-2, NOT separated worse -> committed-readings row 2, separated WORSE -> row 3"
               if (arm, ref) == ("M1", "C0") else
               "reported for %s vs %s; PREREG_M1_PEDGEFIT's rows 2/3 are the readings of the decisive M1-vs-C0 verdict" % (arm, ref))
    return {"title": BESIDE_TITLE, "note": BESIDE_NOTE, "decides": False, "rows": rows,
            "selector": {"key": EDGE03_SELECTOR_KEY, "state": None if sel is None else sel["state"], "reading": None if sel is None else sel.get("reading"),
                         "separated_worse": worse, "applies": applies}}


def judge(arm: str, ref: str, floor: dict, pair_rows: dict, ctx=None) -> dict:
    """The verdict from LITERAL rows. ``pair_rows``: metric -> {ref, arm, delta, ci_lo, ci_hi, separated, undefined}.  ``ctx`` (optional, from
    ``build_verdict``): ``{"rasters": bool, "thresholds": bool}``, only to word WHY a beside edge row is absent. The ``clause3=False`` specs are read
    into ``beside_pedgefit`` and touch nothing else (not ``reads``, not a clause, not the verdict)."""
    spec = ARMS[arm]
    reads, named_guards, unevaluable_guards, beside = [], [], [], []
    primary = None
    for m in spec["metrics"]:
        if not m["clause3"]:
            beside.append(dict({"id": m["id"], "text": m["text"], "role": m["role"], "keys": list(m["keys"]), "status": m["status"]},
                               **_read_beside(floor, pair_rows, m["keys"][0], ctx)))
            continue
        rec = {"id": m["id"], "text": m["text"], "role": m["role"], "keys": list(m["keys"]), "status": m["status"], "proxies": list(m["proxies"])}
        if m["status"] == "COMPUTABLE":
            rec.update(_read_key(floor, pair_rows, m["keys"][0]))
        else:
            rec.update(state="NOT-COMPUTABLE", why=m["reason"], would_compute=m["would_compute"])
        if m["role"] == "primary":
            primary = rec
        elif m["role"] == "guard":
            named_guards.append(rec)
            if rec["state"] != "READ":
                unevaluable_guards.append(rec)
        reads.append(rec)
    exclude = own_primary_keys(arm)                          # the arm's own primary keys AND its own proxies (a map arm is not vetoed by its stand-in)
    c3 = clause3_vetoes(floor, pair_rows, exclude, clause3_scope())
    failing, notes = [], []
    if c3["vetoes"]:
        v = ", ".join("%s (delta %+.6g, 2 x floor %.6g)" % (x["metric"], x["delta"], x["keep_threshold"]) for x in c3["vetoes"])
        failing.append({"clause": 3, "text": "another perception primary is separated worse beyond 2 x its floor: " + v})
    prim_ok = primary is not None and primary["state"] == "READ"
    if prim_ok:
        lr = primary["lever_read"]
        if not lr["separated_good_direction"]:
            failing.append({"clause": 1, "text": "the primary %s is not separated in the good direction (delta %+.6g, separated %s, %s is better)"
                            % (primary["key"], primary["delta"], primary["separated"], "higher" if primary["higher_is_better"] else "lower")})
        if not lr["exceeds_2x_floor"]:
            failing.append({"clause": 2, "text": "|delta| %.6g <= 2 x floor %.6g on the primary %s (floor |C1 - C0| %.6g%s)"
                            % (abs(primary["delta"]), primary["keep_threshold"], primary["key"], primary["floor"],
                               ", DEGENERATE" if lr["floor_degenerate"] else "")})
    not_computable = []
    if not prim_ok:
        not_computable.append("clauses 1-2: the primary '%s' is %s: %s" % (primary["text"] if primary else "?", primary["state"] if primary else "absent",
                                                                          primary.get("why", "") if primary else ""))
    for g in unevaluable_guards:
        not_computable.append("named guard '%s' is %s: %s" % (g["text"], g["state"], g.get("why", "")))
    if c3["unread"]:
        not_computable.append("clause 3 not complete: %d floor row(s) were not read for this arm (%s)" % (len(c3["unread"]), ", ".join(c3["unread"][:6])))
    for k in c3["missing_uncovered"]:                        # a scope key with no floor row and no covering proxies: clause 3 cannot be complete
        not_computable.append("clause 3 not complete: floor has no row for %s: re-read the floor with the current eval" % k)
    alt = None
    if prim_ok or named_guards:
        listed = [r for r in reads if r["role"] in ("primary", "guard")]
        alt = {"all_listed_primaries_kept": bool(all(r["state"] == "READ" and r["lever_read"]["keep_on_this_metric"] for r in listed)),
               "listed": [{"id": r["id"], "key": r.get("key"), "state": r["state"],
                           "keep_on_this_metric": (r["lever_read"]["keep_on_this_metric"] if r["state"] == "READ" else None)} for r in listed],
               "note": "the CONJUNCTION reading of a two-metric sec. 4.3 primary (every listed metric must itself clear clauses 1-2); NOT the decision"}
    strict = [{"id": g["id"], "key": g.get("key"), "separated_worse_any_size": (g["lever_read"]["separated_bad_direction"] if g["state"] == "READ" else None)}
              for g in named_guards]
    out = {"reads": reads, "clause3": c3, "failing_clauses": failing, "not_computable": not_computable, "alternative_reading": alt,
           "named_guards": strict}
    if beside:                                               # additive and arm-specific: an arm with no clause3=False spec has exactly the keys it had
        out["beside_pedgefit"] = _beside_block(arm, ref, beside)
    if spec["kind"] == "control":
        if prim_ok:
            out["verdict"], out["control_outcome"] = "CONTROL-READ", control_outcome(arm, ref, primary)
        else:
            out["verdict"], out["control_outcome"] = "NOT-COMPUTABLE", "the control's primary is not computable"
    elif failing:
        out["verdict"] = "DROP"
    elif not_computable:
        out["verdict"] = "NOT-COMPUTABLE"
    else:
        out["verdict"] = "KEEP"
    return out


# ======================================================================================================================== build
def _pair_rows_from(rows: dict) -> dict:
    return {m: {"ref": r.get("c0"), "arm": r.get("c1"), "delta": r.get("delta"), "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                "separated": bool(r.get("separated")), "undefined": bool(r.get("undefined"))} for m, r in rows.items()}


def _families() -> list:
    return [{"family": f, "status": s} for f, s in _fl().FAMILIES]


def resolve_ref(arm: str, ref):
    if arm not in ARMS:
        raise Refused("unknown arm %r (known: %s)" % (arm, ", ".join(DESIGN_ARMS)))
    s = ARMS[arm]
    if s["kind"] == "floor":
        raise Refused("%s is the replicate floor (r9_e1_floor.py), not a lever arm" % arm)
    r = ref or s["ref"]
    if r is None:
        raise Refused("REFCV9_DESIGN 4.3 names no reference for %s (%s)" % (arm, "; ".join(s["ambiguities"]) or "run-time choice"))
    if r not in s["allowed"]:
        raise Refused("%s cannot be read against %r (accepted: %s)" % (arm, r, ", ".join(s["allowed"]) or "none"))
    return r


def _thresholds_protocol(out, checks, problems, floor, rast_meta, e_arm, e_ref, ref) -> None:
    """The map-threshold PROTOCOL the verdict was read on must be the floor's: both at the fixed 128, or both on each arm's OWN FIT cuts. A floor read
    at 128 judging an arm read on its FIT cut (or the reverse) compares two different operating points and REFUSES. Also: the single evals must carry the
    cut they were given, and a C0 reference must be cut by the very file the floor's C0 was. Adds nothing to ``out`` when neither side has thresholds."""
    tv = (rast_meta or {}).get("thresholds")                    # {"c0": ref side, "c1": arm side} as read now
    tf = (floor.get("map_inputs") or {}).get("thresholds")      # {"c0": the floor's C0, "c1": the floor's C1}
    if not tv and not tf:
        return
    prot = lambda t: "per-arm FIT thresholds" if t else "fixed 128"          # noqa: E731
    same = bool(tv) == bool(tf)
    out["map_thresholds"] = {"protocol_verdict": prot(tv), "protocol_floor": prot(tf), "same_protocol": same,
                             "arm": None if not tv else tv["c1"], "ref": None if not tv else tv["c0"],
                             "floor_c0": None if not tf else tf["c0"], "floor_c1": None if not tf else tf["c1"]}
    _check(checks, "map-threshold protocol equals the floor's (both fixed 128, or both each-arm-FIT)", same,
           "verdict: %s; floor: %s" % (prot(tv), prot(tf)) + ("" if same else " -- re-read the floor (r9_e1_floor.py --map-thresholds-c0/-c1) and this verdict on ONE protocol"))
    if not checks[-1]["ok"]:
        problems.append("comparability: %s -- %s" % (checks[-1]["check"], checks[-1]["detail"]))
        return
    if not tv:
        return
    for tag, e, side in (("arm", e_arm, "c1"), ("ref", e_ref, "c0")):
        nf = (e or {}).get("nearfield")
        if nf is None:
            continue
        s_e, s_v = (nf.get("thresholds") or {}).get("sha256"), tv[side]["sha256"]
        _check(checks, "%s single eval was cut at the thresholds this verdict was given" % tag, s_e == s_v,
               "eval %s vs given %s" % ((s_e or "none")[:12], s_v[:12]))
        if not checks[-1]["ok"]:
            problems.append("comparability: %s -- %s" % (checks[-1]["check"], checks[-1]["detail"]))
    if ref in ("C0", "C1"):
        side = "c0" if ref == "C0" else "c1"
        sf, sv = ((tf or {}).get(side) or {}).get("sha256"), tv["c0"]["sha256"]
        _check(checks, "the %s reference is cut by the file the floor's %s was" % (ref, ref), sf == sv,
               "verdict %s vs floor %s" % (sv[:12], (sf or "none")[:12]))
        if not checks[-1]["ok"]:
            problems.append("comparability: %s -- %s" % (checks[-1]["check"], checks[-1]["detail"]))


def build_verdict(floor_path, paired, eval_arm, eval_ref, arm, ref=None, *, rast_arm=None, rast_ref=None, rast_names=None, rast_stride=1,
                  psw_key="psw", floor_eval_c0=None, boot=None, seed=None, lift_valid=None, thr_arm=None, thr_ref=None, beside_d3=False) -> dict:
    out = {"schema": SCHEMA, "tier": TIER, "arm": arm, "keep_rule": KEEP_RULE, "keep_factor": KEEP_FACTOR, "status": "OK", "verdict": None,
           "estimator": {"answers": "another draw of EPISODES, NOT another training run: the floor |C1-C0| is ONE seed pair (n = 1) and "
                                    "the arm is ONE training run"}}
    try:
        fl = _fl()
        out["families"] = _families()
        floor = _load_json(floor_path, "floor")
        out["floor"] = {"path": str(floor_path), "sha256": _sha256(floor_path), "schema": floor.get("schema"), "status": floor.get("status"),
                        "n_windows": floor.get("n_windows"), "n_clips": floor.get("n_clips"), "gate": floor.get("gate"),
                        "estimator": floor.get("estimator"), "n_rows": len(floor.get("rows") or {})}
        if floor.get("schema") != fl.SCHEMA:
            raise Refused("%s: schema %r != %r (not an r9_e1_floor output)" % (floor_path, floor.get("schema"), fl.SCHEMA))
        if floor.get("status") != "OK" or not floor.get("rows"):
            raise Refused("the floor is not OK (status %r): no verdict is read against a REFUSED / empty floor" % floor.get("status"))
        fe = floor.get("estimator") or {}
        boot = int(fe["B"]) if boot is None else int(boot)
        seed = int(fe["seed"]) if seed is None else int(seed)
        ref = resolve_ref(arm, ref)
        out.update(ref=ref, kind=ARMS[arm]["kind"], mapping=_mapping_entry(arm))
        e_arm, e_ref, ep = fl._load_eval(eval_arm), fl._load_eval(eval_ref), fl._load_eval(paired)
        out["inputs"] = {k: {"path": str(p), "sha256": _sha256(p)} for k, p in (("eval_arm", eval_arm), ("eval_ref", eval_ref), ("paired", paired))}
        if (rast_arm is None) != (rast_ref is None):
            raise Refused("--rast-arm and --rast-ref go together")
        if lift_valid is not None and rast_arm is None:
            raise Refused("--lift-valid needs --rast-arm / --rast-ref: a mask with no rasters would be silently ignored")
        if bool(thr_arm) != bool(thr_ref):
            raise Refused("--map-thresholds-arm and --map-thresholds-ref go together: the arm and its reference are each read on their OWN FIT cuts, or both at 128")
        if thr_arm and rast_arm is None:
            raise Refused("--map-thresholds-arm / -ref need --rast-arm / --rast-ref: thresholds with no rasters would be silently ignored")
        fe_path = floor_eval_c0
        fe_rec = (floor.get("inputs") or {}).get("eval_c0") or {}
        if fe_path is None and fe_rec.get("path") and Path(fe_rec["path"]).is_file():
            fe_path = fe_rec["path"]
        fe_eval = None
        if fe_path is not None and Path(fe_path).is_file() and fe_rec.get("sha256") == _sha256(fe_path):
            fe_eval = _load_json(fe_path, "floor C0 eval")
            out["inputs"]["floor_eval_c0"] = {"path": str(fe_path), "sha256": fe_rec["sha256"]}
        checks = floor_comparability(floor, e_ref, e_arm, ep, boot, seed, fe_eval, ref)
        out["comparability"] = checks
        problems = ["comparability: %s -- %s" % (c["check"], c["detail"]) for c in checks if not c["ok"]]
        if problems:
            raise Refused("; ".join(problems))
        pair_rows = _pair_rows_from(fl.box_rows(e_ref, e_arm, ep, problems, seed))
        rast_meta = None
        if rast_arm is not None:
            mrows, rast_meta = fl.nearfield_rows(rast_ref, rast_arm, names_file=rast_names, stride=rast_stride, psw_key=psw_key, boot=boot,
                                                 seed=seed, e0=e_ref, e1=e_arm, problems=problems, lift_valid=lift_valid, thr0=thr_ref, thr1=thr_arm,
                                                 **({"beside_d3": True} if beside_d3 else {}))     # hold_r9_d3beside: unset = the call before it
            pair_rows.update(_pair_rows_from(mrows))
            _thresholds_protocol(out, checks, problems, floor, rast_meta, e_arm, e_ref, ref)
            if "lift_valid" in (rast_meta or {}):            # set once the mask step ran: the nf_swedge_*lv rows were read on THIS mask
                lv_v, lv_f = rast_meta["lift_valid"], (floor.get("map_inputs") or {}).get("lift_valid")
                sv, sf = (lv_v or {}).get("sha256"), (lv_f or {}).get("sha256")
                out["lift_valid"] = {"verdict": lv_v, "floor": lv_f, "same_mask": sv == sf}
                _check(checks, "lift-valid mask equals the floor's (the _lv_ rows are read on the same cells)", sv == sf,
                       "verdict %s vs floor %s" % ((sv or "none")[:12], (sf or "none")[:12]) +
                       ("" if sv == sf else " -- one side has a mask the other lacks, or the masks differ: re-read the floor and the verdict on ONE --lift-valid"))
                if not checks[-1]["ok"]:
                    problems.append("comparability: %s -- %s" % (checks[-1]["check"], checks[-1]["detail"]))
            if "n_files" in (rast_meta or {}) and floor.get("map_inputs"):
                fm = floor["map_inputs"]
                _check(checks, "raster file / episode counts equal the floor's",
                       (rast_meta.get("n_files"), rast_meta.get("n_episodes")) == (fm.get("n_files"), fm.get("n_episodes")),
                       "arm-vs-ref (%s files / %s episodes) vs floor (%s / %s)" % (rast_meta.get("n_files"), rast_meta.get("n_episodes"),
                                                                                   fm.get("n_files"), fm.get("n_episodes")))
                if not checks[-1]["ok"]:
                    problems.append("comparability: %s -- %s" % (checks[-1]["check"], checks[-1]["detail"]))
            if ref == "C0":
                for m, r in mrows.items():
                    fr = floor["rows"].get(m)
                    if fr and not fr.get("undefined") and _finite(r.get("c0")) and abs(r["c0"] - fr["c0"]) > 1e-9:
                        problems.append("%s: reference C0 near-field value %.12g != the floor's c0 %.12g (a different raster set / stride?)" % (m, r["c0"], fr["c0"]))
        out["map_inputs"] = rast_meta
        if problems:
            raise Refused("; ".join(problems))
        out["n_windows"], out["n_clips"], out["gate"] = e_arm.get("n_windows"), e_arm.get("n_clips"), e_arm.get("gate")
        out["boxes_per_object_noNMS_info"] = _bpo_info(e_arm, e_ref)
        out.update(judge(arm, ref, floor, pair_rows, ctx={"rasters": rast_arm is not None, "thresholds": bool(thr_arm),
                                                          **({"beside_d3": True} if beside_d3 else {})}))
        out["next_lever_on_drop"] = ARMS[arm]["next_lever"]
        out["ambiguities"] = list(ARMS[arm]["ambiguities"])
    except Refused as e:
        out.update(status="REFUSED", verdict="REFUSED", problems=[str(e)])
        out.setdefault("families", [])
    return out


def _bpo_info(e_arm: dict, e_ref: dict) -> dict:
    g = lambda e: (e.get("boxes", {}).get("boxes_per_object_noNMS") or {}).get("point")      # noqa: E731
    return {"arm": _num(g(e_arm)), "ref": _num(g(e_ref)), "design_note_2_4": "sec. 2.4 line 179: 'if B3's DN + quality alone reach <= 1.25 "
            "boxes/object, the in-loss NMS is dropped' -- information for B3 / B3p, NOT part of the keep rule"}


def _mapping_entry(arm: str) -> dict:
    s = ARMS[arm]
    return {"arm": arm, "kind": s["kind"], "variable": s["variable"], "reference_default": s["ref"], "references_accepted": list(s["allowed"]),
            "argv_flags": list(s["argv_flags"]),
            "metrics": [dict({"id": m["id"], "text": m["text"], "role": m["role"], "status": m["status"], "keys": list(m["keys"]),
                              "proxies": list(m["proxies"]), "reason": m["reason"], "would_compute": m["would_compute"]},
                             **({} if m["clause3"] else {"clause3": False})) for m in s["metrics"]]}


# ======================================================================================================================== rendering
def _f(x, nd=6) -> str:
    return "n/a" if x is None else "%.*g" % (nd, x)


def _proxy_note(r: dict) -> str:
    return ("*Emitted proxy keys (sidewalk-only, lift-unmasked; they NEVER stand in for this primary in clauses 1-2, and stay in clause 3's scope for "
            "levers of other families):* " + ", ".join("`%s`" % k for k in r["proxies"]))


def _thresholds_line(v: dict) -> str:
    t = v.get("map_thresholds")
    if not t:
        return ""
    if not t.get("arm"):
        return "Map flags: fixed 128 (verdict and floor)."
    return ("Map flags cut at each arm's OWN FIT-split thresholds (not 128): arm `%s`, reference `%s` (sha256 of the files); protocol %s the floor's." %
            (t["arm"]["sha256"], t["ref"]["sha256"], "EQUAL to" if t.get("same_protocol") else "NOT EQUAL to"))


def _lift_valid_line(v: dict) -> str:
    lv = v.get("lift_valid")
    if not lv:
        return ("Map primary mask: none given to this verdict (no rasters, or no --lift-valid): the `nf_swedge_*` rows exist only when the floor, both "
                "evals and the verdict ran on ONE arm-independent mask.")
    m = lv.get("verdict")
    if not m:
        return "Map primary mask: none, and the floor has none: no `nf_swedge_*` row exists (the map primary reads NO-FLOOR -> NOT-COMPUTABLE)."
    return ("Map primary mask (arm-independent lift-valid): sha256 `%s` (%s, %s episodes) -- %s the floor's." %
            (m.get("sha256"), m.get("source"), m.get("n_episodes"), "EQUAL to" if lv.get("same_mask") else "NOT EQUAL to"))


def _beside_md(b: dict) -> list:
    """The short 'Beside (PREREG_M1_PEDGEFIT rows 2/3 selector)' section: READ rows as a table, every other state as one line with its reason."""
    L = ["", "## " + b["title"], "", b["note"], ""]
    read = [r for r in b["rows"] if r["state"] == "READ"]
    if read:
        L += ["| metric | emitted key | ref | arm | delta (arm - ref) | 95 % CI | separated? | direction | floor abs(C1 - C0), context only |", "|---|---|---|---|---|---|---|---|---|"]
        for r in read:
            fc = r.get("floor_context")
            L.append("| %s | `%s` | %s | %s | %+.6g | [%s, %s] | %s | %s | %s |" % (r["text"], r["key"], _f(r["ref_point"]), _f(r["arm_point"]), r["delta"], _f(r["ci_lo"]),
                                                                              _f(r["ci_hi"]), "yes" if r["separated"] else "no", r["direction"],
                                                                              _f(fc["abs_c1_minus_c0"]) if fc else "no floor row"))
        L.append("")
    for st in ("NOT-COMPUTABLE", "UNDEFINED"):
        grp = [r for r in b["rows"] if r["state"] == st]
        for why in sorted({r["why"] for r in grp}):
            L.append("* **%s** (%d row(s), e.g. `%s`): %s." % (st, sum(1 for r in grp if r["why"] == why), next(r["key"] for r in grp if r["why"] == why), why))
    s = b["selector"]
    if s["state"] == "READ":
        r = next(x for x in b["rows"] if x["key"] == s["key"])
        L += ["", "Selector `%s` (pooled 5-20 m F@0.3 m, paired arm - ref): delta %+.6g [%s, %s], %s -> separated worse: **%s**. %s. It selects nothing else." % (
            s["key"], r["delta"], _f(r["ci_lo"]), _f(r["ci_hi"]), "separated" if r["separated"] else "not separated", "yes" if s["separated_worse"] else "no", s["applies"])]
    else:
        L += ["", "Selector `%s`: **%s**. %s." % (s["key"], s["state"], s["applies"])]
    return L


def render_md(v: dict) -> str:
    L = ["# E1 verdict -- arm %s%s, DEV" % (v.get("arm"), (" vs " + v["ref"]) if v.get("ref") else ""), ""]
    if v["status"] != "OK":
        L += ["**STATUS: REFUSED -- no verdict is quoted.**", ""] + ["* " + p for p in v.get("problems", [])]
        L += ["", "## The four metric families (CLAUDE.md, binding)", "", "| family | status |", "|---|---|"] + ["| %s | %s |" % (x["family"], x["status"]) for x in v["families"]]
        return "\n".join(L) + "\n"
    L += ["**VERDICT: %s**%s" % (v["verdict"], (" -- " + v["control_outcome"]) if v.get("control_outcome") else ""), ""]
    for f in v["failing_clauses"]:
        L.append("* FAILING CLAUSE %d: %s" % (f["clause"], f["text"]))
    for n in v["not_computable"]:
        L.append("* NOT-COMPUTABLE: " + n)
    if v["verdict"] == "KEEP":
        L.append("* all three clauses hold on the pre-registered primary; KEEP means 'cleared the E1 bar', not 'proved' (one seed, n = 1 floor).")
    est = v["estimator"]
    g = v.get("gate") or {}
    L += ["", "Tier **%s** (perception only). Paired estimator: episode-cluster bootstrap (`paired_episode_cluster_bootstrap_stat`) -- it answers *%s*."
          % (v["tier"], est["answers"]),
          "Windows: %s DEV windows / %s episodes (clips), identical for arm, reference and floor (paired alignment drops none). Fixed P=R gate: %s (%s)."
          % (v.get("n_windows"), v.get("n_clips"), g.get("value"), g.get("source")), "",
          "Arm **%s** (%s) against reference **%s**." % (v["arm"], v["mapping"]["variable"], v["ref"]), "",
          "## The keep rule (REFCV9_DESIGN 4.3)", "", v["keep_rule"], "",
          "## Floor provenance", "", "FLOOR.json sha256 `%s` (status %s, %s rows, B %s seed %s). Floor = |C1 - C0|, ONE seed pair (n = 1)."
          % (v["floor"]["sha256"], v["floor"]["status"], v["floor"]["n_rows"], (v["floor"].get("estimator") or {}).get("B"),
             (v["floor"].get("estimator") or {}).get("seed")), _lift_valid_line(v), *([_thresholds_line(v)] if v.get("map_thresholds") else []), "",
          "## The four metric families (CLAUDE.md, binding)", "", "| family | status |", "|---|---|"]
    L += ["| %s | %s |" % (x["family"], x["status"]) for x in v["families"]]
    L += ["", "Perception is the family this rig reads: boxes and the near-field map raster.", "", "## The pre-registered primary (and named guards)", "",
          "| metric (design text) | role | emitted key | state | ref | arm | delta | 95 % CI | separated? | floor | 2 x floor | keep on this metric |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in v["reads"]:
        if r["state"] == "READ":
            L.append("| %s | %s | `%s` | READ | %s | %s | %+.6g | [%s, %s] | %s | %s | %s | %s |" % (
                r["text"], r["role"], r["key"], _f(r["ref_point"]), _f(r["arm_point"]), r["delta"], _f(r["ci_lo"]), _f(r["ci_hi"]),
                "yes" if r["separated"] else "no", _f(r["floor"]), _f(r["keep_threshold"]), "YES" if r["lever_read"]["keep_on_this_metric"] else "no"))
        else:
            L.append("| %s | %s | %s | **%s** | | | | | | | | |" % (r["text"], r["role"], ", ".join("`%s`" % k for k in r["keys"]) or "none emitted", r["state"]))
    for r in v["reads"]:
        if r["state"] in ("UNREAD", "UNDEFINED", "NO-FLOOR"):
            L += ["", "**%s -- %s.** %s" % (r["text"], r["state"], r.get("why", ""))]
            if r["proxies"]:
                L += ["", _proxy_note(r)]
    for r in v["reads"]:
        if r["state"] == "NOT-COMPUTABLE":
            L += ["", "**%s -- NOT-COMPUTABLE.** %s" % (r["text"], r["why"]), "",
                  "*What would compute it:* %s" % r["would_compute"]]
            if r["proxies"]:
                L.append(_proxy_note(r))
    if v.get("alternative_reading"):
        L += ["", "Alternative (conjunction) reading, NOT the decision: every listed primary kept -> **%s**." % v["alternative_reading"]["all_listed_primaries_kept"]]
    c3 = v["clause3"]
    L += ["", "## Clause 3 -- no other perception primary separated worse beyond 2 x its floor", "",
          "Swept over the sec. 4.3 table's perception primaries but the arm's own primary keys and named proxies (excluded: %s): %d rows checked; %d vetoes; "
          "%d not read; %d undefined on the pair; %d without a floor. Other floor rows separated worse beyond 2 x their floor (INFORMATIVE, never a veto): %d."
          % (", ".join("`%s`" % k for k in c3["excluded"]) or "none", c3["checked"], len(c3["vetoes"]), len(c3["unread"]), len(c3["undefined_in_pair"]),
             len(c3["no_floor"]), len(c3.get("informative_worse", [])))]
    if c3.get("informative_worse"):
        L += ["", "Informative (not a veto): " + ", ".join("`%s` %+.6g (2 x floor %s)" % (x["metric"], x["delta"], _f(x["keep_threshold"]))
                                                           for x in c3["informative_worse"])]
    if c3["vetoes"]:
        L += ["", "| vetoing metric | group | delta | 95 % CI | 2 x floor | floor degenerate | a sec. 4.3 table key? |", "|---|---|---|---|---|---|---|"]
        for x in c3["vetoes"]:
            L.append("| `%s` | %s | %+.6g | [%s, %s] | %s | %s | %s |" % (x["metric"], x["group"], x["delta"], _f(x["ci_lo"]), _f(x["ci_hi"]),
                                                                        _f(x["keep_threshold"]), x["floor_degenerate"], x["in_table_primaries"]))
        L.append("")
    if c3["unread"]:
        L += ["", "Not read (clause 3 incomplete): " + ", ".join("`%s`" % m for m in c3["unread"])]
    if c3.get("missing_from_floor"):
        L += ["", "Scope keys the floor has NO row for (%d): %s." % (len(c3["missing_from_floor"]), ", ".join("`%s`" % k for k in c3["missing_from_floor"]))]
        for x in c3["missing_covered"]:
            L.append("* `%s` -- covered by proxies %s (the floor has their rows, so a lever of another family is still vetoed on the map family)."
                     % (x["key"], ", ".join("`%s`" % p for p in x["covered_by_proxies"])))
        for k in c3["missing_uncovered"]:
            L.append("* `%s` -- **UNCOVERED**: floor has no row for %s: re-read the floor with the current eval. Clause 3 is incomplete, so this verdict "
                     "cannot be KEEP (a DROP from another clause still stands)." % (k, k))
    if v.get("beside_pedgefit"):
        L += _beside_md(v["beside_pedgefit"])
    if v.get("boxes_per_object_noNMS_info") and v["arm"] in ("B3", "B3p"):
        b = v["boxes_per_object_noNMS_info"]
        L += ["", "Information (not the keep rule): boxes per detected object, no NMS, at the fixed gate -- arm %s, reference %s. %s"
              % (_f(b["arm"]), _f(b["ref"]), b["design_note_2_4"])]
    L += ["", "## Mapping (from the source of this file)", "", "variable: %s." % v["mapping"]["variable"], "argv lever flags: %s." % (", ".join(v["mapping"]["argv_flags"]) or "none"), ""]
    if v["verdict"] == "DROP":
        L += ["## Next lever (Rule Zero: a refutation is a waypoint) -- from the design text", "", v["next_lever_on_drop"], "",
              "The next lever is named by REFCV9_DESIGN (cited by section and line); where the design names none the text says UNVERIFIED -- it is not invented here."]
    if v.get("ambiguities"):
        L += ["", "## Ambiguities in sec. 4.3 (the bar was NOT changed)", ""] + ["* " + a for a in v["ambiguities"]]
    L += ["", "## Reading notes", "",
          "* The interval is the paired episode-cluster bootstrap: it cannot see training variance (H-ESTIM-SEED-1); the 2 x floor clause is what reads the lever against the rig's own run-to-run noise.",
          "* A floor of exactly 0 is DEGENERATE (two runs agreeing to the last digit); clause 2 then rests on separation alone and clause 3 vetoes any separated worse row.",
          "* Census rows (`prec_gate_*`, `rec_gate_*`, `boxes_per_object_*`) are read at the fixed gate for both arms, not at each arm's own P=R point.",
          "* Inputs: " + "; ".join("%s sha256 %s" % (k, x["sha256"][:12]) for k, x in v["inputs"].items()) + "."]
    return "\n".join(L) + "\n"


def mapping_md() -> str:
    L = ["| arm | kind | reference | metric (design text) | role | emitted key(s) / file:line | status |", "|---|---|---|---|---|---|---|"]
    for a, s in ARMS.items():
        ref = s["ref"] or ("; ".join(s["allowed"]) + " (run-time)" if s["allowed"] else "NONE NAMED")
        if not s["metrics"]:
            L.append("| %s | %s | - | %s | - | - | the floor |" % (a, s["kind"], s["variable"]))
        for m in s["metrics"]:
            if m["status"] == "COMPUTABLE":
                ks = ", ".join("`%s` (%s:%d)" % (k, key_src(k)[0].split("/")[-1], key_src(k)[1]) for k in m["keys"])
                L.append("| %s | %s | %s | %s | %s | %s | COMPUTABLE |" % (a, s["kind"], ref, m["text"], m["role"], ks))
            else:
                L.append("| %s | %s | %s | %s | %s | none; proxies: %s | **NOT-COMPUTABLE** |" % (a, s["kind"], ref, m["text"], m["role"],
                                                                                              ", ".join("`%s`" % p for p in m["proxies"]) or "none"))
    return "\n".join(L) + "\n"


# ======================================================================================================================== state (stdlib)
def _state_mod():
    cands = [os.environ.get("R9E1_STATE_PY"), str(HERE / "r9e1_read_state.py"), str(HERE.parent / "ops" / "r9e1_read" / "r9e1_read_state.py")]
    for c in cands:
        if c and Path(c).is_file():
            spec = importlib.util.spec_from_file_location("r9e1_read_state_ro", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("[r9-e1-verdict] r9e1_read_state.py not found (tried %s)" % [c for c in cands if c])


def _arm_state(st, a: str, seg: list, all_lines: list, root: Path):
    """-> (state, reason, line, run_dir). The chain's own done rule (markers) decides DONE; the LATEST chain event of the arm decides a failure."""
    bad = {a + "-" + k for k in st.BAD_KINDS}
    evs = [l for l in seg if l[1] in bad or l[1] in (a + "-LAUNCH", a + "-DONE", a + "-ALREADY-DONE")]
    launches = [l for l in all_lines if l[1] == a + "-LAUNCH"]
    run = st.run_dir_of(a, launches[-1] if launches else None, root)
    last = evs[-1] if evs else None
    if last is not None and last[1] in bad:
        return "ABORT", "%s did not finish: %s" % (a, last[1]), last[3], run
    probs = st.marker_problems(run)
    if not probs:
        probs = st.checkpoint_problems({a: run}, root)
        if probs:
            return "ABORT", "%s: the run dir disagrees with its done marker: %s" % (a, "; ".join(probs)), last[3] if last else "", run
        return "READY", "%s done: markers and final EMA checkpoint verified" % a, last[3] if last else "", run
    if last is not None and last[1] in (a + "-DONE", a + "-ALREADY-DONE"):
        return "ABORT", "%s-DONE is logged but the run dir disagrees: %s" % (a, "; ".join(probs)), last[3], run
    if any(l[1] == "ALLDONE" for l in seg):
        return "ABORT", "ALLDONE without %s-DONE" % a, [l for l in seg if l[1] == "ALLDONE"][-1][3], run
    return "WAIT", "waiting for the chain: %s %s" % (a, "training" if last is not None else "not launched yet"), "", run


def arm_state(arm: str, ref, status_text: str, root, read_dir) -> dict:
    st = _state_mod()
    root, rd = Path(root), Path(read_dir)
    lines = st.parse_status(status_text)
    seg = st.last_segment(lines)
    d = {"arm": arm, "state": "WAIT", "reason": "", "line": "", "run": "", "ref_run": "", "sig": ""}

    def fin(state, reason, line="", run="", ref_run=""):
        d.update(state=state, reason=reason, line=line, run=str(run), ref_run=str(ref_run))
        d["sig"] = "|".join([state, d["arm"], reason[:80]])
        return d
    if (rd / "READ_ABORTED").exists():
        return fin("ABORT", "the floor read aborted (%s/READ_ABORTED)" % rd)
    fj = rd / "FLOOR.json"
    if not fj.is_file():
        return fin("WAIT", "waiting for the floor read: %s is absent" % fj)
    try:
        fst = json.loads(fj.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return fin("WAIT", "%s is unreadable (being written?)" % fj)
    if fst != "OK":
        return fin("ABORT", "FLOOR.json status %r is not OK: no arm is read against it" % fst)
    s, why, line, run = _arm_state(st, arm, seg, lines, root)
    if s != "READY":
        return fin(s, why, line, run)
    if ref and ref not in FLOOR_ARMS:
        s2, why2, line2, run2 = _arm_state(st, ref, seg, lines, root)
        if s2 != "READY":
            return fin(s2, "reference arm: " + why2, line2, run, run2)
        return fin("READY", why + "; reference " + why2, line, run, run2)
    return fin("READY", why, line, run)


def check_dumps(paths: list) -> list:
    """The same checks r9e1_read.sh runs on C0 / C1, over any number of dumps. -> list of problems (empty = OK)."""
    ds = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    bad = []
    for k in ("order_digest", "n_windows", "n_clips", "step"):
        vals = [d.get(k) for d in ds]
        if len(set(map(str, vals))) != 1:
            bad.append("%s differs: %s" % (k, vals))
    for p, d in zip(paths, ds):
        if d.get("weights_are_ema") is not True or d.get("ckpt_kind") != "ema_model_only":
            bad.append("%s: weights_are_ema=%r ckpt_kind=%r (need an EMA file)" % (p, d.get("weights_are_ema"), d.get("ckpt_kind")))
        if d.get("SMOKE_RESTRICTED") not in (False, None):
            bad.append("%s is SMOKE_RESTRICTED" % p)
    return bad


# ======================================================================================================================== CLI
def _state_cli(argv) -> int:
    ap = argparse.ArgumentParser(prog="r9_e1_verdict.py state")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--status", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        resolve_ref(a.arm, a.ref)
    except Refused as e:
        d = {"state": "ABORT", "reason": str(e), "line": "", "run": "", "ref_run": "", "sig": "ABORT|" + a.arm}
    else:
        p = Path(a.status)
        d = arm_state(a.arm, a.ref or default_ref(a.arm), p.read_text(encoding="utf-8", errors="replace") if p.is_file() else "", a.root, a.read_dir)
    body = "".join("S_%s=%s\n" % (k.upper(), shlex.quote(str(d[k]))) for k in ("state", "reason", "line", "run", "ref_run", "sig"))
    tmp = Path(a.out + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(a.out)
    print(d["state"], d["reason"])
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "state":
        return _state_cli(argv[1:])
    if argv and argv[0] == "mapping":
        print(json.dumps({a: _mapping_entry(a) for a in ARMS}, indent=1) if "--json" in argv else mapping_md(), end="")
        return 0
    if argv and argv[0] == "ref":
        ap = argparse.ArgumentParser(prog="r9_e1_verdict.py ref")
        ap.add_argument("--arm", required=True)
        a = ap.parse_args(argv[1:])
        try:
            r = resolve_ref(a.arm, None)
        except Refused as e:
            print("[r9-e1-verdict] %s" % e, file=sys.stderr)
            return 3
        print(r)
        return 0
    if argv and argv[0] == "dumpcheck":
        ap = argparse.ArgumentParser(prog="r9_e1_verdict.py dumpcheck")
        ap.add_argument("--dump", action="append", required=True)
        a = ap.parse_args(argv[1:])
        bad = check_dumps(a.dump)
        print("; ".join(bad) or "OK")
        return 1 if bad else 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--floor", required=True, help="FLOOR.json (r9_e1_floor.py)")
    ap.add_argument("--paired", required=True, help="r9_e1_eval.py --packs <ARM dump> --packs-b <REF dump> output (delta = ARM - REF)")
    ap.add_argument("--eval-arm", required=True, help="r9_e1_eval.py single-arm output for the arm")
    ap.add_argument("--eval-ref", required=True, help="r9_e1_eval.py single-arm output for the reference")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", help="the reference arm (default: REFCV9_DESIGN 4.3's)")
    ap.add_argument("--rast-arm")
    ap.add_argument("--rast-ref")
    ap.add_argument("--rast-names")
    ap.add_argument("--rast-stride", type=int, default=1)
    ap.add_argument("--psw-key", default="psw")
    ap.add_argument("--floor-eval-c0", help="the floor's own C0 single eval (window file / split are compared against it)")
    ap.add_argument("--lift-valid", help="the ARM-INDEPENDENT lift-validity mask of the MAP PRIMARY (nf_swedge_prec_lv_*; r9_e1_eval.load_lift_valid): a sidecar "
                                         ".npz or a raster dir of an unobs-writing arm.  Its sha256 must equal the FLOOR's map_inputs.lift_valid (one side with a "
                                         "mask and the other without REFUSES); absent -> the map primary is read only if the floor has the row (it cannot without a mask)")
    ap.add_argument("--map-thresholds-arm", help="the ARM's OWN FIT-split map thresholds (r9_map_fit_thresholds.py fit); both -arm and -ref or neither. The floor "
                                                 "must have been read on the same protocol (each arm on its own FIT cuts) or this verdict REFUSES")
    ap.add_argument("--map-thresholds-ref", help="the REFERENCE arm's OWN FIT-split map thresholds")
    ap.add_argument("--map-beside-d3", action="store_true",
                    help="also count M1's two design beside rows on the rasters (lane tolerance-IoU, drivable-boundary recall @0.3 m; hold_r9_d3beside): "
                         "clause3=False, never decide; unset = byte-identical")
    ap.add_argument("--boot", type=int, help="must equal the floor's B (default: the floor's)")
    ap.add_argument("--seed", type=int, help="must equal the floor's seed (default: the floor's)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    v = build_verdict(a.floor, a.paired, a.eval_arm, a.eval_ref, a.arm, a.ref, rast_arm=a.rast_arm, rast_ref=a.rast_ref, rast_names=a.rast_names,
                      rast_stride=a.rast_stride, psw_key=a.psw_key, floor_eval_c0=a.floor_eval_c0, boot=a.boot, seed=a.seed,
                      lift_valid=a.lift_valid, thr_arm=a.map_thresholds_arm, thr_ref=a.map_thresholds_ref, beside_d3=a.map_beside_d3)
    stem = "VERDICT_%s" % a.arm
    (od / (stem + ".json")).write_text(json.dumps(v, indent=1, default=float, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (od / (stem + ".md")).write_text(render_md(v), encoding="utf-8", newline="\n")
    if v["status"] != "OK":
        print("[r9-e1-verdict] REFUSED: %s" % "; ".join(v["problems"]), file=sys.stderr)
        return 2
    print("wrote %s: arm %s vs %s -> %s%s" % (od / (stem + ".json"), a.arm, v["ref"], v["verdict"],
                                             "".join("; clause %d" % f["clause"] for f in v["failing_clauses"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
