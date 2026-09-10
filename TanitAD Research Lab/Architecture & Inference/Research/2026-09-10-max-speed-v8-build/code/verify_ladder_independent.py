"""⛔ THE INDEPENDENT CROSS-CHECK. It must NOT import the builder's ladder.

⭐⭐ WHY THIS FILE EXISTS AND WHY IT REFUSES TO IMPORT ANYTHING FROM
``max_speed_input``: *"A CROSS-CHECK MUST BE DERIVED INDEPENDENTLY OF THE VALUE
IT CHECKS. Re-running the producer's own derivation and finding agreement
measures DETERMINISM, NOT CORRECTNESS."*

MEASURED 2026-09-07, and it is the exact defect this file guards: a builder
rounded its ladder to 4 dp (``round(50/3.6, 4) = 13.8889`` for a step that is
``13.888888...``), then verified the shipped buckets **against that same rounded
ladder**. Re-snapping read **0 % moved** and the artifact passed its own build
gate. Against a ladder a consumer would naturally derive from the km/h integers,
**57.5 % of the corpus (2,631/4,572) moves one step up.**

⇒ Three independence properties, all deliberate:

1. **The ladder is TYPED HERE from road law**, as km/h integers. Nothing is
   imported from ``tanitad.refs.max_speed_input``. If the builder re-pins its
   ladder, this file does NOT follow and the check goes RED -- which is the
   point.
2. **The arithmetic is EXACT RATIONAL** (:class:`fractions.Fraction` over the
   JSON token's own decimal text), so there is no float rounding anywhere in the
   comparison and no 4-dp surface for the defect to live on. This is the
   "analytic target" discriminator, the strongest of the three.
3. **The comparison is in km/h space** (``v_kmh <= k``), while the builder snaps
   in **m/s space** (``v_ms <= k/3.6``). Two different spaces, two different
   authors; agreement is then evidence.

And it ships its own **deliberate-regression arm** (:func:`mutant_rounded_ladder`)
which reintroduces the real historical 4-dp defect and MUST move a large share of
the corpus -- because a check that cannot go RED has never been shown to work.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

# ⛔ TYPED FROM ROAD LAW, NOT IMPORTED. These are the posted limits of the
# corpus's 24 European countries + the United States, as INTEGER km/h.
LADDER_KMH: tuple[int, ...] = (20, 30, 50, 70, 80, 100, 120, 130)

#: m/s per km/h, EXACTLY. 1 km/h = 1000 m / 3600 s = 5/18 m/s.
KMH_PER_MS = Fraction(18, 5)          # multiply m/s by this to get km/h

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def exact_kmh(v_ms_text: str) -> Fraction:
    """The speed in km/h as an EXACT rational, from the JSON token's own text.

    ⛔ Takes the decimal STRING, never a float: ``Fraction("11.19")`` is exactly
    1119/100, while ``Fraction(11.19)`` is the binary float's true value, which
    is not 11.19. The whole point of this file is to have no rounding in it.
    """
    if not _NUM.match(v_ms_text):
        raise ValueError(f"not a plain decimal number: {v_ms_text!r}")
    return Fraction(v_ms_text) * KMH_PER_MS


def snap_up_exact(v_ms_text: str) -> tuple[int, bool]:
    """CEILING, independently: ``min{k : k >= v_kmh}``, exact. Returns (k, over)."""
    v = exact_kmh(v_ms_text)
    for k in LADDER_KMH:
        if v <= k:
            return k, False
    return LADDER_KMH[-1], True


def snap_down_exact(v_ms_text: str) -> tuple[int, bool]:
    """FLOOR, independently: ``max{k : k <= v_kmh}``, exact. Returns (k, under)."""
    v = exact_kmh(v_ms_text)
    best = None
    for k in LADDER_KMH:
        if Fraction(k) <= v:
            best = k
    if best is None:
        return LADDER_KMH[0], True
    return best, False


def mutant_rounded_ladder_on_raw(v_ms_text: str, dp: int = 4) -> int:
    """⚠️ ARM 1 -- and MEASURED INERT on this corpus, which is itself the finding.

    Snap the RAW ``v_hi_ms`` against a ladder whose steps were rounded to ``dp``
    decimals. ``round(50/3.6, 4) = 13.8889`` is very slightly LARGER than the
    true step ``13.888888...``, so this arm can only move a value lying inside
    the sliver ``(13.888888..., 13.8889]`` -- about **1.2e-5 m/s wide**. The
    corpus's ``v_hi_ms`` is emitted at **2 decimal places**
    (``s2_geom_emit_v7.py:311`` ``round(float(v.max()), 2)``), so no value can
    land in it and this arm moves **0** clips.

    ⛔ THAT IS NOT A LICENCE TO CALL THE BUILD VERIFIED. It is the opposite: an
    arm that reads 0 has proven nothing, and reporting it as a passed mutation
    would be the "green forever" failure. It is kept because it **localises the
    real defect** -- the historical failure was never in the snap, it was in
    quantizing TWICE (:func:`mutant_double_quantize`), and this arm is what
    shows the difference.
    """
    steps = [(round(k / 3.6, dp), k) for k in LADDER_KMH]
    v = float(v_ms_text)
    for s, k in steps:
        if v <= s:
            return k
    return LADDER_KMH[-1]


def mutant_double_quantize(v_ms_text: str, dp: int = 4) -> int:
    """⛔⛔ ARM 2 -- THE REAL 2026-09-07 DEFECT, REPRODUCED. Must go RED.

    The historical builder shipped a ``v_max_bucket_ms`` **rounded to 4 dp** and
    the consumer then snapped **that shipped bucket** through the ladder again.
    Because ``round(k/3.6, 4)`` is strictly GREATER than ``k/3.6`` for
    k in {20, 50, 100} (and strictly smaller for the other five steps), the
    re-snap pushes exactly those three buckets one step up.

    ⭐ This arm predicts the published breakdown **without being told it**:
    50->70, 20->30, 100->120, and nothing else. If the counts it prints
    reproduce ``2,631 / 4,572 (57.5 %)``, an independently authored instrument
    has re-derived a published number -- which is the strongest form the
    cross-check can take.
    """
    v = float(v_ms_text)
    # step 1: snap to a step, and ship it ROUNDED to dp -- the defect's surface
    shipped_ms = None
    for k in LADDER_KMH:
        if v <= k / 3.6:
            shipped_ms = round(k / 3.6, dp)
            break
    if shipped_ms is None:
        shipped_ms = round(LADDER_KMH[-1] / 3.6, dp)
    # step 2: the consumer re-snaps the SHIPPED bucket through the exact ladder
    for k in LADDER_KMH:
        if shipped_ms <= k / 3.6:
            return k
    return LADDER_KMH[-1]


def _raw_text_fields(line: str) -> dict:
    """Re-parse a JSON line keeping the DECIMAL TEXT of the two band bounds.

    ``json.loads`` would hand back floats and the exactness would be gone before
    the comparison started, so the numbers are lifted out of the raw text.
    """
    obj = json.loads(line)
    sb = ((obj.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND") or {}
    out = {"clip_id": obj.get("clip_id"),
           "block": obj.get("speed_max_input"),
           "held": sb.get("held")}
    for key in ("v_hi_ms", "v_lo_ms"):
        m = re.search(r'"%s"\s*:\s*(-?\d+(?:\.\d+)?)' % key, line)
        out[key] = m.group(1) if m else None
    return out


def verify(path: Path) -> dict:
    """Score one shipped v8 blob against the independently authored ladder."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]

    n = len(lines)
    # ⭐ SAME-BREATH POSITIVE CONTROLS. Every zero below is a claim about the
    # CONTENT only because these read non-zero in the same pass.
    ctl_clip = ctl_band = 0
    n_block = n_units = n_ceiling = 0
    hi_match = hi_mismatch = lo_match = lo_mismatch = 0
    over_match = under_match = 0
    mut1_moved = 0
    mut2_moved = 0
    mut1_moves, mut2_moves = Counter(), Counter()
    bad_units: list[str] = []
    has_bucket_ms: list[str] = []
    mismatch_examples: list[dict] = []

    for ln in lines:
        r = _raw_text_fields(ln)
        if r["clip_id"]:
            ctl_clip += 1
        if r["v_hi_ms"] is not None and r["v_lo_ms"] is not None:
            ctl_band += 1
        blk = r["block"]
        if not isinstance(blk, dict):
            continue
        n_block += 1
        # ⛔ UNITS MUST BE ON THE WIRE.
        u = blk.get("units", blk.get("control_units"))
        if u is None:
            bad_units.append(str(r["clip_id"]))
        else:
            n_units += 1
        if blk.get("v_max_ms") is not None:
            n_ceiling += 1
        # ⛔ NO m/s BUCKET MAY EXIST -- that key is the 57.5 % defect's surface.
        if any(k in blk for k in ("v_max_bucket_ms", "v_min_bucket_ms")):
            has_bucket_ms.append(str(r["clip_id"]))

        k_hi, over = snap_up_exact(r["v_hi_ms"])
        k_lo, under = snap_down_exact(r["v_lo_ms"])
        if int(blk.get("v_max_bucket_kmh", -1)) == k_hi:
            hi_match += 1
        else:
            hi_mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {"clip": r["clip_id"], "v_hi_ms": r["v_hi_ms"],
                     "shipped": blk.get("v_max_bucket_kmh"), "exact": k_hi})
        if int(blk.get("v_min_bucket_kmh", -1)) == k_lo:
            lo_match += 1
        else:
            lo_mismatch += 1
        over_match += int(bool(blk.get("over_ceiling")) == over)
        under_match += int(bool(blk.get("under_floor")) == under)

        # the two deliberate-regression arms, on the SAME corpus
        km1 = mutant_rounded_ladder_on_raw(r["v_hi_ms"])
        if km1 != k_hi:
            mut1_moved += 1
            mut1_moves[f"{k_hi}->{km1}"] += 1
        km2 = mutant_double_quantize(r["v_hi_ms"])
        if km2 != k_hi:
            mut2_moved += 1
            mut2_moves[f"{k_hi}->{km2}"] += 1

    return {
        "path": str(path), "n_records": n,
        "CONTROL_clip_id_present": ctl_clip,
        "CONTROL_speed_band_present": ctl_band,
        "n_with_speed_max_input_block": n_block,
        "coverage": round(n_block / max(n, 1), 6),
        "n_units_declared": n_units,
        "n_units_MISSING": len(bad_units),
        "n_with_ceiling_value": n_ceiling,
        "n_records_carrying_an_ms_BUCKET_must_be_0": len(has_bucket_ms),
        "ceiling_bucket_MATCH_exact_ladder": hi_match,
        "ceiling_bucket_MISMATCH": hi_mismatch,
        "floor_bucket_MATCH_exact_ladder": lo_match,
        "floor_bucket_MISMATCH": lo_mismatch,
        "over_ceiling_flag_MATCH": over_match,
        "under_floor_flag_MATCH": under_match,
        "mismatch_examples": mismatch_examples,
        # ⚠️ ARM 1 -- measured INERT on a 2-dp corpus, and reported as inert
        # rather than as a passed mutation. See its docstring.
        "MUTANT1_rounded_ladder_on_raw_moved": mut1_moved,
        "MUTANT1_moves": dict(mut1_moves.most_common()),
        "MUTANT1_is_RED": mut1_moved > 0,
        # ⛔ ARM 2 -- THE REAL HISTORICAL DEFECT. This one must go RED.
        "MUTANT2_double_quantize_moved": mut2_moved,
        "MUTANT2_double_quantize_moved_frac": round(mut2_moved / max(n, 1), 6),
        "MUTANT2_moves": dict(mut2_moves.most_common()),
        "MUTANT2_is_RED": mut2_moved > 0,
        "VERDICT_all_buckets_agree": (hi_mismatch == 0 and lo_mismatch == 0
                                      and n > 0 and hi_match == n),
        # ⛔ the gate rides on ARM 2 only: arm 1 reading 0 is a finding, not a
        # proof, and a verdict resting on it would be green forever.
        "VERDICT_mutant_is_RED": mut2_moved > 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("blobs", nargs="+")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    rep = {"instrument": "verify_ladder_independent.py",
           "ladder_kmh_typed_here": list(LADDER_KMH),
           "arithmetic": "exact rational (fractions.Fraction) over the JSON "
                         "decimal text; NO float rounding in the comparison",
           "imports_from_max_speed_input": "NONE (deliberate)",
           "splits": {Path(b).name: verify(Path(b)) for b in a.blobs}}
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    ok = all(s["VERDICT_all_buckets_agree"] and s["VERDICT_mutant_is_RED"]
             for s in rep["splits"].values())
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
