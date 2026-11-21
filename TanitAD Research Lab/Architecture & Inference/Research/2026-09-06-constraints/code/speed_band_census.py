#!/usr/bin/env python3
"""C-ENV-1 step 1 -- WHAT IS `g_tac.goals.SPEED_BAND` ACTUALLY?

The PI asked (instruction 10) for a MAX SPEED constraint the model adapts to.
A speed ENVELOPE is already minted on 4,572/4,572 v7.2 train records as
``g_tac.goals.SPEED_BAND = {v_lo_ms, v_hi_ms, band_s, grounded, grounding,
held}`` -- and ``tanitad.data.v7_labels._goal_audit`` keeps only
``(disputed, time_basis, t_nominal_s, held, grounded, grounding)``, so
``v_lo_ms``/``v_hi_ms`` do not reach even the audit dict, let alone a loss.

Before anyone proposes feeding it, this asks the only question that decides
admissibility: **is the band a CONSTRAINT (what the road allows) or a
DESCRIPTION (what the ego did)?** A band that brackets the ego's own realised
speed is an ego echo -- the nav-echo defect in a speed costume -- and supplying
it at inference would be inadmissible for exactly the reason
``vtarget_guarded`` supplied at inference is.

⛔ NO MODEL IS RUN AND NOTHING IS SCORED. This is a label census. Every number
is a property of the label file, named with its `n` and its denominator.

CONTROLS (this repo's rule: a probe that tunes on what it scores manufactures a
result, and only a control that must read a KNOWN value catches it):
  * ``n_records`` must equal the release count or the read is INCONCLUSIVE.
  * ``SPEED_BAND`` presence must be 4572/4572; anything else means the blob
    moved and every downstream number is void.
  * the band-width histogram must not be single-valued -- a constant band would
    mean the mint is a stub, and a stub reads exactly like a real envelope.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(int(q * (len(xs) - 1)), len(xs) - 1)
    return round(xs[i], 4)


def main(blob: Path, out_json: Path):
    md5 = hashlib.md5(blob.read_bytes()).hexdigest()
    recs = []
    with gzip.open(blob, "rt", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    n = len(recs)

    have_sb = 0
    lo, hi, width, band_lo, band_hi = [], [], [], [], []
    grounded = Counter()
    grounding = Counter()
    held = Counter()
    lon_tok = Counter()
    vtgt = []
    vtgt_vs_band = Counter()
    lon_by_band_pos = {}
    inverted = 0
    zero_lo = 0

    for r in recs:
        g = (r.get("g_tac") or {}).get("goals") or {}
        sb = g.get("SPEED_BAND")
        a_tac = r.get("a_tac") or {}
        tok = a_tac.get("lon")
        lon_tok[tok] += 1
        vt = ((a_tac.get("lon_args") or {}).get("v_target_ms"))
        if vt is not None:
            vtgt.append(float(vt))
        if not sb:
            continue
        have_sb += 1
        vlo, vhi = sb.get("v_lo_ms"), sb.get("v_hi_ms")
        grounded[sb.get("grounded")] += 1
        grounding[str(sb.get("grounding"))[:60]] += 1
        held[sb.get("held")] += 1
        bs = sb.get("band_s") or [None, None]
        band_lo.append(bs[0])
        band_hi.append(bs[1])
        if vlo is None or vhi is None:
            continue
        vlo, vhi = float(vlo), float(vhi)
        lo.append(vlo)
        hi.append(vhi)
        width.append(vhi - vlo)
        if vhi < vlo:
            inverted += 1
        if vlo == 0.0:
            zero_lo += 1
        if vt is not None:
            v = float(vt)
            pos = ("below_lo" if v < vlo - 1e-9 else
                   "above_hi" if v > vhi + 1e-9 else "inside")
            vtgt_vs_band[pos] += 1
            lon_by_band_pos.setdefault(tok, Counter())[pos] += 1

    out = {
        "_what": ("census of g_tac.goals.SPEED_BAND on the banked v7.2 label "
                  "blob -- is the envelope a CONSTRAINT or a DESCRIPTION?"),
        "_evidence_class": "MEASURED (ours; artifact = this json + the blob md5)",
        "_no_model_was_run": True,
        "blob": str(blob),
        "blob_md5": md5,
        "n_records": n,
        "controls": {
            "speed_band_present": f"{have_sb}/{n}",
            "band_width_is_single_valued": (len(set(round(w, 3) for w in width)) == 1
                                            if width else None),
            "n_distinct_band_widths": len(set(round(w, 3) for w in width)),
            "inverted_bands_v_hi_lt_v_lo": inverted,
            "_reads": ("a single-valued width would mean the mint is a STUB; a "
                       "stub reads exactly like a real envelope in a mean"),
        },
        "band_s": {
            "distinct_lo": sorted(set(band_lo))[:5],
            "distinct_hi": sorted(set(band_hi))[:5],
        },
        "v_lo_ms": {"n": len(lo), "mean": round(st.mean(lo), 4),
                    "p05": pct(lo, .05), "p50": pct(lo, .50), "p95": pct(lo, .95),
                    "min": round(min(lo), 4), "max": round(max(lo), 4),
                    "n_exactly_zero": zero_lo},
        "v_hi_ms": {"n": len(hi), "mean": round(st.mean(hi), 4),
                    "p05": pct(hi, .05), "p50": pct(hi, .50), "p95": pct(hi, .95),
                    "min": round(min(hi), 4), "max": round(max(hi), 4)},
        "band_width_ms": {"n": len(width), "mean": round(st.mean(width), 4),
                          "p05": pct(width, .05), "p50": pct(width, .50),
                          "p95": pct(width, .95),
                          "min": round(min(width), 4), "max": round(max(width), 4)},
        "grounded": dict(grounded),
        "grounding_reason": dict(grounding.most_common(5)),
        "held": dict(held),
        "a_tac_lon_tokens": dict(lon_tok.most_common()),
        "v_target_ms": {"n": len(vtgt), "mean": round(st.mean(vtgt), 4),
                        "p05": pct(vtgt, .05), "p50": pct(vtgt, .50),
                        "p95": pct(vtgt, .95)},
        "v_target_vs_SPEED_BAND": dict(vtgt_vs_band),
        "v_target_vs_band_by_lon_token": {
            k: dict(v) for k, v in sorted(lon_by_band_pos.items())},
        "_reads": ("v_target_ms is round(m.v_min,2) -- the MINIMUM speed over "
                   "the plan window (s2_geom_emit_v7.tactical_actions), NOT a "
                   "desired speed. If it lands INSIDE [v_lo, v_hi] on most "
                   "records, the two are the same ego-geometry statistic in two "
                   "costumes, and neither is a road constraint."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"n_records={n}  SPEED_BAND={have_sb}/{n}  md5={md5}")
    print(f"v_lo  mean={out['v_lo_ms']['mean']:.3f} p05={out['v_lo_ms']['p05']} "
          f"p50={out['v_lo_ms']['p50']} p95={out['v_lo_ms']['p95']} "
          f"zeros={zero_lo}")
    print(f"v_hi  mean={out['v_hi_ms']['mean']:.3f} p05={out['v_hi_ms']['p05']} "
          f"p50={out['v_hi_ms']['p50']} p95={out['v_hi_ms']['p95']}")
    print(f"width mean={out['band_width_ms']['mean']:.3f} "
          f"p05={out['band_width_ms']['p05']} p50={out['band_width_ms']['p50']} "
          f"p95={out['band_width_ms']['p95']} distinct={out['controls']['n_distinct_band_widths']}")
    print(f"grounded={dict(grounded)}  held={dict(held)}")
    print(f"v_target vs band: {dict(vtgt_vs_band)}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
