#!/usr/bin/env python3
"""Extract the (v0, v_hi) pair for EVERY v8 clip -- the table every downstream
number in this package is computed from.

WHY A SEPARATE EXTRACTION. The v8 label record carries `speed_max_input.v_max_ms`
(= `g_tac.goals.SPEED_BAND.v_hi_ms`) but carries NO `v0`: the ego speed at the
anchor lives only in the egomotion source. Every claim about "does the channel
echo the ego" is a claim about the JOIN of those two, so the join is built once,
banked, content-asserted, and reused.

UNITS. `v_max_ms` declares `units: "m/s"` in the artifact. `egomotion_source.load`
returns `poses[:, 3] = |v|` in m/s (norm of vx,vy,vz, provider SI). Both m/s. The
extraction re-states this in its own output so a consumer cannot infer it.

ANCHOR. `egomotion_source.RAW_T0_S = 8.0` s at HZ = 10 -> `key_index = 80`.
`s2_geom_emit_v7.tactical_goals` computes `v_hi = max(p[key+20 : key+60, 3])`
(TACTICAL_S = 2..6 s). This script RE-DERIVES v_hi from the poses and checks it
against the label's value: that is the content assertion. If the re-derivation
does not reproduce the label to within rounding, the join is wrong and the run
FAILS -- a silently mis-joined table would make every echo number meaningless
(C140: 20.5 % of a validation was the wrong episode).

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")

from tanitad.data import egomotion_source as ES  # noqa: E402

HZ = 10.0
TACTICAL_S = (2.0, 6.0)


def run(labels_gz: Path, out_json: Path) -> dict:
    rows = [json.loads(x) for x in gzip.open(labels_gz, "rt", encoding="utf-8")
            if x.strip()]
    out, missing, mismatch = [], [], []
    for r in rows:
        cid = r["clip_id"]
        smi = r.get("speed_max_input") or {}
        v_hi_label = smi.get("v_max_ms")
        # UNITS REFUSAL AT THE SOURCE: a record that declares nothing is skipped
        # loudly, never defaulted. Same rule as `anchor_meta.read_anchor_artifact`.
        if v_hi_label is None or smi.get("units") != "m/s":
            missing.append((cid, "no v_max_ms or undeclared units"))
            continue
        try:
            tr = ES.load(cid)
        except Exception as exc:                      # noqa: BLE001
            missing.append((cid, f"no egomotion: {type(exc).__name__}"))
            continue
        p = tr.poses
        key = tr.key_index
        lo = min(len(p) - 1, key + int(round(TACTICAL_S[0] * HZ)))
        hi = min(len(p) - 1, key + int(round(TACTICAL_S[1] * HZ)))
        if hi <= lo or key >= len(p):
            missing.append((cid, f"track too short T={len(p)} key={key}"))
            continue
        seg = p[lo:hi + 1, 3]
        v_hi_re = float(seg.max())
        v_lo_re = float(max(0.0, seg.min()))
        v0 = float(p[key, 3])
        # CONTENT ASSERTION: the re-derivation must reproduce the label.
        if abs(round(v_hi_re, 2) - float(v_hi_label)) > 0.011:
            mismatch.append((cid, round(v_hi_re, 3), float(v_hi_label)))
            continue
        # ⛔ FULL PRECISION, NOT 4 dp. MEASURED: storing `v_hi_ms` at 4 dp rounds
        # it DOWN on ~48 % of clips by up to 4.995e-05 m/s, and the downstream
        # constraint check -- "the ego can never exceed its own realised max" --
        # then fired on 2,203 of 4,572 clips. A programme-standard DISPLAY
        # precision silently became a DATA precision and manufactured a
        # violation rate out of nothing. Round at the point of printing, never
        # at the point of storing.
        out.append({
            "clip_id": cid,
            "v0_ms": float(v0),
            "v_hi_ms": float(v_hi_re),
            "v_lo_ms": float(v_lo_re),
            "v_hi_label_ms": float(v_hi_label),
            "road_class": (r.get("strata") or {}).get("road_class"),
            "country": (r.get("strata") or {}).get("country"),
            "split": r.get("split"),
            "scene": r.get("scene"),
        })
    doc = {
        "units": "m_s",
        "units_note": ("EVERY speed field in this file is METRES PER SECOND. "
                       "Declared, not inferred: m/s vs km/h vs mph is a 1.61x "
                       "spread and this programme published a 396 g anchor "
                       "table from a units error."),
        "anchor": {"raw_t0_s": ES.RAW_T0_S, "hz": HZ,
                   "key_index": int(round(ES.RAW_T0_S * HZ)),
                   "tactical_band_s": list(TACTICAL_S)},
        "source_labels": str(labels_gz),
        "n_records": len(rows),
        "n_joined": len(out),
        "n_missing": len(missing),
        "n_mismatch": len(mismatch),
        "missing_head": missing[:10],
        "mismatch_head": mismatch[:10],
        "rows": out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


if __name__ == "__main__":
    d = run(Path(sys.argv[1]), Path(sys.argv[2]))
    print("records   %d" % d["n_records"])
    print("joined    %d  (%.2f%%)" % (d["n_joined"],
                                      100.0 * d["n_joined"] / max(1, d["n_records"])))
    print("missing   %d   mismatch %d" % (d["n_missing"], d["n_mismatch"]))
    if d["n_joined"]:
        a = np.array([r["v0_ms"] for r in d["rows"]])
        b = np.array([r["v_hi_ms"] for r in d["rows"]])
        print("v0    mean %.3f  median %.3f  max %.3f" % (a.mean(), np.median(a), a.max()))
        print("v_hi  mean %.3f  median %.3f  max %.3f" % (b.mean(), np.median(b), b.max()))
    for c, got, want in d["mismatch_head"]:
        print("MISMATCH %s re=%.3f label=%.3f" % (c, got, want))
