"""P2 - the class-G census of EVERY filter on the refcv6 BOX-GT path, on the
**v7/B1 line** (NOT the parity line).

WHY IT HAD TO BE RE-MEASURED. ``refc_agents.filter_targets_to_visible``'s own
docstring (refc_agents.py:361-367) publishes ``in-field 4,977,314 = 41.06 %``
and ``in-field n decode box 1,899,481 = 15.67 %`` over "the whole 2,308-clip
train join (12,122,129 boxes)". That is the **PARITY** line. refcv6 trains the
**B1** line, whose join is ``b1_train_plus_eval_agents.jsonl.xz`` (4,566 clips).
Quoting one line's fraction for the other is the retraction class this
programme logged today, so this pass reads the B1 join itself.

WHAT IT COUNTS, per the advisory's class-G instruction ("count what it removes,
and test whether the removal correlates with anything"):

  F1  FOV / azimuth   atan2(|cy|, cx) <= 60 deg    (refc_agents.py:385)
  F2  decode box      0 <= cx <= 60, |cy| <= 16    (refc_agents.py:416, from
                                                    SlotDecodeRanges)
  F3  vocabulary      cls in AGENT_CLASSES         (agent_slots.py:196 -> -1)
  F4  --agent-pad     nearest-N truncation         (refc_v3_train.py:2707)
  J   the join's own `occ` flag, cross-checked against F1

and breaks every one down BY CLASS, BY RANGE and BY CLIP so a biased removal is
distinguishable from a uniform one.

Clip ids are emitted as sha256(clip_id)[:12] ONLY.

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p2_b1_join_filter_census.py \
      --join D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import lzma
import math
import pathlib
import time

import numpy as np

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p2_b1_join_filter_census.json"

# The two cuts, read from the code that applies them -- never re-typed.
HALF = float(FOV_HALF_ANGLE_RAD)
X_MAX = float(GRID_DEFAULT.x_fwd_m)
Y_HALF = float(GRID_DEFAULT.y_half_m)
RANGE_BINS = (0.0, 20.0, 40.0, 60.0, 80.0, 120.0, float("inf"))


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--pad", type=int, default=100,
                    help="AgentSeamConfig.queries / --agent-pad (M17 ruled 100)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    vocab = set(ALL_CLASSES)
    t0 = time.time()
    n_lines = n_agents = 0
    n_infield = n_box = n_infield_box = 0
    n_occ0 = n_occ1 = n_occm = 0
    n_occ_agrees = n_occ_disagrees = 0
    n_oov = 0
    n_behind = 0
    n_trunc_lines = n_trunc_boxes = 0
    by_cls = collections.defaultdict(lambda: [0, 0, 0])      # tot, infield, infield&box
    by_rng = collections.defaultdict(lambda: [0, 0])          # tot, infield
    by_clip = {}                                              # sha12 -> [lines, tot, ifb]
    per_frame_counts = collections.Counter()

    opener = lzma.open if str(a.join).endswith(".xz") else open
    with opener(a.join, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ags = d.get("agents")
            if ags is None:
                continue
            n_lines += 1
            cs = sha12(str(d["clip_id"]))
            rec = by_clip.setdefault(cs, [0, 0, 0])
            rec[0] += 1
            n = len(ags)
            per_frame_counts[min(n, 200)] += 1
            if n > a.pad:
                n_trunc_lines += 1
                n_trunc_boxes += n - a.pad
            for g in ags:
                cx = g["cx"]; cy = g["cy"]
                cls = g.get("cls")
                occ = g.get("occ", -1)
                n_agents += 1
                rec[1] += 1
                inf = math.atan2(abs(cy), cx) <= HALF
                inbx = (0.0 <= cx <= X_MAX) and (abs(cy) <= Y_HALF)
                if cx < 0.0:
                    n_behind += 1
                if inf:
                    n_infield += 1
                if inbx:
                    n_box += 1
                if inf and inbx:
                    n_infield_box += 1
                    rec[2] += 1
                if occ == 0:
                    n_occ0 += 1
                elif occ == 1:
                    n_occ1 += 1
                else:
                    n_occm += 1
                if occ in (0, 1):
                    # the join doc: occ == 1 means OUTSIDE the field (~fov_mask)
                    if (occ == 0) == inf:
                        n_occ_agrees += 1
                    else:
                        n_occ_disagrees += 1
                c = by_cls[str(cls)]
                c[0] += 1
                c[1] += int(inf)
                c[2] += int(inf and inbx)
                if str(cls) not in vocab:
                    n_oov += 1
                r = math.hypot(cx, cy)
                for i in range(len(RANGE_BINS) - 1):
                    if RANGE_BINS[i] <= r < RANGE_BINS[i + 1]:
                        b = by_rng[f"{RANGE_BINS[i]:g}-{RANGE_BINS[i+1]:g} m"]
                        b[0] += 1
                        b[1] += int(inf)
                        break
            if a.limit_lines and n_lines >= a.limit_lines:
                break

    # ---- the correlation test class G asks for: per-clip keep fraction ----- #
    fr = np.array([rec[2] / rec[1] for rec in by_clip.values() if rec[1] > 0])
    ln = np.array([rec[0] for rec in by_clip.values() if rec[1] > 0], dtype=float)
    dens = np.array([rec[1] / max(rec[0], 1) for rec in by_clip.values()
                     if rec[1] > 0])
    def _corr(x, y):
        if len(x) < 3 or float(np.std(x)) == 0 or float(np.std(y)) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "v7-B1 (b1_train_plus_eval_agents.jsonl.xz) -- NOT the parity line",
        "join": str(a.join),
        "join_bytes": pathlib.Path(a.join).stat().st_size,
        "cuts_read_from_code": {
            "fov_half_deg": math.degrees(HALF),
            "fov_src": "refc_agents.FOV_HALF_ANGLE_RAD",
            "x_max_m": X_MAX, "y_half_m": Y_HALF,
            "box_src": "bev_raster.GRID_DEFAULT (== SlotDecodeRanges defaults)",
            "pad": a.pad,
        },
        "totals": {
            "n_lines": n_lines, "n_clips": len(by_clip), "n_agents": n_agents,
            "n_infield": n_infield,
            "frac_infield": n_infield / max(n_agents, 1),
            "n_in_decode_box": n_box,
            "frac_in_decode_box": n_box / max(n_agents, 1),
            "n_infield_and_box": n_infield_box,
            "frac_infield_and_box": n_infield_box / max(n_agents, 1),
            "n_behind_ego_cx_lt_0": n_behind,
            "frac_behind_ego": n_behind / max(n_agents, 1),
            "n_out_of_vocabulary": n_oov,
            "frac_out_of_vocabulary": n_oov / max(n_agents, 1),
        },
        "join_occ_flag": {
            "n_occ_0": n_occ0, "n_occ_1": n_occ1, "n_occ_missing": n_occm,
            "agrees_with_computed_fov": n_occ_agrees,
            "disagrees_with_computed_fov": n_occ_disagrees,
            "frac_agree": n_occ_agrees / max(n_occ_agrees + n_occ_disagrees, 1),
        },
        "agent_pad_truncation": {
            "pad": a.pad, "n_lines_over_pad": n_trunc_lines,
            "n_boxes_truncated": n_trunc_boxes,
            "frac_lines_over_pad": n_trunc_lines / max(n_lines, 1),
            "frac_boxes_truncated": n_trunc_boxes / max(n_agents, 1),
            "note": "PRE-filter truncation: refc_v3_train.py:2707 keeps the "
                    "NEAREST pad of the RAW join row, before any visibility cut",
        },
        "by_class": {k: {"n": v[0], "n_infield": v[1], "n_infield_box": v[2],
                         "frac_infield": v[1] / max(v[0], 1),
                         "frac_infield_box": v[2] / max(v[0], 1),
                         "in_vocabulary": k in vocab}
                     for k, v in sorted(by_cls.items(), key=lambda kv: -kv[1][0])},
        "by_range": {k: {"n": v[0], "n_infield": v[1],
                         "frac_infield": v[1] / max(v[0], 1)}
                     for k, v in by_rng.items()},
        "per_clip_keep_fraction": {
            "n_clips": int(len(fr)),
            "min": float(fr.min()) if len(fr) else None,
            "p10": float(np.percentile(fr, 10)) if len(fr) else None,
            "p50": float(np.percentile(fr, 50)) if len(fr) else None,
            "p90": float(np.percentile(fr, 90)) if len(fr) else None,
            "max": float(fr.max()) if len(fr) else None,
            "std": float(fr.std()) if len(fr) else None,
            "corr_with_clip_n_lines": _corr(ln, fr),
            "corr_with_agents_per_frame": _corr(dens, fr),
        },
        "imbalance_in_vocabulary": None,
        "wall_s": round(time.time() - t0, 1),
    }
    inv = [v["n"] for k, v in res["by_class"].items() if v["in_vocabulary"]]
    if inv:
        res["imbalance_in_vocabulary"] = {
            "n_classes": len(inv), "max": max(inv), "min": min(inv),
            "ratio": max(inv) / max(min(inv), 1),
            "n_boxes_in_vocabulary": sum(inv),
        }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("by_class", "by_range")}, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
