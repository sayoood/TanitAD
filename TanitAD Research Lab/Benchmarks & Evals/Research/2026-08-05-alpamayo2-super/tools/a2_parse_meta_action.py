"""Parse Alpamayo's meta-action text into a scored TACTICAL decision — and record
the architectural finding that the taxonomy itself is the leverage.

⭐ THE FINDING IS THE SHAPE, NOT THE SCORE. Alpamayo declares a manoeuvre on THREE
INDEPENDENT AXES:

    Longitudinal: Gentle Deceleration.
    Lateral:      Go Straight.
    Lane:         Lane Keep.

Our flagship declares ONE 5-way softmax over
``[lane_keep, turn_left, turn_right, accelerate, brake_stop]`` — which MIXES the
lateral and longitudinal decisions into a single mutually-exclusive choice. CLAUDE.md
names that mixing as *"our single largest known defect"*, and this is a working system
that does not have it: a vehicle that is simultaneously decelerating AND turning left
is one label in Alpamayo's scheme and is UNREPRESENTABLE in ours.

⛔ THE PARSER IS SEPARATE FROM THE RUN ON PURPOSE. `a2_meta_action.py` banks the raw
generation verbatim; this turns text into classes. A parser that silently mislabels is
indistinguishable from a model that decided wrongly, so every unparsed row is counted
and reported rather than dropped, and the raw string travels with every parsed record.

⚠️ The generation is SAMPLED (temperature 0.6). One draw is not the model's mode. The
seed is recorded; stability across draws is a separate, unrun measurement.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re

# Alpamayo's declared vocabulary, as observed. ⛔ Kept as an OBSERVED list, not an
# asserted one: if a run emits a value outside it, that is a finding about the
# taxonomy, and it must show up as UNKNOWN rather than be coerced into a neighbour.
AXES = ("Longitudinal", "Lateral", "Lane")

#: Projection of Alpamayo's LATERAL axis onto our {left, straight, right} direction
#: classes, which is the only axis where the two schemes are directly comparable.
#: ⛔ The projection is LOSSY IN OUR DIRECTION, not theirs: "Sharp Steer Left" and
#: "Steer Left" both collapse to `left` because our label set has no severity. That
#: loss is the point — it is what a factorised head would recover.
LAT2DIR = {
    "Go Straight": 1, "Steer Left": 0, "Steer Right": 2,
    "Sharp Steer Left": 0, "Sharp Steer Right": 2,
    "Slight Steer Left": 0, "Slight Steer Right": 2,
}
DIRNAME = {0: "left", 1: "straight", 2: "right"}
DIR_YAW_RAD = 0.15


def parse_axes(text: str) -> dict:
    out = {}
    for axis in AXES:
        m = re.search(rf"{axis}:\s*([^.\n<]+)", text)
        out[axis.lower()] = m.group(1).strip() if m else None
    # the Chain-of-Causation is everything before the first axis label
    first = min((text.find(f"{a}:") for a in AXES if f"{a}:" in text), default=-1)
    out["cot"] = text[:first].strip() if first > 0 else None
    return out


def net_yaw(path):
    import numpy as np
    p = np.concatenate([np.zeros((path.shape[0], 1, 2)), path], axis=1)
    d = p[:, 1:] - p[:, :-1]
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    return dh.sum(axis=1)


def kappa(x, y):
    import numpy as np
    labs = sorted(set(x.tolist()) | set(y.tolist()))
    po = float((x == y).mean())
    pe = sum((x == c).mean() * (y == c).mean() for c in labs)
    return None if abs(1 - pe) < 1e-9 else round((po - pe) / (1 - pe), 4)


def main():
    import lzma

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-jsonl", required=True)
    ap.add_argument("--alpamayo-traj", required=True)
    ap.add_argument("--alpamayo-gt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.meta_jsonl) if l.strip()]
    ok = [r for r in rows if "raw" in r]
    parsed, unparsed = {}, []
    for r in ok:
        t = r["raw"].get("raw_outputs")
        s = (t[0] if isinstance(t, list) and t else str(t))
        p = parse_axes(s)
        p["raw"] = s
        p["clip_id"] = r["clip_id"]
        if p["lateral"] is None or p["longitudinal"] is None:
            unparsed.append({"i": r["sample_index"], "raw": s[:200]})
        parsed[r["sample_index"]] = p

    tally = {ax: dict(collections.Counter(
        p[ax] for p in parsed.values())) for ax in ("longitudinal", "lateral", "lane")}

    # --- the one comparable axis: declared LATERAL vs the path Alpamayo actually drove
    A = {int(k): np.asarray(v) for k, v in
         json.loads(lzma.open(a.alpamayo_traj).read()).items()}
    AG = {int(k): np.asarray(v) for k, v in
          json.load(open(a.alpamayo_gt))["gt_xy_by_index"].items()}
    idx = sorted(i for i in parsed
                 if i in A and i in AG and parsed[i]["lateral"] in LAT2DIR)
    res = {"n_rows": len(rows), "n_parsed": len(parsed),
           "n_unparsed_axes": len(unparsed), "unparsed": unparsed[:5],
           "taxonomy_tally": tally}
    if idx:
        d_dec = np.array([LAT2DIR[parsed[i]["lateral"]] for i in idx])
        driven = net_yaw(np.stack([A[i][:20] for i in idx]))
        gtn = net_yaw(np.stack([AG[i][:20, :2] for i in idx]))
        to_cls = lambda ny: np.where(ny > DIR_YAW_RAD, 0,                # noqa: E731
                                     np.where(ny < -DIR_YAW_RAD, 2, 1))
        d_dr, d_gt = to_cls(driven), to_cls(gtn)
        res["declared_vs_driven"] = {
            "n": len(idx), "agreement": round(float((d_dec == d_dr).mean()), 4),
            "kappa": kappa(d_dec, d_dr)}
        res["declared_vs_gt"] = {
            "n": len(idx), "accuracy": round(float((d_dec == d_gt).mean()), 4),
            "kappa": kappa(d_dec, d_gt)}
        res["_projection_note"] = (
            "Alpamayo's LATERAL axis projected onto our {left, straight, right}. ⛔ The "
            "projection is lossy IN OUR DIRECTION: 'Sharp Steer Left' and 'Steer Left' "
            "both become `left` because our label set has no severity. Comparability "
            "is bought by discarding Alpamayo's resolution, and that is the finding.")
    res["_architectural_finding"] = (
        "Alpamayo declares a manoeuvre on THREE INDEPENDENT AXES "
        "(Longitudinal / Lateral / Lane). Our flagship declares ONE 5-way softmax over "
        "[lane_keep, turn_left, turn_right, accelerate, brake_stop], which MIXES "
        "lateral and longitudinal into a mutually-exclusive choice. "
        "'Decelerating AND turning left' is one label there and UNREPRESENTABLE here. "
        "CLAUDE.md names that mixing as our single largest known defect; this is a "
        "working system that does not have it.")
    res["_sampling"] = ("generate_text samples at temperature 0.6; one draw is not the "
                        "model's mode. Stability across draws is UNMEASURED.")
    res["_contamination"] = ("clips are PhysicalAI-AV, which Alpamayo lists as TRAINING "
                             "data; overlap UNRESOLVED")
    res["per_clip"] = {str(i): {k: parsed[i][k] for k in
                                ("longitudinal", "lateral", "lane", "cot", "clip_id")}
                       for i in sorted(parsed)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1)[:2600])
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
