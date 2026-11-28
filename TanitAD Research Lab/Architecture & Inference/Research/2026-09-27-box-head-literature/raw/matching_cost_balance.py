"""How many METRES of centre error can the confidence/class term of the Hungarian cost override?

ANALYTIC, arithmetic only. For a candidate slot the matcher trades a confidence/class cost against a
geometric cost. We report the confidence-cost SPAN (confident slot vs. unconfident slot) divided by the
geometric weight per metre of centre error, i.e. "metre-equivalents".

  OURS (agent_slots.py:216-221, :767-785): cost = 1.0*(-sigmoid(presence)) + 1.0*(-p[true cls])
        + 1.0*(|dcx|+|dcy|) [m] + 0.5*(|dl|+|dw|) [m]
  DETR3D / PETR / BEVFormer / StreamPETR / UniAD configs: cls_cost = FocalLossCost weight 2.0 (the
        Deformable DETR matcher form: pos - neg, alpha 0.25, gamma 2), reg_cost = BBox3DL1Cost weight 0.25
        on the normalised box code whose centre terms are in metres.
Span is evaluated between a slot at p_lo and one at p_hi for the TRUE class.
"""
import json, math, sys

def focal_cost(p, a=0.25, g=2.0):
    neg = (1 - a) * p ** g * -math.log(1 - p + 1e-8)
    pos = a * (1 - p) ** g * -math.log(p + 1e-8)
    return pos - neg

rows = {}
for lo, hi in ((0.05, 0.5), (0.1, 0.9), (0.2, 0.8), (0.01, 0.99)):
    ours_span = 2.0 * (hi - lo)            # presence and class both move from lo to hi, weight 1.0 each
    lit_span = 2.0 * (focal_cost(lo) - focal_cost(hi))
    rows[f"p {lo}->{hi}"] = {
        "ours_conf_span": round(ours_span, 4),
        "ours_metres_equiv (1.0/m)": round(ours_span / 1.0, 3),
        "lit_focal_cost_span (x2.0)": round(lit_span, 4),
        "lit_metres_equiv (0.25/m)": round(lit_span / 0.25, 3),
        "ratio_lit_over_ours": round((lit_span / 0.25) / (ours_span / 1.0), 2),
    }
out = {"doc": __doc__.strip(), "focal_cost_values_x2": {str(p): round(2 * focal_cost(p), 4)
                                                       for p in (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)},
       "spans": rows}
json.dump(out, sys.stdout, indent=1)
