"""Candidate Qwen questions for visibility and illumination, scored against the PI's 60 blind answers (2026-11-08).

Why: on the 40 random audit frames Qwen's visibility scored 20/40 while 35 of the 40 are PI-good, so a constant answer would beat it; illumination has only
SigLIP zero-shot, 30/39 with glare folded into day. The questions below were written before any of them was run.

Selection rule, fixed here: on the 20 non-random audit frames (the rare strata) take the variant with the highest balanced accuracy over the PI classes present,
ties to the lower variant number. Every reported number is on the 40 random frames, which selection never reads. A variant is adopted only if its accuracy on the
random frames is at least that of the control (visibility: vis0, the current question; illumination: SigLIP zero-shot with glare folded into day).
Output: SL/audit_prompts_probs.json (per frame, per variant) and SL/AUDIT_PROMPTS_RESULT.json.
"""
import json
import math
import os
import sys

sys.path.insert(0, "F:/strat_models/pylibs")
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import extract_context as X
import gpu_lock

CANDIDATES = {
    "visibility": {
        "vis0": X.VLM_Q["visibility"],
        "vis1": "Can a driver see far ahead along the road in this image? Normal night driving with headlights, clouds, shadows or traffic ahead do not count as reduced visibility. A. yes, good visibility. B. reduced, fog, haze, heavy rain, snowfall or spray hide things at medium distance. C. poor, only the very near road can be seen. Answer with one letter.",
        "vis2": "Is the air clear in this driving image? A. yes, the air is clear. B. no, there is fog, haze, heavy rain, falling snow or spray in the air. C. no, and it is so dense that only a few metres can be seen. Answer with one letter.",
    },
    "illumination": {
        "ill1": "What is the lighting in this driving image? A. night on a dark road without street lighting. B. night on a street lit by street lamps or city lights. C. twilight, dusk or dawn. D. daylight. Answer with one letter.",
        "ill2": "Look at the sky and the overall brightness of this driving image. When was it taken? A. at night, with the road lit only by headlights. B. at night, with street lamps or city lights. C. at dusk or dawn, with the sun at the horizon or just below it. D. during the day, including cloudy days and low sun. Answer with one letter.",
    },
}
CLASSES = {"visibility": ("good", "reduced", "poor"), "illumination": ("dark", "lit_night", "twilight", "day")}
FOLD = {"glare": "day"}


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)]


def balanced(pairs):
    cl = sorted({y for y, _ in pairs})
    return float(np.mean([np.mean([p == yy for yy, p in pairs if yy == c]) for c in cl])) if cl else 0.0


def score(fr, probs, field, variants, control):
    """Selection on the non-random strata, report on the random frames."""
    cl = CLASSES[field]
    res = {}

    def rows(v, sel):
        out = []
        for a in fr:
            y = FOLD.get(a[field], a[field])
            if sel(a) and y in cl:
                out.append((y, cl[int(np.argmax(probs[str(a["order"])][v]))] if v in probs[str(a["order"])] else control[a["order"]]))
        return out
    for v in variants:
        res[v] = {"select_balanced_acc": round(balanced(rows(v, lambda a: a["stratum"] != "random")), 4)}
        r = rows(v, lambda a: a["stratum"] == "random")
        k = sum(y == p for y, p in r)
        res[v].update(report_correct=k, report_n=len(r), report_pct=round(100 * k / max(len(r), 1), 1), report_wilson=wilson(k, len(r)),
                      recall={c: [sum(1 for y, p in r if y == c and p == c), sum(1 for y, p in r if y == c)] for c in cl})
    best = max(variants, key=lambda v: (res[v]["select_balanced_acc"], -variants.index(v)))
    return res, best


def main():
    A = json.load(open(HERE + "/pilot_audit_answers_PI_60.json"))
    fr = A["frames"]
    assert all(a["frame_idx"] % 5 == 0 for a in fr)
    pf = HERE + "/audit_prompts_probs.json"
    if not os.path.exists(pf):
        X.wait_ram(X.QWEN_MIN_AVAIL_GB, "F:/strat_out")
        body = gpu_lock.acquire()
        try:
            q = X.Qwen()
            imgs = {}
            for c in sorted({a["clip_id"] for a in fr}):
                imgs[c] = X.decode(c, every=30)
            batch = np.stack([imgs[a["clip_id"]][a["frame_idx"] // 5] for a in fr])
            allq = {v: t for f in CANDIDATES.values() for v, t in f.items()}
            X.VLM_Q.clear()
            X.VLM_Q.update(allq)
            X.VLM_NCLS.clear()
            X.VLM_NCLS.update({v: len(CLASSES[f]) for f, d in CANDIDATES.items() for v in d})
            sc = q.score(batch)
        finally:
            gpu_lock.release(body)
        json.dump({str(a["order"]): {v: sc[v][i].tolist() for v in sc} for i, a in enumerate(fr)}, open(pf, "w"))
    probs = json.load(open(pf))
    import base64
    import re
    html = open("F:/strat_out/pilot/audit60/PILOT_AUDIT.html", encoding="utf8").read()
    H = json.loads(base64.b64decode(re.search(r'const HIDDEN_B64 = "([^"]+)"', html).group(1)))
    sig = {a["order"]: FOLD.get(h["src"]["illumination"]["siglip"], h["src"]["illumination"]["siglip"]) for a, h in zip(fr, H)}
    pilot_qwen_vis = {a["order"]: h["src"]["visibility"]["qwen"] for a, h in zip(fr, H)}
    out = {"n_frames": len(fr), "selection": "non-random strata, balanced accuracy", "report": "40 random frames"}
    vis, best_v = score(fr, probs, "visibility", list(CANDIDATES["visibility"]), {})
    ill, best_i = score(fr, probs, "illumination", list(CANDIDATES["illumination"]) + ["siglip_control"], sig)
    r = [a for a in fr if a["stratum"] == "random" and a["visibility"] in CLASSES["visibility"]]
    out["visibility"] = vis
    out["visibility"]["always_good"] = {"report_correct": sum(a["visibility"] == "good" for a in r), "report_n": len(r)}
    out["visibility"]["harness_check_vis0_equals_pilot_qwen"] = [int(sum(CLASSES["visibility"][int(np.argmax(probs[str(a["order"])]["vis0"]))] == pilot_qwen_vis[a["order"]] for a in fr)), len(fr)]
    out["illumination"] = ill
    out["selected"] = {"visibility": best_v, "illumination": best_i}
    out["adopted"] = {"visibility": best_v if vis[best_v]["report_correct"] >= vis["vis0"]["report_correct"] else "vis0",
                      "illumination": best_i if ill[best_i]["report_correct"] >= ill["siglip_control"]["report_correct"] else "siglip_control"}
    json.dump(out, open(HERE + "/AUDIT_PROMPTS_RESULT.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
