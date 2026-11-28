"""score_q3rain.py <answers_q3.json> <answers_q3rain.json> [--key1 ..] [--key2 ..] [--out f]
Pools the Q3 rain frames (11) with the second draw (20) and applies PREREG_Q3RAIN.md's bar."""
import argparse, json, os, sys
SHEET = "F:/TanitAD-scratch/strat1/whead/q3sheet"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SHEET)
from score_q3 import wilson

def load(p):
    d = json.load(open(p, encoding="utf-8")); a = d.get("answers", d)
    return {k: (v if isinstance(v, dict) else {"answer": v}) for k, v in a.items()}

def tally(ans, key):
    by = {k["id"]: k for k in key if k["label"] == "rain"}
    extra = [i for i in ans if i not in {k["id"] for k in key}]
    if extra: raise ValueError(f"unknown frame ids {extra[:3]}")
    dist = {a: 0 for a in ["clear", "fog", "rain", "snow", "cant_tell", "unanswered"]}
    for fid in by:
        a = (ans.get(fid) or {}).get("answer") or "unanswered"
        if a not in dist: raise ValueError(f"bad answer {a!r}")
        dist[a] += 1
    return dist

def reading(n, p, n_total):
    if n < 25: return f"NOT AUDITABLE at this n (answered {n} of {n_total}, needs >= 25)"
    if p >= 0.80: return "SOUND (>= 0.80): no relabel for rain"
    if p >= 0.70: return "BORDERLINE (0.70 to 0.80): stays borderline, interval cannot separate it from either line; rain relabel left to a PI cost decision"
    return "RELABEL CANDIDATE (< 0.70)"

def block(dist):
    n = sum(dist[c] for c in ["clear", "fog", "rain", "snow"]); a = dist["rain"]
    return n, a

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a1"); ap.add_argument("a2")
    ap.add_argument("--key1", default=SHEET + "/key.json"); ap.add_argument("--key2", default=HERE + "/key_q3rain.json"); ap.add_argument("--out")
    x = ap.parse_args()
    d1 = tally(load(x.a1), json.load(open(x.key1))); d2 = tally(load(x.a2), json.load(open(x.key2)))
    pooled = {k: d1[k] + d2[k] for k in d1}
    n, a = block(pooled); n1, a1 = block(d1); n2, a2 = block(d2)
    r = {"pooled": {"frames": sum(pooled.values()), "answered": n, "agree": a, "agreement": a / n if n else None, "wilson95": list(wilson(a, n)),
                    "answers": pooled, "reading": reading(n, a / n if n else 0.0, sum(pooled.values()))},
         "q3_rain_only": {"answered": n1, "agree": a1, "answers": d1},
         "second_draw_only": {"answered": n2, "agree": a2, "agreement": a2 / n2 if n2 else None, "wilson95": list(wilson(a2, n2)), "answers": d2, "note": "reference, not decision-bearing"}}
    s = json.dumps(r, indent=1)
    if x.out: open(x.out, "w", encoding="utf-8").write(s)
    print(s)
if __name__ == "__main__": main()
