"""score_q3.py <answers.json> [--key key.json] [--out scores.json]
Computes exactly the numbers of PREREG_Q3.md: per label class n, agreement, Wilson 95 %, can't tell; pooled and prevalence-weighted overall; 4x4 confusion
(+ can't tell column); the bar reading per class. Answers file = the sheet's JSON {answers: {frame_id: {answer, note}}}."""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLS = ["clear", "fog", "rain", "snow"]
W = {"clear": 30988, "fog": 831, "rain": 1647, "snow": 2585}   # train window counts, RESULT_WEATHERPROBE.md
Z = 1.959964


def wilson(a, n, z=Z):
    if n == 0:
        return (None, None)
    p = a / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def reading(c, n, p):
    if c == "clear":
        return "sanity check only (no bar)"
    if n < 8:
        return "NOT AUDITABLE (n answered < 8)"
    if p >= 0.80:
        return "SOUND (>= 0.80): no relabel for this class"
    if p >= 0.70:
        return "BORDERLINE (0.70 to 0.80): second blind draw of this class"
    return "RELABEL CANDIDATE (< 0.70)"


def score(answers, key):
    by = {k["id"]: k for k in key}
    extra = [i for i in answers if i not in by]
    if extra:
        raise ValueError(f"answers for unknown frame ids: {extra[:3]}")
    conf = {c: {a: 0 for a in CLS + ["cant_tell", "unanswered"]} for c in CLS}
    for fid, k in by.items():
        a = (answers.get(fid) or {}).get("answer")
        if a is None:
            a = "unanswered"
        if a not in conf[k["label"]]:
            raise ValueError(f"bad answer {a!r} for {fid}")
        conf[k["label"]][a] += 1
    per = {}
    ta = tn = 0
    for c in CLS:
        n = sum(conf[c][a] for a in CLS)
        a = conf[c][c]
        p = a / n if n else None
        lo, hi = wilson(a, n)
        per[c] = {"n_frames": sum(conf[c].values()), "n_answered": n, "agree": a, "agreement": p, "wilson95": [lo, hi],
                  "cant_tell": conf[c]["cant_tell"], "unanswered": conf[c]["unanswered"], "reading": reading(c, n, p if p is not None else 0.0)}
        ta += a
        tn += n
    wsum = sum(W.values())
    wavg = sum(W[c] / wsum * per[c]["agreement"] for c in CLS if per[c]["agreement"] is not None)
    wcov = sum(W[c] / wsum for c in CLS if per[c]["agreement"] is not None)
    return {"per_class": per, "pooled": {"agree": ta, "n": tn, "agreement": ta / tn if tn else None, "wilson95": list(wilson(ta, tn)),
                                         "note": "stratified draw, NOT comparable to the 90.0 % of the random draw"},
            "prevalence_weighted": {"value": wavg, "weights_cover": wcov, "weights": W, "evidence": "ESTIMATED reweighting, no interval"},
            "confusion_rows_label_cols_pi": conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--key", default=HERE + "/key.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.answers, encoding="utf-8"))
    ans = d["answers"] if "answers" in d else d
    ans = {k: (v if isinstance(v, dict) else {"answer": v}) for k, v in ans.items()}
    key = json.load(open(a.key, encoding="utf-8"))
    r = score(ans, key)
    s = json.dumps(r, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    print(s)


if __name__ == "__main__":
    main()
