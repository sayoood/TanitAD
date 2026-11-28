#!/usr/bin/env python3
"""ESTIMATE how many NAVSIM-style tokens pool B would carry, and (more important) whether it can carry NAVSIM's OWN tokens at all.
Inputs are LOCAL: TanitAstra compact_metadata (1,252 segments = the 1,192 navtrain logs + the 60 pool-B logs: per frame token, ego (vx, vy, ax, ay, command one-hot), has_route)
and the repo's navtrain token table (103,288 rows; 103,037 resolve to a frame, the difference is the table's non-resolving rows: not investigated).
Features per frame: eligible anchor (3 <= j < n-10 and has_route, NAVSIM navtrain geometry), command class (0 L, 1 S, 2 R, 3 UNKNOWN), speed.
Method: ratio estimators tokens / (anchors passing a signature filter) fitted on navtrain, applied to pool B; session-level bootstrap and leave-one-session-out check give the spread.
Writes raw/token_estimate.json."""
import os, json, gzip, csv, math, collections, random
import numpy as np
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
CM = "D:/TanitAstraResearch/data/e024_corpus_20261010/compact_metadata/"
tok = collections.defaultdict(set)
with gzip.open("C:/Users/Admin/par18/unseenpool/raw/src/navtrain_token_table.tsv.gz", "rt", newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["log"]: tok[r["log"]].add(r["token"])
nt = [l.strip() for l in open("C:/Users/Admin/par18/unseenpool/raw/src/navtrain_logs.txt") if l.strip()]
B = [r["log"] for r in json.load(open(os.path.join(RAW, "poolB_logs.json")))]
sess = lambda l: l.rsplit("_", 2)[0]
def feats(log):
    d = json.load(open(CM + log + ".json")); fr = d["frames"]; n = len(fr)
    rows = []
    for j in range(n):
        f = fr[j]
        if f is None: rows.append(None); continue
        e = f["ego"]; rows.append((math.hypot(e[0], e[1]), int(np.argmax(e[4:8])), bool(f["has_route"]), f["token"]))
    out = {"n": n, "elig": 0, "F1": 0, "F2": 0, "tok": 0, "tok_F1": 0, "tok_F2": 0, "tok_eligible": 0, "unknown_cmd_frames": 0, "stationary_frames": 0}
    for j in range(n):
        r = rows[j]
        if r is None: continue
        v, c, rt, t = r
        out["unknown_cmd_frames"] += int(c == 3); out["stationary_frames"] += int(v < 0.5)
        is_t = t in tok.get(log, ()); out["tok"] += int(is_t)
        if 3 <= j < n - 10 and rt:
            out["elig"] += 1; out["tok_eligible"] += int(is_t)
            if c != 3:
                out["F1"] += 1; out["tok_F1"] += int(is_t)
                if v >= 0.5:
                    out["F2"] += 1; out["tok_F2"] += int(is_t)
    return out
NT = {l: feats(l) for l in nt}; PB = {l: feats(l) for l in B}
def tot(D, k): return sum(v[k] for v in D.values())
summary = {"navtrain": {k: tot(NT, k) for k in ("n", "elig", "F1", "F2", "tok", "tok_eligible", "tok_F1", "tok_F2", "unknown_cmd_frames", "stationary_frames")},
           "poolB": {k: tot(PB, k) for k in ("n", "elig", "F1", "F2", "unknown_cmd_frames", "stationary_frames")}}
summary["poolB"]["logs"] = len(PB); summary["navtrain"]["logs"] = len(NT)
summary["poolB"]["mean_frames_per_log"] = round(summary["poolB"]["n"] / len(PB), 1); summary["navtrain"]["mean_frames_per_log"] = round(summary["navtrain"]["n"] / len(NT), 1)
summary["poolB"]["frac_unknown_cmd_frames"] = round(summary["poolB"]["unknown_cmd_frames"] / summary["poolB"]["n"], 4)
summary["navtrain"]["frac_unknown_cmd_frames"] = round(summary["navtrain"]["unknown_cmd_frames"] / summary["navtrain"]["n"], 4)
summary["poolB"]["frac_stationary_frames"] = round(summary["poolB"]["stationary_frames"] / summary["poolB"]["n"], 4)
summary["navtrain"]["frac_stationary_frames"] = round(summary["navtrain"]["stationary_frames"] / summary["navtrain"]["n"], 4)
# ratio estimators: tokens per anchor passing filter
S_nt = collections.defaultdict(lambda: collections.Counter())
for l, v in NT.items():
    for k in ("elig", "F1", "F2", "tok"): S_nt[sess(l)][k] += v[k]
S_pb = collections.defaultdict(lambda: collections.Counter())
for l, v in PB.items():
    for k in ("elig", "F1", "F2"): S_pb[sess(l)][k] += v[k]
sess_nt = sorted(S_nt); rnd = random.Random(20261022)
def ratio(sel, key): return sum(S_nt[s]["tok"] for s in sel) / max(1, sum(S_nt[s][key] for s in sel))
res = {}
for key in ("elig", "F1", "F2"):
    point = ratio(sess_nt, key); boots = []
    for _ in range(2000):
        sel = [rnd.choice(sess_nt) for _ in sess_nt]; boots.append(ratio(sel, key))
    cnt = sum(v[key] for v in PB.values())
    # per-session matched: each pool-B session uses its own navtrain siblings' ratio (all 30 pool-B sessions have navtrain siblings)
    matched = 0.0; miss = 0
    for s, c in S_pb.items():
        if s in S_nt and S_nt[s][key] > 0: matched += c[key] * S_nt[s]["tok"] / S_nt[s][key]
        else: miss += 1; matched += c[key] * point
    # per-session ratio spread (p10/p90) as the prior probe did
    pr = sorted(S_nt[s]["tok"] / S_nt[s][key] for s in sess_nt if S_nt[s][key] > 0)
    p10, p90 = pr[int(0.1 * len(pr))], pr[int(0.9 * len(pr))]
    res[key] = {"pool_b_anchors": cnt, "navtrain_ratio": round(point, 4), "ratio_boot_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "est_tokens_pooled_ratio": round(cnt * point), "est_tokens_boot_ci95": [round(cnt * float(np.percentile(boots, 2.5))), round(cnt * float(np.percentile(boots, 97.5)))],
                "est_tokens_session_matched": round(matched), "per_session_ratio_p10_p90": [round(p10, 4), round(p90, 4)], "est_tokens_p10_p90": [round(cnt * p10), round(cnt * p90)],
                "pool_b_sessions_without_navtrain_sibling_ratio": miss}
# leave-one-session-out: how well does the matched-by-others ratio predict a held-out navtrain session's token count?
loso = []
for s in sess_nt:
    rest = [x for x in sess_nt if x != s]
    for key in ("F1",):
        pred = S_nt[s][key] * ratio(rest, key); act = S_nt[s]["tok"]
        loso.append((s, act, pred))
act = np.array([a for _, a, _ in loso], dtype=float); pred = np.array([p for _, _, p in loso], dtype=float)
lr = np.log((pred + 1) / (act + 1))
summary["loso_F1_session_level"] = {"n_sessions": len(loso), "median_abs_log_ratio": round(float(np.median(np.abs(lr))), 3), "p10_p90_ratio_pred_over_actual": [round(float(np.exp(np.percentile(lr, 10))), 2), round(float(np.exp(np.percentile(lr, 90))), 2)],
                                    "sum_actual": int(act.sum()), "sum_pred": round(float(pred.sum()))}
# how many navtrain logs have few tokens at all: the "rejection" hypothesis needs logs with many eligible anchors but ~0 tokens to exist inside navtrain
few = [(l, v["elig"], v["tok"]) for l, v in NT.items() if v["elig"] >= 150 and v["tok"] <= 3]
summary["navtrain_logs_with_ge150_anchors_and_le3_tokens"] = {"n": len(few), "of_logs_with_ge150_anchors": sum(1 for v in NT.values() if v["elig"] >= 150)}
summary["poolB_logs_with_ge150_anchors"] = sum(1 for v in PB.values() if v["elig"] >= 150)
json.dump({"summary": summary, "estimates": res, "per_log_poolB": PB}, open(os.path.join(RAW, "token_estimate.json"), "w"))
print(json.dumps({"summary": summary, "estimates": res}, indent=1))
