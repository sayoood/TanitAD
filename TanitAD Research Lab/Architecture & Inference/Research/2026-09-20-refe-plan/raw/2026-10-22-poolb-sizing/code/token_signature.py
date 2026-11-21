#!/usr/bin/env python3
"""What do NAVSIM navtrain TOKENS look like inside their own logs?  (LOCAL: TanitAstra compact_metadata of the 1,192 navtrain logs + the repo's
navtrain token table.)  Per frame: is it a token, ego speed, driving command (ego[4:8] one-hot: L,S,R,U? read from the data), route flag.
Answers: is token membership a thinning at a fixed stride, a speed/command filter, or both?  Writes raw/token_signature.json."""
import os, json, gzip, csv, collections, math
import numpy as np
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
CM = "D:/TanitAstraResearch/data/e024_corpus_20261010/compact_metadata/"
tok = collections.defaultdict(set)
with gzip.open("C:/Users/Admin/par18/unseenpool/raw/src/navtrain_token_table.tsv.gz", "rt", newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["log"]:
            tok[r["log"]].add(r["token"])
nt = [l.strip() for l in open("C:/Users/Admin/par18/unseenpool/raw/src/navtrain_logs.txt") if l.strip()]
speed_tok, speed_non, cmd_tok, cmd_non = [], [], collections.Counter(), collections.Counter()
gaps = collections.Counter(); per_log = {}
elig_in = elig_out = 0; tok_not_elig = 0
for log in nt:
    d = json.load(open(CM + log + ".json")); fr = d["frames"]; n = len(fr)
    el = set(d.get("eligible_anchor_indices") or [])
    idx = []
    for i, f in enumerate(fr):
        if f is None: continue
        e = f["ego"]; v = math.hypot(e[0], e[1]); cmd = int(np.argmax(e[4:8])) if len(e) >= 8 else -1
        is_t = f["token"] in tok[log]
        (speed_tok if is_t else speed_non).append(v)
        (cmd_tok if is_t else cmd_non)[cmd] += 1
        if is_t:
            idx.append(i)
            if i not in el: tok_not_elig += 1
    for a, b in zip(idx, idx[1:]): gaps[b - a] += 1
    per_log[log] = {"n_frames": n, "n_tokens": len(idx), "n_eligible": len(el)}
def q(a): a = np.asarray(a); return {"n": int(a.size), "mean": round(float(a.mean()), 3), "p05": round(float(np.percentile(a, 5)), 3), "p25": round(float(np.percentile(a, 25)), 3), "p50": round(float(np.percentile(a, 50)), 3), "p75": round(float(np.percentile(a, 75)), 3), "frac_lt_0p5": round(float((a < 0.5).mean()), 4)}
out = {"speed_token_frames": q(speed_tok), "speed_nontoken_frames": q(speed_non), "command_token": dict(cmd_tok), "command_nontoken": dict(cmd_non),
       "gap_hist_between_consecutive_tokens_in_frames": dict(sorted(gaps.items())[:30]), "tokens_not_in_eligible_anchor_indices": tok_not_elig,
       "logs": len(nt), "logs_with_0_tokens": sum(1 for v in per_log.values() if v["n_tokens"] == 0),
       "tokens_per_log_quantiles": [int(x) for x in np.percentile([v["n_tokens"] for v in per_log.values()], [0, 5, 25, 50, 75, 95, 100])]}
json.dump({"summary": out, "per_log": per_log}, open(os.path.join(RAW, "token_signature.json"), "w"))
print(json.dumps(out, indent=1))
