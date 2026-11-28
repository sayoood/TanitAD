#!/usr/bin/env python3
"""Token subset for the ego probe (PREREG 'Token subset').  Uses ONLY the W3 export's logged inputs + logged future; no model output.

    make_subset.py --budget 0 --out tokens_all.json            # all 1,058 C3 tokens (+ CRUISE flags)
    make_subset.py --budget 420 --out tokens_subset.json       # ALL CRUISE tokens + seeded draw of the rest, <= 3 per log, up to the budget
    make_subset.py --smoke --out tokens_smoke.json             # 8 tokens, 4 logs (2 with CRUISE tokens, 2 without), 2 per log
"""
import argparse, gzip, hashlib, json, os
import numpy as np

PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
C3 = PK + "/raw/2026-10-11-c3-selection-headroom/tokens.json"
EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEED = 20261106

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.load(open(C3, encoding="utf-8"))
exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
toks = sorted(d["tokens"])
tl, st = d["token_log"], d["stratum"]
cruise = {}
for t in toks:
    r = exp[t]; e = r["ego_statuses"][-1]
    v0 = float(np.hypot(*e["ego_velocity"])); ax0 = float(e["ego_acceleration"][0])
    h = np.asarray(r["human_future_poses"], float); v4 = float(np.linalg.norm(h[7, :2] - h[6, :2]) / 0.5)
    cruise[t] = bool(v0 > 5 and abs(ax0) < 0.3 and abs(v4 - v0) < 1.0)
rng = np.random.default_rng(SEED)
if a.smoke:
    logs = sorted(set(tl.values()))
    cl = [l for l in logs if sum(cruise[t] and tl[t] == l for t in toks) >= 2]
    nl = [l for l in logs if not any(cruise[t] and tl[t] == l for t in toks) and sum(tl[t] == l for t in toks) >= 2]
    pick_c = list(rng.choice(cl, 2, replace=False)); pick_n = list(rng.choice(nl, 2, replace=False))
    sel = []
    for l in pick_c:
        sel += sorted(t for t in toks if tl[t] == l and cruise[t])[:2]
    for l in pick_n:
        sel += sorted(t for t in toks if tl[t] == l)[:2]
elif a.budget <= 0:
    sel = list(toks)
else:
    sel = [t for t in toks if cruise[t]]
    rest = [t for t in toks if not cruise[t]]
    cnt = {}
    for t in sel:
        cnt[tl[t]] = cnt.get(tl[t], 0) + 0
    perm = list(rng.permutation(len(rest)))
    per_log = {}
    for i in perm:
        t = rest[i]
        if len(sel) >= a.budget:
            break
        if per_log.get(tl[t], 0) < 3:
            per_log[tl[t]] = per_log.get(tl[t], 0) + 1
            sel.append(t)
sel = sorted(set(sel))
doc = {"tokens": sel, "token_log": {t: tl[t] for t in sel}, "stratum": {t: st[t] for t in sel}, "cruise_export": {t: cruise[t] for t in sel},
       "provenance": {"from": C3, "export": EXPORT, "seed": SEED, "budget": a.budget, "smoke": a.smoke, "n": len(sel), "n_logs": len(set(tl[t] for t in sel)),
                      "n_cruise_export": int(sum(cruise[t] for t in sel)), "n_cruise_all_1058": int(sum(cruise.values())),
                      "rule": "CRUISE(export)= v0>5 & |ax0|<0.3 & |GT v4(0.5s chord)-v0|<1; subset = all CRUISE + seeded draw of the rest <=3/log up to budget"}}
json.dump(doc, open(a.out, "w"), indent=0)
print(a.out, doc["provenance"], "sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())
