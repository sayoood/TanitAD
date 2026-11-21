#!/usr/bin/env bash
# ⭐ THE READABLE O11 DIAGNOSTIC.
# ⛔⛔ FILTER ON `o11_loss` BEING TRUTHY, NOT ON `o11_excess is not None`.
# The trainer writes TWO lines per step: a real one and a STUB whose numeric
# fields are 0. `o11_excess` is 0.0 in the stub — which is NOT None — so the
# obvious filter admitted them, and every statistic was computed on ~50 % zeros:
# the mean was diluted toward zero and the variance inflated. MEASURED
# 2026-08-24: it reported gnorm median 0.41 (really 2.91), o5 median 0.0153
# (really 0.1340), and an excess t of +0.40 that was an artefact of the padding.
# Same family as C152 — a stub value that is TYPE-CORRECT and range-plausible.
ssh -n -o ConnectTimeout=30 tanitad-thor-wifi 'python3 - <<EOF
import json, statistics as st
raw=[json.loads(l) for l in open("/home/nvidia/v7tiny/o11p30k/train_log.jsonl") if l.strip()]
rows=[r for r in raw if r.get("step") is not None and r.get("o11_loss")]
print("ZZlines %d -> REAL %d (stubs dropped) ZZ" % (len(raw), len(rows)))
def blk(rs, tag):
    x=[r["o11_excess"] for r in rs]
    if len(x)<3: return
    m=st.mean(x); se=st.stdev(x)/len(x)**0.5
    print("ZZ%-11s n=%2d  mean %+9.2e  t %+6.2f  pick %.3f  o5 %.4f ZZ" % (
        tag, len(x), m, m/se if se>0 else 0,
        st.mean([r.get("o11_pick_acc",0) for r in rs]),
        st.mean([r["o5_loss"] for r in rs])))
h=len(rows)//2
blk(rows[:h],"first-half"); blk(rows[h:],"second-half"); blk(rows[-6:],"last-6")
r=rows[-1]
print("ZZlatest  step %d  o11 %.4f  excess %+.4f  pick %.3f  o5 %.4f  gn %.2f ZZ" % (
    r["step"], r["o11_loss"], r["o11_excess"], r.get("o11_pick_acc",-1), r["o5_loss"], r["gnorm"]))
EOF' 2>/dev/null | grep -oE "ZZ[^Z]*ZZ"
