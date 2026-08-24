#!/usr/bin/env bash
# ⭐ THE READABLE O11 DIAGNOSTIC. `o11_excess` is a per-batch estimate over
# EIGHT rows, so a single log line is dominated by noise — I called a trend at
# step 2600 on one high point and it reverted two lines later. This reports a
# WINDOWED MEAN and a paired t against zero, which is what the pre-registration's
# "must rise above 0 and STAY there" actually asks for.
ssh -n -o ConnectTimeout=30 tanitad-thor-wifi 'python3 - <<EOF
import json, statistics as st
rows=[json.loads(l) for l in open("/home/nvidia/v7tiny/o11p30k/train_log.jsonl") if l.strip()]
rows=[r for r in rows if r.get("step") is not None and r.get("o11_excess") is not None]
if not rows:
    print("ZZno-rows ZZ"); raise SystemExit
def blk(rs, tag):
    x=[r["o11_excess"] for r in rs]
    if len(x)<3: return
    m=st.mean(x); sd=st.stdev(x); se=sd/len(x)**0.5
    t=m/se if se>0 else 0.0
    o5=st.mean([r.get("o5_loss",0) for r in rs])
    print("ZZ%-9s n=%2d  mean %+9.2e  t %+6.2f  |sep| %8.2e  o5 %.4f ZZ" % (
        tag, len(x), m, t, st.mean([abs(r.get("o11_sep_rel",0)) for r in rs]), o5))
h=len(rows)//2
blk(rows[:h], "first-half"); blk(rows[h:], "second-half"); blk(rows[-10:], "last-10")
r=rows[-1]
print("ZZlatest    step %d  excess %+9.2e  floor %.4f  ratio %8.2e ZZ" % (
    r["step"], r["o11_excess"], r.get("o11_no_info_floor",1.3863),
    r["o11_excess"]/r.get("o11_no_info_floor",1.3863)))
EOF' 2>/dev/null | grep -oE "ZZ[^Z]*ZZ"
