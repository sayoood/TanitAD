#!/usr/bin/env bash
# Fires the PRE-COMMITTED abort check at step 20,000 — see
# Project Steering/PREREG_O3_MASKED_CELL_30K.md. The criterion was fixed at step
# ~12,800, BEFORE the outcome was known, so it cannot be tuned to the answer.
set -u
while true; do
  R=$(ssh -n -o ConnectTimeout=25 tanitad-thor-wifi 'python3 - <<EOF
import json
rows=[json.loads(l) for l in open("/home/nvidia/v7tiny/o3p30k/train_log.jsonl") if l.strip()]
rows=[r for r in rows if r.get("step") is not None and r.get("o3_loss")]
if not rows: print("ZZWAITZZ"); raise SystemExit
cur=rows[-1]
def at(s):
    c=[r for r in rows if r["step"]<=s]
    return c[-1]["o3_loss"] if c else None
a5, a20 = at(5000), at(20000)
if cur["step"] < 20000:
    print("ZZPROG-%d-o3-%.4f-o5-%.4fZZ" % (cur["step"], cur["o3_loss"], cur.get("o5_loss",0)))
else:
    verdict = "INERT-ABORT" if (a20 is not None and a5 is not None and a20 >= a5) else "FALLING-CONTINUE"
    print("ZZCHECK20K-%s-o3at5k-%.4f-o3at20k-%.4f-o5-%.4fZZ" % (verdict, a5 or -1, a20 or -1, cur.get("o5_loss",0)))
EOF' 2>/dev/null | grep -oE "ZZ[^Z]*ZZ")
  case "$R" in
    ZZCHECK20K*) echo "$R"; break ;;
    ZZWAITZZ|"") : ;;
    *) echo "$R" ;;
  esac
  sleep 600
done
