#!/bin/bash
# R8-NAVFEED-DIAG reader (registered 64d0150 before scoring): wait for Thor's navoff.out marker, pull the score, apply the rule.
set -u
D=/c/Users/Admin/refcv8_pf/navsim_ckpt/navoff; LOG=$D/navoff_waiter.log; mkdir -p $D
mark() { echo "ZZNOW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START
t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZNAVOFFSCORE[D]' /home/nvidia/r8ns/navtrain_sel/navoff.out; grep -c 'ZZREFUS[E]' /home/nvidia/r8ns/navtrain_sel/navoff.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark REFUSED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 18000 ] && { mark TIMEOUT; exit 1; }
  sleep 120
done
mark SCORED
ssh -n tanitad-thor "cd /home/nvidia/r8ns/navtrain_sel/navoff && tar -cf - scores bridge/seam_R8_NAVOFF.manifest.json bridge/seam_R8_NAVOFF.npz cuda_controls.json score.log" | tar -xf - -C $D
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 - > $D/NAVOFF_READ.out 2>&1 <<'EOF'
import importlib.util, json, sys, numpy as np
spec = importlib.util.spec_from_file_location("b8n", "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py")
B = importlib.util.module_from_spec(spec); sys.modules["b8n"] = B; spec.loader.exec_module(B)
R = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"; D = "C:/Users/Admin/refcv8_pf/navsim_ckpt/navoff"
c = json.load(open(D + "/scores/r8sel_navoff_R8_NAVOFF/r8sel_navoff_R8_NAVOFF.counts.json"))
assert c.get("status") == "PASS", c.get("status")
NO = B.read_scores_csv(D + "/scores/r8sel_navoff_R8_NAVOFF/r8sel_navoff_R8_NAVOFF.csv")
LG = B.read_scores_csv(R + "/navtrain_sel/diag/scores/r8sel_diag_ema48852_LEGAL_s0/r8sel_diag_ema48852_LEGAL_s0.csv")
PR = B.read_scores_csv(R + "/navtrain_sel/scores/r8sel_ref_PRIOR_ha0p/r8sel_ref_PRIOR_ha0p.csv")
toks = sorted(LG.index); assert sorted(NO.index) == toks and len(toks) == 2089
ti = B.load_inputs(R + "/inputs/navtrain_sample_inputs.json.gz").loc[toks]; cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]]); cfg = {**B.DEFAULT_CFG}
res = {"id": "R8-NAVFEED-DIAG", "registered": "64d0150 (before scoring)", "tier": "NavSim v1 open-loop PDMS, navtrain sample, POST-HOC diagnostic",
       "PDMS_x100": {"R8_NAVOFF": round(100 * float(NO["PDMS"].mean()), 4), "R8_LEGAL": round(100 * float(LG["PDMS"].mean()), 4),
                     "PRIOR": round(100 * float(PR["PDMS"].mean()), 4)},
       "subs_NAVOFF": {m: round(100 * float(NO[m].mean()), 4) for m in B.SUBS}, "subs_LEGAL": {m: round(100 * float(LG[m].mean()), 4) for m in B.SUBS}}
res["NAVOFF_minus_LEGAL"] = B.pair_record(NO, LG, toks, cl, cfg)
res["NAVOFF_minus_PRIOR"] = B.pair_record(NO, PR, toks, cl, cfg)
for nm, m in (("straight", st), ("turn", ~st)):
    tt = [t for t, k in zip(toks, m) if k]; cc = [x for x, k in zip(cl, m) if k]
    res[f"NAVOFF_minus_LEGAL__{nm}"] = B.pair_record(NO, LG, tt, cc, cfg)
d = res["NAVOFF_minus_LEGAL"]; iv = d["interval"]
if d["delta_x100"] >= 3 and iv.get("positive"):
    v = "FEED IS A DOMINANT CAUSE (>= +3, separated)"
elif d["delta_x100"] <= -3 and iv.get("negative"):
    v = "THE COMMAND HELPS; the loss is elsewhere (<= -3, separated)"
else:
    v = "FEED IS NOT THE CAUSE (within (-3, +3) or interval includes 0)"
res["verdict"] = v
json.dump(res, open(D + "/NAVOFF_READ.json", "w"), indent=1)
print("QQNAVOFF " + json.dumps({"verdict": v, "PDMS": res["PDMS_x100"], "d": d["delta_x100"], "lo": iv.get("lo"), "hi": iv.get("hi"),
                                "straight": res["NAVOFF_minus_LEGAL__straight"]["delta_x100"], "turn": res["NAVOFF_minus_LEGAL__turn"]["delta_x100"]}) + " QQ")
EOF
[ -s $D/NAVOFF_READ.json ] && mark READOK "$(grep -o 'QQNAVOFF.*' $D/NAVOFF_READ.out | cut -c1-500)" || mark READBAD
mark END
