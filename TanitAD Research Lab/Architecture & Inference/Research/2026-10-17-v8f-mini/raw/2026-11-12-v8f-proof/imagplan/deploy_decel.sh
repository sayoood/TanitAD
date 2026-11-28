#!/usr/bin/env bash
# deploy_decel.sh : dev-box side, script polling (no model turns). Card PREREG_V8F_PROOF_ADD40, note N3 (candidate family decel).
#   1. waits until the pod sequence la0 (speed-scaled family) has ended, so that no run of it sees a changed file;
#   2. pushes the rebuilt tool (imag_plan.py, v8f_t1_eval_imagplan.py, pod_imagplan_run.sh) and pod_imagplan_seq3.sh, compares sha256;
#   3. starts ONE nohup chain on the pod: first the two runs that control p5b needs (IPix = cand-imag with the action log on LFT_s0 and on LA_s0),
#      then the remaining plumbing runs of both checkpoints (p1 first part, p1b, p3, cand-cv, cand-label, the shifted run);
#   4. waits for the first two, pulls their logs and reads p4b, p5b, p7 (plumb_read3.py; no unsafe rate is read);
#   5. waits for the rest and pulls their logs.
# Deletes nothing, stops nothing. Last line: ZZ DEPLOY-DECEL DONE or ZZ DEPLOY-DECEL STOP <reason>.
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/imagplan; R=/workspace/v8proof/runs/imagplan; D=/workspace/v8proof/imagplan
ts() { date -u +%H:%M:%SZ; }
cd "$H" || exit 1
k=0
while :; do
  s=$(timeout 60 bash "$P" "ls $R/SEQ_la0_DONE.txt 2>/dev/null | wc -l; grep -c 'ZZ SEQ STOP' $R/seq_la0.log 2>/dev/null" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  set -- $s
  [ "${1:-0}" = 1 ] && break
  [ "${2:-0}" -ge 1 ] 2>/dev/null && { echo "[$(ts)] sequence la0 stopped; the files are pushed all the same (nothing of it runs)"; break; }
  k=$((k+1)); [ "$k" -gt 72 ] && { echo "ZZ DEPLOY-DECEL STOP la0 not ended after 6 h"; exit 2; }
  sleep 120
done
echo "[$(ts)] la0 ended; pushing the rebuilt tool"
for f in imag_plan.py v8f_t1_eval_imagplan.py pod_imagplan_run.sh pod_imagplan_seq3.sh; do
  timeout 180 bash "$P" "cat > $D/$f.new && mv $D/$f.new $D/$f" < "$H/$f" 2>/dev/null
  a=$(timeout 60 bash "$P" "sha256sum $D/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
  b=$(sha256sum "$H/$f" | cut -d' ' -f1)
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "ZZ DEPLOY-DECEL STOP push mismatch $f pod ${a:0:16} local ${b:0:16}"; exit 3; }
  echo "[$(ts)] pushed $f sha256 ${a:0:16}"
done
timeout 90 bash "$P" "cd $D && bash -n pod_imagplan_seq3.sh && bash -n pod_imagplan_run.sh && (nohup setsid bash -c 'export IP_SFX=x IP_REF_SFX=; bash pod_imagplan_seq3.sh lft0x1 leadcond_a LFT_s0 /workspace/v8proof/tree_leadcond IPi; bash pod_imagplan_seq3.sh la0x1 actcons_a LA_s0 /workspace/v8proof/tree_actcons IPi; bash pod_imagplan_seq3.sh lft0x2 leadcond_a LFT_s0 /workspace/v8proof/tree_leadcond \"IPid1 IPid3 IPn IPc IPl IPs\"; bash pod_imagplan_seq3.sh la0x2 actcons_a LA_s0 /workspace/v8proof/tree_actcons \"IPid1 IPid3 IPn IPc IPl IPs\"' > $R/seq_x.nohup 2>&1 < /dev/null &); sleep 8; cut -c1-200 $R/seq_lft0x1.log" 2>/dev/null | sed '/hf_/d'
echo "[$(ts)] pod chain started"
WS_IMAG_TAG=IPix WS_CV_TAG=IPcx bash "$H/wait_seq3.sh" lft0x1 LFT_s0 "IPix" no > "$H/wait_seq_lft0x1.out" 2>&1
WS_IMAG_TAG=IPix WS_CV_TAG=IPcx bash "$H/wait_seq3.sh" la0x1 LA_s0 "IPix" yes > "$H/wait_seq_la0x1.out" 2>&1
grep -h "ZZ\|^p4b\|^p5b\|^p7\|^beside\|MISMATCH\|STOP\|done (" "$H/wait_seq_lft0x1.out" "$H/wait_seq_la0x1.out" | cut -c1-700
echo "ZZ DEPLOY-DECEL P5B READ $(ts)"
WS_IMAG_TAG=IPix WS_CV_TAG=IPcx bash "$H/wait_seq3.sh" lft0x2 LFT_s0 "IPid1x IPid3x IPnx IPcx IPlx IPsx" no > "$H/wait_seq_lft0x2.out" 2>&1
WS_IMAG_TAG=IPix WS_CV_TAG=IPcx bash "$H/wait_seq3.sh" la0x2 LA_s0 "IPid1x IPid3x IPnx IPcx IPlx IPsx" yes > "$H/wait_seq_la0x2.out" 2>&1
grep -h "cmp \|ZZ\|^p7\|MISMATCH\|STOP" "$H/wait_seq_lft0x2.out" "$H/wait_seq_la0x2.out" | cut -c1-400
echo "ZZ DEPLOY-DECEL DONE $(ts)"
