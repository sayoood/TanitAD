#!/usr/bin/env bash
# pod_hbrecord_check.sh -- known-value check on the pod of the v8proof fork: the T1-kin executor of record of the other session (stack/scripts/v8f_t1_eval_hb.py at commit
# bcf267ce, pushed here as $B/v8f_t1_eval_hb_bcf267ce.py) against the study wrapper of card ADD67 under which rows 122 and later of the v8F proof line were measured.
# The pinned tool is not touched: an overlay tree $H/tree_r holds links to the pinned tree plus the executor wrapper.
# Two rollouts at model seed 0 (convention: no rule, pinned readout, 2.0 s cycle):
#   I10R0RecOld_s0  --kappa-max none  must equal the banked dump of I10R0_s0      (141 files, 0 differing)
#   I10R0RecHb_s0   --kappa-max 0.2   must equal the dump of I10R0Hb20_s0 of card ADD67 (the study wrapper's bound under the pinned readout)
# Markers: $H/HBREC_DONE.txt / $H/HBREC_STOP.txt. Nothing is deleted. Start: nohup setsid bash pod_hbrecord_check.sh > hbrec_stdout.txt 2>&1 < /dev/null &
set -u
W=/workspace/v8proof/integ10k; B=$W/build; H=$W/hbrec; PY=/workspace/venv/bin/python
SRC=$B/v8f_t1_eval_hb_bcf267ce.py
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$H/t1"
stop() { echo "$(ts) $*" > "$H/HBREC_STOP.txt"; echo "STOP $*"; exit 3; }
[ -f "$SRC" ] || { echo "no executor wrapper file"; exit 2; }
( set -C; echo "$(ts) pid $$" > "$H/HBREC_START.txt" ) 2>/dev/null || { echo "REFUSED: started before"; exit 2; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free_gb=$(( (mm - an) / 1073741824 ))
echo "$(ts) container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
T=$H/tree_r
mkdir -p "$T/stack/scripts"
[ -e "$W/tree/stack/scripts/v8f_t1_eval_hb.py" ] && stop "THE-POD-TREE-ALREADY-HAS-AN-EXECUTOR-WRAPPER"
[ -e "$T/taniteval" ] || ln -s "$B/tree_ov/taniteval" "$T/taniteval"
[ -e "$T/stack/tanitad" ] || ln -s "$W/tree/stack/tanitad" "$T/stack/tanitad"
for f in "$W"/tree/stack/scripts/*; do
  n=$(basename "$f"); [ -e "$T/stack/scripts/$n" ] || ln -s "$f" "$T/stack/scripts/$n"
done
[ -e "$T/stack/scripts/v8f_t1_eval_hb.py" ] || cp "$SRC" "$T/stack/scripts/v8f_t1_eval_hb.py"
echo "$(ts) overlay tree built; executor wrapper sha256 $(sha256sum "$T/stack/scripts/v8f_t1_eval_hb.py" | cut -c1-16); pinned tool $(sha256sum "$T/stack/scripts/v8f_t1_eval.py" | cut -c1-16)"
ro() { # NAME KAPPA
  local name=$1 kap=$2
  local out=$H/t1/$name tag=${name}_m10_causal_heads
  [ -e "$out" ] && stop "OUTPUT-EXISTS $out"
  mkdir -p "$out/ticks"
  echo "$(ts) rollout $name starts (kappa $kap)"
  PYTHONPATH="$T/stack" timeout 5400 "$PY" "$T/stack/scripts/v8f_t1_eval_hb.py" --kappa-max "$kap" --hb-log "$out/hb_log_$name.json" run --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" --tag "$tag" \
    --tracks /workspace/v8proof/data/inputs/native_tracks.pt --eval-cache "$W/eval_cache" --eval-grid /workspace/v8proof/hierrep/pins/eval_grid_windows.json \
    --eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz --out-dir "$out" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
    --dump-ticks "$out/ticks" --dump-heads --device cuda > "$out/t1.log" 2>&1
  local rc=$?
  [ $rc = 0 ] || { tail -n 5 "$out/t1.log" | cut -c1-300; stop "ROLLOUT-RC $rc $name"; }
  local n; n=$(ls "$out/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$'); [ "$n" = 141 ] || stop "DUMP-COUNT $n $name"
  echo "$(ts) rollout $name done; $(grep -m 1 'v8f_t1_eval_hb' "$out/t1.log" | cut -c1-160)"
}
ro I10R0RecOld_s0 none
"$PY" "$B/cmp_dumps.py" "$W/t1/I10R0_s0/dump_I10R0_s0_m10_causal_heads" "$H/t1/I10R0RecOld_s0/dump_I10R0RecOld_s0_m10_causal_heads" > "$H/CMP_RecOld.txt" 2>&1; r1=$?
echo "$(ts) R1 rc $r1 $(tail -n 1 "$H/CMP_RecOld.txt" | cut -c1-120)"
ro I10R0RecHb_s0 0.2
"$PY" "$B/cmp_dumps.py" "$W/t1/I10R0Hb20_s0/dump_I10R0Hb20_s0_m10_causal_heads" "$H/t1/I10R0RecHb_s0/dump_I10R0RecHb_s0_m10_causal_heads" > "$H/CMP_RecHb.txt" 2>&1; r2=$?
echo "$(ts) R2 rc $r2 $(tail -n 1 "$H/CMP_RecHb.txt" | cut -c1-120)"
"$PY" - "$H/t1/I10R0RecHb_s0/hb_log_I10R0RecHb_s0.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get("bound_counters") or {}
print("hb_log kappa", d.get("kappa_max_per_m"), "tool sha", str(d.get("tool_sha256"))[:16], {k: c[k] for k in c if not isinstance(c[k], dict)})
EOF
if [ "$r1" = 0 ] && [ "$r2" = 0 ]; then echo "$(ts) both equal" > "$H/HBREC_DONE.txt"; echo "ZZ HBREC DONE both equal"; else stop "DIFFERS r1=$r1 r2=$r2"; fi
