#!/usr/bin/env bash
# wait_leadsup_t0.sh : dev-box chain for the lead read of card PREREG_V8F_PROOF_ADD24 (section 3a). Script polling, no model turns. Bounded: 10 h.
# Per arm, once "ARM DONE <ARM>" stands in its pod chain log: (A1) pod_lead_apply.py on the pod (pushed once, sha256 compared), (A2) state extraction on the pod
# (CPU, idle priority; train 600:1800 and eval 0:141, as gw_read_arm_q.sh step 2), (A3) the trained head applied to both state files, (B) ten files pulled
# binary-clean and compared by sha256 with the pod. State files go to C:/Users/Admin/v8proof_c/s1states (F: is short), the small files to read/in on F:.
# When all six arms are pulled: run_leadsup_read.sh (controls first). It trains nothing and uses no GPU on the pod.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
HERE=/f/TanitAD-scratch/v8proof/spatial/leadsup/read
IN=$HERE/in; ST=/c/Users/Admin/v8proof_c/s1states
PB=/workspace/v8proof/probe; CH=/workspace/v8proof/chains
ARMS="LS_s0 LSD_s0 LSH_s0 LS_s1 LSD_s1 LSH_s1"
mkdir -p "$IN" "$ST"
ts() { date -u +%H:%M:%SZ; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
queue_of() { case "$1" in LS_s*) echo leadsup_a;; LSD_s*) echo leadsup_b;; *) echo leadsup_c;; esac; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r' | tr -d ' '; }

# A1: the pod script, once
want=$(sha "$HERE/pod_lead_apply.py")
for att in 1 2 3; do
  timeout 120 bash "$P" "mkdir -p $PB && cat > $PB/pod_lead_apply.py" < "$HERE/pod_lead_apply.py" > /dev/null 2>&1
  got=$(pod 60 "sha256sum $PB/pod_lead_apply.py | cut -c1-64" | tr -cd '0-9a-f\n' | tail -n 1)
  [ "$got" = "$want" ] && break
  sleep 30
done
[ "$got" = "$want" ] || { echo "ZZ LEADSUP-T0 FAIL: pod_lead_apply.py not verified on the pod"; exit 4; }
echo "[$(ts)] pod_lead_apply.py on the pod, sha256 ${want:0:16}"

files_of() { local A=$1; echo "${A}_eval.npz ${A}_eval.npz.json ${A}_train.npz ${A}_train.npz.json ${A}_eval_lead.npz ${A}_eval_lead.npz.json ${A}_eval_lead.npz.head.npz ${A}_train_lead.npz ${A}_train_lead.npz.json ${A}_train_lead.npz.head.npz"; }
dest_of() { case "$1" in *_eval.npz|*_eval.npz.json|*_train.npz|*_train.npz.json) echo "$ST/$1";; *) echo "$IN/$1";; esac; }

one_arm() { local A=$1 Q; Q=$(queue_of "$A"); local RR=/workspace/v8proof/runs/$Q
  [ -f "$IN/$A.PULLED" ] && return 0
  # memory on the pod before adding a CPU job: keep 20 GB free
  local fr; fr=$(pod 60 "echo \$(( (\$(cat /sys/fs/cgroup/memory.max) - \$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2)) / 1073741824 ))" | tail -n 1)
  case "$fr" in ''|*[!0-9]*) echo "[$(ts)] $A: pod memory not readable, later"; return 1;; esac
  [ "$fr" -ge 24 ] || { echo "[$(ts)] $A: pod memory free $fr GB, later"; return 1; }
  # A2 extraction (idempotent)
  timeout 3300 bash "$P" "mkdir -p $PB/states && cd $PB && for sp in 'train 600 1800' 'eval 0 141'; do set -- \$sp; [ -f states/${A}_\$1.npz.json ] || nice -n 19 ionice -c3 /workspace/venv/bin/python extract_states.py --tree /workspace/v8proof/tree_leadsup --ckpt $RR/$A/ckpt_last.pt --cache /workspace/v8proof/data/cache_pod --split \$1 --lo \$2 --hi \$3 --out states/${A}_\$1.npz > states/${A}_\$1.log 2>&1; done" > /dev/null 2>&1
  local n; n=$(pod 60 "ls $PB/states | grep -c -- '^${A}_\\(train\\|eval\\)\\.npz\\.json\$'" | tail -n 1)
  [ "${n:-0}" = "2" ] || { echo "[$(ts)] $A: extraction left ${n:-0} of 2 records: $(pod 60 "tail -n 2 $PB/states/${A}_train.log $PB/states/${A}_eval.log 2>/dev/null | cut -c1-200" | tr '\n' ' ')"; return 1; }
  # A3 head outputs (idempotent, CPU)
  timeout 1500 bash "$P" "cd $PB/states && for sp in eval train; do [ -f ${A}_\${sp}_lead.npz.json ] || CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python $PB/pod_lead_apply.py --tree /workspace/v8proof/tree_leadsup --ckpt $RR/$A/ckpt_last.pt --states ${A}_\$sp.npz --out ${A}_\${sp}_lead.npz > ${A}_\${sp}_lead.log 2>&1; done" > /dev/null 2>&1
  n=$(pod 60 "ls $PB/states | grep -c -- '^${A}_\\(train\\|eval\\)_lead\\.npz\\.json\$'" | tail -n 1)
  [ "${n:-0}" = "2" ] || { echo "[$(ts)] $A: head outputs left ${n:-0} of 2 records: $(pod 60 "tail -n 3 $PB/states/${A}_eval_lead.log $PB/states/${A}_train_lead.log 2>/dev/null | cut -c1-240" | tr '\n' ' ')"; return 1; }
  # B pulls, each compared with the pod sha256
  local sums; sums=$(pod 120 "cd $PB/states && sha256sum $(files_of "$A")")
  local ok=1 f w g d
  for f in $(files_of "$A"); do
    w=$(echo "$sums" | grep -- " $f\$" | cut -c1-64); d=$(dest_of "$f")
    [ -n "$w" ] || { echo "[$(ts)] $A: no pod sha256 for $f"; ok=0; continue; }
    for att in 1 2 3; do
      [ "$(sha "$d")" = "$w" ] && break
      timeout 900 bash "$C" "$PB/states/$f" "$d" > /dev/null 2>&1
    done
    [ "$(sha "$d")" = "$w" ] || { echo "[$(ts)] $A: $f not verified after 3 pulls"; ok=0; }
  done
  [ $ok = 1 ] || return 1
  ln_copy() { [ -f "$IN/$1" ] || cp "$ST/$1" "$IN/$1"; }
  ln_copy "${A}_eval.npz.json"; ln_copy "${A}_train.npz.json"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $A ten files verified against the pod" > "$IN/$A.PULLED"
  echo "[$(ts)] $A: pulled and verified ($(pod 60 "grep -o '\"ckpt_sha256\": \"[0-9a-f]*\"' $PB/states/${A}_eval.npz.json | cut -d'\"' -f4 | cut -c1-16"))"
}

k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd $CH; for c in leadsup_a leadsup_b leadsup_c; do [ -e \$c/\${c}_STOP.txt ] && echo \"STOP \$c \$(head -c 160 \$c/\${c}_STOP.txt)\"; done; grep -h -o -- 'ARM DONE [A-Za-z0-9_]*' leadsup_a/chain.log leadsup_b/chain.log leadsup_c/chain.log | sed 's/ARM DONE //'")
  echo "[$(ts)] done on the pod: $(echo "$S" | tr '\n' ' ' | cut -c1-200)"
  for a in $ARMS; do
    if echo "$S" | grep -q -x -- "$a"; then one_arm "$a"; fi
  done
  n=0; for a in $ARMS; do [ -f "$IN/$a.PULLED" ] && n=$((n+1)); done
  [ "$n" = 6 ] && break
  if echo "$S" | grep -q '^STOP '; then echo "ZZ LEADSUP-T0 STOP MARKER: $(echo "$S" | grep '^STOP ' | tr '\n' ';' | cut -c1-300) ; pulled $n of 6"; [ "$n" -ge 1 ] || exit 2; fi
  [ "$k" -gt 120 ] && { echo "ZZ LEADSUP-T0 TIME LIMIT (10 h), pulled $n of 6"; exit 3; }
  sleep 300
done
echo "[$(ts)] all six arms pulled; waiting for 6 GB of free commit, then the lead read"
j=0; while :; do m=$(free_gb); case "$m" in ''|*[!0-9]*) m=0;; esac; [ "$m" -ge 6 ] && break; j=$((j+1)); [ "$j" -gt 180 ] && { echo "ZZ LEADSUP-T0 NO MEMORY after 3 h (free commit $m GB)"; exit 5; }; sleep 60; done
cd "$HERE" || exit 1
STATES_DIR=$ST bash run_leadsup_read.sh "$IN" "$HERE/out" > "$HERE/out_run.log" 2>&1
rc=$?
echo "[$(ts)] run_leadsup_read rc $rc : $(tail -n 3 "$HERE/out_run.log" | tr '\n' ' ' | cut -c1-400)"
echo "ZZ LEADSUP-T0 READ DONE rc=$rc $(ts)"
