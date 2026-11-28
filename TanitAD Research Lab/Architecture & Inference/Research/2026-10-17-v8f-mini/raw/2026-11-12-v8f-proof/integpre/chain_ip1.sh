#!/usr/bin/env bash
# chain_ip1.sh : dev-box side, started by the lead. Card PREREG_V8F_PROOF_ADD44 (INTEG-PRE). Script polling only; no model turn waits on the pod.
# 1. pushes the tool files into the NEW pod folder /workspace/v8proof/integpre (sha256 compared on both sides; a file that is there with another sha256 stops the chain: nothing is overwritten),
# 2. starts the pod queue pod_integpre_queue.sh with nohup setsid (the queue refuses a second start by its own marker),
# 3. waits with a polling loop that exits on INTEGPRE_DONE.txt, INTEGPRE_STOP.txt or a timeout (IP_CHAIN_MAX_MIN, default 720 minutes),
# 4. pulls the outputs: per rollout a tar of the dump + ticks + t1v8f json (packed on the pod at idle priority, in the pod folder), pulled binary-clean with pcat.sh, sha256 compared with the pod's, extracted into its own
#    folder pull_t1/<RUN>/ ; the archive is deleted after the extraction is checked (literal path, logged with its size in DELETIONS_IP1.log; the pod keeps its archive). Heads, fit records, markers, the queue log and the
#    run records (summary, selfcheck, argv) are pulled as small files with a sha256 compare.
# Last line: ZZ CHAIN-IP1 DONE <time> or ZZ CHAIN-IP1 STOP <reason>. Refuses to start if IP_MAX_WINDOWS is set. Never prints the pod address. Deletes only its own pulled archives.
# DRY=1 prints the plan (files, run names) and touches nothing.
[ -z "${IP_MAX_WINDOWS:-}" ] || { echo "ZZ CHAIN-IP1 STOP IP_MAX_WINDOWS is set (a subset run is not a read of this card)"; exit 5; }
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof/integpre
W=/workspace/v8proof; D=$W/integpre; RR=$W/runs/distrule/t1
PULL=$H/pull_t1; RUNS=$H/pull_runs; SMALL=$H/pull_small; TARS=$H/work/tars; DELLOG=$H/DELETIONS_IP1.log
MAXMIN=${IP_CHAIN_MAX_MIN:-720}
DRY=${DRY:-0}
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${2:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
cd "$H" || exit 1
PUSH="pod_integpre_queue.sh integpre_arms.txt tools/pod_ruleattr_run.sh tools/dist_rule.py tools/cmp_dumps.py tools/v8f_t1_eval_rule.py tools/v8f_t1_eval_rule_mix.py tools/v8f_t1_eval_rule_qhead.py tools/qhead_model.py tools/qhead_fit_n1.py"
# the run names: from the arms file (NEW QUEUE BASE SEED P); NEW is the RQ name
BASES=$(grep -v '^[[:space:]]*#' integpre_arms.txt | grep -v '^[[:space:]]*$' | awk '{print $3}')
RUNNAMES=""
for b in $BASES; do RUNNAMES="$RUNNAMES IR0_$b IRL_$b IRQ_$b"; done
RUNNAMES="$RUNNAMES IRH_RLAm_s0 IRQI_RLAm_s0"
if [ "$DRY" = 1 ]; then
  echo "DRY: push (sha256 compared): $PUSH -> $D (flat)"
  echo "DRY: start: cd $D && nohup setsid bash pod_integpre_queue.sh > $D/start.out 2>&1 < /dev/null &"
  echo "DRY: wait for $D/run/INTEGPRE_DONE.txt or INTEGPRE_STOP.txt, at most $MAXMIN minutes"
  echo "DRY: pull $(echo $RUNNAMES | wc -w) runs: $RUNNAMES"
  exit 0
fi
mkdir -p "$PULL" "$RUNS" "$SMALL" "$TARS" || { echo "ZZ CHAIN-IP1 STOP cannot create the local folders"; exit 3; }
stopc() { echo "ZZ CHAIN-IP1 STOP $1"; exit 4; }

# ---- 1. push
pod "mkdir -p $D" > /dev/null
for f in $PUSH; do
  b=$(sha256sum "$H/$f" | cut -d' ' -f1); n=$(basename "$f")
  a=$(pod "[ -f $D/$n ] && sha256sum $D/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    timeout 180 bash "$P" "cat > $D/$n" < "$H/$f" > /dev/null 2>&1
    a=$(pod "sha256sum $D/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
    [ "$a" = "$b" ] || stopc "push mismatch $n"
    echo "[$(ts)] pushed $n sha256 ${a:0:16}"
  elif [ "$a" = "$b" ]; then echo "[$(ts)] on the pod already: $n sha256 ${a:0:16}"
  else stopc "$n on the pod has another sha256 (${a:0:16} against ${b:0:16}): the folder is not changed"; fi
done
ok=$(pod "cd $D && bash -n pod_integpre_queue.sh && bash -n pod_ruleattr_run.sh && echo syntax-ok" | tr -d ' \n')
[ "$ok" = "syntax-ok" ] || stopc "the pod queue does not pass bash -n"

# ---- 2. start (the queue refuses a second start by its own marker)
pod "cd $D && (nohup setsid bash pod_integpre_queue.sh > $D/start.out 2>&1 < /dev/null &) ; sleep 5; head -n 3 $D/start.out | cut -c1-200" 90
echo "[$(ts)] queue started"

# ---- 3. wait
k=0; st=DONE
while :; do
  s=$(pod "ls $D/run/INTEGPRE_DONE.txt $D/run/INTEGPRE_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) break;; esac
  k=$((k+1)); [ "$k" -gt "$MAXMIN" ] && { st=STOP; echo "[$(ts)] not ended after $MAXMIN minutes"; break; }
  sleep 60
done
n_ref=$(pod "grep -c -- 'REFUSED' $D/start.out 2>/dev/null" | tr -d ' \n')
[ "${n_ref:-0}" -ge 1 ] 2>/dev/null && echo "[$(ts)] the queue start was refused: $(pod "head -n 2 $D/start.out | cut -c1-200")"
pod "tail -n 6 $D/run/integpre_queue.log 2>/dev/null | cut -c1-260"

# ---- 4. pull (also after a stop: whatever finished)
sha() { sha256sum "$1" | cut -c1-64; }
bad=""
for A in $RUNNAMES; do
  if [ -f "$PULL/$A/PULLED.txt" ]; then echo "[$(ts)] $A: already pulled"; continue; fi
  d=$(pod "[ -f $RR/$A/T1_DONE ] && echo yes" | tr -cd 'a-z')
  [ "$d" = yes ] || { echo "[$(ts)] $A has no T1_DONE: not pulled"; continue; }
  tag=${A}_m10_causal_heads; ok=0
  for att in 1 2 3; do
    rs=$(pod "mkdir -p $D/tars && cd $RR/$A && nice -n 19 tar -cf $D/tars/$A.tar dump_$tag ticks t1v8f_$tag.json && sha256sum $D/tars/$A.tar | cut -c1-64" 300 | tail -n 1)
    if [ ${#rs} != 64 ]; then echo "[$(ts)] $A: pack failed on the pod (answer length ${#rs}), attempt $att"; continue; fi
    timeout 900 bash "$C" "$D/tars/$A.tar" "$TARS/$A.tar" > /dev/null 2>&1
    if [ "$(sha "$TARS/$A.tar")" != "$rs" ]; then echo "[$(ts)] $A: tar sha256 mismatch, attempt $att"; continue; fi
    ok=1; break
  done
  if [ $ok != 1 ]; then bad="$bad $A"; continue; fi
  mkdir -p "$PULL/$A" "$RUNS/$A"
  ( cd "$PULL/$A" && tar -xf "$(cygpath -u "$TARS/$A.tar")" ) || { echo "[$(ts)] $A: untar failed"; bad="$bad $A"; continue; }
  nf=$(ls "$PULL/$A/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$' | tr -d '\r')
  if [ "$nf" != 141 ]; then echo "[$(ts)] $A: dump holds $nf ep files, not 141"; bad="$bad $A"; continue; fi
  for f in summary.json selfcheck.json argv.json; do timeout 120 bash "$C" "$RR/$A/$f" "$RUNS/$A/$f" > /dev/null 2>&1; done
  ck=$(pod "sha256sum $W/runs/*/${A#*_}/ckpt_last.pt 2>/dev/null | head -n 1 | cut -c1-64" 120 | tail -n 1)
  echo "tar_sha256 $rs ckpt_sha256 $ck pulled $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PULL/$A/PULLED.txt"
  sz=$(stat -c %s "$TARS/$A.tar")
  rm -f -- "${TARS:?}/$A.tar" && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) deleted $TARS/$A.tar ($sz bytes): extracted and checked ($nf ep files), sha256 $rs equal to the pod, the pod keeps $D/tars/$A.tar" >> "$DELLOG"
  echo "[$(ts)] $A: pulled (141 ep files, tar sha256 ${rs:0:16})"
done
# small files: heads, fit records, markers, queue log (sha256 compared)
for f in $(pod "cd $D && ls heads/*.qhead.npz heads/*.qhead.json run/MISSING_*.txt run/INTEGPRE_*.txt run/integpre_queue.log 2>/dev/null"); do
  n=$(echo "$f" | tr '/' '_'); a=$(pod "sha256sum $D/$f | cut -c1-64" | tail -n 1)
  timeout 120 bash "$C" "$D/$f" "$SMALL/$n" > /dev/null 2>&1
  [ "$(sha "$SMALL/$n")" = "$a" ] || { echo "[$(ts)] small file $f: sha256 mismatch"; bad="$bad $f"; }
done
echo "[$(ts)] small files pulled to $SMALL"
[ -n "$bad" ] && stopc "pull failed for:$bad"
[ "$st" = STOP ] && stopc "the pod queue ended with a STOP marker or the wait timed out (see $SMALL/run_INTEGPRE_STOP.txt)"
echo "ZZ CHAIN-IP1 DONE $(ts)"
