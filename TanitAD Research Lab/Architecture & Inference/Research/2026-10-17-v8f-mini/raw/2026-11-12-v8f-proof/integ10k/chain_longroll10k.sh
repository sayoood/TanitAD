#!/usr/bin/env bash
# chain_longroll10k.sh -- DEV-BOX chain of card PREREG_V8F_PROOF_ADD70 (LONGROLL10K), built from chain_cyclebound10k.sh. Pushes the queue (and the three files of the smoke when absent), compares sha256
# of all four (a mismatch is a STOP), starts the queue detached (it waits on the pod for the end marker of the queue of card ADD69, up to 24 h), waits for LONGROLL10K_DONE.txt / _STOP.txt
# (poll by this script every 60 s, up to 72 h), pulls the small folder small_lr with sha256 compare into integ10k/small_lr. The run folders stay on the pod. Uses no rm, no mv, no kill.
# usage: bash chain_longroll10k.sh [--no-start]      last line: ZZ CHAIN-LONGROLL10K DONE|STOP <why>
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
H=/f/TanitAD-scratch/v8proof; I=$H/integ10k
W=/workspace/v8proof/integ10k; B=$W/build
PULL=$I/small_lr; TARS=$I/work/tars; LOGF=$I/chain_longroll10k.log
NOSTART=0; [ "${1:-}" = "--no-start" ] && NOSTART=1
ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$LOGF"; }
fin() { echo "ZZ CHAIN-LONGROLL10K STOP $*" | tee -a "$LOGF"; exit 3; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
mkdir -p "$PULL" "$TARS"

FILES=("$I/v8f_t1_eval_reader_rule_rzgfhq.py" "$I/check_preroll_b.py" "$I/make_longroll_grid.py" "$I/pod_longroll10k_queue.sh")
pod "mkdir -p $B" > /dev/null
for f in "${FILES[@]}"; do
  n=$(basename "$f")
  [ -f "$f" ] || fin "missing local file $n"
  b=$(sha256sum "$f" | cut -d' ' -f1)
  a=$(pod "[ -f $B/$n ] && sha256sum $B/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
  if [ -z "$a" ]; then
    PODT=180 pod "cat > $B/$n" < "$f" > /dev/null
    a=$(pod "sha256sum $B/$n | cut -d' ' -f1" | tr -cd 'a-f0-9')
    [ "$a" = "$b" ] || fin "push mismatch $n"
    say "pushed $n sha256 ${a:0:16}"
  elif [ "$a" = "$b" ]; then say "on the pod already: $n sha256 ${a:0:16}"
  else fin "$n on the pod has another sha256 (${a:0:16} against ${b:0:16}); the folder is not changed"; fi
done
[ "$(pod "bash -n $B/pod_longroll10k_queue.sh && echo ok" | tr -cd 'a-z')" = ok ] || fin "bash -n of the queue failed on the pod"
[ "$(pod "[ -d $B/tree_ov/taniteval/taniteval ] && [ -e $B/tree_ov/stack/scripts/v8f_t1_eval.py ] && echo yes" | tr -cd 'a-z')" = yes ] || fin "overlay tree tree_ov is missing on the pod"

if [ $NOSTART = 0 ]; then
  s=$(pod "ls $W/START_LONGROLL10K.txt 2>/dev/null | wc -l" | tr -cd '0-9')
  [ "$s" = 0 ] || fin "START_LONGROLL10K.txt exists on the pod: a queue was started"
  pod "cd $W && nohup setsid bash $B/pod_longroll10k_queue.sh > $W/queue_longroll_stdout.txt 2>&1 < /dev/null & sleep 2; ls $W/START_LONGROLL10K.txt 2>&1 | tail -n 1" > /dev/null
  say "queue started"
fi

k=0; st=DONE
while :; do
  s=$(pod "ls $W/LONGROLL10K_DONE.txt $W/LONGROLL10K_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) break;; esac
  [ $NOSTART = 1 ] && break
  k=$((k+1)); [ "$k" -gt 4320 ] && { st=STOP; say "not ended after 72 h"; break; }
  sleep 60
done
[ "$st" = STOP ] && pod "cat $W/LONGROLL10K_STOP.txt | cut -c1-300" | tee -a "$LOGF"

pull_dir() { # name remote-folder local-folder
  local name=$1 rdir=$2 ldir=$3 a b
  [ -f "$ldir/PULLED.txt" ] && return 0
  pod "tar cf $W/pull_$name.tar -C $rdir . && sha256sum $W/pull_$name.tar | cut -d' ' -f1" > "$TARS/$name.sha" 2>/dev/null
  a=$(tr -cd 'a-f0-9' < "$TARS/$name.sha")
  [ ${#a} = 64 ] || { say "no tar of $name"; return 1; }
  bash "$C" "$W/pull_$name.tar" "$TARS/$name.tar" < /dev/null > /dev/null 2>&1
  b=$(sha256sum "$TARS/$name.tar" 2>/dev/null | cut -d' ' -f1)
  [ "$a" = "$b" ] || { say "PULL-MISMATCH $name"; return 1; }
  mkdir -p "$ldir" && tar xf "$TARS/$name.tar" -C "$ldir" && echo "$a" > "$ldir/PULLED.txt"
  say "pulled $name sha256 ${a:0:16} (local tar kept at $TARS/$name.tar for the lead to clear)"
}
# the run folders are not pulled; only small_lr (compare files, grid counts and grids, check output, carry lines, pre-roll logs of the controls, queue log)
pull_dir small_lr "$W/small_lr" "$PULL"
if [ "$st" = DONE ]; then echo "ZZ CHAIN-LONGROLL10K DONE $(ts)" | tee -a "$LOGF"; else echo "ZZ CHAIN-LONGROLL10K STOP queue-stop-or-timeout $(ts)" | tee -a "$LOGF"; fi
