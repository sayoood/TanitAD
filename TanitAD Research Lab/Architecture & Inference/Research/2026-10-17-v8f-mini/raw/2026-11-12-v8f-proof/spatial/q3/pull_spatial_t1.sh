#!/usr/bin/env bash
# pull_spatial_t1.sh ARM...  -- dev-box side: pull the finished T1-kin dump of each arm of one pod queue (env Q, default spatial; Q=dataeff for DATAEFF).
# Per arm: tar of the dump + ticks + t1v8f json (packed on the pod at idle priority), pulled binary-clean, sha256 compared with the pod's,
# unpacked to hiergap/pull/<ARM>/ (the layout hierrep_read.py and hierrep_gaps.py read); the run's small records to hiergap/runs/<ARM>/.
# Deletes nothing. Never prints the address. One line per arm; last line ZZ PULL-SPATIAL DONE or ZZ PULL-SPATIAL FAIL <arms>.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
W=/f/TanitAD-scratch/v8proof; PULL=${PULL_ROOT:-$W/hiergap/pull}; RUNS=${RUNS_ROOT:-$W/hiergap/runs}
TARS=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/d1b9de00-7d76-4030-8bc5-507afd36195b/scratchpad/q3tars
RR=/workspace/v8proof/runs/${Q:-spatial}; PB=/workspace/v8proof/probe
mkdir -p "$PULL" "$RUNS" "$TARS"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
sha() { sha256sum "$1" | cut -c1-64; }
bad=""
for A in "$@"; do
  if [ -f "$PULL/$A/PULLED.txt" ]; then echo "[$(ts)] $A: already pulled"; continue; fi
  tag=${A}_m10_causal_heads
  ok=0
  for att in 1 2 3; do
    rs=$(timeout 300 bash "$P" "mkdir -p $PB/dumps && cd $RR/t1/$A && [ -f T1_DONE ] && nice -n 19 tar -cf $PB/dumps/$A.tar dump_$tag ticks t1v8f_$tag.json && sha256sum $PB/dumps/$A.tar | cut -c1-64" 2>/dev/null | tr -d '\r' | tail -n 1)
    if [ ${#rs} != 64 ]; then echo "[$(ts)] $A: pack failed on the pod (answer length ${#rs}), attempt $att"; continue; fi
    timeout 600 bash "$C" "$PB/dumps/$A.tar" "$TARS/$A.tar" > /dev/null 2>&1
    if [ "$(sha "$TARS/$A.tar")" != "$rs" ]; then echo "[$(ts)] $A: tar sha256 mismatch, attempt $att"; continue; fi
    ok=1; break
  done
  if [ $ok != 1 ]; then bad="$bad $A"; continue; fi
  mkdir -p "$PULL/$A" "$RUNS/$A"
  ( cd "$PULL/$A" && tar -xf "$(cygpath -u "$TARS/$A.tar")" ) || { echo "[$(ts)] $A: untar failed"; bad="$bad $A"; continue; }
  nf=$(ls "$PULL/$A/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$' | tr -d '\r')
  if [ "$nf" != 141 ]; then echo "[$(ts)] $A: dump holds $nf ep files, not 141"; bad="$bad $A"; continue; fi
  for f in summary.json selfcheck.json argv.json; do timeout 120 bash "$C" "$RR/$A/$f" "$RUNS/$A/$f" > /dev/null 2>&1; done
  ck=$(timeout 120 bash "$P" "sha256sum $RR/$A/ckpt_last.pt | cut -c1-64" 2>/dev/null | tr -d '\r' | tail -n 1)
  echo "tar_sha256 $rs ckpt_sha256 $ck pulled $(ts)" > "$PULL/$A/PULLED.txt"
  echo "[$(ts)] $A: pulled (141 ep files, tar sha256 ${rs:0:16}, ckpt ${ck:0:16})"
done
if [ -n "$bad" ]; then echo "ZZ PULL-SPATIAL FAIL$bad"; exit 1; fi
echo "ZZ PULL-SPATIAL DONE"
