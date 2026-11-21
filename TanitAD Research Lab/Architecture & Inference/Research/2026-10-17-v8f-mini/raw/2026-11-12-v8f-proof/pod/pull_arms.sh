#!/usr/bin/env bash
# pull_arms.sh [max_hours]  -- dev-box side: pull every finished HIERREP arm from the pod as it finishes (script polling, no model turns).
# Per arm (name from "ARM DONE <name>" in stageB.log): tar of the T1-kin dump + ticks + t1v8f json (packed on the pod at idle priority), pulled binary-clean,
# sha256 compared with the pod's, unpacked to hiergap/pull/<ARM>/; the run's small records to hiergap/runs/<ARM>/; for HIER arms also the level-1 states
# (extract_states.py on the pod, CPU, idle priority; train 600:1800 and eval 0:141) to spatial/s1states/.
# Ends when the chain has a DONE or STOP marker and every finished arm is pulled, or after max_hours (default 40). Prints one line per arm. Never prints the address.
MAXH=${1:-40}
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
W=/f/TanitAD-scratch/v8proof; PULL=$W/hiergap/pull; RUNS=$W/hiergap/runs; ST=$W/spatial/s1states
HR=/workspace/v8proof/hierrep; RR=/workspace/v8proof/runs/hierrep; PB=/workspace/v8proof/probe
mkdir -p "$PULL" "$RUNS" "$ST"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
sha() { sha256sum "$1" | cut -c1-64; }
t0=$(date +%s)
while true; do
  LIST=$(timeout 60 bash "$P" "grep -o 'ARM DONE [A-Za-z0-9_]*' $HR/stageB.log | cut -d' ' -f3 | tr '\n' ' '; echo; M=none; [ -f $HR/STAGEB_DONE.txt ] && M=DONE; [ -f $HR/STAGEB_STOP.txt ] && M=STOP; echo marker=\$M" 2>/dev/null | tr -d '\r')
  MARK=$(echo "$LIST" | grep -o 'marker=[A-Za-z]*' | cut -d= -f2)
  ARMS=$(echo "$LIST" | grep -v 'marker=' | head -1)
  pending=0
  for A in $ARMS; do
    [ -f "$PULL/$A/PULLED.txt" ] && continue
    tag=${A}_m10_causal_heads
    rs=$(timeout 300 bash "$P" "mkdir -p $PB/dumps && cd $RR/t1/$A && [ -f T1_DONE ] && nice -n 19 tar -cf $PB/dumps/$A.tar dump_$tag ticks t1v8f_$tag.json && sha256sum $PB/dumps/$A.tar | cut -c1-64" 2>/dev/null | tr -d '\r' | tail -1)
    if [ ${#rs} != 64 ]; then echo "[$(ts)] $A: pack failed on the pod (answer length ${#rs}); retry next round"; pending=1; continue; fi
    timeout 600 bash "$C" "$PB/dumps/$A.tar" "$PULL/$A.tar"
    if [ "$(sha "$PULL/$A.tar")" != "$rs" ]; then echo "[$(ts)] $A: tar sha256 mismatch; retry next round"; rm -f "${PULL:?}/${A:?}.tar"; pending=1; continue; fi
    mkdir -p "$PULL/$A" "$RUNS/$A" && tar --force-local -xf "$PULL/$A.tar" -C "$PULL/$A" || { echo "[$(ts)] $A: untar failed"; pending=1; continue; }
    nf=$(ls "$PULL/$A/dump_$tag" 2>/dev/null | grep -c -- '^ep.*npz$' | tr -d '\r')
    if [ "$nf" != 141 ]; then echo "[$(ts)] $A: dump holds $nf ep files, not 141; retry next round"; pending=1; continue; fi
    for f in summary.json selfcheck.json argv.json; do timeout 120 bash "$C" "$RR/$A/$f" "$RUNS/$A/$f"; done
    ck=$(timeout 120 bash "$P" "sha256sum $RR/$A/ckpt_last.pt | cut -c1-64" 2>/dev/null | tr -d '\r' | tail -1)
    extra=""
    case "$A" in HIER_s*)
      timeout 900 bash "$P" "mkdir -p $PB/states && cd $PB && for sp in 'train 600 1800' 'eval 0 141'; do set -- \$sp; [ -f states/${A}_\$1.npz.json ] || nice -n 19 ionice -c3 /workspace/venv/bin/python extract_states.py --tree /workspace/v8proof/tree --ckpt $RR/$A/ckpt_last.pt --cache /workspace/v8proof/data/cache_pod --split \$1 --lo \$2 --hi \$3 --out states/${A}_\$1.npz > states/${A}_\$1.log 2>&1; done; ls states | grep -c -- '^${A}_.*npz.json\$'" > /dev/null 2>&1
      ok=1
      for sp in train eval; do
        timeout 300 bash "$C" "$PB/states/${A}_$sp.npz" "$ST/${A}_$sp.npz"; timeout 120 bash "$C" "$PB/states/${A}_$sp.npz.json" "$ST/${A}_$sp.npz.json"
        want=$(grep -o '"out_sha256": "[0-9a-f]*"' "$ST/${A}_$sp.npz.json" | cut -d'"' -f4)
        [ -n "$want" ] && [ "$(sha "$ST/${A}_$sp.npz")" = "$want" ] || ok=0
      done
      if [ $ok = 1 ]; then extra=" states ok"; else echo "[$(ts)] $A: state files not verified; retry next round"; pending=1; continue; fi ;;
    esac
    echo "pulled $(ts) tar_sha256 $rs ckpt_sha256 $ck files $nf" > "$PULL/$A/PULLED.txt"
    rm -f "${PULL:?}/${A:?}.tar"
    timeout 60 bash "$P" "rm -f ${PB:?}/dumps/${A:?}.tar" > /dev/null 2>&1
    echo "[$(ts)] $A: pulled (141 ep files, tar sha256 ${rs:0:16}, ckpt ${ck:0:16})$extra"
  done
  if [ "$MARK" = DONE ] || [ "$MARK" = STOP ]; then
    if [ $pending = 0 ]; then echo "[$(ts)] chain marker $MARK; every finished arm pulled: $(ls "$PULL" | tr '\n' ' ')"; exit 0; fi
  fi
  if [ $(( ($(date +%s) - t0) / 3600 )) -ge "$MAXH" ]; then echo "[$(ts)] TIMEOUT after $MAXH h; pulled so far: $(ls "$PULL" | tr '\n' ' ')"; exit 3; fi
  sleep 600
done
