#!/bin/sh
# r7m_analysis.sh -- the CPU analyses of PREREG_R7_MATCHED_PHASE_TURN.md, run as soon as their inputs exist (detached child of chain_r7_matched_phase.sh;
# also runnable by hand).  No GPU, no lock.  Nine heading_drift runs + the paired slope contrasts + the ceiling-instrument contrasts, each skipped when its
# output exists, each waiting for COMPLETE trees (battery_summary.json with bars + finished + r8 stamp, never a smoke), polling every 120 s, <= 192 h.
#   a(N)   : --new-root r7step<N>_unc (R7-N), --earlier = the other refcv7 trees already done          -> s7(N), P7(N) vs the embedded refcv7-50,400 arm
#   b(N,e) : --new-root step<N>ema_unc (EMA8-N), --earlier R7-N                                        -> D_ema(N) (point; paired CI: r7m_slope_diff.py)
#   b(N,r) : --new-root step<N>_unc    (RAW8-N), --earlier R7-N                                        -> D_raw(N)
#   sd(N)  : r7m_slope_diff.py (per-seed + pooled P7, D_ema, D_raw, phase fraction, mechanical read of the pre-registered table)
#   ce(N)  : r7m_ceiling_effect.py --contrast EMA8-N:R7-N  (the SAME contrasts on the ceiling-ON and ceiling-OFF instruments; seed 0; needs the filter_off arm)
# Outputs: raw/r7_matched_phase/HEADING_DRIFT_r7m_<what>_<N>.{json,md}, R7M_SLOPE_DIFF_<N>[_ema].{json,md}, R7M_CEILING_<N>.{json,md}.
# Markers ZZR7MHD...ZZ in E:/refcv8_eval_kit/chain/chain_r7_matched_phase.log.  Assert on ARTIFACTS.
. "$(dirname "$0")/r7m_common.sh"
PIDF="$KIT/chain/r7m_analysis.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark HDDUP "analysis daemon pid $(cat "$PIDF") alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
mark HDSTART "pid $$"
lab() { case "$1" in 15000) echo 15k ;; 20000) echo 20k ;; 30000) echo 30k ;; esac; }
tree_done() {   # $1 = tree dir name under the archive
  jq_ok "$ARCL/$1/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j and not j.get('SMOKE_RESTRICTED_S2')"
}
have_json() { jq_ok "$1" "'surfaces' in j or 'trees' in j"; }
hd_job() {   # $1 tag  $2 new tree  $3 new label  rest = LABEL=treedir (earlier)       -> 0 done, 1 not ready, 2 given up, 3 failed this time
  tag=$1; newdir=$2; newlab=$3; shift 3
  out="$RAWL/$tag.json"
  have_json "$out" && return 0
  tree_done "$newdir" || return 1
  for e in "$@"; do tree_done "${e#*=}" || return 1; done
  nA=$(cat "$KIT/chain/r7m_attempts_$tag" 2>/dev/null || echo 0)
  [ "$nA" -ge 2 ] && return 2
  echo $((nA+1)) > "$KIT/chain/r7m_attempts_$tag"
  mark HDRUN "$tag"
  eargs=""; for e in "$@"; do eargs="$eargs ${e%%=*}=$ARCW/${e#*=}"; done
  ( cd "$PKGL" && OMP_NUM_THREADS=6 "$PY" -X utf8 raw/heading_drift/heading_drift.py --new-root "$ARCW/$newdir" --label-new "$newlab" \
        ${eargs:+--earlier $eargs} --out-dir "$RAWW" --tag "$tag" > "$RAWL/run_$tag.log" 2>&1 )
  if have_json "$out"; then mark HDDONE "$tag -> $RAWW/$tag.md"; return 0; fi
  mark HDFAIL "$tag wrote no json (see $RAWW/run_$tag.log)"; return 3
}
sd_job() {   # $1 N  $2 mode (ema|both)
  N=$1; L=$(lab "$N"); r7=r7step${N}_unc; em=step${N}ema_unc; rw=step${N}_unc
  if [ "$2" = ema ]; then out="$RAWL/R7M_SLOPE_DIFF_${N}_ema.json"; else out="$RAWL/R7M_SLOPE_DIFF_${N}.json"; fi
  have_json "$out" && return 0; [ -s "$out" ] && jq_ok "$out" "'surfaces' in j" && return 0
  tree_done "$r7" || return 1; tree_done "$em" || return 1
  [ "$2" = both ] && { tree_done "$rw" || return 1; }
  nA=$(cat "$KIT/chain/r7m_attempts_sd_${N}_$2" 2>/dev/null || echo 0); [ "$nA" -ge 2 ] && return 2
  echo $((nA+1)) > "$KIT/chain/r7m_attempts_sd_${N}_$2"
  rawarg=""; [ "$2" = both ] && rawarg="--raw-tree $ARCW/$rw"
  base="R7M_SLOPE_DIFF_${N}"; [ "$2" = ema ] && base="${base}_ema"
  ( cd "$PKGL" && OMP_NUM_THREADS=6 "$PY" -X utf8 code/r7m_slope_diff.py --label "$N" --r7-tree "$ARCW/$r7" --ema-tree "$ARCW/$em" $rawarg \
        --out-json "$RAWW/$base.json" --out-md "$RAWW/$base.md" > "$RAWL/run_$base.log" 2>&1 )
  if jq_ok "$out" "'surfaces' in j"; then mark SDDONE "$base"; return 0; fi
  mark SDFAIL "$base (see run_$base.log)"; return 3
}
ce_job() {   # $1 N : the ceiling-instrument contrasts (needs the filter_off arm = a seed-0 full roll with the filter acting)
  N=$1; L=$(lab "$N"); out="$RAWL/R7M_CEILING_${N}.json"
  jq_ok "$out" "'trees' in j" && return 0
  tree_done "r7step${N}_unc" || return 1; tree_done "step${N}ema_unc" || return 1; tree_done "step${N}_unc" || return 1
  nA=$(cat "$KIT/chain/r7m_attempts_ce_${N}" 2>/dev/null || echo 0); [ "$nA" -ge 2 ] && return 2
  echo $((nA+1)) > "$KIT/chain/r7m_attempts_ce_${N}"
  ( cd "$PKGL" && OMP_NUM_THREADS=6 "$PY" -X utf8 code/r7m_ceiling_effect.py --seeds 0 \
        --tree R7-$L=$ARCW/r7step${N}_unc --tree EMA8-$L=$ARCW/step${N}ema_unc --tree RAW8-$L=$ARCW/step${N}_unc \
        --contrast EMA8-$L:R7-$L --contrast RAW8-$L:R7-$L --out-json "$RAWW/R7M_CEILING_${N}.json" --out-md "$RAWW/R7M_CEILING_${N}.md" \
        > "$RAWL/run_R7M_CEILING_${N}.log" 2>&1 )
  if jq_ok "$out" "'trees' in j"; then mark CEDONE "R7M_CEILING_${N}"; return 0; fi
  mark CEFAIL "R7M_CEILING_${N} (see run_R7M_CEILING_${N}.log)"; return 3
}
pass_once() {   # returns the number of jobs still pending (not done, not given up)
  pend=0
  for N in 20000 30000 15000; do
    L=$(lab "$N")
    case "$N" in
      20000) EARL="" ;;
      30000) EARL="R7-20k=r7step20000_unc" ;;
      15000) EARL="R7-30k=r7step30000_unc R7-20k=r7step20000_unc" ;;
    esac
    hd_job "HEADING_DRIFT_r7m_r7_$N" "r7step${N}_unc" "R7-$L" $EARL;                                  [ $? -eq 1 ] && pend=$((pend+1))
    hd_job "HEADING_DRIFT_r7m_ema8_$N" "step${N}ema_unc" "EMA8-$L" "R7-$L=r7step${N}_unc";            [ $? -eq 1 ] && pend=$((pend+1))
    hd_job "HEADING_DRIFT_r7m_raw8_$N" "step${N}_unc" "RAW8-$L" "R7-$L=r7step${N}_unc";               [ $? -eq 1 ] && pend=$((pend+1))
    sd_job "$N" ema;                                                                                   [ $? -eq 1 ] && pend=$((pend+1))
    sd_job "$N" both;                                                                                  [ $? -eq 1 ] && pend=$((pend+1))
    ce_job "$N";                                                                                       [ $? -eq 1 ] && pend=$((pend+1))
  done
  return $pend
}
n=0
while :; do
  pass_once; pend=$?
  [ "$pend" -eq 0 ] && break
  n=$((n+1)); [ $((n % 30)) -eq 1 ] && mark HDPOLL "pending jobs: $pend"
  [ "$n" -ge 5760 ] && { mark HDTIMEOUT "192 h; pending $pend"; break; }
  sleep 120
done
mark HDEND "no pending analysis job (done or given up)"
