#!/bin/sh
# SPEC A4 lever L3 (deterministic DDIM, eps = 0) for ONE banked battery tag: a GPU roll behind the
# dev-box gate (lever_eps0.py waits on it; its parent never holds CUDA), then bank + LANDING_READY.
# usage: sh l3_tag.sh <tag> <ckpt path>
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
# SPEC A5: a post-switch tag (the FINAL) is rolled on the 82c2331 tree with the run's post-switch config:
# the caller sets L3_REPO / L3_CONFIG; a pre-switch tag keeps the defaults (its own training tree).
L3_REPO=${L3_REPO:-C:/Users/Admin/ev6}
L3_CONFIG=${L3_CONFIG:-D:/refcv6_eval_kit/ckpt/config.json}
export REFCV6_REPO="$L3_REPO"
export PYTHONPATH="$L3_REPO/stack;$L3_REPO/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
t=$1
ck=$2
WL=$B/raw/post_watch.log
cd $B/code || exit 1
if [ -s $B/raw/$t/levers/eps0.json ] && grep -q '"cells"' $B/raw/$t/levers/eps0.json; then
  echo "ZZL3SKIP_${t}ZZ already done $(date +%FT%T)" >> $WL; exit 0
fi
[ -s $B/raw/$t/panel_s0/manifest.json ] || { echo "ZZL3NOPANEL_${t}ZZ $(date +%FT%T)" >> $WL; exit 1; }
# Master Mind priority (2026-09-26): A6 before L3. If the A6 chain has started and not ended, wait for it
# (at most 10 h; a stuck A6 must not starve the rest of the queue).
AL=$B/raw/a6/a6_chain.log
if [ -f $AL ] && grep -q ZZA6STARTZZ $AL; then
  n=0
  until grep -qE "ZZA6(DONE|FAIL)ZZ" $AL || [ $n -ge 300 ]; do sleep 120; n=$((n+1)); done
fi
# ... and the FINAL-gate dry run on the 35k snapshot (gate_dryrun_35k.sh), which runs right after A6
DL=$B/raw/step35000_gate/dryrun.log
if [ -f $DL ] && grep -q ZZDRYSTARTZZ $DL; then
  n=0
  until grep -qE "ZZDRY(DONE|FAIL)ZZ" $DL || [ $n -ge 150 ]; do sleep 120; n=$((n+1)); done
fi
echo "ZZL3START_${t}ZZ $(date +%FT%T)" >> $WL
$PY lever_eps0.py "$BW/raw/$t" --ckpt "$ck" --config "$L3_CONFIG" > $B/raw/$t/levers_eps0.log 2>&1
if [ -s $B/raw/$t/levers/eps0.json ] && grep -q '"cells"' $B/raw/$t/levers/eps0.json; then
  if $PY bank_tag.py "$BW/raw/$t/levers" "$PKG/raw/$t/levers" > $B/raw/$t/bank_levers_eps0.log 2>&1; then
    { echo ""; echo "## $(date +%F) battery tag $t: A4 lever L3, deterministic DDIM eps = 0 (l3_tag.sh; sanitized)"
      for f in eps0.json EPS0.md; do [ -f "$PKG/raw/$t/levers/$f" ] && echo "$REL/raw/$t/levers/$f"; done
    } >> "$PKG/LANDING_READY.txt"
    echo "ZZL3DONE_${t}ZZ $(date +%FT%T)" >> $WL
  else
    echo "ZZL3BANKFAIL_${t}ZZ $(date +%FT%T)" >> $WL
  fi
else
  echo "ZZL3FAIL_${t}ZZ $(date +%FT%T)" >> $WL
fi
