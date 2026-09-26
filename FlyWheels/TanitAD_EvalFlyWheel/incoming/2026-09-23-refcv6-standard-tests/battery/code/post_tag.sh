#!/bin/sh
# Post-bank processing of ONE battery tag, CPU only (called by post_chain_watch.sh after the chain has
# banked the tag): SPEC A3 recompute (only where the panels predate A3), the A4 lever panel
# (L1 / L2 / L2e), re-rendered tables, re-bank (sanitized), paths appended to LANDING_READY.txt.
# Markers go to raw/post_watch.log (never to chain.log, which other watchers grep).
B=${POST_B:-/c/Users/Admin/ev6_battery}
BW=${POST_BW:-C:/Users/Admin/ev6_battery}
PKG=${POST_PKG:-/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery}
CODE=/c/Users/Admin/ev6_battery/code
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
t=$1
WL=$B/raw/post_watch.log
cd $CODE || exit 1
[ -s $B/raw/$t/battery_summary.json ] || { echo "ZZPOSTNOTAG_${t}ZZ $(date +%FT%T)" >> $WL; exit 1; }
$PY recompute_a3.py "$BW/raw/$t" > $B/raw/$t/A3_recompute.log 2>&1
a3=$?
$PY lever_panel.py "$BW/raw/$t" > $B/raw/$t/levers.log 2>&1
lv=$?
$PY summarize_battery.py "$BW/raw/$t" 0 > $B/raw/$t/TABLES_s0.md 2>&1
[ -s $B/raw/$t/analysis_s1.json ] && $PY summarize_battery.py "$BW/raw/$t" 1 > $B/raw/$t/TABLES_s1.md 2>&1
$PY bank_tag.py "$BW/raw/$t" "$PKG/raw/$t" > $B/raw/$t/bank_post.log 2>&1 || { echo "ZZPOSTBANKFAIL_${t}ZZ $(date +%FT%T)" >> $WL; exit 1; }
if [ -s $B/raw/$t/levers/levers.json ]; then
  $PY bank_tag.py "$BW/raw/$t/levers" "$PKG/raw/$t/levers" >> $B/raw/$t/bank_post.log 2>&1 || echo "ZZPOSTLEVERBANKFAIL_${t}ZZ $(date +%FT%T)" >> $WL
fi
{ echo ""; echo "## $(date +%F) battery tag $t: SPEC A3 recompute + A4 lever panel + re-rendered tables (post_tag.sh; sanitized, sha12 ids)"
  for f in $(ls "$PKG/raw/$t"); do [ -f "$PKG/raw/$t/$f" ] && echo "$REL/raw/$t/$f"; done
  if [ -d "$PKG/raw/$t/levers" ]; then for f in $(ls "$PKG/raw/$t/levers"); do echo "$REL/raw/$t/levers/$f"; done; fi
} >> "$PKG/LANDING_READY.txt"
echo "ZZPOSTDONE_${t}ZZ a3=$a3 levers=$lv $(date +%FT%T)" >> $WL
