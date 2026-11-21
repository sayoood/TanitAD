#!/bin/sh
# SPEC A5: the FINAL battery on the 82c2331 tree (the post-switch A16 hybrid). It replaces
# chain_milestones.sh's final stage, which is DEFERRED (pull_final.sh now prints ZZFINALDEFERREDZZ).
#   1. pull_final_v2.sh: wait for the run's summary.json ("done": true), then the read-only 3-way-md5
#      pull of ckpt.pt + metrics.jsonl + config.json;
#   2. run_battery.py on the 82c2331 tree (REFCV6_REPO + PYTHONPATH) with the run's post-switch config
#      (its --clip-clock-sidecar is remapped to the md5-equal kit copy): full G0 + A1 + A2, 2 seeds;
#   3. bank (sanitized) + LANDING_READY, post_tag.sh (A4 levers, A5 dual-clock TACTICAL), L3 eps0 roll on
#      the same tree, and the cross-checkpoint CURVE over every banked tag.
# Every GPU stage waits on the dev-box gate. Markers: raw/final_v2.log (own) + the chain's final markers
# in raw/chain.log (ZZBFINALDONEZZ / ZZBANKED_finalZZ) for anyone reading the chain log.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
NEW='C:/Users/Admin/ev6_82c2331'
RESUME_CFG_MD5=a3193a4685ce0d07a6ae6b89fdc994a8
LOG=$B/raw/chain.log
FL=$B/raw/final_v2.log
cd $B/code || exit 1
echo "ZZFINALV2STARTZZ $(date +%FT%T)" >> $FL
sh pull_final_v2.sh > $B/raw/pull_final_v2.out 2>&1
out=$(grep -E 'ZZFINAL(OK|FAIL)ZZ' $B/raw/pull_final_v2.out | tail -1)
echo "$(date +%FT%T) $out" >> $FL
case "$out" in
  *ZZFINALOKZZ*)
    M=$(echo "$out" | sed -n 's/.*metrics \(metrics_[0-9T]*\.jsonl\).*/\1/p')
    C=$(echo "$out" | sed -n 's/.*config \(config_final_[0-9T]*\.json\).*/\1/p')
    CM=$(echo "$out" | sed -n 's/.*cfgmd5 \([0-9a-f]*\).*/\1/p')
    if [ "$CM" = "$RESUME_CFG_MD5" ]; then echo "ZZFINALCFGSAMEZZ $CM (== the resume config)" >> $FL
    else echo "ZZFINALCFGCHANGEDZZ $CM (resume config was $RESUME_CFG_MD5; the pulled one is used)" >> $FL; fi
    REFCV6_REPO="$NEW" PYTHONPATH="$NEW/stack;$NEW/taniteval" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_final.pt --tag final \
        --config "$BW/raw/thor_reads/$C" --metrics "$BW/raw/thor_reads/$M" > $B/raw/battery_final.log 2>&1
    echo "ZZBFINALDONEZZ $(date +%FT%T) (chain_final_v2.sh, 82c2331 tree)" >> $LOG
    echo "ZZBFINALDONEZZ $(date +%FT%T)" >> $FL
    if [ -s $B/raw/final/battery_summary.json ]; then
      export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" PYTHONIOENCODING=utf-8
      $PY summarize_battery.py "$BW/raw/final" 0 > $B/raw/final/TABLES_s0.md 2>&1
      [ -s $B/raw/final/analysis_s1.json ] && $PY summarize_battery.py "$BW/raw/final" 1 > $B/raw/final/TABLES_s1.md 2>&1
      if $PY bank_tag.py "$BW/raw/final" "$PKG/raw/final" --with-dumps > $B/raw/final/bank.log 2>&1; then
        { echo ""; echo "## $(date +%F) battery tag FINAL (chain_final_v2.sh, 82c2331 tree; sanitized, sha12 ids)"
          for f in $(ls "$PKG/raw/final"); do [ -f "$PKG/raw/final/$f" ] && echo "$REL/raw/final/$f"; done
        } >> "$PKG/LANDING_READY.txt"
        echo "ZZBANKED_finalZZ $(date +%FT%T)" >> $LOG
        echo "ZZBANKED_finalZZ $(date +%FT%T)" >> $FL
      else
        echo "ZZBANKFAIL_finalZZ $(date +%FT%T)" >> $LOG
        echo "ZZBANKFAIL_finalZZ $(date +%FT%T)" >> $FL
      fi
      unset PYTHONPATH
      env -u REFCV6_REPO sh $B/code/post_tag.sh final
      L3_REPO="$NEW" L3_CONFIG="$BW/raw/thor_reads/$C" sh $B/code/l3_tag.sh final D:/refcv6_eval_kit/ckpt/ckpt_final.pt
    fi ;;
  *) echo "ZZNOFINALV2ZZ $(date +%FT%T) $out" >> $FL ;;
esac
# the cross-checkpoint table over every banked tag (a TABLE, not a fit: SPEC A5 forbids a curve fit across 34,500)
tags=""
for t in step5000 step15000 step20000 step30000 final; do [ -s $B/raw/$t/battery_summary.json ] && tags="$tags $t"; done
PYTHONIOENCODING=utf-8 $PY curve_table.py "$BW/raw" $tags > $B/raw/CURVE.md 2>&1
$PY sanitize_for_bank.py "$BW/raw/CURVE.md" "$PKG/raw" > /dev/null 2>&1
{ echo ""; echo "## $(date +%F) cross-checkpoint table over ALL tags (chain_final_v2.sh)"; echo "$REL/raw/CURVE.md"; } >> "$PKG/LANDING_READY.txt"
echo "ZZFINALV2ENDZZ tags:$tags $(date +%FT%T)" >> $FL
