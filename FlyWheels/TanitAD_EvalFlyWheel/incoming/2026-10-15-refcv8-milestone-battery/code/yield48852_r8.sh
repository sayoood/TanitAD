# yield48852_r8.sh -- SOURCED (needs $PY and $BAT).  NEW file (2026-10-19, EvalFlyWheel).  The sibling of yield40k_r8.sh for the FINAL checkpoint.
# The final chain (chain_final_r8.sh) is the registered read of the programme and outranks every other GPU job.  gpu_lock.py has no queue and no priority, and a chain blocked INSIDE
# `gpu_lock.py acquire --max-wait-s N` wins the next release regardless of any check made before the call -- so the acquire becomes a loop of SHORT acquires, and every pass first asks:
# bfinal_pending -> 0 (= YIELD, do not even try) iff
#   ckpt_full_48852.pt is banked (pull_final_r8.sh renames .part only on ACCEPT: md5 on both ends + stat stable)   AND
#   the final EMA battery_summary.json does not exist yet   OR   the K-draw rolls have no terminal marker yet     AND
#   chain_final_r8 is alive                                                                                        AND
#   its log has no ZZCFN(ABORT|END)ZZ after its last ZZCFNSTARTZZ, and (while the EMA battery is the open item) no ZZCFNEMAFAILZZ.
# Before the weights are banked it does NOT yield (the trainer may still be a day from done: idling the card for that would cost more than it buys).
# Any other state (no bank, chain dead or ended, EMA battery failed and K-draw terminal) -> not pending -> acquire normally.  Total yield per acquire is capped (Y48_CAP, 6 h).  A job never holds the lock while yielding.
# WIRING.  The chains running at 2026-10-19 (chain_r7_matched_phase, chain_r7on_50400, queue_k2k4_r8, chain_retime_trainfit_r8) were started BEFORE this file existed and source yield40k_r8.sh only:
# they cannot be edited under a live run.  Wire this into a chain that has NOT started yet, next to its `. yield40k_r8.sh`, and call acquire_y48 where acquire_y40 was called (it is a drop-in with the same arguments).
Y48_BANK="${Y48_BANK:-/e/refcv8_eval_kit/ckpt/ckpt_full_48852.pt}"
Y48_SUM="${Y48_SUM:-/c/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/battery_summary.json}"
Y48_LOG="${Y48_LOG:-/e/refcv8_eval_kit/chain/chain_final.log}"
Y48_CAP="${Y48_CAP:-21600}"
Y48_SLEEP="${Y48_SLEEP:-120}"
Y48_TRY_S="${Y48_TRY_S:-50}"
bfinal_pending() {
  [ -f "$Y48_BANK" ] || return 1
  _y48c=$(ps -ef 2>/dev/null | grep -c '[c]hain_final_r8'); [ "${_y48c:-0}" -ge 1 ] 2>/dev/null || return 1
  # the chain's own end / abort after its last START: nothing left to protect
  awk '/ZZCFNSTARTZZ/{e=0} /ZZCFN(ABORT|END)ZZ/{e=1} END{exit e}' "$Y48_LOG" 2>/dev/null || return 1
  # the open item: the EMA battery (no summary yet and no EMAFAIL), or the K-draw rolls (no terminal K-draw marker yet)
  if [ ! -s "$Y48_SUM" ]; then
    awk '/ZZCFNSTARTZZ/{e=0} /ZZCFNEMAFAILZZ/{e=1} END{exit e}' "$Y48_LOG" 2>/dev/null && return 0
  fi
  awk '/ZZCFNSTARTZZ/{k=0} /ZZCFNKD(DONE|HAVE|SKIP|FAIL)ZZ/{k=1} END{exit k}' "$Y48_LOG" 2>/dev/null && return 0
  return 1
}
# acquire_y48 <job> <out_json> <logfile> <max_wait_s> : 0 iff WE hold the lock (out_json then says ok true, as before)
acquire_y48() {
  _y48t0=$(date +%s); _y48y=0; _y48said=0
  while :; do
    if [ "$_y48y" -lt "$Y48_CAP" ] && bfinal_pending; then
      [ "$_y48said" = 0 ] && { echo "ZZY48YIELDZZ $(date +%FT%T%z) $1 yields to the final (step 48,852) chain" >> "$3"; _y48said=1; }
      sleep "$Y48_SLEEP"; _y48y=$((_y48y + Y48_SLEEP)); continue
    fi
    [ "$_y48said" = 1 ] && { echo "ZZY48RESUMEZZ $(date +%FT%T%z) $1 after ${_y48y} s" >> "$3"; _y48said=0; }
    _y48a=$(date +%s)
    "$PY" "$BAT/gpu_lock.py" acquire --job "$1" --pid $$ --max-wait-s "$Y48_TRY_S" --out "$2" >> "$3" 2>&1
    "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$2" 2>/dev/null && return 0
    # A REFUSAL is not a WAIT (same rule as acquire_y40): ok false with an "error" field, or ok false returned faster than one try window, means the tool gave up.
    "$PY" -c "import json,sys; sys.exit(0 if 'error' in json.load(open(sys.argv[1])) else 1)" "$2" 2>/dev/null && return 3
    [ $(( $(date +%s) - _y48a )) -lt "$Y48_TRY_S" ] && return 3
    [ $(( $(date +%s) - _y48t0 )) -gt "$4" ] && return 3
  done
}
