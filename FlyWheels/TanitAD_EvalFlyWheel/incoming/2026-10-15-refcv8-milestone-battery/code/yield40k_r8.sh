# yield40k_r8.sh -- SOURCED (needs $PY and $BAT). 2026-10-19 03:55 Berlin, orchestrator.
# The step-40k EMA battery (chain_40k_r8.sh) is the PI-ruled bar read and outranks the refcv7 chains and K2/K4. gpu_lock.py has
# no queue and no priority, and a chain blocked INSIDE `gpu_lock.py acquire --max-wait-s 172800` would win the next release
# regardless of any check made before the call. So the acquire becomes a loop of SHORT acquires, and every pass first asks:
# b40ema_pending -> 0 (= YIELD, do not even try) iff
#   ckpt_full_40000.pt is banked (snap40k renames .part only on ACCEPT)  AND
#   the 40k EMA battery_summary.json does not exist yet                  AND
#   chain_40k_r8 is alive                                                AND
#   its log has no ZZC40(EMAFAIL|ABORT|END)ZZ after its last ZZC40STARTZZ.
# Any other state (no snapshot, summary written, 40k chain dead or failed) -> not pending -> acquire normally.
# Total yield per acquire is capped (Y40_CAP, 6 h). A job never holds the lock while yielding.
Y40_CKPT="${Y40_CKPT:-/e/refcv8_eval_kit/ckpt/ckpt_full_40000.pt}"
Y40_SUM="${Y40_SUM:-/c/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/battery_summary.json}"
Y40_LOG="${Y40_LOG:-/e/refcv8_eval_kit/chain/chain_40k.log}"
Y40_CAP="${Y40_CAP:-21600}"
Y40_SLEEP="${Y40_SLEEP:-120}"
Y40_TRY_S="${Y40_TRY_S:-50}"
b40ema_pending() {
  [ -f "$Y40_CKPT" ] || return 1
  [ -s "$Y40_SUM" ] && return 1
  _y40c=$(ps -ef 2>/dev/null | grep -c '[c]hain_40k_r8'); [ "${_y40c:-0}" -ge 1 ] 2>/dev/null || return 1
  awk '/ZZC40STARTZZ/{e=0} /ZZC40(EMAFAIL|ABORT|END)ZZ/{e=1} END{exit e}' "$Y40_LOG" 2>/dev/null || return 1
  return 0
}
# acquire_y40 <job> <out_json> <logfile> <max_wait_s> : 0 iff WE hold the lock (out_json then says ok true, as before)
acquire_y40() {
  _y40t0=$(date +%s); _y40y=0; _y40said=0
  while :; do
    if [ "$_y40y" -lt "$Y40_CAP" ] && b40ema_pending; then
      [ "$_y40said" = 0 ] && { echo "ZZY40YIELDZZ $(date +%FT%T%z) $1 yields to the step-40k EMA battery" >> "$3"; _y40said=1; }
      sleep "$Y40_SLEEP"; _y40y=$((_y40y + Y40_SLEEP)); continue
    fi
    [ "$_y40said" = 1 ] && { echo "ZZY40RESUMEZZ $(date +%FT%T%z) $1 after ${_y40y} s" >> "$3"; _y40said=0; }
    _y40a=$(date +%s)
    "$PY" "$BAT/gpu_lock.py" acquire --job "$1" --pid $$ --max-wait-s "$Y40_TRY_S" --out "$2" >> "$3" 2>&1
    "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$2" 2>/dev/null && return 0
    # A REFUSAL is not a WAIT: ok false with an "error" field, or ok false returned faster than one try window, means the tool
    # gave up (crash / probe failure / stub) -- return as the old single blocking call did, never tight-loop for max_wait.
    "$PY" -c "import json,sys; sys.exit(0 if 'error' in json.load(open(sys.argv[1])) else 1)" "$2" 2>/dev/null && return 3
    [ $(( $(date +%s) - _y40a )) -lt "$Y40_TRY_S" ] && return 3
    [ $(( $(date +%s) - _y40t0 )) -gt "$4" ] && return 3
  done
}
