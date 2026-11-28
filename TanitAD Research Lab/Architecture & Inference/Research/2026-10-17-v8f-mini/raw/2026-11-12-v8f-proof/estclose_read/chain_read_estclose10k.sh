#!/usr/bin/env bash
# chain_read_estclose10k.sh -- pod: waits for the queue of card ADD79 (ESTCLOSE10K_DONE.txt / ESTCLOSE10K_STOP.txt), links the new run folders into the pull folder
# of the readers and runs podread_estclose10k.sh once. CPU only. It deletes, moves and stops nothing. Refuses to run twice (START_READ_ESTCLOSE10K.txt).
# Last line of the log: ZZ CHAIN-READ-ESTCLOSE10K END <what>.
set -u
W=/workspace/v8proof/integ10k
P=/workspace/v8proof/podread
RD=$P/root/integ10k
PL=$RD/pull
OUT=$P/out/read_estclose10k
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
( set -C; ts > "$W/START_READ_ESTCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_ESTCLOSE10K.txt exists"; exit 2; }
echo "[$(ts)] chain start, waiting for the queue"
n=0
while [ ! -f "$W/ESTCLOSE10K_DONE.txt" ] && [ ! -e "$W/ESTCLOSE10K_STOP.txt" ]; do
  n=$((n + 1)); [ "$n" -le 240 ] || { echo "ZZ CHAIN-READ-ESTCLOSE10K END WAIT-TIMEOUT"; exit 3; }
  sleep 60
done
[ -e "$W/ESTCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-ESTCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/ESTCLOSE10K_STOP.txt")"; exit 3; }
echo "[$(ts)] queue done: $(cut -c1-120 "$W/ESTCLOSE10K_DONE.txt")"
mkdir -p "$PL"
for d in "$W"/t1/I10RQ50e*_s[01]; do nm=$(basename "$d"); [ -e "$PL/$nm" ] || ln -s "$d" "$PL/$nm"; done
echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50e')"
cd "$RD" || { echo "ZZ CHAIN-READ-ESTCLOSE10K END NO-READ-DIR"; exit 3; }
HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz bash "$P/podread_estclose10k.sh" \
  --pull "$PL" --code "$P/code" --grid-dir "$W/longroll" --fresh-grid /workspace/v8proof/hierrep/pins/eval_grid_windows.json \
  --midlook-json "$P/out/read_midlook10k/READ_MIDLOOK10K.json" --est-dir /workspace/v8proof/closeread/est --w "$W" --tree "$W/tree" \
  --out "$OUT" --wrapper-sha a747abeb51ef041f --read-dir "$RD"
rc=$?
echo "[$(ts)] podread rc $rc"
echo "ZZ CHAIN-READ-ESTCLOSE10K END rc $rc"
