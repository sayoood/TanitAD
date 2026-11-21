#!/usr/bin/env bash
# chain_read_constclose10k.sh -- pod: waits for the queue of card ADD80 (CONSTCLOSE10K_DONE.txt / CONSTCLOSE10K_STOP.txt) AND for the go file GO_READ_CONSTCLOSE10K.txt (written by the
# lead once the hold that carries the card is landed), links the new run folders into the pull folder of the readers and runs podread_constclose10k_n1.sh once (reader of note N1).
# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_CONSTCLOSE10K.txt).
# Last line of the log: ZZ CHAIN-READ-CONSTCLOSE10K END <what>.
set -u
W=/workspace/v8proof/integ10k
P=/workspace/v8proof/podread
RD=$P/root/integ10k
PL=$RD/pull
OUT=$P/out/read_constclose10k
EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
( set -C; ts > "$W/START_READ_CONSTCLOSE10K.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_CONSTCLOSE10K.txt exists"; exit 2; }
echo "[$(ts)] chain start, waiting for the queue and the go file"
n=0
while :; do
  [ -e "$W/CONSTCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-CONSTCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/CONSTCLOSE10K_STOP.txt")"; exit 3; }
  [ -f "$W/CONSTCLOSE10K_DONE.txt" ] && [ -f "$W/GO_READ_CONSTCLOSE10K.txt" ] && break
  n=$((n + 1)); [ "$n" -le 300 ] || { echo "ZZ CHAIN-READ-CONSTCLOSE10K END WAIT-TIMEOUT"; exit 3; }
  sleep 60
done
echo "[$(ts)] queue done: $(cut -c1-120 "$W/CONSTCLOSE10K_DONE.txt"); go: $(cut -c1-160 "$W/GO_READ_CONSTCLOSE10K.txt")"
mkdir -p "$PL"
for d in "$W"/t1/I10RQ50eC*_s[01]; do nm=$(basename "$d"); [ -e "$PL/$nm" ] || ln -s "$d" "$PL/$nm"; done
echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50eC')"
cd "$RD" || { echo "ZZ CHAIN-READ-CONSTCLOSE10K END NO-READ-DIR"; exit 3; }
HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz bash "$P/podread_constclose10k_n1.sh" \
  --pull "$PL" --code "$P/code" --grid-dir "$W/longroll" --fresh-grid /workspace/v8proof/hierrep/pins/eval_grid_windows.json \
  --midlook-json "$P/out/read_midlook10k/READ_MIDLOOK10K.json" --const-dir /workspace/v8proof/closeread/est_const --w "$W" --tree "$W/tree" \
  --out "$OUT" --wrapper-sha a747abeb51ef041f --estclose-json "$EJ" --read-dir "$RD"
rc=$?
echo "[$(ts)] podread rc $rc"
echo "ZZ CHAIN-READ-CONSTCLOSE10K END rc $rc"
