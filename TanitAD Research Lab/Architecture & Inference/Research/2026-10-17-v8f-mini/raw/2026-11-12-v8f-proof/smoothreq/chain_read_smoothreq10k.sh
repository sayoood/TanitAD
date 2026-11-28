#!/usr/bin/env bash
# chain_read_smoothreq10k.sh -- pod: waits for the queue of card ADD85 (SMOOTHREQ10K_DONE.txt, or both SMOOTHREQ10K_DONE_s0.txt and SMOOTHREQ10K_DONE_s1.txt; any SMOOTHREQ10K_STOP*.txt ends it) AND for the go file
# GO_READ_SMOOTHREQ10K.txt in $W (written by the lead once the hold that carries the card is landed), links the run folders into the pull folder of the readers and runs podread_smoothreq10k.sh once.
# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_SMOOTHREQ10K.txt). The wrapper hash of the queue is checked here as well.
# Last line of the log: ZZ CHAIN-READ-SMOOTHREQ10K END <what>.
set -u
W=/workspace/v8proof/integ10k
P=/workspace/v8proof/podread
RD=$P/root/integ10k
PL=$RD/pull
OUT=$P/out/read_smoothreq10k
EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json
SD=/workspace/v8proof/closeread/est_smooth
TBJ=${SR_TBJSON:-$P/out/read_tabclose10k/READ_TABCLOSE10K.json}
WSHA=02a58e655e5c1979
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
( set -C; ts > "$W/START_READ_SMOOTHREQ10K.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_SMOOTHREQ10K.txt exists"; exit 2; }
echo "[$(ts)] chain start, waiting for the queue and the go file"
n=0
while :; do
  for sf in "$W"/SMOOTHREQ10K_STOP*.txt; do [ -e "$sf" ] && { echo "ZZ CHAIN-READ-SMOOTHREQ10K END QUEUE-STOPPED $(cut -c1-200 "$sf")"; exit 3; }; done
  { [ -f "$W/SMOOTHREQ10K_DONE.txt" ] || { [ -f "$W/SMOOTHREQ10K_DONE_s0.txt" ] && [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]; }; } && [ -f "$W/GO_READ_SMOOTHREQ10K.txt" ] && break
  n=$((n + 1)); [ "$n" -le 300 ] || { echo "ZZ CHAIN-READ-SMOOTHREQ10K END WAIT-TIMEOUT"; exit 3; }
  sleep 60
done
echo "[$(ts)] queue done (done markers: $(ls "$W" | grep -c "^SMOOTHREQ10K_DONE")); go: $(cut -c1-160 "$W/GO_READ_SMOOTHREQ10K.txt")"
got=$(sha256sum "$W/build/v8f_t1_eval_reader_rule_rzgfhqet.py" | cut -c1-16)
[ "$got" = "$WSHA" ] || { echo "ZZ CHAIN-READ-SMOOTHREQ10K END WRAPPER-SHA $got"; exit 3; }
mkdir -p "$PL"
for d in "$W"/t1/I10RQ50e{STKM,TKD,Z00,S90,S50,S7B,W7B,TK,STK,L100,W00,W50,W70,W90,S70}KF5P30Hb20_s[01]; do nm=$(basename "$d"); [ -e "$PL/$nm" ] || ln -s "$d" "$PL/$nm"; done
echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50e')"
cd "$RD" || { echo "ZZ CHAIN-READ-SMOOTHREQ10K END NO-READ-DIR"; exit 3; }
HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz bash "$P/podread_smoothreq10k.sh" \
  --pull "$PL" --code "$P/code" --grid-dir "$W/longroll" --fresh-grid /workspace/v8proof/hierrep/pins/eval_grid_windows.json \
  --midlook-json "$P/out/read_midlook10k/READ_MIDLOOK10K.json" --estclose-json "$EJ" --tabclose-json "$TBJ" --tab-dir "$SD" --w "$W" --tree "$W/tree" \
  --out "$OUT" --wrapper-sha "$WSHA" --read-dir "$RD"
rc=$?
echo "[$(ts)] podread rc $rc"
echo "ZZ CHAIN-READ-SMOOTHREQ10K END rc $rc"
