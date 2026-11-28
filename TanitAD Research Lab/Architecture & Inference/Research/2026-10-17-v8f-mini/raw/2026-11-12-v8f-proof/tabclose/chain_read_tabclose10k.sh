#!/usr/bin/env bash
# chain_read_tabclose10k.sh -- pod: waits for the queue of card ADD84 (TABCLOSE10K_DONE.txt, or TABCLOSE10K_DONE_<part>.txt with TB_PART=A|B; TABCLOSE10K_STOP.txt) AND for the go file GO_READ_TABCLOSE10K.txt
# (written by the lead once the hold that carries the card is landed), links the new run folders into the pull folder of the readers and runs podread_tabclose10k.sh once
# (output folder read_tabclose10k, or read_tabclose10k_<part> with a part switch).
# CPU only. No process listing or stopping, no file removal or move. Refuses to run twice (START_READ_TABCLOSE10K[_part].txt). The wrapper hash of the queue is checked here as well.
# Last line of the log: ZZ CHAIN-READ-TABCLOSE10K END <what>.
set -u
PART=${TB_PART:-ALL}
case "$PART" in A|B|ALL) ;; *) echo "REFUSED: TB_PART must be A, B or ALL"; exit 2;; esac
PS=""; [ "$PART" = ALL ] || PS="_$PART"
W=/workspace/v8proof/integ10k
P=/workspace/v8proof/podread
RD=$P/root/integ10k
PL=$RD/pull
OUT=$P/out/read_tabclose10k$PS
EJ=$P/out/read_estclose10k_n2/READ_ESTCLOSE10K.json
SD=/workspace/v8proof/closeread/est_tab
WSHA=02a58e655e5c1979
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
( set -C; ts > "$W/START_READ_TABCLOSE10K$PS.txt" ) 2>/dev/null || { echo "REFUSED: START_READ_TABCLOSE10K$PS.txt exists"; exit 2; }
echo "[$(ts)] chain start, waiting for the queue and the go file"
n=0
while :; do
  [ -e "$W/TABCLOSE10K_STOP.txt" ] && { echo "ZZ CHAIN-READ-TABCLOSE10K END QUEUE-STOPPED $(cut -c1-200 "$W/TABCLOSE10K_STOP.txt")"; exit 3; }
  [ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break
  n=$((n + 1)); [ "$n" -le 300 ] || { echo "ZZ CHAIN-READ-TABCLOSE10K END WAIT-TIMEOUT"; exit 3; }
  sleep 60
done
echo "[$(ts)] queue done: $(cut -c1-120 "$W/TABCLOSE10K_DONE$PS.txt"); go: $(cut -c1-160 "$W/GO_READ_TABCLOSE10K.txt")"
got=$(sha256sum "$W/build/v8f_t1_eval_reader_rule_rzgfhqet.py" | cut -c1-16)
[ "$got" = "$WSHA" ] || { echo "ZZ CHAIN-READ-TABCLOSE10K END WRAPPER-SHA $got"; exit 3; }
mkdir -p "$PL"
for d in "$W"/t1/I10RQ50e{TK,STK,L100,W00,W50,W70,W90,S70}*_s[01]; do nm=$(basename "$d"); [ -e "$PL/$nm" ] || ln -s "$d" "$PL/$nm"; done
echo "[$(ts)] links of the new runs: $(ls "$PL" | grep -c '^I10RQ50e')"
cd "$RD" || { echo "ZZ CHAIN-READ-TABCLOSE10K END NO-READ-DIR"; exit 3; }
HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz bash "$P/podread_tabclose10k.sh" \
  --pull "$PL" --code "$P/code" --grid-dir "$W/longroll" --fresh-grid /workspace/v8proof/hierrep/pins/eval_grid_windows.json \
  --midlook-json "$P/out/read_midlook10k/READ_MIDLOOK10K.json" --estclose-json "$EJ" --tab-dir "$SD" --part "$PART" --w "$W" --tree "$W/tree" \
  --out "$OUT" --wrapper-sha "$WSHA" --read-dir "$RD"
rc=$?
echo "[$(ts)] podread rc $rc"
echo "ZZ CHAIN-READ-TABCLOSE10K END rc $rc"
