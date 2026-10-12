
# ---------------------------------------------------------------------------------------------------- noLW main
# PI 2026-10-12: "yes, run V-R8-noLW after the ladder". Functions above are ladder_chain.sh lines 1-196 VERBATIM
# (md5 f71a25fd of the source). W = the noLW dir, never the ladder's W. Waits for the ladder chain PID to exit, then
# starts ONLY if the chain's last ZZSTARTZZ segment holds ZZALLDONEZZ and no ZZSTOPZZ. Trains under the same GPU lock,
# evaluates the `base` row on Thor (batch 8, both sampler seeds), strips to model_final.pt + md5, touches PULL_READY.
rows_for() { echo base; }
LW=${LW:-/home/nvidia/refcv8_ladder/W}
CHAIN_PID=${CHAIN_PID:?CHAIN_PID required}
log "ZZNOLW-WAIT chain pid $CHAIN_PID"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 300; done
seg=$(awk '/ZZSTARTZZ/{s=""} {s=s"\n"$0} END{print s}' "$LW/chain.log")
printf '%s\n' "$seg" | grep -q "ZZSTOPZZ" && { log "ZZNOLW-NOT-STARTED the ladder chain logged ZZSTOPZZ"; exit 4; }
printf '%s\n' "$seg" | grep -q "ZZALLDONEZZ" || { log "ZZNOLW-NOT-STARTED the ladder chain ended without ZZALLDONEZZ"; exit 4; }
preflight
N=$(cat "$LW/N.txt"); [ "$N" = 3000 ] || die "ladder N is '$N', expected 3000"
log "ZZNOLW-START N=$N argv_sha256=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["argv_sha256"])' "$W/argv/V-R8-noLW.json")"
train_arm V-R8-noLW
eval_arm V-R8-noLW "$N"
log "ZZNOLW-ALLDONE"
