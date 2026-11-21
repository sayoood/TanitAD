#!/usr/bin/env bash
# RECOVER-FT: pull the six runs' checkpoints, summaries and markers from box 2 to the dev box, and verify each ckpt's sha256 against the box-2 marker. Run AFTER CHAIN_RECOVER.done exists (box 2's USB HDD: a ~0.5 GB read, no telemetry needed).
# Usage: bash pull_recft.sh            (idempotent: re-pulls what is missing or differs)
set -u
DEST=${RECFT_DEST:-/e/Projects/TanitAD-artifacts/v8f-recover}
R=D:/tanitad/runs/v8f_mini_full
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$DEST/recft_markers/logs" "$DEST/recft_markers/pin"
$SCP -r "tanitad-box2:$R/recft/*" "$DEST/recft_markers/" 2>&1 | grep -v '^\*\*'
bad=0
for t in R_0:M2b_s0_recft_R C0_0:M2b_s0_recft_C0 R_1:M2b_s1_recft_R C0_1:M2b_s1_recft_C0 C0rep_0:M2b_s0_recft_C0rep C0rep_1:M2b_s1_recft_C0rep; do
  tag=${t%%:*}; run=${t##*:}; d="$DEST/$run"; mkdir -p "$d"
  [ -f "$DEST/recft_markers/RUN_$tag.done" ] || { echo "SKIP $tag: no RUN_$tag.done (see RUN_$tag.FAILED)"; bad=1; continue; }
  for f in ckpt_last.pt summary.json argv.json selfcheck.json train_log.jsonl train.log; do $SCP "tanitad-box2:$R/$run/$f" "$d/$f" 2>/dev/null || echo "  (no $f for $run)"; done
  $SCP "tanitad-box2:$R/gate_recft/$run/selfcheck.json" "$d/gate_selfcheck.json" 2>/dev/null
  want=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['evidence']['ckpt_sha256'])" "$(cygpath -m "$DEST/recft_markers/RUN_$tag.done")")
  got=$(sha256sum "$d/ckpt_last.pt" | cut -c1-64)
  if [ "$want" = "$got" ]; then echo "OK $tag $run ckpt sha256 ${got:0:16}"; else echo "MISMATCH $tag: marker $want != pulled $got"; bad=1; fi
  ( cd "$d" && sha256sum ckpt_last.pt summary.json argv.json > SHA256SUMS.txt )
done
echo "pull done (bad=$bad) -> $DEST"; exit $bad
