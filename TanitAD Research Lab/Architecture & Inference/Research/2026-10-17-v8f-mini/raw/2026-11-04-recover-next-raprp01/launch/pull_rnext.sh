#!/usr/bin/env bash
# RECOVER-NEXT: pull the runs' (six registered + the two RAPR-P01 runs) checkpoints, summaries and markers from box 2 to the dev box, and verify each ckpt's sha256 against the box-2 marker. Run AFTER CHAIN_RNEXT.done exists (box 2's USB HDD: a ~0.5 GB read, no telemetry needed).
# Usage: bash pull_rnext.sh            (idempotent: re-pulls what is missing or differs; a run without RUN_<tag>.done is skipped and reported)
# Destination: E:/Projects/TanitAD-artifacts/v8f-recover/ -> M2b_s{0,1}_recft_{RAPR,RTL,RP01,RAPRP01}/ and rnext_markers/ (the RECOVER-FT runs in the same folder are not touched).
# Test hooks (dev box): RNEXT_DEST, RNEXT_SCP (the whole scp command), RNEXT_PY. Production runs set none of them.
set -u
DEST=${RNEXT_DEST:-/e/Projects/TanitAD-artifacts/v8f-recover}
R=D:/tanitad/runs/v8f_mini_full
SCP=${RNEXT_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${RNEXT_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
mkdir -p "$DEST/rnext_markers/logs" "$DEST/rnext_markers/pin"
$SCP -r "tanitad-box2:$R/rnext/*" "$DEST/rnext_markers/" 2>&1 | grep -v '^\*\*'
bad=0
for t in RAPR_0:M2b_s0_recft_RAPR RAPR_1:M2b_s1_recft_RAPR RTL_0:M2b_s0_recft_RTL RTL_1:M2b_s1_recft_RTL RP01_0:M2b_s0_recft_RP01 RP01_1:M2b_s1_recft_RP01 RAPRP01_0:M2b_s0_recft_RAPRP01 RAPRP01_1:M2b_s1_recft_RAPRP01; do
  tag=${t%%:*}; run=${t##*:}; d="$DEST/$run"; mkdir -p "$d"
  [ -f "$DEST/rnext_markers/RUN_$tag.done" ] || { echo "SKIP $tag: no RUN_$tag.done (see RUN_$tag.FAILED)"; bad=1; continue; }
  for f in ckpt_last.pt summary.json argv.json selfcheck.json train_log.jsonl train.log; do $SCP "tanitad-box2:$R/$run/$f" "$d/$f" 2>/dev/null || echo "  (no $f for $run)"; done
  $SCP "tanitad-box2:$R/gate_rnext/$run/selfcheck.json" "$d/gate_selfcheck.json" 2>/dev/null
  want=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['evidence']['ckpt_sha256'])" "$(cygpath -m "$DEST/rnext_markers/RUN_$tag.done" 2>/dev/null || echo "$DEST/rnext_markers/RUN_$tag.done")")
  got=$(sha256sum "$d/ckpt_last.pt" 2>/dev/null | cut -c1-64)
  if [ -n "$want" ] && [ "$want" = "$got" ]; then echo "OK $tag $run ckpt sha256 ${got:0:16}"; else echo "MISMATCH $tag: marker $want != pulled $got"; bad=1; fi
  ( cd "$d" && sha256sum ckpt_last.pt summary.json argv.json > SHA256SUMS.txt )
done
echo "pull done (bad=$bad) -> $DEST"; exit $bad
