#!/usr/bin/env bash
# LEVELHEAD arm CL: pull the two runs' (CL_0, CL_1) checkpoints, summaries and markers from box 2 to the dev box and verify each ckpt's sha256 against the box-2 marker. REFUSES without CL_PI_GO.
# Run AFTER CHAIN_CL.done exists (box 2's USB HDD: a ~0.5 GB read). Usage: bash pull_cl.sh   (idempotent: re-pulls what is missing or differs; a run without RUN_CL_<s>.done is skipped and reported)
# Destination: E:/Projects/TanitAD-artifacts/v8f-recover/ -> M2b_s{0,1}_recft_CL/ and cl_markers/ (the other runs in that folder are not touched).
# Releases OUR BOX2_BUSY lock (moved aside, never deleted) only when bad=0 and the lock text is the chain's. Test hooks (dev box): CL_DEST, CL_SCP, CL_PY, CL_LOCK. Production runs set none of them.
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to contact box 2 without the go of the PI}"
DEST=${CL_DEST:-/e/Projects/TanitAD-artifacts/v8f-recover}
R=D:/tanitad/runs/v8f_mini_full
SCP=${CL_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${CL_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
LOCK=${CL_LOCK:-/c/Users/Admin/box2_ship/BOX2_BUSY.txt}
mkdir -p "$DEST/cl_markers/logs" "$DEST/cl_markers/pin"
$SCP -r "tanitad-box2:$R/cl/*" "$DEST/cl_markers/" 2>&1 | grep -v '^\*\*'
bad=0
for t in CL_0:M2b_s0_recft_CL CL_1:M2b_s1_recft_CL; do
  tag=${t%%:*}; run=${t##*:}; d="$DEST/$run"; mkdir -p "$d"
  [ -f "$DEST/cl_markers/RUN_$tag.done" ] || { echo "SKIP $tag: no RUN_$tag.done (see RUN_$tag.FAILED)"; bad=1; continue; }
  for f in ckpt_last.pt summary.json argv.json selfcheck.json train_log.jsonl train.log; do $SCP "tanitad-box2:$R/$run/$f" "$d/$f" 2>/dev/null || echo "  (no $f for $run)"; done
  $SCP "tanitad-box2:$R/gate_cl/$run/selfcheck.json" "$d/gate_selfcheck.json" 2>/dev/null
  want=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['evidence']['ckpt_sha256'])" "$(cygpath -m "$DEST/cl_markers/RUN_$tag.done" 2>/dev/null || echo "$DEST/cl_markers/RUN_$tag.done")")
  got=$(sha256sum "$d/ckpt_last.pt" 2>/dev/null | cut -c1-64)
  if [ -n "$want" ] && [ "$want" = "$got" ]; then echo "OK $tag $run ckpt sha256 ${got:0:16}"; else echo "MISMATCH $tag: marker $want != pulled $got"; bad=1; fi
  ( cd "$d" && sha256sum ckpt_last.pt summary.json argv.json > SHA256SUMS.txt )
done
if [ "$bad" = 0 ] && [ -f "$LOCK" ] && head -c 36 "$LOCK" | grep -q '^LEVELHEAD CL two-run chain'; then
  mv "$LOCK" "$(dirname "$LOCK")/BOX2_BUSY.cl_chain_done_$(date +%Y%m%d_%H%M%S).txt"; echo "lock released (moved aside)"
elif [ "$bad" = 0 ]; then echo "lock not released: no BOX2_BUSY.txt of the chain at $LOCK"; fi
echo "pull done (bad=$bad) -> $DEST"; exit $bad
