#!/bin/bash
# push_steer_enc.sh -- push the ADD100 arm S data-side kit to the pod and compare sha256 per file.
# Usage: bash /f/TanitAD-scratch/v8proof/cropread/steer_enc/push_steer_enc.sh
# Pod target: /workspace/v8proof/cropread/kit/steer_enc/   (code files are overwritten with `cat >`; nothing is deleted or moved)
# Never prints the pod address, the token or the ssh command line.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSSH=/c/Users/Admin/v8pod/pssh.sh
DEST=/workspace/v8proof/cropread/kit/steer_enc
FILES="steer_encode.py ctl_s3.py ctl_s8.py fake_steer.py pod_steer_enc_queue.sh"
if [ -n "${EXTRA_FILES:-}" ]; then FILES="$FILES $EXTRA_FILES"; fi
timeout 60 bash "$PSSH" "mkdir -p $DEST" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
bad=0
for f in $FILES; do
  if [ ! -f "$HERE/$f" ]; then echo "MISSING local $f"; bad=1; continue; fi
  loc=$(tr -d '\r' < "$HERE/$f" | sha256sum | cut -d' ' -f1)
  tr -d '\r' < "$HERE/$f" | timeout 120 bash "$PSSH" "cat > $DEST/$f" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
  rem=$(timeout 60 bash "$PSSH" "sha256sum $DEST/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$loc" = "$rem" ]; then echo "OK   ${loc:0:16}  $f"; else echo "DIFF local=${loc:0:16} pod=${rem:0:16}  $f"; bad=1; fi
done
if [ $bad -eq 0 ]; then echo "PUSH_STEER_ENC ALL OK"; else echo "PUSH_STEER_ENC FAILED"; fi
exit $bad
