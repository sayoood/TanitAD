#!/bin/bash
# Dev box: emit ship_restage_v2.txt = a script to paste/pipe on the pod (podx.sh style) that recreates restage_v2.sh and restage_helpers.py
# under /workspace/ops/restage_v2 and verifies their sha256 against the local copies.  No token, pod address or secret is involved.
set -eu
cd "$(dirname "$0")"
out=ship_restage_v2.txt
{
  echo 'set +H; mkdir -p /workspace/ops/restage_v2'
  for f in restage_v2.sh restage_helpers.py; do
    echo "base64 -d > /workspace/ops/restage_v2/$f <<'ZZB64'"; base64 -w 76 $f; echo 'ZZB64'
  done
  echo 'chmod +x /workspace/ops/restage_v2/restage_v2.sh'
  for f in restage_v2.sh restage_helpers.py; do
    s=$(sha256sum $f | cut -c1-64)
    echo "[ \"\$(sha256sum /workspace/ops/restage_v2/$f | cut -c1-64)\" = $s ] && echo \"ZZSHIP_OK $f ${s:0:16} ZZ\" || echo \"ZZSHIP_BAD $f ZZ\""
  done
  echo 'bash -n /workspace/ops/restage_v2/restage_v2.sh && echo "ZZSYNTAX_OK ZZ"'
  echo '# START (only after <TOKEN_FILE> is placed; prefix P6_RESUME_REV=... for the rollback):  setsid -f nohup bash /workspace/ops/restage_v2/restage_v2.sh > /workspace/ops/restage_v2/driver.out 2>&1 < /dev/null'
} > $out
wc -c $out
