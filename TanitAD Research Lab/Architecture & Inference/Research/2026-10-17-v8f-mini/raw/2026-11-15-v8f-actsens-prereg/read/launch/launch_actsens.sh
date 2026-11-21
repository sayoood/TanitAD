#!/usr/bin/env bash
# ACTSENS box-2 chain launcher (MM, 2026-11-15). Detached through WMI (CreateFlags 0x08000200) -> cmd.exe /c launch_actsens.cmd < NUL -> Git usr/bin bash.
# PI go: "Yes, launch it". Code of record 6b2bca1f on agent/v8f-actsens-20261115. Package: C:/Users/Admin/actsens_pkg (copy of the landed raw/.../build/box2/, blob-checked).
export PATH="/usr/bin:/mingw64/bin:$PATH"
export ACTSENS_COMMIT=81cee6c1760068cf76c6f0465d1fffd9bc1d7215   # AMEND_ACTSENS_1 code of record (run 2); run 1 used 6b2bca1f and stopped at the C3 gate
export ACTSENS_SSH=/usr/bin/ssh
export ACTSENS_SFTP=/usr/bin/sftp
unset ACTSENS_DRY ACTSENS_PRED_NAME ACTSENS_PRED_STATE ACTSENS_AMENDMENT
LOGD=/c/Users/Admin/actsens_launch
echo "LAUNCH $(date '+%F %T') bash pid $$" >> "$LOGD/launcher.log"
cd /c/Users/Admin/actsens_pkg || { echo "EXIT no-package $(date '+%F %T')" >> "$LOGD/launcher.log"; exit 3; }
bash /c/Users/Admin/actsens_pkg/actsens_chain.sh >> "$LOGD/chain_console.log" 2>&1
rc=$?
echo "EXIT rc=$rc $(date '+%F %T')" >> "$LOGD/launcher.log"
exit $rc
