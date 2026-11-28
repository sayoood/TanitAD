#!/usr/bin/env bash
# ACTRSP box-2 chain launcher (READY, NOT RUN). Detached through WMI (CreateFlags 0x08000200) -> cmd.exe /c launch_actrsp.cmd < NUL -> Git usr/bin bash (pattern of C:/Users/Admin/actsens_launch/launch_actsens.sh).
# Preconditions: B-CAL amendment frozen by the MM, the amended trainer landed on ONE commit, ARGVS_SHA256 re-pinned if the argvs moved, G-SUITE green, MM read of the pilot plan.
export PATH="/usr/bin:/mingw64/bin:$PATH"
export ACTRSP_COMMIT="${ACTRSP_COMMIT:?set ACTRSP_COMMIT to the full 40-hex sha of the landed ACTRSP code commit}"   # a13f6f16983010d2a96e14c16aef8273626fcd73 unless an amendment moves it
export ACTRSP_SSH=/usr/bin/ssh
export ACTRSP_SFTP=/usr/bin/sftp
unset ACTRSP_DRY ACTRSP_PRED_NAME ACTRSP_PRED_STATE ACTRSP_AMENDMENT
LOGD=/c/Users/Admin/actrsp_launch; mkdir -p "$LOGD"
echo "LAUNCH $(date '+%F %T') bash pid $$" >> "$LOGD/launcher.log"
cd /c/Users/Admin/actrsp_pkg || { echo "EXIT no-package $(date '+%F %T')" >> "$LOGD/launcher.log"; exit 3; }
bash /c/Users/Admin/actrsp_pkg/actrsp_chain.sh >> "$LOGD/chain_console.log" 2>&1
rc=$?
echo "EXIT rc=$rc $(date '+%F %T')" >> "$LOGD/launcher.log"
exit $rc
