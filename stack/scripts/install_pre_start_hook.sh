#!/usr/bin/env bash
# =============================================================================
# Install the boot-persistent /pre_start.sh stub (idempotent, fail-safe)
# =============================================================================
# RunPod's image /start.sh runs `bash /pre_start.sh` on EVERY container start
# (verified in /start.sh -> execute_script "/pre_start.sh"). We use that as the
# reboot hook. The stub is intentionally TINY and ALWAYS exits 0: /start.sh runs
# under `set -e` and calls it BEFORE sshd starts, so a failing stub would brick
# the pod (no SSH). All real logic is in pod_boot_hook.sh on the persistent
# volume; the stub only backgrounds it.
#
# Note: /pre_start.sh lives on the container's ephemeral rootfs. It survives a
# plain "Restart Pod" (docker restart preserves the writable layer) but a full
# container RE-CREATE wipes it — re-run this script (or let the orchestrator do
# so) to re-establish the stub after a recreate. pod_boot_hook.sh + all state
# live on /workspace and persist regardless.
# -----------------------------------------------------------------------------
set -u
HOOK="${HOOK:-/workspace/TanitAD/stack/scripts/pod_boot_hook.sh}"
STUB="${STUB:-/pre_start.sh}"
BEGIN="# >>> tanitad-boot >>>"
END="# <<< tanitad-boot <<<"

tmp="$(mktemp)"
# Preserve any pre-existing non-tanitad content, dropping a prior tanitad block.
if [ -f "$STUB" ]; then
  awk -v b="$BEGIN" -v e="$END" 'BEGIN{skip=0} $0==b{skip=1;next} $0==e{skip=0;next} skip==0{print}' "$STUB" > "$tmp"
fi
{
  if [ ! -s "$tmp" ]; then echo '#!/usr/bin/env bash'; else cat "$tmp"; fi
  echo "$BEGIN"
  echo "# Launch the TanitAD supervisors in the background; MUST NOT fail pod boot."
  echo "( setsid nohup bash ${HOOK} </dev/null >>/workspace/ops/boot.log 2>&1 & ) >/dev/null 2>&1 || true"
  echo "$END"
  echo "exit 0"
} > "$STUB"
chmod +x "$STUB" 2>/dev/null || true
rm -f "$tmp" 2>/dev/null || true

echo "installed ${STUB} (backgrounds ${HOOK} on every container start):"
echo "-----"
cat "$STUB"
echo "-----"
exit 0
