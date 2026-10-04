#!/bin/bash
# retry wrapper: alternates the classic LFS path (xet disabled) and the xet path; terminal markers QQDONE / QQGAVEUP
cd /home/nvidia/refcv8_v9labels || exit 9
for a in 1 2 3 4 5 6; do
  if [ $((a % 2)) -eq 1 ]; then export HF_HUB_DISABLE_XET=1; else unset HF_HUB_DISABLE_XET; fi
  echo "attempt $a xet_disabled=${HF_HUB_DISABLE_XET:-0} start $(date -u +%FT%TZ)"
  timeout 3600 /home/nvidia/venvs/tanitad-train/bin/python hf_push_v9_thor2.py
  rc=$?
  echo "attempt $a rc=$rc end $(date -u +%FT%TZ)"
  if [ $rc -eq 0 ]; then echo "QQDONE attempt=$a"; exit 0; fi
  sleep $((90 * a))
done
echo "QQGAVEUP"
exit 2
