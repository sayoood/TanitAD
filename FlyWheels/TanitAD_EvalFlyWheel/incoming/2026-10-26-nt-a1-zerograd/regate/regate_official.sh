#!/bin/bash
# OFFICIAL re-gate of nt-a1-s0 (PI go 2026-10-26, D-NT-A1-ZGFAIL; r8_rc_gate included as a DISCLOSED exemption) -- MM
set -u
NT=/home/nvidia/r8ns/nt; RUN=$NT/runs/nt-a1-s0; ENV=$NT/runs.d/nt-a1-s0.env; RG=$NT/regate_20261026
[ -f "$RUN/ZEROGRAD_FAIL.json" ] || { echo "ZZREGATE-NOFAILMARKERZZ"; exit 2; }
cp -p "$ENV" "$RG/nt-a1-s0.env.pre_regate"
{ echo ""
  echo "# MM 2026-10-26 Berlin, PI go (D-NT-A1-ZGFAIL): re-gate with the launch gate's PRE-DECLARED dead groups (launch_gate.py profile"
  echo "# refcv8-nt live_dead_admitted whose flags are in this argv; derived by stack/scripts/zerograd_allow_from_gate.py, 10e9dfd)"
  echo "# + core.decoder.r8_rc_gate as a DISCLOSED exemption, NOT pre-registered (same rc-is-None guard, refc.py:3887-3891)."
  cat "$RG/allow_final.sh"; } >> "$ENV"
bash -c "source '$ENV'; echo n_allow=\${#ZEROGRAD_ALLOW[@]}"
mv "$RUN/ZEROGRAD_FAIL.json" "$RG/ZEROGRAD_FAIL.20261026T1753Z.json"
cd $NT && setsid nohup bash $NT/launch_tree/stack/ops/sup_refcv7.sh nt-a1-s0 $NT/runs.d > $NT/sup_nt-a1-s0.regate.out 2>&1 < /dev/null &
echo "ZZREGATE-STARTEDZZ"
