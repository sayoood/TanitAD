#!/usr/bin/env bash
# wait_pull_arm.sh <queue> <ARM> [max_minutes]  -- waits (script polling, no model turns) for "ARM DONE <ARM>" in the pod chain log of <queue>,
# then pulls the checkpoint (binary pull, sha256 compared, sha256 file beside it) to spatial/ckpt/<ARM>/ and the T1-kin dump through pull_spatial_t1.sh.
# Deletes nothing. Last line: ZZ WAITPULL DONE <ARM> or ZZ WAITPULL FAIL <ARM> <reason>.
Q=${1:?queue}; A=${2:?arm}; MAXM=${3:-180}
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
W=/f/TanitAD-scratch/v8proof; D=${CK_ROOT:-$W/spatial/ckpt}/$A; PULL=${PULL_ROOT:-$W/hiergap/pull}; RR=/workspace/v8proof/runs/$Q
n=0
while true; do
  k=$(timeout 60 bash "$P" "grep -c -- \"ARM DONE $A\" /workspace/v8proof/chains/$Q/chain.log" 2>/dev/null | tr -cd "0-9")
  [ -n "$k" ] && [ "$k" -ge 1 ] && break
  n=$((n+1)); [ $n -gt $MAXM ] && { echo "ZZ WAITPULL FAIL $A timeout"; exit 1; }
  ping -n 61 127.0.0.1 > /dev/null
done
mkdir -p "$D"
want=$(timeout 120 bash "$P" "sha256sum $RR/$A/ckpt_last.pt | cut -c1-64" 2>/dev/null | tr -cd "0-9a-f\n" | tail -n 1)
ok=0
for att in 1 2 3; do
  timeout 1200 bash "$C" "$RR/$A/ckpt_last.pt" "$D/ckpt_last.pt" > /dev/null 2>&1
  got=$(sha256sum "$D/ckpt_last.pt" | cut -c1-64)
  [ ${#want} = 64 ] && [ "$got" = "$want" ] && { ok=1; break; }
done
[ $ok = 1 ] || { echo "ZZ WAITPULL FAIL $A ckpt-sha256"; exit 1; }
echo "$want  ckpt_last.pt" > "$D/ckpt_last.sha256"
for f in argv.json selfcheck.json; do timeout 60 bash "$C" "$RR/$A/$f" "$D/$f" > /dev/null 2>&1; done
echo "$A ckpt ok sha256=$want"
Q=$Q bash "$W/spatial/q3/pull_spatial_t1.sh" "$A" | tail -n 2
[ -f "$PULL/$A/PULLED.txt" ] || { echo "ZZ WAITPULL FAIL $A dump"; exit 1; }
echo "ZZ WAITPULL DONE $A"
