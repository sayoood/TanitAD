#!/bin/bash
# MM 2026-11-23: card ADD125 (DSCALE-LEAD-EQP, 9dfbc7f3) on the LoRA pod (PI in the MM chat, 2026-11-23 ~06:40Z: "Yes, run both" = RK, then ADD125).
# The installed ADD123 kit, unchanged (pod_dscale_lead_2x.sh a01c5ca8, tok_probe_u_ck.py 5888527b, TREE_PINS_2X.txt ad681c67), with the four
# settings of HANDOVER_ADD125.md (87a96684) section 2: OUT=out_eqp, STEPS_U2=61568, TRAIN_LIMIT_S=6600, MAX_H=1.5.
# Waits (cap 8 h) for the ROLLHIST-RK chain's last line ("ZZ ROLLHIST-RK DONE|FAIL ZZ" in /root/rollhist/rh_rk_chain.log), then
# precheck -> cpucheck -> smoke -> run -> extract -> checklog, then the no-.pt tar for the coordinator fork.
# The kit takes flock /root/gpu_queue.lock for each GPU process itself, so this script holds NO lock.
# Marker: the pod owner's priority marker /root/N1DW_READY, so TanitADRuled's next slice skips. While it waits, every 20 s it creates the marker
# ONLY IF ABSENT (noclobber, atomic) with its own token as content (the RK watcher clears its own token once the RK chain holds the lock; this
# re-sets it then, before the RK chain ends). It removes the marker at the end ONLY if the content equals its own token. Never overwrites a marker.
# smoke exit 4 = projection above MAX_H (1.5 h): stop, back to the PI (no retry). A training stop also goes back to the PI.
# v2 (06:25Z): adds KIT_SHA.txt to $OUT before the tar (the fork's request); v1 stopped while waiting, before any lock or marker.
# Log /root/dscale_lead/mm_add125_chain.log; last line ZZ ADD125 DONE ZZ | ZZ ADD125 STOP ZZ <why> | ZZ ADD125 FAIL ZZ <stage> rc
export KIT=/root/dscale_lead/kit TREE=/root/dscale_lead/kit/tree OUT=/root/dscale_lead/out_eqp
export P10=/root/dscale_lead/kit/p10_prior.pt TRACKS=/root/dscale10k/merged/native_tracks_union.pt
export STEPS_U2=61568 TRAIN_LIMIT_S=6600 MAX_H=1.5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=/root/dscale_lead/mm_add125_chain.log
R=/root/rollhist/rh_rk_chain.log
MK=/root/N1DW_READY
MINE=/root/dscale_lead/add125_marker_mine
TOKEN="ADD125-$$-$(date +%s)"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
mark() { if ( set -C; printf '%s\n' "$TOKEN" > "$MK" ) 2>/dev/null; then printf '%s\n' "$TOKEN" > "$MINE"; say "marker set ($MK) with own token"; fi; }
unmark() { if [ -e "$MINE" ] && [ -e "$MK" ] && [ "$(cat "$MK" 2>/dev/null)" = "$(cat "$MINE")" ]; then rm -f "$MK"; say "marker removed (own token)"; else say "marker not removed: absent or not ours"; fi; rm -f "$MINE"; }
finish() { unmark; echo "$1" >> "$L"; exit "$2"; }
[ -e "$OUT" ] && { echo "ZZ ADD125 STOP ZZ $OUT exists (never overwritten)" >> "$L"; exit 1; }
say "start (token $TOKEN); waiting for the ROLLHIST-RK chain's last line in $R"
mark
for i in $(seq 1 1440); do
  grep -q "ZZ ROLLHIST-RK " "$R" 2>/dev/null && break
  [ -e "$MK" ] || mark
  sleep 20
done
grep -q "ZZ ROLLHIST-RK " "$R" 2>/dev/null || finish "ZZ ADD125 STOP ZZ ROLLHIST-RK did not end within 8 h" 1
say "ROLLHIST-RK ended: $(grep "ZZ ROLLHIST-RK " "$R" | tail -n 1)"
[ -e "$MK" ] || mark
cd "$KIT" || finish "ZZ ADD125 FAIL ZZ cd rc=1" 1
for s in precheck cpucheck smoke run extract checklog; do
  say "stage $s start"
  bash pod_dscale_lead_2x.sh "$s" >> "$L" 2>&1
  rc=$?
  say "stage $s rc=$rc"
  if [ "$s" = smoke ] && [ "$rc" -eq 4 ]; then finish "ZZ ADD125 STOP ZZ smoke projection above $MAX_H h: read $OUT/SMOKE_ESTIMATE.json, back to the PI" 4; fi
  if [ "$rc" -ne 0 ]; then
    if [ -s "$OUT/DSCALE_2X_STOP.txt" ]; then finish "ZZ ADD125 STOP ZZ $s: $(head -c 300 "$OUT/DSCALE_2X_STOP.txt" | tr '\n' ' ')" 1; fi
    finish "ZZ ADD125 FAIL ZZ $s rc=$rc" 1
  fi
done
# the fork's request (2026-11-23, control e0 of card ADD125): the three kit files' sha256 as they stand now, inside the tar
(cd "$KIT" && sha256sum pod_dscale_lead_2x.sh tok_probe_u_ck.py TREE_PINS_2X.txt) > "$OUT/KIT_SHA.txt" || finish "ZZ ADD125 FAIL ZZ kit sha" 1
say "KIT_SHA.txt: $(cut -c1-16 "$OUT/KIT_SHA.txt" | tr '\n' ' ')"
unmark
cd /root/dscale_lead && tar --exclude='*.pt' -czf out_eqp_nockpt.tar.gz out_eqp && sha256sum out_eqp_nockpt.tar.gz > out_eqp_nockpt.tar.gz.sha256 || finish "ZZ ADD125 FAIL ZZ tar" 1
say "tar $(cat out_eqp_nockpt.tar.gz.sha256)"
echo "ZZ ADD125 DONE ZZ" >> "$L"
