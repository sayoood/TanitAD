#!/bin/bash
# MM 2026-11-20: start the ROLLHIST chain only after TanitADRuled's hpf1 decode (agreed with that session: about 8 min of GPU).
# Their decode starts only while /root/N1DW_READY is absent, so this gate holds the marker off until the decode is in.
# Done signal (theirs): /root/ruled/runs/hpf1_share0.03/pred/_done exists AND /root/ruled/logs/gpu_slices.log has "end hpf1_share0.03_infer rc=".
# Any rc ends the wait (a failed decode is theirs to rerun; the pod must not idle). Cap: 3 h from start, then the chain starts anyway.
# Then it starts the marker watcher (which sets the marker) and the chain, exactly as HANDOFF_B2 LAUNCH steps 1-2. Read-only on /root/ruled.
L=/root/rollhist/rh_launch_after_hpf1.log
D=/root/ruled/runs/hpf1_share0.03/pred/_done
G=/root/ruled/logs/gpu_slices.log
END=$(( $(date +%s) + 10800 ))
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
say "gate start: waiting for $D and 'end hpf1_share0.03_infer rc=' in $G"
while :; do
  if [ -f "$D" ] && grep -q "end hpf1_share0.03_infer rc=" "$G" 2>/dev/null; then
    say "hpf1 decode done: $(grep "end hpf1_share0.03_infer rc=" "$G" | tail -n 1)"
    break
  fi
  if [ "$(date +%s)" -gt "$END" ]; then say "cap 3 h reached without the done signal; starting the chain anyway"; break; fi
  sleep 30
done
nohup setsid bash /root/rollhist/rh_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_chain.sh < /dev/null > /dev/null 2>&1 &
say "marker watcher and chain started (RH_READOUT=causal_g1x)"
