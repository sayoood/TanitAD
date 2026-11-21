#!/usr/bin/env bash
# either_marker.sh <go marker> <marker 1> <marker 2>   writes <go marker> as soon as marker 1 or marker 2 exists (bounded: 120 h). No GPU, no load.
# Purpose: the next chain starts when its predecessor is DONE or has STOPPED, so one stopped chain never idles the pod.
GO=$1; A=$2; B=$3; n=0
while [ ! -f "$A" ] && [ ! -f "$B" ]; do n=$((n+1)); [ $n -gt 7200 ] && { echo "neither marker after 120 h" > "$GO.TIMEOUT"; exit 1; }; sleep 60; done
if [ -f "$A" ]; then w=$A; else w=$B; fi
echo "predecessor marker seen: $w ($(date -u +%Y-%m-%dT%H:%M:%SZ))" > "$GO"
