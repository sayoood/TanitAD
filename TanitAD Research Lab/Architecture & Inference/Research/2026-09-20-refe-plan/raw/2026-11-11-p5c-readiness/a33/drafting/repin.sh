#!/usr/bin/env bash
# recompute the two pins the A3.3 edits move (PIN_GATES, PIN_RJ5PATH) in a1/p5c_pod_label.sh
cd "$(dirname "$0")"
G=$(sha256sum a1/p5c_table_gates.py | cut -d' ' -f1); R=$(sha256sum a3/p5c_rj5_path.py | cut -d' ' -f1)
sed -i -E "s/^PIN_GATES=[0-9a-f]{64}/PIN_GATES=$G/; s/^PIN_RJ5PATH=[0-9a-f]{64}/PIN_RJ5PATH=$R/" a1/p5c_pod_label.sh
grep -c $'\r' a1/p5c_pod_label.sh
grep -E "^PIN_(GATES|RJ5PATH)=" a1/p5c_pod_label.sh | cut -c1-90
