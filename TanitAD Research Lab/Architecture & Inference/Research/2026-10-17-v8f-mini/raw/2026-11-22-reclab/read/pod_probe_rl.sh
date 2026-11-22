#!/bin/bash
# MM 2026-11-22, read only: the RL arms' label census (selfcheck) and the swap counters from the train logs.
R=/root/rollhist
for t in RL_s0 RL_s1; do
  D=$R/runs/$t
  echo "=== $t"
  ls $D $D/selfcheck 2>&1 | tr '\n' ' '; echo
  python3 - "$D/selfcheck/selfcheck.json" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("no selfcheck.json", e); sys.exit(0)
def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            q = p + "." + k if p else k
            if any(s in q.lower() for s in ("speed_label", "relabel", "label", "probe", "n_eligible", "refus", "identity", "emitted", "tick")):
                if not isinstance(v, (dict, list)):
                    print(q, "=", v)
            walk(v, q)
    elif isinstance(o, list) and len(o) < 50:
        for i, v in enumerate(o):
            walk(v, p + "[%d]" % i)
walk(d)
EOF
  echo "--- train log swap counters"
  L=$(ls $D/train/*.log $D/train/log*.txt $D/*.log 2>/dev/null | head -5); echo "logs: $L"
  for f in $L; do grep -iE "swap|label_diff|relabel|dagger" "$f" | head -8; grep -iE "swap|label_diff|relabel|dagger" "$f" | tail -3; done
done
