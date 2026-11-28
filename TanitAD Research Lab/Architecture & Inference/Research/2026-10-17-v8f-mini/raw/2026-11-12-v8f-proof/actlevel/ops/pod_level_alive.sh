#!/bin/bash
# pod_level_alive.sh <chain>:<arm> ... -- runs ON THE POD via ssh stdin, CPU, reads only. Control v5 of note ADD37_N6 (the level term is alive): for each arm
# the train log must show the consistency term and, for an arm with a level weight, the level term with a finite value and a positive row count on every
# logged step. Prints one line per arm. Starts nothing, deletes nothing.
W=/workspace/v8proof
for ca in "$@"; do
  c=${ca%%:*}; a=${ca##*:}
  f=$(ls $W/runs/$c/$a/train_log.jsonl $W/runs/$c/$a/out/train_log.jsonl 2>/dev/null | head -n 1)
  [ -n "$f" ] || { echo "LEVEL $a NO-LOG"; continue; }
  ARM_=$a /workspace/venv/bin/python - "$f" <<'PYEOF'
import json, math, os, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
body = [r for r in rows if isinstance(r, dict) and "step" in r]
fin = lambda r, k: isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
kc = [r for r in body if "act_cons" in r]
kl = [r for r in body if "act_level" in r]
nl = [r.get("act_level_n", 0) for r in kl]
print("LEVEL", os.environ["ARM_"], "logged", len(body), "last_step", body[-1]["step"] if body else None,
      "| cons rows", len(kc), "finite", sum(fin(r, "act_cons") for r in kc), "first_last", (round(kc[0]["act_cons"], 4), round(kc[-1]["act_cons"], 4)) if kc else None,
      "| level rows", len(kl), "finite", sum(fin(r, "act_level") for r in kl), "first_last", (round(kl[0]["act_level"], 4), round(kl[-1]["act_level"], 4)) if kl else None,
      "min_n", min(nl) if nl else None, "| keys", sorted(k for k in (body[-1] if body else {}) if k.startswith("act_")))
PYEOF
done
