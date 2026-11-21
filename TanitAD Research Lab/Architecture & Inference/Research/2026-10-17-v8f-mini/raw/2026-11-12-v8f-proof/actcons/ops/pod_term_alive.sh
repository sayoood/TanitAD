#!/bin/bash
# pod_term_alive.sh <chain>:<arm> ... -- runs ON THE POD via ssh stdin, CPU, reads only. Control of note ADD37_N4 section 4 (the term is alive): for each arm the
# train log must show the consistency term with a finite value on every logged step. Prints one line per arm. Starts nothing, deletes nothing.
W=/workspace/v8proof
for ca in "$@"; do
  c=${ca%%:*}; a=${ca##*:}
  f=$(ls $W/runs/$c/$a/train_log.jsonl $W/runs/$c/$a/out/train_log.jsonl 2>/dev/null | head -n 1)
  [ -n "$f" ] || { echo "TERM $a NO-LOG"; continue; }
  ARM_=$a /workspace/venv/bin/python - "$f" <<'PYEOF'
import json, math, os, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
body = [r for r in rows if isinstance(r, dict) and "step" in r]
k = [r for r in body if "act_cons" in r]
fin = [r for r in k if isinstance(r["act_cons"], (int, float)) and math.isfinite(r["act_cons"])]
lv = [r for r in body if "act_level" in r]
nv = [r.get("act_cons_n", 0) for r in k]
print("TERM", os.environ["ARM_"], "logged", len(body), "with_term", len(k), "finite", len(fin), "all_finite", bool(k) and len(k) == len(fin) == len(body),
      "term_first_last", (round(k[0]["act_cons"], 4), round(k[-1]["act_cons"], 4)) if k else None, "min_n", min(nv) if nv else None, "last_step", body[-1]["step"] if body else None, "level_rows", len(lv))
PYEOF
done
