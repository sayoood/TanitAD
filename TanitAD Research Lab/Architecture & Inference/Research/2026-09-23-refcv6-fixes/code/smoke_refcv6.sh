#!/usr/bin/env bash
# refcv6 post-reboot smoke on Thor: THE launch line, short, with every per-step reading on.
#
#   CODE=<shipped tree> TAG=<name> BATCH=<B> CONFLICT_EVERY=<N> [WORKERS=4] ./smoke_refcv6.sh
#
# Reads back ONLY from the artifact (metrics.jsonl), and emits opaque ZZ tokens so no monitor
# greps its own command line: ZZSMOKE-<tag>-<rows>-<maxmemGB>-<s_per_step>-<evalrows>-<evalerrs>ZZ
set -u
: "${CODE:?}" "${TAG:?}" "${BATCH:?}" "${CONFLICT_EVERY:?}"
OUT=/home/nvidia/refcv6_run/smoke/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"
D=/home/nvidia/data
CODE="$CODE" OUT="$OUT" BATCH="$BATCH" STEPS="${STEPS:-40}" CONFLICT_EVERY="$CONFLICT_EVERY" \
WORKERS="${WORKERS:-4}" EVAL=1 EVAL_EVERY=20 EVAL_BATCHES=2 SAVE_EVERY=1000000 LOG_EVERY=1 \
AGENT_JOIN="$D/joins/b1_train_plus_eval_agents.jsonl.xz" \
JOIN3D="$D/join3d/b1_train_plus_eval_agents_3d.jsonl.xz" \
  bash "$CODE/ops/run_refcv6.sh" > "$OUT/smoke.log" 2> "$OUT/smoke.stderr.log"
echo "exit=$? (NOT the verdict)" >> "$OUT/smoke.log"
/home/nvidia/venvs/tanitad-train/bin/python - "$OUT" "$TAG" <<'PYEOF'
import json, sys
out, tag = sys.argv[1], sys.argv[2]
rows, ev, everr = [], 0, 0
try:
    for l in open(out + "/metrics.jsonl", encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if "eval_error" in r:
            everr += 1
        elif any(k.startswith("eval_") for k in r):
            ev += 1
        elif isinstance(r.get("step"), int) and "elapsed_s" in r:
            rows.append(r)
except FileNotFoundError:
    pass
mem = max((r.get("cuda_max_mem_gb", 0.0) for r in rows), default=0.0)
sps = 0.0
if len(rows) >= 12:
    a, b = rows[9], rows[-1]            # skip the first steps (warm-up, cudnn autotune)
    sps = (b["elapsed_s"] - a["elapsed_s"]) / max(1, b["step"] - a["step"])
print("ZZSMOKE-%s-%d-%.2f-%.3f-%d-%dZZ" % (tag, len(rows), mem, sps, ev, everr))
PYEOF
