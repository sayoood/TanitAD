#!/bin/bash
# POD-SIDE, launched DETACHED by post_merge_to_p6gate.sh.   Step (d): hard-link (else atomic copy) the KEPT twins' parts from metric_cache_twins_3c into metric_cache,
# where P6's labeller reads them (`--metric-cache`) and where P6's lanes stage looks before it would build.   Idempotent and resumable.
# Env in: ENV3C OUT MC MC_TWIN P6PKG P6N HOW HELPERS.   WRITES under $MC (new names only; an existing different file is a refusal) and $P6N.
set -u
export PYTHONDONTWRITEBYTECODE=1                  # no __pycache__ written into the 3-C or P6 code trees
source "$ENV3C"                                   # PY
echo $$ > "$P6N/parts.PID"
rm -f "$P6N/PARTS_FINISHED" "$P6N/parts_apply.out"
nice -n 10 $PY - parts --bank "$OUT" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode apply --how "$HOW" --report "$P6N/parts_copy_report.json" \
    < "$HELPERS" > "$P6N/parts_apply.out" 2>&1
echo $? > "$P6N/parts_apply.rc"
touch "$P6N/PARTS_FINISHED"
