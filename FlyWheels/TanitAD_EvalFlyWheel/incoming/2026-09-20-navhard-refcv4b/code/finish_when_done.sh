#!/usr/bin/env bash
# W7 2026-09-20 — the DETACHED FINISHER: waits for the live navhard run to write summary.json, then
# produces every remaining deliverable and stages them by EXPLICIT PATH.
#
# ⛔ WHY IT EXISTS. The run finishes ~8 h after it starts. "I'll collect the numbers next time" is
# how an artifact ends up on one disk and nowhere else — the failure this programme's operating
# standard was written to make structurally impossible. This closes the loop without a human.
#
# ⛔ IT ASSERTS ON THE ARTIFACT, NEVER ON AN EXIT CODE. It waits for summary.json to EXIST, and each
# step's status is read back from the file it was supposed to write. No step is chained behind a
# pipe (`$?` after a pipeline is the LAST element's status — measured twice on 2026-09-07).
# ⛔ It NEVER commits and NEVER pushes, and it stages only its own explicit paths — never a
# directory — because this is a shared branch.
set -u
REPO="D:/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b"
RUN="${1:-$REPO/taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T144140Z-navsim_v2-refcv4b_b1_v72_40k-b11027}"
BANK="C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920"
EXPORT="C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"
LOG="$PKG/raw/finisher.log"
MAXWAIT=$((14 * 3600))

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

say "finisher armed; waiting for $RUN/summary.json (max ${MAXWAIT}s)"
T0=$(date +%s)
while [ ! -f "$RUN/summary.json" ]; do
  if [ ! -f "$RUN/bench_run.json" ] && [ $(( $(date +%s) - T0 )) -gt "$MAXWAIT" ]; then
    say "GIVING UP: no summary.json after ${MAXWAIT}s and no bench_run.json either"
    exit 3
  fi
  # bench_run.json without summary.json means the run ENDED without producing one (FAILED/REFUSED):
  # that is a real outcome and must be reported, not waited on forever.
  if [ -f "$RUN/bench_run.json" ]; then
    sleep 20
    [ -f "$RUN/summary.json" ] || { say "RUN ENDED WITHOUT summary.json — see $RUN/bench_run.json"; exit 4; }
  fi
  sleep 60
done
say "summary.json present after $(( $(date +%s) - T0 ))s"

# ⛔⛔ "summary.json EXISTS" IS NOT "THE RUN SUCCEEDED" — MEASURED 2026-09-20, ON THIS SCRIPT.
# The suite writes a summary for a FAILED run too, with every arm `status: FAILED` and every
# headline `UNAVAILABLE`. The first version of this finisher waited on the FILE, found it, and
# produced a FINISH_REPORT.md that LOOKED like a result: an arm table, correct headers, and
# `UNAVAILABLE` in every cell. That is the exact class this package spent the day auditing for —
# a success criterion disconnected from the thing being checked — committed by my own guard.
# ⇒ gate on the CONTENT: at least one arm must have status OK, or this writes a FAILURE report and
# stops. The count is computed here and printed, so the gate can be audited from the log.
# ⛔ cd FIRST. v1 of this branch wrote `> FAILURE_REPORT.md` with a RELATIVE path before the
# `cd "$PKG"` below, so the failure report landed in the REPO ROOT (found there 2026-09-21) and the
# `git add $PKG/FAILURE_REPORT.md` that followed failed on a path that did not exist.
cd "$PKG" || exit 2
# ⛔ ONE shared judge (code/judge_run.py), never a copy of its logic — the existence test survived in
# retry_scoring.sh precisely because the fix had been made in THIS file and not propagated.
"$PY" "$PKG/code/judge_run.py" "$RUN" --require CV,STOP,ECHO,A1 > raw/finisher.judge.txt 2>&1
J=$(grep '^JUDGE=' raw/finisher.judge.txt | tail -1)
NOK=$("$PY" -c "import json,sys;print(json.loads(open(sys.argv[1],encoding='utf-8').readline()).get('n_ok',0))" \
      raw/finisher.judge.txt 2>/dev/null || echo 0)
say "content judge: ${J:-<none>} (arms OK: ${NOK:-0}/4)"
if [ "$J" != "JUDGE=SUCCESS" ]; then
  {
    echo "# ⛔ RUN FAILED — no arm was scored"
    echo
    echo "Run: \`$RUN\`"
    echo
    echo 'A `summary.json` EXISTS, and it is not a result: the suite writes one for a failed run too,'
    echo 'with every arm `status: FAILED` and every headline `UNAVAILABLE`. ⛔ Nothing in this run is'
    echo 'quotable. The per-arm failure records are the evidence:'
    echo
    echo '```'
    for a in CV STOP ECHO A1; do
      "$PY" -c "
import json,sys
try:
    d=json.load(open(sys.argv[1],encoding='utf-8'))
    print('%-5s %-18s rc=%s wall=%ss retryable=%s' % (d['arm'], d['status'], d['rc'], d['wall_s'], d['retryable']))
    for f in d.get('failures',[])[:3]: print('      ', str(f)[:150])
except Exception as e: print('%-5s (no counts.json)' % sys.argv[2])
" "$RUN/raw/$a/$a.counts.json" "$a" 2>/dev/null
    done
    echo '```'
    echo
    echo '## What survived, and what must NOT be redone'
    echo
    "$PY" -c "
import json
try:
    m=json.load(open(r'$RUN/raw/model/model_run.json',encoding='utf-8'))
    for k,v in m.get('arms',{}).items():
        print('* model seam **%s**: %s model rows, %s CV stand-ins, %.1f s — INTACT, adopt it with \`--reuse-seams\`'
              % (k, v.get('n_model_rows'), v.get('n_cv_standin_rows'), float(v.get('seconds') or 0)))
except Exception as e: print('* (no model_run.json: inference did not complete either)')
"
  } > FAILURE_REPORT.md
  say "wrote FAILURE_REPORT.md; ⛔ no FINISH_REPORT.md, because there is nothing to report"
  cd "$REPO" || exit 2
  git add -- "$PKG/FAILURE_REPORT.md" >> "$LOG" 2>&1
  git add -- "$PKG/raw/finisher.log" >> "$LOG" 2>&1
  exit 5
fi

cd "$PKG" || exit 2

"$PY" code/read_summary.py --run "$RUN" --arm A1 --floors CV,STOP,ECHO \
      --out raw/bar_verdict.json > raw/read_summary.stdout.txt 2> raw/read_summary.stderr.txt
say "read_summary artifact: $([ -s raw/bar_verdict.json ] && echo WRITTEN || echo MISSING)"

"$PY" code/decompose.py --run "$RUN" --arm A1 --vs STOP,CV,ECHO --inputs "$EXPORT" \
      --out raw/decomposition_A1.json > raw/decompose.stdout.txt 2> raw/decompose.stderr.txt
say "decomposition artifact: $([ -s raw/decomposition_A1.json ] && echo WRITTEN || echo MISSING)"

for f in ECHO STOP CV; do
  "$PY" code/decompose.py --run "$RUN" --arm "$f" --vs CV,STOP --inputs "$EXPORT" \
        --out "raw/decomposition_${f}.json" > /dev/null 2>&1
done
say "floor decompositions: $(ls raw/decomposition_*.json 2>/dev/null | wc -l) files"

"$PY" code/manifest.py --run "$RUN" --bank "$BANK" --pkg "$PKG" --out MANIFEST.json \
      > raw/manifest.stdout.txt 2>&1
say "manifest: $([ -s MANIFEST.json ] && echo WRITTEN || echo MISSING)"

{
  echo "# FINISH REPORT — produced automatically when the run wrote summary.json"
  echo
  echo "Run: \`$RUN\`"
  echo
  echo '## The pre-registered bar (BAR-W7-1), as a literal comparison'
  echo
  echo '```'
  cat raw/read_summary.stdout.txt 2>/dev/null
  echo '```'
  echo
  echo '## Where A1 stands vs the floors (decomposition)'
  echo
  echo '```'
  cat raw/decompose.stdout.txt 2>/dev/null
  echo '```'
  echo
  echo '## Criteria gates (0 violations required)'
  for a in CV STOP ECHO A1; do
    echo
    echo "### $a"
    echo '```'
    head -4 "$RUN/criteria/$a.txt" 2>/dev/null || echo "(no criteria file for $a)"
    echo '```'
  done
  echo
  echo '## W5 report + failure gallery'
  echo
  echo "index: \`$RUN/report/index.html\` — $([ -s "$RUN/report/index.html" ] && echo PRESENT || echo ABSENT)"
  echo
  echo '⚠️ RESULT.md has NOT been rewritten by this finisher — the numbers, their evidence classes'
  echo 'and the verdict prose are a human/agent judgement, not a template fill. This file is the'
  echo 'input to that write-up, and every number in it comes from summary.json.'
} > FINISH_REPORT.md

cd "$REPO" || exit 2
# ⛔ ONE `git add` PER FILE. MEASURED 2026-09-20: a single multi-path `git add` hit
# `fatal: pathspec '…/bar_verdict.json' did not match any files` and **aborted the whole call**, so
# files that DID exist were never staged — and the per-file blob check then read INCONCLUSIVE for
# all of them, which looks like a mount problem rather than one missing sibling path.
for f in "$PKG/FINISH_REPORT.md" "$PKG/MANIFEST.json" \
         "$PKG/raw/bar_verdict.json" "$PKG/raw/decomposition_A1.json" \
         "$PKG/raw/read_summary.stdout.txt" "$PKG/raw/decompose.stdout.txt" \
         "$PKG/raw/finisher.log" "$PKG/raw/bench_launch.log"; do
  [ -f "$f" ] && git add -- "$f" >> "$LOG" 2>&1
done
for f in "$PKG/FINISH_REPORT.md" "$PKG/raw/bar_verdict.json" "$PKG/raw/decomposition_A1.json"; do
  i=$(git ls-files --stage -- "$f" 2>/dev/null | awk '{print $2}')
  w=$(git hash-object "$f" 2>/dev/null)
  if [ ${#i} -ne 40 ] || [ ${#w} -ne 40 ]; then say "INCONCLUSIVE staging $f"
  elif [ "$i" = "$w" ]; then say "VERIFIED staged $f"
  else say "MISMATCH staging $f"; fi
done
say "finisher done"
