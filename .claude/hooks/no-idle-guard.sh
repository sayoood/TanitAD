#!/bin/bash
# TanitAD no-idle guard — a Stop hook that refuses to let an AUTONOMOUS turn end on a report.
#
# WHY THIS EXISTS: across one session the PI flagged idling four times. The pattern is always the
# same — check the fleet, find the top item blocked, write a well-organised report, end the turn.
# CLAUDE.md forbids it and I broke the rule anyway, so the rule needs a MECHANISM, not intent.
#
# WHAT COUNTS AS WORK: a commit. Not a report, not a plan, not a status table. If the turn produced
# no commit, the backlog was not advanced, and the turn should continue.
#
# ⛔ THREE GUARDS so this can never trap the PI in a loop or nag during conversation:
#   1. stop_hook_active  -> exit immediately. A hook that re-fires on its own block is an
#                           infinite loop that burns credit. This is the non-negotiable one.
#   2. marker file       -> only enforces while `.claude/autonomous.on` exists. Conversational
#                           turns (answering a question, discussing a result) legitimately produce
#                           no commit and MUST NOT be blocked. The PI switches the mode.
#   3. time window       -> a commit within IDLE_MIN minutes counts as this turn's work.
#
# Turn it on :  touch .claude/autonomous.on
# Turn it off:  rm .claude/autonomous.on

set -u
IDLE_MIN=${TANITAD_IDLE_MIN:-30}

payload=$(cat)

# GUARD 1 — never re-fire on our own block.
case "$payload" in
  *'"stop_hook_active":true'*|*'"stop_hook_active": true'*) exit 0 ;;
esac

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

# GUARD 2 — autonomous mode only.
[ -f .claude/autonomous.on ] || exit 0

# GUARD 3 — did this turn produce a commit?
last=$(git log -1 --format=%ct 2>/dev/null) || exit 0
[ -n "$last" ] || exit 0
now=$(date +%s)
age_min=$(( (now - last) / 60 ))
[ "$age_min" -lt "$IDLE_MIN" ] && exit 0

# No commit in the window, and we are in autonomous mode -> the turn was a report. Continue it.
cat <<EOF
{"decision":"block","reason":"⛔ NO-IDLE GUARD: this autonomous turn produced no commit in the last ${age_min} minutes. A report is not work (CLAUDE.md). Do NOT summarise again and do NOT ask which item to do next — asking is itself the idle failure. Open 'Project Steering/BACKLOG.md', take the HIGHEST-VALUE UNBLOCKED item (not the cheapest), execute it, and commit. All GPUs are stopped, so every 0-GPU item is unblocked: the paper update, the DATA-vs-ARCH launch preflight, the sitclf L0 gold-set re-score, the REF-C improvement plan, four-family panels on banked windows. If you genuinely believe every item is blocked, say WHICH item and WHAT unblocks it, one line each — do not write a status report."}
EOF
exit 0
