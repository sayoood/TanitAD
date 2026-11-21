#!/usr/bin/env bash
# r9gate3_zg_env.sh <profile> <argv-file> <controls-env> <tree>  -- print the manifest's zero-gradient block for ONE arm.
#   stdout = the controls env's lines (ZEROGRAD_CONTROL ...) with its ZEROGRAD_ALLOW lines dropped, then the ZEROGRAD_ALLOW line DERIVED from the launch gate's
#   pre-registered dead groups for THIS arm's argv (stack/scripts/zerograd_allow_from_gate.py --emit bash).  Why (MEASURED 2026-11-10, refcv9 E2 K0): the chain
#   appended a controls env carrying a literal ZEROGRAD_ALLOW=(), so the supervisor's G5 failed 3 pre-registered dead params (core.decoder.r8_rc_gate,
#   r8_rc_to_cond.weight/.bias) that the derivation tool exempts: a tested tool whose output never reached its caller.
#   REFUSES (rc 2, NOTHING on stdout) if the tool fails or prints no ALLOW line, if the controls env is unreadable or has no ZEROGRAD_CONTROL,
#   or if it carries a NON-EMPTY ZEROGRAD_ALLOW that differs from the derived one.  ($PY = python, default python3.)
set -u
[ $# -eq 4 ] || { echo "usage: r9gate3_zg_env.sh <profile> <argv-file> <controls-env> <tree>" >&2; exit 2; }
PROFILE=$1; ARGVF=$2; CTRL=$3; TREE=$4; PYX="${PY:-python3}"
refuse() { echo "r9gate3_zg_env REFUSED: $*" >&2; exit 2; }
[ -s "$CTRL" ] || refuse "controls env $CTRL is absent or empty"
[ -s "$ARGVF" ] || refuse "argv file $ARGVF is absent or empty"
DERIVED="$("$PYX" "$TREE/stack/scripts/zerograd_allow_from_gate.py" --profile "$PROFILE" --argv "$ARGVF" --emit bash < /dev/null)" || refuse "zerograd_allow_from_gate.py failed (profile $PROFILE, argv $ARGVF)"
case "$DERIVED" in ZEROGRAD_ALLOW=\(*\)) ;; *) refuse "derived output is not one ZEROGRAD_ALLOW=(...) line";; esac
case "$DERIVED" in *$'\n'*) refuse "derived output has more than one line";; esac
# parse both sides by sourcing them in subshells (one regex per line), so the comparison is on values, not spelling
CTRL_N="$( ( ZEROGRAD_CONTROL=(); . "$CTRL" >/dev/null 2>&1; echo "${#ZEROGRAD_CONTROL[@]}" ) )"
[ "${CTRL_N:-0}" -ge 1 ] 2>/dev/null || refuse "$CTRL has no ZEROGRAD_CONTROL"
CTRL_ALLOW="$( ( ZEROGRAD_ALLOW=(); . "$CTRL" >/dev/null 2>&1; printf '%s\n' "${ZEROGRAD_ALLOW[@]}" ) | sort )"
DER_ALLOW="$( ( eval "$DERIVED"; printf '%s\n' "${ZEROGRAD_ALLOW[@]}" ) | sort )"
if [ -n "$(printf '%s' "$CTRL_ALLOW" | tr -d '[:space:]')" ] && [ "$CTRL_ALLOW" != "$DER_ALLOW" ]; then
  refuse "$CTRL carries a non-empty ZEROGRAD_ALLOW that differs from the derived one for $PROFILE (a hand-written exemption is never mixed in silently)"; fi
grep -v '^[[:space:]]*ZEROGRAD_ALLOW=' "$CTRL" | grep -v '^[[:space:]]*$'
printf '%s\n' "$DERIVED"  # EMIT-ALLOW
