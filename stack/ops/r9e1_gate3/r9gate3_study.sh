#!/usr/bin/env bash
# r9gate3_study.sh -- the ONE definition of "which refcv9 study is this chain / gate / bind working on".  SOURCED, never executed:
#   Thor side   r9gate3_chain.sh  r9gate3_thor_gate.sh  r9gate3_rec_explore.sh   (this file sits NEXT TO THEM in $R/kit; r9gate3_restart.sh ships it FIRST)
#   dev box     r9gate3_restart.sh  r9gate3_devbox_pinned.sh                       (next to them in the repo / hold; r9gate3_launch_pinned.ps1 gets it as -Study / -Stage)
#   bind        stack/ops/r9e1_bind.sh   (stack/ops/r9e1_gate3/ relative to itself; it takes the study from --argv-prefix, NEVER from R9_STUDY)
#
#   study  root                     argv prefix  launch-gate profile  explore arm  default arms  default REC12
#   e1     /home/nvidia/r8ns/r9e1   r9e1         refcv9-e1rig         C0           C0 C1         d297ba563d8f
#   e2     /home/nvidia/r8ns/r9e2   r9e2         refcv9-e2            K0           K0            (none: an e2 chain needs its own REC_C12 stamp)
#   e3     /home/nvidia/r8ns/r9e3   r9e3         refcv9-full          (1st arm)    (none)        (none: an e3 chain needs its own REC_C12 stamp)
#   e3 is ARM-AGNOSTIC (the E2 GO arm V is not known when this file is written): it has NO default arm and NO explore arm -- the operator names the arms
#   (r9gate3_restart.sh 2nd argument: 'V' for E3 alone, 'V_d1' for a D1-only re-entry after E3 ended; 'V V_d1' chains both but see E3_STUDY_NOTE.md) and r9_resolve_explore_arm
#   picks the first arm that is not a D1 branch (a D1-only list: its first arm).  A D1 arm is `<V>_d1` (r9e3_<V>_d1.argv.json, r9_e3_make_argv.py) and is gated under profile refcv9-full-d1 (r9_profile_of_arm).
#
# SELECTOR  R9_STUDY=e1|e2|e3.  UNSET = e1 = what every one of these scripts did before this file existed (same root, argv prefix, profile, explore arm,
# default arms, default REC12, dev-box stage, ship stage, markers and log lines).  A SET value that is not a study name -- INCLUDING THE EMPTY STRING --
# REFUSES (a message, the marker ZZR9STUDY-REFUSEDZZ, return 10): a typo must never fall back to e1, and an e1 chain must never read an e2 root.
# The selector reaches every child by being EXPORTED here (chain -> thor_gate final, restart -> ssh'd chain / explore / rec_explore); the one hop an
# environment cannot cross is the WMI launch (the provider's environment, not the caller's), so r9gate3_launch_pinned.ps1 writes it into the command line.
#
# VARIABLES (set by r9_study_load; never edit a script's copy of a value -- change it HERE):
#   R9_STUDY R9_ROOT R9_PREFIX R9_PROFILE R9_EXPLORE_ARM R9_DEFAULT_ARMS R9_DEFAULT_REC12 R9_DEVSTAGE R9_SHIP_STAGE R9_STUDY_NOTE
#   FUNCTIONS: r9_profile_of_arm <arm> (the gate profile of ONE arm: the study's, except an e3 `*_d1` arm -> refcv9-full-d1);
#              r9_resolve_explore_arm "<ARMS>" (fills an EMPTY R9_EXPLORE_ARM with the first non-D1 arm; a no-op for e1 / e2).
#   R9_STUDY_NOTE is "" for e1 and " study=e2 ..." for the others: it is appended to a status line so an e1 line stays what it was.
# R9_STUDY_NOAUTOLOAD=1 sources the functions only (r9e1_bind.sh: it must not read R9_STUDY).
R9_STUDIES="e1 e2 e3"
# THE GPU-LOCK WAIT (seconds), ONE value for every gate3 waiter of the ONE Thor lock: r9gate3_thor_gate.sh (explore + final) waits this long; the chain's
#   training wait is 2x (a timeout there STOPS the whole chain) and its explore deadlines (stage A smoke, stage C end) are extended by it, because the
#   explore itself may be waiting for the lock.  Why (MM 2026-11-04): two studies' chains (e1 arms, e2 K0) and the record queues share a NON-FIFO lock
#   and an arm trains ~5 h, so the old 3 h gate wait (6 h training, 90 min / 2 h explore) handed the loser of a boundary race a refusal -- a lost arm
#   -- instead of a later turn.  12 h > any single holder.  Not study-specific: set here, outside r9_study_load.  A non-integer or empty value REFUSES (return 10).
R9_LOCK_WAIT_S="${R9_LOCK_WAIT_S-43200}"      # UNSET = 12 h; an EMPTY value is refused, like an empty R9_STUDY
case "$R9_LOCK_WAIT_S" in
  ""|*[!0-9]*) echo "REFUSED: R9_LOCK_WAIT_S '$R9_LOCK_WAIT_S' is not a whole number of seconds"; echo "ZZR9LOCKWAIT-REFUSEDZZ"; return 10 2>/dev/null || exit 10;;
esac
r9_study_load() {   # r9_study_load <name>: set the R9_* variables of a study (and export R9_STUDY); return 1, setting NOTHING, on an unknown name
  case "${1-}" in
    e1) R9_ROOT=/home/nvidia/r8ns/r9e1; R9_PREFIX=r9e1; R9_PROFILE=refcv9-e1rig; R9_EXPLORE_ARM=C0; R9_DEFAULT_ARMS="C0 C1"
        R9_DEFAULT_REC12=d297ba563d8f; R9_DEVSTAGE=F:/refcv9_e1/gate; R9_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship; R9_STUDY_NOTE="";;
    e2) R9_ROOT=/home/nvidia/r8ns/r9e2; R9_PREFIX=r9e2; R9_PROFILE=refcv9-e2; R9_EXPLORE_ARM=K0; R9_DEFAULT_ARMS="K0"
        R9_DEFAULT_REC12=""; R9_DEVSTAGE=F:/refcv9_e2/gate; R9_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e2
        R9_STUDY_NOTE=" study=e2 profile=refcv9-e2 root=/home/nvidia/r8ns/r9e2";;
    e3) R9_ROOT=/home/nvidia/r8ns/r9e3; R9_PREFIX=r9e3; R9_PROFILE=refcv9-full; R9_EXPLORE_ARM=""; R9_DEFAULT_ARMS=""
        R9_DEFAULT_REC12=""; R9_DEVSTAGE=F:/refcv9_e3/gate; R9_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e3
        R9_STUDY_NOTE=" study=e3 profile=refcv9-full root=/home/nvidia/r8ns/r9e3";;
    *) return 1;;
  esac
  R9_STUDY="$1"; export R9_STUDY
}
r9_profile_of_arm() {   # r9_profile_of_arm <arm> -> the launch-gate profile of that arm on stdout (needs a loaded study)
  # IDEMPOTENT (MM review 2026-11-11): r9gate3_chain.sh and r9gate3_devbox_pinned.sh assign R9_PROFILE="$(r9_profile_of_arm "$A")" INSIDE their per-arm loop, so the base
  # is the study's profile with any -d1 stripped -- built from the mutable R9_PROFILE, a D1 arm's refcv9-full-d1 leaked into the next arm (and a 2nd D1 arm got -d1-d1).
  local _base="${R9_PROFILE%-d1}"
  case "${R9_STUDY-}:${1-}" in
    e3:?*_d1) echo "${_base}-d1";;     # the D1 decay branch: its own profile / token (launch_gate.PROFILES["refcv9-full-d1"], the 4-flag delta)
    *) echo "$_base";;
  esac
}
r9_resolve_explore_arm() {   # r9_resolve_explore_arm "<ARMS>": an arm-agnostic study (e3: R9_EXPLORE_ARM empty) explores its FIRST arm that is not a D1 branch (a D1-only list: its first arm); return 1 on an empty list
  [ -n "${R9_EXPLORE_ARM-}" ] && return 0
  local _a; for _a in ${1-}; do
    case "$_a" in *_d1) ;; *) R9_EXPLORE_ARM="$_a"; return 0;; esac
  done
  for _a in ${1-}; do R9_EXPLORE_ARM="$_a"; return 0; done     # ONLY D1 branches (a D1-only re-entry after E3 ended): the first one is explored, its source checkpoint exists by then
  return 1
}
r9_study_of_prefix() {   # r9_study_of_prefix <argv prefix> -> the study name on stdout, return 1 (nothing printed) when no study owns that prefix
  local _s; for _s in $R9_STUDIES; do
    if ( r9_study_load "$_s" && [ "$R9_PREFIX" = "${1-}" ] ); then echo "$_s"; return 0; fi
  done
  return 1
}
if [ -z "${R9_STUDY_NOAUTOLOAD:-}" ]; then
  if ! r9_study_load "${R9_STUDY-e1}"; then
    echo "REFUSED: unknown R9_STUDY '${R9_STUDY-}' (known: $R9_STUDIES; UNSET means e1, an empty value is refused)"; echo "ZZR9STUDY-REFUSEDZZ"
    return 10 2>/dev/null || exit 10
  fi
fi
