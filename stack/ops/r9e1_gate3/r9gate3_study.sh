#!/usr/bin/env bash
# r9gate3_study.sh -- the ONE definition of "which refcv9 study is this chain / gate / bind working on".  SOURCED, never executed:
#   Thor side   r9gate3_chain.sh  r9gate3_thor_gate.sh  r9gate3_rec_explore.sh   (this file sits NEXT TO THEM in $R/kit; r9gate3_restart.sh ships it FIRST)
#   dev box     r9gate3_restart.sh  r9gate3_devbox_pinned.sh                       (next to them in the repo / hold; r9gate3_launch_pinned.ps1 gets it as -Study / -Stage)
#   bind        stack/ops/r9e1_bind.sh   (stack/ops/r9e1_gate3/ relative to itself; it takes the study from --argv-prefix, NEVER from R9_STUDY)
#
#   study  root                     argv prefix  launch-gate profile  explore arm  default arms  default REC12
#   e1     /home/nvidia/r8ns/r9e1   r9e1         refcv9-e1rig         C0           C0 C1         d297ba563d8f
#   e2     /home/nvidia/r8ns/r9e2   r9e2         refcv9-e2            K0           K0            (none: an e2 chain needs its own REC_C12 stamp)
#
# SELECTOR  R9_STUDY=e1|e2.  UNSET = e1 = what every one of these scripts did before this file existed (same root, argv prefix, profile, explore arm,
# default arms, default REC12, dev-box stage, ship stage, markers and log lines).  A SET value that is not a study name -- INCLUDING THE EMPTY STRING --
# REFUSES (a message, the marker ZZR9STUDY-REFUSEDZZ, return 10): a typo must never fall back to e1, and an e1 chain must never read an e2 root.
# The selector reaches every child by being EXPORTED here (chain -> thor_gate final, restart -> ssh'd chain / explore / rec_explore); the one hop an
# environment cannot cross is the WMI launch (the provider's environment, not the caller's), so r9gate3_launch_pinned.ps1 writes it into the command line.
#
# VARIABLES (set by r9_study_load; never edit a script's copy of a value -- change it HERE):
#   R9_STUDY R9_ROOT R9_PREFIX R9_PROFILE R9_EXPLORE_ARM R9_DEFAULT_ARMS R9_DEFAULT_REC12 R9_DEVSTAGE R9_SHIP_STAGE R9_STUDY_NOTE
#   R9_STUDY_NOTE is "" for e1 and " study=e2 ..." for the others: it is appended to a status line so an e1 line stays what it was.
# R9_STUDY_NOAUTOLOAD=1 sources the functions only (r9e1_bind.sh: it must not read R9_STUDY).
R9_STUDIES="e1 e2"
r9_study_load() {   # r9_study_load <name>: set the R9_* variables of a study (and export R9_STUDY); return 1, setting NOTHING, on an unknown name
  case "${1-}" in
    e1) R9_ROOT=/home/nvidia/r8ns/r9e1; R9_PREFIX=r9e1; R9_PROFILE=refcv9-e1rig; R9_EXPLORE_ARM=C0; R9_DEFAULT_ARMS="C0 C1"
        R9_DEFAULT_REC12=d297ba563d8f; R9_DEVSTAGE=F:/refcv9_e1/gate; R9_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship; R9_STUDY_NOTE="";;
    e2) R9_ROOT=/home/nvidia/r8ns/r9e2; R9_PREFIX=r9e2; R9_PROFILE=refcv9-e2; R9_EXPLORE_ARM=K0; R9_DEFAULT_ARMS="K0"
        R9_DEFAULT_REC12=""; R9_DEVSTAGE=F:/refcv9_e2/gate; R9_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e2
        R9_STUDY_NOTE=" study=e2 profile=refcv9-e2 root=/home/nvidia/r8ns/r9e2";;
    *) return 1;;
  esac
  R9_STUDY="$1"; export R9_STUDY
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
