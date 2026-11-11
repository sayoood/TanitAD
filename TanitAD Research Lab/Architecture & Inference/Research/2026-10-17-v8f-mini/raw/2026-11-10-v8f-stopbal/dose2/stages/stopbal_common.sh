#!/usr/bin/env bash
# Sourced by every second-dose stage. Thin wrapper: it points the first dose's stopbal_common.sh (not edited; all fail-closed transport rules, ssh / sftp helpers, disk and GPU gates, put / pull
# verification) at the dose-2 package, so PKG = theory/stopbal/dose2 (its stopbal_check.py is the dose-2 layer over the first dose's checker). Everything else is the first dose's, unchanged.
D2_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; D2_HERE="$(cygpath -m "$D2_HERE" 2>/dev/null || echo "$D2_HERE")"
export STOPBAL_PKG="${STOPBAL_PKG:-$D2_HERE}"
D2_BASE="${STOPBAL_BASE:-$D2_HERE/../box2}"          # the first dose's package (read, never edited)
source "$D2_BASE/stages/stopbal_common.sh"
