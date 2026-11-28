#!/usr/bin/env bash
# r9e2_make_read_tree.sh <launch_tree> <kit_root> [<out_tree>]
# Builds the FROZEN READ TREE of the refcv9 E2 read: a copy of the e2 launch tree (stack/ + taniteval/) plus the FlyWheels packages the roll reads
# (r9e2_roll.sh `need` lines, r8_repoint.BATTERY_CODE, roll_seed_r8_train_pf.NF_HOLD).  Same idea as E1's launch_tree_frozen_<commit>.
#   <launch_tree>  the e2 launch tree (Thor: /home/nvidia/r8ns/r9e2/launch_tree; <launch_tree>.commit holds its commit)
#   <kit_root>     a dir that holds FlyWheels/TanitAD_EvalFlyWheel/incoming/<package>/... exported from git with core.autocrlf=false
#                  (Thor: /home/nvidia/r8ns/r9e2/inputs/kit_pkgs)
#   <out_tree>     default: <dirname launch_tree>/read_tree_<first 12 of the commit>  (REFUSED when it already exists: a frozen tree is never rewritten)
# ⚠️ The e2 chain restarts at a later tip: the launch tree CHANGES then.  RE-RUN this script on the NEW launch tree and point R9E2_TREE at the new output
#    (r9e2_read_env.thor.sh reads R9E2_TREE from the environment when set).  The tree is read-only data: it never touches the GPU.
set -eu
LT="${1:?usage: r9e2_make_read_tree.sh <launch_tree> <kit_root> [<out_tree>]}"; KR="${2:?kit_root}"
INC=FlyWheels/TanitAD_EvalFlyWheel/incoming
[ -d "$LT/stack" ] && [ -d "$LT/taniteval" ] || { echo "[make_read_tree] $LT has no stack/ + taniteval/" >&2; exit 4; }
if [ -n "${3:-}" ]; then OUT="$3"; else
  C="$(tr -d '[:space:]' < "$LT.commit" 2>/dev/null || true)"; [ -n "$C" ] || { echo "[make_read_tree] no $LT.commit: pass <out_tree>" >&2; exit 4; }
  OUT="$(dirname "$LT")/read_tree_${C:0:12}"
fi
[ ! -e "$OUT" ] || { echo "[make_read_tree] refusing: $OUT exists (a frozen tree is never rewritten)" >&2; exit 4; }
# the packages: (dir below $INC, one file that must exist in it)
PK="2026-10-20-refcv8-nav-feed-fix/PREREG_NAV_FEED.md 2026-10-20-refcv8-nav-feed-fix/code/overlay/navfeed_overlay.py
2026-11-03-refcv9-camblind/code/camblind_roll_r9.py
2026-10-27-refcv9-anticheat/PREREG_ANTICHEAT_REFCV8.md 2026-10-27-refcv9-anticheat/PREREG_ANTICHEAT_REFCV8.sha256 2026-10-27-refcv9-anticheat/code/anticheat_common.py
2026-10-25-refcv9-mapsel/PREREG_MAPSEL.md 2026-10-25-refcv9-mapsel/PREREG_MAPSEL.sha256 2026-10-25-refcv9-mapsel/code/mapsel_roll.py
2026-09-28-refcv7-standard-tests/battery/code/bank_tag.py"
for f in $PK; do [ -s "$KR/$INC/$f" ] || { echo "[make_read_tree] kit_root misses $INC/$f" >&2; exit 4; }; done
mkdir -p "$OUT.tmp" && cp -a "$LT/." "$OUT.tmp/" || exit 4
mkdir -p "$OUT.tmp/$INC"
for d in 2026-10-20-refcv8-nav-feed-fix 2026-11-03-refcv9-camblind 2026-10-27-refcv9-anticheat 2026-10-25-refcv9-mapsel; do
  [ ! -e "$OUT.tmp/$INC/$d" ] || { echo "[make_read_tree] the launch tree already holds $INC/$d" >&2; rm -rf "${OUT:?}.tmp"; exit 4; }
  cp -a "$KR/$INC/$d" "$OUT.tmp/$INC/$d" || exit 4
done
mkdir -p "$OUT.tmp/$INC/2026-09-28-refcv7-standard-tests/battery" && cp -a "$KR/$INC/2026-09-28-refcv7-standard-tests/battery/code" "$OUT.tmp/$INC/2026-09-28-refcv7-standard-tests/battery/code" || exit 4
mv "$OUT.tmp" "$OUT" && echo "[make_read_tree] built $OUT from $LT + $KR ($(du -sm "$OUT" | cut -f1) MB)"
