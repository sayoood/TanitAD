#!/usr/bin/env bash
# mapft_ship.sh -- DEV-BOX side.  File-ship the MAPFT kit and the launch tree to the Jetson Thor.  md5-verified at BOTH ends,
# never git on Thor (CLAUDE.md: pods/Thor have no git credentials; a `git fetch` hangs and a checkout resets the tree).
#
#   mapft_ship.sh kit                        small (~100 KB): the Thor-side scripts + the smoke argv   -> /home/nvidia/r8ns/mapft/{kit,argv}
#   mapft_ship.sh tree <SHA40> [--repo R]    `git -c core.autocrlf=false archive SHA stack taniteval` (~457 MB) -> /home/nvidia/r8ns/mapft/launch_tree
#   mapft_ship.sh --dry-run <mode> ...       builds the tars LOCALLY, prints their md5 and the exact remote commands; touches NOTHING on Thor
#
# WHAT RUNS ON THOR: `tar -x` of the shipped archive into the MAPFT root (its own directory: it never touches NT's launch tree
# /home/nvidia/r8ns/nt/launch_tree).  The `tree` mode moves an existing launch_tree ASIDE (never deletes it).  Run only after NT A1's
# done-marker (the brief: nothing is copied to Thor while it trains).  The 457 MB tree is ~7 s of LAN and ~20 s of extraction.
#
# Markers (opaque, disjoint from anything grepped):  ZZMAPFTSHIP-KIT-OKZZ  ZZMAPFTSHIP-TREE-OKZZ  ZZMAPFTSHIP-REFUSEDZZ
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MODE="${1:-}"; shift || true
HOST=tanitad-thor; ROOT="${MAPFT_ROOT:-/home/nvidia/r8ns/mapft}"
THOR_PY="${MAPFT_THOR_PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
# MAPFT_LOCAL=1 (dev-box REHEARSAL ONLY): the "Thor" is a local directory ($MAPFT_ROOT); rsh runs the command in a local bash, rcp is cp.  Never set on launch day.
DEV_PY="${DEV_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MAPFT_REPO:-C:/Users/Admin/tanitad-push}"
STAGE="${MAPFT_SHIP_STAGE:-/f/refcv9_mapft_scratch/ship}"          # F: (NTFS); never G:
export MSYS_NO_PATHCONV=1
say() { echo "[mapft-ship:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZMAPFTSHIP-REFUSEDZZ"; exit 10; }
rsh() { if [ "$DRY" = 1 ]; then say "DRY ssh: $*"; return 0; fi; if [ "${MAPFT_LOCAL:-0}" = 1 ]; then bash -c "$1"; return $?; fi; ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@"; }
rcp() { if [ "$DRY" = 1 ]; then say "DRY scp: $1 -> $HOST:$ROOT/"; return 0; fi; if [ "${MAPFT_LOCAL:-0}" = 1 ]; then mkdir -p "$ROOT" && cp -p "$1" "$ROOT/"; else scp -q -o BatchMode=yes "$1" "$HOST:$ROOT/"; fi; }
md5of() { md5sum "$1" 2>/dev/null | cut -c1-32; }

# Thor-side kit = what mapft_prelaunch.sh / mapft_postrun.sh read at $ROOT/kit ($KIT) + the smoke argv the `smoke` mode wants at $ROOT/argv
KIT_FILES="mapft_prelaunch.sh mapft_grade_arm.py mapft_postrun.sh make_mapft_argv.py mapft_kfrozen.py mapft_gsuite_precheck.py refcv8_launch_argv_base.json mapft_argv_smoke.json pi_cost_approval_mapft.json"

case "$MODE" in
kit)
  mkdir -p "$STAGE" || die "cannot create $STAGE"
  for f in $KIT_FILES; do
    [ -s "$KIT/$f" ] || die "missing kit file $KIT/$f"
    # a CR in a shell script breaks it on Thor; a CR in the JSON changes its sha256 (the base argv's sha is pinned in make_mapft_argv.py)
    [ "$(tr -cd '\r' < "$KIT/$f" | wc -c)" = 0 ] || die "$f contains CR characters (a CRLF checkout/edit?)"
  done
  T="$STAGE/mapft_kit.tar"; rm -f "$T"
  tar -cf "$T" -C "$KIT" $KIT_FILES || die "tar failed"
  LM="$(md5of "$T")"; [ "${#LM}" = 32 ] || die "could not md5 $T"
  say "local tar $T md5=$LM bytes=$(stat -c %s "$T") files: $KIT_FILES"
  rsh "mkdir -p $ROOT/kit $ROOT/argv $ROOT/logs" || die "ssh mkdir failed"
  rcp "$T" || die "scp failed"
  RM="$([ "$DRY" = 1 ] && echo "$LM" || rsh "md5sum $ROOT/mapft_kit.tar | cut -c1-32" | tr -d '\r')"
  [ "$RM" = "$LM" ] || die "remote md5 '$RM' != local '$LM' (the transfer is corrupt: re-ship, never extract)"
  rsh "tar -xf $ROOT/mapft_kit.tar -C $ROOT/kit && chmod +x $ROOT/kit/*.sh && cp -p $ROOT/kit/mapft_argv_smoke.json $ROOT/argv/mapft_argv_smoke.json" || die "remote extract failed"
  # per-file md5 comparison, local vs remote, so a half-extracted kit cannot pass on the tar's md5 alone
  WANT="$(cd "$KIT" && md5sum $KIT_FILES | cut -c1-32 | tr '\n' ' ')"
  GOT="$([ "$DRY" = 1 ] && echo "$WANT" || rsh "cd $ROOT/kit && md5sum $KIT_FILES | cut -c1-32 | tr '\n' ' '" | tr -d '\r')"
  [ "$(echo $WANT)" = "$(echo $GOT)" ] || die "per-file md5 mismatch after extract: want [$WANT] got [$GOT]"
  rsh "bash -n $ROOT/kit/mapft_prelaunch.sh && bash -n $ROOT/kit/mapft_postrun.sh && echo SYNTAX-CLEAN" | grep -q SYNTAX-CLEAN || [ "$DRY" = 1 ] || die "bash -n failed on Thor"
  say "kit installed at $ROOT/kit (per-file md5 verified)"; echo "ZZMAPFTSHIP-KIT-OKZZ";;

tree)
  SHA="${1:-}"; shift || true
  while [ $# -gt 0 ]; do case "$1" in --repo) REPO="$2"; shift 2;; *) die "unknown argument $1";; esac; done
  [ "${#SHA}" = 40 ] || die "usage: tree <SHA40> [--repo R]  (40 hex chars)"
  C12="${SHA:0:12}"; GD="$REPO/.git"
  [ -d "$GD" ] || die "no git dir $GD"
  # the commit must exist and be the one you mean: print its subject so a typo is visible, and require the load-bearing paths
  git --git-dir="$GD" cat-file -e "$SHA^{commit}" 2>/dev/null || die "commit $SHA not found in $GD"
  for p in stack/scripts/launch_gate.py stack/scripts/refc_v3_train.py stack/tanitad/train/map_only_ft.py stack/scripts/assert_map_only_finetune.py stack/ops/sup_refcv7.sh stack/ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md; do
    git --git-dir="$GD" cat-file -e "$SHA:$p" 2>/dev/null || die "commit $SHA lacks load-bearing path $p"
  done
  say "commit $SHA :: $(git --git-dir="$GD" log -1 --format=%s "$SHA" | cut -c1-110)"
  mkdir -p "$STAGE" || die "cannot create $STAGE"
  T="$STAGE/launch_tree_$C12.tar"; rm -f "$T"
  git --git-dir="$GD" -c core.autocrlf=false archive --format=tar "$SHA" stack taniteval > "$T" || die "git archive failed"
  LM="$(md5of "$T")"; BYTES="$(stat -c %s "$T")"
  [ "${#LM}" = 32 ] && [ "${BYTES:-0}" -gt 100000000 ] || die "implausible archive ($BYTES bytes): the G:/ mount can truncate git reads -- re-run, never ship it"
  # byte-faithfulness: the box-audit literals the gate and bind check are blob md5s; a CRLF-converted export fails them
  X="$STAGE/x_$C12"; rm -rf "$X"; mkdir -p "$X"
  tar -xf "$T" -C "$X" stack/ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md stack/ops/refcv8_box_audit/raw/visibility/gbo_frameset.json || die "cannot extract the box-audit files from the tar"
  [ "$(md5of "$X/stack/ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md")" = 594c71196cc5bbd527fee40b2cb0e3f1 ] || die "box-audit prereg md5 != blob literal (CRLF-converted export)"
  [ "$(md5of "$X/stack/ops/refcv8_box_audit/raw/visibility/gbo_frameset.json")" = b291404c36f83c3e397e3b90367e8e7b ] || die "box-audit frame-set md5 != blob literal"
  say "local tar $T md5=$LM bytes=$BYTES (box-audit literals verified)"
  rsh "mkdir -p $ROOT" || die "ssh mkdir failed"
  rcp "$T" || die "scp failed"
  RM="$([ "$DRY" = 1 ] && echo "$LM" || rsh "md5sum $ROOT/launch_tree_$C12.tar | cut -c1-32" | tr -d '\r')"
  [ "$RM" = "$LM" ] || die "remote md5 '$RM' != local '$LM' (re-ship; nothing was extracted)"
  # old launch_tree moves ASIDE (never deleted here); the new one lands at the exact path the argv files embed (<ROOT>/launch_tree)
  rsh "cd $ROOT && { [ ! -e launch_tree ] || mv launch_tree launch_tree_aside_\$(date +%H%M%S); } && mkdir launch_tree && tar -xf launch_tree_$C12.tar -C launch_tree" || die "remote extract failed"
  # the import probe runs from /tmp with an explicit PYTHONPATH and must print a path INSIDE the new tree (editable-install shadow trap)
  P="$([ "$DRY" = 1 ] && echo "IMP $ROOT/launch_tree/stack/tanitad/__init__.py" || rsh "cd /tmp && PYTHONPATH=$ROOT/launch_tree/stack:$ROOT/launch_tree/taniteval $THOR_PY -c 'import tanitad; print(\"IMP\", tanitad.__file__)'" | tr -d '\r' | grep '^IMP ')"
  case "$P" in "IMP $ROOT/launch_tree/stack/"*) ;; *) die "import probe resolved outside the new tree: '$P'";; esac
  # one file the commit is known to carry, compared by blob content (md5 of the tree member vs the Thor file)
  for p in stack/scripts/launch_gate.py stack/tanitad/train/map_only_ft.py; do
    W="$(md5of "$X/$p" 2>/dev/null)"; [ -n "$W" ] || { tar -xf "$T" -C "$X" "$p" && W="$(md5of "$X/$p")"; }
    G="$([ "$DRY" = 1 ] && echo "$W" || rsh "md5sum $ROOT/launch_tree/$p | cut -c1-32" | tr -d '\r')"
    [ "${#W}" = 32 ] && [ "$W" = "$G" ] || die "$p: local $W != Thor ${G:-unreadable}"
  done
  echo "$SHA" > "$STAGE/launch_tree_commit.txt"
  rsh "echo $SHA > $ROOT/launch_tree.commit" || die "could not record the commit on Thor"
  say "launch tree $C12 installed at $ROOT/launch_tree; $P"; echo "ZZMAPFTSHIP-TREE-OKZZ";;

*) sed -n 2,13p "${BASH_SOURCE[0]}"; exit 2;;
esac
