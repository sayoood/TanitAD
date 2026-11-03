#!/usr/bin/env bash
# r9gate3_restart.sh <SHA40> -- DEV-BOX one-command re-entry of the E1 gate chain on a NEW tip (after the G-EVAL rig fix is landed).
#   ship tree -> suite plan + files -> kit -> the gate's own record job (OVF records must PASS) -> dev-box pinned driver (WMI) -> Thor explore + chain.
# Stops at the first refusal.  Everything long runs DETACHED; this script itself takes ~2 min.
export PATH="/usr/bin:/bin:$PATH"; export MSYS_NO_PATHCONV=1; set -u
SHA="${1:?usage: r9gate3_restart.sh <SHA40> [ARMS quoted, default C0 C1] [REC12 = the records ovf_<ARM>_<c12> suffix]}"; C12="${SHA:0:12}"; R=/home/nvidia/r8ns/r9e1; GD=C:/Users/Admin/tanitad-push/.git
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; KIT=C:/Users/Admin/refcv8_pf/refcv9_warm/kit; PLANGEN=C:/Users/Admin/refcv8_pf/nt_build/trainer/code/make_thor_suite_plan.py
DEVPY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe; TIPBASE=d297ba563d8fddc0ce9cfb8380408878a3778fb2
die() { echo "REFUSED: $*"; exit 10; }
ARMS="${2:-C0 C1}"; REC12="${3:-d297ba563d8f}"
echo "$REC12" | grep -qx "[0-9a-f]\{12\}" || die "REC12 must be 12 hex chars"
git --git-dir=$GD cat-file -e "$SHA^{commit}" || die "commit $SHA not in the mirror"
[ "$(git --git-dir=$GD ls-remote origin agent/arch-inf-20260803 | cut -c1-40)" = "$SHA" ] || die "$SHA is not the remote tip (git ls-remote)"
(cd $KIT && R9W_ROOT=$R R9W_SHIP_STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship bash r9w_ship.sh tree $SHA) | tail -1 | grep -q ZZR9WSHIP-TREE-OKZZ || die "tree ship"
NS=F:/refcv9_e1/gate/nts_$C12; mkdir -p F:/refcv9_e1/gate
"$DEVPY" -I $PLANGEN --repo C:/Users/Admin/tanitad-push --tip $TIPBASE --cand $SHA --out $NS > F:/refcv9_e1/gate/plan_$C12.log 2>&1 || die "suite plan (see F:/refcv9_e1/gate/plan_$C12.log)"
ssh -n tanitad-thor "mkdir -p $R/suite_$C12" && scp -q $NS/tree_tip.tar $NS/overlay.tar $NS/plan_thor.json tanitad-thor:$R/suite_$C12/ || die "suite files scp"
W="$(cd $NS && md5sum tree_tip.tar overlay.tar plan_thor.json | cut -c1-32 | tr '\n' ' ')"; G="$(ssh -n tanitad-thor "cd $R/suite_$C12 && md5sum tree_tip.tar overlay.tar plan_thor.json | cut -c1-32 | tr '\n' ' '")"
[ "$W" = "$G" ] || die "suite files md5 differ"
scp -q $HERE/r9gate3_chain.sh $HERE/r9gate3_thor_gate.sh $HERE/r9gate3_rec_explore.sh tanitad-thor:$R/kit/ && ssh -n tanitad-thor "chmod +x $R/kit/r9gate3_*.sh" || die "kit scp"
ssh -n tanitad-thor "mkdir -p $R/gate3 && echo '$ARMS' > $R/gate3/ARMS && echo $REC12 > $R/gate3/REC_C12 && cat $R/gate3/REC_C12" | grep -qx "$REC12" || die "arms/records stamp"
# the early record check runs on the arms whose records EXIST now (R9_PRECHECK_ARMS, default = all arms); the chain's own final gate re-reads
# every arm's records before it mints that arm's token, so an arm whose record is still being built only delays its own gate
PRE="${R9_PRECHECK_ARMS:-$ARMS}"
for A in $PRE; do ssh -n tanitad-thor "cd $R && bash kit/r9gate3_rec_explore.sh $A; grep -c 'G-MAP-OVERFIT PASS\|G-BOX-OVERFIT PASS' $R/gate3_explore_rec_${C12}_$A.log" | tail -1 | grep -qx 2 || die "OVF records not admissible for $A at $C12 (a module in their closures changed: re-run r9e1_bind.sh)"; done
[ -e F:/refcv9_e1/gate/pinned_status.txt ] && mv F:/refcv9_e1/gate/pinned_status.txt F:/refcv9_e1/gate/pinned_status.$(date +%H%M%S).old
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $HERE/r9gate3_launch_pinned.ps1)" -Sha $SHA -Arms "$ARMS" || die "pinned launch"
ssh -n tanitad-thor "cd $R || exit 1; nohup setsid bash kit/r9gate3_thor_gate.sh explore C0 > logs/r9gate3_explore_${C12}_C0.out 2>&1 < /dev/null &" ; sleep 2
ssh -n tanitad-thor "cd $R || exit 1; nohup setsid bash kit/r9gate3_chain.sh >> logs/r9gate3_chain.out 2>&1 < /dev/null &" ; sleep 3
ssh -n tanitad-thor "tail -2 $R/logs/r9gate3_chain.out | cut -c1-160"; echo "ZZR9GATE3-RESTART-OKZZ $C12"
