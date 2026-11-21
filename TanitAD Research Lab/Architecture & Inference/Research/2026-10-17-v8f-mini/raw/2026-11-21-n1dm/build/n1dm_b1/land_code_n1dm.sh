#!/bin/bash
# land_code_n1dm.sh -- MM 2026-11-21: the N1DM code (the 14 files of changed_files.txt, from build/n1dm_b1/tree) as ONE commit on a NEW branch
# agent/v8f-n1dm-20261121 whose parent is 1f217bf0 (agent/v8f-n1dh-20261120). Nothing lands on agent/arch-inf-20260803 (RefCV9's closure freeze).
# Scratch-index plumbing on the C: mirror; the repo's tools/secret_scan.py --tree on a staging copy first. Usage: bash land_code_n1dm.sh [--dry]
set -euo pipefail
DRY=${1:-}
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1
M=C:/Users/Admin/tanitad-push
BASE=1f217bf0
BR=agent/v8f-n1dm-20261121
SP=C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
STG=F:/TanitAD-scratch/l6s
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P=$(git -C $M rev-parse --verify "${BASE}^{commit}")
case "$P" in 1f217bf0*) ;; *) echo "base is not 1f217bf0: $P"; exit 1;; esac
NB=$(git -C $M ls-remote origin refs/heads/agent/v8f-n1dh-20261120 | cut -f1)
echo "parent $P ; remote agent/v8f-n1dh-20261120 = $NB"
[ -z "$(git -C $M ls-remote origin refs/heads/$BR)" ] || { echo "branch $BR exists on the remote: refusing"; exit 1; }
mapfile -t FILES < <(tr -d '\r' < $B/changed_files.txt)
[ ${#FILES[@]} = 14 ] || { echo "expected 14 files, got ${#FILES[@]}"; exit 1; }
# 1. staging copy + secret scan
rm -rf "F:/TanitAD-scratch/l6s"; mkdir -p "$STG" "$SP/l6_tools"
for f in "${FILES[@]}"; do mkdir -p "$STG/$(dirname "$f")"; cp "$B/tree/$f" "$STG/$f"; done
git -C $M show "$P:tools/secret_scan.py" > "$SP/l6_tools/secret_scan.py"
scrc=0; "$PY" -I "$SP/l6_tools/secret_scan.py" --tree "$STG" --json "$B/secret_scan_code.json" > "$B/secret_scan_code.log" 2>&1 || scrc=$?
tail -n 3 "$B/secret_scan_code.log"
[ $scrc -eq 0 ] || { echo "SECRET SCAN rc $scrc: refusing"; exit 1; }
# 2. scratch index from the parent; blobs must equal the build's newhash.txt (git hash-object of the tree files)
export GIT_INDEX_FILE=$SP/idx_l6
rm -f "$GIT_INDEX_FILE"; git -C $M read-tree $P
for f in "${FILES[@]}"; do
  b=$(git -C $M hash-object -w --no-filters "$B/tree/$f")
  want=$(awk -v p="$f" '$2==p{print $1}' $B/newhash.txt)
  [ "$b" = "$want" ] || { echo "blob of $f is $b, newhash.txt says $want"; exit 1; }
  git -C $M update-index --add --cacheinfo 100644,$b,"$f"
done
TREE=$(git -C $M write-tree)
# 3. revert guard: exactly the 14 paths change (A or M), nothing else
git -C $M diff-tree -r --name-status $P $TREE > $SP/l6_diff.txt
nX=$(awk '$1!="A" && $1!="M"' $SP/l6_diff.txt | wc -l)
nT=$(wc -l < $SP/l6_diff.txt)
diff <(cut -f2 $SP/l6_diff.txt | sort) <(printf '%s\n' "${FILES[@]}" | sort) || { echo "REVERT GUARD: changed paths differ from changed_files.txt"; exit 1; }
echo "diff vs parent: $(awk '$1=="A"' $SP/l6_diff.txt | wc -l) A, $(awk '$1=="M"' $SP/l6_diff.txt | wc -l) M, other $nX, total $nT"
[ "$nX" = 0 ] && [ "$nT" = 14 ] || { echo "REVERT GUARD"; exit 1; }
if [ "$DRY" = "--dry" ]; then echo "ZZ L6 DRY OK ZZ tree $TREE"; rm -f "$GIT_INDEX_FILE"; exit 0; fi
MSG="v8f N1DM: --aux-route modes, a mode-routed plan head (K = 3 zero-init residual branches of head_o, soft route posterior)

H-V8F-NAVRB2-1, the N1DH card's NO-GAIN lever after N1DH read N1DH-COST. New mode --aux-route modes
= the N1DH ce half (route_head CE 0.5 on own-code rows) + K = 3 zero-init Linear(1024 -> 24, no bias)
branches built last under a forked RNG; plan coef = head_o(h) + sum_k p_k branch_k(h), with
p = softmax(route_head(h)).detach(), soft, no argmax, no route_emb. off and cond unchanged bit for bit.
G-DVB (bad_aux_route), built_levers, G-LIVE probe aux_route_probe, checker kind N1DM in strat1_check.py.
Tests: test_v8f_n1dm.py 39/39 (Z0a/Z0b/Z0c, Z1, REACH with exact-0.0 controls, FORMULA, permutation),
census tests updated; 21 mutants RED; whole-file regression of 108 files against the 1f217bf0 base.
Parent 1f217bf0 (agent/v8f-n1dh-20261120). Pre-registration PREREG_V8F_N1DM.md; not run on a pod.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
C=$(git -C $M commit-tree $TREE -p $P -m "$MSG")
echo "commit $C tree $TREE"
git -C $M push origin $C:refs/heads/$BR 2>&1 | tail -3
R2=$(git -C $M ls-remote origin refs/heads/$BR | cut -f1)
[ "$R2" = "$C" ] && echo "ZZ L6 PUSHED-OK $BR $C" || { echo "ZZ L6 PUSH-MISMATCH remote=$R2 local=$C"; exit 1; }
rm -f "$GIT_INDEX_FILE"
