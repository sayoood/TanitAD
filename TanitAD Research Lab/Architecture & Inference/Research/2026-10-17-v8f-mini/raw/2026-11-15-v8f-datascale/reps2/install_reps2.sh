#!/bin/bash
# install_reps2.sh -- DSCALE10K REPS2: put the kit on the LoRA pod (dev box, git-bash). NOT RUN by the builder (the builder wrote only under /root/dscale10k/dry_reps2/ on the pod).
# usage (MM): bash F:/TanitAD-scratch/dscale10k/reps2/install_reps2.sh
# Writes ONLY NEW files under /root/dscale10k/blockreps2/ (it refuses if that directory exists) and the two NEW argv files argv/UE_s2.A.json, argv/UE_s3.A.json (gen_argv_reps2.py refuses to overwrite a differing one).
# Nothing existing is modified: no argv, token, run, lock, marker or process is touched. Steps: (1) tar the kit, ship it, sha256 every file at both ends; (2) on the pod: patch_checker_reps2.py builds
# blockreps2/study/dscale10k_check.py from blockreps1/study/dscale10k_check.py (the pod copy that minted UE_s1's token); (3) gen_argv_reps2.py writes the two argv and PAIRING_PROOF.json (CPU, 1 s); (4) bash -n of every script.
set -u
HERE=F:/TanitAD-scratch/dscale10k/reps2
KIT="blockreps2.sh blockreps2_prep.sh chk_arm_reps2.sh gen_argv_reps2.py init_reps2.py assemble_evidence_reps2_ue.py verify_token_reps2.py patch_checker_reps2.py fake_leaf_reps2.sh reps2_marker_watch.sh dryrun_reps2.sh"
source /c/Users/Admin/lorapod/podenv.sh
die() { echo "INSTALL-REFUSED: $1"; exit 1; }
cd "$HERE" || die "no $HERE"
for f in $KIT; do [ -s "$f" ] || die "missing $f"; done
pssh "test ! -e /root/dscale10k/blockreps2" || die "/root/dscale10k/blockreps2 already exists (never overwritten)"
pssh "test -s /root/dscale10k/blockreps1/study/dscale10k_check.py && test -s /root/dscale10k/argv/UE_s0.A.json" || die "the reps1 checker or UE_s0.A.json is missing on the pod"
pssh "mkdir -p /root/dscale10k/blockreps2/study" || die "mkdir failed"
tar cf - $KIT | ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "tar xf - --no-same-owner -C /root/dscale10k/blockreps2" || die "ship failed"
sha256sum $KIT | awk '{n=$2; sub(/^\*/,"",n); print $1"  /root/dscale10k/blockreps2/"n}' > "$HERE/install_expected.sha256"
ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "sha256sum -c -" < "$HERE/install_expected.sha256" || die "a shipped file differs from the dev-box file"
pssh "cd /root/dscale10k && /root/venv/bin/python blockreps2/patch_checker_reps2.py" || die "patch_checker_reps2.py failed"
pssh "cd /root/dscale10k && /root/venv/bin/python blockreps2/gen_argv_reps2.py | tail -n 1" || die "gen_argv_reps2.py failed (a third change, or an existing differing argv)"
pssh "cd /root/dscale10k/blockreps2 && for f in *.sh; do bash -n \$f || exit 1; done && chmod +x *.sh && echo SYNTAX-OK" || die "bash -n failed"
echo "ZZ INSTALL-REPS2 DONE ZZ  (next: the launch commands in HANDOFF_REPS2.md)"
