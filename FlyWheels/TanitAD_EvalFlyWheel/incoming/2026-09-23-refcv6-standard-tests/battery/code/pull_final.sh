#!/bin/sh
# DEFERRED (SPEC A5, 2026-09-26). The FINAL is a post-switch checkpoint (A16 hybrid: F3 cascade loss +
# true label clock from step 34,500, resumed on commit 82c2331). Its G0 target (the in-run eval) carries
# the cascade term and corrected-clock labels, so it must be rolled and gated on the 82c2331 tree with the
# run's post-switch config. chain_final_v2.sh does exactly that; pull_final_v2.sh keeps this script's
# read-only 3-way-md5 pull (the original is kept as pull_final_orig.sh). The pre-switch chain
# (chain_milestones.sh) therefore ends after step 30000 with ZZNOFINALZZ carrying this marker.
echo "ZZFINALDEFERREDZZ final handled by chain_final_v2.sh on the 82c2331 tree (SPEC A5)"
exit 3
