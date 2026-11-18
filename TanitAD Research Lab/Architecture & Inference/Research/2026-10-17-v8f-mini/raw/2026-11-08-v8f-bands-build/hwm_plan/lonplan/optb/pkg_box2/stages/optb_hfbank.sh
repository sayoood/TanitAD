#!/usr/bin/env bash
# optb_hfbank.sh <tag>: bank the run's dev-box bank ($PULLS/optb_<tag>/) PRIVATE to HF and verify it (optb_hfbank.py). Needs OPTB_HF_REPO (owner/name); unset = refuse (rc 16), the chain records HF_BANK_PENDING and ends in STOP, never DONE.
# The token is read in place from Keys.txt by the python script; this stage never prints it. HF_HUB_DISABLE_XET=1; the tanitad venv.
source "$(dirname "${BASH_SOURCE[0]}")/optb_common.sh"; TAG="${1:?tag}"; OPTB_STAGE="hfbank:$TAG"
need_pkg optb_hfbank.py
[ -n "${OPTB_HF_REPO:-}" ] || die 16 "OPTB_HF_REPO is unset (owner/name of the PRIVATE HF repo)"
BD="$PULLS/optb_$TAG"; [ -s "$BD/BANK.done" ] || die 15 "$BD has no BANK.done: nothing to upload"
export HF_HUB_DISABLE_XET=1
"$PY" -I "${OPTB_HFBANK_PY:-$PKG/optb_hfbank.py}" --tag "$TAG" --bank-dir "$BD" --repo "$OPTB_HF_REPO" || die 12 "HF bank of $TAG failed or did not verify"
say "ok"
