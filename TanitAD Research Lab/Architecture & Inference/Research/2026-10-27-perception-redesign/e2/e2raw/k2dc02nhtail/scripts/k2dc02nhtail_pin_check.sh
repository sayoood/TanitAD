#!/bin/bash
# k2dc02nhtail_pin_check.sh [ROOT [MANIFEST_SHA256 [NLINES]]] -- Thor, READ ONLY. PREREG_K2DC02S1 sec. 2 item 4: before anything reads the comparator planner rolls of the K2DC02-tail read
# (cmp/E2_K0_pf = K0t, cmp_k2dc/E2_K2_pf = K2DCt, stageB/E2_K2_pf = K2DC02t, stageB/E2_K0_pf = K0t's second copy) every file must re-hash equal to the pinned manifest, and the manifest itself must hash to the pin.
#   ROOT          default /home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02 (holds K2DC02S1_CMP_PIN.sha256 and the four roll dirs). Only the tests pass another ROOT.
#   MANIFEST_SHA256  default 3d367ad922e4b250c1782b6204dad7b56ce9c0d83eae5955fe1959c5512f311c (the PREREG's pin). NLINES default 3524.
# Checks, each REFUSES (rc 3) on failure, in this order:
#   1 the manifest exists and its own sha256 == the pin;  2 it has NLINES lines, every path is one of the four pinned roll dirs (no other prefix, no `..`) and all four are present;
#   3 sha256sum -c over EVERY listed file (a missing or changed file refuses);  4 the four dirs hold NO file the manifest does not list (an extra file refuses);
#   5 the two K0t copies (cmp/E2_K0_pf, stageB/E2_K0_pf) carry identical hash lists.
# Writes nothing. Ends ZZK2DC02NHTAILPIN-OKZZ (rc 0) or ZZK2DC02NHTAILPIN-REFUSED <why> (rc 3).
export PATH=/usr/bin:/bin:$PATH; set -u
ROOT=${1:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02}
PIN=${2:-3d367ad922e4b250c1782b6204dad7b56ce9c0d83eae5955fe1959c5512f311c}
NL=${3:-3524}
MAN=$ROOT/K2DC02S1_CMP_PIN.sha256
die() { echo "ZZK2DC02NHTAILPIN-REFUSED $*"; exit 3; }
DIRS="cmp/E2_K0_pf cmp_k2dc/E2_K2_pf stageB/E2_K0_pf stageB/E2_K2_pf"
[ -s "$MAN" ] || die "no manifest $MAN"
GOT=$(sha256sum "$MAN" | cut -d' ' -f1); [ "$GOT" = "$PIN" ] || die "manifest sha256 $GOT != pin $PIN"
N=$(wc -l < "$MAN" | tr -d ' '); [ "$N" = "$NL" ] || die "manifest has $N lines, the pin says $NL"
BAD=$(grep -cvE '^[0-9a-f]{64}  (cmp/E2_K0_pf|cmp_k2dc/E2_K2_pf|stageB/E2_K0_pf|stageB/E2_K2_pf)/[^ ]+$' "$MAN"); [ "$BAD" = 0 ] || die "$BAD manifest line(s) outside the four pinned roll dirs or malformed"
grep -q '[.][.]/' "$MAN" && die "manifest has a .. path"
for d in $DIRS; do grep -q "  $d/" "$MAN" || die "manifest lists no file under $d"; done
CHK=$(cd "$ROOT" && sha256sum -c --quiet "$MAN" 2>&1); RC=$?
[ $RC = 0 ] || die "sha256sum -c failed: $(printf '%s' "$CHK" | head -3 | tr '\n' ' ')"
NF=0; for d in $DIRS; do NF=$((NF + $(cd "$ROOT" && find "$d" -type f | wc -l))); done
[ "$NF" = "$N" ] || die "the four dirs hold $NF files, the manifest lists $N (an unlisted file exists, or a listed one is not a regular file)"
A=$(grep '  cmp/E2_K0_pf/' "$MAN" | sed 's|  cmp/E2_K0_pf/|  |' | sort); B=$(grep '  stageB/E2_K0_pf/' "$MAN" | sed 's|  stageB/E2_K0_pf/|  |' | sort)
[ "$A" = "$B" ] || die "the two K0t copies carry different hash lists"
echo "ZZK2DC02NHTAILPIN-OKZZ $N files re-hashed equal to $MAN (manifest sha256 $GOT); no unlisted file in $DIRS"
