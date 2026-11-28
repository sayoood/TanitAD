#!/bin/bash
# Builds the TF-N7 run tree: COPIES (never links; NTFS) of the TF-N6 run tree plus the N7 files. The TF-N6 tree and the pinned A1 tree are only READ.
# usage: build_run_tree.sh <dest run dir, e.g. C:/Users/Admin/n7nt/run> [<src N6 run tree, default C:/Users/Admin/n6nt/run>]
# Needs ~140 MB on the destination drive. Does NOT fetch inputs/n6t/n6_plans_N6T.npz (HF, see HANDOFF.md) and never deletes anything.
set -euo pipefail
DEST="${1:?destination run dir}"
SRC="${2:-C:/Users/Admin/n6nt/run}"
HOLD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF="$(cd "$HOLD/.." && pwd)"
[ -d "$SRC/w" ] && [ -d "$SRC/n5" ] && [ -d "$SRC/n6" ] || { echo "REFUSING: $SRC is not a TF-N6 run tree"; exit 1; }
[ ! -e "$DEST/n7" ] || { echo "REFUSING: $DEST/n7 exists (not overwriting a tree)"; exit 1; }
mkdir -p "$DEST/n6" "$DEST/n7" "$DEST/inputs/n6t"
for d in inputs n2 n4 n5 repo w; do cp -r "$SRC/$d" "$DEST/"; done
# n6/: the PINNED TF-N6 files only (NOT N6_PINS.json, NOT the card)
for f in n6_make_pins.py n6_nt_analyze.py n6_nt_build.py n6_nt_chain.py test_n6_chain.py; do cp "$SRC/n6/$f" "$DEST/n6/$f"; done
# n7/: the new files (the three n7_* generator modules are the CHAIN COPIES in this hold)
for f in n7_lib.py n7_select.py n7_analyze.py n7_nt_build.py n7_nt_chain.py n7_nt_analyze.py n7_make_pins.py test_n7_chain.py fixture_navtrain_tiers.json; do cp "$HOLD/$f" "$DEST/n7/$f"; done
# inputs/n6t/: the banked TF-N6 artifacts the N7 build reads (the N6-T plan file is fetched separately)
cp "$TF/n6_tol/run/build/n6_build_report.json" "$DEST/inputs/n6t/n6_build_report.json"
tar -xJf "$TF/n6_tol/run/harness_csv_n6.tar.xz" -C "$DEST/inputs/n6t" "refe_tfn6_n6t/refe_tfn6_n6t.csv"
mv "$DEST/inputs/n6t/refe_tfn6_n6t/refe_tfn6_n6t.csv" "$DEST/inputs/n6t/refe_tfn6_n6t.csv"
rmdir "$DEST/inputs/n6t/refe_tfn6_n6t"
echo "tree built at $DEST. STILL NEEDED: inputs/n6t/n6_plans_N6T.npz (sha256 1c79202f25b6747799d7cb6512d4985a7490cfc7619433c969c4d8a67c93a5ee, HF evidence/2026-11-13-tf-n6), inputs/ledger_K.json (banked ledger + TF-N7 n = 50), n7/PREREG_TF_N7.md (frozen card), n7/N7_PINS.json"
