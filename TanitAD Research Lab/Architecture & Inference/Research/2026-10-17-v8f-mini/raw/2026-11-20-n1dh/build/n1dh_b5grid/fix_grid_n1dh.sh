#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 3): put the canonical eval-141 grid into /root/strat1/n1dh/tree, the file the trainer's DAGGER_EVAL_GRID (K-B3) reads
# for a dagger-bank arm. It is git blob c489b553 of the commit of record 1f217bf0; it was left out of the N1DH tree, and chk_tree refused any
# "TanitAD Research Lab" path. Keeps *.bgrid copies; moves done/tree.done and the failed chain log aside (renamed, nothing deleted).
set -euo pipefail
N=/root/strat1/n1dh
cd $N
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
G="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
SRC="/root/strat1/n1dc/tree/$G"
BLOB=c489b553fe9355cf50e6007bdf395bd4a07a6e9a
PY=/root/venv/bin/python
gblob() { $PY -I -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1"; }
[ ! -e "tree/$G" ] || { echo "the grid is already in the tree"; exit 1; }
[ ! -e "tree/TanitAD Research Lab" ] || { echo "the tree already has a Research Lab dir"; exit 1; }
[ "$(gblob "$SRC")" = "$BLOB" ] || { echo "source grid is not blob $BLOB"; exit 1; }
grep -q "eval_grid_windows" all_blobs.txt && { echo "all_blobs already lists a grid"; exit 1; }
for x in n1dh_chain.sh all_blobs.txt; do [ -e $x.bgrid ] && { echo "$x.bgrid exists"; exit 1; }; cp -p $x $x.bgrid; done
mkdir -p "$(dirname "tree/$G")"
cp -p "$SRC" "tree/$G"
[ "$(gblob "tree/$G")" = "$BLOB" ] || { echo "copied grid differs"; exit 1; }
printf '%s %s\n' "$BLOB" "$G" >> all_blobs.txt
# chk_tree: exactly one Research Lab file (the grid) instead of none
OLDL='  [ ! -e "$TREE/TanitAD Research Lab" ] || return 1'
NEWL='  [ "$(find "$TREE/TanitAD Research Lab" -type f | wc -l)" = 1 ] || return 1      # N1DH Amendment 3: the one Research Lab file is the eval-141 grid (K-B3), listed in all_blobs.txt'
grep -qxF "$OLDL" n1dh_chain.sh || { echo "chk_tree line not found"; exit 1; }
$PY -I - "$OLDL" "$NEWL" <<'EOF'
import sys
o, n = sys.argv[1], sys.argv[2]
p = "/root/strat1/n1dh/n1dh_chain.sh"
s = open(p, encoding="utf-8").read()
assert s.count(o + "\n") == 1
s = s.replace(o + "\n", n + "\n")
s = s.replace("# no extra file (a stray eval_grid_windows.json is gone)", "# no extra file (Amendment 3: the eval-141 grid is listed in all_blobs.txt)")
s = s.replace("chain start commit=1f217bf0 (n1dh)", "chain start commit=1f217bf0 (n1dh, Amendment 3)")
open(p, "w", encoding="utf-8").write(s)
EOF
bash -n n1dh_chain.sh
[ "$(diff n1dh_chain.sh n1dh_chain.sh.bgrid | grep -c '^[<>]')" = "6" ] || { echo "n1dh_chain.sh: not exactly 3 lines changed"; diff n1dh_chain.sh n1dh_chain.sh.bgrid; exit 1; }
[ "$(diff all_blobs.txt all_blobs.txt.bgrid | grep -c '^[<>]')" = "1" ] || { echo "all_blobs: not exactly one line added"; exit 1; }
[ "$(wc -l < all_blobs.txt)" = "3440" ] || { echo "all_blobs is not 3440 lines"; exit 1; }
# re-run the tree check exactly as the chain will (its step re-runs because tree.done moves aside)
TREE=$N/tree
(cd $TREE && sha256sum -c $N/n1dh_tree.sha256) | tail -1
$PY $N/blobcheck.py $TREE $N/all_blobs.txt | tail -1
nf=$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l); echo "tree files $nf (want 3440)"; [ "$nf" = "3440" ] || exit 1
[ "$(find "$TREE/TanitAD Research Lab" -type f | wc -l)" = 1 ] || { echo "Research Lab file count"; exit 1; }
mv done/tree.done done/tree.done_bgrid
mv n1dh_chain.log n1dh_chain_1f217bf0_kb3.log
mv runs/mint_N1DH_s0.log runs/mint_N1DH_s0_kb3.log
mv runs/N1DH_s0/gate/gate.log runs/N1DH_s0/gate/gate_kb3.log
echo "$(date -u +%FT%TZ) N1DH Amendment 3: *.bgrid copies of n1dh_chain.sh all_blobs.txt; grid blob $BLOB added to tree + all_blobs; done/tree.done -> tree.done_bgrid; chain log -> n1dh_chain_1f217bf0_kb3.log; mint log -> runs/mint_N1DH_s0_kb3.log; gate log -> runs/N1DH_s0/gate/gate_kb3.log" >> $N/deletions.log
echo "ZZ GRID FIX OK ZZ"
