#!/usr/bin/env bash
# Thor side of the gate's evidence ship: verify EVERY shipped file by md5, then build the two trees
# from ONE git archive of the tip (tip/ = the archive as is; cab/ = the archive + the gate's
# files, each md5-verified after it is written), and assert which tanitad each tree imports.
#   bash setup_thor_trees.sh <gate dir> <tree tar> -> <gate dir>/SETUP_OK or SETUP_FAIL (the verdict)
set -u
G="${1:?gate dir}"
TAR="${2:?the tree tar in the gate dir (git -c core.autocrlf=false archive of the tip)}"
cd "$G" || exit 1
rm -f SETUP_OK SETUP_FAIL
fail() { echo "$1" > SETUP_FAIL; echo "SETUP_FAIL: $1"; exit 1; }
md5sum -c --quiet md5.txt > md5_check.txt 2>&1 || fail "md5 check failed: $(head -3 md5_check.txt)"
n=$(grep -c . md5.txt); echo "md5 OK over $n files" > md5_check.txt
for t in tip cab; do
  [ -e "$t" ] && fail "$t already exists (a fresh dir is required)"
  mkdir "$t" && tar -xf "$TAR" -C "$t" || fail "extract into $t failed"
done
( cd overlay && find . -type f | sed 's|^\./||' ) | while IFS= read -r f; do
  mkdir -p "cab/$(dirname "$f")"
  cp "overlay/$f" "cab/$f"
  a=$(md5sum < "overlay/$f" | cut -c1-32); b=$(md5sum < "cab/$f" | cut -c1-32)
  [ "$a" = "$b" ] || { echo "overlay $f md5 $b != $a" > SETUP_FAIL; exit 1; }
  echo "overlay $f $b" >> overlay_check.txt
done
[ -e SETUP_FAIL ] && exit 1
for t in tip cab; do
  PYTHONPATH="$G/$t/stack:$G/$t/taniteval" /home/nvidia/venvs/tanitad-train/bin/python -c \
    "import tanitad,sys; f=tanitad.__file__; print(f); sys.exit(0 if f.startswith(sys.argv[1]) else 3)" \
    "$G/$t/stack" >> import_check.txt 2>&1 || fail "tanitad in $t imports from elsewhere"
done
mkdir -p keys
date -u > SETUP_OK
echo SETUP_OK
