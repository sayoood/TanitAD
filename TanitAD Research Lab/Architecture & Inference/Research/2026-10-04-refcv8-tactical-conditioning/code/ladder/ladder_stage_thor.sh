#!/usr/bin/env bash
# Stage the ladder's launch tree + harness on Thor (dev box, Git Bash). SMALL by design (Thor disk, MM 2026-10-04):
# only the subtrees the trainer / loader / harness import -- stack/{tanitad,scripts,ops} + taniteval/taniteval
# (~18 MB) -- never the 267 MB taniteval results or stack/experiments.
#   SRC   a local tree whose files ARE the launch commit's blobs (asserted by the caller: `git archive` of the landed
#         commit with core.autocrlf=false, or the gated candidate re-verified blob-by-blob against the landing)
#   TAG   names the Thor dir: /home/nvidia/refcv8_ladder/tree_<TAG>; `tree` is then pointed at it
# Every file is md5-listed on the dev box and re-checked ON THOR from the shipped bytes; a mismatch refuses the switch.
set -eu
SRC=$1; TAG=$2
HOST=${HOST:-tanitad-thor}
CODE_SRC=${CODE_SRC:-"$(cd "$(dirname "$0")" && pwd)"}
R=/home/nvidia/refcv8_ladder
TMP=${TMPDIR:-/tmp}/ladder_stage_$$
mkdir -p "$TMP"
( cd "$SRC" && find stack/tanitad stack/scripts stack/ops taniteval/taniteval -type f -not -path "*__pycache__*" \
    -not -name "*.pyc" | sort > "$TMP/files.txt" )
( cd "$SRC" && tar -cf "$TMP/tree.tar" -T "$TMP/files.txt" && xargs -d '\n' md5sum < "$TMP/files.txt" > "$TMP/MD5SUMS" )
( cd "$CODE_SRC" && tar -cf "$TMP/code.tar" ladder_arms.py ladder_eval.py ladder_i0.py ladder_score.py ladder_chain.sh \
    && md5sum ladder_arms.py ladder_eval.py ladder_i0.py ladder_score.py ladder_chain.sh > "$TMP/CODE_MD5SUMS" )
echo "files $(wc -l < "$TMP/files.txt")  tar $(wc -c < "$TMP/tree.tar") B"
timeout 120 ssh -n "$HOST" "mkdir -p $R/tree_$TAG $R/code $R/W"
timeout 600 scp -q "$TMP/tree.tar" "$TMP/MD5SUMS" "$TMP/code.tar" "$TMP/CODE_MD5SUMS" "$HOST:$R/"
timeout 300 ssh -n "$HOST" "set -e; cd $R/tree_$TAG && tar -xf $R/tree.tar && md5sum -c --quiet $R/MD5SUMS \
  && cp $R/MD5SUMS $R/tree_$TAG/MD5SUMS && cd $R/code && tar -xf $R/code.tar && md5sum -c --quiet $R/CODE_MD5SUMS \
  && rm -f $R/tree.tar $R/code.tar && ln -sfn $R/tree_$TAG $R/tree && echo ZZSTAGED-\$(wc -l < $R/MD5SUMS)-ZZ"
rm -rf "$TMP"
