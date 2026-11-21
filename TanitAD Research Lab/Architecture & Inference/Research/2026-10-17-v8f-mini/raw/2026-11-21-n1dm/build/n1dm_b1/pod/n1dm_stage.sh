#!/bin/bash
# n1dm_stage.sh COMMIT NFILES -- STAGE the N1DM pod folder. Run by the MM, once, AFTER the PI's second go and after the patch is landed (COMMIT = the full 40-hex sha of the landed N1DM commit; NFILES = the number of
# stack/ + taniteval/ entries that differ from the N1DH tree, from n1dm_delta_files.txt). It reads /root/strat1/n1dh (the N1DH tree = 1f217bf0 + the eval-141 grid, the kit) and writes ONLY /root/strat1/n1dm.
# Inputs, scp'd by the MM to /root/strat1/n1dm/incoming/ first: n1dm_delta.tar (the changed + new files, repo-relative), n1dm_tree.sha256, all_blobs.txt, and the scripts of this folder (n1dm_*.sh, hb_dump_n1dm.sh,
# pairing_n1dm.sh, dry_mint_n1dm.sh, render_argv_n1dm.py, n1dm_zuse_ckpt.py).
# Steps: (1) copy the N1DH tree, (2) unpack the delta over it and verify every delta file's sha256, (3) verify the whole tree against all_blobs.txt (git blob sha1, 1 grid file included), (4) stage the kit (hb wrapper
# sha256 74fd8445, kclip.py sha256 8e78e69e, blobcheck.py, read_tools) from n1dh, (5) install the scripts with the two placeholders substituted, bash -n each, refuse any @@ left.
# Exit 0 and the last line `ZZ N1DM STAGE OK ZZ` only if everything passed. Nothing in /root/strat1/n1dh is modified.
R=/root/strat1; S=$R/n1dh; N=$R/n1dm; IN=$N/incoming; PY=/root/venv/bin/python
COMMIT=$1; NF=$2
fail() { echo "ZZ N1DM STAGE FAIL ZZ $1"; exit 1; }
[ ${#COMMIT} = 40 ] && echo "$COMMIT" | grep -qE '^[0-9a-f]{40}$' || fail "COMMIT must be 40 hex chars"
echo "$NF" | grep -qE '^[0-9]+$' || fail "NFILES must be a number"
[ -d $S/tree ] || fail "no $S/tree"
[ -d $IN ] || fail "no $IN (scp the inputs first)"
for f in n1dm_delta.tar n1dm_tree.sha256 all_blobs.txt n1dm_chain.sh n1dm_arm.sh n1dm_gsuite.sh n1dm_dumps.sh hb_dump_n1dm.sh pairing_n1dm.sh dry_mint_n1dm.sh render_argv_n1dm.py n1dm_zuse_ckpt.py n1dm_launch.sh n1dm_marker_watch.sh; do
  [ -s $IN/$f ] || fail "missing $IN/$f"
done
[ -e $N/tree ] && fail "$N/tree already exists (refusing to overwrite; move it aside)"
mkdir -p $N/runs $N/done $N/out $N/b2 $N/hb
cp -a $S/tree $N/tree || fail "copy of the N1DH tree"
(cd $N/tree && tar -xf $IN/n1dm_delta.tar) || fail "unpack delta"
cp $IN/n1dm_tree.sha256 $IN/all_blobs.txt $N/
(cd $N/tree && sha256sum -c $N/n1dm_tree.sha256 > $N/stage_tree_verify.log 2>&1) || fail "delta sha256 (see $N/stage_tree_verify.log)"
$PY $S/blobcheck.py $N/tree $N/all_blobs.txt > $N/stage_blobcheck.log 2>&1 || fail "tree vs all_blobs.txt (see $N/stage_blobcheck.log): $(tail -c 300 $N/stage_blobcheck.log)"
[ "$(find $N/tree -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $N/all_blobs.txt)" ] || fail "file count != all_blobs.txt lines"
[ "$(find "$N/tree/TanitAD Research Lab" -type f | wc -l)" = 1 ] || fail "exactly one Research Lab file (the eval-141 grid) is required"
grep -q "eval_grid_windows.json" $N/all_blobs.txt || fail "all_blobs.txt does not list the eval-141 grid"
cp $S/hb/v8f_t1_eval_hb.py $N/hb/ && (cd $N/hb && echo "74fd8445e377e763d17b1fa4c60753f23b8240eb94d0bdf7f4633a9e3ef80608 *v8f_t1_eval_hb.py" | sha256sum -c) || fail "hb wrapper"
cp $S/kclip.py $S/blobcheck.py $N/ && (cd $N && echo "8e78e69e202e6fa1422ae680d6bbda3c24af0786ae5234b9f744fefb4004f53c *kclip.py" | sha256sum -c) || fail "kclip.py"
mkdir -p $N/read_tools && cp $S/read_tools/*.py $N/read_tools/ || fail "read_tools"
for f in n1dm_chain.sh n1dm_arm.sh n1dm_gsuite.sh n1dm_dumps.sh hb_dump_n1dm.sh pairing_n1dm.sh dry_mint_n1dm.sh n1dm_launch.sh n1dm_marker_watch.sh render_argv_n1dm.py n1dm_zuse_ckpt.py; do
  sed "s/@@N1DM_COMMIT@@/$COMMIT/g; s/@@N1DM_NFILES@@/$NF/g" $IN/$f > $N/$f || fail "install $f"
  grep -q '@@N1DM_' $N/$f && fail "placeholder left in $f"
  case $f in *.sh) bash -n $N/$f || fail "bash -n $f";; *.py) $PY -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" $N/$f || fail "compile $f";; esac
done
chmod +x $N/*.sh
sha256sum $N/n1dm_*.sh $N/hb_dump_n1dm.sh $N/pairing_n1dm.sh $N/dry_mint_n1dm.sh $N/render_argv_n1dm.py $N/n1dm_zuse_ckpt.py > $N/stage_scripts.sha256
echo "ZZ N1DM STAGE OK ZZ"
