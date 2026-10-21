#!/bin/bash
# md5 every navtest frame-bank file on the dev box (source) and on Thor (destination); write both lists + the diff into the package.
set -u
L=/c/Users/Admin/refcv8_pf/navtest_read/raw/ship_verify; mkdir -p $L
cd /d/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank || exit 2
find . -type f | sort | xargs md5sum | sed 's# \*\./#  ./#' > $L/bank_md5_devbox.txt
ssh -n tanitad-thor 'cd /home/nvidia/r8ns/navtest/frame_bank && find . -type f | sort | xargs md5sum' > $L/bank_md5_thor.txt
echo "devbox_files=$(wc -l < $L/bank_md5_devbox.txt) thor_files=$(wc -l < $L/bank_md5_thor.txt)"
if diff <(sed 's/^\([0-9a-f]*\) *\*\{0,1\}/\1  /' $L/bank_md5_devbox.txt) $L/bank_md5_thor.txt > $L/bank_md5_diff.txt; then echo QQBANKMD5_MATCH; else echo QQBANKMD5_DIFF; fi
