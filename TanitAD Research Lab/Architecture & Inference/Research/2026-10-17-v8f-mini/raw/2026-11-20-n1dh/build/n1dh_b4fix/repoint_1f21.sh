#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 2): repoint /root/strat1/n1dh from f2e91e33 to 1f217bf0 (one test file, stack/tests/test_v8f_n1dc.py, which becomes the 15th
# N1DH delta file vs bcf3ba3f). Keeps *.bf2e9 copies; moves the failed G-SUITE result and chain log aside (renamed, nothing deleted).
set -euo pipefail
N=/root/strat1/n1dh
NEW=1f217bf015ebd6d69b888e424a2675970ff72075
OLD=f2e91e33fbe69586d7e2c464f3c1eda7ac7ddcfc
cd $N
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
[ -e done/gsuite.done ] && { echo "gsuite.done exists; refusing"; exit 1; }
f=stack/tests/test_v8f_n1dc.py
NEWSHA=646e0608567062b6ecbc4904288c5593361dd89f5be0c584fd92b1a41051cb47
[ "$(sha256sum tmp_fix/test_v8f_n1dc.py.fixed | cut -d' ' -f1)" = "$NEWSHA" ] || { echo "fixed test sha"; exit 1; }
[ "$(sha256sum tree/$f | cut -d' ' -f1)" = "0aa25e8a29b93081a8dcdd62c463acead8d8f5b4ab490fde88c1558191e83974" ] || { echo "tree test is not the f2e91e33 blob"; exit 1; }
for x in n1dh_arm.sh n1dh_gsuite.sh n1dh_chain.sh n1dh_tree.sha256 all_blobs.txt; do [ -e $x.bf2e9 ] && { echo "$x.bf2e9 exists"; exit 1; }; cp -p $x $x.bf2e9; done
cp -p tree/$f tmp_fix/test_v8f_n1dc.py.tree_bf2e9
cp tmp_fix/test_v8f_n1dc.py.fixed tree/$f
sed -i "s/^COMMIT=$OLD$/COMMIT=$NEW/" n1dh_arm.sh n1dh_gsuite.sh
sed -i "s/^NFILES=14$/NFILES=15/; s/N1DH files (8 modified, 6 added)/N1DH files (9 modified, 6 added)/" n1dh_gsuite.sh
printf '%s *%s\n' "$NEWSHA" "$f" >> n1dh_tree.sha256
sed -i "s/^a90b24d39d07f92ba79d57cb904f74238f01302a stack\/tests\/test_v8f_n1dc.py$/02dc9af0af717af85fec86fbdc4455baf4cbb433 stack\/tests\/test_v8f_n1dc.py/" all_blobs.txt
sed -i "s/chain start commit=f2e91e33 (n1dh)/chain start commit=1f217bf0 (n1dh)/" n1dh_chain.sh
for x in n1dh_arm.sh n1dh_gsuite.sh; do grep -q "^COMMIT=$NEW$" $x || { echo "$x not repointed"; exit 1; }; done
grep -q "^NFILES=15$" n1dh_gsuite.sh || { echo "NFILES"; exit 1; }
[ "$(wc -l < n1dh_tree.sha256)" = "15" ] || { echo "tree list is not 15 lines"; exit 1; }
grep -q "^02dc9af0af717af85fec86fbdc4455baf4cbb433 stack/tests/test_v8f_n1dc.py$" all_blobs.txt || { echo "blob list"; exit 1; }
[ "$(diff all_blobs.txt all_blobs.txt.bf2e9 | grep -c '^[<>]')" = "2" ] || { echo "more than one line changed in all_blobs"; exit 1; }
[ "$(diff n1dh_gsuite.sh n1dh_gsuite.sh.bf2e9 | grep -c '^[<>]')" = "6" ] || { echo "n1dh_gsuite.sh: not exactly 3 lines changed"; exit 1; }
grep -q "chain start commit=1f217bf0 (n1dh)" n1dh_chain.sh || { echo "banner"; exit 1; }
bash -n n1dh_arm.sh && bash -n n1dh_gsuite.sh && bash -n n1dh_chain.sh
(cd tree && sha256sum -c ../n1dh_tree.sha256) | tail -2
/root/venv/bin/python blobcheck.py tree all_blobs.txt | tail -1
BT=/root/strat1/n1dc/tree       # n1dh_gsuite.sh line 13: BT=$R/n1dc/tree with R=/root/strat1
nd=$( (diff -rq -x __pycache__ $BT/stack tree/stack; diff -rq -x __pycache__ $BT/taniteval tree/taniteval) | grep -v '^$' | wc -l)
echo "tree vs base $BT: $nd differing entries (want 15)"; [ "$nd" = "15" ] || { echo "delta count"; exit 1; }
mv gsuite_result_n1dh.json gsuite_result_n1dh_f2e91e33.json
mv n1dh_chain.log n1dh_chain_f2e91e33.log
echo "$(date -u +%FT%TZ) N1DH Amendment 2: *.bf2e9 copies of n1dh_arm.sh n1dh_gsuite.sh n1dh_chain.sh n1dh_tree.sha256 all_blobs.txt + the test file; failed G-SUITE result -> gsuite_result_n1dh_f2e91e33.json, chain log -> n1dh_chain_f2e91e33.log; commit $NEW" >> $N/deletions.log
echo "ZZ REPOINT 1f217bf0 OK ZZ"
