#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 1): repoint /root/strat1/n1dh from cc8c98a1 to f2e91e33 (one test file). Keeps *.bcc8c copies. Nothing deleted.
set -euo pipefail
N=/root/strat1/n1dh
NEW=f2e91e33fbe69586d7e2c464f3c1eda7ac7ddcfc
OLD=cc8c98a1d1c4d53e59b7568cc40dcaad2015bc01
cd $N
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
[ -d done ] && [ "$(ls done | wc -l)" != "0" ] && { echo "N1DH done markers exist; refusing"; exit 1; }
f=stack/tests/test_v8f_n1dh.py
[ "$(sha256sum tmp_fix/test_v8f_n1dh.py | cut -d' ' -f1)" = "b8fd9a94531752fa6533d10aadab534fc15b400a34a7d7c138b1bd239db0fd3d" ] || { echo "fixed test sha"; exit 1; }
[ "$(sha256sum tree/$f | cut -d' ' -f1)" = "2c6724dfffc781f280d2687cca788affc072033d722240867e1bc9e2088a6e26" ] || { echo "tree test is not the cc8c98a1 blob"; exit 1; }
for x in n1dh_arm.sh n1dh_gsuite.sh n1dh_chain.sh n1dh_tree.sha256 all_blobs.txt; do cp -p $x $x.bcc8c; done
cp -p tree/$f tmp_fix/test_v8f_n1dh.py.bcc8c
cp tmp_fix/test_v8f_n1dh.py tree/$f
sed -i "s/^COMMIT=$OLD$/COMMIT=$NEW/" n1dh_arm.sh n1dh_gsuite.sh
sed -i "s/^2c6724dfffc781f280d2687cca788affc072033d722240867e1bc9e2088a6e26 \*stack\/tests\/test_v8f_n1dh.py$/b8fd9a94531752fa6533d10aadab534fc15b400a34a7d7c138b1bd239db0fd3d *stack\/tests\/test_v8f_n1dh.py/" n1dh_tree.sha256
sed -i "s/^17472d655c2d7a9b9f8a1a982a6959e87421244a stack\/tests\/test_v8f_n1dh.py$/3792022ec9fa558bf2fe269361b8591bd9bec703 stack\/tests\/test_v8f_n1dh.py/" all_blobs.txt
sed -i "s/chain start commit=cc8c98a1 (n1dh)/chain start commit=f2e91e33 (n1dh)/" n1dh_chain.sh
for x in n1dh_arm.sh n1dh_gsuite.sh; do grep -q "^COMMIT=$NEW$" $x || { echo "$x not repointed"; exit 1; }; done
grep -q "^b8fd9a94531752fa6533d10aadab534fc15b400a34a7d7c138b1bd239db0fd3d \*stack/tests/test_v8f_n1dh.py$" n1dh_tree.sha256 || { echo "tree sha list"; exit 1; }
grep -q "^3792022ec9fa558bf2fe269361b8591bd9bec703 stack/tests/test_v8f_n1dh.py$" all_blobs.txt || { echo "blob list"; exit 1; }
[ "$(diff n1dh_tree.sha256 n1dh_tree.sha256.bcc8c | grep -c '^[<>]')" = "2" ] && [ "$(diff all_blobs.txt all_blobs.txt.bcc8c | grep -c '^[<>]')" = "2" ] || { echo "more than one line changed in a list"; exit 1; }
bash -n n1dh_arm.sh && bash -n n1dh_gsuite.sh && bash -n n1dh_chain.sh
(cd tree && sha256sum -c ../n1dh_tree.sha256) | tail -2
/root/venv/bin/python blobcheck.py tree all_blobs.txt | tail -1
echo "$(date -u +%FT%TZ) N1DH Amendment 1: *.bcc8c copies of n1dh_arm.sh n1dh_gsuite.sh n1dh_chain.sh n1dh_tree.sha256 all_blobs.txt and the test file; commit $NEW" >> $N/deletions.log
echo "ZZ REPOINT f2e91e33 OK ZZ"
