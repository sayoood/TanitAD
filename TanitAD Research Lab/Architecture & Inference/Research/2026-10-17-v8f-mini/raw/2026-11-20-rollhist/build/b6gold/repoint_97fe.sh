#!/bin/bash
# MM 2026-11-21 (ROLLHIST Amendment 3): repoint /root/rollhist from 43319227 to 97fe0a56 (one test file). Keeps *.b43319 copies. Moves, never deletes.
set -euo pipefail
R=/root/rollhist
NEW=97fe0a56767320dbc5f4b823d67fb07a3209a745
OLD=433192275ba50f11c67bfc5df2e110d9cf465df2
cd $R
[ "$(tail -n1 rh_chain.log)" = "ZZ ROLLHIST FAIL ZZ step gsuite" ] || { echo "chain log last line is not the gsuite FAIL; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
[ "$(ls bank | wc -l)" = "0" ] && [ "$(ls runs | wc -l)" = "0" ] || { echo "bank or runs not empty; refusing"; exit 1; }
# 1. the one test file into tree_cdb
want=f5c69eae97e9718e
got=$(sha256sum tmp_gold/test_v8f_t1_readout.py | cut -c1-16)
[ "$got" = "$want" ] || { echo "new test sha $got != $want"; exit 1; }
cp -p tree_cdb/stack/tests/test_v8f_t1_readout.py tmp_gold/test_v8f_t1_readout.py.b43319
cp tmp_gold/test_v8f_t1_readout.py tree_cdb/stack/tests/test_v8f_t1_readout.py
# 2. blob list and commit pins
[ "$(sha256sum all_blobs_97fe0a56.txt | cut -c1-16)" = "95d6bbb6eead3fcb" ] || { echo "blob list sha"; exit 1; }
for f in rh_env.sh rh_gate_record.py rh_chain.sh rh_gsuite.sh; do cp -p $f $f.b43319; done
sed -i "s/^COMMIT=$OLD$/COMMIT=$NEW/" rh_env.sh
sed -i "s/^COMMIT = \"$OLD\"$/COMMIT = \"$NEW\"/" rh_gate_record.py
sed -i "s/all_blobs_43319227\.txt/all_blobs_97fe0a56.txt/g" rh_chain.sh rh_gsuite.sh
grep -q "^COMMIT=$NEW$" rh_env.sh || { echo "rh_env.sh not repointed"; exit 1; }
grep -q "^COMMIT = \"$NEW\"$" rh_gate_record.py || { echo "rh_gate_record.py not repointed"; exit 1; }
[ "$(grep -c all_blobs_97fe0a56.txt rh_chain.sh)" -ge 2 ] && [ "$(grep -c all_blobs_97fe0a56.txt rh_gsuite.sh)" -ge 2 ] || { echo "blob list name not repointed"; exit 1; }
grep -n "all_blobs_43319227\|^COMMIT.*$OLD" rh_env.sh rh_gate_record.py rh_chain.sh rh_gsuite.sh && { echo "old pin left"; exit 1; } || true
bash -n rh_env.sh && bash -n rh_chain.sh && bash -n rh_gsuite.sh
/root/venv/bin/python -c "import ast,sys; ast.parse(open('rh_gate_record.py').read())"
# 3. the tree verifies against the new list
/root/venv/bin/python blobcheck.py tree_cdb all_blobs_97fe0a56.txt | tail -1
[ "$(find tree_cdb -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < all_blobs_97fe0a56.txt)" ] || { echo "file count"; exit 1; }
# 4. moves (logged)
mv done done_43319227; mkdir done
mv rh_chain.log rh_chain_43319227.log; touch rh_chain.log
mv gsuite_result_rh.json gsuite_result_rh_43319227.json
echo "$(date -u +%FT%TZ) MOVED done -> done_43319227; rh_chain.log -> rh_chain_43319227.log; gsuite_result_rh.json -> gsuite_result_rh_43319227.json; *.b43319 copies of rh_env.sh rh_gate_record.py rh_chain.sh rh_gsuite.sh and the test file (Amendment 3, commit $NEW)" >> deletions.log
echo "ZZ REPOINT 97fe0a56 OK ZZ"
