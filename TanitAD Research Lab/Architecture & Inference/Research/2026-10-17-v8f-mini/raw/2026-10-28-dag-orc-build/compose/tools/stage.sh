#!/bin/bash
# build stage/ + diff.patch in the deliverable dir from the work tree
set -e
T=F:/TanitAD-scratch/compose_tree
O=F:/TanitAD-scratch/compose_tmp/orig
D=E:/Projects/TanitAD-artifacts/dagland/ego_dagger_compose
rm -rf $D/stage; mkdir -p $D/stage/stack/scripts $D/stage/stack/tanitad/data $D/stage/stack/tests
for f in stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_mini_data.py stack/tests/test_v8f_ego_dagger_compose.py stack/tests/ego_dagger_compose_golden.json; do cp $T/$f $D/stage/$f; done
: > $D/diff.patch
for f in stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_mini_data.py; do
  diff -u --label a/$f --label b/$f $O/$(basename $f) $T/$f >> $D/diff.patch || true
done
for f in stack/tests/test_v8f_ego_dagger_compose.py stack/tests/ego_dagger_compose_golden.json; do
  diff -uN --label a/$f --label b/$f /dev/null $T/$f >> $D/diff.patch || true
done
grep -c $'\r' $D/stage/stack/scripts/v8f_mini_train.py $D/stage/stack/tanitad/data/v8f_mini_data.py $D/stage/stack/tests/test_v8f_ego_dagger_compose.py || true
(cd $D/stage && sha256sum stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_mini_data.py stack/tests/test_v8f_ego_dagger_compose.py stack/tests/ego_dagger_compose_golden.json) > $D/SHA256SUMS.txt
cat $D/SHA256SUMS.txt; wc -l $D/diff.patch
