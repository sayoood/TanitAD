#!/bin/bash
# verify: (1) RULETRAIN.patch applies to a clean copy of the LEADSUP tree and reproduces the PINS sha256; (2) apply_edits_train.py replayed on a clean copy of the LEADSUP trainer reproduces the same trainer sha256.
D=F:/TanitAD-scratch/v8proof/ruletrain
W=C:/Users/Admin/v8proof_c/ruletrain/tmp/verify2
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p $W/a/stack/scripts $W/a/stack/tanitad/data $W/a/stack/tests $W/b/stack/scripts
SRC=C:/Users/Admin/v8proof_c/leadsup/tree_patch
cp $SRC/stack/scripts/v8f_mini_train.py $W/a/stack/scripts/
cp $SRC/stack/scripts/v8f_mini_train.py $W/b/stack/scripts/
cd $W/a && patch -p1 --binary < $D/RULETRAIN.patch
echo "--- sha256 after patch vs PINS"
for f in $(grep -E '^(changed|new)' $D/PINS.txt | awk '{print $2}'); do
  want=$(grep -E "^(changed|new) +$f " $D/PINS.txt | awk '{print $NF}')
  got=$(sha256sum $W/a/$f | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "OK $f" || echo "BAD $f"
done
echo "--- edit script replay"
$PY $D/checks/apply_edits_train.py $W/b
want=$(grep -E "^changed +stack/scripts/v8f_mini_train.py" $D/PINS.txt | awk '{print $NF}')
got=$(sha256sum $W/b/stack/scripts/v8f_mini_train.py | cut -d' ' -f1)
[ "$want" = "$got" ] && echo "OK replay trainer sha256 $got" || echo "BAD replay $got != $want"
