#!/bin/bash
# MM 2026-11-21 (ROLLHIST Amendment 6): run the study checker (mint and run) with cwd /root/dscale10k, where the DSCALE10K merge record's relative input path (in/manifest_10k.jsonl) resolves. Keeps chk_arm_rh.sh.bcwd.
set -euo pipefail
R=/root/rollhist; F=$R/chk_arm_rh.sh
[ -e $F.bcwd ] && { echo ".bcwd exists"; exit 1; }
A='if [ "$CMD" = mint ]; then'
[ "$(grep -cxF "$A" $F)" = "1" ] || { echo "anchor"; exit 1; }
cp -p $F $F.bcwd
awk -v a="$A" '$0==a{print "cd $DS                  # ROLLHIST Amendment 6: the DSCALE10K merge record lists in/manifest_10k.jsonl RELATIVE to /root/dscale10k (C-MERGE re-hashes it); every other path here is absolute"} {print}' $F.bcwd > $F
[ "$(diff $F $F.bcwd | grep -c '^[<>]')" = "1" ] || { echo "diff size"; exit 1; }
bash -n $F
(cd /root/dscale10k && /root/venv/bin/python -c "
import json,hashlib
r=json.load(open('merged/merge_record.json'))
bad=[k for k,v in r['inputs'].items() if hashlib.sha256(open(v['path'],'rb').read()).hexdigest()!=v['sha256']]
print('inputs differing from cwd /root/dscale10k:', bad)
assert not bad")
echo "$(date -u +%FT%TZ) ROLLHIST Amendment 6: chk_arm_rh.sh gains 'cd \$DS' before the checker call; copy chk_arm_rh.sh.bcwd kept" >> $R/deletions.log
sha256sum $F
echo "ZZ CWD FIX OK ZZ"
