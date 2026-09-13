#!/bin/bash
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
B="<scratchpad>/v2/build_frames_v2.py"
O=C:/Users/Admin/qwenvis/v2
export PYTHONIOENCODING=utf-8
rm -rf $O/frames_legacy $O/seq_*
for spec in "4fbd97b6a4b7 0.25,0.55,0.80" "73495082f98b 0.30,0.60,0.85" \
            "0d90d20036a3 0.35,0.70" "e4ca802fbd5f 0.40,0.75" \
            "2bb37d62b419 0.35,0.70" "6924358fafe0 0.30,0.65"; do
  set -- $spec
  $PY "$B" --clip $1 --fracs $2 --out $O/frames_legacy || echo "ZZLEGACY-FAIL-$1ZZ"
done
echo "ZZLEGACY-DONE $(ls -d $O/frames_legacy/*/ | wc -l)ZZ"
for clip in 4fbd97b6a4b7 73495082f98b 0d90d20036a3 6924358fafe0; do
  $PY "$B" --clip $clip --hz 5 --out $O/seq_${clip:0:8} || echo "ZZSEQ-FAIL-${clip:0:8}ZZ"
  echo "ZZSEQ-DONE-${clip:0:8} $(ls -d $O/seq_${clip:0:8}/*/ | wc -l)ZZ"
done
echo ZZBUILD-ALL-DONEZZ
