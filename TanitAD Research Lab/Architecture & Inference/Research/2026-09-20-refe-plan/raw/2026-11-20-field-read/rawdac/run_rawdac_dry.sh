#!/bin/bash
cd /c/Users/Admin/swp/rawdac; . ./env.sh
O=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read/rawdac
for arm in C1 C0; do
  a=$(echo $arm | tr A-Z a-z)
  for s in 0 1; do
    PYTHONPATH="$NAV;$REFE" "$PYN" "$FD/field_rawdac.py" score --arm $arm --proposals $SEAMS/stage2_p5c_$a.proposals.npz --out $O/rawdac_${arm}_s${s}of2.jsonl --shard $s/2 --tokens "$C3" > $O/rawdac_${arm}_s${s}of2.log 2>&1 < /dev/null &
  done
  wait
  echo "ZZDRY_ARM_DONE $arm $(date -u +%FT%H:%MZ)" >> $O/dry_progress.txt
done
# RED: 150-token rolled arm for C1
PYTHONPATH="$NAV;$REFE" "$PYN" "$FD/field_rawdac.py" score --arm C1 --red --limit 150 --proposals $SEAMS/stage2_p5c_c1.proposals.npz --out $O/rawdac_C1_RED.jsonl --tokens "$C3" > $O/rawdac_C1_RED.log 2>&1 < /dev/null
echo "ZZDRY_ALL_DONE $(date -u +%FT%H:%MZ)" >> $O/dry_progress.txt
