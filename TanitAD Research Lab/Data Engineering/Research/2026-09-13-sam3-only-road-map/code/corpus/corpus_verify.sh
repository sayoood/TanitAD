#!/bin/bash
# Before the corpus launch: (A) the validated driver on [bogus clip, valid clip] = deliberate regression, the bogus clip's failed
# prebuild poisons the valid clip; (B) the corpus driver on the same list: the bogus clip fails alone, the valid clip's map must be
# bit-identical to production_test2 (the validated production output).
V=/home/nvidia/sam3map/corpus_verify; SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
mkdir -p $V; cd $SM/eval || exit 1
printf "<bogus-clip-id>\n41257700ce38\n" > $V/clips.txt
export PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PROD_SCRATCH=$V/scratch_old $PY sam3map_prod_v1_validated.py $V/clips.txt $V/old_out > $V/old.log 2>&1 < /dev/null
echo "A exit $?" >> $V/verify.log
PROD_SCRATCH=$V/scratch_new SRC_CAM=$SM/data/frontwide SRC_CAL=$SM/data/calib PROD_WAIT_S=0 PROD_MAX_FAILS=2 PROD_KEEP_FAILED_WORK=20 \
  $PY sam3map_prod.py $V/clips.txt $V/new_out > $V/new.log 2>&1 < /dev/null
echo "B exit $?" >> $V/verify.log
S=$($PY -c 'import hashlib; print(hashlib.sha256(b"41257700ce38").hexdigest()[:12])')
for f in sam3mapgt.npz worldmap.npz; do $PY gt_npz_equal.py $SM/production_test2/$S.$f $V/new_out/$S.$f >> $V/verify.log 2>&1; done
echo "ZZVERIFY-END" >> $V/verify.log
