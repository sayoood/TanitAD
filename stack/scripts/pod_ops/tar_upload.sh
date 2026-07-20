#!/bin/bash
set -e
cd /workspace/data/comma2k19/_epcache
ionice -c3 nice -n19 tar cf /workspace/tmp_val.tar comma2k19-val-61c46fca8f7f
echo VAL_TARRED
ionice -c3 nice -n19 tar cf /workspace/tmp_train.tar comma2k19-train-b40a21eb5216
echo TRAIN_TARRED
cd /workspace/TanitAD/stack
python /workspace/tar_upload.py
rm -f /workspace/tmp_val.tar /workspace/tmp_train.tar
echo ALL_DONE
