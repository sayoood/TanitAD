#!/bin/bash
# pod3 data fetch: resumable comma tar download + extraction (survives ssh exit via setsid)
cd /workspace/data || exit 1
wget -c --tries=20 --waitretry=30 -o /workspace/data/wget_train.log \
  https://huggingface.co/datasets/Sayood/tanitad-comma2k19-episodes/resolve/main/comma_train.tar
echo TRAIN_TAR_DONE
tar -xf comma_val.tar && tar -xf comma_train.tar && echo EXTRACT_DONE
ls -d /workspace/data/*/
