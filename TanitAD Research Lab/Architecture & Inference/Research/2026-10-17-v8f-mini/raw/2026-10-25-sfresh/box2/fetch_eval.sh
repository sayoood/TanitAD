#!/bin/bash
# fetch the 141 eval clips from box 2 by sftp (READ ONLY); retries, resumes by skipping complete files
cd "$(dirname "$0")"
for attempt in 1 2 3 4 5 6 7 8; do
  : > batch_eval.sftp
  miss=0
  while read -r perm n u g size mon day tm name; do
    f=../cache/eval/clips/$name
    if [ ! -f "$f" ] || [ "$(stat -c %s "$f")" != "$size" ]; then
      rm -f "$f"; echo "get /D:/tanitad/cache_full/eval/clips/$name ../cache/eval/clips/$name" >> batch_eval.sftp; miss=$((miss+1))
    fi
  done < eval_clips_ls.txt
  echo "attempt $attempt missing $miss"
  [ $miss = 0 ] && { echo FETCH-COMPLETE; exit 0; }
  sftp -b batch_eval.sftp tanitad-box2 > sftp_eval_$attempt.log 2>&1
  sleep 5
done
echo FETCH-INCOMPLETE; exit 1
