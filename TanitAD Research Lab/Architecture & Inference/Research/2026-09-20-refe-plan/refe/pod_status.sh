#!/bin/bash
# One status line for the session's event waiter (read-only; nothing is written). Fields, in order:
# sft3_evals sft3_exit sft3_update pdm_exit sft4_chain_lines sft4_evals sft4_exit sft4_update oom_kill lane_log_lines
# scorer_finetune_procs. Opaque markers (ZZS ... ZZE) so a client never matches its own echoed command.
cnt() { if [ -f "$2" ]; then grep -c -- "$1" "$2"; else echo 0; fi; }
lastupd() { [ -f "$1" ] && grep '"event": "train"' "$1" | tail -1 | grep -o '"update": [0-9]*' | grep -o '[0-9]*$'; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
S3=/workspace/data/refe_sft3/sft.log
S4=/workspace/data/refe_sft4/sft.log
u3=$(lastupd $S3); u4=$(lastupd $S4)
echo "ZZS $(cnt '"event": "eval"' $S3) $(cnt '^ZZEXIT' $S3) ${u3:-0} $(cnt ZZPDM_TRAIN_EXIT /workspace/data/refe_sft2/pdm_train.log)" \
     "$(lines /workspace/data/refe_sft4/chain.log) $(cnt '"event": "eval"' $S4) $(cnt '^ZZEXIT' $S4) ${u4:-0}" \
     "$(awk '$1=="oom_kill"{print $2}' /sys/fs/cgroup/memory.events) $(lines /workspace/data/refe_sft2/lane_pause.log)" \
     "$(pgrep -fc 'python scorer_finetun[e].py') ZZE"
