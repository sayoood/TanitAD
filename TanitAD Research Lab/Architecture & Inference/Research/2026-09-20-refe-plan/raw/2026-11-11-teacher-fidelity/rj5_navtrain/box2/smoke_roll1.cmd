@echo off
call D:\tanitad\jobs\rj5_env.cmd
echo ZZ roll1 start %TIME%
"%PY%" D:\tanitad\rj5\rj5\rj5_roll.py --bank D:\tanitad\rj5\bank\bank_p7_rank0_slim.jsonl.gz --db-dir D:\tanitad\rj5\db --out-dir D:\tanitad\rj5\out\smoke1 --arms R0,RJ5 --logs-file D:\tanitad\rj5\bank\smoke_logs.txt --tokens-file D:\tanitad\rj5\bank\smoke_tokens.txt --tag p1
echo ZZ roll1 rc=%ERRORLEVEL% %TIME%
