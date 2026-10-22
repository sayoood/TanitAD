#!/bin/bash
# PERC2PLAN-R8 -- the QUEUED launch (built + smoked 2026-10-21 19:54 Berlin; not launched: Thor busy with fanceil s0/DEC s1, then perc8).
# Run ONLY when neither /home/nvidia/r8ns/fanceil/ nor /home/nvidia/r8ns/perc8/ has any process (the chain refuses otherwise: ZZCBUSY).
# ~12 os-only rolls x ~23 min (fan-roll rate on Thor) = ~4.6-4.8 h; resumable (re-run skips rolls whose meta says C4 pass + rc 0).
ssh -n tanitad-thor 'cd /home/nvidia/r8ns/perc2plan && nohup setsid bash code/chain_p2p_thor.sh > out/chain.out 2>&1 < /dev/null & disown; sleep 20; cat /home/nvidia/r8ns/perc2plan/out/chain.log'
# progress:  ssh -n tanitad-thor 'cat /home/nvidia/r8ns/perc2plan/out/chain.log; tail -2 /home/nvidia/r8ns/perc2plan/out/*/p2p.log'
# after ZZCEND in chain.log:  bash /c/Users/Admin/refcv8_pf/perc2plan/code/pull_and_analyze.sh   (writes PERC2PLAN.json + RESULT.md)
