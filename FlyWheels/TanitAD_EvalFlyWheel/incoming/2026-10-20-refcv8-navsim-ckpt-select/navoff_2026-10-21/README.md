# R8-NAVFEED-DIAG result (registered 64d0150 before scoring) -- FEED IS NOT THE CAUSE

refcv8 EMA-48,852 with the NavSim command withheld (bridge arm R8_NAVOFF, nav_cmd=None), seed 0, on the registered navtrain
sample (2,089 tokens / 30 logs), Thor CUDA roll + Thor CPU score with the selection run's driver.

| | PDMS |
|---|---|
| R8_NAVOFF | 64.14 |
| R8_LEGAL | 63.39 |
| PRIOR | 73.84 |

NAVOFF - LEGAL +0.75 [-1.60, +3.04] (log-cluster bootstrap B 2000) -> **FEED IS NOT THE CAUSE**; straight +4.24 [+2.75, +6.32],
turn -7.05 [-10.62, -2.97]. Full record: `NAVOFF_READ.json`. Code: `code/r8ns_navoff_LAUNCH.sh` (Thor), `code/navoff_waiter.sh`
(dev box reader). POST-HOC diagnostic, never a bar; one training seed, one inference seed; NavSim v1 open loop.
