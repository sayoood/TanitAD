# HANDOFF_B2 (N1DW build, phase B2) 2026-11-19

State: pod tree verified; argv + pairing done; G-SUITE RUNNING detached; mint WAITING (detached); chain staged, NOT started.

Pod paths (all under /root/strat1/n1dw/): tree (134f5da1 = navrbd/tree + 10 files; 3,429 of 3,429 files equal their git blobs via blobcheck.py; 1 extra file eval_grid_windows.json inherited from the base tree), runs/, b2/ (argv/, kclip_*, sampler_p_*, scripts), out/.

1. G-SUITE: `/root/strat1/n1dw/n1dw_gsuite.sh`, base junit/navrbd (0c3b374f, same pod), waves 3,3,2 shards (coordinator: <= ~3 busy procs, pod cap 15.3 cores), taskset 48-95 nice 10, memory guard = growth < 20 GB. Started 15:08Z (an earlier 4-shard wave was killed by PID and restarted).
   Sentinel: last line of /root/strat1/logs/n1dw_gsuite.log = `ZZ N1DW GSUITE DONE ZZ` / `FAIL ZZ <why>`. Result: /root/strat1/n1dw/gsuite_result_n1dw.json (n_regressions 0, new_test_failures [], missing_in_new []). Known base failures (text_encoding, ego_dagger_compose flag_off t2_bank keys, launch_intent flag_off) must appear in BOTH junit sets; compare failure text.
2. Mint waiter (already launched, detached): `/root/strat1/n1dw/mint_n1dw.sh` -> log /root/strat1/logs/n1dw_mint.log. It waits for the suite sentinel, runs binding_preflight_n1dw.py (b2/binding_preflight.json), then takes `flock /root/gpu_queue.lock`, runs the R2 repeat (2x selfcheck-only of N1D_s0 argv), mints the 4 tokens (runs/<arm>/gate/gate_token.json) and writes b2/r2_result.json via r2_compare.py. Sentinel `ZZ N1DW MINT DONE ZZ` / `FAIL`. If it died, relaunch the same line (idempotent: minted arms skipped).
3. Chain: `/root/strat1/n1dw/n1dw_chain.sh` (staged). Launch line, MM sets the priority marker first:
   `nohup setsid bash /root/strat1/n1dw/n1dw_chain.sh < /dev/null > /root/strat1/logs/n1dw.log 2>&1 &`
   Sentinel in /root/strat1/logs/n1dw.log: `ZZ N1DW CHAIN DONE ZZ` / `FAIL ZZ <reason>`. Each arm under flock; dumps (D1-HO follow + zero) detached on CPU after each arm. EVAL-141 dumps NOT in the chain: in NAVRBD they were a dev-box step (navrbd_dumps.py, navrbd_t0.py) on pulled checkpoints.
Next: read suite + mint sentinels, fill the PENDING fields of REPORT_B2.json, then the MM launches the chain.
