# HANDOFF_B4 (2026-11-18)

State: pod tree, inputs, chain, argv files, dump scripts are staged and verified. The G-SUITE of 0c3b374f runs detached on the pod (CPU, 2 waves of 4 shards, about 23 min as for the NAVRB2 fix2 suite; started 18:50:43Z).

Sentinels (pod):
- G-SUITE: last line of `/root/strat1/logs/navrbd_gsuite.log` is `ZZ NAVRBD GSUITE DONE ZZ` or `ZZ NAVRBD GSUITE FAIL ZZ <why>`. Result: `/root/strat1/gsuite_result_navrbd.json`; compare log `/root/strat1/logs/suite_compare_navrbd.log`.
- A watcher (`/root/strat1/navrbd/b4_finish.sh`, CPU) then runs `binding_preflight.py` (the CPU half of mint: data sha256, suite binding, nav_rb(d) report, tree manifest, pairing, G-HYG) and writes `/root/strat1/navrbd/b4/finish.log` (`ZZ B4 FINISH DONE ZZ` or `FAIL`), `binding_preflight.json`, `gsuite_result_navrbd.json`.

Exact next steps for the next agent:
1. Read the G-SUITE result: `n_regressions` must be 0, `new_test_failures` empty, `missing_in_new` empty. The two known base failures (test_text_encoding_is_explicit, test_v8f_ego_dagger_compose flag-off test) appear in both base and new junit.
2. Read `binding_preflight.json`: every `binding_problems` and `pair_problems` list empty.
3. Fill `REPORT_B4.json` fields marked PENDING (G-SUITE counts, preflight).
4. The MM decides the launch: `nohup setsid flock /root/gpu_queue.lock bash /root/strat1/navrbd/navrbd_chain.sh < /dev/null > /root/strat1/logs/navrbd.log 2>&1 &`, then `dump_rbd_all.sh` (CPU).

Base for the suite comparison is `junit/navrb2_fix2` (the e71782a3 run on the same pod, 2026-11-18 06:29Z), not a re-run.
