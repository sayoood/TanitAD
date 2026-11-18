# MM note: NAVRB2 pod G-SUITE DONE at dfea5c9d, 0 regressions (2026-11-18 ~00:10Z pod clock)

Evidence class MEASURED: `gsuite_result_navrb2.json`, written by `strat1_check.py suite`, plus the run log.

| run | tree / base | regressions | cause | fix |
|---|---|---|---|---|
| 1 | new8 + 14 files / junit/new8 | 43 | new8 is a later weather-head tree, not 4dedac45: its `v8f_mini_eval.py` reads `cfg.ctx_pool`, which the 4dedac45 model lacks | rebuild on **new9**, which matches 4dedac45 blob for blob (3,407 blobs, 0 mismatched, 0 missing, 0 extra), with base junit/new9 |
| 2 | new9 + 14 files (f95de67a) / junit/new9 | 7 | all "off is the tip bit for bit" (lead / level / otband): `built_levers()` always carried `nav_rb: off` | the key is emitted only when the channel is built; the checker reads `.get("nav_rb", "off")`. Commit **dfea5c9d** (2 files; dev box 240 passed on the 8 affected files) |
| 3 | new9 + 14 files (dfea5c9d) / junit/new9 | **0** | - | - |

* **Run 3 counts.** Base: 14,929 tests, 12,603 pass, 435 fail, 205 error. New: 14,993 tests, 12,666 pass, 435 fail, 205 error. So 64 tests exist only in the new tree: 63 pass and 1 skips. The failure and error counts equal the base's (the dev-box and pod environment failures).
* **What this record shows.** The 14-file NAVRB2 change (`../navrb2.patch` against 4dedac45, regenerated) breaks no test that passes on 4dedac45.
* **Pod state.**
  * `/root/strat1/navrb2/tree` = 4dedac45 + the 14 files of dfea5c9d, 14 of 14 sha256 (`pod/navrb2_tree.sha256`).
  * `COMMIT=dfea5c9d` in `pod/navrb2_gsuite.sh` and `pod/navrb2_chain.sh`.
  * `queue_navrb2.sh` waits for DSCALE block 3's end line, then takes the GPU lock.
  * The chain refuses without this DONE line.
* **Banked record.** The note `../commit/MM_NOTE_NAVRB2_COMMIT.md` names new8 as the base tree. The tree the arms train on is new9 (4dedac45) + dfea5c9d, as this record shows.
