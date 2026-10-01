# like-for-like 016 -- inputs and outputs

| artifact | where | sha256 |
|---|---|---|
| like_for_like_016.py (this run's script) | repo: eval/like_for_like_016.py | `9732caa9537f028bdead9af3c71a2b5045e9e5e807458d60eb074e27b2745043` |
| like_for_like_016.json, build_record.json, selftest.json | repo: eval/raw/like_for_like_016/ | (written with this manifest) |
| RESULT_like_for_like_016.md | repo: eval/ | (written with it) |
| repaired ep015 seam (md5 065b967c587df31f7cdc26a96c702981) | data dir: D:/Projects/TanitAD/data/refe_navtest/seams/refe_lfl016_ep015_repaired.npz; repo copy: eval/raw/like_for_like_016/refe_lfl016_ep015_repaired.npz | `107b035b6fed10b7d52709ccd4401e224af124eee9231ca67b40f85bbba6a88f` |
| rebuilt unrepaired ep015 seam (md5 a3d2a0e57f0784c42d3552c83f94e5b2) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_lfl016_ep015_unrepaired.npz (poses bit-identical to the shipped ep015 seam) | `7e0e33c6f1a53b0fe72fea0cae6f739c986d8d07ebbcdd16560bc9cd26da4b3e` |
| shipped ep015 seam (the pipeline's) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep015.npz | `5ed8656adfb479d1b2c7538a032214ad6cb1e9eb3a47248a6ed83f71acbda854` |
| shipped ep016 seam (the pipeline's) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep016.npz | `475027dcca649c742f83e00132b4cbcb8532075bae8d0cd04a5c2099d27eb469` |
| ep015 native dump (stop_candidate_probe.py) | data dir only: D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz | `c1667b481d2e228a3883fb84e9443e7215235cfa6fb48751265a1b2fadc813e0` |
| A1_sub200_tokens.json (md5 114deb5bf6d2631a9cdd6e8c0724273d) | repo: D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json | `52a59595136387051e39cebced530e484277536e147be9caf383989e9d192032` |
| harness csv, sub200_ep015 | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep015/refe_sub200_ep015.csv | `b7dc808223cefa17133a40564ea03e5087fbac906392ab53b731526fa99042ac` |
| harness csv, sub200_ep016 | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep016/refe_sub200_ep016.csv | `2464e8834c552002d489280b3273836c653ebc81a8f3f65acc10e0fcca47b969` |
| harness csv, unrepaired | data dir: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_unrepaired/refe_lfl016_ep015_unrepaired.csv; repo copy: eval/raw/like_for_like_016/harness/refe_lfl016_ep015_unrepaired/ | `b7dc808223cefa17133a40564ea03e5087fbac906392ab53b731526fa99042ac` |
| harness csv, repaired | data dir: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_repaired/refe_lfl016_ep015_repaired.csv; repo copy: eval/raw/like_for_like_016/harness/refe_lfl016_ep015_repaired/ | `a0bbd26f2cc4cb5050026c412eaf426ec38e6196adbd6baa565b21e9d1162869` |
| harness refe_lfl016_ep015_unrepaired.calls.jsonl | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_unrepaired/refe_lfl016_ep015_unrepaired.calls.jsonl | `1fe0c4989ab38342dd68e81a3d508580ce7d1c2c7b8b6a1b7aeca922c0a76f3a` |
| harness refe_lfl016_ep015_unrepaired_hooks.json | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_unrepaired/refe_lfl016_ep015_unrepaired_hooks.json | `92e6c93c233343a2f10913e852e70d808918b80ae3f8d6d7046f5f1353936000` |
| harness refe_lfl016_ep015_unrepaired_manifest.json | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_unrepaired/refe_lfl016_ep015_unrepaired_manifest.json | `1d9ed22d906a243d971327d9a09ab5e76f73aea83f508885ebf216daed0b34be` |
| harness refe_lfl016_ep015_unrepaired_rows.jsonl | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_unrepaired/refe_lfl016_ep015_unrepaired_rows.jsonl | `1b69225e7826283777282a57b5e50b0a7d13f8ce38ed3f9461d9794c4bbb09d3` |
| harness refe_lfl016_ep015_repaired.calls.jsonl | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_repaired/refe_lfl016_ep015_repaired.calls.jsonl | `33aa9932ba85d0101078a5a3eaf1b614c0117be6fda6adc7ca9f6a4942bbe4be` |
| harness refe_lfl016_ep015_repaired_hooks.json | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_repaired/refe_lfl016_ep015_repaired_hooks.json | `99f0b484c8568d3de660e47b62548bf221a00aba637f799b842cfdd5ffa2afa8` |
| harness refe_lfl016_ep015_repaired_manifest.json | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_repaired/refe_lfl016_ep015_repaired_manifest.json | `5598f92ed6747e4a218b6482e91175111370ba3135255163f8c6ed53068a4848` |
| harness refe_lfl016_ep015_repaired_rows.jsonl | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_lfl016_ep015_repaired/refe_lfl016_ep015_repaired_rows.jsonl | `bb3986de40f1a5cd46f6ca80b3794d3f21895fbcf3aeb2d9512a7cef0c23b6d7` |

Harness stdout logs: eval/raw/like_for_like_016/logs/. W3's devkit run directories: D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/{refe_lfl016_ep015_unrepaired, refe_lfl016_ep015_repaired} (Archive only).
Checkpoints (read for md5 only): D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt md5 d7c59f4f2fbcbde3e2dec8f67d63a7e7; D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch016.pt md5 eacd8e8fa76ce55eead1779a08876c03 (data dir only).

## Extension: all 64 proposals

| artifact | where | sha256 |
|---|---|---|
| like_for_like_016_64.json, build64_record.json, oracle_trend_ep015rep_ep016.json, logs/oracle_trend.log | repo: eval/raw/like_for_like_016/ | (written with this manifest) |
| compact scores of the repaired table (token, pdms, sub, valid, pick) | repo: eval/raw/like_for_like_016/table_sub200_ep015_repaired_scores.npz | `e6924aeccd52dbf80f244ef9fb7905efacfaed802fb5042ea2fa59fcb0430d03` |
| the repaired E-6 table oracle_trend.py reads (+ gates.json) | data dir only: D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015_repaired/table.npz | `b9c7b589ec98113c3e995ad39723038a73527810df703c775f881f4718932cd7` |
| the pipeline's ep015 / ep016 E-6 tables | data dir only: D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/table.npz / D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep016/table.npz | `732969ea14bf48f248f67d40509e7d039c11ddd74cba8ba5d2d1ded45d0bd87a` / `5c2cd22944fc60628b36ea762a78327ea4e90db3a861fe4929746bb75b4bc002` |
| 64 repaired proposal seams (md5 each in like_for_like_016_64.json) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep015_repaired/ | -- |
| 64 harness outputs (csv sha256 each in like_for_like_016_64.json) | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep015_repaired_p00 .. refe_sub200_ep015_repaired_p63/; stdout logs D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015_repaired/logs/ | -- |
