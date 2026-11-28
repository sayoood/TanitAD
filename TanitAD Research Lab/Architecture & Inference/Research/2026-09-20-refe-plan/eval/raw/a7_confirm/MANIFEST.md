# SPEC_NAVTEST Amendment 7 confirmation -- inputs and outputs

| artifact | where | sha256 |
|---|---|---|
| a7_confirm.py (this run's script) | repo: eval/a7_confirm.py | `843c50af2699ecd8892fe7037d2b0687b877165765acf62b0fe13d5cab3e6179` |
| a7_confirm_ep015.json | repo: eval/raw/a7_confirm/ | (this manifest is written with it) |
| RESULT_A7_a7confirm_ep015.md | repo: eval/ | (written with it) |
| a5_confirm_tokens.json (inputs; md5 3098d1784a9b99ba9d5a1a0a672a7ed6) | repo: eval/raw/a5_confirm/ | `9b5dc47f295f948c5774852e25cfd127ec6dce1e2a9e48165ec66629591f1788` |
| snap_epoch015.pt (md5 d7c59f4f2fbcbde3e2dec8f67d63a7e7) | data dir only: D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt | `982533ae47bc067b7910bc860517ea8d841737eb4ff0214a78cfe04898637be2` |
| shipped seam (the pipeline's) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_a7confirm_ep015.npz | `208f5ff441d6710a6539470b42dcfc848a2dc13ed611e758dad3b5f54943da01` |
| repaired seam | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_a7confirm_ep015_repaired.npz | `268b0b6850e90e37e418d99d6c29e841222f93b5381b2391470e5f187fb75aa3` |
| path-tangent seam (reported) | data dir only: D:/Projects/TanitAD/data/refe_navtest/seams/refe_a7confirm_ep015_repaired_tangent.npz | `f586f25964af441bcce2f2879885618ff75ad5b35ab4be736d2304428fe3922a` |
| native executed-plan record | data dir only: D:/Projects/TanitAD/data/refe_navtest/proptable/a7confirm_ep015/native_executed.npz | `eed73cb57bee18cbc2898316de792a3750ecb6eff2ea61d3c3bf77b7a85d1d44` |
| pipeline proposal dump | data dir only: D:/Projects/TanitAD/data/refe_navtest/proptable/a7confirm_ep015/proposals.npz | `8b073e6631f2b31a06c052258d2c9acfc893901fb6b0935640bb5d8bf8fb97a5` |
| harness csv, shipped | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_a7confirm_ep015/refe_a7confirm_ep015.csv | `d4e8bd24bf30038003b16cd58ea633c162aa881689890ddc44c423b24d0e56d6` |
| harness csv, repaired | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_a7confirm_ep015_repaired/refe_a7confirm_ep015_repaired.csv | `37831c8d6aea2f94863cd3cf3ee0e434261c088a1dc8ace13ae53f3017298837` |
| harness csv, tangent | data dir only: D:/Projects/TanitAD/data/refe_navtest/score/refe_a7confirm_ep015_repaired_tangent/refe_a7confirm_ep015_repaired_tangent.csv | `5c8dc6016c0a4df7990ff38ff53129af03541f446ac3b6eb6c9a9bbd10580177` |

Harness run directories (W3's devkit output): D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/{refe_a7confirm_ep015, refe_a7confirm_ep015_repaired, refe_a7confirm_ep015_repaired_tangent} (Archive only).
Families blocks: D:/Projects/TanitAD/data/refe_navtest/proptable/a7confirm_ep015\families_shipped.json, D:/Projects/TanitAD/data/refe_navtest/proptable/a7confirm_ep015\families_repaired.json (data dir only).
