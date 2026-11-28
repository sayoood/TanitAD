# READER_FREEZE (written BEFORE the reader touched any real metric)

UTC: 2026-11-12T15:24:05Z

Before this freeze the real data was touched only by `load_real` (sha256 checks, key coverage, label counts): no IoU, AUC or any S1 metric was computed on real data.

| file | sha256 |
|---|---|
| s1_read.py | 72b4488abd7ceb039463613b5e1a41b8a42b64d78eb44478ee12a80af9222126 |
| test_s1_read.py | c669a763da864f644f446daf9be5ccb35e8b07a4c39d6dc1505c1af284841682 |
| inputs_manifest.json | 70b8a8fad5db77af023567176f5a01ad929ad54b8efd32b32ca6aed347daf154 |
| card PREREG_VDAC_P6.md | 5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4 (equals its .sha256 file) |
| card PREREG_VDAC_P6_A1.md | 8f121bdf9d6f19acec8927f12c50d3f7be2832c70c7f1fe40140d5eb8cc4ed70 (equals its .sha256 file) |
| card PREREG_VDAC_P6_A2.md | c3d563586be6479ea11455f92d7a5295c9e1658e2d7ba1cc77b27e87e5c8b417 (equals its .sha256 file) |
| card PREREG_VDAC_P6_A3.md | a7708573d9db16609bca0faf255ea4c879bda8723ce82e9df72cd27df763d570 (equals its .sha256 file) |
| card PREREG_VDAC_P6_A4.md | 045a10c8f7129c0ac2b168624a982915d50724e4d9b8960a91715fe083254cf5 (equals its .sha256 file) |

Tests: 18 synthetic-fixture tests, all OK (test_results.txt).

## Fix after a crash (a crash is not a peek: it happened in the all-48 sensitivity block AFTER the metrics, before any number was printed or written)

UTC: 2026-11-12T15:26:06Z. Cause: `city_of_log` held only the 45 logs with frames, but the all-48 sensitivity needs the 3 empty logs' cities (KeyError '2021.08.09.17.55.59_veh-28_00960_01031'). Fix: `city_of_log` = all 48 cities from A3 (one line in load_real). No metric, bar, estimator or verdict rule changed. Tests: 18 OK again.

| file | new sha256 |
|---|---|
| s1_read.py | 58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e |
