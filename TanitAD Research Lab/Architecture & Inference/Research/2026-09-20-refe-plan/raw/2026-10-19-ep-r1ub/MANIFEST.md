# MANIFEST: EP-R1ub (agent EP-R1UB, 2026-10-19)

Repo target directory: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-ep-r1ub/`.
Staging root (the bytes): `C:/Users/Admin/par18/epr1ub/out/` (paths.json rows `[repo path, abs source, "raw"]`).
Nothing is committed or pushed. Each row: file, bytes, sha256.

| file (under the repo target directory) | bytes | sha256 |
|---|---|---|
| `DRAFT_PREREG_NCDAC_RESOLUTION.md` | 6146 | `fe83adb4938b05eae372feeb630d082e5b668edb864e1f8f788a4f283d44b7d3` |
| `DRAFT_PREREG_NCDAC_RESOLUTION.sha256` | 99 | `2dab305f3eb4cf484dc0fb3007d75fbf2f4a7e1bde912b2d98a4405dbca69295` |
| `EPR1_ABANDON_PROPOSED.json` | 1431 | `8b6146212bf106d52625a6ad84a433fe9c07223c577f7fb40450dcfc1b35f52e` |
| `PREREG_EPR1UB.md` | 17738 | `214f2d379dc7cded632817759e50b4a770ad26111081338abdad130409d5ed00` |
| `PREREG_EPR1UB.sha256` | 83 | `3f27000d885d7b89e33b5bd2bca98085d611bcd73ed766e83ae9e4602e7c6524` |
| `PREREG_EPR1UB_AMENDMENT1.md` | 2928 | `4e23b64dac8faba225b11b24e6488354bacc1a55e29f4043d334b039c42a235a` |
| `PREREG_EPR1UB_AMENDMENT1.sha256` | 94 | `82e8ff335feeb0192f1347ac5d1a7b5ca15ea0a68a0dc5fed0331b4948161877` |
| `RESULT.md` | 13221 | `796ad7543cccd52cc66486b7191014fafbd6119c67b9eea735ffea1a95064ed0` |
| `code/CODE_SHA256_EPR1UB.json` | 1463 | `e48d3253aa2ac27cdc235ec45145c5fe97fdede0fc3bf95e15e3b39c9176ea19` |
| `code/epr1ub.py` | 45523 | `92355a316a6ef966f6d22f7f60b97f2875887ae9efda250809e54a57ac6f305e` |
| `code/epr1ub_gate.py` | 55362 | `5a91545437744d775ec1b54eb22f3ece2d17d572a1cdf4cb70c8e700cfb698b3` |
| `code/launch_epr1ub.ps1` | 5414 | `b39a313543b18650926ab07acd877959f5af1d4cbcd6a6ee84926143f44b0c46` |
| `code/null_control.py` | 3440 | `339acd3bf5ffd3ffa06cf6428efb254e1d932e0c0549c225d038d233618243a0` |
| `code/test_epr1ub.py` | 37385 | `a11278f58feb0c42c72a036a6974531549f94267856777187b90b50ef755eb45` |
| `code/test_epr1ub_gate.py` | 18054 | `0fd5745ef0a711325f7dcdddfe94248d0e521526125d1ebb03fc16dd2d68d2a1` |
| `raw/f10/epr1ub.log` | 1482 | `5f034c5fd0bcb5225185e48cacfb90ca6a32f5279a67376e8ccf0f8d72c99b1d` |
| `raw/f10/epr1ub_gate_token.json` | 9819 | `310b564e9f9f0ff4b72ce4dfc15b88b06b999bd52d2933b26fa626828fba6446` |
| `raw/f10/epr1ub_result.json` | 33910 | `10562ac7defe299b5bd051648a024b81c0334b503276a51f5e2beb120f49cafa` |
| `raw/f10/epr1ub_result.sha256` | 85 | `31623c4489c03428c5fd1bd91f5d254888df38db5d17ea8130da4975fc10d2ed` |
| `raw/f10/mint.log` | 244 | `8f5fcf872543402c32be0e20b551c559193a45a5154de359d36629d596a22e12` |
| `raw/f10_null/null_control.json` | 5975 | `383ed3b0ca7500b2bc4f3f9afa13b3c882b8d030e78b512876c3a29e1b9abb43` |
| `raw/test_epr1ub.log` | 16295 | `e680036605e66241f7378778dbbcc4a8b38b6b87fd4c4ef37fa54de5e4503eee` |
| `raw/test_epr1ub_gate.log` | 2081 | `1ac5c1f0b69e80b381304b6bd144075b571933c6b928c1636c56ec90e855e30b` |

## Where everything lives

* `code/`: code, STAGED not landed (mirrors the EP-R1 layout raw/2026-10-18-ep-r1/code/); hashes also in code/CODE_SHA256_EPR1UB.json
* `raw/f10/`: the run (disk copy: C:/Users/Admin/epr1ub/run/f10 and gate/f10_run)
* `raw/f10_null/`: the post-hoc noise-row control (disk copy: C:/Users/Admin/epr1ub/run/f10_null)
* PREREG_EPR1UB.md (sha256 `214f2d379dc7cded632817759e50b4a770ad26111081338abdad130409d5ed00`) and PREREG_EPR1UB_AMENDMENT1.md (sha256 `4e23b64dac8faba225b11b24e6488354bacc1a55e29f4043d334b039c42a235a`): staged here.
* `EPR1_ABANDON_PROPOSED.json`: staged; the Master Mind lands it as the abandonment file `refe_head_confirm.read_abandon` reads (it takes effect only when landed).
* `DRAFT_PREREG_NCDAC_RESOLUTION.md`: DRAFT, NOT run, needs registration.
* Disk only, outside the repo (not staged because they are large or private to the run): the fitted rows are not written (no feasible c*), the bank at `C:/Users/Admin/srttc_s0/work/bank` (S0, not mine), helper scripts `C:/Users/Admin/par18/epr1ub/*.py`.
* Landed parents used (frozen, byte-exact from tip `1afc54f6fa5e`): see `code/CODE_SHA256_EPR1UB.json`.

## Integration asks (escalated, not buried)

1. Land `code/` (5 new files + `null_control.py` if wanted) so the gate token's `landed: false` bindings become blobs; the blobs must equal the hashes in `code/CODE_SHA256_EPR1UB.json`.
2. Land `EPR1_ABANDON_PROPOSED.json` as the EP-R1 abandonment file (R7 and R9) and open the shared look for SR-TTC alone if you accept N.
3. Register and run `DRAFT_PREREG_NCDAC_RESOLUTION.md` Part A (CPU, seconds).
4. Add the GAC row `D-REFE-EPR1UB-1`, update `H-REFE-RL1-NCRES` and `H-EPR1`, and add the BACKLOG items (text in RESULT.md section 6).
