# MANIFEST: P7 navtest read (2026-11-16)

Every file is a byte-identical copy (sha256 re-checked against the source at copy time). Skipped on purpose: `*.npz`, `fam_P6.json`/`fam_P7.json` (49.7 MB each), `rows_s*.jsonl` (28-29 MB each), the harness `*_hooks.json` (12 MB), `*.calls.jsonl`, the harness `.log` (3 MB). They stay on the dev box / HF.

| file (under `PK/raw/2026-11-16-p7-read/`) | source | sha256 | bytes | IPv4-like | hf_ |
|---|---|---|---|---|---|
| `chain/CHAIN.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/CHAIN.log` | `a6799e336a5c4a7982d41e95087ab59158ecdc00b3667f5bd667f6280eca7cfe` | 8365 | 0 | 0 |
| `chain/chain_stdout.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/chain_stdout.log` | `a6799e336a5c4a7982d41e95087ab59158ecdc00b3667f5bd667f6280eca7cfe` | 8365 | 0 | 0 |
| `chain/command_adherence_p7_full.json` | `E:/Projects/TanitAD-artifacts/p7_full_read/command_adherence_p7_full.json` | `553aa47d33a24174aaf1741c298a9952a33434ff4ecee3ff68b1b7b53dcbd7b2` | 21660 | 0 | 0 |
| `chain/fam_ci_P6_vs_P7.json` | `E:/Projects/TanitAD-artifacts/p7_full_read/fam_ci_P6_vs_P7.json` | `3d19f0342d965f4c5cc0ae89c3ab44f48e095b5614f3cf6e48e67f9d22f59612` | 120531 | 0 | 0 |
| `chain/logs/adherence.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/adherence.log` | `d45f815102ac934fdcf067b8d8cc8d8b6b64add74de96211e08eddb9ec0fbbd6` | 152 | 0 | 0 |
| `chain/logs/fam_arm.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/fam_arm.log` | `86216324e5e8d056ca737af69ad19fd4d628973c31bfa33b6bdba1b728f92f4b` | 408 | 0 | 0 |
| `chain/logs/fam_base.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/fam_base.log` | `86216324e5e8d056ca737af69ad19fd4d628973c31bfa33b6bdba1b728f92f4b` | 408 | 0 | 0 |
| `chain/logs/fam_ci.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/fam_ci.log` | `3e9a9ebc521ce7aedb20622e2dd48f8641c9398e2f70a6f858043f07cc6f9a68` | 71 | 0 | 0 |
| `chain/logs/rep.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/rep.log` | `99011a4125c02f09110c3ca13deff90c90eaf3d28935bf402a2c7c9830b852c6` | 75 | 0 | 0 |
| `chain/logs/score.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/score.log` | `06a413dea5f3ff096338576bf5ddfb796452f0e37a9b28ac26524686719d6bfd` | 67 | 0 | 0 |
| `chain/logs/seam1.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/seam1.log` | `25bd769715e24be0a2316d733007de967a0cd134d6aed5455d733271a2172eed` | 71 | 0 | 0 |
| `chain/logs/stage3.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/stage3.log` | `dd6fb1926aa7b8a25fc161ba1a394cc066ad9be55f9d92ea4861c27209e615f8` | 85 | 0 | 0 |
| `chain/logs/transp_s0.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/transp_s0.log` | `ecb6a2e20c7a75b8dc37c4db17f66d40927bf553c13ee25e87628e13eb53047b` | 37 | 0 | 0 |
| `chain/logs/transp_s1.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/logs/transp_s1.log` | `8b4d28a7ec542332252df2b880672a6dea85bbe181166912aea737c12531985c` | 37 | 0 | 0 |
| `chain/markers/adherence.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/adherence.DONE` | `6851674c5d6d834dfe4af7147d6b6fb4f4381dd2fcf9b4d2ac164be01e517ec6` | 133 | 0 | 0 |
| `chain/markers/chain.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/chain.DONE` | `74811751c8a977efcaf04c70a58c4fa64951c6c28687e08b4c95bdfa499465e5` | 57 | 0 | 0 |
| `chain/markers/families.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/families.DONE` | `93b85caf2fed0b35946a8819615a694699fb77b70078db36da093bc81aeeb5ab` | 389 | 0 | 0 |
| `chain/markers/ident.json` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/ident.json` | `2ca5fdf4bd78201ad9e036dd49713f32654f7ba315a28cd3266d3a98a890268a` | 227 | 0 | 0 |
| `chain/markers/rep.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/rep.DONE` | `de53b31835b1b682314de8d58f29e599d30f89d6ccdfe71d8a4a14c841425713` | 116 | 0 | 0 |
| `chain/markers/score.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/score.DONE` | `9000dbb655db5e1938e75cd34d65581a08bb91f46bec405cbbc8b05784f2134b` | 377 | 0 | 0 |
| `chain/markers/seam1.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/seam1.DONE` | `dd14a964bb09894965d51737bae5d38bfd60c3e813efb0346c7371446fdda86c` | 273 | 0 | 0 |
| `chain/markers/seam1.FAILED.attempt_1` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/seam1.FAILED.attempt_1` | `7463b459670e5304386cb41cc69d91f009a4bb21e1d2a86ad9647f6825cb8e25` | 843 | 0 | 0 |
| `chain/markers/seam1.FAILED.attempt_2` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/seam1.FAILED.attempt_2` | `7463b459670e5304386cb41cc69d91f009a4bb21e1d2a86ad9647f6825cb8e25` | 843 | 0 | 0 |
| `chain/markers/stage3.DONE` | `E:/Projects/TanitAD-artifacts/p7_full_read/markers/stage3.DONE` | `534f836e15a573a987433ae850a679b42529a6f2f445821ecae9b3b083383c78` | 109 | 0 | 0 |
| `chain/families_substeps_done.json` | `E:/Projects/TanitAD-artifacts/p7_full_read/families/substeps_done.json` | `f4c694831aacbdc862328414ec4ae7053de13d79e79fe123b96bc8d6fecdb673` | 244 | 0 | 0 |
| `stage3/r2_result.json` | `E:/Projects/TanitAD-artifacts/p7_full_read/stage3/r2_result.json` | `831b5c2b97c650f4bb82381e61c20b33fb08ec07e40db5659a87a9fc0d382aa3` | 13968 | 0 | 0 |
| `stage3/p7_full_families.log` | `E:/Projects/TanitAD-artifacts/p7_full_read/stage3/p7_full_families.log` | `5e29cabc756e2e4a2b10fa9487b0f56d5b4d32a036fd95e6b87e8f446bb8432e` | 1140 | 0 | 0 |
| `stage2_p7_full/result.json` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/result.json` | `94a4ce43bd0a9ad71df38d346627d45c18d3bc276d838f5a4b72f2f48a7a8db6` | 2795 | 0 | 0 |
| `stage2_p7_full/score_status.json` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/score_status.json` | `b23a1b378696844f5ea6c3a0fce7ecd4b637ec1e5e0e0e73d6177eb66e265c49` | 353 | 0 | 0 |
| `stage2_p7_full/score_1.log` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/score_1.log` | `1fc8769edf935307a68276103be1018286af113ca96132afdd9bb35ac9fb8039` | 309 | 0 | 0 |
| `stage2_p7_full/seam.log` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/seam.log` | `16db4162206882e0bc930f789a168124a04ff675b3f6686a05b724988ccc7fb2` | 2971 | 0 | 0 |
| `stage2_p7_full/p7_full_families.log` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/p7_full_families.log` | `5e29cabc756e2e4a2b10fa9487b0f56d5b4d32a036fd95e6b87e8f446bb8432e` | 1140 | 0 | 0 |
| `stage2_p7_full/deployed_families.log` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full/deployed_families.log` | `5e29cabc756e2e4a2b10fa9487b0f56d5b4d32a036fd95e6b87e8f446bb8432e` | 1140 | 0 | 0 |
| `stage2_p7_full_rep/result.json` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full_rep/result.json` | `39182f4b814c6b2ec0b12ed5a23fdd47b7ca56b033774d99ebe2f882d1882420` | 1429 | 0 | 0 |
| `stage2_p7_full_rep/seam.log` | `F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2/p7_full_rep/seam.log` | `3a9ee29c4be3efc5ce3d883f1a66daca7b74211d4795d3d57b66994b9d2d5a7c` | 2871 | 0 | 0 |
| `seams/stage2_p7_full.report.json` | `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.report.json` | `a577b92b16ddec1eb76831c93fbfc9f3c9d59300d35b82edbe189a703a5633aa` | 1602 | 0 | 0 |
| `seams/stage2_p7_full_rep.report.json` | `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full_rep.report.json` | `d18093974ad1586937361e536e76f951591bc9cd34ffb7415deb53df194a00c3` | 1501 | 0 | 0 |
| `score/refe_stage2_p7_full.csv` | `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.csv` | `b209a6e35dc634873cf03520d844f1e73649fed8fa212a1e9903db9da901636f` | 901345 | 0 | 0 |
| `score/refe_stage2_p7_full.counts.json` | `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.counts.json` | `f1cc9ed98a351ead8cb7f3fad2fadfc4372fc6d2fb757320f23768b7e1e81a0e` | 2589 | 0 | 0 |

Totals: 39 files; IPv4-looking strings 0; `hf_` tokens 0. Nothing masked.
