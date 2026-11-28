# Bank record: the v8f-recover fine-tune checkpoints on private HF (2026-11-19)

* Repo: private HF dataset `Sayood/tanitad-v8f-recover-runs`.
* Contents: 23 objects, all equal by sha256 to the local copies (`BANK_V8F_RECOVER.json`, `not_equal` empty).
  * 22 checkpoints: `M2b_s{0,1}_recft_<arm>/ckpt_last.pt` for C0, C0rep, CL, CLR, LLR, R, RAPR, RAPRP01, RP01, RPT and RTL.
  * One tar of the small files (`v8f_recover_small.tar`, sha256 cd5a97e5).
* Source: `E:/Projects/TanitAD-artifacts/v8f-recover`. That copy is unchanged; only the local staging tar was deleted (`DELETIONS.log`).
* Why: NATIVE40 (`2026-11-19-native40/PREREG_NATIVE40.md`) reads LLR_s0 and LLR_s1 from these checkpoints. Before this bank, they existed on one disk only.
* Script: `bank_recover.py`. The token comes from the environment, never from argv.
