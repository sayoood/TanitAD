
## Amendment 3 (2026-10-17, before any SFT-4R launch): what the run is, now that the labels exist and the launch gate is binding

> Folded in from `raw/2026-10-16-sft4r-readiness/AMENDMENT_3_DRAFT.md` (drafted 2026-10-16, S4R-1). The code already cited "Amendment 3"
> (`refe/launch_gate.py`, the `refe_sft4r` profile; `refe/test_sft4r_gate.py`; `refe/sft4r_chain.sh`); until now this file ended at Amendment 2
> (`grep -c "Amendment 3"` was 0 on the branch tip `3b8c39f`, MEASURED 2026-10-17), so the registered text did not say what the launch will do.
> This amendment records ONLY facts already fixed in code or in this file. It moves **no lever**. The PI can overrule it. No SFT-4R step has run.
> Earlier sections are unchanged and are not re-written.

1. **No relabel step on the launch pod.** "Order on a new pod" steps 1-3 assumed a pod without labels.
   - The pod that will run SFT-4R already holds the complete PDM labels (train 174,912 rows, held-out 3,137; built by the P2 and S1 chains) and the lane labels.
   - `refe/sft4r_relabel.sh` is therefore NOT run. `refe/sft4r_chain.sh` v2 **audits** the labels read-only and refuses otherwise:
     `pdm_label_complete.py --expect 174912` / `--expect 3137`; lane train >= 2,109; held-out lane and teacher-lane == 3,137.
   - "Changed 1" (COMPLETE labels) is unchanged. It is now also asserted from the trainer's own `data` event by the launch gate:
     `train_sets == 174,912`, `lane_labels.pdm_train == 174,912`, `pdm_train_stale == 0`.
2. **The run length follows from "Changed 1"; it is not a new lever.**
   - One epoch of 174,912 sets at batch 8 x accumulation 8 is `ceil(174,912 / 8 / 8) = 2,733` updates (`scorer_finetune.py:1363-1365`).
   - The "2,335 updates / ~9.3 h" in "Order on a new pod" is the lost run's epoch at 146,185 labels.
   - At the lost run's 14.4 s per update the epoch is ~10.9 h (ESTIMATED; not re-measured).
   - The chain's post-run check requires exactly 2,733 updates and `train_sets == 174,912`.
3. **Binding launch gate.** CLAUDE.md (PI, 2026-09-26): no training run starts without a PASS token from `refe/launch_gate.py`.
   - Profile `refe_sft4r` pins the whole argv token for token, and asserts the effective recipe: lr 1e-4, warm-up 50, batch 8, accumulation 8, 1 epoch, seed 0,
     `--b-mode expreward`, lam 1.0, tau_s 0.1, lane weight 1.0.
   - It forbids `--max-updates`, `--epo*` and `--aux-*`, and requires G-EGR (item 4).
   - The chain verifies the token twice: after minting, and again immediately before the trainer starts.
4. **G-EGR is a required check** ("Changed 3" is unchanged): the gate's own result JSON (`hf_egress.py gate`) is bound into the token.
   No checkpoint is pushed in this arm (`egr_require_ckpt: false`); the scorer states are.
5. **G-SUITE** for this profile runs `test_sft4r_chain.py`, `test_sft4r_gate.py`, `test_suite_nesting.py` and `test_launch_gate.py --static --no-banked`;
   every one must print its PASS token with 0 skips.
   - `check_suite` now refuses to nest: a suite script can never re-run a suite that contains it. Root cause of the 2026-10-16 runaway (~80 python processes):
     `test_sft4r_gate.py` is itself a G-SUITE entry and called `launch_gate.py run`, which reaches G-SUITE. This changes no experiment.
6. **Operational, no lever:** `refe/sft4r_chain.sh` takes the path of the HF token file from the REQUIRED environment variable `SFT4R_HF_TOKEN_FILE`.
   The source carries no default path (the repo is public). Unset or empty, the chain refuses at stage `precheck`, first, before reading anything on the pod.
7. **Unchanged:** base checkpoint (md5 `b54773f8...`), trainable set, arms A and B in one pass, selection rule, held-out read every 400 updates,
   stage-1/stage-2 criteria, the two deliberate regressions (empty PDM -> G9 RED, `REFE_SFT4_MUTATE_G3=1` -> G3 RED), and "Next arm, by outcome".
