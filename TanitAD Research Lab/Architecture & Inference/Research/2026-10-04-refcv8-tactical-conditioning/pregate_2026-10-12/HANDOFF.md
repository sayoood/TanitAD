# HANDOFF -- refcv8 A2 pre-gate (2026-10-12)

**No Thor job is running** (all 12 recorded child PIDs dead; every `gate_*.log` ends in `GATE_CHILD_RC=0`; no watchdog kill).
Nothing here needs collecting. What is left, in order of cheapness:

1. **B1 (profile defect, code owner):** `PROFILES['refcv8']` needs `--join-defect-masks` in `model_input_flags`
   (`stack/scripts/launch_gate.py:173`, inherited at `:647`). Re-run the model job after the fix:
   `ssh tanitad-thor 'W=/home/nvidia/pregate_d014ff2; setsid nohup bash $W/code/run_check.sh model G-HYG,G-DVB,G-EVAL > $W/raw/run_model2.out 2>&1 < /dev/null &'`
   (the ctx `work/ctx_model.json` pins the OLD tree; for a fixed tree rebuild it with `code/mk_ctx_pregate.py` after changing `T`/`COMMIT`).
   Expected: G-HYG and G-DVB PASS (they did in diagnostic d1), G-EVAL then walks into B2-B4.
2. **B2 inputs:** a real RECORDED `--eval-remap-overrides` JSON for `--agent-rig-extrinsics --vis1-sidecar --nav-compliance-tau-file --clip-clock-sidecar` (my identity-mapped `work/remap_overrides_A2_DIAG.json` is a diagnostic, not a record) and `--eval-kit /home/nvidia`.
3. **B4 needs an owner before launch:** the trainer model's eval-mode forward is not reproducible under the gate's forked RNG (18 outputs). Reproduce with
   `PREGATE_TRACE_DIGESTS=$W/raw/digest_trace_new.json PREGATE_STAMPS=$W/work/stamps/config_DIAG_PATCHED.json PREGATE_KIT=/home/nvidia PREGATE_REMAP=$W/work/remap_overrides_A2_DIAG.json bash $W/code/run_diag.sh dN --join-defect-masks G-EVAL`
   (about 100 s, CPU, 82 GB free needed >= 40 GB). Open question: is it r8-specific? A refcv7 control through the same path was not run.
4. **GPU-gated (not run):** G-LIVE + G-CKPT smoke (also produces the config.json G-EVAL must rebuild from), G-MAP-OVERFIT / G-BOX-OVERFIT records + closures, the ladder's R8-RECIPE / R8-BUDGET decisions (each changes the argv hash -> re-run every check).
5. **G-SUITE:** produce the Thor full-suite verdict for `d014ff2` (`gate_verdict.py` -- location not found at d014ff2), then `launch_gate.py suite-bind --verdict <verdict.json> --cand-tree <clean tree> --commit d014ff2fd695c306abf769af0572a17e8d713991 --out <bound.json>` (`launch_gate.py:5584`, args `:6652-6657`) and pass `--suite-verdict`.
6. **G-SUITE-PINNED (dev box, stage `devbox`, `launch_gate.py:485`):** needs `--git-dir C:/Users/Admin/tanitad-push/.git` (the gate archives the commit itself and uses a throwaway repo, never tanitad-push's index) -- the real `run` or `check --ctx <ctx with options.git_dir> --checks G-SUITE-PINNED`; imported into the Thor gate dir with `--import-evidence`.

Re-run the cheap in-process rules any time (about 10 s, includes the manifest): `ssh tanitad-thor 'cd /home/nvidia/pregate_d014ff2/work && CUDA_VISIBLE_DEVICES= nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python ../code/step2.py'` (reads `work/argv_PROVISIONAL.json`).
