# Brief: adapt the T2-A kit to v8F-mini, steps E1 and E2 (the policy and the in-loop encoder), with their reproduction gates

PI approval 2026-11-06 ("Approve T2 path"). This agent does CPU engineering only.

Read first and follow: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (preamble).

**Repo and files**
* Never commit or push. Never delete files: move them aside. LF line endings.
* Read the repo via `git -C C:/Users/Admin/tanitad-push show 24b97ba9:<path>`, or export only the dirs you need to `F:/TanitAD-scratch/t2v8f/tree/`.
* Do not touch other sessions' processes or files.
* Do not edit `stack/ops/launch_gate_thor_env_failures_mapft.json` or `stack/scripts/launch_gate.py`.

**Compute and network**
* No downloads, no network, no ssh to box 2, no GPU (`CUDA_VISIBLE_DEVICES=-1`).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. PYTHONPATH joins with `;` and must not resolve `tanitad` from G:
  (memory `windows-pythonpath-separator.md`): assert `tanitad.__file__`.

**Context budget:** ≤ 150k tokens; read by grep anchors. Hand off at about 130k.

**Banned words in what you write:** void, superseded, retracted, formerly, "no longer", correction, withdrawn, "closed loop". Say
"T2 re-perception rollout"; existing file names may be quoted.

## What to read
* **The sizing:** `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-06-t2-sizing-v8f/T2_SIZING_V8F.md`,
  sections 2 and 5: the input gap table and steps E1 to E7.
* **The kit:** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-prep/` (`code/t2a_policies.py`, `t2a_run.py`,
  `stub_gsplat_renderer.py`, `tests/test_t2a_dry.py`) and `.../2026-10-17-t2a-closedloop-box2/` (the banked run, `closedloop_drive.py`).
* **v8F-mini:**
  * `stack/tanitad/models/v8f_mini.py`: `forward_infer` and `INFERENCE_INPUT_KEYS`.
  * `stack/tanitad/data/v8f_mini_data.py`.
  * The T1-kin tool `stack/scripts/v8f_t1_eval.py`: `WindowState`, `V8FMiniWindows.item`, the causal bucket, the HACC post-map.
  * The `reindexed` tool the floor reads used: `E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_build/tool/v8f_t1_eval_ri_lf6.py`,
    the floor flags off.
  * `DESIGN_T1_V8F.md`.
* **The encoder route that built the training cache:** `pull_encode_pooled.py` (grep the tip). Find the exact canon: cylindrical
  projection, 256x640, DINOv3-L/16, bf16, pooled 4x10, fp8 storage. Also find which route actually built box 2's `cache_full`; the sizing
  marks this UNVERIFIED.
* **Checkpoints:** `E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt` (C0). The DAG-OACC base is the init
  named in `v8f-recover/M2b_s0_recft_C0/argv.json`.

## Build (into `E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/`)
1. **E1 `code/t2a_v8f_policy.py`, a `V8FMiniT2Policy` with the kit's policy interface.**
   * It loads a v8F checkpoint and keeps a 4-frame feature deque on the 0.2 s grid.
   * It builds the model's inference items exactly as the T1-kin tool does: measured ego history including yaw rate from the
     executed track, nav, and the causal speed bucket.
   * It runs `forward_infer` and returns the 6 s plan (30 knots).
   * Optional, as in the tool: the HACC post-map, and the P10 arm from the same forward's prior.
   * ⛔ Inference reads only `INFERENCE_INPUT_KEYS`. Reuse the key guard.
2. **E2: an in-loop encoder in the same package.** It reproduces the training cache's canon from a rendered or logged front-wide frame:
   reprojection, resize, DINOv3-L, pool, and the fp8 round trip if the cache stores fp8. Weights come from a local offline snapshot only;
   find one on this disk (the HF cache, `E:`, `F:`). If none exists, build against a stub and name the blocker. ⛔ No G: path.
3. **Gates, as tests under `tests/`, CPU, each with a deliberate-regression arm that must go RED.**
   * **G1, reproduction.** For one eval-141 window, the policy fed the BANKED cached features gives the T1-kin tool's plan at tick 0
     bit for bit (or to the tool's own tolerance, stated).
   * **G2, encoder fidelity.** The in-loop encoder on the LOGGED frame of a cached window reproduces the banked tokens. Report rel-L2
     and cosine against the cache, and compare with the 2.5–2.8 % residue in the sizing. If DINOv3-L is too slow on CPU, use one frame
     and say so.
   * **G3, zero control.** A policy wrapper that returns the P10 prior gives exactly the prior.
   * **G4, key guard.** A peeking variant is caught.
   * Mutation proof for each gate.
4. Run the tests and write `PINS.txt` (sha256 of each code file and checkpoint read).

## End
Write `t2v8f/HANDOFF_E1E2.md` (≤ 2k tokens): what is built, gate results with numbers, open items for E3 (playback executor), E4 (kit
on the tip tree), E5 (scorer) and E6 (the remaining reachability gates), and the blockers.

Hand back in ≤ 300 words: the gate results, blockers, and a deliverable manifest.
