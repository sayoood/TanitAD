# BRIEF: build K-OWN-SPLIT for the N1DM read (MM, 2026-11-21 ~20:55Z)

Read first: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (the preamble binds you). Stage nothing in git: the MM lands. End with a deliverable manifest.

## Goal (one deliverable)
Build the read-side diagnostic **K-OWN-SPLIT** of `F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DM.md`, section 2.3 (iii). It has **no bar and decides no outcome**. The card's words: "the route head's accuracy on own-code vs GT-code rows of a probe batch; a large gap names the shortcut. ... if it fires, the next arm masks the decoder loss through the branches to own-code rows."

Why it exists: in N1DM the 3 branches train on ALL rows. On teacher-forced rows (prob p_gt 0.2) O's plan feature `h` holds the GT tactical code (future-derived), and the soft route posterior `p = softmax(route_head(h)).detach()` is read from that `h`. If the head is much more accurate on GT-code rows, the branches see a cleaner route signal in training than they ever get at inference.

## Recommended design (verify every source fact; quote file:line)
* Source: `F:/TanitAD-scratch/n1dc/read_n1dm/tree/stack/tanitad/models/v8f_mini.py`. `_forward(b, p_gt, use_labels, gen=None, code_override=None)` is at :1156. The own/GT switch is at :1189-1197: `own` is all True; with `use_labels and p_gt > 0` under `code_mix == "whole"`, `use_t = (rand < p_gt) & lab_t["has"]` and `own = ~use_t`. `out["route_logits"]` and `out["route_own"]` are at :1214. The CE mask is at :1397.
  * Check `cfg.code_mix` for the N1DM and N1DH arms; under "kind" the mask differs, so say what you found.
* **Paired form, on the SAME labelled windows:** `model.eval()`, `torch.no_grad()`. Run the forward twice:
  * `p_gt=1.0, use_labels=True`: every row with `lab_t.has` reads the GT code, so `route_own` is False there.
  * `p_gt=0.0`: every row reads its own code.
* Metric: the route head's argmax accuracy against `route_y`, on rows with `route_ok & lab_t.has`.
  * Report the own-code arm, the GT-code arm, and the paired difference (GT minus own).
  * Estimator: `taniteval.ci` paired episode-cluster bootstrap, B 2000, seed 0, per seed. Say that it answers only "another draw of EPISODES".
  * Beside it, no bar: the natural p_gt 0.2 draw, with own vs GT rows split by `route_own`.
  * Also report n rows, n episodes and class shares.
* Probe windows: the D1-HO windows in TRAINING batch format (`lab_t`, `route_y`, `route_ok`), built by the trainer's own path. K-LABEL already does this, pinning `route_stage` to D1-HO labels on 5,736 real windows. Find K-LABEL in the N1DH/N1DM build (`F:/TanitAD-scratch/n1dc/build/`, `stack/scripts/v8f_mini_train.py` `route_stage`) and reuse it unedited.
  * The inference batch of `n1dm_t0route_lib.run_tick0` carries NO labels (it is key-guarded), so it cannot give the GT-code arm.
  * If the dev box cannot build these batches, design the tool to run on pod CPU instead (stage the script, do NOT run it on the pod) and say why.
* **Controls (each must read its known value; at least one mutation must turn a named test RED):**
  * **K-ID:** p_gt 0 twice gives bit-identical logits and a difference of exactly 0.0.
  * **K-MASK:** at p_gt 1, `route_own` is False exactly on the `has` rows; at p_gt 0 it is all True.
  * **K-PERM:** labels permuted across rows read the chance level.
  * **K-UNIF:** a copy with `route_head` weight and bias zeroed gives a constant argmax, so both arms read the same class share and the gap is exactly 0.
* Write `RESULT_*_KOWNSPLIT.json`. A model without a route head is a REFUSAL, never a number.

## Run it on real checkpoints now (MEASURED, diagnostic only)
* **N1DH_s0 and N1DH_s1:** `F:/TanitAD-scratch/n1dc/read_n1dh/ckpts/N1DH_s{0,1}/ckpt_last.pt` (cond mode, same route head and same own mask). Their result is N1DH's K-OWN-SPLIT, reported beside N1DM.
* The N1DM checkpoints are not on the dev box yet (they arrive with `read_n1dm/pull_n1dm.sh`). Smoke the modes path on the tiny 4-step arm from the build, if its checkpoint is under `F:/TanitAD-scratch/n1dc/build/n1dm_b1/`.
* Wire the tool into `read_n1dm/run_n1dm_reads.sh` / `.cmd` as a step after Z-POST, for N1DM_s0 and N1DM_s1. Extend `dry_check_n1dm.sh` so the step is checked. Keep `.bk1` copies of every file you edit, run the dry check, and read its LAST line: it exits 0 even on FAIL.

## Constraints (binding)
* Dev box CPU only, 2 threads, ONE python process at a time.
  * Before each python run: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`.
  * Keep F: at 6 GB free or more.
  * Never touch the dev-box GPU lock or use the GPU.
* Python: the venv above with `-I`. PYTHONPATH uses `;` with Windows paths; assert that `tanitad.__file__` resolves inside `read_n1dm/tree/stack`.
  * The tree path contains `Architecture & Inference`: sed and bash patsub drop `&`.
  * pytest rc 4 ("no tests ran") is a FAIL.
* Do NOT edit `read_n1dm/tree/` (the archive of code 573ad839, checked blob by blob), `E:/Projects/TanitAD/stack`, `launch_gate.py` or `v8f_t1_eval*.py`. New files go only in `F:/TanitAD-scratch/n1dc/read_n1dm/`.
* No pod or box-2 access. Write no IP, port or id anywhere. No `*token*` or `*secret*` file names.
* Delete your own temp by literal absolute path and log each deletion (path, size) in `F:/TanitAD-scratch/n1dc/DELETIONS.log`.
* Every number carries its evidence class, the tier (T0, training-format forward) and the estimator. In .md files avoid the words: void, superseded, retracted, formerly, correction, withdrawn, "no longer", "closed loop".
* Budget: stay under about 150k context. Hand back at most 300 words plus the manifest (file paths and what each is). Put the full detail in `read_n1dm/KOWNSPLIT_BUILD.md`.

## Priority order (if you run short, bank what you have)
1. The tool + controls + test file, green.
2. N1DH_s0/s1 real numbers.
3. Wiring into the read chain + dry check PASS.
