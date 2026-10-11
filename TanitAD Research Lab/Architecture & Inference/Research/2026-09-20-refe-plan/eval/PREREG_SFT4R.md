# SFT-4R: SFT-4 re-run on the COMPLETE PDM labels, with automatic egress (pre-registration)

> Written 2026-10-11 ~07:55 UTC, while `PREREG_SFT4.md` Amendment 5's read (arm B at update 400, 923 tokens) was
> running and **before its result was seen**. It is landed in git before that result is looked at, so nothing in this
> design is conditioned on it.
> Why a re-run: `STOP_CONSEQUENCES_2026-10-11.md`. The pod stopped and nothing of SFT-4 beyond update 400 left it, so
> the registered SFT-4 read (stage 1 at the final update, then stage 2) was never made.

## What it is, and what it is not

- ⚠️ **A NEW arm, not a resume and not a replicate.** The lost run trained on 146,185 of 174,912 PDM labels
  (`PREREG_SFT4.md` Amendment 4). This one trains on all of them. Nothing from the lost run is a comparator. Its u800
  held-out read (+2.27 for each arm) is INHERITED, from the handoff only, and is quoted nowhere as a baseline.
- **Not a replicate of the u400 checkpoint.** Whatever Amendment 5's read returns, it is about that checkpoint, not
  about this recipe at its final update.
- **It runs whatever Amendment 5's read returns.** An ADOPT there does not replace this run, and a NOT PROVEN or REFUTED
  does not cancel it. Neither reads the recipe at its final update; this run does.

## Design: identical to `PREREG_SFT4.md` (Design + Amendment 1) except where stated

- **Unchanged:**
  - base `model_final.pt` (md5 `b54773f8…`, on HF `Sayood/tanitad-refe-vitl16-navtrain10-grow-final`);
  - trainable only `score_q_mlp`, `score_dec`, `score_head`; the 7-row head (rows 0–5 copied, row 6 weight 0, bias 1.5);
  - arms A and B in one pass over the same batches;
  - selection rule `navsim_v1_lane`;
  - AdamW lr 1e-4, 50-update warm-up, cosine to 0; weight decay 0.01; batch 8 × accumulation 8; 1 epoch; seed 0; bf16;
    held-out read every 400 updates;
  - the launch gate G1, G2, G3 (Amendment 1's fp32 bound), G4, G5, G7, G9, G10, and both deliberate regressions:
    an empty PDM dir turns G9 RED, and `REFE_SFT4_MUTATE_G3=1` turns G3 RED.
- **Changed 1: the PDM labels must be COMPLETE.** `onpolicy_relabel_pdm.py` has finished on every served training set
  (174,912) and every held-out set (3,137) before launch.
  - The chain refuses to arm until the relabel's exit marker is present AND the row count equals the served-set count
    minus the sets the 5 s guard excludes. Both numbers go in `chain.log` and the `data` event.
  - ⛔ Amendment 4's "launch on what is written" does not apply here.
- **Changed 2: lane rows.** The seeded 15,000-set subset (seed 20261004), using the rows written by launch.
  - Floor: 2,109, the lost run's count. Below it, the chain refuses.
  - The count is logged. The held-out lane side file must be present (G10).
- **Changed 3: automatic EGRESS (new; the lesson of the stop).**
  - **What is pushed:** at every held-out read and at exit, the chain pushes to the private HF model repo
    `Sayood/tanitad-refe-sft4r`:
    - `scorer_states_u<N>.pt` (both arms; ~34 MB each, MEASURED on the surviving u400 file, which holds `A`, `B` and
      `update`);
    - `sft.log`, `chain.log`, `preflight.log`, `mutate.log`, and the gate output.
  - **Full checkpoints are not pushed.** `eval/build_sft_ckpt.py --base model_final.pt --states <scorer_states> --arm
    {A,B}` rebuilds them on the dev box. This read uses one such rebuild.
  - **PDM labels:** `train_pdm/` and `heldout_pdm/` go to the dataset repo `Sayood/tanitad-refe-navtrain-banks` under
    `pdm_labels/` as soon as the relabel exits, before SFT-4R arms. P2 needs them too.
  - **Size, pre-push (binding, HF quota):**
    - scorer states: ≤ 7 reads × 34 MB ≈ 0.24 GB;
    - logs: negligible;
    - PDM labels: measured with `du -sb` on the pod before the push. The account's private total, read from the API
      listing, plus this push must stay inside the Pro allowance.
  - **The success marker is the REMOTE listing, never the local file.** After each push, the chain lists the repo and
    requires every pushed path to be present with a size equal to the local file. It writes `ZZEGRESS_OK <N>` or
    `ZZEGRESS_FAIL <N> <reason>` to `chain.log`.
  - A failed push does not stop training. It is retried at the next read. Two consecutive failures stop the chain
    **after** the next checkpoint is written, so the pod never runs on for hours producing what cannot leave it.
  - **Egress gate `G-EGR`, before the first training step:** the gate's own JSON is pushed and must appear in the
    remote listing with the right size. Deliberate regression: a push to a repo name the token cannot write must
    make `G-EGR` RED.
  - ⚠️ **The tooling does not exist yet.** No upload code exists in `refe/` or `eval/`. MEASURED 2026-10-11: a grep for
    `upload_folder|upload_file|HfApi` over both directories finds 0 files. Building `refe/hf_egress.py` with a
    mocked-API test is a launch blocker for this run. The pod authenticates with a token the PI provisions on it;
    the token is never passed in argv and never copied there by an agent.

## Stage 1 and stage 2: unchanged from `PREREG_SFT4.md`

- **Stage 1** (pod, label-only, the 3,137 held-out sets of 24 logs, PDM truth):
  - primary: pick(arm, navsim_v1_lane) − pick(base, navsim_v1) × 100;
  - paired log-cluster bootstrap, 10,000 resamples, seed 20260927;
  - an arm PASSES iff its lower bound > 0 **at the final update**;
  - if both pass, B goes forward only if pick(B) − pick(A) > 0; otherwise A;
  - no interim read decides anything.
- **Stage 2** (dev box, never on the pod): the winner, rebuilt from its pushed scorer states, with `eval/stage2_sft.py
  --rule navsim_v1_lane` in the unchanged seam (route fix ON, A7 ON).
  - Tokens: SPEC_NAVTEST Amendment 5's 923 confirmation tokens, against the deployed system's own scores (85.61 PDMS).
  - **ADOPT** iff the lower bound > 0 AND no longitudinal or lateral family component separates adversely.
  - **REFUTED** iff the upper bound < 0.
  - Otherwise **NOT PROVEN**.
  - STRATEGIC is UNAVAILABLE in NAVSIM and is reported as such.
- ⚠️ **The 923 tokens are being REUSED, and this is stated in advance.** They have already decided SFT-3 arm A and
  will have read SFT-4's u400 checkpoint. SFT-4R is the third scorer read on them. Each read is a fixed, pre-registered
  rule and none tunes on the tokens, but repeated reads on one confirmation set accumulate selection. ⇒ **On ADOPT,
  the full 12,146-token navtest is a REQUIRED confirmation before any claim, not a courtesy.**
- The interval answers *another draw of logs* only: one fine-tune seed, deterministic inference.

## Order on a new pod, and what blocks it

1. **PI:** the pod. First check whether the old pod is Stopped (its disk may still hold the labels; copy them off
   first) or Terminated. Then provision a new pod and put a write-scoped HF token on it.
2. Restore the banks from HF on the pod, and re-fetch navtrain.
3. PDM relabel, train and held-out, ~11.3 h CPU (ESTIMATED from 258 sets / min); then push the labels (egress).
   The lane relabel runs beside it on Amendment 2's priority (the PDM relabel first).
4. `hf_egress.py` and `G-EGR` must exist and pass. Then SFT-4R: gate → mutations → run, ~9.3 h GPU (2,335 × 14.4 s,
   ESTIMATED from the lost run's rate).
5. P2 (`eval/PREREG_P2.md`) follows on the same complete labels. Its chain gets the same egress step, by an amendment
   to PREREG_P2 landed before its launch.

## Next arm, by outcome (Rule Zero)

- **ADOPT:** first the full 12,146-token navtest and the four families. Then a training-seed replicate (seed 1, same
  flags; H-ESTIM-SEED-1) before the lever is claimed. Deploying it is the PI's call.
- **Stage 1 fails for both arms, or stage 2 is NOT PROVEN / REFUTED:** P2 still runs, because it fixes the second
  defect (the detached visual context, `REVIEW_7_GAP_TO_PAPER.md`), which a scorer-only fine-tune cannot reach.
  - SFT-4R's log shows which half failed: the held-out ranking against PDM truth, the AUC of NC / DAC / DDC, and the
    lane AUC.
  - If held-out passes but navtest does not, the yardstick gap (`RESULT_SFT3.md`, yardstick check) is the next
    diagnosis.

## Amendment 1 (2026-10-11, before any SFT-4R launch): the tooling exists; what is and is not proven

Landed with this amendment. Nothing in stage 1, stage 2 or their rules changes.

- **The egress tooling exists.** The "tooling does not exist yet" blocker above is closed by `refe/hf_egress.py`:
  - subcommands `push`, `gate`, `watch` and `verify`;
  - `refe/test_hf_egress.py` passes 13/13 against a mocked API.
  - **Live check:** `raw/2026-10-11-hf-egress/egress_verify_live.json` read the 1,288,484,600 B `model_final.pt` back
    from its HF repo. The size matched on the attribute, the LFS size and the listing. A +1 B control read as a
    mismatch.
- **The success marker, as built:** `<utc> ZZEGRESS_OK <tag> <reason>` or `<utc> ZZEGRESS_FAIL <tag> <reason>` in
  `chain.log`. This replaces `ZZEGRESS_OK <N>` above. The gate writes `<out>/g_egr_result.json` and prints
  `ZZEGR_PASS` or `ZZEGR_FAIL`.
  - The lines carry a timestamp prefix. ⇒ the chain greps `" ZZEGR_PASS "`, never `"^ZZEGR_PASS"`, which can never
    match.
- **HF quota read (MEASURED 2026-10-11):**
  - `usedStorage` comes only from a per-repo `repo_info(..., expand=["usedStorage"])`. The list endpoints reject the
    expand.
  - The account's private total, summed that way, is **199,397,374,781 B against the 1e12 B allowance**.
  - An unknown total fails closed. No push proceeds on a quota the tool could not read.
- **The PDM labels go to `pdm_labels_sft4r/`, not `pdm_labels/`**, in `Sayood/tanitad-refe-navtrain-banks`. The prefix
  names the run that produced them, so they cannot be confused with, or overwrite, any earlier upload. P2 reads them
  from there.
- **The PDM completeness check exists:** `refe/pdm_label_complete.py`, with 13/13 in its test.
  - It recomputes the eligible count from the sets and credits only rows of the served (newest) `ckpt_step`.
  - It refuses on a missing DONE marker, failed sets, a row-count mismatch, or a printed `served sets:` line that
    disagrees with its own recount.
  - The held-out pass runs `--served-only`. That is an assumption, because SFT-4's own command line was never banked.
    It is guarded by `--expect 3137`: a wrong assumption stops the chain instead of training on it.
- **The G3 deliberate regression is wired in the trainer**, not only in a test.
  - `refe/scorer_finetune.py:184` perturbs a scorer copy under `REFE_SFT4_MUTATE_G3=1`; the gate key is at line 459.
  - `refe/sft4r_pod.sh` has the `mutate_g3` mode, and the chain refuses unless it turns G3 RED.
  - ⚠️ **It has not been RUN.** The preflight needs the pod's data, so the RED reading is enforced at launch by the
    chain, not shown here.
- **The held-out lane labels are RESTORED, not recomputed.** They come from
  `raw/2026-10-04-lane-discipline/heldout_{navsim,teacher}_lane_labels.jsonl.xz` (`refe/sft4r_relabel.sh` step 1),
  the labels SFT-4 trained against.
- ⛔ **NAMED GAP — the train lane-subset queue.** The `--queue` source of SFT-4's 15,000-set subset was never written
  into a banked script.
  - The candidate is the HF banks' `onpolicy_queue`.
  - `sft4r_relabel.sh` refuses (`ZZLANE_TRAIN_EXIT 9`) unless `LANE_QUEUE_SRC` names the restored directory. Nothing
    is guessed.
  - The chain then requires ≥ 2,109 train teacher-lane rows.
- **What the three pod scripts have had: `bash -n` only.** `refe/sft4r_relabel.sh`, `refe/sft4r_chain.sh` and
  `refe/sft4r_pod.sh` have NOT been dry-run against a fake root.
  - Every flag they pass was traced to its argparse definition on 2026-10-11.
  - Their first real execution is on the new pod, and every step there is fail-closed.
