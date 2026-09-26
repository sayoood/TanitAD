# RESULT — refcv6 standard held-out battery (four families): loader, G0, pipeline

**Stream:** EvalFlyWheel, refcv6 standard tests, four-family held-out battery (the NavSim suite is a sibling stream).
**Author:** EvalFlyWheel agent, working from the Master Mind session, 2026-09-23/24 and 2026-09-26.
**Pre-registration:** `SPEC.md`, registered 2026-09-23 23:33 Berlin (sha256 `32625b9a…`), before any refcv6 forward. Five amendments, each written BEFORE the measurement it governs:
* **A1** (09-24 01:11, `1ae8f6af…`): new deliberate regressions M2–M4, after the as-registered M1 had no power (§1).
* **A2** (09-26 09:50, `7bfca6cf…`): the wrapper control is redefined to isolate the merge from bf16 arithmetic, after step 5000's wrapper clause read FAIL (§1b).
* **A3** (09-26 12:46, `fd65da5c…`): the paired LATERAL yaw-rate cell is masked to steps with a path tangent (four_families' own `pair_valid`), after the shared unmasked cell was measured inflating refcv4b 6.4× (§5 F10). No bar reads yaw-rate.
* **A4** (09-26 12:58, `e94e1ab6…`): a zero-training LEVER PANEL, reported with no bar, under RULE ZERO. L1 inference-seed average; L2 causal-hold blend `w·os + (1−w)·ha` (2-fold episode-disjoint cross-fit, identity and shuffled-plan controls); L2e echo blend (diagnostic only); L3 deterministic DDIM (eps = 0, one extra GPU roll). Each lever is a DIFFERENT planner from the registered arm.
* **A5** (09-26 13:55, `a8095594…`): the A16 mid-run switch. Per-step stamps; the FINAL runs on the 82c2331 tree with the run's post-switch config; a tree-equivalence control; TACTICAL under BOTH label clocks with controls C1–C4; no curve fit across step 34,500.
**Evidence class:** MEASURED (ours) unless stated. Every number cites a JSON under `raw/`.

> ⛔ **Every number from the kit checkpoint (step 1000) is PIPELINE VALIDATION ONLY (SPEC §3.6).**
> No bar is evaluated on it, and no step-1000 number is a result about refcv6.

> ⛔⛔ **THE RUN WAS SWITCHED MID-WAY (PI: "Stop now, resume with fixes", 2026-09-26 13:30).** It stopped at step 34,500 and resumed on 82c2331. F3's cascade loss trains from there, and tactical labels are read on the true clip clock.
> * Checkpoints at step ≤ 34,500 (5k, 15k, 20k, 30k) carry the pre-switch stamp below.
> * **The FINAL is the post-switch HYBRID:** *"hybrid: F3 cascade loss + true label clock from step 34,500"*.
> * A comparison across the switch mixes training time with the fix, and is never attributed to the fix alone (SPEC A5).
>
> ⛔ **RUN-DEFECT STAMP (pre-switch checkpoints)** (Master Mind audit
> `92337fa6`; register `D-REFCV6-F3-WHITELIST`, `D-REFCV6-LABEL-CLOCK`, landed `9d16c441`):
> **"F3 detach-only (no per-stage loss), F4 on the last layer only; tactical labels ~0.37 s early."**
> * Decoder stages 0–2 carry frozen, randomly-initialised modulations: their heads are bit-identical at 1k / 5k / 30k.
> * The tactical band is read on the provider row at `row × 0.1`; the true anchor is at 8.369 s. On the eval split, 598 / 5,699 (10.5 %) of tactical-supervised windows lie outside the true ±2 s band.
> * **Since A5, TACTICAL is scored under BOTH label clocks at every checkpoint.** Label tables are built once with 82c2331's own V3Dataset. The OLD table reproduces the old-tree roll labels bit for bit on 4,754 / 4,754 windows, and the corrected clock's eval census equals the run's own. PRIMARY is OLD for a pre-switch checkpoint and CORRECTED for the FINAL. The trajectory-derived tactical κ reads no label and is not exposed.
> * A cache row is 0.100667 s (programme-wide). Rates in m/s assume 0.1 s, i.e. +0.67 %, identically for every arm, baseline and control.
> * ⛔ **No weakness below is attributed to the registered refcv6 design while these hold.**

## STATUS — where the work stands (kept current; read this first if I am stopped again)

| checkpoint | G0 as registered | G0-A1 | G0-A2 (operative gate) | battery (T1, S2 + S6, 2 inference seeds) |
|---|---|---|---|---|
| step 1000 (kit) | VOID (M1 no power) | **PASS** | **PASS** (fp32 wrapper 6.7e-7; bf16 P1 8.4e-4; W1/W2 detected 1/26) | pipeline validation only: seed 0 rolled on GPU; panel on CPU (§3) |
| step 5000 | FAIL (wrapper 3.9e-3 + M1 VOID) | **FAIL** (wrapper clause only) | **PASS** (fp32 wrapper 1.9e-6; §1b) | **DONE, both seeds (§4)**: BAR-R6-1..4 **FAIL**, BAR-R6-5 **PASS**. The A4 L2 blend beats the echo (−0.0122 sep, both seeds). Banked `raw/step5000/` |
| step 30000 | — | — | — | RUNNING in `chain_milestones.sh` (full G0 + A1 + A2, 2 seeds; old tree = its training tree) |
| step 15000 / 20000 (optional) | — | — | — | `chain_optional.sh`: after the L3 rolls, never started after 12:00 09-27 (ckpts pulled read-only, md5 `46eb2185…` / `f4320c89…`) |
| FINAL (step 50,400; post-switch HYBRID) | — | — | — | `chain_final_v2.sh` on the **82c2331 tree** (SPEC A5). It waits for `summary.json` (Thor ETA ~09-27 19:30 Berlin), pulls `ckpt.pt`, metrics and config with 3-way md5, then runs. The old chain's final stage is DEFERRED |

**Unattended chains.** `code/chain_milestones.sh` runs step 5000 and then step 30000, both on their training tree. It logs to `raw/chain.log` on the dev box. Its final stage is DEFERRED (SPEC A5): `pull_final.sh` now prints `ZZFINALDEFERREDZZ`, and `code/chain_final_v2.sh` runs the FINAL on the 82c2331 tree.
* It banks each tag, sanitized, into `raw/<tag>/`.
* It appends the paths to `LANDING_READY.txt`.
* Every GPU stage waits on the dev-box gate. The orchestrator holds no CUDA context.

**Post-bank watcher** (`code/post_chain_watch.sh`, started 13:07 09-26, markers in `raw/post_watch.log`). After the chain banks a tag it runs `post_tag.sh <tag>`: the A3 recompute (step 5000 only, whose panels predate A3), the A4 lever panel, re-rendered tables, a re-bank and a LANDING_READY append. After step 30000 is banked, while the chain waits on Thor, it runs the L3 eps0 rolls for step 5000 and step 30000 behind the gate; after the final, L3 for the final.

**If I am stopped:** every chain keeps going unattended. Read these marker logs, plus each tag's `battery_summary.json` and `RESULT_SECTION.md` (`code/result_section.py <tag_dir>`):
* `raw/chain.log`: `ZZB30000DONEZZ`, `ZZBANKED_step30000ZZ`, then `ZZNOFINALZZ … ZZFINALDEFERREDZZ` (by design, A5) and `ZZCHAINMENDZZ`.
* `raw/post_watch.log`: `ZZPOSTDONE_<tag>ZZ`, `ZZL3DONE_<tag>ZZ`.
* `raw/optional.log`: `ZZOPTDONE_15000ZZ`, `ZZOPTDONE_20000ZZ`.
* `raw/final_v2.log`: `ZZFINALWAITZZ` polls, `ZZFINALOKZZ`, `ZZBFINALDONEZZ`, `ZZBANKED_finalZZ`, `ZZFINALV2ENDZZ`.

The headline is BAR-R6-1 at BOTH seeds of the FINAL tag (`raw/final/battery_summary.json` → `bars`).

**Where I stopped** (updated each turn): 2026-09-26 ~14:15. Step 5000 is done and banked; step 30000 is running; the FINAL chain is polling Thor.

<!-- HEADLINE -->

---

## 1. G0: the loader reproduces the run's own in-run eval

**What was reproduced.** `metrics.jsonl`'s eval row at step 1000, i.e. 8 fixed batches of 16 = 128 windows. The windows are `randperm(len(e_ds), Generator().manual_seed(12345))[:128]` over the 23,772 eval windows. Each value is the trainer's own `compute_losses_v3` output, and the checkpoint is the kit's `ckpt_step1000.pt` (md5 `7c3ad3c1…`).

**How.**
* **The model** is rebuilt by `code/refcv6_loader.py`, which replays `refc_v3_train.train()` block by block with line references, and loads the checkpoint STRICTLY:
  * 1,101 keys, 0 missing, 0 unexpected;
  * `param_breakdown` equals the run's `config.json` (total 100,893,747);
  * the kit anchor file equals the checkpoint's anchor buffers (max |Δ| 0.0).
* **The eval dataset** is the trainer's own, and its census equals the run's `config.json` stamps number for number:
  * 139 episodes → 23,772 windows;
  * agent join 22,663/23,772 windows labelled and 776,801 prefilter boxes, pad 397;
  * SAM3 GT 23,430/23,772 windows (no_file 342, on the same 2 clips as on Thor).
* **Levers:** `--trunk-compile` is dropped (no Triton; it wraps the call only). bf16 + NHWC, frozen + folded BN, dedup and chunk 8 are kept.
* **The forward is micro-batched** (3,3,3,3,4) inside the single `model(...)` call, so every loss reduction still runs over the 16-row batch. **Wrapper control:** micro vs plain at batch 4 with the DDIM draw zeroed differs by at most 8.4e-4 relative over 81 terms, which passes the 1e-3 bar.
* **The DDIM draw cannot be replayed:** the run drew `eps` from its training RNG. Stochastic terms are therefore judged against 8 dev-box inference seeds (SPEC §2).

**Result, 82 judged terms** (`raw/g0_step1000/g0_step1000_table.md`):

| class | n | criterion | read |
|---|---|---|---|
| COUNT (data / geometry) | 42 | exact (±1e-5) | **42/42 exact** |
| SMOOTH (deterministic in the seed) | 20 | ≤ 1 % each, median ≤ 0.2 % | **20/20**. Median rel dev **0.029 %**; max 0.43 % (`goal2s_err_m`) |
| MATCHED (Hungarian set losses) | 15 | ≤ 15 % each, median ≤ 5 % | **15/15**. Median **0.015 %**, max 0.28 % |
| STOCHASTIC (move with the DDIM seed) | 5 | in the 8-seed 99 % PI | **5/5**. Examples: `eval_traj` **1.21898** in-run vs 1.21337 ± 0.00913 (PI [1.167, 1.259]); `eval_loss` **34.26996** vs 34.2399 ± 0.0235 (PI [33.81, 34.67]) |
| EXCLUDED | 1 | — | `eval_goal_gate_grad`, the gradient left by the last training backward, not an eval quantity |

**The verdicts, reported as registered:**
* ⛔ **G0 as registered = VOID.** The pre-registered deliberate regression **M1** was the lift bank without `equalize_bottom_rows`, i.e. `refcv3_arm.py:1101`'s build. It moved **0 of 20 SMOOTH terms** beyond 1 % (e.g. `eval_map` 0.88997 → 0.88927). The equalized strip is 43 image rows, about 3 rows of the stride-16 map. SPEC §2 says a gate that cannot fail is VOID, so it is.
  * ⚠️ **As run, the verdict printed FAIL** on two COUNT terms (`eval_lon_tac`, `eval_tacv6_lon_ce`). That was **my implementation bug**: a substring test for `n_` matched `lo[n_]tac`. I found and fixed it before the G0 output existed (01:07). Under the rule the SPEC's examples define (a standalone `n` token), both are SMOOTH and read 0.07 % and 0.014 %. Both verdicts are banked (`g0_step1000.as_run.json`, `g0_step1000.json`).
* ⭐ **G0-A1 = PASS** (SPEC Amendment A1). A1 has three new deliberate regressions on the same 128 windows, inference seed 0, and **all three are detected** (`raw/g0_step1000/g0_step1000_A1_table.md`):
  * **M2**, max-speed input withheld: 6 terms out, e.g. `tacv6_goal_bce` +12.5 %;
  * **M3**, ego-history window zeroed: 4 terms out, e.g. `eval_traj` +4.0 % (outside the PI) and `anchor_acc` 0.055 → 0.141;
  * **M4**, the tactical-goal `pos_weight`/mask absent, which is `refcv3_arm.load_model`'s literal state: 4 terms out, `tacv6_goal_bce` −26 %.

⇒ **The loader builds the model the run trained, and the eval it scored; and the gate can see wiring defects of the kinds that matter.** Its one measured blind spot is M1, the lift's bottom-strip mask, which is below the loss's resolution at step 1000.


### 1b. Step 5000: G0-A1 = FAIL on the wrapper clause alone; the discriminating experiment; G0-A2 = PASS

**The as-registered reading** (`raw/step5000/g0.json`, `g0_A1.json`; `ckpt_5000.pt`, md5 `8a1e4da0…`):
* **Reproduction:** all 82 terms are within tolerance. SMOOTH median rel dev 0.096 %, MATCHED median 0.19 %, and COUNT is exact.
* **Mutations:** M2, M3 and M4 are all detected (4 / 7 / 4 terms out).
* **The wrapper control is the only failure:** plain vs micro[1,3] at batch 4, DDIM eps zeroed, read **3.921e-3** against its 1e-3 bar. The worst terms were `box3d_yaw` 3.9e-3, `goal2s_err_m` 2.0e-3 and `goal_tac` 1.5e-3.
* ⇒ **G0-A1 at step 5000 = FAIL**, as registered, and it stays FAIL.

**The discriminating experiment** (SPEC Amendment A2, written 09:50 **before** it ran; `raw/step5000/g0_A2_wrapper_probe.json`; tool `code/wrapper_probe.py`):

| condition | max Wrapper rel | max Floor rel | terms over bar | W1 / W2 detected |
|---|---|---|---|---|
| P1 as run (cuDNN TF32 on, bf16 trunk) | 3.921e-03 | 2.785e-05 | 30 of 64 | — |
| P2 cuDNN TF32 off + deterministic, bf16 trunk | **3.921e-03** (unchanged) | 2.792e-05 | 30 of 64 | — |
| **P3** fp32 trunk, TF32 off, deterministic | **1.891e-06** | 9.743e-08 | **0 of 64** | **1 / 27** |

**What it shows.**
* cuDNN TF32 and algorithm determinism explain **nothing**: P1 and P2 read identically.
* The whole 3.9e-3 is the **bf16 trunk under a change of batch SIZE** (4 → 1+3). A permutation at the same size moves the terms by only ~1e-7 to 2.8e-5.
* With the trunk in fp32 the micro-batched forward agrees with the plain one to **1.9e-6** (worst: `traj`). That is under the A2 bar, max(3 × floor, 1e-5), on every term.
* Both deliberate wrapper regressions are **detected**:
  * W1, 0-dim outputs merged as part 0: 1 term (`goal_score_absmean`);
  * W2, micro-parts concatenated in reverse: 27 terms.
* ⇒ **The wrapper's merge is exact. The step-5000 reading is precision, not a merge defect.**
* ⇒ **G0-A2 at step 5000 = PASS.** Both verdicts are always reported.
* **The same probe at step 1000** (`raw/g0_step1000/g0_step1000_A2_table.md`): P1 8.44e-4 and P2 8.45e-4, which reproduce the §2 reading; **P3 6.7e-7**; W1/W2 detected 1/26 ⇒ **G0-A2 PASS**. So the bf16 batch-size sensitivity **grows with training** (8.4e-4 at step 1000 → 3.9e-3 at step 5000), while the fp32 merge stays exact at both. That is why a fixed 1e-3 bar on the bf16 reading was the wrong instrument.
* ⚠️ The battery itself does not use the wrapper: its rolls are single-window forwards. The bf16 batch-size sensitivity (~4e-3 relative on loss terms) is ~2 orders below the DDIM inference-seed spread of the plans.

---

## 2. The battery: surfaces, arms, instrument

## A. What was built, and why each piece exists

| piece | what it does | why it could not be reused as-is |
|---|---|---|
| `code/refcv6_loader.py` | Rebuilds refcv6 **exactly as `refc_v3_train.train()` builds it**, block by block with line references, and loads the checkpoint STRICTLY. It also rebuilds the held-out eval dataset exactly as the trainer's eval block does. | `taniteval/tools/refcv3_arm.py::load_model` builds the BEV lift bank **without `equalize_bottom_rows`** (`refcv3_arm.py:1101`; the run passes 43, `refc_v3_train.py:6826-6830`). It also sets none of the model-level attributes the loss reads (`_cls_class_weight`, `_rig_camera`, the tactical-goal `pos_weight`/mask, …). |
| `code/refcv6_roll.py` | Runs refcv3_arm's own `run_dump` (the dump contract, nav controls, model-free controls and lead join are all refcv3_arm's). Three seams are patched: the model builder, the corpus builder, and the model call. The call is wrapped so refcv6 receives **`ego_poses` (the observed window) and `v_max_ms`/`v_max_valid` (the eval sidecar)**. Two further arms run under a forked RNG: max-speed withheld, and the 30 km/h obedience forward. An **exact per-window backbone memo** (`torch.equal` frames, BN frozen) removes the 5–6× repeated trunk passes. | refcv3_arm's call never passes `ego_poses` or `v_max_ms`, and the refcv6 build **refuses both omissions** (`refc.py:4413`, `refc_v3.py:1937`). So refcv3_arm cannot roll the full refcv6 arm; the 2026-09-19 E9 run was a `--size tiny` checkpoint without these channels. |
| `code/microbatch.py` | Splits the ONE `model(...)` call inside `compute_losses_v3` into micro-batches (3,3,3,3,4) and merges the outputs. Every loss reduction therefore still runs over the in-run eval's 16-row batch. | A 16-row refcv6 forward does not fit the 8 GB RTX 4060: the frames alone are 1.96 GB fp32 before the trunk normalises them twice. |
| `code/reproduce_inrun_eval.py` | GATE G0 (SPEC §2): the trainer's own `compute_losses_v3` over the trainer's own 128 fixed windows, 8 inference seeds, a wrapper control, and deliberate-regression arm M1. | — |
| `code/g0_mutations.py` / `code/wrapper_probe.py` | G0-A1 (M2–M4 deliberate loader regressions) and G0-A2 (the fp32 merge-isolating wrapper probe with its floor and W1/W2 regressions). | — |
| `code/roll_seed.py`, `code/chain_milestones.sh`, `code/pull_milestone.sh`, `code/pull_final.sh` | Per-seed GPU child process; the unattended milestone chain; read-only md5-verified checkpoint pulls (the final only after `summary.json`, with md5 checked on Thor before and after the copy and on the dev box). | — |
| `code/refcv6_panel.py` | The panel dump: refcv6 + STOP + the banked baselines as arms on the SAME windows, with pairing verified per window. Also the analysis (refcv3_arm, imported), cross-model paired families, the v6 tactical decoder metrics, the frozen acceptance instruments, the 6 s read, and the VOID gates. | refcv3_arm pairs arms inside one roll only; it has no cross-checkpoint pairing and no STOP control. |
| `code/run_battery.py` | ONE command: gate → G0 → rolls per inference seed → panels → bars. | — |
| `code/check_pairing_surface.py` | MODEL-FREE proof that the 416×1024 kit clips and the 256×640 baseline clips carry bit-identical poses and actions, and that the banked GT equals the GT recomputed from the kit. | — |
| `code/gpu_gate.py` | The brief's dev-box gate: used < 4,300 MiB, no other python compute, ≥ 8 GB free RAM. The verdict is written as JSON. | — |

**Surfaces (MODEL-FREE, measured before registration, `raw/pairing_surface.json`).**
* **S2** = the banked refcv4b / refcv5-v2 dev-box grid (stride 5, instants 0.5–2.0 s) on the 139 kit clips: **4,754 windows / 139 episodes**.
  * On all 139 clips the 256×640 and 416×1024 caches carry **bit-identical `poses` and `actions`**.
  * The GT recomputed from the kit poses equals the banked `g` on **4,754/4,754** windows (max |Δ| 0.0).
  * ⇒ refcv6 and the banked baselines are scored on **identical clips, windows and GT futures**, each on its own input geometry.
* **S6** = the S2 windows whose 60-step future is inside the clip, read at 1–6 s from the full 8-slot plans.

**Arms and controls.** `os` is refcv6's own selection, fed exactly as `compute_losses_v3` feeds it. With it run:
* the nav controls: shuffled, withheld (`nav_cmd=None`), flipped;
* the max-speed-withheld arm;
* the model-free controls: `ha` (hold-action), `ha0` (**constant velocity**), `ha0_ext` (echo), `stop` (**STOP**);
* `oracle_sel` (T0 ceiling);
* the banked baselines refcv4b and refcv5-v2 (seeds 0 and 1).

The **VOID gates** are checked on every panel: model-free arms bit-identical across all dumps, `g` bit-identical, STOP displacement 0, `ha0` curvature 0.

**Controls that had to read a known value, and did (all MEASURED):**
* wrapper micro vs plain: 8.4e-4 max relative at step 1000 under bf16. Under A2 the merge-isolating fp32 probe reads **1.9e-6** at step 5000, and both deliberate wrapper regressions are detected (§1b);
* frame memo on the real trunk: max |Δ| **0.0** (`raw/controls/frame_memo_exactness_cpu.txt`);
* inference seeds **must differ** on a sampler arm. The first CPU smoke read **4.5e-8 m** between seeds 0 and 1, which exposed the loader resetting the caller's RNG (fixed with `fork_rng`, §5). After the fix: **0.22 m**;
* AUROC/AP against sklearn: identical to 1e-15; 1.0 / 0.0 / 0.5 on separable / reversed / tied;
* prefetch + memo roll vs a plain roll, same seed on CPU: `g` and model-free arms 0.0, `os` ≤ 4.8e-6 m.

**Estimator.**
* Points are FULL-SET pooled means.
* Intervals: `taniteval.ci.episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`, n_boot 2000, seed 0, cluster = clip. ⛔ `overlapping_holdout_se` is not used anywhere.
* Every separated cell answers the **episode** question. The **inference** question is answered by rolling two inference seeds. The **training-run** question is **NOT MEASURED** (one training seed, `H-ESTIM-SEED-1`).

**Tier.** Every number is **T1 = self-action OPEN loop** (one forward, own selection). It is not closed loop and not driving (EVAL_DOCTRINE 2026-09-02). `os`'s T1 stamp is UNRULED for an action-free model, as for refcv3/refcv4b/refcv5-v2. `oracle_sel` is **T0**.



---

## 3. The battery on the kit checkpoint (step 1000): PIPELINE VALIDATION ONLY

> ⛔ **Not a result.** Step 1000 is ~2 % of the 50,400-step run. No bar is evaluated (SPEC §3.6). The numbers below prove the instrument runs end to end on the real surface and that its controls read their known values. They are not a reading of refcv6.

**Device.**
* The roll ran on the dev-box **RTX 4060 (GPU, bf16 trunk as trained)**, inference seed 0: 4,754 windows / 139 episodes, 0 skipped. Provenance stamp: `REAL_CHECKPOINT_ON_REAL_CORPUS`.
* The full panel (pairing gates, analysis, cross-model paired families, tactical, acceptance, S6) ran on **CPU**; it does no forward passes.
* ⚠️ **The seed-1 roll of step 1000 was NOT run.** The orchestrator deadlocked on its own GPU gate (§5 F7) and was stopped. The replicate path is validated on the CPU smoke instead (seeds 0 vs 1 differ by 0.22–0.39 m, as a sampler arm must). Step 1000 carries no bar, so nothing is lost. The step-5000 run rolls both seeds with the fixed orchestrator.

**VOID gates on the real surface (all PASS, `raw/step1000/void_gates.json`, `panel_record.json`).**
* The model-free arms `ha`/`ha0`/`ha0_ext` and `g` are **bit-identical** between the refcv6 dump and all three banked baseline dumps on 4,754/4,754 windows.
* STOP displacement is exactly 0, and its ADE equals the mean GT distance exactly (max |Δ| 0.0).
* `ha0`'s own curvature is exactly 0 and its tactical κ is exactly 0.
* The selection profile is non-degenerate: 44 distinct anchors, modal share 0.454.

**The instrument reads the banked baselines at their registry values.**
* The banked refcv4b dump, re-analysed on its own 141 episodes with the current `refcv3_arm`, reproduces MODEL_REGISTRY §4.8 **exactly**: os 0.2965 [0.2697, 0.3280], ha 0.2996, ha0_ext 0.2874, ha0 0.6723 (`raw/controls/registry_reproduction_control.json`).
* On the 139-clip surface:
  * refcv4b − ha0_ext = +0.0088 [−0.0058, +0.0249], not separated (registry, 141 eps: +0.0101 [−0.0050, +0.0273]);
  * refcv5-v2 − ha0_ext = **+0.0204 [+0.0046, +0.0383], separated worse** (registry: +0.0205 [+0.0043, +0.0390]).

  Same readings, two clips fewer.

**Step-1000 refcv6, for the record only.**
* `os` ADE 0–2 s: 0.9015 [0.8385, 0.9716] m against ha0_ext 0.2886 and CV (`ha0`) 0.6764. Target-speed acc@0.5 is 0.48, and traj-derived lat/lon κ are 0.20/0.05.
* The tactical v6 decoder's lat κ is 0.138; with nav zeroed it collapses to the majority class (κ 0.000).
* Full tables: `raw/step1000/TABLES.md`.

⭐ **A by-product, MEASURED, about the BANKED baselines (not refcv6): at their own 6 s horizon, both beat the echo control.**
* **S6** is the S2 windows with a valid 6 s future: 3,668 windows / 139 episodes, instants 1–6 s, read from the banked full plans (`plan_full_nav_true`). The GT is recomputed and asserted equal from both dumps' own banked futures.
* refcv4b − ha0_ext = **−0.8254 m [−1.1031, −0.5708], separated**.
* refcv5-v2 (seed 0) − ha0_ext = **−0.8338 [−1.1206, −0.5756], separated** (seed 1 agrees).
* The echo control is 3.548 m at 1–6 s; refcv4b is 2.723 m and refcv5-v2 2.714 m.
* ⇒ LEADERBOARD 1e's "neither beats doing nothing" is a **2 s** statement. **At 6 s both do**, with the usual caveats:
  * T1, one training seed each;
  * the route input is oracle nav, derived from the ego's future;
  * the echo control degrades with horizon by construction.
* Not pre-registered as a bar; reported as a measurement with its estimator (`raw/step1000/cross_paired_s6.json`).

⛔ **A finding about the frozen OBEDIENCE instrument (refcv6_acceptance, frozen 2026-09-16): as defined, it is structurally unsatisfiable on 97.5 % of its population.**
* The population is "GT max > 40 km/h". **2,236 of those 2,294 windows start ABOVE the forced 30 km/h** (v0 median 15.4 m/s).
* `planned_max_speed` counts the first interval from v0, so no plan starting above the ceiling can obey. **2,229/2,294 windows have NO compliant candidate** in the 117-anchor fan, and all 2,229 have v0 > 30 km/h.
* On the 58 windows with v0 ≤ 30 km/h, 35 obey and none lacks a candidate.
* The instrument's own `rows_with_no_compliant_candidate` exposes this; but the bar (≥ 99 %) cannot be met by ANY model on this population.
* ⚠️ **Escalated as an instrument-design question.** Restrict the population to v0 ≤ ceiling, or score a deceleration-toward-the-ceiling criterion. Either way it is the owner's or PI's call, because the bar was frozen before training and I do not move it.


## 4. Step 5000: the first milestone reading (T1 primary; S2 + S6; inference seeds 0 and 1)

> ⛔ **Pre-switch checkpoint.** The stamp applies: *F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early*. No weakness below is attributed to the registered design.
>
> Step 5000 is ~10 % of the 50,400-step run. This is an early-checkpoint reading, not the headline.
>
> Evidence class: MEASURED. The full digest is `raw/step5000/RESULT_SECTION.md` (rendered from the JSONs by `code/result_section.py`); full tables are `raw/step5000/TABLES_s0.md` and `TABLES_s1.md`.

**Gate.** G0 as registered: **FAIL**. G0-A1: **FAIL**, on the wrapper clause only (3.921e-3). **G0-A2 (operative): PASS**, with an fp32 wrapper of 1.89e-6 (§1b).

**Bars.** SPEC §3.4 requires each bar to hold at BOTH inference seeds. Surface S2, ADE 0–2 s, paired episode-cluster bootstrap, 4,754 windows / 139 episodes.

| bar | seed 0 | seed 1 | verdict |
|---|---|---|---|
| **BAR-R6-1** (primary) `os − ha0_ext` | +0.0885 [+0.0695, +0.1094] sep | +0.0885 [+0.0695, +0.1092] sep | **FAIL** |
| BAR-R6-2 `os − ha` | +0.0761 [+0.0569, +0.0974] sep | +0.0761 [+0.0569, +0.0971] sep | FAIL |
| BAR-R6-3 `os − refcv4b` | +0.0797 [+0.0574, +0.1017] sep | +0.0797 [+0.0577, +0.1023] sep | FAIL |
| BAR-R6-4 `os − refcv5-v2` | +0.0681 [+0.0456, +0.0909] sep | +0.0681 [+0.0455, +0.0909] sep | FAIL |
| BAR-R6-5 S6 `os − ha0_ext`, ADE 1–6 s | **−0.4729 [−0.7531, −0.1780] sep** | **−0.4601 [−0.7453, −0.1658] sep** | **PASS** |

* **Inference-seed replicate.** `os(s0) − os(s1)` = +0.0000 [−0.0025, +0.0027]. The planner does sample: the max per-window path difference is 3.33 m, and the mean |Δ| at 2 s is 0.21 m.
* **The ~0.08 m gaps are ~30× that floor.** The failure is not inference noise.

**What the gap to the echo is made of** (paired, seed 0; seed 1 agrees to the third decimal).
* It is spread across both families:
  * along-track +0.0486 sep;
  * speed MAE +0.0495 sep;
  * cross-track +0.0588 sep;
  * heading +0.42° sep;
  * yaw-rate +0.0039 rad/s sep (A3 cell).
* Against refcv4b, speed is not separated (+0.0134). The deficit there is lateral (cross-track +0.0689 sep, heading +0.50° sep) with a small along-track term (+0.0280 sep).

**Levels (seed 0).**

| arm | ADE 0–2 s | cross-track m | tgt-speed acc |
|---|---|---|---|
| `os` | 0.3771 [0.3458, 0.4138] | 0.1668 | 0.810 |
| echo `ha0_ext` | 0.2886 | 0.1079 | 0.866 |
| hold `ha` | 0.3010 | 0.1238 | 0.866 |
| refcv4b | 0.2974 | 0.0979 | 0.832 |
| refcv5-v2 | 0.3090 | 0.0994 | 0.834 |
| CV `ha0` | 0.6764 | 0.3170 | 0.705 |

* `os` beats constant velocity by −0.2993 sep.
* Distance keeping (69 episodes with a lead): min time-gap 4.12 s, min TTC 24.07 s. `os` is closing on 515 windows, against 446 for the echo.

**No programme arm clears BAR-R6-1 on this surface.** Seed 0, S2:
* refcv4b − echo = +0.0088 [−0.0058, +0.0249], not separated;
* refcv5-v2 − echo = +0.0204 [+0.0046, +0.0383], separated worse.

The primary bar is one the programme has not yet cleared with any arm.

**Other readings.**
* **Nav.** Withholding nav costs +0.0246 sep and shuffling it +0.0100 sep. Withholding max-speed costs +0.0020, not separated.
* **T-FLIP: FAIL** at both seeds. follows_FED is 0.25 against a bar of 0.50, but already above refcv5-v2's 0.205. true − shuffled is 0.1534 (seed 1: 0.1335), against a bar of 0.38; refcv5-v2 reads 0.099.
* **OBEDIENCE: FAIL.** This is structural (§3, F8): 2,216 of 2,294 rows have no compliant candidate.

**TACTICAL under both label clocks (SPEC A5).**
* The declared tactical heads are **deterministic in the inference seed**: their logits are bit-identical in the seed-0 and seed-1 dumps. Their readings therefore carry no inference variance.

| head (v6 behaviour decoder) | OLD clock ⭐ primary | CORRECTED clock |
|---|---|---|
| LAT acc [CI] · κ (n in band) | 0.7169 [0.6427, 0.7857] · 0.2418 (1,141) | 0.7134 [0.6392, 0.7821] · 0.2300 (1,106) |
| LAT, nav zeroed | 0.6713 · **0.0066** (the majority rate is 0.6713) | 0.6718 · 0.0068 |
| LON acc [CI] · κ (n in band) | 0.3707 [0.3003, 0.4432] · 0.1821 | 0.3680 [0.2965, 0.4418] · 0.1786 |

* The lateral decision is nav-driven: with nav zeroed, κ collapses to ~0.
* FOLLOW_LANE goal AUROC: 0.7372 (OLD) / 0.7369 (CORRECTED).
* CORRIDOR_OFFSET sits exactly at the n_pos floor under OLD (200) and falls below it under CORRECTED (193 → UNSCOREABLE). SPEED_BAND is positive on every in-band window, so its AUROC is undefined. Every other token is below the floor under both clocks.
* **STRATEGIC: NOT APPLICABLE, n = 0.** The strategic layer is OFF.

**S6 (1–6 s).**
* `os` ADE is 3.0749 / 3.0877 m (seeds 0 / 1).
* It beats the echo (BAR-R6-5 PASS), but is worse than refcv4b (+0.3524 / +0.3652 sep) and refcv5-v2 (+0.3609 / +0.3737 sep).

**⭐ RULE ZERO. BAR-R6-1 failed at step 5000; the next lever, and its result (SPEC A4, reported with no bar).**
* A4 was registered at 12:58, before either seed's lever reading existed. The step-5000 `os − echo` surface had already been seen, so this checkpoint's lever readings are post-registration measurements on a seen surface.
* **L2, the causal-hold blend `w·os + (1−w)·ha`** (w cross-fitted per instant on the other episode-parity fold; w = 0.20 / 0.20–0.25 / 0.30 / 0.40 at 0.5 / 1 / 1.5 / 2 s):

  | cell | seed 0 | seed 1 |
  |---|---|---|
  | blend − echo | **−0.0122 [−0.0200, −0.0042] sep** | **−0.0123 [−0.0200, −0.0045] sep** |
  | blend − `ha` | −0.0246 sep | −0.0247 sep |

  * The shuffled-plan control reads **exactly +0.0000** (w = 0 on both folds), so the gain is not shrinkage.
  * The identity control passes.
  * ⇒ **The committed A4 interpretation fires. refcv6's plan already carries 0–2 s information that a causal kinematic hold lacks, and a cross-fitted composite beats the echo bar that no programme arm has cleared.**
  * The next lever is a **residual-on-kinematic-prior output parameterisation**, a training change. It is blocked on an MM/PI decision.
* **L2e** (echo blend, diagnostic only): −0.0202 sep at both seeds.
* **EXPLORATORY context** (not pre-registered; `raw/step5000/levers/context_baseline_blends.json`): **the L2 gain is generic, not refcv6-specific.** The same cross-fit applied to the banked baselines, on the same windows:

  | arm blended with `ha` | arm − echo | blend − echo | w |
  |---|---|---|---|
  | refcv4b | +0.0088 ns | **−0.0315 [−0.0398, −0.0227] sep** | ≈ 0.5 |
  | refcv5-v2 s0 | +0.0204 sep | **−0.0267 [−0.0350, −0.0178] sep** | ≈ 0.45–0.6 |
  | refcv6 at step 5000 | +0.0885 sep | −0.0122 sep | 0.2–0.4 |

  * Every REF-C arm's 0–2 s error is complementary to the causal kinematic hold, so every one clears the echo bar by composition.
  * ⇒ A residual-on-kinematic-prior output is a **programme-wide** lever, not a refcv6-only one.
  * At step 5000, refcv6's composite is still the weakest of the three.
* **L1** (seed average): `os_avg − echo` = +0.0831 sep, still failing. The sampling term is only 0.0054 m, so inference variance is not the lever.
* **L3** (deterministic DDIM): queued by the watcher for after step 30000.

<!-- STEP30000 -->

<!-- FINAL -->


---

## 5. Defects found: in existing tools, in the data, and my own

| # | what | class | status |
|---|---|---|---|
| F1 | **`taniteval/tools/refcv3_arm.py` cannot roll the full refcv6 arm.** Its model call never passes `ego_poses` or `v_max_ms`/`v_max_valid`, and a refcv6 build **refuses** both omissions (`refc.py:4413`, `refc_v3.py:1937`). Its `load_model` also builds the lift bank **without `equalize_bottom_rows`** (`:1101`) and sets **none of the model-level loss attributes**; M4 above measures what that costs on the tactical-goal BCE (−26 %). The 2026-09-19 E9 "refcv3_arm runs on refcv6" result was a `--size tiny` checkpoint without these channels. | "built, tested, unreachable from its caller" (CLAUDE.md, 2026-09-10) | ⚠️ **ESCALATED.** The battery works around it in-package (`refcv6_roll.py` patches three seams); the durable fix belongs in refcv3_arm, and that is the owner's call |
| F2 | **The dev box's local 416×1024 eval-139 rebuild is NOT the run's input.** 139/139 files differ by md5 from Thor's cache. On 3 sampled clips, `poses`/`actions` are identical but **~67–73 % of pixel values differ by up to 3/255** (`raw/kit_checks/eval139_content_compare_3clips.json`). | a true artifact quoted outside its scope | Only the kit copy (Thor's files) is used. ⚠️ Do not substitute the local build for any refcv6 number |
| F3 | **My loader reset the caller's RNG.** `train()`'s `torch.manual_seed(args.seed)` is replayed in `build_model`, which ran AFTER refcv3_arm seeded `--infer-seed`. So every "inference seed" was seed 0, and seeds 0 and 1 read bit-identical within 4.5e-8 m. | mine; a check that shares the defect | ✅ Caught by the CPU smoke's seed-replicate control **before any GPU roll**, fixed with `torch.random.fork_rng`; after the fix seeds differ by 0.22 m |
| F4 | **My G0 COUNT rule** was a raw substring test for `n_`, which matched `lo[n_]tac` / `tacv6_lo[n_]ce` and printed a FAIL. | mine | ✅ Fixed at 01:07, before the G0 output existed, to the SPEC's own examples (a standalone `n` token). Both verdicts are banked |
| F5 | **M1, my pre-registered deliberate regression, had no power** (0/20 SMOOTH terms moved). | a gate that cannot fail | G0 as registered = **VOID**, reported as such. Amendment A1 (registered before running) added M2–M4, and all three were detected ⇒ G0-A1 PASS |
| F7 | **My orchestrator deadlocked on its own GPU gate.** After the step-1000 seed-0 roll the process held a CUDA context (1,826 MiB), and `gate_wait()` did not exclude its own pid, so it waited on itself: 11 WAIT rows from 03:39 Berlin. It also blocked the sibling NavSim stream. The Master Mind caught it. | mine; a check that shares the defect it checks for | ✅ Stopped by explicit PID. Fixes: (a) `gpu_gate.evaluate()` always drops the caller's own pids; (b) the orchestrator **never initialises CUDA**, because every GPU stage (G0, G0-A1, each seed roll) is a child process (`roll_seed.py`), and `parent_cuda_initialized` is recorded False. `test_gpu_gate.py` 6/6, including a mutation arm that reproduces the defect. The 8 GB RAM rule is unchanged |
| F9 | **The §2 wrapper control conflated merge exactness with bf16 batch-size numerics.** Its 1e-3 bar was never measured. It held at step 1000 (8.4e-4) and read 3.9e-3 at step 5000. | an unmeasured tolerance; a control whose question was wider than its purpose | ✅ Amendment A2, registered before measuring: an fp32-deterministic wrapper probe with its own floor and two deliberate wrapper regressions. The merge is exact (1.9e-6); G0-A2 PASS. The as-registered FAIL is kept and reported |
| F8 | **The frozen OBEDIENCE bar is unsatisfiable on 97.5 % of its population** (§3). | instrument design | ⚠️ **ESCALATED** (owner / PI). Reported, not changed |
| F10 | **The paired yaw-rate cell in the shared `taniteval/tools/refav1_arm.py::_components` is scored on steps with NO path tangent.** `four_families` masks yaw-rate with `pred.pair_valid & gt.pair_valid` (`four_families.py:786-791`); `_components` masks heading but not yaw-rate (`refav1_arm.py:2058`). MEASURED at step 5000 seed 0 (`code/probes/yaw_mask_probe.py`): refcv4b 0.2034 rad/s unmasked vs 0.0318 masked (6.4×; 235 windows > 1 rad/s carry 83 % of the sum, every one with a stopped/crawling GT step); the cell `os − refcv4b` read −0.1683 rad/s separated and is −0.0042 [−0.0075, −0.0011] on valid steps (40× smaller); `refcv4b − ha0_ext` +0.1721 → +0.0073. Every other paired cell is bit-identical before/after. The same `_components` feeds `paired_openloop.py`, `openloop_suite.py`, `stratified_openloop.py`. | a check that shares the defect it checks for (a mask applied to heading, forgotten on its derivative) | ✅ In-package: SPEC **A3** (registered before any unread panel), `LAT_yaw_rate_mae_radps_valid` in every paired table, the unmasked cell labelled DEFECTIVE; `code/test_yaw_valid.py` 3/3 with a mutation arm that re-introduces the defect and is caught. ⚠️ **ESCALATED**: the shared instrument and any landed paired yaw-rate claim (candidates: `2026-09-05-mm-decisions.md` M26 0.2176 → 0.0427 rad/s; `PREREG_REFCV6.md` L1–L4 yaw-rate clause) are the owner's to re-read |
| F6 | **GPU contention.** 7 REFe `augment_search.py` shards (another stream) held python compute on the RTX 4060 23:18–00:00, and the brief's gate correctly WAITED. My first gated launcher then died on an MSYS `/c/` path inside a Python string (the MSYS-path trap). | environment; mine | ✅ Relaunched with Windows paths. No battery number was affected |


---

## 6. Deliverable manifest

`PKG` = `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/`. Repo copies are landed by the Master Mind from `LANDING_READY.txt`. Batch 1 is `9f52118b` on `agent/arch-inf-20260803`.

| artifact | where it lives | only in one place? |
|---|---|---|
| `SPEC.md` (registration + A1–A5) and its sha records `raw/SPEC_SHA256_*.txt` | repo:`PKG` (batch 1) + dev box `C:/Users/Admin/ev6_battery/` | no |
| battery code, `code/*.py`, `code/*.sh` (loader, roll, panel, runner, G0/A1/A2 tools, gate + test, chain, pulls, renderers, sanitizer) | repo:`PKG/code/` + dev box `C:/Users/Admin/ev6_battery/code/` | no |
| model-free pairing proof, kit checks, controls | repo:`PKG/raw/{pairing_surface.json,kit_checks,controls}` | no |
| G0 / G0-A1 / G0-A2 at step 1000 and step 5000 | repo:`PKG/raw/g0_step1000/`, `PKG/raw/step5000/` (+ dev box `raw/`) | no |
| step-1000 pipeline validation (analysis, cross-paired, tactical, S6, acceptance, tables) | repo:`PKG/raw/step1000/` | no |
| battery tags step5000 / step30000 / step15000 / step20000 / final (JSON, logs, tables, `RESULT_SECTION.md`, `TACTICAL_CLOCKS.md`, `levers/`, dump tarball ≤ 19 MiB) | auto-banked to `PKG/raw/<tag>/` by the chains (`chain_milestones.sh`, `chain_optional.sh`, `chain_final_v2.sh`) and `post_tag.sh`, then landed from `LANDING_READY.txt` | dev box only until landed |
| per-window decision sidecars (`dump_s*/decisions/*.npz`, ~25 KB/window) and the panel/S6 dumps | dev box `C:/Users/Admin/ev6_battery/raw/<tag>/` | **YES, dev box only**: derivable from the banked dumps plus the banked baseline dumps; too large to land |
| checkpoints `ckpt_step1000.pt`, `ckpt_5000.pt`, `ckpt_30000.pt`, `ckpt_final.pt` + `MD5SUMS` | dev box `D:/refcv6_eval_kit/ckpt/` (read-only copies of Thor files) | copies; the originals are on Thor |
| SAM3 GT for the 137 eval clips, byte-identical to Thor | dev box `D:/refcv6_eval_kit/data/sam3_gt_eval_thor137/` (+ `_GT_RECORD.json` landed) | copy |
| A3 evidence: yaw-mask probe, A3 recompute records | repo:`PKG/raw/a3/` + dev box `raw/a3/` | no |
| A5 evidence: label tables (both clocks, 23,772 eval windows) + controls C1–C4, tree-equivalence roll, clock census vs the run, resume config | repo:`PKG/raw/label_clock/`, `PKG/raw/a5/` + dev box | no |
| the 82c2331 code tree the FINAL runs on (`git archive` of stack/, taniteval/, tools/, products/P7-TanitEval, lead block, refcv5_compare, sidecar) | dev box `C:/Users/Admin/ev6_82c2331/` | **dev box only**, but it is a pure export of commit `82c2331` (reproducible with `git archive`) |
| clip-clock sidecar (md5 `78466f99…` == Thor == audit package) | dev box `D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl` | copy; the original is landed in the audit package |

