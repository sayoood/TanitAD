# PREREG L2-FRESH-STEPA READ 2 (amendment to `PREREG_L2_FRESH.md`): the FINAL L2 EP head on universe D, the scene-unseen navtrain tokens with a local nuPlan DB

**Status: DRAFT** (written 2026-10-19 by the L2-FRESH-STEPA agent for the Master Mind; it has NO force until a sealed copy exists; no D label or score exists; nothing in it was run on a GPU, a pod or the network).
It is READ 2 of the Master Mind's ruling on READ 1 (`MM_RULING_L2_FRESH.md`, carried verbatim in the sealed `PREREG_L2_FRESH.md` section 1): *"O3 => READ 2 on universe D, under its own amendment, sealed before any D label or score ... READ 2 is prepared NOW, whatever READ 1 shows. On O1 it is the log-fresh confirmation that B cannot give; on O3 it is the next arm."*
Parent: `PREREG_L2_FRESH.md` sha256 `5006ed844979578f4c42599cd79b09c15bd07862d9c9bda94b21481d5e8d4e54` (sealed 2026-10-19T16:25+02:00). Everything not changed here is the parent's (heads, recipe, outcome rule, four families, seal pattern).

## 1. Headline findings of the preparation (MEASURED unless stated)
1. **D is 99 tokens over 84 logs, not "about 86 clusters".** The 103 DB-bearing log-less tokens minus the 4 whose log is one of the 24 held-out logs. `raw/fresh_sample_D_scene_unseen.json`; token-list sha256 `80de7dd150cf2e68b3794cf27d155cae5d23acc0c397fa3aadd07ddb0e49b435`.
2. **D is NOT log-fresh for P2: 83 of its 84 logs are inside P2's PDM-label bank (M-PDM) and inside the final-run target bank (M-SCENE) through OTHER tokens of those logs** (per D log: median 65 M-PDM tokens / 113 sets, median 67 M-SCENE tokens). Token-level overlap is 0 / 0. Exactly ONE D log has no exposure at all (`2021.06.14.18.42.45_veh-12_02520_02585`, 1 token). So D is **token-unseen, LOG-SEEN by P2 (83/84)**. It is log-fresh for the L2 HEAD (its fit logs are the 24 held-out logs; D shares none: `stepA_overlap` tokens 0 / logs 0).
   This is the same structure as candidate C' of the parent (its section 2): any log outside the 24 is inside M-PDM, so no navtrain universe is log-fresh for P2. The parent called D "the only candidate unseen by P2 at the scene level"; that is true of TOKENS only.
3. **The interval will be WIDER than B's, not narrower (ESTIMATED, variance model in `code/readtwo_geometry.py`, from Step A's own per-token terms).** Clustering costs little at D (design effect about 1.02, one or two tokens per log), but D has 99 tokens against B's 400: the A1 half-width at the banked L2 OOF operating point (A1 about 0.826) is **about 0.054** (P2: 0.070), against Step A's measured 0.045. `interval_aware` O1 then needs an A1 POINT of about **0.854** (B: about 0.846). A point near the banked OOF value reads **O3 on D too**. The ruling's reasoning (the binding term is the number of log clusters) is right about the design effect and incomplete about the token count.
4. The bridge, the fetch list and the carve are built and tested (section 6); the D bundle, the labels and the read script are NOT built (section 7).

## 2. The universe (fixed by the carve record; no D label or score read)
| item | value |
|---|---|
| universe | the 251 navtrain tokens whose log the token table does not record (never served to P2's PDM-label loss, absent from `train_grow`); 103 have a nuPlan DB on this box (`raw/logless251_probe.json`, `code/probe_logless251.py`); 148 have none |
| excluded | 4 tokens whose DB log is one of Step A's 24 logs: `62d4aa026e4d5d05`, `8329f21a48755d57`, `8d81371457855252`, `d6e6ab2532535021` |
| carve | **99 tokens / 84 logs** (69 logs with 1 token, 15 with 2); status SHORT: all taken |
| sampler, seed | `stepA.sample_stratified` through `fresh_carve.carve` (mode log_disjoint_c), seed **20261020** (new, recorded; Step A 20260927, B 20261019). n_requested 400 > universe 99, so the seed has no effect; the sampler's draw at n = 99 is asserted to be the identity (`sampler_identity_at_n_equals_universe: true`) |
| token-list sha256 | `80de7dd150cf2e68b3794cf27d155cae5d23acc0c397fa3aadd07ddb0e49b435` |
| overlap with Step A / the L2 fit set | 0 tokens, 0 logs |
| overlap with the plan | none of D's tokens is a held-out plan token (tested) |
| files | `raw/fresh_sample_D_scene_unseen.json`, `raw/readtwo_plan_D.json` (sha256 `9a4c87c004acac9c20a57fa8ae8cce70ddc718ca7f67177a3a732d3ef04039c2`; token_log, db, token_ts_us, planned frame root) |

## 3. Exposure, MEASURED (`raw/readtwo_exposure.json`; sources by file:line)
Sources (the same ones the G0 check used): `code/fresh_carve.py:34` `served_keys.tsv.gz` and `:35` `navtrain_token_table.tsv.gz` (M-PDM tokens = served keys with `heldout_log == "0"` at `:104`; their logs through the token table at `:105`); `:36` `g0_traingrow_scan2.json` and `:106` `token_to_logs` (M-SCENE = `train_grow`); the original check is `l1run/code/g0_overlap.py:5` (SOURCE A) and `:49` (`--scan`).
| level | M-PDM | M-SCENE |
|---|---|---|
| tokens of D in the bank | **0 of 99** (PASS) | **0 of 99** (PASS) |
| logs of D with at least one bank token | **83 of 84** (G0 A1.2 reads FAIL at LOG level) | **83 of 84** |
| bank tokens per D log (min / median / max, sum) | 0 / 65 / 575, 8,020 | 0 / 67 / 582, 8,190 |
| bank sets per D log (M-PDM) | 0 / 113 / 1,025, 14,082 | - |
The deliberate-overlap control reads FAIL as required. **What this licenses:** a statement about P2's exposure, not about the L2 head; the head never fitted on any D log. **Decision D1 for the Master Mind:** declare the log-level G0 A1.2 FAIL a stated exposure for this universe (as Amendment A1.3 did for M-SCENE), with the stamp **"token-unseen, LOG-SEEN by P2 (83/84 logs); log-fresh for the L2 head"**. Until ruled, a READ 2 report must not say "unseen logs".

## 4. The read (parent section 3 unchanged, with these differences)
* Heads: the SAME two final heads of the parent (sha256 `bded124e1322b56d9bfd749f73b22c86430ef41f6829856927e5a08831320df4`, `86f0b20f6ab0d2c46af6b3ae9874b73cc9d94a8065cf7fc0344acbeadcfd4654`); P2 as the reference; the verdict is the most severe of the two seeds.
* **The log-OOF companion does not apply:** D's logs are not among the 24 logs of the fold structure, so a D token has no held-out fold (`compatible_logs` refuses). It is omitted and the report says so.
* **Estimator: the paired bootstrap with clusters = LOGS (84)**, 10,000 resamples, seed 20260927. A log with two tokens is one cluster. It answers "another draw of LOGS" only; one training seed per head (H-ESTIM-SEED-1).
* **Rule: `interval_aware`, verbatim as in the parent section 5.1.** Expected half-width (ESTIMATED; MEASURED half-widths on Step A: 0.073 for P2, 0.045 for the L2 OOF read), by universe size, same cluster-per-token structure, L2 OOF operating point:
| tokens | 99 (D) | 222 (C' size) | 247 (D plus the 148 DB-less, a data job) | 400 | 1,000 |
|---|---|---|---|---|---|
| A1 half-width | 0.054 | 0.036 | 0.034 | 0.027 | 0.017 |
| O1 needs an A1 point of at least | 0.854 | 0.836 | 0.834 | 0.827 | 0.817 |
  (variance model: `HW(m) = 1.96 * sqrt(mean(e^2) / m) / mean(den)` from Step A's per-token A1 terms times the square root of a design effect estimated from Step A's own clustered interval; weak point: D's per-token variance is assumed to be Step A's. `raw/readtwo_geometry.json`.)
* **Stated expectation, before any D label or score:** an A1 point near the banked OOF value (0.83) classifies **O3 on D**; O1 needs about 0.854. A READ 2 that returns O3 does not by itself move the programme.
* A3 reported with the parent's pre-stated consequence (paired dA3 upper CI below 0 on both seeds records an EP-RELIAB-class risk).
* Stamps: "token-unseen, LOG-SEEN by P2 (83/84); log-fresh for the L2 head" (D1), SELECTION tier, one training seed per head. STRATEGIC family PAUSED (ITEM 25), as in the parent.

### 4.1 Next arms (PROPOSED; the Master Mind rules at the seal)
| verdict | next arm |
|---|---|
| O1 | Step B with the L2 head under its own amendment (parent section 5.4); READ 2 is the log-fresh-for-the-head confirmation |
| O2 | L2-BRANCH (parent) |
| O3 | PROPOSED: per L1 section 4 / Amendment 1 C7, "re-run once; if still straddling, treat as O2" applies after READ 1 and READ 2 have both straddled: treat as O2 (L2-BRANCH), or widen only if the MM prefers the data job (247 tokens) or the single navtest look; **decision D3** |

## 5. The fetch (needs PI APPROVAL; the Master Mind asks; this package has no pod access and fetched nothing)
* **What:** the nearest camera frame to each D token's timestamp, per camera: **396 jpgs** (99 tokens x 4 cameras; the probe's 412 = 103 tokens x 4 includes the 4 excluded tokens' 16 jpgs, which D does not need). Maximum timestamp offset 40.07 ms. List: `raw/readtwo_fetch_list.txt` (one `<log>/<CAM>/<hash>.jpg` per line), sha256 of the sorted list `8ab4f8dc95276b3965b00652cbd2c38998dee6988a8df06a53d1d1ca4a9b0acb`; with offsets `raw/readtwo_fetch_list.json`.
* **Size:** about **77 MiB** for 396 files (ESTIMATED: 0.195 MiB per jpg from the earlier pod copy of 5,488 files = 1,069.9 MiB, INHERITED from `p2heldout/HANDOFF_2.md`); the MM's brief says about 85 MB for the 412.
* **Source:** the pod's navtrain pixels, `/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg` (the layout the earlier copy recorded; absence at one directory is not absence: probe `ls -d navtrain_pixels/*/<log>` and `find /workspace -maxdepth 4 -name '<log>*'` as the second path before declaring a file missing; mind CRLF in a stdin list, which once read as "MISSING").
* **Read-only, under the P6 overlap rule (`eval/P6_ON_11VCPU.md` section 9, `refe/p6_pod.sh` `guard_watch`):** the pod may be training the P2 seed replicate. Read with `nice -n 19` and `ionice -c 3` (idle class), CUDA hidden, **batches of at most 120 MB** (the whole list is about 77 MiB: ONE batch), and the **`P6_GUARD` skip**: before each batch take the trainer-step guard (`p6_overlap.py trainer-guard --metrics <replicate metrics.jsonl> --base <baseline> --tol 0.05`); a slow read (ratio above 1.05) twice in a row means the batch is NOT started (skipped, retried after the pause, 600 s doubling to 1 h), exactly as the chain pauses its stage. No readable baseline = no fetch (UNVERIFIED which trainer is live now: not probed). No pod address is written into any file.
* **Landing and proof:** `C:/Users/Admin/par18/l2fresh/frames_D/<log>/<CAM>/<hash>.jpg` (the layout `FrameResolver._find` matches; this package's own directory, no other agent's); sha256 of every file on both sides equal, re-hashed from disk after the copy (the earlier copy's method; tool `p2heldout/copy/podcopy.py`); then the planner's own `FrameResolver` must resolve all 99 tokens (4 frames within 50 ms; the resolver call pattern is in `code/probe_logless251.py`; the check is written with the build, section 7).
* The DBs of the 99 tokens are on this box already (`readtwo_plan_D.json` `db`).

## 6. Gates (L1 / parent gates, plus the bridge)
* **G0:** token level PASS, log level FAIL read as exposure D1 (section 3); control reads FAIL.
* **G2b BRIDGE (BUILT, TESTED on a fake planner, NOT RUN: needs the dev-box GPU):** D has no banked fan, so P2's fan comes from the bundle code, which must first reproduce the banked fan, logits and pick **bit for bit on 3 B tokens** (IDENTITY; tokens `raw/bridge3_sample.json` sha256 `13b153d6c6f4ba47764e5381bd2e2cbeef77e8ff325a486342268511a2189d0a`) with a **RED arm**: one pixel of one channel of camera 0 raised by one 8-bit level must make the gate FIRE on every token (and the perturbation must have been applied once per token, or the bridge FAILS). `code/bridge_g2b.py` runs the bundle script itself (`runpy`, byte-identical; certified: `make_stepA_bundle.py` LF sha256 `8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237`) and re-compares every recorded forward to the banked rows. Tests (`code/test_readtwo.py`, a fake planner through the real CLI): PASS case; a gate blind to the perturbation FAILS; a perturbation that never reaches the forward FAILS; an identity mismatch FAILS. The exact locked command is in `READY.md` section G. `--dry-run` (CPU) already confirms the 3 tokens have banked rows, DBs and resolvable frames.
  **Limit, stated:** the bridge certifies L1's script. The D bundle needs a patched copy (banked lookups skipped for tokens with no banked row); the bridge must be re-run with `--bundle-script <the patched copy>` and the patch must be a pinned diff touching only the banked-lookup lines.
* **G1 (label identity) and the in-fan labels:** for B the true values of the 64 in-fan proposals come from the banked table; for D there is none, so the 64 proposals of every D token and the perturbations must be PDM-labelled by a patched `label_perturbations.py`, with G1 on the bridge tokens (the labeller must reproduce the banked in-fan labels of the 3 B tokens, <= 1e-6, shifted-trajectory control must change the label).
* **G2a, G2c, G3, C-FEAT:** as in the parent and L1.

## 7. What is NOT built (honest list; a day-scale build agent, ESTIMATED from `READY.md` section E)
1. the patched bundle script and its pinned diff; the bridge re-run on it (GPU, locked);
2. the patched labeller for fresh tokens (all 64 in-fan proposals + the perturbations) and G1 on the bridge tokens;
3. `analyse_readtwo.py` (a clone of `analyse_fresh.py` with D's pins, cluster = log, no companion) and the sealed amendment that pins the D carve, the heads, the patched code and the bridge result;
4. the fetch (PI approval) and its sha256 proof;
5. a `launch_gate.py`-style review of the D chain before any GPU hour (PI rule 2026-09-26 applies to training launches; this chain trains nothing, but the reachability discipline is the same: the first run of the patched bundle and labeller on fresh tokens is their first test).

## 8. Decisions for the Master Mind
* **D1** the stamp and the declared log-level exposure (section 3);
* **D2** whether a 99-token READ 2 (expected half-width about 0.054, O3 again if A1 is near 0.83) is worth the fetch and the build, against the 247-token variant (the 148 DB-less tokens need DBs: a data job) or the A1.5(ii) / navtest look;
* **D3** the O3-on-READ-2 rule (section 4.1);
* **D4** PI approval for the fetch (section 5).
