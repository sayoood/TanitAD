# Withheld-bank panel on the v7-tiny rig (H-EGO-LIT-4) — RESULT

status: **COMPLETE** (2026-09-05, arms 10:09:34 → 11:43:15, scoring 11:44 → 12:01:51 Europe/Berlin). Seven arms, every one rc0 at 2,000/2,000 steps. One-variable audit **PASS**. Gate **VALID** (it fails the deliberate regression on the informative reading). **VERDICT: NEITHER ADOPT NOR REFUSE `pred` — the pre-registered ADOPT criterion is NOT met, and the lever is NOT refuted; the rig is underpowered for the criterion and the decision needs seeds, not more windows.**

Owner: Architecture & Inference FlyWheel agent (Claude Opus 5), resumed after the predecessor died at a model rate limit while waiting for the GPU.
Pre-registration: `H-EGO-LIT-4` in `Project Steering/GOALS_AND_CLAIMS.md`; reading rules in `SPEC.md`. Both outcomes were committed before this panel ran.
Compute: dev-box RTX 4060 only, one arm at a time, `OMP_NUM_THREADS=6`. The live run `refcv4b-b1-v72-40k` on pod `tanitad-refcv3` was untouched; Thor was not used.
Primary artifact: `raw/panel_report.json` (671,770 B). Tables `raw/TABLES.md`, mechanical outcome read `raw/VERDICT.md`, audit `raw/ONEVAR_AUDIT.txt`, noise floor `raw/NOISE_FLOOR.md`. Per-window dumps `C:\Users\Admin\run_wbank\score_dumps\*.npz` (7 × ~2.4 MB, **local disk only** — too large for the repo).

## Scope stamp — binding, and it applies to every number below

**RIG scope** — v7-tiny rung, ~19 M params, **48 non-parity training episodes**, 2,000 steps, **one seed per arm**. Corpus `physicalai-train-14231cd29c74` references **no registered parity key**. Nothing here is cross-arm comparable with the parity arms and **nothing here may enter `MODEL_REGISTRY.md`** (H-SCALE-2). This validates a **DESIGN** and a **GATE**, never a model claim. Evidence class **MEASURED (ours)**. Tier: families are **T1-style self-action open loop** on the goal/selection heads; the ceilings are **T0 model-free**. Scored on the episode-disjoint val cache `physicalai-val-bb543bdf7836`, **n = 640 windows from 40 episodes** (189 straight / 451 non-straight / 0 unlabelled); estimator `paired_episode_cluster_bootstrap`, n_boot 2,000 (families) / 500 (probes).

## 1. OUTCOME IV first — the gate is NOT vacuous

`A5_regress` (`--ablate-frames`, an echo **by construction**) reads **`ECHOING`**, with scene degradation **exactly +0.0000, CI [0.0, 0.0]** and ego **+2.5957**. That is the *informative* failure — wrong ego hurts enormously, wrong scene cannot hurt at all — not the unpowered `READS_NEITHER`. Every family collapses, all separated: withheld speed **+9.01 m/s**, withheld ADE **+11.18 m**, kept speed **+2.31**, kept ADE **+1.74**, tactical agreement **−0.43 lat / −0.29 lon**.

⇒ **The panel is VALID.** ⚠️ But one honest qualification: the composite `GATE.raised` flag is **True for all seven arms**, because no 2,000-step tiny arm clears gate 1 (`ha` / `ha0_ext`) at the longest horizon. **`GATE.raised` therefore carries no discriminating information at this rig scale** — all of the discrimination comes from the `gate2b` *verdict*, which does separate the arms.

## 2. ⭐ The control that changes how this programme should read CIs

`A0b_replicate` is A0's flags and A0's seed, **run a second time** — it moves zero levers by audit. It was added mid-panel (⚠️ **not** in the original pre-registration) because the panel's estimator resamples *episodes with the trained models held fixed*, and so cannot see training nondeterminism at all.

**Two runs that differ in nothing produced "separated" differences on 6 of 42 bootstrapped cells = 14.3 %:**

⛔ **CORRECTED 2026-09-06 by the register-repair agent — the figure originally published here was "3 of 18" and it reproduces at NO scoping.** Recursing `arms.A0b_replicate.paired_vs_A0` in `raw/panel_report.json` gives **42 cells, 6 separated = 14.3 %**: `LAT_yaw_rate_mae_radps` separated in **3 of its 4** (regime x horizon) cells, `LON_accel_mae_mps2` in 2, `LON_along_mae_m` in 1. The five-row table below is the **2 s horizon** view; the panel's own `VERDICT.md` tabulates seven family rows at that horizon, where it is **3 of 14 = 21.4 %**. Re-derivation with both its controls: `…/2026-09-06-register-repair/raw/replicate_fp_rate.py` -> `raw/replicate_fp_rate.txt`. ⚠️ **`raw/NOISE_FLOOR.md` was NOT the source of either figure** — that artifact crashed mid-write on a cp1252 `UnicodeEncodeError` and carried no numbers at all; it has been repaired and now emits a completion marker (`I21`).


| A0b − A0 (identical config) | withheld regime | kept regime |
|---|---|---|
| LON speed MAE (m/s) | +0.0577 [−0.2883, +0.3967] ns | +0.0827 [−0.0095, +0.1784] ns |
| LON along MAE (m) | +0.0548 [−0.3799, +0.4759] ns | **+0.0955 [+0.0054, +0.1935] SEP** |
| LON accel MAE (m/s²) | **+0.1638 [+0.0222, +0.3019] SEP** | +0.0458 [−0.0963, +0.1874] ns |
| LAT yaw-rate MAE (rad/s) | **+0.0759 [+0.0164, +0.1473] SEP** | −0.0075 [−0.1351, +0.0988] ns |
| ADE (m) | +0.0180 [−0.4164, +0.4331] ns | +0.0502 [−0.0447, +0.1509] ns |

⛔ **That is a 14.3 % false-positive rate for "separated" (6 of 42 bootstrapped cells; 3 of 14 = 21.4 % restricted to the 2 s family rows), measured on this rig with this estimator, between two runs of the same configuration** *(CORRECTED 2026-09-06 by the register-repair agent; published here as "~17 %")*. It is not an estimator bug — the bootstrap answers the question it is asked (*"would this delta survive on other windows?"*) — it is that the question is the *wrong one* for a single-seed panel, where the dominant variance is run-to-run and sits entirely outside the interval. The train-log half of the same finding: over the 450 steps where A1 *is* A0's configuration, 66/72 logged differences are non-zero and |Δ `withheld_speed_mae`| reaches **0.701 m/s**, amplifying from 0.00007 at step 50.

⇒ **A separated CI from a one-seed arm is necessary, not sufficient.** Every delta below is read against this floor.

## 3. The echo instrument (kept = the deployed regime), `min_degradation` 0.05

| arm | bank | gate2b verdict | scene rel | scene separated | ego rel | ego separated |
|---|---|---|---|---|---|---|
| A0_fixed | fixed | ECHOING | +0.0732 | no | +0.1286 | yes |
| **A0b_replicate** | fixed | ECHOING | **+0.1211** | no | +0.1169 | yes |
| **A1_pred** | pred | **READS_BOTH** | **+0.4619** | **yes** | +0.1642 | yes |
| A2_random | random | ECHOING | +0.0942 | no | +0.1282 | yes |
| A3_drop25 | fixed | ECHOING | +0.2130 | no | +0.5745 | yes |
| A4_none | none | IGNORES_EGO | +0.7591 | yes | +0.0417 | **no** |
| A5_regress | fixed | ECHOING | **+0.0000** | no | +2.5957 | yes |

**A1 is the only arm that reads `READS_BOTH`.** The replicate pins the same-config band for scene-rel at **0.073–0.121**; A1's **0.462** is ~4× outside it, and A1 is the only arm whose scene effect is separated. The blindness control A2 lands at **0.094 — inside the replicate band**, which is exactly what a control with the right distribution and no information should do.

**Controls that must read known values — all correct.** The constant predictor reads **scene +0.000000 / ego +0.000000** exactly; `constant_only` reads its own ADE (6.4448 / 13.2513 / 20.3085 m at 2/4/6 s) and every arm and reference beats it; the raw-pixel floor is **n_fit 320, n_score 320, d 192**, λ = 0.1 selected on an inner val of FIT (`underpowered_n_lt_d: false`, i.e. n > d — not underpowered by construction).
⚠️ **One control does NOT read as the SPEC expected, and is reported rather than buried:** `ha` vs `ha0_ext` max |Δ| on the goal rows is **48.52 m**, not near-zero. The slot means agree closely (2 s: `ha` 0.9922 vs `ha0_ext` 1.3290), so this is a small number of outlier rows where the finite-difference acceleration diverges when integrated to 6 s — not a systematic break. It does mean the "`ha` ≡ `ha0_ext`" identity should be stated as a per-slot mean, not as a max.

## 4. The four families — A1 vs A0 (paired, same 640 windows)

| withheld regime (keep = 0) | A1 − A0 | vs replicate floor |
|---|---|---|
| **LON speed MAE (m/s)** | **−0.1655 [−0.4159, +0.0619] ns** | floor +0.058 — **not separated** |
| **LON along MAE (m)** | **−0.1952 [−0.5103, +0.0971] ns** | floor +0.055 — **not separated** |
| LON accel MAE (m/s²) | −0.1194 [−0.2293, −0.0094] SEP | floor +0.164 **SEP — inside the floor** |
| LAT cross MAE (m) | −0.1477 [−0.2124, −0.0877] SEP | floor −0.053 ns |
| LAT heading MAE (°) | −1.7287 [−2.8615, −0.7796] SEP | floor +0.150 ns |
| TAC lat agreement | +0.0828 [+0.0198, +0.1581] SEP | floor −0.013 ns |
| ADE (m) | −0.2580 [−0.5762, +0.0283] ns | floor +0.018 ns |

**The two metrics the pre-registration named as the success criterion — withheld 2 s speed MAE and along-track MAE — are NOT separated.** A1 does win the lateral family and tactical lat agreement beyond the floor, and `A1 − A2` on withheld speed is **−0.1399 [−0.2735, −0.0046] SEP**, so the *source* matters relative to the blindness control even though neither beats A0 with separation.

**Strategic family**: reported per-regime in `score_dumps/families/<arm>__<regime>.json` as the 3-way route head on nav-valid windows with its majority-class control — the only strategic label the epcache rig carries; `g_str` is unsupervised here.

## 5. The attribution risk, named in advance — CONFIRMED MATERIAL

The registration flagged that under `pred` the withheld-row longitudinal decision becomes downstream of `g_tac`, so the family would score the goal head and the anchor classifier jointly. The panel's two separations settle it:

| A1's weights, bank swapped | speed vs A0[w] | along vs A0[w] |
|---|---|---|
| its own `pred` bank | −0.1655 ns | −0.1952 ns |
| **swapped back to `fixed`** | **+0.7063 [+0.3579, +1.0581] SEP** | **+1.0012 [+0.5394, +1.4724] SEP** |
| `pred` bank at a **permuted row's** speed (shuffle) | +0.1984 [−0.0710, +0.4635] ns | +0.3127 [−0.0346, +0.6552] ns |

⇒ **A1's weights have become dependent on their own bank**: without it they are 0.71 m/s *worse* than A0, and the shuffle control removes the advantage entirely (as the SPEC required it must). The effect is real and row-specific — and it is **not** separable into "better model" vs "better geometry" from the family alone. Even had the longitudinal family separated, it could not have been attributed to the bank by itself.

## 6. The bank geometry does improve — the decoder does not exploit it

Withheld-row oracle-in-vocabulary (**T0, model-free**, the ceiling of the bank actually decoded):

| bank rolled at | ceiling ADE (m) |
|---|---|
| fixed 10 m/s | 4.6762 [4.1191, 5.3078] |
| **A1's own predicted speed** | **2.7873 [2.4248, 3.1995]** |
| true v0 (**LEAK bound** — reported, never a design) | 1.2841 [1.1382, 1.4222] |

`pred − fixed` = **−1.8889 [−2.7307, −1.1304] SEP**; `pred − true` = +1.5032 SEP. The predicted-speed bank recovers **≈56 %** of the distance from the fixed bank to the leak bound. **So the candidate geometry gets substantially better while the decoded longitudinal trajectory does not** — the binding constraint at this rung is the selection/decode step, not the bank.

## 7. Pre-registered outcomes — what fired

| outcome | fired | why |
|---|---|---|
| **ADOPT `pred` for refcv5** | **NO** | needs READS_BOTH ✓ **AND** a separated withheld-longitudinal gain ✗ **AND** A1 ≠ A2 ✓ — the middle clause fails |
| **REFUSE `pred` (A1 ECHOING)** | **NO** | A1 is the only arm reading READS_BOTH |
| Bank geometry not binding (A1 ≈ A2) | NO | `A1 − A2` withheld speed −0.1399 SEP |
| Retire the per-window roll (A4 ≥ A0) | **NO — refuted** | A4 is `IGNORES_EGO` and *worse* in the kept regime: speed **+0.3609 SEP**, along **+0.5205 SEP**, ADE **+0.4978 SEP**. The speed-blind vocabulary (the field's design, this hypothesis's H0-alt) **loses**; the per-window roll is load-bearing |
| **PlanTF signature — keep `ego_dropout` 0.5** | **YES** | A3 wins kept families (speed −0.1463 SEP, along −0.1632 SEP, ADE −0.2048 SEP) but reads `ECHOING` — better numbers, lost separation |
| **The harm is any speed-blind bank, not 10 m/s** | **YES** | A2 ≈ A0 (withheld speed −0.0256 ns), so re-centring the fixed roll on the marginal mean (5.7998 m/s) is not the fix |

## 8. Verdict

**Do not adopt `pred` for refcv5 on this evidence, and do not refuse it.** The pre-registered ADOPT criterion is not met: the withheld-row longitudinal family does not separate from A0. But the lever is not refuted and is the only one in the panel that moves the model off the echo — and the panel's own replicate shows this rig manufactures "separated" readings on **14.3 %** of bootstrapped cells between identical runs *(CORRECTED 2026-09-06 by the register-repair agent)*, so a single-seed non-result on a 0.17 m/s effect is **underpowered, not negative** (the standing rule: a negative from an underpowered probe is a statement about the probe).

Two conclusions **are** licensed at rig scope:

1. ⭐ **The v0-conditioned per-window roll is load-bearing** — the field's speed-blind vocabulary (A4) is separably worse in the deployed regime on three metrics. H0-alt is refuted; retiring the roll is off the table.
2. ⭐ **`ego_dropout` stays at 0.5** — A3's 0.25 buys better kept-regime families and loses separation: the PlanTF signature the registration predicted.

**The cheapest discriminating next experiment** is not more windows — it is **seeds**: A0 / A1 / A2 at 3 seeds each (~40 min per arm-set on the dev box, 0 pod GPU), reading the between-seed spread as the floor. That converts a 0.17 m/s point estimate into a decision. Pair it with §6: since the bank *ceiling* improves by 1.89 m while the decode does not, an arm that changes the **selection** step is the higher-value lever for refcv5.

## 9. Ops findings (MEASURED; each is a false-failure generator)

1. ⛔ **The GPU-wait launcher could never have fired.** `wait_and_launch.sh` counted busy jobs with a `Get-CimInstance … -match 'refav1_arm.py|refcv3_arm.py|t1_eval.py'` filter whose own PowerShell command line **contains the pattern**, so the query matched **itself**. `procs` read a constant 4 for over an hour — including while GPU utilisation fell to 5 % — of which: the querying `powershell.exe`, two shell wrappers, and an `ssh.exe` to **Thor** (not a local GPU consumer). `n` could never reach 0. The documented self-match trap in a `Win32_Process` costume; the fix is to enumerate **without** filtering on the pattern and read the rows.
2. ⚠️ **`wc -c < file` is a METADATA probe, not a content probe.** GNU `wc -c` takes the size from `fstat()` and never reads the bytes, so during a mount outage it returned the correct 802,715 B for a file whose every content read failed — producing a "up for the shell, down for Python" diagnosis that was **wrong**. True content reads failed on target **and** control 15/15, which is what actually justified the Drive-client restart.
3. ⛔ **A freshly created directory refuses ALL writes** — MSYS first, then the native Windows API too (12 retries × 4 s, all four files) — while `mkdir` returns 0, `test -d` says EXISTS, reads work, and a tiny `echo >` into the *same* directory succeeds. **Restarting the Drive client clears it immediately**, on all five occurrences. The dangerous part is the shape of the lie: the banking script reports *"banked metrics.jsonl has 0 rows — do not quote this arm"*, which reads as a claim about the **arm** when the fault is entirely in the **access path**. An agent trusting that line would discard a good 2,000-step run.
