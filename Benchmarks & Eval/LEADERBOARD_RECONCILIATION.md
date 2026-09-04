# LEADERBOARD reconciliation — 2026-08-23

*Companion to `Benchmarks & Eval/LEADERBOARD.md`. Records what changed in the 2026-08-23 rebuild,
every registry↔raw-JSON check performed, and every row that could **not** be verified and why.
**Quotable sources used: `Project Steering/MODEL_REGISTRY.md` and raw eval JSON only.** No number
was taken from `PROJECT_STATE.md`, a weekly report, a changelog or any prose summary.
Scope: **INTERNAL rows only** — NavSim / nuScenes external rows are owned by two sister agents and
are left as a `PENDING` merge skeleton in LEADERBOARD §9.*

---

## 1. Top findings

### F-1 ⛔ TOP PRIORITY — the whole canonical leaderboard was **T0 presented as driving performance**

`Benchmarks & Eval/LEADERBOARD.md` §1–§4 carried **no tier stamp at all**, and §2 was headed
*"Driving capability — TanitEval v2 tier-0 · the standard read"*.

**Verified from source, not from prose:**

- `taniteval/taniteval/rollout.py:184` — `fa = torch.stack([ep.actions[t + window : t + window + fwd_k] …])`
  → the **expert's future actions** are fed into `rollout_decode` (`rollout.py:189`).
- `taniteval/taniteval/rollout.py:151-160` (module docstring, verbatim): *"…is fed the expert's true
  future actions, so the number it produces is a **world-model fidelity** decode of a known control
  sequence … it just may not be quoted as driving or as hierarchy."* It stamps
  `actions_source="expert_future"`.
- `taniteval/taniteval/rollout.py:237` — *"`pc2_pass` will be False here BY CONSTRUCTION."*
- `Project Steering/EVAL_DOCTRINE.md` (BINDING, PI 2026-08-07), T0 row: *"may be quoted as
  'prediction quality' — ⛔ NEVER 'driving performance'"*; rule 2: *"A capability claim … requires
  T1 or better."*

⇒ **Every ADE in §1, §1b, §1c, §2, §3, §4 and §5 of the old page is a TIER T0 number**, and the §2
heading was a doctrine violation. **Fixed:** §0.1 tier block added; §1/§2/§3.0/§4 restamped
`TIER T0 — prediction quality, not driving`; the §2 heading retracted in place. **No number changed.**

**Aggravating factor:** a name collision made the violation easy to miss.
`taniteval.driving` emits `BLOCK = "taniteval.driving/tier0"` and
`surface: "4 waypoints 0.5 s apart (tier-0)"` — that *"tier-0"* is the **metric-suite** tier (sparse
4-knot vs dense), a **different axis** from EVAL_DOCTRINE's T0/T1/T2. The rebuilt §0.1 names the
collision explicitly.

### F-2 ⛔ The primary tier (T1) was absent from the leaderboard entirely

The programme's PRIMARY offline eval — the T1 action-closed-loop campaign of 2026-08-11/12 — had
**no row on this page for 11 days**, although its artifacts are **banked in-repo and md5-verified**.
It is now LEADERBOARD §1a, ahead of the T0 tables.

Its headline is the single most important number on the page and was nowhere:

| arm · surface | tier | ADE dense m [ep-cluster CI95] | source |
|---|:--:|---|---|
| `stage-a-repaired` · `cl` | **T1** | **9.3697** [6.6822, 12.2576] | `…/2026-08-18-v58f-artifact-banking/gates/four_families/ff_stageA_cl.json` |
| `stage-a-repaired` · `ha` (hold-action control) | **T1** | **0.4246** [0.3500, 0.5132] | `ff_stageA_ha.json` |
| `stage-a-repaired` · `ol` | T0 | 0.3659 [0.2926, 0.4521] | `ff_stageA_ol.json` |
| `v5f-30k` · `cl` | **T1** | **23.9837** [21.4420, 26.3470] | `ff_v5f30k_cl.json` |
| `v5f-30k` · `ha` | **T1** | 0.9597 [0.8361, 1.0879] | `ff_v5f30k_ha.json` |
| `v5f-30k` · `ol` | T0 | 0.9397 [0.8162, 1.0679] | `ff_v5f30k_ol.json` |

**The same checkpoint on the same windows reads 0.3659 at T0 and 9.3697 at T1 — 25×**, and a
hold-action control beats both closed-loop arms by 22–25×.

### F-3 ⛔ FORBIDDEN-ESTIMATOR AUDIT — result: **no live verdict rests on `overlapping_holdout_se`**

Checked every internal row. Findings:

- The `heldout ± ci95` column in §1 is **explicitly labelled DEPRECATED** and is decorative — the
  existing convention. **Kept**, per the brief, and the label strengthened.
- `taniteval/driving.py` **refuses** the estimator in code: every `driving_*.json` carries
  `estimator.deprecated_and_refused: "overlapping_holdout_se"`. Verified by reading
  `driving_refb-10k.json`.
- Every point estimate and interval in §1/§1b/§1c/§2/§3.0/§4 is `episode_cluster_bootstrap` (or the
  paired form) — verified field-by-field across **27** `driving_*.json` files.
- **One banned verdict still stands and is NOT decorative:** `planner_beats_cv` on the
  `planner_p2` row. MODEL_REGISTRY §6 states it is *"banned on both sides"* and that its flip **is**
  reachable. The old page printed **✗**. **Fixed:** rendered ⛔ **UNDECIDED**.
- §0's quoted narrowing band was **stale**: *"1.28–2.06× across 10 arms"*. The registry's current,
  larger sample is **1.107–3.100×, median 1.499×, over 27 dumps = 25 distinct arms**. **Corrected.**

### F-4 ⛔ The three-way win/tie/LOST doctrine was being violated by the page's own §1

§0 declares *"win / tie / LOST is three-way on purpose"*, then §1's `beats CV` column rendered
two-way ✅/✗ and printed **✗** for two arms whose own JSON says `favours: "tie"`:

| arm | old page | raw JSON `verdict.ade_vs_cv` |
|---|---|---|
| `refb-10k` | `✗ +0.0005` | `delta 0.0005, ci [−0.0982, 0.0951], separated false, favours "tie"` |
| `refb` | `✗ −0.0252` | `delta −0.0252, ci [−0.1007, 0.0496], separated false, favours "tie"` |

MODEL_REGISTRY §6 already flags `refb-10k` as *"⚠️ TIE (flip)"* and says explicitly that the honest
verdict is *"neither the ✗ this table used to print nor the ✅ `LEADERBOARD.md` prints"*.
⚠️ **Note the registry's own description of the leaderboard is itself stale** — the leaderboard
printed **✗**, not ✅, by the time of this rebuild. Either way the correct rendering is **TIE**.
**Fixed:** the column is now three-way and reads `favours` straight from the JSON. Two further arms
added in this rebuild are also TIE (`flagship-v4.1-10k`, `flagship-v4.2-step4000`).

### F-5 ⛔ A "not evaluated" row that had been evaluated for a month

Old §1 last row: *"Flagship v3enc | — | running | 272.9 M | 🟥 not evaluated"*.

- `taniteval/results/driving_flagship-v3enc-10k.json` → `headline.ade_0_2s.mean = **1.9654**`,
  `lo 1.6556`, `hi 2.2859`, `estimator episode_cluster_bootstrap`, `n_windows 881 / n_episodes 40`.
  The file has been in-repo since the 2026-07-25 re-emission.
- MODEL_REGISTRY §6 ranks it **11** with exactly those values, and its own change-note says
  *"Flagship v3enc is no longer '🟥 not evaluated'"*.

⇒ **the LEADERBOARD was wrong and the registry was right.** Fixed.

---

## 2. Registry ↔ raw-JSON cross-check

**Method.** Every `taniteval/results/driving_*.json` (27 files) was parsed and
`headline.ade_0_2s.{mean,lo,hi,estimator}`, `headline.fde_2s.mean`, `headline.miss_2m.mean` and
`verdict.ade_vs_cv` extracted, then compared to MODEL_REGISTRY §6 and to the old LEADERBOARD §1/§2.
`eff_*.json` was parsed for `<precision>.plan_step.{p50_ms,p99_ms}`.

### 2.1 ✅ Full agreement — registry §6 == raw JSON == rebuilt page (4 dp)

`flagship-30k` 0.4271 [0.3675, 0.4871] · `refc-xl-30k` 0.4714 [0.3896, 0.5556] ·
`refc-base-30k` 0.4728 [0.3835, 0.5699] · `flagship-v16-ab-ft` 0.4375 [0.3423, 0.5501] ·
`refb-v2-30k` 0.5913 · `refc-xl` 0.6048 · `flagship-speed` 0.6152 · `refb-v2-20k` 0.6435 ·
`refb-10k` 0.8372 · `refb` 0.8629 · `flagship-v3enc-10k` 1.9654 · `refa-dinov2` 2.1675 ·
`flagship-nospeed` 3.0175 · `refa-dynin-30k` 3.0471 · `flagship-v2-6k` 5.9396 ·
`flagship-v4.1-10k` 0.8522 [0.7468, 0.9800] · `flagship-v4.2-step4000` 0.9869 [0.8795, 1.1088].
Every one carries `estimator: "episode_cluster_bootstrap"`, `n_windows 881`, `n_episodes 40`,
`n_boot 2000`.

`refc-small-30k` **0.5261 [0.4295, 0.6262]** — verified from
`TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-07-22-refc-small-30k/refc-small-30k.json`
→ `driving.headline.ade_0_2s`. ⚠️ **It has no `taniteval/results/driving_refc-small-30k.json`** — the
only §1-ranked arm whose scored block lives outside `taniteval/results/`. Noted, not a mismatch.

**Floors** (`driving_flagship-30k.json → floor_values`): CV 0.8377 / 1.7406 / 0.3042 / speed MAE
0.4678 / along 1.0955 / cross 1.0089 / heading 6.623 / κ-sign 0.6103 and hold-v0 0.7876 / 1.6521 /
0.2917 / 0.4818 / 1.1040 / 0.9137 / 6.3441 / 0.5119 — **all eight cells of both rows match the page
exactly.**

**Latency** (`eff_*.json → plan_step`): `flagship-30k` fp32 97.3199 / 122.7717, tf32 p50 97.6981,
amp16 p50 123.8325 · `refc-xl-30k` 44.0647 / 44.4441, 27.7808, 20.9993 · `refc-base-30k` 21.7785 /
22.3272, 15.8115, 15.8779 — **all match §5 to the rounding shown.**

**T1 four families** — `ff_stageA_cl.json` reproduces MODEL_REGISTRY §1.14 exactly: speed MAE
9.7291, speed bias +9.3892, along MAE 9.2655, along final bias +18.5801, accel MAE 19.0948, ego
progress 1.7279 / 1.0994, target-speed 0.3398 / 0.5069 / 0.6564, yaw-rate 4.9188, curvature 0.018586
(bias −0.0024), cross 0.7446, TAC lat κ 0.3795, **lon κ 0.0405**, 5-way κ 0.1404, goal bearing MAE
4.8098°, goal range ratio 1.7584. `ff_v5f30k_cl.json` likewise (speed MAE 26.9356, along 23.8965,
cross 0.9993).

**v5f T0** — `i4a_none.json` reproduces MODEL_REGISTRY §1.8 exactly: ADE@2s **0.4011132656**, oracle
**0.1975271977**, sel_gap **0.2035860683**, miss@2m **0.1486946652**, wp4 0.5190916 / 0.2452900,
`seam_norm_ratio_max` 0.099, `wm_canary_ade_2s` 1.2450082. Ablations: zero **7.6492510**, shuffle
**1.2492018** — match §1.14.

**W7-w4r** — `w7_w4r_k32_gate.json`: selected **3.614226**, frozen-in-run **4.415884**, oracle
**0.127303**, threshold 0.4505, PASS false — match §1.14.

**v5.8f CI** — `v58f_rescore_ci.json`: rescorer-top8-kincost **0.4815 [0.3928, 0.5771]**,
frozen-argmax **0.7933 [0.6414, 0.9757]**, both `episode_cluster_bootstrap` — match §1.14
(the registry rounds the interval to [0.393, 0.577]).

### 2.2 ⚠️ Registry↔raw disagreements found

| # | disagreement | evidence | resolution on the page |
|---|---|---|---|
| **M-1** | **MODEL_REGISTRY §1.14 publishes TWO different heading MAEs for the same arm/surface without saying so.** Its T1 table row `stage-a-repaired · cl` reads heading **5.3945°**; its FF-rescore prose reads **3.8776°** | **Both are in the same file.** `ff_stageA_cl.json → four_families.lateral.heading_mae_deg = 3.8776` (pooled over valid steps) vs `→ intervals.metrics.LAT_heading_mae_deg.mean = 5.3945` (per-window, then mean; CI [3.6203, 7.5348]). Same divergence on all six arms (`stageA·ha` 2.4189 / 3.9859; `v5f·cl` 2.7171 / 3.6204; `v5f·ha` 2.9279 / 4.5954; `stageA·ol` 1.3816 / 1.8351; `v5f·ol` 2.1637 / 3.3483) | **BOTH published in §1a.3, each labelled with its reducer.** ⚠️ This is the **same defect class** the LEADERBOARD already documents in §1c ("three intervals REFUSED on purpose" — pooled mean vs mean-of-per-window-means) — but at T0 the interval is *refused*, at T1 it is *published*. Work item **W-5** |
| **M-2** | Registry §6 says `LEADERBOARD.md` prints **✅** for `refb-10k` beats-CV; the page actually printed **✗** | old `LEADERBOARD.md` §1 row 6 | The registry's *description of the leaderboard* is stale; its *verdict* (TIE) is correct and is what the page now prints (F-4) |
| **M-3** | Registry §6 reading 3 quotes flagship tick **103.42 / 93.76 / 104.49 ms** and REF-C-XL amp16 **26.12 ms**; committed artifacts say **97.32 / 97.70 / 123.83** and **21.00** | `eff_flagship-30k.json`, `eff_refc-xl-30k.json`, re-read 2026-08-23 | ⚠️ **Not a new finding** — the registry itself stamps these six figures **"⛔ UNRESOLVED SOURCE (2026-08-03) … Do not re-cite"** and records the same artifact values. The page quotes the artifact and carries the conflict as **W-8** |

### 2.3 Rows that could NOT be verified, and why

| row | why not verified | how it is marked |
|---|---|---|
| `planner_p2` open-loop ADE | **the open-loop CEM arm (`plan_wp`) was never dumped per-window.** The *closed-loop* windows ARE banked (`…/2026-07-26-closedloop-artifact-rerun/raw_windows/p2win_flagship-30k.pt`, 221 win / 20 ep) and both gates were re-decided 2026-08-16 without flipping — but there is no decision-grade open-loop point estimate | ADE cell blank with the reason; `beats CV` = ⛔ **UNDECIDED**; §12 lists the ~400 s of GPU that closes it |
| CTRV / best-of-3 / ego-status floors | not emitted by `taniteval/results/*`; they come from `…/2026-08-02-ctrv-floor/raw/ctrv_readjudication.json` and `bench.py:485-511 / :558` | §0.6 cites the artifact and the emitter line; the two CV forms (0.8248 split-mean / 0.8377 full-set) are both printed |
| §1b re-adjudication deltas, §1c four-families-vs-floors | from `…/2026-08-02-ctrv-floor/raw/{ctrv_readjudication,four_families_vs_floors}.json`; **the raw JSON was not re-parsed cell-by-cell in this pass** (1.3 MB; the pass prioritised the T0/T1 headline tables) | carried forward unchanged with their artifact paths; **INHERITED from the 2026-08-02 block, not re-verified 2026-08-23** |
| §5.5 T2 AlpaSim / low-OOD block, §5.1 deployment block, §7 different-corpus blocks, §8 historical | raw JSONs live under `…/incoming/2026-07-22-*`, `2026-07-23-*`, `2026-07-15-*`; **not re-parsed in this pass** | carried forward unchanged with their artifact paths and their existing retraction notices; **INHERITED** |
| §9 external rows (NavSim / nuScenes / Bench2Drive / competitor envelope) | **out of scope by the brief** — two sister agents own them | §9 is a `PENDING` skeleton; the carried-over NavSim figures are explicitly stamped **INHERITED, not re-checked, replace or confirm** |
| `flagship-v2corpus-30k` | registry status is 🟢 RUNNING with an ETA **26 days in the past**, no completion/final-eval row anywhere, host pod1 reported `/dev/nvidia*` empty. No eval JSON exists | §12.2: **status UNVERIFIED — re-probe before anything quotes or waits on it** |
| `flagship-v1arch-v2bal-30k` on C1 | ⛔ **inadmissible** — 21 of the 40 canonical val episodes are inside its 9,000-clip training pool (`Project Steering/LEAK_v1arch_val_2026-08-05.md`) | only its C4 OOD-val numbers are quoted (§2.6), with the leak stated |
| `refa-ijepa-4brain-speed-15k` | ⛔ **val ~80 % leaked into train** — the number is unusable | no row; listed in §12.2 |
| **v7-tiny line** | ⛔ **no MODEL_REGISTRY row exists.** Its results live in `Project Steering/GOALS_AND_CLAIMS.md` (H-RANK-1…10, H-INIT-1, H-GATE-1/2, H-REP-1) and under `…/Research/2026-08-19-simwam-analysis/` | **deliberately excluded** — quotable-source rule. §11-D + §12.2 **W-12: mint a registry row first** |
| `v6F-SW-30k` | in MODEL_REGISTRY **§12 only**, stamped *"Everything in this section is a WORLD-MODEL DIAGNOSTIC and may NEVER be quoted as driving performance"* | no leaderboard row; §12.2 records that a driving number needs a T1 eval |

---

## 3. Regeneration commands — checked, both still valid

The old page's regeneration block had never been re-verified. Both commands exist:

| command | verified at | note |
|---|---|---|
| `python -m taniteval.runner driving-all` | `taniteval/taniteval/runner.py:496` — `dra = sub.add_parser("driving-all")` | ✅ exists |
| `python -m taniteval.driving --leaderboard` | `taniteval/taniteval/driving.py:1113-1114` → `leaderboard_md()` at `driving.py:1063`, columns at `driving.py:1056` | ✅ exists |

⛔ **But the block was materially misleading and is now corrected on the page:** these two commands
regenerate **the T0/C1 tables only**. §1a (T1), §2.5 (v5f/v5.8f), §2.6/§2.7 (OOD-val) and §5.5 (T2)
come from `taniteval/tools/t1_eval.py`, `taniteval/tools/ff_rescore.py`,
`taniteval/tools/eval_four_families.py` and per-campaign gate emitters. A "regenerate the
leaderboard" that runs only those two lines rebuilds **one tier of three**.

⚠️ Also enforced on the page: **a census over `windows_*.pt` must import `taniteval.dump_census` and
honour `taniteval/results/dump_exclusions.json`** (EVAL_DOCTRINE rule 6, class C126). 27 dumps are
**25 distinct arms**: `windows_overfit_refa-dynin-30k.pt` ≡ `windows_refa-dynin-30k.pt`
(bit-identical on all seven keys) and `windows_refc-v12-identity.pt` ≡ `windows_refc-xl-30k.pt`
(`max |Δpred| = 7.63e-06`). The old page listed neither exclusion in §1 and could double-count.

---

## 4. Four-family coverage after the rebuild

| block | tier | LONG | LAT | TACTICAL | STRATEGIC | work items |
|---|:--:|:--:|:--:|:--:|:--:|---|
| §1a T1 (C3, 6 844 win / 40 ep) | **T1** | ✅ (distance-keeping ⛔) | ✅ all four members | ✅ factored + goal-setting | ⛔ n/a-corpus, n stated | W-1, W-3 |
| §1c T0 vs floors (C1) | T0 | ✅ (distance-keeping ⛔) | ✅ (3 intervals refused) | ⛔ n = 0 | ⛔ n = 0 | W-1, W-3, W-4, W-5 |
| §2.5 v5f / v5.8f (C2) | T0 | ✅ (distance-keeping ⛔) | ✅ | ⛔ | ⛔ | W-1, W-3, W-4 |
| §2.6 v1arch OOD-val (C4) | T0 | ✅ **incl. distance-keeping (n = 2 846)** | ✅ | ✅ κ 0.6033 | ⚠️ measured **and it is a constant predictor** | — |
| §2.7 unicycle line (C4) | T0 | ✅ **incl. distance-keeping (n = 419 LEAD)** | ✅ | ✅ executed-manoeuvre | ⛔ no map | W-3 |
| §1 / §2 / §3.0 / §4 canonical (C1) | T0 | partial | partial | ⛔ | ⛔ | W-4, W-6 |
| §5.5 T2 | T2 | ⛔ | ⚠️ corridor only | ⛔ | ⛔ | W-10 |

**Every ⛔ above is written on the page as `NOT MEASURED — WORK ITEM` with its reason and its n.**
None was silently dropped, and none was fabricated.

⭐ **The single largest structural change:** the old page's §1c already had three
`UNAVAILABLE` family rows, but §1, §2, §3, §4 and §5 had **no family reporting at all** and the ADE
horizon sweep of §1 was, in effect, presented as the result. The rebuild (a) reports all four
families per tier, (b) adds explicit `NOT MEASURED — WORK ITEM` rows wherever a family is absent,
and (c) demotes the ADE tables under an explicit T0 stamp.

---

## 5. Escalations to the orchestrator

1. ⛔ **`MODEL_REGISTRY.md` §1.14 needs a one-line fix** — its T1 table and its FF-rescore prose
   publish two different heading MAEs for the same arm/surface (M-1) without naming the reducer. The
   registry is not editable by this agent.
2. ⛔ **`MODEL_REGISTRY.md` §6's description of `LEADERBOARD.md`'s `refb-10k` cell is stale** (M-2).
   Cosmetic, but it is a registry statement about a leaderboard rendering that has since changed.
3. ⚠️ **`flagship-v2corpus-30k` status must be re-probed** — 🟢 RUNNING with an ETA 26 days past.
4. ⚠️ **The CTRV floor patch has been validated and UNMERGED since 2026-08-02**
   (`…/incoming/2026-08-02-ctrv-floor/`, patch + 11 tests). Until it lands, `driving.py:304` keeps
   auto-scoring every new arm against two straight-line floors.
5. ⚠️ **v7-tiny needs a `MODEL_REGISTRY` row** before any of its numbers can appear here.
6. ⚠️ **Both `build_lead_block.py` citations in the registry point at `tools/build_lead_block.py`,
   which does not exist** — the instrument is at `taniteval/tools/build_lead_block.py`. The registry
   already annotates this in §1.13/§1.14; the leaderboard now states the correct path.

---

## 6. Amendment **2026-09-03** — the SCHEMA GAP, and what would close it

*PI instruction, verbatim: **"Document the results in the leaderboard even if the old includes
partial criteria."** This section is the honest statement of that partiality. ⛔ **No old row was
retro-filled and no number was invented**; the leaderboard's §0.9 carries the same content inline so
a reader of the page alone cannot miss it.*

### 6.1 The gap in one paragraph

The leaderboard's schema was designed around **`ADE + tier + estimator`** and was extended in the
2026-08-23 rebuild to carry **the four metric families**. Two binding rules have landed **since**
every table on the page was written, and the schema has **no column for either**: the
**`loop` axis** (PI 2026-09-02) and the **`ha0` trivial floor** (`D-REFAV1-HA0-ARM`). Three families
remain partly or wholly unmeasured on the older corpora, for reasons that are recorded per cell.
**The result is that the page's numbers are individually correct and collectively under-specified**:
a reader can tell what each arm scored, but not — without §0.8/§0.9 — whether the number was produced
in a loop that closes, nor whether the arm beat the strongest trivial baseline.

### 6.2 The six omitted criteria, and the cost of closing each

| id | omitted criterion | closes with | GPU? |
|---|---|---|---|
| **G-1 / W-18** | the **`loop` column** (open / closed) | a **lookup on the tier**, plus header corrections at 13 sites on the page and `EVAL_DOCTRINE.md:11` (**a separate stream owns the header edits**) | ⛔ **no** |
| **G-2 / W-19** | the **`ha0` column** — constant velocity at the **measured** `v0` | ⛔ **a re-RUN of every T1 arm with `ha0` in the battery.** It cannot be rescored from banked dumps: `ha0` is a rollout, not a statistic | ✅ **yes** |
| **G-3 / W-3** | **STRATEGIC** on every PhysicalAI block | ⛔ **nothing on this corpus** — settled at five probes: no map, no lane graph, no junction label, no route signal, and both available label sources inadmissible. Needs the VLM PH0→PH1→PH2 pipeline or an external corpus | — |
| **G-4 / W-4** | **TACTICAL** on C1 and C2 | a **hierarchy-traversing rescore**; already closed at source for future T1 runs (`t1_eval.py` passes `tactical_from_traj=True`) | partly |
| **G-5 / W-1** | **LONGITUDINAL distance-keeping** on C1, C2, C3 | a **JOIN**, not new data — `obstacle.offline` covers 97.44 % of the corpus and the instrument exists at `taniteval/tools/build_lead_block.py`. Copy the C4 recipe | ⛔ **no** |
| **G-6 / W-5** | a **per-window reducer** for heading / yaw-rate / curvature | implement it inside `four_families`; three C1 intervals are currently **refused on purpose**, and the same hazard is **unflagged at T1** | ⛔ **no** |

⭐ **W-19 is the one that changes readings, not just completeness.** MEASURED on the refcv3 arm
package's own fixture: paired **`ha − ha0` = +0.1330 [−0.0502, 0.2326], NOT separated**. `ha` holds
the last **observed** `(a, κ)`, which drifts even where the human drives straight ⇒ **an arm can beat
`ha` by doing nothing at all.** §1a's headline — *the hold-action control beats the repaired arm 22×
and the v5f arm 25×* — is therefore stated against a floor whose **own** margin over the true floor is
unknown. The direction of that headline is not in doubt; its **magnitude against `ha0` is unmeasured**.

### 6.3 What was ADDED on 2026-09-03, and what was deliberately NOT added

| added | where |
|---|---|
| §0.8 — the `loop` column + **13 mislabelled sites enumerated with file:line** | `LEADERBOARD.md` |
| §0.9 — criteria coverage: the six omitted criteria + **row-by-row quotable-as-is vs needs-re-measurement** | `LEADERBOARD.md` |
| §1d — the **B1-corpus line**: refcv3 identity + admissibility, the four families **PENDING with the filename that fills them**, the T0 in-training read with the three reasons it may not be promoted, and refav1's step-1000 T1 read | `LEADERBOARD.md` |
| W-18 … W-22 | `LEADERBOARD.md` §12.1 |

⛔ **NOT added, deliberately:**

1. **No refcv3 family number.** `taniteval/results/refcv3-30k-openloop-*.json` **does not exist** —
   verified by two differently-bound probes (`ls`+`grep`; .NET `Directory.GetFiles`), 105 files in
   `taniteval/results/`, **zero** matching `refcv3|refav1|openloop`. The cells say `⏳ PENDING` and
   name the file.
2. **No header corrections.** Every *"closed loop"* mislabel is listed with file:line and **left in
   place** — a separate stream owns it, and two streams editing the same 13 sites is a merge conflict.
3. **No retro-fill of any old row.** An unmeasured cell stays unmeasured.
4. **`eval_traj` was NOT put in an ADE column**, although it is refcv3's only trajectory-shaped
   number. From source: it is a **mean L1 per COORDINATE** (`refc_v3_train.py:462–465`), not an L2
   norm; and it is scored on the **GT-nearest** anchor `a_star` (`:457–459`, `:463`), which the model's
   own selection matches on only **57 %** of windows. It is a **loose lower bound**, not an ADE.

### 6.4 A refusal that no later work can undo

⛔ **refcv3's in-training eval can never carry a confidence interval** —
`refcv3_epoch_read.json → estimator_refusal.ci_available = false`. Every `eval_*` value is **already**
the pooled mean over the 160 held-out windows; the file carries **no per-window values and no episode
index**, both of which `taniteval/ci.py` requires. **No estimator can recover an interval from it**,
so every statement derived from that series is a **DIRECTION, not a verdict** — and the programme's
binding paired-episode-cluster-bootstrap rule is **unsatisfiable by that artifact**. The fix is
**forward-looking only**: the in-training eval must dump per-window values with episode ids (W-21).

### 6.5 Escalations added by this amendment

7. ⭐⚠️ **REGISTRY — the claim I wrote went STALE INSIDE THE HOUR, and the correction is the finding.**
   At **21:2xZ**, three differently-bound probes (Bash `grep -n -E`, PowerShell
   `Select-String -LiteralPath`, .NET `File.ReadAllText` + `Regex.Matches`) agreed on **0** hits for
   `refcv3` / `refc_v3` / `refc-v3` / `refav1` / `refa_v1` / `REF-C v3` / `REF-A v1` across all
   **381,371** characters of `MODEL_REGISTRY.md`, and I wrote *"no row for either arm"*. At
   **21:5xZ** a re-probe read **385,121** characters and **7 / 10 / 3** hits: a sibling stream had
   minted **§4.5 `refcv3-b1-v72-30k`** while the section was being written.
   ⇒ **refcv3 HAS a registry row; `refav1` still does not** (0 hits on the re-probe), so **W-20
   stands, narrowed to refav1 alone** — same class as W-12 (v7-tiny).
   ⭐ **The corroboration is worth more than the correction:** §4.5 was written independently and
   reaches the **same pending verdict** as §1d — one-shot / no action input read from
   `refc_v3.py:480`, the B1 non-parity corpus, the **ORACLE** nav caveat, the `ha0`-margin framing,
   the UNRULED tier, all four families **NOT MEASURED**, and the *same* artifact filename
   (`taniteval/results/refcv3-30k-openloop-*.json`) named as the thing that will close them.
   ⚠️ **The lesson is the standing one — the repo advances mid-session.** Re-probe before shipping an
   absence claim, and most of all when the claim is being written **into a page about stale absence
   claims**. Recorded here rather than silently corrected, because the silent version teaches nobody.
8. ⛔ **The tier ruling for refcv3 is OPEN and it is the PI's** (BACKLOG R30): does the doctrine admit
   at **T1** a model that consumes **no actions at all**? §1d stamps `T1 / status: UNRULED` and frames
   every read as a **margin over `ha0`**, which keeps the numbers correct either way.
9. ✅ **The apparent parameter-count conflict is RESOLVED — and neither number was wrong.**
   `config.json → param_breakdown.total` reads **107,032,901**; the tasking brief stated
   **107,082,365 / 544 tensors**; Δ **49,464**. `MODEL_REGISTRY.md` §4.5 measured both and reconciles
   them: **107,082,365 is the ALL-TENSOR element count over 544 tensors, of which 201 are BUFFERS
   totalling exactly 49,464 elements** ⇒ `107,082,365 − 49,464 = 107,032,901` **parameters**.
   ⭐ **Quote 107,032,901.** *(A worked example of the standing scope rule: two true measurements of
   different objects read as a contradiction. The fix was to find what each one counted, not to pick
   a winner.)*
10. ⚠️ **refcv3's nav token is an ORACLE** — `config.json` states
    `nav_cmd_derivation = "v7.2 nav_command token (oracle, provenance ego-future; allow_oracle_nav=True)"`.
    Under the binding rule that a supplied route is optimistic by construction, **every refcv3 number
    carrying nav is an optimistic bound**, and `os_navshuf` / `os_navzero` are what bound that
    optimism. ✅ The goal path is clean on the other axis: `goal_provenance` records
    `contains_situation_classifier_output: false`.
