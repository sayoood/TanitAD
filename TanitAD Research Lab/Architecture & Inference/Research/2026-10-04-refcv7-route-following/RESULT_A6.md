# RESULT A6 — the deployable time-localised nav (announced entries, soft rule) CLEARS the bar on a dense capture of the held-out episodes

**Status: COMPLETED after A7.** A6 as registered STOPPED at its gate G1 (Part B below, kept unchanged); `SPEC_ADDENDUM_A7.md` (sha256 `8e846a84…`, 2026-10-04T10:42:57Z)
replaced G1 by G1′ and changed nothing else. Part A is the run that followed.

**Stamp on every number in Part A.**
* **Evidence class:** MEASURED. Source `raw/a6_dense_score.json` (every arm, both draws, every control), tables generated from it by `code/make_tables_a6.py` →
  `raw/TABLES_a6_generated.md`; G1′ in `raw/a7_g1prime.json`. Code `code/nav_ann_a6.py`, `code/announced_a6.py`, `code/run_route_a6.py`.
* **Tier:** OPEN-LOOP single-shot planning on logged frames of the held-out eval139 clips; never closed loop (EVAL_DOCTRINE). Not a driving-performance claim.
* **Model:** refcv7-r101-s0, step 50,400, **one training seed**; Thor `ckpt.pt` (md5 `d5f104ee…`, strict load, step asserted) — the file that produced A5's `eval_s0g`/`eval_s1`.
  The brief's md5 `b418d0fc…` is the MODEL-ONLY export `ckpt_50400.pt` of the same step (D: `refcv7_eval_kit/ckpt/MD5SUMS`); that the weights are the same is shown by the
  overlap control below (214 windows bit-identical to A5's capture), not by comparing the two files.
* **Nav is an ORACLE** (v8 `nav_30s`, ego-future) restricted to ANNOUNCED entries (the nav_command builder's rule, PI 2026-08-29 suppression included).
* **Intervals:** paired episode-cluster bootstrap vs V0, B = 2000 (the SPEC §5 draws). It answers *"another draw of EPISODES"*; the seed-1 column answers the INFERENCE question;
  TRAINING variance is not measured (`H-ESTIM-SEED-1`). **These are the SAME 139 held-out episodes as A5 — denser windows, not new episodes.**
* **Scope of the claim A6 can support** (A6 SPEC): *"confirmed on a denser capture of the same held-out episodes"*, not *"replicated on new episodes"*.

## A.1 Bottom line

1. **G1′ PASSES:** (1) the builder's rule on `manoeuvre_sequence[0]` reproduces `nav_command` on **4,719 / 4,719**; (2) `announced(entries[0])` reproduces the side on **1,799 / 1,799**
   of the records where `entries[0]` is `seq[0]` and `time_s ≤ 30` (1,655 turn commands, 144 FOLLOW commands), and on **68 / 68** suppression-named records. 2,920 records are outside
   the population (their `entries[0]` is not `seq[0]`); 0 are excluded by `time_s > 30`.
2. **Capture:** **4,634 windows** over all 139 episodes = **2,317 GT-turn windows (every eval window with |terminal heading| ≥ 30°: 849 left, 1,468 right)** + 2,317 seeded (seed 0) random
   others (1,347 straight, 276 gentle, 694 unclassified), sampler seeds 0 and 1. **Wall 2,072.8 s + 2,061.0 s** (GPU forward 1,804.8 s / 1,795.1 s; each pass includes a ~127 s model build);
   one `setsid nohup` job, GPU lock held 10:48:46 → 11:57:46 UTC (69 min), released at exit. ETA given from the first 50 windows: 23 s ⇒ 0.46 s/window ⇒ ≈ 35 min per seed (actual 34.5 min).
3. **T3a (REPORTED ARM) CLEARS all four criteria.** Seed 0: turn ΔADE **−0.131 [−0.207, −0.062]**, turn dir-correct **+0.064 [+0.034, +0.104]**, straight ΔADE **+0.002 [−0.008, +0.012]**,
   all-window ΔADE **−0.076 [−0.116, −0.036]**. Seed 1 (sampler replicate): turn ΔADE **−0.127 [−0.203, −0.057]**, dir-correct **+0.065 [+0.032, +0.107]**.
4. **T3a-c (control) FAILS as required:** turn ΔADE −0.011 [−0.085, +0.045], dir-correct −0.004 [−0.025, +0.015], all-window −0.005 [−0.049, +0.030]; seed 1 turn −0.005 [−0.058, +0.043]
   (criteria N/Y/Y/N). Paired, the true announced series beats the donor series: turn **−0.120 [−0.230, −0.008]** (seed 1 −0.123 [−0.219, −0.031]).
5. **Reading-rule branch: "T3a passes, T3a-c fails"** ⇒ a deployable time-localised nav lifts route following at inference on refcv7 as trained, **confirmed on denser windows of the same episodes**;
   it ships as an opt-in inference rule and L2 enters refcv8 (the rule's own wording).
6. **Size, honestly:** T3a captures **0.270** of the B1t bound (dADE_turn −0.131 / −0.486 recomputed on this capture; per-draw ratio 95 % [0.151, 0.449]); on the 1,112-window A5 grid scale
   (B1t −0.566) it is 0.232. Re-weighted to NATURAL window frequency (post-hoc), the all-window ΔADE is **−0.015 [−0.026, −0.005]** m (seed 1 −0.014 [−0.026, −0.002]); the bar's
   "all windows" of −0.076 is a property of the turn-enriched sample (50 % turn windows).
7. **The secondary arms:** T3 (foil, all entries) also CLEARS (turn −0.159 [−0.254, −0.080]); **T2a (hard filter, announced) CLEARS on the dense capture** (turn −0.241 [−0.461, −0.031], seed 1 −0.239
   [−0.453, −0.032], straight +0.009 [−0.000, +0.020]) — A5's T2 failure on 107 turn windows was a power problem, not an absence — but its CI is ~3× wider than T3a's and it FAILS on A5's own 214 windows (below).

## A.2 Controls (all PASS before any arm was scored)

| control | measured | verdict |
|---|---|---|
| G1′ (1): builder rule on `seq[0]` vs `nav_command` | 4,719 / 4,719 | PASS |
| G1′ (2): `announced(entries[0])` where `entries[0]` is `seq[0]`, `time_s ≤ 30` | 1,799 / 1,799; suppression-named 68 / 68 | PASS |
| seed-0 / seed-1 / window-list windows identical | n = 4,634 | PASS |
| selection-time GT (CPU, from poses) == the GT the forward captured | max abs diff 0.0 m, 0 validity mismatches (n = 4,634) | PASS |
| GT-turn windows captured == selected | 2,317 == 2,317 | PASS |
| **overlap identity with A5's capture** (214 windows in both sets; same per-window seed) | seed 0 vs `eval_s0g` and seed 1 vs `eval_s1`: fan, score, GT max abs diff **0.0**, `sel_idx` mismatches **0** | PASS |
| capture identities C1 (traj == fan[pick], E9 / decoder argmax recomputed, forwards per window) | 0.0 / 0 / 0 / 1 on both passes | PASS |
| K0 join: bank `nav` == record `nav_command` side | 4,634 / 4,634 (both seeds) | PASS |
| K1 clock vs the replay bank's `t_label_s` (A5, re-run) | 5.0e-5 s over 2,059 windows | PASS |
| K2 nav_tl == clip token at t_rel ≈ 0 (now on **dense captured** windows) | **11 / 11** (+ 17 / 17 supplementary) | PASS |
| T3a-c derangement | the same seeded derangement as A5's T2c (digest `968842…`, 0 fixed points) | – |
| G: guard | 0 modules imported from G: in `raw/a7_g1prime.json` and in `raw/a6_dense_score.json` | PASS |

## A.3 The windows and the nav

* Turn windows come from **43 of the 139 episodes** (A5's 8-per-episode grid saw 38; the A6 SPEC's "same 38 turn episodes" understates it by 5).
* On the 2,317 GT-turn windows: nav_ann active on **1,021** (44 %), nav_tl (all entries) on 1,129, the shipped clip token on 1,125; nav_ann has the correct side on 1,007 and the opposite side on 14.
  nav_ann differs from nav_tl on 108 GT-turn windows (117 of 4,634 overall) — the contested clips' turns.
* On GT-straight windows (1,347) nav_ann is active on **21** where the clip token was active on **448**.

## A.4 The bar on the dense capture (EVAL; paired episode-cluster bootstrap vs V0; seed 1 = sampler replicate)

| arm | picks changed | turn ΔADE [CI] | turn dir-correct Δ [CI] | straight ΔADE [CI] | all ΔADE [CI] | seed 1: turn ΔADE [CI] | seed 1: turn dir-correct Δ [CI] | criteria 1/2/3/4 | bar |
|---|---|---|---|---|---|---|---|---|---|
| **T3a (REPORTED)** | 0.041 | −0.131 [−0.207, −0.062] | +0.064 [+0.034, +0.104] | +0.002 [−0.008, +0.012] | −0.076 [−0.116, −0.036] | −0.127 [−0.203, −0.057] | +0.065 [+0.032, +0.107] | Y/Y/Y/Y | **CLEARS** |
| T3 (foil: all entries) | 0.044 | −0.159 [−0.254, −0.080] | +0.070 [+0.040, +0.110] | +0.002 [−0.008, +0.012] | −0.092 [−0.144, −0.046] | −0.146 [−0.228, −0.072] | +0.070 [+0.037, +0.112] | Y/Y/Y/Y | CLEARS |
| T2a (hard filter, announced) | 0.072 | −0.241 [−0.461, −0.031] | +0.084 [+0.040, +0.135] | +0.009 [−0.000, +0.020] | −0.138 [−0.266, −0.015] | −0.239 [−0.453, −0.032] | +0.082 [+0.039, +0.132] | Y/Y/Y/Y | CLEARS |
| **T3a-c CONTROL** | 0.021 | −0.011 [−0.085, +0.045] | −0.004 [−0.025, +0.015] | −0.000 [−0.008, +0.007] | −0.005 [−0.049, +0.030] | −0.005 [−0.058, +0.043] | −0.006 [−0.025, +0.009] | N/Y/Y/N | **FAILED (required)** |
| B1t (label-side bound) | 0.092 | −0.486 [−0.748, −0.267] | +0.185 [+0.115, +0.270] | 0.000 | −0.286 [−0.427, −0.152] | −0.478 [−0.727, −0.264] | +0.187 [+0.114, +0.274] | – | bound |
| ORACLE-117 | 0.676 | −2.072 [−2.448, −1.743] | +0.137 [+0.086, +0.198] | −0.946 [−1.117, −0.782] | −1.656 [−1.883, −1.420] | −2.045 | +0.137 | – | bound |

**A5's own 214 windows as a subset of the dense capture** (107 GT-turn + 61 straight + 11 gentle + 35 unclassified; reported separately as the SPEC asks):
T3a CLEARS (turn −0.107 [−0.214, −0.016], dir-correct +0.056 [+0.011, +0.114], straight +0.027 [+0.000, +0.088], all −0.055 [−0.122, +0.000]; seed 1 turn −0.097 [−0.197, −0.013]);
T3 CLEARS; **T2a FAILS** (turn −0.157 [−0.343, +0.034], seed 1 [−0.302, +0.060] — consistent with A5's T2); T3a-c FAILS. The subset's T3a turn CI is 1.4× as wide as the dense one (and its all-window CI 1.5×): the denser capture buys precision, not a different answer for T3a.

## A.5 Four metric families (SPEC §2.1 format; dense set; sampler seed 0; per family, never pooled)

| GT-turn (2,317) | ADE | FDE | LON \|along\| 6 s | LON along 6 s signed | LON speed MAE 0–2 s | LAT \|cross\| 6 s | LAT heading MAE 0–2 s ° | LAT curv MAE | LAT term. heading err ° | TAC dir correct | STR nav compl. (clip token) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 3.297 | 10.713 | 7.368 | +3.437 | 0.346 | 6.330 | 5.276 | 0.010 | 23.771 | 0.815 | 0.685 |
| T3a | 3.166 | 10.230 | 7.082 | +3.283 | 0.341 | 5.973 | 5.192 | 0.009 | 21.485 | 0.880 | 0.812 |
| T3 | 3.138 | 10.133 | 6.977 | +3.178 | 0.337 | 5.935 | 5.191 | 0.009 | 21.341 | 0.886 | 0.812 |
| T2a | 3.056 | 9.857 | 6.629 | +2.811 | 0.327 | 5.920 | 5.242 | 0.010 | 20.893 | 0.899 | 0.845 |
| T3a-c | 3.285 | 10.696 | 7.244 | +3.270 | 0.343 | 6.393 | 5.290 | 0.010 | 24.314 | 0.811 | 0.665 |
| B1t | 2.810 | 9.034 | 6.172 | +2.298 | 0.321 | 5.369 | 4.996 | 0.009 | 17.886 | 1.000 | 0.948 |
| ORACLE | 1.225 | 3.621 | 2.418 | +0.144 | 0.224 | 2.194 | 3.917 | 0.007 | 13.681 | 0.953 | 0.869 |

| GT-straight (1,347) | ADE | FDE | LON \|along\| 6 s | LON along 6 s signed | LON speed MAE | LAT \|cross\| 6 s | LAT heading MAE ° | LAT curv MAE | LAT term. heading err ° | straight-keeping | STR nav compl. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 1.678 | 5.353 | 4.995 | +0.224 | 0.252 | 1.048 | 0.455 | 0.002 | 2.289 | 0.963 | 0.029 |
| T3a | 1.680 | 5.359 | 4.997 | +0.225 | 0.252 | 1.052 | 0.457 | 0.002 | 2.303 | 0.960 | 0.038 |
| T2a | 1.687 | 5.377 | 5.000 | +0.167 | 0.253 | 1.084 | 0.462 | 0.002 | 2.400 | 0.955 | 0.054 |
| T3a-c | 1.678 | 5.358 | 4.981 | +0.235 | 0.251 | 1.068 | 0.461 | 0.002 | 2.380 | 0.958 | 0.036 |
| ORACLE | 0.733 | 2.228 | 1.882 | −0.250 | 0.158 | 0.812 | 0.461 | 0.002 | 2.490 | 0.941 | 0.029 |

TACTICAL (`four_families.tactical_from_trajectory`, 0–2 s, 4,634 windows) lateral κ / longitudinal κ: V0 0.8511 / 0.5205, T3a 0.8627 / 0.5312, T3 0.8635 / 0.5337, T2a 0.8606 / 0.5609,
T3a-c 0.8511 / 0.5242, ORACLE 0.8997 / 0.8001. The gains are lateral (terminal heading −2.3° on turns, |cross| at 6 s −0.36 m) plus 6-s along-track (|along| −0.29 m), not 0–2 s speed.
LONGITUDINAL distance keeping UNAVAILABLE (no lead tracks). STRATEGIC decision UNAVAILABLE (`--no-strategic`); the nav-compliance column is against the CLIP token as in RESULT.md §2.1.
All arms, all classes, the gentle/unclassified rows and every CI are in `raw/a6_dense_score.json`.

## A.6 Post-hoc (descriptive; NOT part of the bar or the reading rule)

* **How broadly does T3a's gain sit?** It changes **151** of 2,317 turn picks (122 better, 29 worse) in **19** episodes; the 5 largest gains are −53.1 of the −304.1 total (17 %, against ≈ 100 % in A5).
  Per turn episode: **16 improve, 3 worsen, 24 unchanged** (43 turn episodes); leave-one-episode-out turn ΔADE ranges −0.140 … −0.113; the largest single episode holds 16 % of the gain.
  T2a: 13 improve, 5 worsen, 25 unchanged; leave-one-out −0.303 … −0.188.
* **Announced vs not:** T3a and T3 differ on 15 windows (14 GT-turn); restricting to announced entries costs 0.028 m of turn ΔADE (−0.159 → −0.131), i.e. 18 % of T3's turn gain
  came from the obstacle-pass turns a navigation system would not announce.
* **Both sampler draws pooled:** T3a turn −0.129 [−0.203, −0.061], straight +0.003, all −0.075; T2a turn −0.240 [−0.457, −0.031].
* **Natural-frequency weights** (turn windows 1; straight 9.583, gentle 8.707): all-window ΔADE T3a −0.0150 [−0.0263, −0.0045], T3 −0.0186, T2a −0.0237 [−0.0583, +0.0060], T3a-c +0.0017 [−0.0110, +0.0152].

## A.7 What this does and does not show

* **Shows:** on 4,634 windows (2,317 GT-turn) of the held-out episodes, an inference-time soft nav-compliance term (×10) driven by ANNOUNCED, time-localised oracle nav improves turn
  direction by 6.4 pp and turn ADE by 0.13 m on both sampler draws, costs nothing on straight driving (+0.002 m), and a mismatched nav series does not (paired difference −0.12 m).
* **Does not show:** (a) generalisation to NEW episodes (same 139); (b) training-run robustness (one checkpoint; `H-ESTIM-SEED-1`); (c) anything with a deployable, non-oracle route
  (the nav is the ego-future oracle; A6's "announced" removes only what the builder itself would not command, i.e. 3 eval entries on 2 clips / 103 train entries on 66 clips);
  (d) closed-loop behaviour. The effect is small against the oracle gap: 0.27 of B1t, 0.06 of the ORACLE-117 turn gain (−0.131 of −2.072); 56 % of GT-turn windows carry no nav signal at all.
* **Next lever (Rule Zero):** the cleared arm is an inference rule on a fixed checkpoint, so the larger measured levers remain training levers: the selector's speed choice
  (RESULT.md §2.3 R-SPEED, bound −0.6 m all-window) and a vision-only turn cue for the 56 % of turn windows nav cannot reach. L2 (time-localised ANNOUNCED nav as a refcv8 training input)
  is the reading rule's own recommendation. Nothing blocks the analysis; deploying it is the PI's decision.

## A.8 Judgement calls and what I could not do

* **Checkpoint:** see the stamp (ckpt.pt d5f104ee…, same file as A5). **Random sample:** drawn from ALL non-turn windows (including the 694 unclassified), `np.random.default_rng(0)`, without replacement.
* **G1′ (2), "time_s ≤ 30" for FOLLOW commands** (no `time_s`): read as satisfied; both sub-populations are reported (turn 1,655 / 1,655, follow 144 / 144), so the verdict does not depend on it.
* **Mechanics debug:** before seed 1 existed, the scoring script was run once with seed 0 standing in for the replicate to debug it; that output was discarded (not in `raw/`), the controls were
  expected to fail on the replicate comparison and did, and the bars were not used. The registered scoring is the single run on the real two-seed capture, repeated once after adding
  the post-hoc blocks (every non-post-hoc key bit-identical).
* **Not done:** no new episodes, no second training seed, no closed loop. Thor: the capture job exited at 11:57:46 UTC; no python process of this agent is alive on Thor (checked at ~11:58 UTC,
  the processes then running belong to other streams); outputs remain at `/home/nvidia/refcv7_post/route_a6/{out,logs,code}`.
* The earlier G: disclosure (Part B §5) stands.

---

# PART B — the stop at G1 (history, kept exactly as written when A6 stopped)

## B.0 RESULT A6 as it stood when it STOPPED at G1 (no capture launched, no arm scored)

**Status: A6 stopped as its own SPEC requires** (`SPEC_ADDENDUM_A6.md`, sha256 `b4099578…`, registered 2026-10-04T10:32:25Z: *"If G1 fails, A6 stops and
reports."*). Nothing on Thor was started, the GPU was not touched, and no ΔADE of any A6 arm exists. Everything below is a COUNT or a reproduction rate.

**Stamp.** Evidence class MEASURED; source `raw/a6_g1.json`, `raw/a6_announced_table.json`, `raw/a6_g1_consequences.json`; code `code/announced_a6.py`,
`code/a6_g1_consequences.py`, tests `code/test_announced_a6.py` (10/10). Label files: train md5 `b45377a1…` (4,572 records), eval `eefc38d1…` (147 records).
The builder called is `D:/Projects/TanitAD/stack/scripts/s2_geom_emit_v7.py` (md5 in the JSON); its `nav_command()` and `is_turn()` are CALLED, and the
thresholds (15° / 140 m / 8.0 m/s) are read from it, not written here.

## 1. announced(entry, record)

It calls the builder's own `nav_command(poses, 0, [that entry's manoeuvre], suppress_turn=record.turn_suppression.applied)` and returns whether the token it
commands equals the entry's token. The entry's manoeuvre is the record's `manoeuvre_sequence` element with the same `t_start_s` and `dyaw_deg` (the tuple
`(t_start, t_end, dyaw, R, v_min)` the builder uses). Two facts about the builder that decide what it can and cannot remove:

* **`nav_30s.entries` never contains a road curve.** `build_v8_nav30s.py` drops every manoeuvre whose `is_turn` is False; I re-evaluated the builder's `is_turn` on
  **every** `manoeuvre_sequence` element of all 4,719 records: **0** disagreements with the stored flag. So at entry level the "curve" half of announced() removes nothing.
* **The suppression is read by the emitter as `applied` (`s2_geom_emit_v7.py:893`) but by the nav_30s builder as `suppressed` (`build_v8_nav30s.py:83`)**, so
  nav_30s kept every contested turn. announced() reads `applied` and removes them. The builder applies the flag to the whole clip, so announced() removes **every**
  turn entry on a suppressed record: train **103** turn entries on 66 records (28 of those records hold more than one turn entry), eval **3** entries on 2 records
  (`bbd162dfe0eb` has a second, different turn at 14.4 s — the −152° stop-and-turn — which is also removed).

## 2. G1 — FAILED as registered (97.01 % < 99 %)

| criterion (SPEC A6) | required | measured | verdict |
|---|---|---|---|
| announced(`entries[0]`) reproduces the shipped `nav_command.token` side, train + eval | ≥ 99 % of records | **4,578 / 4,719 = 97.01 %** (train 4,438/4,572 = 97.07 %; eval 140/147 = 95.24 %) | **FAIL** |
| …on records whose `nav_command.reason` names a suppression | 100 % | **68 / 68** (66 train + 2 eval) | PASS |

**All 141 mismatches fall in exactly two structural classes; there are 0 "other".**

| class | n | why announced(entries[0]) cannot match |
|---|---|---|
| **curve-first** | **70** (66 train + 4 eval) | `nav_command` applies its rule to `seq[0]` only. When `seq[0]` is a road curve it returns FOLLOW ("curve, not a turn") and never looks further; `entries[0]` is a LATER, real turn (`is_turn` true; the 4 eval ones: −33°/R 39 m, −49°/19 m, −69°/38 m, −91°/20 m). announced(that entry) = True, shipped = FOLLOW. |
| **turn beyond the nav_30s cap** | **71** (68 train + 3 eval) | `nav_command` looks 35 s ahead and returns TURN with `time_s` 30.1–35; `nav_30s` caps at 30.0 s, so `entries[0]` is FOLLOW-only and there is no entry to announce. |

**Diagnostics (NOT a substitute for the registered control; reported so the resolution is one step):**
* the builder's rule applied to **`seq[0]`** — the object it is actually applied to — reproduces the shipped `nav_command` on **4,719 / 4,719 (100 %)**, suppression-named 68/68;
* a stricter variant, "an entry counts as announced only if it is also `seq[0]`", would pass the curve-first class but still fail on the 71 beyond-cap records: **98.50 % < 99 %**.
  So no definition of announced() on `entries[0]` can pass the registered control; the control, not the function, is mis-specified.

## 3. What this means for the A6 arms (WINDOW COUNTS ONLY, A5's 1,112-window EVAL grid; no arm scored)

A5's post-hoc finding was "6 clips where `nav_command` suppressed a turn: 4 × curve, 2 × contested". Under the faithful announced():

* **The 2 contested clips are removed; the 4 curve-first clips are NOT** — their turn is a real turn after a road curve, and the builder's own rule announces it.
* Of A5's **7 GT-turn windows with "clip token FOLLOW, nav_tl active"** (the 54 % of T2's turn gain), **5 lie on the contested clips** (removed) and **2 on curve-first clips** (still active).
* nav_ann differs from A5's nav_tl on **10 of 1,112 windows** (GT-turn 5, GT-straight 2, gentle 1, unclassified 2); nav_ann is active on 117 windows against 127.
  Suppressing only the contested turn instead of the whole clip changes 1 further window (118 active).
* ⇒ **T3a and T3 differ on at most 10 windows (5 GT-turn)** of the A5 grid. On a denser capture of the same 38 turn episodes the same two clips carry the difference.
  The reading rule's branch "T3a fails while T3 passes ⇒ the gain needs non-announced turns (curves, obstacle passes)" would therefore be decided by the two contested clips
  alone, and "curves" play no role at all.

## 4. What I need from the Master Mind (the SPEC is fixed; I changed nothing)

1. **Amend G1.** Candidates that are independent of announced()'s outcome: (a) G1 on `seq[0]` (the builder's input) — measured 100 % on 4,719; (b) literal `entries[0]` on the records
   where `nav_command` and `nav_30s` are comparable, i.e. excluding the two classes above, which are identifiable from the record without calling announced() (curve-first: `seq[0]`
   not a turn and `entries[0]` a turn; beyond-cap: shipped TURN with `time_s` > 30.0) — then 4,578 / 4,578 by construction, which is weaker than (a).
2. **Rule on the two semantics the SPEC leaves open:** (i) is a real turn behind a road curve "announced" (mine: yes — the entry is a turn; the builder's `seq[0]`-only look is a
   limitation of `nav_command`, not of a navigation system); (ii) is the contested flag clip-level (the builder's own, mine) or per contested turn (changes 1 window of the A5 grid).
3. **Say whether to proceed.** The capture does not depend on announced(): when you re-register A6, I can start it at once. Plan (not started): every eval139 window with GT 6-s heading change
   ≥ 30° at stride 1 (≈ 2,300 ESTIMATED) + an equal-count seeded random sample of the others, sampler seeds 0 and 1, the package's `run_route.py` path, `ckpt_50400`, under
   `flock /home/nvidia/refcv7_post/thor_gpu.lock`, one `setsid nohup` job; ETA will be stated after timing the first 50 windows (A5-era forward cost: 0.39 s per window MEASURED,
   RESULT.md §0 × ≈ 9,200 forwards (2 seeds × ≈ 4,600 windows, ESTIMATED from 107/1,112 × 23,772 turn windows, doubled for the random sample) ⇒ ≈ 1 h of GPU). Scoring code for T3a / T3 / T2a / T3a-c reuses `code/nav_tl_a5.py` unchanged apart from the announced flags.

## 5. Disclosures

* **G: was probably read once, by accident.** My very first exploratory import of the builder (`import s2_geom_emit_v7` to see whether it loads) ran before I set `PYTHONPATH`; the venv's
  editable `tanitad` finder maps to the G: checkout (`__editable___tanitad_0_0_1_finder.py`: `MAPPING['tanitad'] = G:\…\stack\tanitad`), so that import read Python files from G: and may have written
  `__pycache__` there. I did not look at G: to check. Every later run set `PYTHONPATH=D:/Projects/TanitAD/stack` first, and `announced_a6.py` now refuses to run if any imported module lives on G:
  (the JSON records the builder path and md5; the module list had 0 G: entries on the runs that produced these files).
* The launch-tree `taniteval` (ev7) and the D: working-tree `tanitad` must not share a process. announced() therefore runs in its own process and hands the scoring process a table.
* No Thor access in this stage; nothing remains running there from this stage. The A5-stage Thor work was reported earlier.
