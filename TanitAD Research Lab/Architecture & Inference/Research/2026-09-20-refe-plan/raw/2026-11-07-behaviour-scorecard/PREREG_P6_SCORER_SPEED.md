# PREREG_P6_SCORER_SPEED -- has P6's scorer lost u400's speed pessimism? (stream BEHAV, written 2026-11-07, BEFORE P6 ends)

**Status: FROZEN before any P6 proposal, logit or label exists on the 1,058 C3 tokens.**
- sha256 in `PREREG_P6_SCORER_SPEED.sha256`. Every P6 stage below refuses to run unless this file hashes to that value.
- **Tier T0** (open loop, one step, logged ego history). Nothing here is a driving claim.
- **Evidence classes:** MEASURED = ours with an artifact path. PUBLISHED = none used. INHERITED is marked. ESTIMATED is marked.
- P6 is training on the pod; its end is expected about 2026-11-08 19Z (INHERITED from the brief, not re-verified).

## 0. What has been seen at freeze time
- **Seen, all of u400:**
  - YSEL P1/P2/P3 (`refe_selection_probe_20261107/`) and the YSEL2 prereg;
  - the u400 reference numbers of section 2, computed from banked data by `u400_reference.py` (sha in section 9).
- **Not seen:** any P6 output on the C3 tokens. No P6 proposal, logit, pick or PDM label on them exists.
- P6's training-side BCE components may be seen in its log. They say nothing about within-token speed rank-correlation, and no bar below is set from them.
- The P6 full-navtest PDMS read is a separate instrument. If it is seen before this read, that is disclosed in the result, and no bar here changes.

## 1. Hypothesis (HYPOTHESIS, one sentence)
**H-P6-SPEED-1.** P6's scorer, trained on the PDM labels of its OWN proposals, online, lag 0, has lost u400's speed pessimism: its TTC and NC heads no longer fall with proposal speed beyond what the truth does, and its picks on CRUISE no longer slow down.

**Where "trained on its own proposals online" is implemented** (read 2026-11-07, `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`):
- `train.py:2049-2051`: `--labeller {bank,online}`, help text "online = NAVSIM PDM targets on each sample's OWN current proposals ... (deferred score backward, lag 0)".
- `train.py:1141, 1143-1147`: `online = (a.labeller == "online")`. It REFUSES `--grow`, `--scorer-mode onpolicy`, `--onpolicy-targets`, `--pdm-labels`, `--pdm-only` with it, and requires `--metric-cache` for the pdm backend.
- `train.py:1296-1298`: the run prints "ONLINE PDM labeller ... the fixed candidate bank is NOT loaded".
- `train.py:1400-1406`: the `OnlineLabeller` is built.
- `train.py:1725-1735`: the labelled array is asserted by sha256 to be the micro-batch's own detached proposals, and lag 0 is asserted. The loss is BCE of the score against the labels, `bce = F.binary_cross_entropy_with_logits(P["score"], lab_t)` (`:1732`), `l_score` (`:1735`).
- **P6 has 6 score outputs, not 7:** `model.py:96` `n_score_components: int = 6`, `model.py:563` `nn.Linear(cfg.dec_width, cfg.n_score_components)`, and `SPEC_P6_PAPER_FAITHFUL.md` §1 "SFT-4's 7th lane output is not built". Its rule is `navsim_v1` (G-DVB row, `SPEC_P6` §5). So P6 has **no lane factor** in its pick.
- The cited lines are the working-tree reading of 2026-11-07. The launch argv of the P6 run (`--labeller online`) is **UNVERIFIED by me**: the executing agent quotes it from the pod's `config.json` into the result (section 6, G6-ARGV).

## 2. u400 reference values (MEASURED, `u400_reference.json`, reads banked data only)
Population: the 1,058 C3 tokens, 154 CRUISE (`tokens_cruise.json`: gt_valid, v0 > 5 m/s, |ax0| < 0.3, |GT v4 - v0| < 1). Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile (`sel_common.log_boot_idx`). "One checkpoint, deterministic planner": it answers "another draw of logs" only.

| quantity | u400 value |
|---|---|
| **stratum-weighted CRUISE slow-pick share** (v4/v0 < 0.85; W_zero 529/12146, W_random 11617/12146) | **0.412 [0.275, 0.569]**. By stratum: zeroed 0.407 (n = 91), random 0.413 (n = 63). Unweighted 0.409. |
| **same, u400 re-picked WITHOUT the lane factor** (rule `navsim_v1` on u400's own logits: the lane-confound control) | **0.366 [0.229, 0.521]**; paired Delta vs banked **-0.046 [-0.104, -0.0005]**; 61 CRUISE picks change |
| within-token Spearman, TTC head, predicted vs true (tie-correct average ranks) | pred **-0.546 [-0.640, -0.449]** (n = 154); true **-0.094 [-0.181, -0.004]** (defined on **52** tokens); paired gap **-0.451 [-0.565, -0.341]** |
| same, NC head | pred **-0.491** (paired subset; -0.443 on all 154); true **-0.094 [-0.178, -0.014]** (defined on **38** tokens); gap **-0.396 [-0.517, -0.285]** |
| same, DAC / EP | DAC gap -0.257 [-0.372, -0.178] (n = 119); EP gap -0.351 [-0.484, -0.201] (n = 151) |

**Two corrections these numbers force on `P1_DIAG.md` (stated, not softened):**
1. **The "true" within-token correlation in P1 is not tie-safe.** `p1_diag.py::spearman_rows` ranks by `argsort(argsort(.))`, which breaks ties by proposal INDEX. The true NC and TTC labels are constant within most CRUISE tokens (all 64 proposals pass), so P1's "true -0.049 / -0.039" on 154 tokens mostly measures the correlation of proposal index with speed. Tie-correct, the true value is defined on only 52 (TTC) and 38 (NC) of 154 tokens, and reads about -0.09. **The direction of P1 stands (predicted -0.5, true about -0.1); the quoted "true" figures do not.** The legacy estimator reproduces `p1_diag_all.json` exactly (gate G6-REPRO) and is reported beside the pre-registered one.
2. **The stratum weighting does not matter on ALL logs** (0.407 vs 0.413). `P1_DIAG.md`'s "headline share is inflated" holds for the TRAIN logs (0.200 vs 0.304) and not for the 154-token ALL set, whose weighted share is 0.412 and whose CI is wide (51 logs).

## 3. Instrument: the YSEL P1 machinery on a P6 dump, as an ADD-ON to the P6 read
Reuse, unchanged: `sel_common.py` (loader, `step_speed`, bootstrap), `p1_diag.py` (its Q1-Q4 blocks), and the selc chain (`selc_dump.py` GPU dump, `selc_score.py` 64-column harness table). Their sha256 are in `CODE_PIN.json` of `refe_selection_confirm_20261107/` (sha `86e7b9fdb5a3479cfefdddc9cee906be1e48f8134666c8346b7b92a79284f3ea`).

**What must differ, and nothing else (a P6 variant, written and pinned BEFORE the P6 checkpoint is read):**
| item | u400 (selc_dump.py) | P6 variant |
|---|---|---|
| checkpoint | `model_sft4_B_u400.pt` (sha256 prefix `8b4dba4818b5cc96`) | the P6 FINAL checkpoint, the same file the P6 navtest read scores; sha256 recorded and pinned |
| scorer outputs / rule | 7 logits, `navsim_v1_lane` | **6 logits, `navsim_v1`** (`p0*p1*(5p2+5p3+2p4)/12`) |
| nav input | goal points | **command** (`nav_mode=command`; NAVSIM `driving_command`; `eval/P6_PLANNER_COMMAND.md`) |
| identity reference | C3 `table.npz` (u400) | the P6 read's own full-navtest seam npz and csv |
Everything else (token loop, GPU lock policy, gt future, human8, `ego_real` vector) is the selc code. The variant `selc6_dump.py` + `selc6_analyze.py` (= `p1_diag.py` with 6 heads, `lane=False`, tie-correct ranks) have their sha256 written to `CODE_PIN6.json`, whose own sha256 is written into the result before any P6 token is run. Any change after that is an `AMENDMENT_1.md`, written before any result is read.

**Stages (CPU unless marked):**
1. **dump (GPU, dev box, YAX lock, never on the training pod).** The 1,058 C3 tokens, tokens list = `tokens_all.json` (sha in section 9). Keeps per token: logits [64,6], pick, raw traj [64,20,3], executed props8, `ego`, gt future. ETA ESTIMATED 0.7 h.
2. **score (CPU, W3's NAVSIM v1.1 harness, `selc_score.py table`).** All 64 proposal columns x 1,058 tokens = 2 chunks x 64 = 128 calls. ETA ESTIMATED 1.3-1.8 h (scaled from `POWER.md` 448 calls = 4.6-6.2 h).
3. **analyse (CPU).** `selc6_analyze.py` writes `p6_speed_result.json` and `P6_SPEED_RESULT.md`.

**Gates; any red = INSTRUMENT FAILURE, no verdict, no number quoted:**
- **G6-PIN:** this file's sha256; `CODE_PIN6.json`; token files; `PREREG_SEL.md` = `e00fa28a...`.
- **G6-ARGV:** the P6 run's `config.json` shows `labeller = online`, `label_backend = pdm`, `nav_mode = command`, rule `navsim_v1`, `n_score_components = 6`. Quoted into the result. (If P6 did NOT train online, H-P6-SPEED-1 is void and the result says so.)
- **G6-DISJOINT:** none of the 1,058 tokens' logs appears in P6's training token list.
- **G6-0 (pick):** the dump's own pick equals the offline torch-float32 argmax of the `navsim_v1` aggregate on 1,058/1,058. **RED:** argmin must disagree on every token; the aggregate with a flipped sign on p3 must disagree on at least 1 token.
- **G6-1 (fidelity):** the dump's executed pick poses equal the P6 read's full-navtest seam to <= 1e-3 m on 1,058/1,058. **RED:** a cyclic shift of the reference must fail.
- **G6-2 (CRUISE set):** the dump's `ego` vector equals `ego_real` of the ego probe on 1,058/1,058, and the CRUISE formula reproduces `tokens_cruise.json` on 1,058/1,058. (The ego vector is model-independent, so this must hold.)
- **G6-3 (harness):** the banked-pick column equals the P6 read's csv PDMS bit for bit on 1,058/1,058. **RED:** poses shifted +2.0 m in y must differ on at least 1 token.
- **G6-4 (table):** 64 x 1,058 columns finite and valid, and PDMS recomputed from the six sub-scores (`NC*DAC*(5EP+5TTC+2C)/12`) equals the harness PDMS to 1e-9.
- **G6-REPRO:** `selc6_analyze.py` run on the u400 C3 table reproduces `u400_reference.json` exactly (max abs diff 0.0), and the legacy estimator reproduces `p1_diag_all.json` `Q4_within_token_spearman_vs_speed` for NC, DAC, EP, TTC (|diff| <= 1e-12). **RED:** swapping predicted and true flips the sign of every gap.

## 4. Statistics (defined now)
- `ratio_k = v4_k / v0`, `v4` = the one-step 0.2 s chord at t = 4 s of the RAW proposal (`sel_common.step_speed`, idx 19), `v0 = ego_real[:, 6]`.
- `rho(X, Y)` = Spearman with **average ranks**, over the valid proposals of one token. A token with fewer than 5 valid proposals, or constant X or Y, is undefined for that head.
- Heads: TTC (index 3) and NC (index 0) are PRIMARY. DAC (1) and EP (2) are reported (DAC is DESCRIPTIVE; EP is the opposite-sign half of the problem). Predicted = sigmoid of the logit; true = harness sub-score.
- `S_h` = mean over the 154 CRUISE tokens of `rho(pred_h, ratio)`.
- `T_h` = mean over CRUISE tokens where defined of `rho(true_h, ratio)`.
- `G_h` = mean over CRUISE tokens where BOTH are defined of `rho(pred_h, ratio) - rho(true_h, ratio)`.
- **Slow pick:** `ratio[pick] < 0.85`. `share_P6` = stratum-weighted share, W_zero 529/12146 and W_random 11617/12146, over CRUISE tokens. Draws that leave a stratum with no CRUISE token are dropped and the count reported.
- All intervals: paired log-cluster bootstrap, the same 10,000 x n_logs draws for every statistic and for u400 (`log_boot_idx`, seed 20260927).

## 5. Committed outcomes (both written now)
- **(A) head calibration:** point `G_TTC` in [-0.15, +0.15] **and** point `G_NC` in [-0.15, +0.15].
- **(B) not subset-driven:** `|S_h - T_h| <= 0.15` for TTC and NC, with `S_h` on all 154 CRUISE tokens. (The true side is defined on only about 40-50 tokens, so (A) alone could be a subset artefact.)
- **(C) behaviour:** `share_P6 <= 0.20` **and** the paired Delta (P6 minus u400 banked) has CI-hi < 0 **and** the paired Delta (P6 minus u400-without-lane, 0.366) has CI-hi < 0. (The 0.20 margin is about half of u400's 0.412 and 0.17 below the lane-free control; ESTIMATED, set without any P6 data.)

| verdict | condition | what it commits us to (Rule Zero) |
|---|---|---|
| **HOLDS** | (A) and (B) and (C) | The root fix is present in P6 at T0. **Not attributed to the online labeller:** P6 also differs from u400 in command input, 6 outputs / no lane, un-detached context, 25 epochs. Next: the closed-loop BEHAVIOUR_SCORECARD tier-1 read on P6, as the first checkpoint scored on it. SCORER-GYM's first experiment drops in priority. If the sim crawl persists on P6, the speed-pessimism is exonerated and the next suspects are the ax loop (`PREREG_AXOFF`) and image informativeness (GREY regime, `refe_ego_probe_20261106/RESULT.md`). |
| **REFUTED** | at least one of `G_TTC`, `G_NC` has point < -0.15 (the pessimism persists) **and** not (C) | Online own-proposal labels did not remove the pessimism. **Next arm: the scorer retrain, SCORER-GYM's first experiment** (speed-balanced TTC/NC targets or hard pairs: slow-unsafe vs fast-safe). BLOCKED on GPU and PI authorisation; named. |
| **NOT PROVEN** | anything else, with the sub-case named in the result | **N1** (A and B hold, C fails): calibrated heads, yet picks still slow. Selection-geometry or generation: run the `p1_diag.py` Q1/Q2 blocks on P6 (is a faster, equally good proposal present? which term carries the margin?), then the proposal set. **N2** (C holds, A or B fails): behaviour moved without head calibration, e.g. through the EP head or the missing lane factor; read Q2. **N3** (A holds, B fails): subset-driven; needs the larger token set (below). |
- **Precedence:** INSTRUMENT FAILURE first. A verdict is never issued over a red gate.
- **Power (ESTIMATED from u400's own CIs):** the paired gap CIs on u400 have half-width 0.11-0.12 at n = 38-52 tokens, so a P6 point inside +-0.15 is not itself an equivalence proof. That is why (B) and (C) exist and why the result reports every CI. If N3 occurs, the next read is the same instrument on >= 2x fresh navtest logs, with a NEW prereg and no retuning.

## 6. Reported with it (DESCRIPTIVE; can never change the verdict). Four families, tier T0.
- **LONGITUDINAL:** weighted and unweighted slow-pick share; overspeed share (v4/v0 > 1.15); plan v4 minus GT v4 on CRUISE (u400: -0.535 [-1.391, +0.221] m/s, `refe_ego_probe_20261106/RESULT.md`); mean plan v4/v0 (u400 0.934 [0.803, 1.054] on A0).
- **LATERAL (what the proposals allow):** ADE to the human future on CRUISE; |y| at 4 s; DAC failure rate; DDC mean.
- **Safety:** NC and TTC failure rates of the pick; changed-pick counts.
- **TACTICAL:** n/a, n = 0 (no manoeuvre-decision label on these tokens; REFe emits no tactical output here).
- **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Not a pass, not omitted.
- Also from `p1_diag.py`: Q1 (a fast, no-worse proposal exists on x % of slow picks; u400 85.7 % [70.6, 97.4]), Q2 (dominant margin term; u400 lane 31/55), Q3 (human-nearest proposal), Q4 calibration table by speed bin.
- Open-loop PDMS on the 1,058 mix is NOT a statistic of this prereg (the C3 sample is stratified; the zero-floor mix of `PREREG_SEL_CONFIRM` section 3 is the only admissible one).

## 7. Limits (named, not hedged)
- **One checkpoint, deterministic planner:** no claim about training or inference variance (`H-ESTIM-SEED-1`, `CLAUDE.md`). A HOLDS here is "this P6 checkpoint has the property", not "online labelling produces the property". The cheapest attribution test, if HOLDS: a second training seed of the P6 recipe (compute, PI).
- **Confound by construction:** P6 has no lane factor. The lane-free u400 control (section 2) bounds that part at about 5 points of slow-pick share.
- **Same 1,058 tokens as u400's analysis** (selected as 529 zeroed + 529 random). u400 was not tuned on them; P6's train set must not contain their logs (G6-DISJOINT).
- 154 CRUISE tokens over 51 logs: the share's CI is about +-0.15 (u400: [0.275, 0.569]).

## 8. Compute and ownership
- Dump on the dev box GPU under the YAX lock, after P6's checkpoint is on the dev box and the P6 navtest read exists. CPU scoring after.
- Chain: reuse `selc_chain.py`'s stage runner with the P6 variants. Idempotent stages, markers under `markers/`, `selc6_*.DONE|FAILED`.
- **Who writes `selc6_dump.py` / `selc6_analyze.py`:** the first agent launched after P6's end, from this file, BEFORE it opens the checkpoint. The MM lands this file and its `.sha256` before that agent starts.

## 9. Pins (sha256)
| file | sha256 |
|---|---|
| `refe_ego_probe_20261106/tokens_cruise.json` | `57db811e644218c00020cd6d25b417b604ffc9af7b81612976d0d52f02263f78` |
| `refe_ego_probe_20261106/tokens_all.json` | `421aac1d51da3beb3497ac92494ed7dacd99efe62e758e49ae5ada63359fa566` |
| `refe_selection_probe_20261107/sel_common.py` | `50727023525c86b2be503db5c7f84d56de9b2be98f260aab700e4c9438f27d23` |
| `refe_selection_probe_20261107/p1_diag.py` | `7bd7220dffa1ae117bc1803ccdf8a4c76163d17bdfabbe3016dde560e2ca8b9d` |
| `refe_selection_probe_20261107/p1_diag_all.json` | `1645b6572f593f94ffcd46c7e8977ba0aec7182bcb5cffb660c416b06b1d9a6c` |
| `refe_selection_confirm_20261107/CODE_PIN.json` | `86e7b9fdb5a3479cfefdddc9cee906be1e48f8134666c8346b7b92a79284f3ea` |
| `refe_selection_confirm_20261107/selc_dump.py` | `cf16d85b93dd470ed620f8f646219919a9fc3b9d8570b5e82b5a99d49d07e8d0` |
| `refe_selection_confirm_20261107/selc_score.py` | `3fd4997a05e55240376fbb71af7c0459f9ccb2fa6ddf420a737648a26fb6ae09` |
| C3 `table.npz` (`.../raw/2026-10-11-c3-selection-headroom/`) | `cf30519051bea03b5a950975477806ce9b1ef141ac270dbe5781ea53f6dbc7c0` |
| `refe_behaviour_20261107/u400_reference.py` | `ee124c5b3f4114ff4129a8d872ec3129b928dc7807cbbf306361ef0f5e2e3a11` |
| `refe_behaviour_20261107/u400_reference.json` | `1ad5776a19a02179cb22a63fc9d20e9a04db34d4a2ffd6b6f14796e8121682e5` |
