# DRAFT PREREG_REFE_STOPLINE: a map-carried stop-line input on the nuPlan / NAVSIM rig (REFe)

**Status: DRAFT for the Master Mind (MM) to review and freeze** (2026-11-16). Drafting only: nothing in this card has run, no flag exists, no GPU or pod was touched. It must be frozen with a `.sha256` before any stop-line feature of a navtest token is computed and before any number below is scored.

**Authority.** The PI's go, 2026-11-16, verbatim: *"3 yes draft"*. That approves drafting this card. It approves no compute, no code landing and no venue. Those are section 0's decisions.

**Lineage.** STOPPERC Q5 read RES416-FLAT (`.../2026-10-17-v8f-mini/raw/2026-11-15-v8f-stopperc-q5-416/run/VERDICT_Q5.md`). The frozen card's own next lever (`AMENDMENT_7_Q5_416.md` section 2.4, RES416-FLAT row) is "an input that carries the stop location": on PhysicalAI-AV only an ego-future label, on nuPlan / NAVSIM a map stop line. `QUESTIONS_416.md` Q2 named this card.

**Path shorthands.** `V8` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. `SC` = `PK/raw/2026-11-07-behaviour-scorecard`. Code anchors are at branch tip `origin/agent/arch-inf-20260803` = `68850034`; they move, and the freeze re-anchors them.

**Evidence classes.** MEASURED (ours, artifact path) / PUBLISHED / INHERITED (another agent or document, not re-verified here) / ESTIMATED / HYPOTHESIS / UNVERIFIED. Model numbers also carry a tier.

---

## 0. Decisions for the MM / PI (nothing runs before these are answered)

### D1. Admissibility: is a map-carried stop line an admissible INFERENCE input? (PI question; `QUESTIONS_STOPLINE.md` Q1)
The input proposed here is a function of three things only: the static HD map (nuPlan `map.gpkg`), the ego's current global pose at t0, and the rig's existing navigation command. It is not a situation-classifier output and it is not vision. No PI ruling names map inputs. The rules read both ways.

**The case that it is admissible.**
* *"Nav is an INPUT at training AND inference — stated by the PI many times; it is not an oracle that disappears at deployment."* (`CLAUDE.md`, strategic-pause section). A stop line is static content of the same navigation map the route comes from.
* The goal-input rule admits route signals and forbids only one thing: *"yes a goal input is admissible, at the same time, we need to be careful not to include the result of the situation classification in the goal input."* (Sayed, 2026-08-03). The stop-line feature is computed before the model runs, from map + pose + command. It cannot be computed from the situation classifier's output.
* The vision-only rule is scoped to the scenario classifier: *"for ground truth data of scenario classification you can use both ego and other label, for inference only vision."* (Sayed, 2026-08-03). REFe has no scenario or situation classifier: the scorecard reads TACTICAL from executed behaviour because "REFe emits no tactical output" (`SC/BEHAVIOUR_SCORECARD.md` section 2). The planner already takes non-vision inputs: ego kinematics 7-D and the command (`PK/refe/model.py:185-192`).
* The horizon ruling puts map-like content in the strategic band: *"(6se to 30 sec incl. av checkpointsssss, detectedstrategic informtio nd derived goals from it)"* (PI, 2026-11-08). `CLAUDE.md` also records that *"the strategic-brain topology must come from AlpaSim or an external corpus"*.

**The case that it is not.**
* The programme's inference stance leans to vision for anything vision can see. The PI's time-of-day ruling: *"estimate rougly the poeriod of day and linked illumanation from pure images and past frames no additional informaio"* (2026-11-08). A painted stop line is visible.
* The rig's own design line: *"Labels may use the map (labels may use privileged signals); the map is never a model input."* (`PK/refe/navsim_lane.py`, module docstring). It is a REFe convention, not a PI ruling, but this card would break it.
* The DriveZero student REFe reproduces takes *"the multi-view images of the current frame, the ego kinematics, and a navigation command"* (quoted at `PK/refe/model.py:177-178`). A map arm is not like-for-like with the paper's 94.55 PDMS (PUBLISHED, DriveZero Table A13, as cited in `PK/REFE_MODEL.md` section 2; not re-read here).
* NAVSIM's sensor-agent interface carries no map. MEASURED from source (devkit 2.0.0 tree, `F:/Archive/devbox-C/navsim/devkit/navsim/common/dataclasses.py:139-156`): `AgentInput` = `ego_statuses`, `cameras`, `lidars`, and `EgoStatus.in_global_frame` is False for `AgentInput`. The map (`Scene.map_api`, `:334-345`) reaches only agents built with `requires_scene` (`agents/abstract_agent.py`). The rig's harness is NAVSIM v1.1; the same fields there are UNVERIFIED. A map arm therefore needs the current global pose, which is not a NAVSIM sensor-agent input. On a car that pose comes from localisation against the map.
* `CLAUDE.md`: *"A supplied route is optimistic by construction on PhysicalAI — our only route supplier there is the ego's own future path. Prefer a predicted goal."* Section 2 shows that the scorecard's own stop-line selection is exactly such a future-path supplier.

**Drafter's recommendation (the decision is the PI's).** Admissible for the REFe planner, under four conditions, each enforced by a gate in section 8:
1. location only: static map + current pose + the command the model already receives; no future frame, no route list from the log, no signal state;
2. never an input to any situation or scenario classifier head;
3. results are reported as a **map-equipped configuration**, never pooled with camera-only rows, the paper-reproduction table or NAVSIM sensor-agent comparisons;
4. a positive result transfers to the product only as a mechanism claim ("stop coverage is an information gap"). The product's input would be a vision-predicted stop location trained with the map line as its LABEL, which is section 6's follow-on.

The **signal state** (OpenScene `traffic_lights`, `is_red` per connector) is recommended **inadmissible** as an input. It is the thing vision must perceive, and an input carrying it would replace perception. It stays admissible as a label.

### D2. Base model and labels (MM; REFe MM to agree)
* **Base:** the P5-C base, by its own rule. P7 `model_final` sha256 `2613fc17b9e5d63342c9f42120c5612a6ec7061fda72f83dac46ddd2e70fd3fe` (9,075 steps, ZZP7_DONE 2026-11-16 06:09:27Z) if SPEC_P7 section 5's read says ADOPT or NOT PROVEN; P6 if REFUTED or INVALID. INHERITED: `E:/Projects/TanitAD-artifacts/dagland/LOOP_STATE_P6_2026-11-06.md`, RESTART ENTRY 2026-11-16 ~08:40Z. The P7 read was relaunched at 12:06Z (attempt 3) and has not reported.
* **Labels:** the bank raw `traj` labels of P5-C's arm C0, unchanged, for both arms here. Only one lever moves.
* **Recommendation:** P7 base, C0's labels and C0's argv verbatim plus the stop-line flags. That makes P5-C's C0 a candidate control (D4).

### D3. Venue, owner and schedule (PI for spend; REFe MM for the pod)
* **Training venue:** REFe's L40 RunPod pod, the only box that runs the REFe trainer with its online labeller. It is owned by the REFe MM session ("SFT-4/P2 state after weekly-limit and pod stops (fork)"). The queue today: P5-C C0, then C1, then C0r if C1 − C0 separates beyond B (INHERITED, LOOP_STATE entries 2026-11-16 08:40Z-12:15Z; PI rulings 11-12 / 11-14).
* **This card borrows nothing from that stream.** It needs from the REFe MM:
  * a pod slot after P5-C's last arm trains;
  * a land of the `refe/train.py` and `refe/model.py` wiring after that arm, because a land touching the trainer between a bind and a restart invalidates the bound record (the auto-memory rule on closure lands, 2026-11);
  * a `refe/launch_gate.py` profile;
  * the P5-C C0 checkpoint and its seam dumps, if D4 says reuse.
* The LoRA pod (RTX 4000 Ada, ~20 GB) is not a candidate. P7 planned a 44 GB L1 memory bound with micro-batch 2 as the fallback (INHERITED, `PK/SPEC_P7.md` section 1), and the PI assigned that pod to DATASCALE storage on 2026-11-16.
* **Eval venue:** the dev box, the route the P7 read uses (`stage2_sft.py` seam, shared GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`). Risk, INHERITED: the P7 read's seam died twice on dev-box memory (0xC0000005 after 15,185 s; ArrayMemoryError on commit exhaustion).
* **Cost:** section 7. Committed ESTIMATED 57-86 L40-h plus ~23 dev-box GPU-h; conditional +28.6 L40-h and ~12 thor3-h.

### D4. Control reuse (MM, with the REFe MM)
* **Option A, own control:** C_s0 and C_s1 trained here at the same code of record as S. Clean. Costs one extra arm.
* **Option B, reuse P5-C's C0 as C_s0:** C0 has the same base, labels and argv minus our default-off flag. It trains before our wiring lands, so the code of record differs. Reuse is admissible only if control C-ID0b (section 5) shows that our code, with `--stopline-input off`, reproduces C0's code bit for bit on the forward and the first 4 optimizer steps on 64 tuples. C_s1 still trains here.
* **Recommendation: B if C-ID0b passes, else A.** Saves ~28.6 L40-h.

### D5. Tier stamp on this rig (MM ruling, no definition depends on it)
* The rig's families and adherence instruments stamp navtest plans **"T1 (NAVSIM v1.1 plan, self-action open loop; the command is an INPUT of the agent)"** (`PK/eval/command_adherence.py` docstring; `families_run.py` passes `tier="T1"`).
* `SC/BEHAVIOUR_SCORECARD.md` section 0 calls the same navtest proxies **T0**.
* **This card's rule until ruled:** navtest reads are stamped **"T1 first tick (navtest plan from logged inputs, one plan per token)"**, as v8f's "T1-kin first tick". They are a planner diagnostic, never driving performance. Driving claims need the T2-class sim stage (S3), stamped as the scorecard stamps it.

### D6. The T2-class sim stage (PI for time; sim session for the suite)
* It runs only on SL-ADDS (section 6), on the scorecard's nuPlan-seeded TL-G and TL-S slots.
* The suite and thor3 belong to the sim session ("Cosmos render proof and paused T2 suite", per the scorecard owner note in memory). Its suite is paused.
* **Ask:** about 12 thor3-h (section 7). n = 4 traffic-light scenarios can REJECT and cannot ADOPT (scorecard section 4).

---

## MM rulings (2026-11-16, on the draft; the card stays a DRAFT until the PI answers Q1 and Q2)
* **D4: option B if C-ID0b passes, else A.** P5-C's C0 stands in for C_s0 only when our code with `--stopline-input off` reproduces C0's code bit for bit on the forward and the first 4 optimizer steps on 64 tuples (control C-ID0b). On any differing bit, C_s0 trains here (option A, +28.6 L40-h). C_s1 trains here in both cases. The reuse also needs the REFe MM's agreement, since that session owns C0.
* **D5: navtest reads are T1.** `Project Steering/EVAL_DOCTRINE.md` defines T1 as self-action open loop (the planner supplies its own actions, perception fixed at t0) and T0 as consuming recorded future actions. A navtest plan from logged inputs at t0 is T1, so the stamp here is "T1 first tick (navtest plan from logged inputs, one plan per token)", a planner diagnostic and never driving performance. `BEHAVIOUR_SCORECARD.md` section 0 labels the same proxies T0, which disagrees with the doctrine's definitions; the MM tells the REFe MM (scorecard owner). No number in this card is compared across the two labels.

## 1. Question

**Primary.** On the REFe planner, does a map-carried stop-line input (location and type of the next controlled stop line on the commanded lane path) move stop behaviour at navtest? Stop behaviour is read as:
* stop coverage (hit rate on human line-stop windows at a fixed operating point);
* stop-point AUROC;
* stop-gap error;

with the false-stop rate held, and with the cost to every other family cell measured.

**Echo sub-question.** Is the gain carried by the line's specific location (real − donor), by its mere presence (real − withheld), or by future information leaking through the selection (S-PATH pass)?

**Priors (HYPOTHESIS, written before any data):**
* STOP+ at seed 0: MEDIUM. The input carries what vision lacked on PhysicalAI (Q5: four eliminations), and the teacher sees signal states (section 2, L4).
* SL-COSTS: not small. The input says where, never whether. Navtest has many green pass-throughs, and 1,340 + 1,836 = 3,176 of 11,456 audited tokens traverse a signalised connector on the human path (MEASURED, section 2).

---

## 2. Admissibility and leak checks

### 2.1 The input, fixed here (ESTIMATED design, 13-D)
| field | width | definition |
|---|---|---|
| `has_line` | 1 | 1 iff a controlled stop line lies ahead within 80 m on the S-CMD path (2.3) |
| `s_line` | 1 + 8 | arc length from the front-bumper projection to the line along the S-CMD path, clipped to [0, 80] m: s/80 plus 4 Fourier bands (sin, cos), the goal encoder's band count (`n_goal_freq` 4, `PK/refe/model.py:115`) |
| `type` | 2 | one-hot: TRAFFIC_LIGHT (intersection type 1, or a connector listing a type-2 stop polygon), STOP_SIGN (intersection type 2). Type codes: `SC/AMENDMENT_3.md` section 1 |
| `lat` | 1 | the line point's lateral offset in the ego frame, clipped to ±10 m, divided by 10 |

* **Stop-line point:** the START of the controlled connector's baseline in the gpkg, `SC/AMENDMENT_6_STOP_LINE.md` ruling 1. The same definition the scorecard scores against, so input and score cannot disagree on where the line is.
* **Not in the input:** signal state, any frame after t0, the OpenScene route list `roadblock_ids`, the human path, any agent.

### 2.2 L1: does the scorecard's own selection encode the ego's future? Yes. MEASURED.
The scorecard selects the governing connector as *"the first connector traversed on the extended human path that appears in `traffic_lights`"*, where the path is extended *"for up to 60 further frames (30 s)"* past the window (`SC/AMENDMENT_3.md` section 4). `stop_line_s` is a station on the human path (`SC/AMENDMENT_6_STOP_LINE.md` ruling 2).

MEASURED by re-counting the banked rows (`SC/amendment3/feasibility_rows.csv.gz`, 11,456 navtest rows; my counts equal `feasibility_report.json`):

| governing-connector source | rows |
|---|---|
| none | 6,635 |
| `traversed_window` (on the human path inside the window) | 1,340 |
| `traversed_ext` (found only on the human path AFTER the window, up to +30 s) | 1,836 |
| `route_ahead` (the selector's fallback on the frame's `roadblock_ids`) | 1,645 |
| TL-G rows (red, then green, human proceeds) | 69: 38 window / 31 ext |
| **TL-S 5 s rolling rows (human stopped at red to the window end)** | **55: all 55 `traversed_ext`** |

**Reading.** On exactly the stop-at-red population, the scorecard's line exists in 55 of 55 rows only because the human later drove through that connector. As an input that is the PhysicalAI ego-future echo in map form. **The scorecard selection (S-PATH) is therefore forbidden as an input.** It appears here only as the eval-time echo probe (2.4).

### 2.3 The deployable selection: S-CMD (no future, no log route)
1. **Current lane at t0:** the lane whose centreline is nearest the rear axle with heading within 30 deg; ties go to the smaller fid. It uses the t0 pose only.
2. **Walk the lane graph forward from the ego's projection up to 80 m.**
   * At a junction, take the outgoing connector (`exit_lane_fid` = the current lane, per the backwards naming, `SC/AMENDMENT_3.md` section 1) whose map `turn_type_fid` matches the command: LEFT→1, RIGHT→2, STRAIGHT→0.
   * If the command is UNKNOWN, or nothing matches, take the connector with the smallest |dpsi|.
   * Then continue on its entry lane.
3. **The first connector on the walk in a TRAFFIC_LIGHT or STOP_SIGN intersection gives the line** (its baseline start). If the front bumper is already past that start, it is skipped.
4. **Command source per row:**
   * navtest: the token's `driving_command`;
   * navtrain rows: `row["command"]` when present, else the twin's own goal rule (`PK/refe/nav_command.py`).

   The command used to select is always the command the model is fed, checked by G-DVB.

**S-LANE** is the same walk with no command; at a fork it takes all successors. It is computed for the audit only. Where every successor's line start coincides within 0.5 m, S-CMD = S-LANE and the command added nothing. The audit prints how often the command mattered.

### 2.4 L1 echo probe: the S-PATH pass (eval only, never deployable)
* The S arm is evaluated a second time with lines from the scorecard rule (2.2), on TL tokens.
* On rows where S-PATH and S-CMD give different lines (or only S-PATH gives one), (S-PATH pass − S-CMD pass) is the value of future-selected information, paired.
* It is reported and decides nothing, except through control C-LEAK (section 5).

### 2.5 L2: the log route
* The OpenScene frame field `roadblock_ids` is not in `AgentInput` (section 0, D1).
* Its construction from the expert's driven roadblocks is UNVERIFIED. The selector's `route_ahead` fallback finds a connector within 40 m on 1,645 rows where the 30-s human path finds none (MEASURED, 2.2), so it is not a pure crossing echo. It is still a log-derived route whose provenance is unread.
* **Not used.** An S-ROUTE pass is out of scope. If wanted it is a beside pass, with its provenance read first.

### 2.6 L3: label-input overlap
* The windows and the stop-gap metric use the line position, and the input carries it. The timing (does the human stop) comes only from the human future, which is never in the input.
* A model that stops at every line would score well on line-stop windows. It is caught by two committed cost cells:
  * the false-stop rate on GO-at-line windows (the human proceeds through the line);
  * the LEAD-STOP hit rate (stops not caused by a line).
* The donor pass isolates location information from presence.

### 2.7 L4: signal state and the teacher
* **Inputs.** The signal state is not an input (D1).
* **Labels.** The labels come from the DriveRL teacher, which receives signal states: `traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(i + k))` at `PK/refe/build_teacher_rollouts.py:268-269` (MEASURED by reading source).
* **Open.** Whether the teacher's targets stop before red lines is UNVERIFIED. S0-LAB measures it before any GPU is spent, because without line-anchored targets no input can teach line stops.

### 2.8 L5: the command
* NAVSIM's `driving_command` may itself be derived from the future route. The exact OpenScene rule is UNVERIFIED (`PK/refe/nav_command.py` docstring).
* It is the rig's admitted nav input. S-CMD is a deterministic function of (map, t0 pose, command), so its information about the future is bounded by what the command already carries.

### 2.9 L6: the situation path
* REFe has no situation classifier.
* If one is added, or this input is carried to v8f, the stop-line feature must not reach that head. A G-DVB entry asserts no autograd path from `sl_embed` to any classifier head (vacuous on REFe today, enforced if a head exists).

### 2.10 L7: train / inference identity
* One builder (sha-pinned) builds the training sidecar and the navtest sidecar.
* Its header records `selection: "cmd"`, `builder_sha256` and `future_frames_read: 0`. The gate refuses any other header.

---

## 3. Stages and arms

### S0: zero GPU (CPU, dev box; navtrain rows wherever navtrain meta and maps live)
| step | what | output |
|---|---|---|
| S0-B | builder + tests (section 8), sidecars for navtest (12,146 tokens) and the training manifest's rows | `stopline_navtest.npz`, `stopline_train.npz`, headers, sha256 |
| S0-SEL | selection audit: has_line rate; S-CMD vs S-LANE (did the command matter); S-CMD vs S-PATH on TL tokens (agreement within 0.5 m, rows where only one exists) | `S0_SEL.json` |
| S0-COV | coverage: on navtest, human LINE-STOP windows (section 4.1) as a share of all human-stop windows, with n windows and n logs | `S0_COV.json` |
| S0-LAB | teacher label audit: on training rows whose S-CMD line is TRAFFIC_LIGHT, within 40 m, with the governing connector listed red at t0 (label-side use of `traffic_lights`), the share of teacher targets that end stopped (final-pose speed ≤ 0.5 m/s) with the front bumper ≤ line + 1.0 m. Reported with the human's share on the same rows | `S0_LAB.json` |

### S1: training (L40 pod; each run needs a launch-gate PASS)
| arm | init | argv | seed | status |
|---|---|---|---|---|
| **S_s0** | base (D2) | P5-C C0's argv verbatim + `--stopline-input cmd --stopline-sidecar <stopline_train.npz>` | 0 | committed |
| **C_s0** | base | P5-C C0's argv verbatim + `--stopline-input off` (named, default-off rule) | 0 | committed (or P5-C C0 under D4-B) |
| **C_s1** | base | as C_s0 | 1 | committed: the same-flag replicate that sets the training floor F |
| S_s1 | base | as S_s0 | 1 | conditional on SL-ADDS: the lever claim needs it |

All arms run 2,000 steps, as P5-C's approved arm form (INHERITED). The step count is fixed here and never tuned on a result.

### S2: navtest reads (dev box; T1 first tick)
| pass | arm | tokens | role |
|---|---|---|---|
| real | S_s0, C_s0, C_s1 (S_s1) | all 12,146 | committed |
| donor | S_s0 | has_line tokens | echo split: lines from a donor token in a different log of the same city with has_line 1, permutation seed 20261116 |
| withheld | S_s0 | has_line tokens | the 13-D input all zero (what S sees with no line) |
| S-PATH | S_s0 | TL tokens with an S-PATH line | echo probe (2.4) |

### S3: T2-class sim (conditional on SL-ADDS; sim session; PI go)
* S_s0 vs C_s0 on the scorecard's tier-2 TL-G 2 + TL-S 2 slots, with the baseline twice (scenario seeds 20261107 / 20261108), per scorecard sections 3-4.
* Primary rules: c2 (stop on red), c1 (no stall when clear), c4 (launch).

---

## 4. Read

### 4.1 Windows and labels (labels may use the human future and the signal state; inputs may not)
At each navtest token t0, with s measured along the S-CMD path:
* **LINE-STOP:** has_line, and the human is stopped by t0 + 4 s (speed ≤ 0.5 m/s at the last scene pose), with the front bumper ≤ line + 1.0 m, and no stopped or slowing lead in the corridor between the ego and the line. "Lead" is the scorecard's definition, `SC/BEHAVIOUR_SCORECARD.md` section 1.
* **GO-AT-LINE:** has_line, and the human's front bumper crosses the line within 4 s.
* **LEAD-STOP:** the human is stopped by t0 + 4 s with a lead in the corridor ahead (line or not).
* Other has_line windows are reported beside.
* **Stop score:** score = −r, where r = planned travel distance over 4 s / max(v0 · 4 s, 4 m), on the executed plan the seam scores (8 poses at 0.5 s).
* **Plan "stops":** the speed between the last two poses is ≤ 0.5 m/s. This operating point is fixed and never tuned.

### 4.2 Committed cells (S − C, paired on identical tokens)
| cell | family | definition |
|---|---|---|
| **AUROC** | TACTICAL | AUROC of the stop score, LINE-STOP vs GO-AT-LINE |
| **HIT** (coverage) | TACTICAL | the share of LINE-STOP windows where the plan stops |
| **FS** (cost) | TACTICAL | the share of GO-AT-LINE windows where the plan stops |
| **LEADHIT** (cost) | TACTICAL | the share of LEAD-STOP windows where the plan stops |
| **GAP** | LONGITUDINAL | on LINE-STOP windows where the plan stops: \|plan stop station − human stop station\|, m. Overshoot past the line is counted beside |
| **RED-X** | TACTICAL | the share of has_line windows with the governing connector listed red at t0 where the plan's front bumper passes the line within 4 s |
| **PDMS** guard | all | NAVSIM PDMS ×100 and sub-scores NC, DAC, EP, TTC, C |

### 4.3 Four families per band (T1 first tick; the plan horizon is 4 s: `horizon_steps` 20 at 5 Hz, `PK/refe/model.py:94`)
| family | operative 0-2 s | tactical 2-4 s (4-6 s: n/a, beyond the plan horizon, n = 0) | strategic 6-30 s |
|---|---|---|---|
| **LONGITUDINAL** | speed error, along-track; headway, time-gap and TTC to the lead (`PK/eval/instruments/families_run.py` → taniteval `four_families_block`) | the same + GAP + EP | n/a at T1 first tick (4 s plan), n = 0; S3: scorecard a1, a2, b1, b2, d2, c4 |
| **LATERAL** | heading, curvature, yaw-rate error, cross-track (same instrument) | the same + DAC | n/a at T1 first tick; S3: e1, e2, d1 |
| **TACTICAL** | AUROC, HIT, FS, LEADHIT, RED-X, NC; executed stop/go decision at the line vs human (confusion with n) | the same cells | n/a at T1 first tick; S3: c1-c4 and the executed-decision detector's LINE_R / LINE_G classes (`SC/tactical_events/tactical_detector.py`) |
| **STRATEGIC** (in full; the pause ended 2026-11-16) | (i) route decision executed: command adherence, turn-active accuracy and Cohen's kappa with the 3x3 confusion (`PK/eval/command_adherence.py`; 12,146 tokens, turn-active 4,076, INHERITED counts). PREREG_P6 files this row under TACTICAL; this card reads it once, here. (ii) **goal / route setting quality: NOT COMPUTABLE, n = 0**: REFe sets no goal or route; the route is the supplied command | the same rows over the 4 s plan | T1 first tick: n/a, n = 0 (4 s plan). S3: route completion over the 15 s scenario (the 6-15 s part of the band) and junction-turn agreement with the command; the 15-30 s part stays n/a (scenario length) |

### 4.4 Estimators, one per question (each interval names the question it answers)
* **Another draw of LOGS:** the paired log-cluster bootstrap. B 10,000, seed 20260927, logs sorted, `rng.integers(0, G, G)` per resample: the stream of `proxy_eval.boot` (`PK/eval/proxy_eval.py:313-327`, called by `stage3_full.py` with N_BOOT, SEED = 10000, 20260927; INHERITED from `PK/eval/command_adherence.py`).
  * AUROC, HIT, FS, LEADHIT, GAP and RED-X are computed per resample on the same resampled logs, the way `command_adherence.py` computes kappa.
  * The family block uses its own estimator, taniteval's episode-cluster bootstrap with eid = log_name and B 2,000 (rig convention; its seed is pinned at freeze, UNVERIFIED here).
  * 95 % percentile intervals throughout.
* **Another TRAINING RUN:** the floor F(cell) = |C_s1 − C_s0| on the same tokens. An effect counts only above F (`H-ESTIM-SEED-1`; v7-tiny replicate 4 of 42 = 9.5 % "separated", `CLAUDE.md`). A lever claim needs S_s1 to agree.
* **Another INFERENCE RUN:** zero by construction. The REFe planner is deterministic: a replicate was bit-identical on 265 / 265 tokens (INHERITED, `SC/BEHAVIOUR_SCORECARD.md` section 3.2). Control C-DET re-checks it per arm.
* **S3:** scorecard section 4 (paired bootstrap, cluster = scenario / log, 10,000 draws, seed 20260927; sim floor from the two baseline seeds).

---

## 5. Controls, each with the value it must read (evaluated first; any miss refuses the read)
| id | control | must read |
|---|---|---|
| **C-LEAK** (deliberate regression) | Every frame after t0 is shuffled in the scene dicts and the sidecar is rebuilt. Mutation: the S-PATH builder under the same shuffle | deployable builder: sidecar **bitwise equal** on all tokens; S-PATH builder: **changes on ≥ 1 token** (otherwise the test is blind) |
| C-CMD | (a) constructed fork whose branches have different line starts, command flipped; (b) shared-line fork | (a) feature changes; (b) feature bitwise equal |
| C-A6 | S-PATH builder against the scorecard's `stop_line_xy` on the TL-G and TL-S pools (A6 definition) | max distance ≤ 0.01 m |
| C-MAP | `map.gpkg` sha256 equal between the training venue and the dev box | equal |
| **C-ID0** | S_s0 at step 0 (zero-init `sl_embed`) against the base, plans on 64 navtest tokens | bitwise |
| C-ID0b (D4-B only) | our code with `--stopline-input off` against P5-C C0's code: forward + 4 optimizer steps on 64 tuples | bitwise, else D4-A |
| **C-ZERO** | G-LIVE with an all-zero sidecar | `sl_embed.weight` gradient **exactly 0.0**; with the real sidecar **non-zero** |
| **C-PLANT** (planted signal, model level) | G-PLANT at gate time: 64 tuples whose targets are replaced by "stop at s_line − 2 m", 200 steps, real input vs donor input | real: stop-station error < 1.0 m; donor: > 3.0 m |
| C-PLANT-R (planted signal, read level) | a synthetic dump whose plan stops exactly on LINE-STOP windows | AUROC 1.0, HIT 1.0, FS 0.0; the outcome code returns SL-ADDS |
| **C-SELF** (identity) | S_s0 vs S_s0, and real vs a donor permutation equal to the identity | dAUC, ΔHIT, ΔGAP exactly 0.0, CI [0.0, 0.0] |
| C-LABEL | the label appended as a score | AUROC 1.0 |
| C-DET | each arm's seam re-run on 64 tokens | bitwise |
| C-UNPAIRED (deliberate regression, tests) | the decision code reading S's CI lower bound against C's point | goes red |
| C-COUNT | n windows / n logs per class, per pass | printed; the S2 read refuses if LINE-STOP < 100 windows or < 30 logs (then S0-COV's branch applies) |

---

## 6. Committed outcomes (each names what it triggers; none ends at a refutation)
**Predicates** on the S_s0 − C_s0 pair at T1 first tick:
* **STOP+:** at least one of
  * dAUC point ≥ max(+0.020, F) and CI lower bound > 0;
  * ΔHIT point ≥ max(+0.05, F) and CI lower bound > 0.

  The +0.020 is STOPPERC's RES-ADDS bar, reused unchanged.
* **ECHO-SKILL:** on S_s0, real − donor has a CI lower bound > 0 on AUROC or HIT.
* **GUARD:** ΔPDMS lower CI > −0.5 (×100), the programme's bar (`SC/BEHAVIOUR_SCORECARD.md` section 4.3). Also, no family cell (incl. FS, LEADHIT, RED-X) is separated worse beyond F.
* **ECHO-LEAK:** on disagreeing rows, S-PATH − S-CMD has CI lower > 0. It is reported; it is never a pass.

**S0 outcomes (before any GPU):**
| outcome | rule | triggers |
|---|---|---|
| **LABELS-NOT-ANCHORED** | S0-LAB teacher share < 0.50 (ESTIMATED bar, fixed here) | S1 does not launch. Next lever: labels that stop at the line. The rig's other teacher route (RJ5 → Tc20s → P5-R, the P5-C C1 label path) is audited the same way on CPU, then a label card goes to the PI with its cost |
| **POWER-SHORT** | S0-COV LINE-STOP < 100 windows or < 30 logs on navtest | Add P7's session reserve (16 sessions, 10,141 scenes, unseen in training; INHERITED, `PK/SPEC_P7.md`) as a second read population, with the REFe MM's agreement; it is fixed at freeze before S1 |
| PROCEED | otherwise | S1 |

**S2 outcomes:**
| outcome | rule | triggers (committed now) |
|---|---|---|
| **SL-ADDS** | STOP+ and ECHO-SKILL and GUARD | (1) S_s1, because a lever claim needs both seeds; (2) on both seeds, S3 (PI go, sim session); (3) the product card: a vision stop-location head trained on nuPlan cameras with the map line as its LABEL (vision at inference) |
| **SL-PRESENCE** | real − withheld separated on AUROC or HIT; real − donor not; GUARD | The model uses presence and type, not location. Next arm: location through the decoder (the line point as a cross-attended map token) plus a stop-gap auxiliary loss, under its own card |
| **SL-COSTS** | STOP+ and not GUARD (FS, LEADHIT or PDMS worse) | The input says where, not whether. Next lever: a vision traffic-light-state auxiliary on the camera tokens, with OpenScene `traffic_lights` as its LABEL (vision at inference), under its own card |
| **SL-NULL** | not STOP+, GUARD holds, C-PLANT passed | The input reaches the plan (C-PLANT) but training did not use it. Next lever, by S0-LAB's number: dose (4,000 steps) if the labels are line-anchored, else the label lever above |
| **SL-REGRESSES** | not STOP+, and GUARD fails | REJECT. Next arm: input dropout (withheld with p 0.5 at train) |
| **INVALID** | any control misses | No read. Fix the control's cause and re-read the same dumps |

* SL-ADDS at seed 0 alone is reported as **ONE-SEED** (scorecard section 4 rule ii).
* S3 can REJECT with n = 4. It cannot ADOPT.

---

## 7. Cost (ESTIMATED from the rates below; venue named)
* **L40 step time:** 51.2-52 s/step for P7 at batch 4, accum 64 (INHERITED, pod `train.log` reads quoted in LOOP_STATE_P6 entries for steps 508, 1,224, 2,744, 6,907 and 6,964). One 2,000-step arm ≈ 2,000 × 51.5 s ≈ **28.6 L40-h** (ESTIMATED). The PI-approved P5-C arm form was ~27 L40-h each (INHERITED).
* **Navtest seam:** 1,587 tokens in 2,149 s on the dev box (INHERITED, the P7 tail probe, LOOP_STATE 2026-11-16 11:35Z) → 1.35 s/token → 12,146 tokens ≈ **4.6 h** plus ≥ 1 h scoring (ESTIMATED).
* **The has_line share** is bounded above by the 4,821 of 11,456 rows with any governing connector (MEASURED, 2.2); S-CMD's own share comes from S0-SEL.

| stage | venue | committed | conditional |
|---|---|---|---|
| S0 | dev-box CPU (navtest); navtrain rows where navtrain meta + maps are (the pod, which already holds a map table for the label runner, INHERITED; UNVERIFIED for the full gpkg set) | builder ~1 agent-day; CPU 2-25 h (ESTIMATED, navtrain row count unread) | |
| S1 | REFe L40 pod after P5-C | S_s0 + C_s1 = 57.2 L40-h (D4-B), or + C_s0 = 85.8 L40-h (D4-A); G-LIVE + G-PLANT ≤ 2 L40-h | S_s1 +28.6 L40-h |
| S2 | dev-box GPU (shared lock) | 3 full seams ≈ 17 h + 3 subset passes ≈ 5.5 h ≈ **23 h** | S_s1 seam ≈ 5.6 h |
| S3 | thor3 (sim session) | none | 12 scenario runs × ~1 h ≈ **12 thor3-h** (rate INHERITED, scorecard section 3) |

* **Calendar (ESTIMATED):** the L40 pod is busy with P5-C C0 and C1 (~57 h), plus C0r if it runs. S1 starts no earlier than ~3 days out. S0 needs no GPU and can start on the freeze.

---

## 8. Reachability plan (the rule: a mechanism that is tested is not proof that it runs)
**Hop table** (anchors at `68850034`, `PK/refe/`; the freeze re-anchors them):
| hop | where | proof |
|---|---|---|
| 1 argparse | `train.py`, beside `--nav-mode` (`:2211`, whose own default is `"goal"`): `--stopline-input {off,cmd}` **default `off`**; `--stopline-sidecar PATH` default None; `cmd` without a sidecar raises | G-HYG; the mutation "cmd, no sidecar" must raise |
| 2 config | `REFeConfig` (`model.py:58-205`): declared fields `stopline_mode`, `stopline_dim` 13; an undeclared attribute is refused (`model.py:200-204`) | G-HYG typo arm (`stopline_mod`) raises |
| 3 build | `model.py` beside `cmd_embed` / `ego_enc` (`:537-541`): `self.sl_embed = nn.Linear(13, dec_width, bias=False)`, zero-init, built iff mode ≠ off | G-DVB |
| 4 forward | `model.py:771`: `ego_tok = ego_enc(cat[ego, cmd_embed(goal)]) + sl_embed(sl)`; mode on with `sl` None raises | C-ID0 bitwise at step 0; deleting the `+ sl_embed(sl)` term must turn G-LIVE red |
| 5 data | TargetBank row → `sl` from the sidecar by token (twins: built with the twin's command) → batch | G-DVB data entry; the "all rows get row 0" mutation is red (feature variance across the batch must be > 0) |
| 6 loss | unchanged WTA + score loss; **both** backward sites (`train.py:1911` online deferred, `:1930` plain) | G-LIVE: non-zero `sl_embed` gradient through each site separately; C-ZERO reads exactly 0.0 |
| 7 identity | `IDENTITY_ARGS` / `IDENTITY_DEFAULTS` (`train.py:1071`, `:1079`) gain the new dests | resume / identity tests |
| 8 eval | `planner.py`, the seam harness, `eval_checkpoint.py` and G-EVAL build `sl_embed` and read the navtest sidecar; `--stopline-eval {real,donor,withheld,path}` exists on the eval side only | G-EVAL: identical model; a seam run without `sl` on an S checkpoint raises |
| 9 checkpoint | `sl_embed` in the state dict | G-CKPT round trip bitwise |

**Default-off rule.**
* `--stopline-input` defaults to `off`, the class that silently disables a wired lever.
* Every arm names it in its argv: `cmd` for S, `off` for C. An S argv without it is refused by G-DVB ("declared cmd, built off" cannot happen silently).

**G-DVB entries** (`refe/gate_checks.py` `DVB_KINDS`, new profile `refe_sl` in `refe/launch_gate.py` cloned from P5-C C0's profile; a flag without an entry is refused by the gate itself):
| dest | declared = built | mutation that must FAIL |
|---|---|---|
| `stopline_input` cmd | `sl_embed` present, shape (dec_width, 13), all zero at step 0; the forward reads `sl` | flag cmd with the module not built; module built with flag off |
| `stopline_sidecar` (data) | sha256 equals the pin; key coverage of the training manifest ≥ 99.9 % (printed); header `selection == "cmd"`, `future_frames_read == 0`, `builder_sha256` equals the pin; the command per row equals the fed command | an S-PATH sidecar (header `path`); rows shifted one token; a row whose selection command differs from the fed command |
| `stopline_dim` 13 | the Linear's `in_features` equals the sidecar width | width 12 |
| no classifier path (L6) | no autograd path `sl_embed` → any classifier head (vacuous on REFe today) | wire `sl` into a classifier head |

**Other gates.**
* **G-LIVE** (finite losses; non-zero `sl_embed` gradient; C-ZERO);
* **G-PLANT** (C-PLANT);
* **G-EVAL**, **G-CKPT**;
* **G-SUITE** (clean-tree tests, 0 regressions);
* one PASS token per run, bound to commit, argv sha256 and the sidecar sha256.

The trainer wiring lands only after P5-C's last arm trains (D3).

---

## 9. What this card does not license
* "The REFe student drives better at lights." Only S3's T2-class read speaks to driving, and only to reject at n = 4.
* Any claim on PhysicalAI-AV or v8f. That corpus has no map, and the transfer path is the vision stop-location head (SL-ADDS trigger 3).
* Comparison with the paper's 94.55 or the NAVSIM sensor-agent leaderboard.
* A signal-state input in any form.
