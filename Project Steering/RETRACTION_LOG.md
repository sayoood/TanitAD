# RETRACTION LOG — append-only

**Purpose: this is the program's self-learning mechanism.** A retraction that only records *what* was
wrong teaches nobody. Each entry records the **ROOT-CAUSE CLASS**, because the class is what recurs.

**Binding rule (CLAUDE.md §Operating standard 4): before asserting in a known class, read this file.**
Append; never delete. A wrong claim that stays visible is worth more than a tidy document.

---

## The classes, ranked by how often they have bitten us

| # | class | recognition signal |
|---|---|---|
| **C1** | **Faster-moving source than the harness** — quoting a trainer log, a HUD overlay, or a progress print as an eval result | the number came from something that updates every 50 steps |
| **C2** | **Absence from a single probe** — one path, one process name, one directory | "X does not exist" after exactly one check |
| **C3** | **Mechanism instead of measurement** — a plausible causal story asserted as fact | the sentence contains "because" and no artifact path |
| **C4** | **Inherited without re-verification** — prose, a summary, or another agent's claim quoted as primary | cannot name the file the number lives in |
| **C5** | **Scalar off a noisy curve at one point** — single-row reads, unstable fits | a ratio or exponent with no window, R², or n |
| **C6** | **Confounded comparison** — the contrast varies more than one thing | two arms differ in ≥2 respects and only one is named |
| **C15** | **A tensor's semantics taken from its NAME rather than from its construction site** | a `.pt` key is used as if its name were a specification, and the only check run is one the name would pass anyway |

---

## Entries

| date | retracted claim | class | what it cost / would have cost |
|---|---|---|---|
| 07-21 | *"v1.6 ADE 0.4420 — best in the program"* | **C1** | trainer in-loop val, **~10 % optimistic** vs canonical (0.4886 heldout). Entered MODEL_REGISTRY as a headline — the one thing §0 forbids |
| 07-21 | *"v1.6 failed decisively"* (step-2500 spike) | **C5** | transient; it recovered and finished at parity. Nearly killed a healthy run |
| 07-21 | *"v3enc's failure is a generalisation gap (in-sample 0.79–0.85 vs held-out 0.393)"* | **C3/C4** | `ego_linear_r2` is an in-batch fit reading **0.595 on a randomly-initialised encoder**; v1 never logged it — no control existed |
| 07-21 | *"v1 reached v3enc's level at step 450 (~23×)"* | **C5** | single row from a curve swinging 0.38–0.82. Bucket means say **~3.5–5×**. Had propagated into the registry |
| 07-21 | *"decorr strangled speed capacity"* (D-031) | **C3** | `decorr_w = 0.0 if step < 10000` — **never on**. A restart aimed here would have burned the last budget on a refuted lever |
| 07-21 | *"the broken route labels explain v3enc"* | **C6** | `nav_valid_frac` is 0.21–0.25 in **all four arms including the deployed v1** that scores probe 0.861 |
| 07-21 | *"our pods cannot render / no EGL devices"* (dated 07-09) | **C2** | Vulkan ICD is in `/etc/vulkan/icd.d/`, not `/usr/share/`. **Blocked AlpaSim + CARLA for 12 days** |
| 07-21 | *"no agent state exists; `lead_state` is a stub"* | **C2** | `obstacle.offline` — 3D tracks on **96.90 %** of our corpus. Our ingest reads **2 of 36** features |
| 07-21 | *"CEM planning is infeasible — 723 ms, 7× over budget"* | **C3** | measured **20.8 ms** at K=8. My arithmetic assumed re-encoding per candidate; you encode once and broadcast. Nearly killed v3's whole thesis |
| 07-21 | *"three code sites BLOCK CUDA-graph capture"* | **C3** | capture succeeded with **exact** equivalence — allocations inside a capture use the graph's private pool |
| 07-21 | *"one graph over 20 steps beats per-step capture"* | **C3** | differ by **7.7 µs/step**. Inter-step CPU round-trips were never the cost |
| 07-21 | *"encoder caching → 84.74 ms"* | **C5** | measured **95.11**; stage arithmetic overestimates when the encoder hides behind rollout launches |
| 07-21 | *"the fan is a SPEED fan ⇒ strategic choice is a ~2 % lever"* | **C6** | REF-C evaluates with `nav_cmd=None` → a decoder that never had a route input learns the marginal. **Nearly designed the hierarchy away** |
| 07-21 | *"VLM turn detection 89.3 %"* | **C4/C6** | does not reproduce (**80.6 %**), *and* it was AGREEMENT not RECALL — on a 74 %-straight corpus, always-"straight" scores ~74 % while detecting zero turns |
| 07-21 | *"6 of 9 ROUTE tokens never minted"* | **C4** | it is **4 of 9** |
| 07-21 | *"R11: `combined_tick_harness.py` is not in HEAD"* | **C4** | asserted from a triage doc predating the merges, without running `git ls-files`. Written **into the registry while writing the section that forbids it** |
| 07-21 | *"deploy tick 11.16 ms / 89.6 Hz"* (propagated 2 days) | **C1/C6** | a 1-frame encode + K=9 select, on a **different GPU, checkpoint and corpus** — not the 20-step rollout the leaderboard scores. Real planning tick: **103 ms** |
| 07-21 | *"REF-C-XL finishes 0.006 m behind flagship v1"* | **C6** | a difference of **split-means**; full-set gap is **0.0443 m** and the paired test says **not separated** — they tie |
| 07-21 | *"`obstacle.offline` unblocks agent-relational tactics ⇒ ingest all 197 chunks this week"* (`DATA_STRATEGY_FOR_HIERARCHY.md` §0.2/§7 row 1) | **C3** | a mechanism ("agents constrain the ego") priced at **12.4 GB + 2–3 eng-days** with no measurement behind it. The pre-registered gate reads **+1.16 % [−0.92, +3.19]** on the 2 s longitudinal target, ≤ +1.83 % out to a **6 s** horizon. The *labels* are exactly as described — the *inference from them to tactical prediction* was the unmeasured step. Cost of finding out: **1.1 GB, 0 GPU-h** (`Research/2026-07-21-lead-state-gate.md`) |
| 07-21 | *"LAL-v2 is implemented but UNMERGED — 12 days idle"* (E5; escalated twice, incl. into `V4_FLAGSHIP_DESIGN.md` P5e + escalation 1b) | **C4 + C2** | **It merged on 2026-07-09, the day of the intake** (`3784e34`; `stack/tanitad/eval/metrics.py:202-251`; `stack/tests/test_lal_v2.py` green). Inherited from `TANITEVAL_V2_METRIC_SUITE.md` §7 E5 — which asserts "unmerged" **on the same line that cites the merged file path** — without running `git ls-files` or grepping the target module. Second probe never taken. Would have spent an eval-agent's session re-merging code that was already in HEAD. ⚠️ **Mirror image of the stranding failure the Agent Operating Standard was written for: a stale escalation demanding work git already contains.** The real residual is **one line** (`taniteval/taniteval/rollout.py:94` keeps 4 of 20 steps) |
| 07-21 | *"v4 stands at **1** encoder-touching lever of a limit of 2"* (§9.1, §12.1 #6) | **C4** | a **duplicated, orphaned table** in `V4_FLAGSHIP_DESIGN.md` §9 totalled 1 while the table directly above it totalled 2; the "1" was then quoted forward twice. **The strict count is 2 of 2 — the door is CLOSED.** The under-count is the dangerous direction: it advertises headroom for a third lever, which is [PM] DO-NOT-CARRY #6 and the *"actual repeat root cause"* that cost v2 and v3enc their attributability |
| 07-21 | *"v1 trained at 6.37 s/step"* (`summary.json:wallclock_s`, carried in REGISTRY §1.2 against `GATE_PROTOCOL.md`'s 10.89 — "UNRESOLVED" for a day) | **C4** | **Settled by arithmetic once the log reached the repo: 10.888 s/step** (`sum(step_s)` = 326,638.2 s = 90.73 h over 29,999 steps). In-loop step time is a **subset** of wallclock, so 90.73 h of it cannot sit inside the claimed 53.1 h wallclock — `wallclock_s` is not the full run's. ⚠️ Do **not** re-derive from `median(step_s)/50` either: that gives 9.240, a 15 % under-read, because `step_s` is accumulated over `--log-every`. Every GPU-day estimate in the program depended on this |
| 07-22 | *"TanitDataSet-C PUSHED to HF"* (LOOP_STATE item 0 + chat, 00:32) | **C1 (async-completion overclaim)** | Reported an async action as DONE while it was still running — then it **died committing 0/17 files** (uploader was a child of the Claude session; a restart killed it). The repo was created but **empty**. ⚠️ New sub-pattern of C1: *"launched" ≠ "completed"* for uploads/trainings/pushes. **Verify the terminal marker** (`committed: N/N`, `UPLOAD_COMPLETE`, `"done": true`) before claiming completion — never the launch log. Relaunched detached 07:00 |
| 07-22 | *"REF-C collides at-fault in closed loop"* (headline in chat + LOOP_STATE, from **n=1**) | **C5** | The n=1 scene (`01d503d4`, 41-actor highway) was the **worst case**; the n=12 AlpaSim suite shows **33 % at-fault (4/12) — REF-C passes ~half** (base 6/12). The "both collide" headline over-read a single worst-case scene. ⚠️ **A closed-loop failure rate from n=1 is scene-dependent — caveat as worst-case until n ≥ ~12.** I *did* flag "n=1 directional", so this is a refinement not a false claim — but the headline propagated into the live state before the suite corrected it. The durable read: REF-C **fails ~half** closed-loop, and **base ≥ XL** (score 0.345 vs 0.246) — scale gives no closed-loop advantage. |
| 07-22 | *"REF-C fails ~half closed-loop / collides closed-loop"* (chat + LOOP_STATE + leaderboard §5.5) | **C6** | The open-loop-in-AlpaSim control (Sayed's idea) measured REF-C's open-loop ADE **on the AlpaSim reconstructions** = **1.52, 3.21× the real-footage 0.47** (consistent across 4 scenes, 288 preds). REF-C is fed NuRec reconstructions **~3× off its training distribution** → the closed-loop failure rate **confounds model quality with reconstruction-OOD**; the base-vs-XL *ordering* survives (same OOD both) but "REF-C collides" is NOT a clean model indictment. ⚠️ **A closed-loop eval on reconstructed scenes measures model × reconstruction-fidelity — run the open-loop-vs-known control BEFORE attributing failure to the model.** Corroboration: flagship v1 (same OOD input) PASSES the scene REF-C crashes (n=1, rollout `71f9740c`). |
| 07-23 | *"flagship v1 beats REF-C closed-loop / drives where REF-C collides"* (chat + LOOP_STATE + leaderboard §5.5 + imagination synthesis §8, from **n=1** rollout `71f9740c`) | **C5** | The n=1 scene (`01d503d4`, wide highway) was a **lucky scene**. The n=12 PAIRED AlpaSim suite (same 12 scenes, same NuRec input, f-theta verified live) REVERSES it: **REF-C base beats flagship v1** — pass **8/12 vs 2/12**, mean score **0.496 vs 0.066**, paired Δ **−0.430 [−0.646, −0.215]**, sign-test **8-0** (p=0.008), pass-McNemar 6-0 (p=0.031); **collisions TIED** (1-1, p=1.0). Mechanism (MEASURED, `flagship_vs_refc_suite_results.json`): v1's tactical head is a **high-deviation planner** (plan_dev **1.12 vs 0.34**, 3.3×) → failure mode is **offroad, not collision**; the lone n=1 wide swerve that avoided a collision does **not** generalize. ⚠️ Same class as the 07-22 REF-C-n=1 entry directly above — **a closed-loop win/loss from n=1 is scene-dependent; never headline it until n ≥ ~12.** Still within-sim RELATIVE / ~3.2× OOD (isolates the planner, not a real-world rate). Residual confound: **480×854 vs native 1080×1920** (the n=1 pass was native) → a native-res paired re-run is the cheapest discriminator. |
| 07-23 | *"the own-encoder ablation died silently — PID gone, GPU idle, no results"* (orchestrator, mid-loop; I sent a relaunch instruction) | **C2** | It had **COMPLETED**, not crashed: both experiments hit the terminal `ALL_CAMCOND_DONE` marker and wrote `camcond_{rig,multirig}.json` — into `/workspace/tmp/idm/`, NOT the `/workspace` root I `ls`'d. **"PID gone + GPU 0 MiB + file-not-seen" is *identically* the success signature** (the bash launcher exits and frees the GPU on completion) and the failure signature — the discriminating checks are the **terminal marker** and the **artifact in the RIGHT dir**, neither of which I ran before alarming. The owning agent verified completion and correctly REFUSED my relaunch, saving a wasted re-run of a decision-grade experiment. ⚠️ **Before declaring a detached job dead: grep its terminal marker + `find` its output artifact across subdirs — not just `ls` one dir + check `/proc`.** Mirror of the 07-21 "our pods cannot render" C2 (probed `/usr/share`, not `/etc`). |
| 07-24 | *"from-scratch canary DESCENDING, co-evolution CONFIRMED"* (chat + LOOP_STATE, from **n=1** eval point) | **C5** | Read a "clean descent" from a SINGLE point (step-500: 15.67→9.15); the step-1000 point **bounced to 10.70** — the plan-free canary is NOISY pre-coupling (Phase A, λ_plan=0). Same class as the 07-21 v1.6 step-2500-spike entry: **a trend from one eval delta is not a trend.** ⚠️ The co-evolution claim ITSELF holds on the right signals (val-ade 2.72→0.89, wm-loss 29.6→4.64, oracle 0.70→0.52 — all monotone-improving); it was the words "canary descending CONFIRMED" that were the n=1 over-read. **The decisive canary read is Phase-B (step 2000+), when the planner couples — that is where v4.x degraded.** Corrected same-iteration (2 points), pre-Sayed-decision — cost 0, but the headline reached chat. |
| 07-24 | *"CEM search over the frozen WM = 0.132 (4.5×) → the planner is the headroom on a frozen WM"* (chat + LOOP_STATE + amortised-MPC doc "product path") | **C6** | The search arm **peeks at the expert's ACTUAL future** as its cost (hindsight-privileged); the deployable feedforward W does not. So 0.132-vs-0.599 varies **two** things — planner quality AND access-to-the-future — not one. The learned-value CRUX (`ade3edfb`) settled it: a DEPLOYABLE search (learned value, no GT future) = **1.02, SEP-WORSE than W 0.599** (value learns only E[cost]=mean trajectory; CEM adversarially fools it). Every deployable frozen-WM route hits the **~0.60 aleatoric wall** (feedforward 0.599 · bigger 0.60 · distill 1.40 · learned-value 1.02). The W→0.132 gap is **prediction-vs-hindsight, NOT controllable planning headroom.** ⚠️ **A privileged-input arm is not a headroom estimate** — name the input asymmetry before quoting a gap as "headroom". Amortised-MPC "product path" superseded. (A learned value CAN pay off in a genuine CLOSED-LOOP setting where the ego controls the future — different eval, needs a sim/reward.) |
| 07-24 | *"the closed-loop program is gated on ONE thing — a faithful low-OOD renderer"* (07:57 program report §3 + chat; synthesis of Gate-1/D2/RefcCL) | **C3** | Over-generalized — it's **two separable problems**, and I never probed whether an asset we ALREADY have serves. **(A) road-keeping/drift** (the D2/RefcCL Pareto trade) needs low-OOD + **on-policy** but NOT reactive agents → our real-footage harness (on-policy OOD **1.02–1.19×** vs NuRec 3.2×) IS a sufficient low-OOD *training* source, **renderer-free** — it was just eval-only. Only **(B) reactive-agent COLLISION** truly needs a renderer. The real escape is **on-policy training** (correct); "needs a renderer" was the over-broad part. Caught same-day by my own commissioned research (`ac2f8f58`); the `LOWOOD-CL-TRAIN` test (running) settles (A) for ~1 pod-day. ⚠️ **Before quoting a capability as "blocked / needs new X", probe whether an existing asset already serves it.** **→ UPDATE (same-day, `a1f26c92`): the test came back BOUND — on-policy training on our instrument does NOT close (A) (base rarely fails → objective starved). So (A) DOES need a better instrument (map/tolerance-band). The reframe was directionally useful (worth the cheap test) but its "(A) doesn't need a renderer" over-corrected; the original "needs a better instrument/renderer" STANDS. Net lesson survives: probe the existing asset — we did, cheaply, and it was insufficient.** |
| 07-24 | *"the closed-loop-improvement direction is DECISIVELY CLOSED; the recovery lever is Pareto-bound on EVERY axis"* (chat + LOOP_STATE; from D2/RefcCL/LOWOOD-CL all showing raw-ADE "worse") | **C3 (over-claimed closure)** | The tolerance-band re-score (`a1f26c92`) showed the "ADE trade" was **largely a knife-edge-L2-metric artifact** — under a fair lane-tolerance band the ADE-cost VANISHES (CI∋0) for 3/4 configs, −74% for naive. NOT a hard Pareto wall; the real residual is a small, **n=12-underpowered departure signal** (a measurement question, cheap). ⚠️ **META-PATTERN this session (my 4th over-claim of closure): "canary descent confirmed" (C5) · "planner is the headroom" (C6) · "closed-loop needs a renderer" (C3) · now "closed-loop closed"** — a CHEAP follow-up reopened EACH. **Lesson: before declaring a direction "closed / bound / resolved", run the cheapest metric-or-power check FIRST — the closure claim is the single one most worth a $0 test.** Cost each time: ~0 (caught same-session), but the firm claim reached chat/reports. |
| 07-24 | *"D2/RefcCL recovery-augmentation HALVES held-out lane-departures + generalizes (beats Gate-1's memorization)"* (chat + LOOP_STATE + 07:57 & 12:57 program reports; from **n=12**) | **C5** | The departure "win" (+0.0089 **S** at n=12) **REVERSES at n=40** (2-fold cross-fit, 1.83× power): **−0.0302 S — the recovery FT departs 3.3× MORE**, not less; ADE worse under both metrics. The n=12 held-out win was **favorable-split noise** — a ~1 pp departure effect on 12 episodes is underpowered. Confound flagged: cross-fit trains 20-ep vs the original 28-ep folds (part is data-reduction), but the unbiased full-corpus estimate is neg+separated → not robustly promotable. ⚠️ **Second measurement lesson: n=12 held-out is underpowered for ~1 pp departure effects — use full-corpus cross-fit (all-40 held-out) for EVERY closed-loop claim** (pairs with the tolerance-band-ADE lesson from the C3 above). Same-day whipsaw with that C3: the tolerance re-score REOPENED the direction (ADE-cost = metric artifact), this powered eval CLOSED it (departure-benefit = noise). Net: recovery-aug is NOT a net win on road-keeping at full power. Durable + un-retracted: the machinery, encoder-FT-safety, the 2 metric lessons. |
| 07-25 | *"the `git commit -- <pathspec>` segfault triggers at 178+ files"* → then *"no, it is the pathspec SHAPE (space-containing dirs / multi-pathspec)"* — **BOTH asserted into CLAUDE.md, both WRONG** (commits `e405cd4`, `2c44ae6`) | **C8 — premature root cause from too few points** *(same class as the C8 Glob artifact, and a cousin of C5)* | The crash is **INTERMITTENT (~50 %), independent of count and shape**: the *identical* single-file `git commit -- CLAUDE.md` **segfaulted on attempt 1 and succeeded on attempt 2**. Each wrong theory was consistent with every point I had *at that moment* (81/149 ✓ vs 178/677 ✗; then single-file ✓ vs 2-file ✗) — which is exactly why a pattern over ~4 observations of a flaky process is not a root cause. ⚠️ **BINDING LESSON: before asserting a mechanism for an intermittent failure, RE-RUN THE IDENTICAL FAILING COMMAND. One repeat would have shown flakiness and prevented both claims.** *(Cost: ~3 doc rewrites, no data loss. Third premature-certainty error this session — the recurring failure mode is generalizing a mechanism before testing reproducibility, and the cheap check is always "run it again unchanged".)* Corrected guidance now in CLAUDE.md §Git-hygiene: retry 2–3×, and clear the stale `.git/index.lock` between attempts or the retry reports a phantom "another git process" error. |
| 07-25 | *"the YouTube-IDM non-CC scale-up will return a decision-grade lift verdict"* (planned + briefed; **NOT delivered**) | **NEW CLASS — operational churn against a rate-limited LIVE THIRD-PARTY source** | The harvest was **hard-blocked by YouTube** mid-run ("Sign in to confirm you're not a bot"; confirmed a block not throttling — a single isolated request also fails). **Root cause is ours, not theirs:** the pipeline was *iterated against the live source* — single → parallel ×3 restarts → GeoCalib rework → 3 smokes + a 65-clip run — and the cumulative burst volume tripped the anti-bot. The 80-clip pilot's clean run held only because it ran **once, at low volume**, which we mistook for headroom. ⚠️ **BINDING LESSON: validate a pipeline on a 2-clip smoke, then make ONE gentle wide run. Never iterate architecture against a rate-limited live source, and never in parallel bursts.** ✅ **Correctly NOT worked around:** no cookies/sign-in, no player-client evasion — bot-detection bypass is out of bounds regardless of how much it would unblock, and the block is a rate-limit signal to respect (be gentler), not an obstacle to route around. **Do NOT auto-retry**; resume deliberately after cooldown with the staged gentle config (`W=2 TARGET=400 SEEDS=4 --sleep 4`). What SURVIVES un-retracted: the non-CC harvest works (real non-CC clips, license recorded per pointer), privacy holds (full-res face/plate/body blur, raw mp4 deleted, pointers-not-bytes), GeoCalib per-video geometry is integrated deadlock-free, and two real traps were fixed (GeoCalib's `opencv-python` dep silently clobbered pinned cv2 4.11 → killed `CascadeClassifier` → **broke the privacy blur**; 8-worker thread oversubscription, loadavg 98 at 81% idle CPU). **The pilot's 80-clip DIRECTIONAL win (+0.563, ~92% of ceiling) remains the ONLY YouTube result — the scale claim is NOT upgraded.** |
| 07-25 | *"flagship-v1.6 = ⭐ best ADE in the program"* — **stood in the §1.4b HEADER for 4 days AFTER its own body retracted it** (C1, 07-21), plus a second instance surviving in the §1.4b narrative | **C4 — PROPAGATION: a retraction that does not edit the HEADLINE has not landed** | The 07-21 retraction corrected the prose and **left the section header standing**, so the registry — the ONLY quotable source — headlined a claim its own body refuted 14 lines later. Decision-grade re-derivation (paired episode-cluster bootstrap, B=2000, 40 eps / 881 windows, corr 0.453, reproduces digit-for-digit): **Δ(v1.6 − v1) = +0.0104 m [−0.0888, +0.1147] — NOT separated, and v1.6 is BEHIND on the point estimate (0.43746 vs 0.42711).** Not a power artifact: the paired half-width (±0.1018) is *tighter* than the invalid quadrature (±0.1199) and still spans 0. ⚠️ **TWO BINDING LESSONS: (1) a retraction sweep MUST re-read section HEADERS, not just body prose — the headline is the highest-visibility surface and the last place anyone looks; (2) the sweep MUST be MULTILINE — a second instance wrapped across a newline (`the best` / `ADE in the program`) and evaded line-based grep, i.e. PRESENCE-detection hitting the same trap as the C8 absence-detection artifact.** Also settled a live self-contradiction (§1.4b stated its G-A gates as `❌ ❌` on held-out and `✅ ✅` on full-set): **both are TIES → UNRESOLVED, not pass/fail** (v1.6 vs REF-C-XL Δ −0.0340 [−0.1060, +0.0511], also not separated). Third registry-carries-a-retracted-claim finding in one day (cf. the n=1 REF-C closed-loop reading, corrected the same session). Header + narrative + gates now corrected. Note: `LEADERBOARD.md` was already fully correct — **the derived document out-disciplined the source of truth**, which is the wrong way round and is why header sweeps now matter. |
| 07-25 | *"launch the v2-corpus flagship with `--no-labels-v2`, so it differs from the running arm ONLY in corpus"* — **my own launch decision, wrong on BOTH counts** | **C8 — inference from a verified fact to an unverified comparison** | The *fact* I checked was true (`v2_labels: false` **is** in `flagship-v4-fromscratch-30k`'s config). The **inference was not**: that run is **`train_flagship_v4.py` with `labels="v3"` — a different trainer AND architecture**, so it was never the comparable control and "differs only in corpus" was never available from it. The real same-trainer control is **`flagship4b-v2-30k`** (REGISTRY §1.3: identical `--config flagship4b`, same **286,339,251** params, parity corpus, abandoned step 7,800) — and **its `--v2` IMPLIES `--labels-v2`**. So my choice produced **2 axes (corpus + labels)** where plain `--v2` gives **1 (corpus only)**; it also lost the more accurate curvature-gated labeler, so it was worse for the production goal too. **Caught by the launching subagent, which recorded the confound instead of hiding it; I verified against the registry and restarted ~20 min into a 90-h run** (the cheapest possible moment). ⚠️ **BINDING LESSON: verifying a fact is not verifying the COMPARISON it is supposed to license. Before pre-registering "differs only in X", confirm the control is the same trainer + architecture + params — from the REGISTRY, not from whichever run happens to be in front of you.** *(Fourth premature-certainty error this session; the recurring shape is a true premise carrying an unchecked inference.)* |
| 07-24 | *"TanitEval's fuller version (bench/closedloop/runner/report/registry) is STRANDED on worktree `dazzling-villani-bb4728`, not merged to main"* AND *"lake ingest modules (ingest/hf_export/license_guard/schema/filtering) are STRANDED on worktrees, not in main"* (chat to Sayed + LOOP_STATE) | **C8 — surface-read / tool artifact** | BOTH FALSE. The **Glob tool sorts results by modification time and truncates at 100** ("Showing 100 of 250/112"). Main-tree files with older mtime got cut past result 100, so I saw only the `.claude/worktrees/*` copies and inferred main lacked them. Ground truth: a NARROW glob (`stack/tanitad/lake/*.py`, `taniteval/taniteval/*.py`) + the subagent's git check (merge-base==worktree HEAD, main 4 commits AHEAD, diff 100% CRLF noise) show **main has ALL modules and is the newest/most-complete copy**; the worktrees are fully superseded. ⚠️ **BINDING LESSON: never infer "absent from main / stranded" from a Glob result — it is mtime-sorted + capped at 100. Verify presence/absence with `git ls-files <path>` or a NARROW non-truncating glob, the tools that own the fact.** Same root-class as the "absence from ONE probe is not absence" rule (CLAUDE.md §Operating-standard-2), here applied to PRESENCE-of-stranding. 2nd premature-certainty error this session (the 1st: the C5 above). No cost incurred (caught same-iteration by the subagent I briefed on the false premise — which is why the brief said "verify, don't assume") but the false claim reached Sayed. |
| 07-25 | *"the closed-loop-improvement direction is BOUND — on-policy training on our instrument does NOT close road-keeping (base rarely fails → objective starved)"* (the `LOWOOD-CL-TRAIN` verdict, chat + LOOP_STATE + 07-24 program report; itself the UPDATE that walked back a same-day C3) | **C6 — the BOUND verdict was HORIZON-CONFOUNDED; the instrument was measured at 2 s on an 18 s event** | The pre-registered horizon experiment (E1a, paired **common-start**, identical 43 held-out windows at K=20 vs K=185, episode-cluster bootstrap B=2000) fires **Outcome A decisively**: corridor-departure **0.0035 → 0.5877 overall** (junction **0.025 → 0.8414**), peak-XTE **0.35 m → 38.94 m**, paired Δ **+0.5842 [0.5071, 0.6565], p=1.0, SEPARATED** — while the OOD-envelope ratio stays **≤1.30** (genuine in-distribution failure, not extrapolation). The `LOWOOD-CL` instrument reported "base rarely fails → objective starved" **because it scored departures in a 2 s window**; at the 18.5 s horizon that matches a real junction crossing, base fails on **59–84 %** of windows — the objective was **never starved, it was blindfolded**. E2a localizes the fix: the lateral offset **is** representable (oracle R² **0.72**, ceiling ρ **0.91**) and the loss is **91 % downstream** (planner ignores available info), **neither truncation (0.01 %) nor conditioning (0.11 %)** → the lever is the **training objective**, i.e. failure-gated CL-SFT (E1b), renderer-free. ⚠️ **BINDING LESSON: a closed-loop "bound/closed" verdict inherits the horizon of the metric that produced it — before declaring a direction closed, confirm the failure-detection horizon MATCHES the event's timescale (a 2 s ADE-window cannot see an 18 s corridor drift).** Pairs with the C3/C5 "run the cheapest reopening check before claiming closure" meta-pattern — here the reopening check was a **paired horizon sweep on data we already had**, cost ~0 GPU-h. **5th over-claim-of-closure reopened by a cheap follow-up this session** — the pattern is now so reliable it is itself the strongest evidence that "closed" should never be a same-session verdict. What SURVIVES: the instrument is still map/agent-free (no collision/off-road); E1b is a pre-registered experiment with a falsifier, not a promised win; the 2-metric power/tolerance lessons stand. |

| 07-25 | *"the control beats the treatment ⇒ test the hierarchy or DROP the claim"* — finding F1 + proposal #6 of my own **independent chief-scientist review** (`Reviews/2026-07-25-…/00_CHIEF_SCIENTIST_REVIEW.md`), delivered to Sayed in chat | **C6 — REPEAT of the 07-21 hierarchy confound, from a different door** *(and the review that flagged "premature closure" as the program's dominant failure mode committed it in its own headline)* | **Caught by Sayed** ("it's not about changing the thesis — the question is how to prove it consequently"). The adverse reading rests on **six independently MEASURED confounds, every one of which makes the hierarchy UNDETECTABLE rather than absent**: (1) REF-C evaluates with **`nav_cmd=None`** — the route input is never exercised in the comparison *(this is the IDENTICAL confound already logged on 07-21, where it "nearly designed the hierarchy away")*; (2) the deployed 0.452 m number **structurally bypasses all three brains** — the hierarchy is not in the loop being scored; (3) nav→strategic has **`route_skill = 0.0`** (pure command-echo; `nonav_route_beats_majority` FAILS straight 240/240) — a **broken implementation, not a refuted concept**; (4) the **D5/D6 topology gates have NEVER RUN**; (5) the corpus is **74 % straight / 0 % semantic** with a 2-parameter CTRV oracle topping the table — there are almost no route decisions to make; (6) the metric is **ADE@2s**, and strategic value is a **10–20 s** quantity — the same instrument my own review proved blind to an 18 s failure (E1a) three sections earlier. ⚠️ **BINDING LESSON: absence of evidence is not evidence of absence when the instrument is provably incapable of detecting the effect. Before recommending that a hypothesis be DROPPED, enumerate what would have to be true for the effect to be VISIBLE to the test that failed to find it — and check each condition.** A null from an instrument that cannot see the effect is not a null; it is a missing experiment. *(Second-order lesson: an audit is not exempt from the error classes it audits. This review's own §8 named C6-confounded-comparison as 19 % of all retractions while its headline finding was one.)* **Consequence:** proposal #6 WITHDRAWN and replaced by the **Hierarchy Proof Program** (`Reviews/2026-07-25-…/01_EXECUTION_PLAN.md` Part A) — four pre-conditions (working route input · hierarchy actually in the loop · horizon-capable instrument · decision-rich corpus) that must hold **before** any hierarchy-vs-flat number is admissible, then a **6-prediction discriminating battery** (horizon-growth · junction-concentration · route-counterfactual · compositional generalization · data-efficiency · recovery) pre-registered with falsifiers. The thesis is not on trial; the measurement apparatus is. **Cost: zero** (caught same-session, before any GPU was spent on a "drop the hierarchy" path) — but the recommendation reached Sayed in chat, which is exactly the propagation surface this log exists to police. |

| 07-25 | *"Checkpoint landed and the GPU is free"* — my own status to Sayed in chat, and to the eval agent via SendMessage (*"your transfer completed"*), authorizing it to proceed to the eval | **C1/C2 — PRESENCE mistaken for COMPLETENESS** *(the "launched ≠ completed" class, in its file-transfer form)* | I ran `find … -name "*.pt"`, saw `ckpt_step15000.pt` in the destination, and reported the transfer **complete**. It was **not**: the eval pod's `/root` had **3.0 GB free against a 3.24 GB checkpoint** (overlay fs **99 % full**), so scp wrote a **truncated file — 48.9 MB short, md5 mismatch — and did NOT fail loudly.** The file existed at the right path with the right name and a plausible size. **Caught by the eval agent**, which md5-verified before loading (as its brief required) instead of trusting the path; it deleted the truncated file, re-probed capacity with a real **4 GB `dd` write** (505 MB/s) rather than `df` alone, re-targeted to `/workspace`, and restarted the relay. ⚠️ **BINDING LESSON: a file's PRESENCE at the destination is not evidence a transfer completed. Verify the terminal marker — md5 (or at minimum an exact byte-size match against the source) — before any consumer reads it.** A partial transfer is indistinguishable from a complete one by `ls`/`find`, and **a full destination disk truncates silently**. ⚠️ Second lesson, a genuine nuance against a standing rule: `CLAUDE.md` says *"never judge pod disk with `df`"* — that rule is about the **MooseFS `/workspace` quota**, which `df` hides. Here the constrained volume was the pod's **overlay `/root`**, where `df` reported the truth (99 % full). **The right generalization is "confirm capacity with a real write at the ACTUAL destination path", not "df is always wrong".** *(Cost: **zero** — caught before any eval ran. Had it not been, the eval would have loaded a corrupt checkpoint and produced either a crash or, far worse, a plausible-looking wrong ADE that entered the record as the flagship's mid-training number.)* Third time this session a completion was asserted from an indirect signal rather than the owning check. |

| 07-25 | **E2 / H26: *"ctx→tactical is a LOAD-BEARING hierarchy seam, maneuver-acc Δ +0.0439 CI-separated"*** — the single measured leg supporting the hierarchy in the 30k panel, quoted in `PROGRAM_OVERVIEW`, the R3 audit, HPP-0, and by me in chat today | **C4/C5 — THE ESTIMATOR PRODUCED THE EFFECT** *(a new sub-class: not a stale headline and not an n=1 read — a **biased statistic manufacturing a positive**)* | `hierarchy.py` used the deprecated `_jack` (`overlapping_holdout_se`). Migrating it to the paired episode-cluster bootstrap shows `_jack` does not merely narrow intervals — **it moves the POINT ESTIMATE**: the published **+0.0439 is a `_jack` artifact; the true full-set paired delta is +0.0148** (bias **×2.97**; ×3.28 on v4.2b, ×1.76 on the v1 artifact; mechanism reproduced synthetically up to **×4.29 including a SIGN FLIP**). Under the correct estimator the seam **fails all three gates on the POINT ESTIMATE ALONE** — 0.0148 < MIN_ACC 0.02 · 0.0050 < MIN_COS 0.01 · 0.0437 < MIN_ADE_M 0.05 — so no widening of intervals is even required to kill it. ⇒ **the honest read is now 0 of 3 hierarchy seams load-bearing** (was 1/3): `intent→operative` harmful, `nav→strategic` separated *by construction* (the echo), `ctx→tactical` retracted. ⚠️ **BINDING LESSON: a deprecated estimator is not only an interval problem. Before trusting ANY historical positive, check whether the statistic that produced it also biases the mean — `_jack`/`overlapping_holdout_se` does, by up to ×4.29 with sign flips.** Every pre-2026-07-25 number computed through `_jack`/`_agg` is suspect in BOTH its interval and its central value; the closed-loop panel showed the same defect at 1.5–5.9 % the same day. ⚠️ Second lesson, preserved in code: `_jack`'s `separated` was **one-sided**; a naive port to a two-sided test would have flipped a **harmful** seam to LOAD-BEARING, so the panel now emits both `separated` and `separated_positive` and load-bearing reads the latter. **CONTEXT THAT IS NOT A CONSOLATION BUT IS LOAD-BEARING FOR INTERPRETATION: this does NOT bear on whether the hierarchy works.** All three seams were measured under **PC1-violated conditions** — the route target is a lookup of the route input (`route_skill = 0.0` by construction, 27 % coverage), and the scored path bypasses the hierarchy entirely (PC2). *0/3 seams under a broken instrument is what an **untested** hypothesis looks like, not a refuted one.* **And the other hierarchy-supporting result got STRONGER and is now admissible: H18 grounding dominance corrects UP to Δ +2.9568 m (from +2.6979) and would need an 8.65× interval widening to un-separate — against a worst-ever-measured 2.06×.** |
| 07-25 | *"ADE is **98.6 % longitudinal** by squared-error energy ⇒ the lateral channel gets ~1.4 % of the signal and is **nearly invisible**"* — my own framing to Sayed in chat and in `LATERAL_VS_LONGITUDINAL_ANALYSIS.md` §1.1, written in answer to his lateral-deviation question | **C5 — single-artifact scalar quoted as a program constant** *(the "bucket means, never single rows" rule, applied to myself within hours of citing it)* | The 98.6 % is **`ep_00020`-specific and does not replicate.** Measured across **8 committed arms**: energy share ranges **0.607–0.976**, and the deployed **flagship v1 is 0.873** ⇒ lateral ≈ **13 %**, not 1.4 % — **an order of magnitude off** for the model that matters. ⚠️ **The structural claim SURVIVES** (longitudinal dominates in 8/8 arms, 61–98 %, so lateral is systematically under-weighted and the M1–M6 program stands) — but *"nearly invisible"* was an n=1 overstatement and is withdrawn; the quotable form is *~2–39 % depending on arm, ~13 % on the deployed model.* ✅ **What DOES replicate is the decisive claim: the compounding law holds 8/8** — lateral error grows several-fold faster than longitudinal, which is why the 18.5 s failure is a *lateral* corridor departure (E1a peak XTE 38.94 m) while the 2 s metric reads longitudinal. **BINDING LESSON: I flagged the n=1 caveat in the doc and then quoted the headline number without it in chat — a caveat that does not travel with the number has not been applied.** Cost: zero (corrected same-day, same-session, before any decision); the overstatement did reach Sayed in chat. |

| 07-25 | *"flagship-v4-fromscratch val ade@2s ~0.48 and descending → **v1's 0.427** with 2/3 of training left"* — the trainer's in-loop val quoted against v1's eval number, carried in `LOOP_STATE` for days and repeated by me to Sayed in chat (*"WM loss 2.10 and still descending… holding through full coupling"* framed as being on the descent to 0.427) | **C1 — a faster-moving source than the harness, in its subtlest form: TWO DIFFERENT STATISTICS, not two different runs** | The first decision-grade eval of this arm (step 15,000, `episode_cluster_bootstrap` B=2000, harness pinned by recomputing v1 from its own dump to **0.4271 exactly**) measures **ADE@2s = 0.5839 m [0.4962, 0.6821]** — and paired against v1, **Δ +0.1568 m [+0.0630, +0.2504], CI-SEPARATED BEHIND v1.** The trainer's ~0.48 was never comparable: **on the SAME forward pass the trainer's dense-20 statistic reads 0.4596 while the v1-comparable 4-waypoint value reads 0.5839.** So the number that looked like "0.48, closing on 0.427" was a *different metric definition* — the arm is ~0.16 m behind v1 at half schedule, not ~0.05 m. ⚠️ **BINDING LESSON — this is the C1 rule's hardest case and it defeated the existing wording.** The standing rule says *"trainer logs watch curves; only `eval_*.py` output is quotable."* Both numbers here are "val ADE@2s" by name, from the same checkpoint, on the same clean split — they differ **only in the waypoint set they average over (dense-20 vs the 4 gate waypoints)**. **A metric NAME is not a metric DEFINITION: before comparing two numbers, confirm the reduction (which waypoints, which horizon, which aggregation), not just the label and the split.** *(Cost: no GPU-day — the run continues and is healthy on its own terms — but a materially optimistic read of the flagship's position reached Sayed repeatedly, and it was the number underpinning "on track for v1".)* **What is NOT retracted and stands on the same eval:** the arm beats both trivial floors CI-separated (CV 0.8377 by +0.2538; hold-v0 0.7876), the WM canary at **2.0739** vs the planner's 0.5839 shows the trunk still converging with the head carrying the arm — exactly the from-scratch co-evolution signature — and this is a **descent position at 15k of 30k, not a verdict**. ⭐ **New finding from the same eval, directly relevant to the lateral axis:** v4@15k's longitudinal share of squared error is **0.619 vs v1's 0.873** — i.e. it is **38.1 % LATERAL**, and its lane-scale cross-track tail is **8× v1's** (6.4 % vs 0.8 % of windows with peak \|XTE\| > 1.75 m; p90 1.4277 vs 0.7119 m). An undecomposed L2 hides precisely this, which is why the lat/lon split is now mandatory on every reported error. |

| 07-26 | **H2 `L1_gate` decision-relevance lift = "2.22× [1.30, 3.14] at 3 m"** — the number the whole H2 label rested on (substrate audit, carried into `H2_PHASE1_PLAN.md` and reported to Sayed) | **C5 — WINNER'S CURSE / ARGMAX-OF-A-SWEEP on a small sample** *(the "bucket means, never single rows" rule, in its most seductive form: a real-looking effect with a plausible mechanism and a CI that excluded 1)* | Held-out at 3.0 m the lift is **1.16× [0.9975, 1.3272]** (paired episode-cluster bootstrap, B=2000, **2,159 episode-clusters**, zero clip overlap with the sweep) — **CI includes 1.0**, far below the 1.5× bar; excess lift attenuates **1.22 → 0.16 (7.6×)**. **Both pre-registered PASS criteria fail ⇒ the label is refuted as a capability target.** The root cause is MEASURED, not inferred: on the sweep's **own two chunks**, the 105 clips it did *not* draw give **0.99× [0.53, 1.53]** — same geography, same rig, same code, **zero effect**; 80-clip subsamples at a *fixed* 3.0 m span **0.42–2.14**; **P(lift ≥ 2.22) = 2.0 %**; and **2 of 24 held-out chunks individually reproduce a "2.2×, CI excludes 1" result** — i.e. the original finding is exactly what noise produces at this n. ⚠️ **BINDING LESSON: a threshold chosen at a sweep's argmax carries the sweep's selection, and its CI is then not a CI. Any cut selected after seeing a sweep must be re-confirmed on held-out data at that single cut before it is quotable — and a physically plausible mechanism is NOT evidence the cut is real** (here the lane-width story was *right in shape* — monotone decay crossing 1.0 at ≈3.5–4.0 m — while the peak's *location* was noise; the mechanism made the artifact credible). ✅ **The process worked as designed and this is the system functioning, not failing:** the substrate agent **flagged its own post-hoc selection**, the orchestrator made it a **pre-registered stop gate before any GPU**, the executing agent **refused to re-scope onto the surviving ≤1.5 m peak**, and the cost was **~2 CPU-hours instead of a pod-week and a paper**. ✅ **Un-retracted, and it strengthens:** the need-**RATE** reproduced out-of-sample **to three digits** (1.832 % vs 1.83 %) on **27× the episodes** ⇒ **0.67 % residual ⇒ 84.8–85.6 % of surround-camera compute saveable.** *The label's frequency generalises perfectly; its decision-relevance does not exist.* ⚠️ Consequence for how C-EFF may be stated: the trigger is **geometric presence**, whose safety-relevance is precisely what this retraction removes — so the admissible claim is *"an off-front agent is geometrically proximate on 0.67 % of frames"*, **never** *"the ego needed another camera on 0.67 % of frames."* |
| 07-26 | *"we discard 57 % of the front camera, so the cheapest capability win is to WIDEN THE CROP, not activate a second camera"* — **my own** §2 recommendation in `H2_PHASE1_PLAN.md`, promoted to "the first experiment" and reported to Sayed as the headline reframe | **C3 — a mechanism asserted as a cost conclusion without doing the arithmetic** | The geometry was right and the inference was backwards. The split is **63.6 % recoverable-by-crop / 36.4 % residual**, which *looks* decisive for widening — but **covering the full front field costs 2.30× the native pixels, ALWAYS ON**, against **1.007 cameras/frame** for selective activation. **Selective activation is ~2.2× CHEAPER than widening the crop.** I compared *coverage* (what fraction of triggers a wider crop would catch) and silently treated it as *cost*, never pricing the always-on token increase against an occasional second encoder pass. ⚠️ **BINDING LESSON: "X covers most of Y" is a COVERAGE statement; converting it to "X is the cheaper fix" requires the cost model, and for always-on-vs-occasional the duty cycle usually dominates the per-event cost.** *(Cost: zero — E0 was cheap and it is what produced the refutation, so the experiment was worth running even though its motivating claim was wrong. But the recommendation reached Sayed as "the first experiment we should run, and it reframes the workstream."* ✅ Both E0 outcomes had been pre-committed as informative, which is why the wrong prediction cost nothing.) |

| 07-26 | **E1b's shipped evaluator (`e1b_eval.py`) would have reported SUCCESS on a run that is pre-registered BOUND** — its verdict logic never consulted the guardrails at all; guardrails (b) anchor-metrics and (c) OOD, peak XTE, and the lateral/longitudinal split were **not implemented**; and open-loop (a) was computed **unpaired** | **NEW CLASS — C10: THE EVALUATOR DOES NOT IMPLEMENT ITS OWN PRE-REGISTRATION** *(the pre-registration was correct, complete, and committed in advance — and the code that renders the verdict silently ignored most of it)* | Caught by the executing agent, which **implemented the missing guardrails and fixed the unpaired test BEFORE running**, then declared every change in its report §8 rather than quietly shipping a passing number. Had it not: a **large, real, CI-separated closed-loop win** (junction departure −0.4270, peak XTE −35.90 m) would have been announced as **SUCCESS** while open-loop had degraded **CI-separated worse (+0.1947 [+0.1415, +0.2522])** — i.e. the program would have promoted a Pareto trade as a clean win, on a verdict its own pre-registration forbids. ⚠️ **BINDING LESSON: a pre-registration is not in force until the EVALUATOR implements it. Before the deciding run, execute the evaluator against a synthetic case that MUST fail each registered guardrail, and confirm it renders the failing verdict.** A guardrail that is written in a document and absent from the code is not a guardrail — it is a comment. *(Sibling to the 07-25 finding that `run_gate.py`'s `matched_step_ratio` SystemExit escaped from a diagnostic and aborted every verdict: in both cases the ADJUDICATING TOOL, not the science, was the defect. The instrument layer deserves the same adversarial scrutiny as the claims.)* |
| 07-26 | **E1b's replay term was treated as a catastrophic-forgetting guard on the strength of its TRAINING loss** (replay loss fell 1.83 → 1.61, read as "open-loop is being preserved") | **NEW CLASS — C11: A TRAINING-SET LOSS IS NOT A GENERALIZATION GUARD** | Held-out open-loop ADE@2s **rose 41 %** over the same interval (0.4747 → 0.6693, paired Δ +0.1947 **separated worse**), and anchor accuracy fell 0.6815 → 0.6163 while anchor-traj L1 rose 0.1775 → 0.2399 — **all three guardrails failed while the replay loss it was supposed to protect them was improving.** The replay was measured on **parity-TRAIN**, the thing it was regularising toward, so it could only ever report success. ⚠️ **BINDING LESSON: a regulariser must be monitored on HELD-OUT data, never on the corpus it replays. If the guard and the objective share a distribution, the guard is a second copy of the objective.** ⇒ **Concrete, cheap fix for the successor run:** gate the replay on **held-out** open-loop ADE + anchor metrics (early-stop or λ-schedule on the held-out signal), not on the replay loss. This is a *fixable mechanism*, not a wall — which is why the BOUND verdict here does **not** close the direction (see the standing consequence below). |

| 07-26 | **E1's null was attributed to the TRIGGER (`L1_gate`'s geometry), and the workstream was re-scoped on that reading** — when the **RESPONSE** half of the label was the unpowered one | **NEW CLASS — C12: A COMPOSITE LABEL'S NULL BLAMED ON THE WRONG HALF** | `L1_gate` = *trigger* (an agent off-front and geometrically proximate) **AND** *response* (the ego did something about it). E1 refuted the composite at 1.16× and I reported the diagnosis as *"the label captured geometric presence, not decision-relevance"* — i.e. I blamed the trigger. The **2×2 decomposition on the SAME frames** (L2 work, MEASURED) shows where the signal actually was: old-trigger × old-response **1.10×** (reproduces the refutation) → **new-trigger × OLD-response 1.85× [1.38, 2.28]** → both-new **2.41×**. **The new trigger separates even under the unamended response** — so the trigger *was* the weaker half and my reading happened to be right, **but I had not established it**, and E1's own four-flat-thresholds table already contained the evidence that the response was insensitive. It was noted in that report and not acted on. ⚠️ **BINDING LESSON: when a composite AND-label returns a null, you have learned nothing about WHICH conjunct failed. Decompose the 2×2 (old/new × old/new) on the same frames before re-scoping a workstream — the ablation is nearly free and it is the only thing that licenses a diagnosis.** *(Cost: none — the re-scope I proposed was directionally correct and the L2 redesign fixed both halves anyway. But the reasoning was unsupported at the time it was reported to Sayed.)* |

| 07-26 | **`SOURCE_REGISTRY` recorded nuScenes as `CC-BY-NC-4.0, share_alike=False`** — i.e. permissive-within-NC, freely mixable into the research tier | **C4 — LICENCE INFERRED FROM A SHORT NAME. SECOND INSTANCE OF THIS EXACT CLASS** *(after the 2026-07-13 ZOD correction)* | nuScenes is **`CC-BY-NC-SA-4.0` — COPYLEFT**. Confirmed by two independent PUBLISHED probes (the authors' own paper, arXiv:1903.11027, verbatim; plus a corroborating secondary corpus). The authoritative terms page returned **empty on 4 fetches across 3 URLs** and is documented to carry *modifications* to the CC grant, so the entry is recorded as the **floor** of the restriction and the conservative branch was taken — evidence and caution agreed. A second trap was separated in passing: **the nuScenes devkit code is Apache-2.0; the DATA is not** — conflating them is how the short-name error propagates. Now `share_alike=True`, routing to `shards/nc-research/sharealike/nuscenes/`, with **8 regression tests** (registry-grounded and end-to-end, each of the three C-tier gates refusing independently). No guard weakened; the correction propagates into the push safety-check's SA set with zero edits there. ⚠️ **BINDING LESSON: a licence is a document, not a string. Never populate `SOURCE_REGISTRY` from a remembered short name — fetch the terms, and when the authoritative page cannot be retrieved, record the CONSERVATIVE branch and mark it as a floor.** Twice now this class has put a copyleft corpus one guard away from the commercial tier. ⭐ **And the consequence is strategic, not clerical: ShareAlike raises the question of whether a world model TRAINED on SA data is itself a derivative — a decision about our WEIGHTS that must be made BEFORE ingest, not after.** Escalated to the PI; not agent-decidable. Also flagged: `nuplan` is absent from the registry entirely, and BDD100K's licence is **UNVERIFIED** — both are the same class waiting to happen. |

| 07-26 | *"the IDM does not recover yaw rate — R² **0.010** at scale (n=9,420); only steer and speed are real"* — reported to Sayed as the detailed per-channel answer that justified his dissatisfaction | **C5 — A POOLED METRIC DESTROYED BY 9 CORRUPT LABEL ROWS, and pooling across two corpora hid it** | **Deleting 9 windows out of 4,195 — those whose GT yaw is PHYSICALLY IMPOSSIBLE — moves pooled `yaw_rate` R² from 0.105 to 0.497.** All nine are **comma2k19** frames at **v ≈ 0**, where heading is `arctan2` of an **undefined ENU velocity**. On **PhysicalAI the SAME head reads yaw R² 0.9035**, and a linear probe there reads 0.746 — so the model was recovering yaw all along and the metric was reporting a **label defect in the other corpus**. comma's own smooth-fit **ceiling is 0.352**: no model can score well against a label that noisy. ⚠️ **BINDING LESSON: R² has an unbounded left tail, so a handful of impossible GT rows can dominate it entirely — and POOLING ACROSS CORPORA WITH DIFFERENT LABEL QUALITY converts one corpus's defect into a verdict on the model. Report per-corpus, and check the LABEL's own achievable ceiling before concluding a channel is unlearnable.** *(Cost: zero GPU — but it reached Sayed as "the model does not recover yaw", which is the opposite of what the evidence supports, and it was the number that most justified his dissatisfaction.)* Same session, the reverse error was also found: the v1 card's *"steer … DROPPED as unusable"* is **inherited prose with no measuring artifact** (C4) — steer actually reads **R² 0.742**. One channel was written off that works; another was written off on 9 bad rows. **⤷ SUPERSEDED 2026-07-27 by C29 — deletion was the wrong fix.** The 9 impossible rows were the visible tip: **50 windows (1.19 %)** carry an undefined-at-standstill heading, and *repairing* the label beats *deleting* it (pooled **0.8108 vs 0.4967**, and it discards nothing — 4,195 windows kept vs 4,186). The **`R² 0.010` at n = 9,420** retracted in this row is **STALE-PENDING, not corrected**: it lives on the v1 card's own 9,420-window pooled split, and **no repaired measurement exists there** — the 0.8108 figure is a different substrate (4,195 windows, `v_min` 0.5) and must not be pasted in. Per corpus on that substrate: comma2k19 **+0.0114 → +0.3308**, PhysicalAI **+0.9035 unchanged**. ⭐ And the honesty condition C29 carries: comma-only MAE **−42.5 %** but **medAE −1.1 % and nMedAE 8.0 % WORSE** — the repair fixes the tail and the summary statistic, **not typical accuracy**. Inventory: `…/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`. **⤷ FORWARD POINTER 2026-07-27 (`heading-default` pass; this row is NOT rewritten and `0.010` keeps its date):** the re-score was run — dev box, nothing retrained — on the **comma half** of this very 9,420-window split (`cm_[40:70]`, 30 clips / 4,140 windows, `episode_id`-disjoint from the head's own comma training clips). ⛔ The **pooled** figure is still **NOT re-issued** and stays STALE-PENDING. Comma-only: yaw **R² +0.000048 (repair OFF) → −0.000421 (ON)** — ⭐ **the repair does NOT move R² on this substrate**, so pasting the `+0.3308` anchor here would have published **+0.33 where the measurement is ≈ 0**. MAE **0.2288 → 0.1527 (−33.3 %**, paired episode-cluster bootstrap [−0.150, −0.012], separated), medAE −2.6 %, **nMedAE 10.1 % WORSE**, ρ flat — the honesty condition reproduces **in shape but not in magnitude**. MEASURED cause: **one wholly-stationary clip** (300 frames, zero observable frames, v_max 0.039 m/s) that the repair deliberately leaves alone, carrying **84 impossible labels up to 15.28 rad/s**; raising `v_min` cannot remove it. ⚠️ **BINDING LESSON, added to C5's own:** a label repair that is *correct* can still leave a metric *unmoved*, because a repair and an **admissibility** decision are different things — `hold_heading_through_standstill` returns an `observable` mask for exactly this and **no caller in the repo uses it**. Record: `…/incoming/2026-07-27-heading-default/HEADING_DEFAULT.md` §5. **⤷ SECOND FORWARD POINTER 2026-07-27 (`anchor-settlement`; row still NOT rewritten):** the `+0.3308` quoted twice in this row as "what IS measured elsewhere" is now **WITHDRAWN** — **2 of its 22 comma val episodes are, BY CONTENT (sha256 of raw pose bytes AND raw sensor bytes), inside the scoring head's own comma TRAINING set**; content-clean value **−0.746** (CI [−1.574, −0.177]). The `R² 0.010` retracted in this row stays **STALE-PENDING** and is **not** replaced. What survives: a RETRAINED head reads **+0.3038 [+0.054, +0.479]** on content-verified-disjoint comma clips — comma yaw is TESTABLE and the deployed head does not do it. See **C43**. |

| 07-26 | *"the true wheelbase is **2.85 m for ~90 %** of clips and 3.165 m for ~10 %"* — from the PhysicalAI feature probe, carried into the PI decision brief and into my own measurement brief as the premise | **C2 — A DISTRIBUTION READ FROM A SINGLE CHUNK** *(statistic from one probe; the same class as the 07-25 "df hid the quota" and "one GPU sample" errors, applied to a data distribution)* | The figure came from **`vehicle_dimensions.chunk_0000` alone — a 100 %-US chunk.** Across the **197 chunks our parity corpus actually draws from** there are **FIVE** wheelbases: **2.730 m (47.0 %, the MODE — where 2.9 is +6.23 % wrong)**, 2.850 (**1.8 %**, not 90 %), 3.135 (13.9 %), 3.165 (25.5 %), 3.216 (11.8 %). **98.2 % of clips carry a >5 % error**, and the populations are **geographically coherent** — `I(wheelbase; country) = 0.769` of `H = 1.880` bits ⇒ **40.9 % explained by country**, with the 2.85 m slice **100 % United States**. ⚠️ **BINDING LESSON: a DISTRIBUTION cannot be read from one chunk, one shard, or one file — sample the population the corpus actually draws from, and report the support, not just the mode.** Correcting the premise **made the defect bigger**, which is the direction that matters: a wrong premise flattered the problem. *(Cost: zero — the measurement that corrected it was the one commissioned to act on it, and it verified the join rather than trusting it: 2376/2376 committed `episode_id`s reproduce, the 24 skips land exactly on 1798…1941, 40/40 val ids match, and a query-grid trap was caught — `labels/egomotion` spans ~140 s, not 20 s — with the corrected reconstruction matching the built episodes at corr 0.99998.)* |
| 07-26 | *"`steer` is r = 0.9865 redundant with ω/v, so the wrong wheelbase constant may be nearly irrelevant to LEARNING"* — **my own** reasoning, given to Sayed in the decision brief as the nuance that "changes the calculus" | **C6 — A REDUNDANCY MEASURED ON THE DATASET, ASSERTED ABOUT THE MODEL** | The redundancy is real **in the data** — `R²(steer | ω/v)` out-of-sample, episode-disjoint, v > 2 m/s = **0.9919 [0.9841, 0.9964]**. **But ω IS NOT AN INPUT CHANNEL TO flagship-v1.** Measured against what the model *actually receives*: **`R²(steer | accel, v0/10) = 0.0002 [−0.0021, +0.0006]`** — the model cannot reconstruct steer from its own inputs at all, and **zeroing steer costs +0.343 m**. A third, independent measurement agrees: `IDM_V2_RESULTS.md` §3.4 found *derived* steer (0.424) **worse** than regressed (0.520), verdict *"do not drop steer"*. ⚠️ **BINDING LESSON: "channel X is redundant with Y" is a statement about the DATA. Before inferring that a defect in X is harmless to a MODEL, check that Y is in that model's INPUT SET.** I inferred harmlessness from a correlation the network never sees. *(Cost: zero — it was the stated reason to lean toward "do nothing", and the measurement I commissioned refuted it before any decision was taken. Also reconciled: the 0.9865 is a **mean of within-episode** correlations, reproduced at 0.9786; pooled is 0.9931 at v > 2 m/s and **0.0125 at all speeds** — the low-speed regime destroys it.)* |

| 07-26 | *"the OOD-envelope ratio stays ≤ 1.30, so this is **genuine in-distribution failure, not extrapolation**"* — the certificate attached to E1a's horizon result, repeated in `GATE_PROTOCOL.md` §0.1 **(written by me)**, `RETRACTION_LOG.md`, `LOOP_STATE.md`, `E1a_E2a_RESULTS.md:173`, and **live as a guard constant in two eval scripts** (`e1b_eval.py:403`, `e1c_common.py:34`, both `<= 1.30`) | **NEW CLASS — C13: A GUARD THAT CANNOT FAIL IS NOT A GUARD** *(a saturating estimator cited as a certificate — it could never have falsified the claim it was used to support)* | `OODMap.ratio_arr` uses `np.interp`, which **CLAMPS at |dlat| = 3.0 m and |dψ| = 12°.** The ratio therefore **saturates**: it is a **lower bound**, structurally incapable of exceeding its own 1.5 threshold no matter how far out of distribution the rollout goes. MEASURED on the v4 30 k co-primary, where the ratio read a reassuring **1.2741**: **54.63 % of steps exceed 3 m and 90.24 % of windows leave the measured envelope.** E1a's rule was always a **disjunction** — high ratio **OR** steps leaving the envelope — and **only the ratio half was ever evaluated**, in every artifact. Sweep: **194 OOD nodes across 14 committed artifacts — 14 carry a factually FALSE verdict string, 17 fail to declare saturation, 118 (the whole E1a family) quote a ratio with no verdict at long horizon, 15 carry no envelope evidence at all.** ⚠️ **BINDING LESSON: before citing a guard as a certificate, ask what value would make it FAIL — and confirm the estimator can actually reach that value. A metric that saturates below its own threshold is not evidence of safety; it is evidence of nothing.** *(Sibling to C9 — there the instrument was pointed at the wrong horizon; here it is pointed at the right quantity but cannot register enough of it. Both produce confident silence.)* ✅ **WHAT STANDS, and it is the bulk of it:** the horizon finding is **untouched** — corridor departure **0.0035 → 0.5877** on the same 43 windows, paired Δ **+0.5842 separated**, and it now **replicates independently on the v4 arm** (0.0146 → 0.6388, Δ +0.6241 separated). The gate-primary change, §0.7, §0.8 and the C9 class all stand on that finding, not on the certificate. ❌ **WHAT IS WITHDRAWN:** the claim that the failure is *in-distribution*. At K=185 the correct verdict is **EXTRAPOLATION for every arm measured** — so a bar registered there would be calibrated on extrapolated dynamics, which is exactly why the 30 k co-primary was (correctly, if for weaker reasons than I knew) registered **report-only**. **Fixed in code:** `taniteval/ood.py` makes out-of-envelope fractions first-class, implements the real disjunction, reports *which clause fired and why the other could not*, and **raises** on an inconsistent verdict. **Still live and owed:** the two `<= 1.30` guard constants in `e1b_eval.py` / `e1c_common.py` adjudicate on the void criterion. |

| 07-26 | *"the v4 30 k gate returned **NOT-CONTINUE → RESTART**"* — quoted by **me** in chat and carried into the drumbeat brief as a machine verdict | **C10 — THE EVALUATOR DOES NOT IMPLEMENT ITS PRE-REGISTRATION** *(and a **RECURRENCE**: commit `3ff5499` fixed this exact class and a sibling instance survived the fix)* | The instrument returned **`NOT_YET`**, not RESTART: *"step 29999 < pre-registered gate step 30000."* A 30,000-step run indexes **0…29999**, so `run_gate.py` compared the trainer's **0-indexed** counter against a **1-indexed** count and **refused a COMPLETE 59-hour run** (`config.json/args/steps = 30000`, `train_log.jsonl` spans 0→29999 over 661 rows, `metrics.json final_step 29999`, trainer exited, 212,544.6 s wallclock). Both projections — `raw/GATE_30K_verdict_A_no_coprimary.json` and `…_B_coprimary_registered.json` — carry `NOT_YET` and are explicitly labelled **"NOT the verdict"**. ⚠️ **BINDING LESSON: a gate that cannot tell a finished run from an unfinished one is not a gate — and an off-by-one in the ADMISSIBILITY check is more dangerous than one in a threshold, because it produces NO verdict rather than a wrong one, and a missing verdict gets filled in by hand and then quoted as if the machine had spoken.** ✅ **CREDIT + WHAT STANDS:** the gate agent caught this itself (§D3), adjudicated **against the card's own text with every criterion printed**, and said so in the report — so the process was sound and **the OUTCOME is unchanged**: `wm_canary_ade_2s` **1.1409** vs ≤0.55 and `miss_at_2m` **0.2123** vs ≤0.10 are MEASURED failures by ≥2×, a conjunction containing a hard FAIL is unsatisfiable, and with `restarts_used 0/2` the resolved call is **RESTART**. ❌ **WHAT IS WITHDRAWN:** the *provenance* — I attributed to `run_gate.py` a verdict it never rendered. **Fixed in code and VERIFIED MEASURED 2026-07-26 12:35 UTC** (`stack/scripts/run_gate.py::step_reached`): 29999/30000 → `reached=True`, convention **NAMED** as `0-indexed (trainer convention)`; **29998/30000 → still refused** under either convention; 30500/30000 → `1-indexed`. The repaired guard **can still fail**, so it is not a C13 sibling. Per the gate agent's instruction the v4 gate is **NOT** re-rendered — the fix applies to future gates. |

| 07-26 | *"λ_plan = 1.0 lets the **planner loss starve the world model** — that would explain a failing WM canary AND a within-run regression"* — **MY OWN** leading hypothesis, written into the v4 restart-lever brief as the mechanism to test, and repeated to Sayed in chat | **C3 — A MECHANISM ASSERTED WITHOUT READING THE CODE THAT IMPLEMENTS IT** *(and the mechanism does not exist: I named a culprit parameter without checking what the parameter DOES)* | **λ_plan is not a loss weight at all.** `train_flagship_v4.py:140` sums the five terms **unweighted** (`total = wm + planner + fac + sm + strat`); λ_plan is a **gradient scale on the trunk↔planner seam** (`flagship_v4.py:211`), and the module docstring states `lambda_plan == 1` is **"a strict no-op"**. So λ_plan=1.0 is the **ABSENCE of an intervention**, not an aggressive setting — the starvation I described cannot occur as stated, and 30 seconds of reading the source would have shown it. Scored anyway on the pre-registered rule (timestamped before any loss value was read): over Phase C the WM term fell **−31.7 %** (A-progressive needed ≥ 0 to stall) and the planner's share rose **+2.9 %** (A-level needed ≥ 10 %) → **A REJECTED, B ACCEPTED**; `lam_mult` was **1.0 at all 601 logged steps** and the canary controller never fired. ⚠️ **BINDING LESSON: before naming a hyper-parameter as a root cause, read its implementation — not its name.** A plausible-sounding mechanism attached to a parameter that does something else is worse than no hypothesis, because it aims a GPU-week at the wrong subsystem. *(Cost: zero, and only because the brief pre-registered BOTH outcomes and told the agent not to rescue the hypothesis — the agent rejected mine on the data and found the real failure instead. This is the pre-registration earning its keep on ME.)* |

| 07-26 | *"the learned longitudinal gate `sel_gate` is the regression: it grew **+43.5 %** while `sel_gap` grew **+43.6 %**, and `vt_speed` is hard-wired to `v0` so the 'target-speed-aware' term is a pure constant-velocity preference"* — the v4-lever agent's **own** mechanism, compelling and nearly written up | **C8 — PREMATURE ROOT CAUSE FROM A COINCIDENT TREND** *(caught by the agent itself, before publication, by running the counterfactual instead of writing the story)* | Two numbers moving together by 43.5 % / 43.6 % is not a mechanism. The counterfactual settles it: `sel_gate := 0` on a **frozen fan** gives paired Δ **−0.0100 [−0.0191, −0.0020] — the gate HELPS, separated.** The narrative was inverted. ⚠️ **BINDING LESSON: a matching percentage between a candidate cause and its effect is the weakest possible evidence — ablate it before you believe it.** ✅ Narrows the real target to `refined_logits` and its objective. *(Cost: zero. Exemplary — the agent had a publishable-looking story and spent the compute to refute it instead.)* |

| 07-26 | **Every v4 training term improved monotonically to 30 k while HELD-OUT selection got separated-WORSE** — and the run was allowed to continue to 30 k on that basis | **C11 — TRAINING-SET LOSS IS NOT A GENERALIZATION GUARD** *(program-level; clean textbook instance)* | EVAL-GRADE on the same **881 windows / 40 episodes**, 15 k→30 k: WM canary **−45.0 %**, fan quality **−21.9 %**, `oracle_in_fan` **−16.7 %** — **all better**. Only *selection* degraded: `sel_gap` **+43.6 %**, `miss_at_2m` **+25.5 %**. Paired episode-cluster bootstrap: `ade_0_2s` **+0.0584 [+0.0043, +0.1179] SEPARATED**, and **LARGER on the deployable produced surface** (**+0.0985 [+0.0374, +0.1631]**, p = 1.0) — so it is **not** a goal-oracle artefact. Decomposed: along-track **+0.0581 SEPARATED WORSE**, cross-track **−0.0257 SEPARATED BETTER** — **the regression is longitudinal**, consistent with the standing longitudinal-lever finding. **Onset:** `sel_gap` climbs monotonically from ~step **11,000**, with a persistent level shift at **26,000** coinciding with exactly one schedule event — the cosine LR reaching **4.95 % of peak**. Nothing coincides with λ_plan's saturation at 8,000. ⭐ **THE NUMBER THAT REFRAMES v4: the selector throws away 0.4093 m.** The fan's best is **0.2330 m**; the pick is **0.6423 m** (v1 = **0.4271**). **v4's PROPOSALS ALREADY BEAT v1 BY ~0.19 m — the world model is not the problem, the picker is.** ⚠️ **BINDING LESSON: a monotonically improving training loss licenses NOTHING about held-out behaviour, and a composite system can regress on the ONLY axis that ships while every component metric improves.** ⚠️ **Instrument gap found: `rank_acc`, `frac_sel_2x_worse_than_oracle`, `sel_gate` and `sel_pen_span` are computed EVERY step and DISCARDED by the row-writer (`train_flagship_v4.py:693-703`) — the exact diagnostics for this failure were computed 601 times and thrown away.** Log-only fix, must land before any restart. |
| 07-26 | ⛔ **The `<= 1.30` OOD guardrail is not merely *saturating* — its threshold sits ABOVE the criterion's ARITHMETIC CEILING.** Closes the *"still live and owed"* line of the C13 entry above, and sharpens it: that entry says the ratio *"is structurally incapable of exceeding its own **1.5** threshold"*, which left open whether the **1.30** constants the two live scripts actually use could fire. They cannot. | **C13 — A GUARD THAT CANNOT FAIL IS NOT A GUARD** *(same class, now proved ANALYTICALLY rather than observed empirically)* | `OODMap.ratio_arr = 1 + clip((interp(|dlat|) - base)/base, 0, inf) + clip((interp(|dpsi|) - base)/base, 0, inf)`, and `np.interp` is piecewise-linear through the P1 sweep points and CLAMPS outside them — so its **supremum over ALL possible inputs is a constant computable from `lowood_flagship_ci.json` alone, with no model, no rollout and no GPU**: `1 + 0.16267 + 0.136218 =` ⭐ **1.298888**. **1.298888 < 1.30.** `e1b_eval.py:403` (`c_ood_in_band: ood_ft <= 1.30 + 1e-9`) and `e1c_common.py:34` (`OOD_BAND = 1.30` → `Gc_ood_in_band`) are therefore **TAUTOLOGIES, by a margin of 0.001112** — and `RATIO_EXTRAPOLATION_X = 1.5` inside the *fixed* `taniteval/ood.py` is unreachable by **0.201112**, so **clause 1 of E1a's disjunction is dead and only clause 2 can ever fire.** MEASURED corroboration of the proof, repo-wide: **181 OOD nodes re-adjudicated, 139 carry a ratio, and the old test passes on 139/139 — it has never once failed, and it could not have.** `Gc` was evaluated at **17 E1c frontier checkpoints + 2 smoke + 1 in E1b = 20 times, 20 passes**; the largest ratio recorded anywhere is **1.2919**. ⛔ **And in E1b it is the ONLY guardrail that "held"**: `a_openloop_ade2s_ok false · b_anchor_acc_ok false · b_anchor_traj_l1_ok false · c_ood_in_band TRUE (1.1339)`. The BOUND verdict is unchanged, but the artifact publishes a passing OOD guardrail that carried **zero bits**. ⭐ The cleanest single exhibit sits in the gate's own artifact: the v4 **junction** stratum at K=185 records `ood_peak_ratio = 1.2989` — **the supremum, to 4 dp** — beside `frac_windows_any_step_out_of_envelope = 1.0`, i.e. **the estimator pinned at its ceiling while 100 % of windows are outside the envelope.** ⚠️ **BINDING LESSON, cheaper than the one C13 already states: before trusting a threshold, COMPUTE THE ESTIMATOR'S RANGE. `sup(estimator)` is often closed-form from the calibration artifact — if the threshold lies outside that range, the test is decided before any experiment runs.** Asking *"what value would make this fail?"* is necessary; asking *"can the estimator produce that value?"* is what settles it. ⇒ **Adjudication (GATE_PROTOCOL §0.7): `c_ood_in_band` / `Gc_ood_in_band` are INSTRUMENT-FAIL (VOID)** — they may not contribute to a kill conjunction *nor to a pass one*, and must be PRINTED as void. The replacement needs no new science: `ood.verdict`'s clause 2 (out-of-envelope fractions) is model-dependent, unbounded, and fires hard. *(MEASURED: `…/2026-07-26-horizon-envelope-closeout/artifacts/ood_blast_radius.json`.)* |
| 07-26 | ⛔ **`POD2_EVAL_HOST.md` headline #10 stamps `MEASURED` on a compound claim whose envelope half its OWN BODY labels `HYPOTHESIS`**: *"register K = 60 … It also satisfies the gate report's own §10.1 instruction to 'register at a horizon where the envelope holds'"* — while §4.5 says of that same clause *"that has NOT been measured, and it must be"*. | **C4 — HEADLINE OUTRUNS ITS OWN BODY** *(the "4-day stale headline" pattern `registry_lint` CHECK 2 exists for, in its forward form: not a stale retraction left standing, but an evidence class promoted on the way UP into the summary table)* | The **yield** half is `MEASURED` and stands untouched: junction clusters 232 (K=20) → 207 (K=60) → 204 (K=70) → 196 (K=75) → 58 (K=185), so K ≤ 70 remains the hard ceiling for any stratified verdict. The **envelope** half is now MEASURED and it is **FALSE**: on the arm the co-primary is registered on (`flagship-v4-fromscratch` @29999, `ckpt_md5 8771c1d9…`, closed loop, 40 eps), the fraction of windows leaving the P1 envelope is **0.1226 at K=20**, **0.5122 at K=60** and **0.5854 at K=70** — a MAJORITY at the recommended primary — and the first departure occurs at **k = 8 (0.8 s)**. **There is NO K in the gate protocol's admissible range (20 < K ≤ 190) at which the closed loop is a measurement rather than an extrapolation.** ⚠️ **BINDING LESSON: an evidence class is a property of a CLAIM, not of a paragraph. When a summary row merges a MEASURED clause with a HYPOTHESIS clause, the row's class is the WEAKER of the two — and the honest fix is to SPLIT the row, not to average the classes.** ⭐ **What this does NOT do is lower the recommended K** — the reading the pre-registration committed to in advance and then had to refuse: because the envelope already fails at 0.8 s it cannot discriminate between K=20 and K=60. **The envelope is not a horizon problem, it is a RENDERER-VALIDATION problem**, and it indicts the standing 2 s instrument as hard as the proposed 6 s one (junction stratum at K=20: **58.79 %** of windows already outside). *(MEASURED: `…/2026-07-26-horizon-envelope-closeout/artifacts/ksweep_results.json`.)* |
| 07-26 | ⛔ **`stack/scripts/eval_flagship_v4.py` cannot be IMPORTED on pod2 — the host the program had just designated as its n ≥ 200 eval box.** Introduced the same day, in commit `87131fd`, whose own headline is *"the eval pod was 62 % stale and MISSING corridor.py"*. | **C2 — ABSENCE FROM A SINGLE PROBE, in its INTERPRETER-VERSION form** *("it imports on the host I tested" is a one-probe claim)* | Line 478 puts a **multi-line expression inside an f-string replacement field** — **PEP 701, Python ≥ 3.12 ONLY**. **pod2 runs 3.11.10** → `SyntaxError: unterminated string literal` at import time. Consequence: **every v4 eval path is un-runnable there**, including `v4_corridor_cl.py` and `taniteval.clhorizon.run_v4`, both of which import `load_v4_from_ck` from this module — so **the registered closed-loop co-primary could not have been re-rendered on the designated eval host.** `compileall` under 3.11 over the whole tree finds exactly one sibling, `vlm_kin_crossval.py:117` (backslash in an f-string expression, also ≥ 3.12); `taniteval/` is clean. ⚠️ **BINDING LESSON: "it runs" is a claim about an ENVIRONMENT, not about code. When a repo executes on more than one host, the portability floor is part of the contract — pin it and CHECK it. `python3 -m compileall` under the OLDEST supported interpreter is a 30-second step that catches this whole class.** ✅ **FIXED and staged, behaviour-preserving**: both suffixes built outside the f-string, output strings asserted equal on both branches; `vlm_kin_crossval`'s header written `r"kin \\ vlm"` so its TWO output backslashes reproduce byte-for-byte. `compileall` clean under 3.11, `import eval_flagship_v4` succeeds on pod2, `pytest -q` on the four v4/vlm test modules → **70 passed**. |
| 07-26 | ⛔ **`taniteval/clhorizon.py::run_v4` — the entry point written so *"the co-primary is not stranded behind a driver in `incoming/`"* — raises on its first rollout step and has evidently never been executed.** | **C2 — A PINNED UNIT TEST MISTAKEN FOR AN EXECUTION PROBE** *(the module's `test_port_is_tensor_identical_to_the_driver` exercises `corridor_rollout` with a stub planner; nothing exercises `run_v4`, so "the port is verified" was a claim about a different function)* | `run_v4` builds episodes with `_data.load_frames`, which wraps each one in `RawEp` — exposing frames as **`.feats`** (`taniteval/data.py:220`) — and hands them to `corridor_rollout`, whose default `frames_of` reads **`ep.frames`**. **REPRODUCED on pod2: `AttributeError: 'RawEp' object has no attribute 'frames'`.** The committed gate driver used `load_episode(...)` directly and is unaffected, which is why it went unnoticed. One-line fix: `_data.load_raw`. ⚠️ **BINDING LESSON: un-stranding a capability is only DONE when the NEW entry point has been RUN end-to-end. A test that pins an internal function bit-identical to its ancestor proves the physics moved; it does not prove the front door opens.** ⛔ **NOT patched by me** — the module landed hours earlier from a sibling stream and is pinned by that test; editing a sibling's mid-flight tree is the exact hazard the pod2 standup refused. **Escalated as a one-line owner fix**; this run used `_data.load_raw`, which is also the surface-matching choice, since it is what the committed driver used. |

| 07-26 | *`ENV_YAW_MAX = 12.0` — annotated **`# MEASURED envelope limit`** in `taniteval/corridor.py:110` and **`# MEASURED — P1 envelope`** in `run_gate.py:614`*, and relied on as the yaw edge by every closed-loop OOD verdict in the program | **NEW CLASS — C14: A SWEEP'S GRID END RE-LABELLED AS A MEASURED LIMIT** *(a default argument string mistaken for an experimental result — the instrument reported where we stopped looking, and we recorded it as where the effect stopped)* | **The sweep stopped at 12° because the STRING stopped at 12°.** `lowood_probe.py:228` and `lowood_ci.py:114` both read `--yaw-grid default="0,1,2,3,5,8,12"`. **No criterion selecting a yaw edge exists anywhere in P1.** ⚠️ **And the two axes were never set by a common criterion:** P1's own report puts the yaw no-degradation edge at **≤ 2°**, with the paired Δ CI-separated from **3°** onward — so `ENV_YAW_MAX` sat **FOUR sweep points deep into separated degradation** while `ENV_LAT_MAX = 3.0` sits at its **FIRST**. Evidence class was **INHERITED**, never MEASURED. **MEASURED 2026-07-26** (881 windows / 40 clusters, episode-cluster bootstrap B=2000 **on the edge itself**): usable yaw edge **15.47° [12.14, 17.88]** — the CI **lower bound TOUCHES the shipped 12°**, i.e. **1.29× at the point estimate and NO widening at the lower bound**; information **fully destroyed at 26.41° [18.33, 29.63]**, corroborated **model-free** by the FOV half-angle at **25.70°** (two independent instruments, one answer). At the shipped 12° the warp has already **destroyed 34.7 % of usable information and FABRICATED 26.4 % of pixels**. ⛔ **AND WIDENING IT RESCUES NOTHING — settled before any GPU ran:** even at **yaw = ∞** the *lateral* clause alone leaves **3.75 %** of K=20 windows outside (junction **18.13 %**), and MEASUREMENT requires **zero**. K=60 would need **39.25°** (junction 60.03°) — **1.5×–2.3× past total destruction** — and junction p90 at the standing 2 s horizon is **28.14°, already past it**. ⇒ **CLOSED-LOOP NUMBERS ARE EXTRAPOLATIONS AT EVERY ADMISSIBLE HORIZON AND MUST BE LABELLED SO PERMANENTLY.** This closes the last branch `GATE_30K_RESULTS.md` §10.1 left open. ⭐⭐ **THE REDIRECT THAT MATTERS MORE THAN THE RETRACTION: the envelope is NOT a renderer-fidelity envelope.** The yaw warp is geometrically **EXACT for arbitrary depth** (`max|ΔH| = 0.000e+00` over 30 conditions). Roughly **half of what the envelope measures is OUR ARM'S OOD SENSITIVITY** — measured on v1 and applied to v4. ⇒ **the lever is TRAINING-TIME OFF-PATH AUGMENTATION, not rendering.** We were about to buy a renderer to fix a training problem. ⛔ **DO NOT RESURRECT THE OOD RATIO:** extending the sweep raises `sup(ratio_arr)` to ≈**1.52**, clearing the 1.5 threshold by 0.02 — **C13 in a new costume**. Clause 1 stays VOID. ⚠️ **BINDING LESSON: before recording a limit, ask whether the instrument could have reported a LARGER value. If the grid, string, range or loop bound could not have exceeded it, you have measured your own configuration, not the world.** Sibling of C13 (a guard that cannot fail) and C9 (an instrument at the wrong horizon) — all three are instruments structurally incapable of reporting the answer they were cited for. *(Discipline note, and it is why this is CONFIRMED not PROVISIONAL: pre-registration written before measuring; the C13 gate applied to P1's **own** criterion, which PASSES (`dead_black` +1.5619) so an easy exculpatory outcome was refused **on evidence**; **80 reproduction checks against the committed P1 artifact, 0 mismatch**; and one **self-refutation recorded rather than overwritten** — the agent interpolated the junction stratum to a minority, then measured 0.5165 and reported the correction. The grid-terminus claim was independently re-verified by me directly in both source files.)* |

| 07-26 | *"**ZOD is the publishable twin and is one human action away** — the only commercially usable candidate found"* — **MY OWN** recommendation, put to Sayed as a decision item and **APPROVED BY HIM ON IT** (*"take ZOD"*) | **C4 — A CANDIDATE REJECTED FOR REASON A, PROMOTED FOR REASON B, WITHOUT CHECKING PROPERTY C** *(the property that made B worth anything). I verified the LICENCE and never verified the CONTENT.* | **ZOD HAS NO LANE GRAPH.** Its lane annotations are **2-D IMAGE-SPACE MARKING POLYLINES** with **no successor, predecessor, neighbour, connectivity or topology field, and no map module at all**. Confirmed at **four** independent probes: the licensor's annotations page, the ICCV paper, the MIT devkit source `zod/anno/lane.py`, and **a fourth check I ran myself** — the full field set is `uuid` · `geometry` · `type` · `colored` · `instance_id` · `cardinality` (+ road-painting booleans), with `geometry` typed `List[List[float]]  # [[x1, y1], …]`. ⚠️ **How the error was made:** the prior survey correctly rejected ZOD **for ACCESS** (an application is required) and never assessed its **contents**. I then promoted it to *"the publishable twin"* on the strength of its **licence alone** — `CC-BY-SA-4.0`, share-alike but not non-commercial — because the licence was the scarce property I was hunting. **A commercially usable licence over a corpus that lacks the lane graph buys nothing**, and the lane graph was the entire reason the twin was wanted (S1 branch selection, S2 lane selection, HP-4). ⚠️ **BINDING LESSON: when promoting a candidate that was previously rejected, re-derive its VALUE from scratch — the earlier rejection means nobody ever checked the properties beyond the one that killed it.** A rejected candidate carries no verified attributes at all, only a verified defect. ⭐ **WHAT REPLACES IT — and it is better on the axis that mattered: OVERTURE MAPS `transportation`.** Commercially usable (**ODbL-1.0**), **no gate** (2 endpoints, HTTP 200), and a **byte-verified routable graph — 20,000/20,000 segments carry ≥ 2 connectors**, read via **parquet range requests (3.3 MB fetched, not 14 GB)**. It additionally carries **`prohibited_transitions`** (turn restrictions) and **`destinations`** (signposted route/goal) — **which AV2 lacks entirely**. ⚠️ Bounds: **road-level, not lane-level**, so it is NOT a substitute for AV2 on lane selection; and a licence trap was caught in passing — the widely-quoted *"CDLA-Permissive-2.0"* is the **foundation default**, while the transportation theme is **copyleft ODbL**. ⚠️ **Whether an ODbL-trained MODEL is a "Derivative Database" is UNSETTLED** and is not agent-decidable — so the follow-on *"map-match comma2k19 onto Overture ⇒ ship-tier"* is **NOT established**: the same report that proposes it also states the licence question is open, and both cannot be true at once. *(Cost: near zero, and only because the check was commissioned before anything was spent — no email was sent, no account created, no application submitted. But the recommendation had ALREADY REACHED THE PI AND BEEN APPROVED, which is exactly the failure `BOOST_PROGRAM` M1/M2 exist to prevent: I recommended at PROVISIONAL and it was acted on as if CONFIRMED.)* |

| 07-26 | *"`canary_rollout`'s docstring carries a v1-line reference of **~0.452** for `wm_canary_ade_2s`"* — carried into the blind-imagination brief as the v1 number to establish | **C4 — INHERITED WITHOUT RE-VERIFICATION, in its `heldout`-vs-`full_set` form** | **0.452 is not an independent measurement.** It is the **`heldout` split-mean 0.4522** of `ade_0_2s` itself (`MODEL_REGISTRY §1.2`) — i.e. the deprecated `overlapping_holdout_se` central value of a number whose `full_set` value is **0.4271**. ⭐ **And the deeper correction is that the two quantities are THE SAME QUANTITY:** `metric_dynamics.rollout_decode` appends the model's own `z_hat` and **never re-encodes a frame** (`:241`), so `taniteval.rollout.collect`'s headline `ade_0_2s` **already is** a blind-imagination rollout. The program has been measuring blind imagination all along and reading it at one horizon (k = 20) under the expert's true actions. ⚠️ **Second-order consequence, MEASURED: that number is decoded with `grounding.step["op"]`, whose forward-consistency loss was trained over `op_fwd_k = 4` steps (0.4 s) — so the program's headline is read 5× beyond its decoder's calibration**, and swapping to the 20-step-calibrated `step["str"]`/`["tac"]` moves v1 from **0.3839 → 0.1950 / 0.1865** on 596 episode clusters. **BINDING LESSON: before "establishing" an inherited reference number, check whether it is a DIFFERENT ESTIMATOR OF A NUMBER YOU ALREADY HAVE.** *(MEASURED: `…/2026-07-26-blind-imagination/artifacts/{gate_reproduction,horizon_curve}.json`.)* |
| 07-26 | ⛔ **MY OWN pre-registered headline rule.** *"`T_blind` = the largest N such that imagination is separated-better than frozen-last-frame at every N′ ≤ N"* — contiguity anchored at **N = 1** | **C13 — A CRITERION THAT CANNOT FIRE** *(the guard-cannot-fail class, applied to a HEADLINE STATISTIC rather than to a guardrail — and written into a pre-registration, which is exactly where it is hardest to notice)* | Arms (a) imagination, (b) frozen-last and (c) full-observation decode a **bit-identical first transition by construction** — they share the observed window and only diverge from step 2 (pinned by `test_first_step_is_identical_across_state_sources`). So the paired Δ at step 1 is **exactly 0.0** and its bootstrap lower bound is **exactly 0.0 in all 2,000 draws**: the rule returns **`T_blind = 0` for every arm, in every regime, regardless of the data.** It did — 5/5 regimes, `frac_draws_T_blind_is_zero = 1.000` — and it was caught only by reading `delta_at_step1_m` beside the verdict. Repaired by anchoring contiguity at step 2 (amendment A4, pinned in code as `bi_analyze.T_CONTIGUITY_START_STEP`); the repair is strictly *more permissive*, so it cannot manufacture the run's negative result. ⚠️ **BINDING LESSON: C13's question — "what value would make this fail?" — must be asked of the STATISTIC, not only of the guardrail. Where two arms share a construction, the statistic is DEGENERATE exactly where the arms agree, and a rule anchored there is decided before any data arrives.** *(MEASURED: `…/2026-07-26-blind-imagination/artifacts/t_blind.json`.)* |
| 07-26 | *"PhysicalAI-AV's wheelbase is **2.85 m for ~90 % of clips**, 3.165 m for the other ~10 % — so the pipeline's 2.9 constant is off by **1.5 %**"* — the feature probe's figure, which set the SIZE of the proposed fix and propagated into `GEOMETRY_INTEGRITY_AUDIT.md:40,77` and into the option-A/B/C decision brief | **C2 — A DISTRIBUTION READ FROM A SINGLE PROBE** *(the absence-class in its statistic form: one chunk quoted as the corpus)* | Measured on **`vehicle_dimensions.chunk_0000` alone**, and that chunk is **100 % United States**. Over the **197 chunks the parity corpus actually draws from** there are **five** wheelbases — **2.730 (47.0 %, the MODE) · 2.850 (1.8 %) · 3.135 (13.9 %) · 3.165 (25.5 %) · 3.216 (11.8 %)** — so the quoted majority population is **1.8 % of the corpus, not ~90 %**, and the corpus-weighted error is **+6.23 % to −9.83 %, not 1.5 %**. ⚠️ **The sign is INVERTED for 47 % of clips.** The reason one chunk misled so badly is itself measurable: `I(wheelbase; country) = 0.769` of `H = 1.880` bits, so **40.9 % of the wheelbase entropy is explained by country** — chunks are geographically coherent, which makes any single chunk a *systematically* biased sample of this variable rather than a noisy one. ⚠️ **BINDING LESSON: a DISTRIBUTION quoted from one shard is a C2, not a small-sample. Before quoting a corpus statistic, check whether the shard key correlates with the variable — if it does, one shard is not a sample of the corpus at all.** *(Cost: it would have justified a fix sized 1.75 % against a corpus whose modal error is 6.23 %. Caught by the option-C measurement before any code changed; the corrected figures are what PI-approved option B is built on. MEASURED: `…/2026-07-26-wheelbase-impact/wheelbase_population.json`.)* |
| 07-26 | *"`trafficsim` is an owned-but-never-switched-on asset; the 4-brain tactical gate's **[−0.21, +0.14] m vs a 4.5 m noise floor** failure may therefore be a CONFIGURATION artefact, and that negative result void"* — carried into a task brief as the reason to re-run the gate | **C1 — STALE SOURCE USED TO IMPEACH A NEWER MEASUREMENT** *(a circular premise: the doc cited as evidence that the feature was off is OLDER than the run that turned it on — and the number being impeached came FROM that run)* | `RUN_RECIPE.md:26` (*"trafficsim (disabled by default)"*, written **07-22**) is **still literally true and still the default** — it describes the config default, not run history. The `[−0.21, +0.14]` figure is `GATE_RESULTS.md` §2.4, produced **07-26** by a session whose §2.1 records **fetching the CATK weights (sha256 `7c5a89bc…`, re-verified byte-identical this session), building the PyG extensions from source, and running the service** — with `IS_REPLAY: false` proven (agents 8.37–78.17 m off their logged tracks). **trafficsim WAS enabled when the gate failed.** ⚠️ **BINDING LESSON: before using doc A to impeach measurement B, check that A is NEWER than B. A default documented in a recipe is not a record of what has been run.** ⭐ **The probe was still worth it — it surfaced a different, true finding:** `disabled.yaml` sets `endpoints.trafficsim.skip: true`, and `traffic_service.py:simulate_traffic` shows `skip` is **literal REPLAY** — so **every published TanitAD closed-loop number (REF-C n=12, flagship-vs-REF-C n=12, native-1080) ran against non-reactive replayed traffic.** Paired comparisons stand; "no closed-loop number has ever involved a reactive agent" is now on the record. *(MEASURED: `…/2026-07-26-trafficsim-wheelbase/TRAFFICSIM_WHEELBASE.md` §1.)* |

| 07-27 | *"`base_rank` in `raw/v5_v4_windows_reduced.pt` is the argsort of the deployed grafted score, best first"* — and, downstream of it, V5 §5.2's breadth axis labelled *"keep the top-n candidates by the AS-TRAINED base ranking"* | **NEW CLASS — C15: A TENSOR'S SEMANTICS TAKEN FROM ITS NAME RATHER THAN FROM ITS CONSTRUCTION SITE** | **`base_rank` is not a rank.** It is `[the as-trained pick] ++ [anchors 0..255 in INDEX order, pick removed]`. First caught by the E-H1 stream on 881/881 rows; **independently re-verified here on 881/881, plus a second, sharper probe the first did not run: `881/881` rows have a tail that is STRICTLY INCREASING in anchor index**, i.e. columns 1.. carry *no score information whatsoever* (`…/2026-07-27-lambda-tau-sweep/raw/eh2_gate.json::G4`). The name was persuasive because **column 0 is genuinely the deployed pick**, so `fan_err4.gather(1, base_rank[:, :1]).mean()` = **0.8563** = `F_flat` exactly — a real consistency check that passes for a reason unrelated to the claim, which is what made the wrong reading survive. ⚠️ **BINDING LESSON: a check that passes on column 0 says nothing about columns 1..N. Verify a tensor's semantics against the CODE THAT WROTE IT, not against a spot check that the name predicts.** ⭐ **The conclusions built on it SURVIVE and are not weakened** — E-V5-3's finding is that *letting the imagination rule consider more candidates makes it worse*, which holds for **any nested family** of candidate sets, and index order is still a nested family; *"breadth costs −10.66 m"* stands. ✅ **FIXED AT SOURCE, three places**: `code/v5_cost_curve.py` (comment rewritten, variable renamed `nested_order`, dump gains a `nested_order` key and a `_base_rank_IS` note; the legacy `base_rank` key is kept so staged `.pt` files stay readable), and `V5_IMAGINATION_SELECTION.md §5.2` (correction block, conclusions explicitly preserved). |

---

## Standing consequences

- **C1** ⇒ trainer logs watch curves; **only `eval_*.py` output is quotable**.
- **C2** ⇒ two probes minimum, and prefer the tool that *owns* the fact (`nvidia-smi --query-compute-apps`
  over `ps`; `git ls-files` over a triage doc; a real `dd` over `df`).
- **C3** ⇒ if a claim would change a decision, measure it first; a mechanism is a hypothesis until then.
- **C4** ⇒ mark INHERITED explicitly, and never let INHERITED decide a GPU-day.
- **C5** ⇒ bucket means, never single rows; no exponent without window + R² + n.
- **C6** ⇒ name every difference between the arms before reading the contrast.
- **C15 — A NAME IS NOT A SPECIFICATION** ⇒ before building on a dumped tensor, **read the code that
  wrote it**, and design the check so it could FAIL. `base_rank` passed a real consistency test
  (column 0 → 0.8563) that the wrong reading also predicts. Applies to every reduced dump in the
  program; the cheap fix is a `_<key>_IS` docstring key beside every non-obvious tensor.
- **C14 — GRID END ≠ MEASURED LIMIT** ⇒ **ask whether the instrument could have reported a LARGER
  value.** If a grid, default string, loop bound or range could not have exceeded the number you are
  recording, you have measured **your own configuration**, not the world. Record such a value as
  `INHERITED (instrument bound)` and never as MEASURED. ⚠️ Completes a trio with **C13** (a guard that
  cannot fail) and **C9** (an instrument at the wrong horizon): all three are instruments *structurally
  incapable* of reporting the answer they were cited for, and all three produce **confident silence**.
- **C10 — EVALUATOR ≠ PRE-REGISTRATION** ⇒ **an instrument that refuses to render a verdict is not neutral —
  the gap gets filled by hand and then re-quoted as machine output.** Before a gate adjudicates a GPU-week,
  run its ADMISSIBILITY path against the exact shape of a finished run (final step, step naming, units), not
  just its thresholds. **MEASURED twice now on the same defect** (`3ff5499`, then a surviving sibling), which is
  why the repair NAMES the indexing convention it used: an implicit convention is an unauditable one, exactly as
  with an implicit horizon (C9). ⚠️ When an instrument returns no verdict and a human supplies one, **the report
  must say which is which** — the v4 report did; my chat summary did not.
- **C9 — HORIZON-BLIND INSTRUMENT** *(new class, added 2026-07-26)* ⇒ **a metric measured at a horizon
  shorter than the failure it is meant to detect will report success indefinitely.** Not a sampling
  error, not a confound, not a stale source — the instrument is simply pointed at the wrong timescale,
  so *more data and tighter intervals make it more confidently wrong.* **MEASURED on the same 43
  windows:** corridor departure **0.0035 @ K=20 → 0.5877 @ K=185** while the paired **ADE@2s delta is
  0.0109 [−0.0, 0.0312], not separated** — the 2 s instrument recorded essentially nothing while the
  arm departed its corridor on 59 % of windows (junction 84 %, peak XTE 38.94 m). **It hid the dominant
  failure mode by ~168×.** ⇒ *Standing consequence:* **every gate verdict must NAME its metric's horizon
  and n** (`GATE_PROTOCOL.md` §0, enforced in `run_gate.py`: K≤20 refused as the blind horizon, K>190
  refused as structurally impossible, `INCOMPLETE` when a registered co-primary is unmeasured). The
  decisive demonstration, one arm and one checkpoint: **REF-C base-30k passes `ade_0_2s` 0.4728 against
  a 0.60 bar → old gate CONTINUE; corridor@K185 0.5877 / junction 0.8414 fail → new gate RESTART.**
  Same weights, opposite decisions. ⚠️ Sibling to C1 (*faster-moving source*) and to the 07-25
  metric-definition case (*a metric NAME is not a metric DEFINITION*): all three are "the number was
  computed correctly and answers a different question than the one asked."
- **C16 — FABRICATING INTERMEDIARY: the source text never existed** *(new class, added 2026-07-27)* ⇒
  **a summarising fetch layer can invent a verbatim quotation, a section name and numbers, and they
  read exactly like primary evidence.** MEASURED in the latent-action research stream: fetching
  **arXiv:2605.20223 as a PDF** returned a named section, a **verbatim quote** — *"latent actions
  cannot inherently recover metric/scale information"* — and a *"when not to use latent actions"*
  recommendation. **All three returned NOT FOUND at full-text and abstract depth.** The same path
  invented **three numbers** for a second paper (MVP-LAM).
  ⚠️ **And note WHICH claim was fabricated: the single most load-bearing one in the brief** — it would
  have settled the stream's crux question *by citation instead of by reasoning*. The hallucination
  landed precisely on the question the agent most wanted answered. **Treat convenient primary-source
  quotes with more suspicion than inconvenient ones, not less.**
  **Distinct from C4 (inherited/propagation), which is why it gets its own class.** C4 is *"a real
  claim travelled without re-verification"* and its remedy is to re-verify the chain. Here **there is
  no chain — the artifact never existed**, so re-verifying the relay finds nothing wrong. Sibling to
  the C13/C14/C9 trio in that the instrument produces **confident output it is structurally unable to
  support**, but worse: those report a real measurement of the wrong thing, this reports a measurement
  of nothing.
  ⇒ **STANDING RULE: PDF-summarisation output is a model-generated summary, NOT source text, and is
  INADMISSIBLE as a quotation.** A verbatim quote or a cited number must come from HTML full text or
  the abstract, retrieved directly. **Every citation records the depth at which it was verified**
  (`CITATIONS.md` in the latent-action stream is the reference implementation — depth per row, with
  the fabricated entries quarantined rather than deleted). A PUBLISHED evidence class is only as good
  as the retrieval that produced it: **`PUBLISHED` now requires a stated fetch depth.**
- **C17 — MARGINAL MISTAKEN FOR CONDITIONAL** *(new class, added 2026-07-27)* ⇒ **a marginal
  distribution that looks wrong tells you NOTHING about whether the conditional is wrong, and the
  conditional is what the system actually uses.** MEASURED: the trajectory fan's candidate speed
  tracks ego speed at slope **−0.129** where ground truth tracks it at **+1.0003** — a genuinely
  un-conditioned marginal, and true. From it I inferred that the fan was mis-placed and that
  longitudinal admissibility was the highest-value engineering task. **It cost nothing at all:
  100.0 % of windows already contain a candidate within 0.5 m/s of the speed the car actually took**
  (mean gap **0.0525 m/s**), and restricting the oracle to speed-matched candidates moves it
  **+0.0000 m [0.0000, 0.0000]**. **The fan was WIDE, not MIS-PLACED.**
  ⚠️ **Two independent streams converged on the wrong lever, and convergence read as confirmation.**
  One measured the *span* (108.7 m per window vs a 25.40 m ground truth), the other the *unreachable
  anchors* (94.36 %) — **both marginal, neither conditional.** Agreement between two measurements of
  the same marginal is not replication of a claim about the conditional; it is the same blind spot
  twice. ⇒ **Before acting on "the distribution of X is wrong", measure `P(good | context)` — the
  quantity the system consumes — not `P(X)`.**
- **C18 — CORRELATION WITHOUT SLOPE** *(new class, added 2026-07-27)* ⇒ **a correlation coefficient
  can be near-perfect while the slope says the opposite of what you conclude; the SLOPE carries the
  physics and the correlation carries only the tightness.** MEASURED in the same stream: **r = −0.974**
  on a relationship whose slope is **−0.129** against a required **+1.0**. The correlation says the
  points lie on a line; **only the slope says which line**, and a strong correlation on the wrong
  slope reads as strong evidence for the wrong model. ⇒ **Quote a slope with its units, or do not
  quote the relationship.** Sibling to the standing rule that an exponent is inadmissible without its
  fit window, R² and n — same failure, one dimension down.
- **C19 — A STRATUM WIN IS NOT A DEPLOYABLE WIN** *(new class, added 2026-07-27)* ⇒ **an effect
  measured inside a stratum must be multiplied by that stratum's FIRING RATE before it can be
  compared to anything, and a stratum mean cannot be compared across gates with different firing
  rates at all.** MEASURED: the good-world-model stratum shows **0.7085 → 0.3330, −0.3754** — *"a
  53 % cut in selector error"* — but it fires on only **22.7 %** of windows, so **deployed as a policy
  it is worth −0.0852 [−0.1190, −0.0548]. Quoted bare, it overstates the deployable win by 4.4×**
  (0.227 × 0.3754 = 0.0852). It had already propagated bare into two documents.
  ⚠️ **The correct frame is a POLICY over ALL windows** — *"use B where the gate fires, else A"* —
  because that is the only form in which two gates are commensurable. Converting to it reversed the
  conclusion: the gate was worth **3.4× LESS** than simply applying the treatment **ungated**, and
  even an *unattainable perfect* gate lost to it. ⇒ **Every conditional result reports its
  `selected_frac` beside its effect, and reports the whole-set policy value. A gate that fires on
  everything or nothing is degenerate and must be visible as such** — in this same stream **two
  PURE-NOISE gates would have been written up as separated wins** without that column.
- **C20 — OPTIMISE THE OBJECTIVE YOU ARE PAID FOR, NOT ITS LEGIBLE CORRELATE** *(new class, added
  2026-07-27)* ⇒ **when a decision rule can be aimed either at the quantity you actually want or at an
  interpretable proxy that correlates with it, aiming at the proxy is a measurable, quantified loss —
  not a stylistic choice.** MEASURED, same features and same folds: gating on **predicted C2-vs-A0
  utility** (the thing we are paid for) recovers **−0.1397**; gating on the **predicted canary** (the
  legible correlate we had named as the missing instrument) recovers **−0.0383**. **Aiming at the
  correlate costs 3.6×** — and the canary is a weak gate *even as an oracle*, recovering only 9.3 %
  of the available headroom on v1's world model. ⇒ **State what the rule is optimising and why it is
  the payoff rather than a stand-in.** Related to C9/C13/C14 (instruments structurally unable to
  report the answer they are cited for), but distinct: this instrument reports its own quantity
  correctly — **the quantity is simply not the one that pays.**
- **C21 — A DOCSTRING IS NOT A MEASUREMENT** *(new class, added 2026-07-27; a sharper instance of the
  "prose lied to us" rule `CLAUDE.md` opens with)* ⇒ **a module docstring and a usage example in a
  `--help` block are PROSE. Grepping them and calling it verification is the same error as quoting a
  weekly report.** RETRACTED: *"verified — the flagship trains on a comma2k19 + PhysicalAI mix."*
  **It does not.** `flagship4b-speedjerk-30k` trains on **PhysicalAI alone, 100 %**: `--data cached`
  **discards every cache dir after the first** (`train_flagship4b.py:186-188`), confirmed
  independently by the registry command **and** by the run's own committed config JSON. The
  *"0.40/0.60 mix"* is a **stale docstring belonging to a different run** (`p0-sB01-realmix`).
  ⚠️ **What makes this expensive rather than merely wrong: the false premise became a DECISION.** It
  produced a three-option comma2k19 dilemma (per-corpus geometry / letterbox / drop), that dilemma was
  **escalated to the PI as a decision he needed to make**, and it was written into **three agent
  briefs** as a binding constraint. **None of it existed. There is no comma mixture to break.**
  ⇒ **A claim about what a RUN did is answered by the run's config JSON, its launch command, or the
  code path that consumes the flag — never by the docstring above it.** The evidence class for
  "grepped a docstring" is **not** MEASURED; at best it is a hypothesis to go and check.
  *(Also flagged for repair: `GEOMETRY_INTEGRITY_AUDIT.md:26` and `train_flagship4b.py:3-4` both assert
  the corpus mix the deployed model does not have.)*
- **C22 — BOUND QUOTED AS CAPABILITY** *(new class, added 2026-07-27)* ⇒ **an oracle result is an
  UPPER BOUND on what an achievable version could deliver, and the distance between the two is not a
  detail — it can be the whole result, and it can have the opposite sign.** MEASURED: handing the
  selector the **true 2 s goal position** recovered **88.0 % of the fan's headroom, separated,
  replicated ×3** — reported (by me, to the PI) as the first thing in the program to clear both bars.
  The achievable version is **separated-WORSE**: break-even needs **σ₀ = 0.955 m** radial RMS
  (**0.721 m** for the realistic *biased-regressor* family), the best out-of-fold head achieves
  **1.330 m** ⇒ **recovery −10.4 %**, and the latent-only head a strategic brain would actually carry
  is **+0.0464 [+0.0164, +0.0792]** — damage, on all three fans, with the curve reaching **+8.37 m**.
  ⇒ **Every oracle number is quoted with (a) the achievable value or (b) an explicit "not yet
  measured", and a bound NEVER licenses a decision on its own.** Sibling to **C19** (a stratum win is
  not a deployable win): both are *"the number is real and it is not the number that would ship."*
- **C23 — ORACLE SHAPED AS EGO STATE** *(new class, added 2026-07-27; caught PRE-FIT, which is the
  point)* ⇒ **a feature can carry future information while sitting in a dump beside genuine
  present-time state, and its NAME will not tell you.** MEASURED: `head_deg` is the **future net
  heading change** over the window and sits next to `v0` in every fan dump. Fitting a "deploy-time"
  head on it would have produced a strong, plausible, entirely leaked result — the same shape as
  REF-A's I-JEPA val leak. ⇒ **Before fitting anything called deploy-time, audit every input for
  future content by DEFINITION, not by name.**
- **C24 — RMS PLACED ON A NOISE CURVE** *(new class, added 2026-07-27)* ⇒ **an achieved RMS error and
  a synthetic-noise σ are not the same quantity, and reading one off the other's curve mis-states the
  damage — here by 5.7×.** A noise sweep injects zero-mean isotropic error; a real estimator is
  **biased and correlated with the target**, which is why the `SHRINK` family sits **25 % stricter**
  than the isotropic one. ⇒ **Place a measured estimator on a requirement curve only via a family
  that matches its error STRUCTURE, and say which family you used.**
- **C25 — AN UNPAIRED POINT-ESTIMATE LADDER QUOTED AS A MEASURED EFFECT** *(new class, added
  2026-07-27)* ⇒ **a monotone-looking sequence of point estimates is not a measured trend. Bootstrapping
  each rung AGAINST CHANCE tells you nothing about whether one rung differs from the NEXT — and the
  contrast between rungs is the only quantity the claim is about.** RETRACTED: *"vision enters at
  rank ≈ 16"* (`3.659× → 3.685× (k16) → 3.000× → 2.116× → 1.59×`), which sat under **VALIDATED** in the
  v5 PREP card and propagated into **≥4 documents and three agent briefs, mine included**.
  **The paired test cost seconds — the raw held-out scores were already banked — and it kills the
  headline: the "peak at k=16" is `+0.00085 [−0.02204, +0.02299]` against ego alone, a CI 27× WIDER
  THAN THE EFFECT. It cannot distinguish "16 dimensions of vision help" from "vision contributes
  zero."** ⭐ **The decisive control was cheaper still: the image-ONLY ladder is FLAT to 5 decimal
  places.** If 16 were the visual state's information content, image-alone would peak at 16 and fall.
  ⚠️ **Three defects rode along, and each is its own trap:** the ladder **spliced two instruments**
  (rungs 1–4 a linear ridge, rung 5 a 2.17 M-param attention head on a different baseline); it measured
  a **linear probe on a binary anticipation target, not the predictor it was cited to constrain**; and
  the **"replicated by two independent streams, all ten arms selecting r=16" ran three arms that do not
  read the PCA rank at all** — *a replication of something else.*
  ⇒ **Before quoting a dose–response, ADJACENT-RUNG CONTRASTS OR NOTHING**, and state the instrument
  for every rung. ⭐ **And note what the refutation bought: it CANCELLED a planned experiment** —
  re-measuring the ladder on wider crops would have reproduced the same shape regardless of crop
  content. *A false premise does not only mislead a decision; it funds work that cannot inform one.*
- **C26 — A RIG-CORRELATED FABRICATION IN THE DEPLOYED INPUT, PRESENT IN EVERY NUMBER SINCE D-016 R1**
  *(new class, added 2026-07-27: **the preprocessing itself was the confound**)* ⇒ **a geometry fix can
  introduce a corpus-identifiable artefact that correlates perfectly with a latent grouping variable,
  and nothing downstream will flag it because every metric is computed on the fabricated pixels.**
  MEASURED on 10 real clips: today's canonical crop **replicate-pads 0.00 % of rows on rig A and
  11.21 % on rig B** — and the rig split is **29.1 % of clips**, stamped consistently across all three
  120° f-theta cameras. ⇒ **roughly a third of our training frames carry invented pixels the rest do
  not, in a pattern that identifies the rig.** This model demonstrably eats shortcuts (zeroing `v0`
  moves the imagined decode **×93.7** while the perceived decode is bit-exactly unchanged), so a
  free rig label is not a harmless artefact.
  ⚠️ **Scope: this is in EVERY number the program has produced since D-016 R1, not only in v5's
  future.** It is not retracted here — it is **flagged as an unquantified common-mode confound** on
  the whole post-D-016 record, and the cost of quantifying it must be stated before anything is
  restated. **A widened CROP inherits it and makes it worse (120° → 19.91 %); the cylindrical
  projection removes the fabrication entirely** (an explicit mask, not replicate-pad), cutting rig B
  to **0.69 %** at 100°.
  ⭐ **And the residual is the subtle half: a rig-correlated BLACK region is still a rig-correlated
  signal.** Masking converts fabricated content into *identifiable absence*, which a shortcut-hungry
  model can use just as well. **The clean fix is a vertical field BOTH rigs fully observe** — made
  expressible, deliberately **not** chosen, because choosing it is a measurement someone still owes.
  ⇒ **Before trusting a preprocessing "fix", ask what it FABRICATES and whether the fabrication
  correlates with anything.** Sibling to C17 (marginal vs conditional): both are defects invisible to
  every downstream metric because the metric is computed *after* the defect.
- **C27 — REAL-vs-SHUFFLED MEASURES HARM AVOIDED, NOT BENEFIT GAINED** *(new class, added 2026-07-27;
  the sharpest methodological catch of the session)* ⇒ **a two-arm design that contrasts a REAL feature
  against a SHUFFLED one cannot tell you whether the feature helps — only whether corrupting it hurts.
  A feature the model has learned to route around will still show "real ≫ shuffled".** MEASURED: the
  within-PhysicalAI geometry test gives **geometry-vs-nothing NOT separated** while **real-vs-shuffled
  IS separated**. ⚠️ **Reported on its own, that second number reads as "geometry works, p < 0.05" — and
  it is the number a motivated analyst would reach for.** The three-way **real / shuffled / NONE**
  design is what catches it, and only that design.
  ⇒ **Every ablation carries a NONE arm.** A shuffled control bounds *corruption sensitivity*; only an
  absent arm bounds *contribution*. Sibling to **C13** (a guard that cannot fail) and **C19** (a stratum
  win is not a deployable win): all three are *"the number is real and it answers a question nobody
  asked."*
- **C28 — A CONSTANT WHERE THE QUANTITY IS PER-CLIP** *(new class, added 2026-07-27)* ⇒ **when a
  physical quantity is measured per clip and code stores it as a program-wide constant, every
  downstream number inherits an error nobody can see, and the constants BREED.** MEASURED: PhysicalAI
  camera height is **per-clip, 1.245–1.607 m — a 29 % spread**. **All three constants circulating in
  this program (1.5 / 1.43 / 1.22 m) are wrong, and 1.22 is BELOW THE OBSERVED MINIMUM.** Four files
  hard-code one; overlays are off by up to **29 %**. ⚠️ **And the obvious proxy fails too: rig identity
  does NOT predict height** — rig medians are **1.5 % apart** while within-rig spread is **29 %**, so
  "handle the two rigs" would not have fixed it.
  ⇒ **Before a constant is written, check whether the corpus ships the value per unit. Three
  unreconciled values for one quantity is not a documentation problem — it is the signal that the
  quantity was never constant.**
- **C29 — THE MODEL WAS RIGHT AND THE LABEL WAS WRONG** *(new class, added 2026-07-27)* ⇒ **a channel's
  worst metric is a hypothesis about the MODEL only if the labels have been audited; ours had not.**
  MEASURED: comma2k19 heading is `arctan2` of ENU velocity and is **undefined at standstill** —
  **26.27 % of frames below 0.5 m/s are physically impossible, and 0.000 % above it** (PhysicalAI:
  zero in every bin). Repairing the label moved the **already-deployed head, with nothing retrained,**
  from pooled `yaw_rate` **R² 0.105 → 0.811**. On the 50 changed windows the vehicle was stationary
  (max 0.53 m/s), the label claimed up to **9.47 rad/s**, and **the model had predicted 0.023.**
  ⭐ **Reported against its own interest, which is what makes it trustworthy: R² and MAE (−33 %) move,
  but medAE moves 0.5 % and nMedAE gets slightly WORSE.** The repair fixes **the tail and the summary
  statistic, not typical accuracy** — a distinction a headline R² erases.
  ⇒ **Bin the residuals by a physical covariate BEFORE blaming the architecture.** The defect here was
  visible in one speed-bin table and had survived every previous investigation of that channel.
  **⤷ FORWARD POINTER 2026-07-27 (`anchor-settlement`; this row is NOT rewritten and keeps its date):**
  ⛔ **the `0.105 → 0.811` and its comma component `+0.0114 → +0.3308` are WITHDRAWN** — **2 of the
  22 comma val episodes** they were measured on are, BY CONTENT (sha256 of raw pose bytes *and* raw
  `frames_u8` bytes), bit-identical to 2 of the scoring head's own comma TRAINING clips. Content-clean
  comma value: **−0.746**. The **pooled** figure inherits the contamination through its comma half and
  must not be quoted either (it never should have been — *per corpus, never pooled*).
  ⚠️ **What C29 got RIGHT is untouched and is the durable part:** the label defect is real, the
  26.27 %/0.000 % speed-bin table is a fact about the LABEL measured on all 64 val segments,
  **the model was right and the label was wrong**, and PhysicalAI is unaffected (`n_pai_changed = 0`,
  yaw R² **+0.903482** bit-identical — re-measured 2026-07-27, not inherited). What does not survive
  is the *size* of the gain. See **C43**.
- **C30 — RECOVERY CONDITIONAL ON AN UNREPORTED BACKGROUND** *(new class, added 2026-07-28)* ⇒ **a
  "% of headroom recovered" is not a property of the treatment alone — it is a property of the
  treatment AND the background the other axis was held at, and the background can move the answer
  further than the treatment does.** MEASURED: **at fixed n and fixed along-track error, recovery
  spans +13.3 % … +29.2 % PURELY on the cross-track background — a 15.9-point swing — and SEPARATION
  FLIPS inside that range.** ⇒ E-GOAL-1's headline **+23.6 % was conditional on a background it never
  named**, and so is every other recovery figure this program has published.
  ⚠️ **How it was caught is the instructive part: the registered bridge FAILED.** E-GOAL-1's
  background could not be rebuilt at n = 600 (it needs v4 latents that exist only on the 881 windows),
  and **the registered substitute deviated +5.6 recovery points and flipped `by_speed` to separated at
  n = 40 — using it would have MANUFACTURED a CONFIRM.** The stream reported the bridge failure
  instead of quietly substituting, which is the only reason the class exists.
  ⇒ **A recovery number without its background is INADMISSIBLE.** State the background, and if it
  cannot be reproduced at the new n, say so rather than substituting.
- **C31 — A PREDICATE THAT STOPS DISCRIMINATING AT HIGH n** *(new class, added 2026-07-28; the INVERSE
  of under-powering, and far more dangerous because it looks like a stronger result)* ⇒ **a decision
  rule tuned at small n can become vacuous at large n: as intervals tighten, an arm carrying NO
  information starts to "separate" too.** MEASURED in the stream's **own pre-registered primary**: at
  n = 600 a deliberately information-free arm separates at **+9.1 %**. Every real arm still passed —
  which is exactly why nobody would have looked.
  ⇒ **More data does not automatically make a predicate safer; it can dissolve it.** The claim was
  rescued by replacing "does it separate?" with a **direct contrast**: history vs
  noise-in-the-same-columns **−0.0504 [−0.0519, −0.0490]**, against a tight null for
  dropped-vs-fake-history (**−0.0001 [−0.0006, +0.0004]**) — establishing that **64 % of the recovery
  is speed history** and the lead block is **7.9× smaller**. **Re-run your negative control at the new
  n whenever n changes materially. A control validated at n = 40 is not validated at n = 600.**
- **C32 — AN ABLATION CREDITED TO THE WRONG COLUMN** *(new class, added 2026-07-28)* ⇒ **an ablation
  tells you which COLUMN SET carries an effect, never WHY — and when one column is silently defective,
  a neighbouring block absorbs its credit and looks like the mechanism.** MEASURED: E-GOAL-2 credited
  **64 %** of the goal-head recovery to a "speed history" block (`dv_0p5`, `dv_1p0`, `v_lag_*`). It is
  worth **0.9 of 46.3 recovery points — 2.0 %.** The real lever is **one 0.1 s speed difference**:
  `v` alone is **−19.4 %, separated-WORSE**, while **`v + ax_fd` is +46.3 %, a tight null against the
  full ten-column head (+0.0002 [−0.0023, +0.0027]).**
  ⭐ **Root cause, and it was replicated on the ORIGINAL stream's own corpus with its own fitter, folds
  and seed** (both its anchors reproduce exactly): **`egomotion`'s NATIVE `ax` is a poor derivative of
  the speed the target integrates — correlation 0.759.** `v + ax_fd` (0.1 s backward difference)
  reaches **0.9270 m**, a null against the whole block at 0.9305, while native `v + ax` reaches
  **1.1808 m — 0.2539 m worse for one column choice.** **The lag block was a PROXY for a derivative the
  native channel failed to supply.**
  ⇒ **Two streams that appear to disagree about a mechanism may both be right about the statistics and
  wrong about the cause. Before crediting a block, check whether a single column in it is defective —
  and check the channel against a derivative you compute yourself.** ⚠️ **Consequence had it stood: v5
  would have shipped a 1-second history buffer to buy 2 % of the effect.** Fixed at source
  (`lead_state_gate.py` now emits `ax_fd`; **`ax` deliberately NOT redefined**, because committed
  artifacts carry that name).
- **C33 — A RESAMPLED RESIDUAL UNDER-STATES A TRAINED HEAD** *(new class, added 2026-07-28; unusually,
  a class about being too PESSIMISTIC)* ⇒ **estimating a learned component by resampling a residual
  assumes its errors are UNCORRELATED with the windows that matter. A real head's errors are
  correlated — and here that correlation was worth points, not a penalty.** MEASURED: the resampled
  estimate was **+25.4 %**; the trained head delivered **+46.3 % (OOF) / +50.7 % (deployable)** —
  **1.82×**, exceeding its pre-registered bar by 2.4×. Decomposed at matched RMS: **+3.9 points** come
  from the head being *correlated* rather than decorrelated, and **+17.0 points** from it simply being
  more accurate (0.7449 vs 0.9305 m).
  ⇒ **A resampled estimate is a LOWER bound on a trained component as often as an upper one, and which
  it is cannot be assumed — state the direction as unknown and go and train the thing.** Sibling to
  **C22** (bound quoted as capability) with the sign reversed: there a bound over-stated a capability;
  here a proxy under-stated one, and **declining to license the +25.4 % is what preserved the finding.**
- **C34 — A LEVER MEASURED AGAINST THE WRONG COUNTERFACTUAL** *(new class, added 2026-07-28)* ⇒ **"how
  much does X recover" is meaningless until you name what X is being added TO, and the natural
  comparator — the deployed arm — is usually the wrong one, because it differs from the treatment by
  MORE than X.** MEASURED: the goal input's recovery was reported as **+46.3 %** against the
  **as-trained** selector. But **a trained selector with NO goal already recovers +35.62 %** — so most
  of that headline was *capacity*, not *goal*. **The goal's capacity-matched marginal is +26.31 points,
  an over-credit of 1.76×.**
  ⭐ **And the capacity-matched contrast turned out to be the BACKGROUND-INVARIANT quantity**: it is
  identical on two backgrounds (**−0.0811 [−0.0904, −0.0720]** and **[−0.0888, −0.0732]**) whose totals
  differ by **15 recovery points**. That is a partial answer to **C30** — *the marginal transports where
  the total does not.*
  ⇒ **Match capacity before attributing an effect to information. Report the marginal, not the total.**
- **C35 — A REQUIREMENT CURVE IS A PROPERTY OF THE CONSUMER, NOT OF THE SIGNAL** *(new class, added
  2026-07-28)* ⇒ **"the supplier must reach σ₀" is only true for the consumer the curve was measured
  on; a different consumer can have no such threshold at all.** MEASURED on an identical degradation
  ladder: the **fixed rule** goes destructive at **1.128 m** and reaches **−111.78 % at 2.256 m**, while
  the **trained selector** is **+16.73 %, separated-BETTER at that same 2.256 m and never crosses
  zero.** ⇒ E-GOAL-3's σ₀ and its "gate the goal channel on measured accuracy" recommendation are
  **re-scoped to the fixed rule** and must not be quoted as properties of the goal.
  ⇒ **Before writing a spec for a supplier, ask which consumer the threshold was measured against —
  and whether that is the consumer you will ship.**
- **C36 — AN INPUT CAN BE WORTH POINTS WHILE CARRYING NO INFORMATION** *(new class, added 2026-07-28;
  the most counter-intuitive of the set)* ⇒ **a derived feature that is an exact function of columns the
  model already has adds ZERO information and can still be worth a large, separated gain — as an
  INDUCTIVE BIAS.** MEASURED: `g_along` = GBM(`v`, `ax_fd`) at **R² = 0.999894**, and the no-goal arm is
  **fed both `v` and `ax_fd`.** The goal is therefore informationally empty — and worth **+26.3
  separated recovery points.**
  ⚠️ **The decision this changes: funding a strategic SUPPLIER (AlpaSim, an external mapped corpus, any
  goal-signal acquisition) is the wrong lever at this feature list** — a supplier buys the accuracy
  term, which end-to-end is worth **+2–4 points**, not the +26. Corroborated independently: a naive
  **`2·v0`** goal delivers **+62.07 %** through the trained selector where the *fixed rule* turns the
  same goal into **−18.55 %, separated-WORSE** — **a 16.7×–33.6× collapse in the value of accuracy.**
  ⇒ **Before buying a signal, test whether a crude version of it captures the gain. If it does, you are
  buying structure, not information, and you can build structure yourself.**
- **C37 — A GATED RESULT RELAYED AS ITS UNGATED SIBLING** *(new class, added 2026-07-28; the relay was
  MINE, and it happened inside a brief whose own trap list warns about this class)* ⇒ **when a document
  publishes a rule and several fitted variants of it, the strongest number on the page is usually a
  FITTED one — and relaying it under the plain rule's name ships the fitter.** RETRACTED: *"C2 ungated:
  0.8563 → 0.5196, −0.3366 [−0.4507, −0.2310], a 39 % cut."* **`−0.3366 / 0.5196–0.5221` is
  `learned_gate_ALL_ridge_tau0` — a fitted ridge gate firing on 66.97 % of windows over a 73-feature
  bank INCLUDING the 2-WM ensemble family**, i.e. exactly the gate the same stream showed is
  **dominated by its own prerequisite**.
  ✅ **The true ungated value: 0.8563 → 0.5645, −0.2918 [−0.4233, −0.1598], separated,
  `selected_frac` 1.000** — a **34.1 %** cut, not 39 %. My figure overstated it by **0.0448 m
  (1.154×)**, and I gave it to the PI twice.
  ⚠️ **The source document was INTERNALLY CORRECT** — §1.2 publishes −0.2918 and §5.2 recommends it.
  **The defect was entirely in the relay**, which is what makes this distinct from C4: nothing was
  wrong upstream, and re-verifying the chain would have found nothing wrong *with the chain*.
  ⭐ **What caught it: the implementing agent was told to reproduce −0.3366 and REFUSED to call its
  −0.2918 a failure** — it re-derived the cost matrix from raw geometry (881/881 identical picks) and
  reported the label as the defect. *"Had I reproduced −0.3366 I would have shipped a gate."*
  ⇒ **When quoting a headline out of a multi-arm document, quote the ARM NAME and its `selected_frac`,
  not just the number.** A `selected_frac` below 1.000 means you are looking at a gate.
- **C38 — A RIG-CORRELATED SIGNAL THAT SURVIVES THE GEOMETRY FIX, BECAUSE IT WAS NEVER GEOMETRY**
  *(new class, added 2026-07-27; found while CLOSING C26, and it is the half of C26 nobody had
  measured)* ⇒ **when a confound is removed by construction, verify the residual ON REAL PIXELS —
  the mechanism you fixed may not be the only one producing the signal you were worried about.**
  ✅ **C26 IS CLOSED for its own mechanism.** The clean field `176x624` (a centred slice of the built
  `256x640`/120°/cylindrical frame) has a replicate-pad/mask fraction of **0.0000000000 on rig A and
  0.0000000000 on rig B — maximum, not mean, over 240 real clips (120/120)**, against
  0.0000159 / 0.0900333 at the frame the corpus is built at.
  ⛔ **AND THE RIG IS STILL READABLE FROM THE PIXELS.** Real all-zero fraction inside the clean slice:
  **0.0000834 (A) vs 0.0079316 (B) — 95×**. ~97 % of it is **TRANSIENT** (night, tunnels), i.e.
  genuine image content: **the two rigs recorded systematically different imaging conditions.**
  **No choice of frame removes it** — it is a corpus-BALANCE confound wearing a preprocessing
  confound's clothes. ~3 % is persistent black beyond the rectangle-based mask (the lens image
  circle, θ 56–61°): **0.0000000 on rig A, 0.0002711 on rig B**.
  ⚠️ **And the sample size decided the answer.** At **n = 24** the same diagnostic said `160x592`
  would be free of persistent black; at **n = 240** the union covers **5.57 %** of the rig-B frame and
  **no useful centred frame is free of it**. The small sample fitted scene content and would have
  shipped a needless 3.5 pp of field loss. *"You cannot crop your way out of night scenes."*
  ⇒ **Two rules. (a) A fix verified only by the instrument that defined the defect is unverified —
  the ray map said 0.0000 while the pixels said 95×. (b) A residual measured on ≤ tens of clips is a
  direction, not a rate.** Sibling to **C26** (the defect it closes), to **C13** (a guard that cannot
  fail), and to **C5** (a scalar off too little data).
- **C39 — A FIELD REQUEST THE SENSOR NEVER SATISFIED, MASKED INTO INVISIBILITY** *(new class, added
  2026-07-27)* ⇒ **an "achieved == requested" geometry check can pass on the FRAME while failing on
  the CLIPS, because the check probes one clip and the shortfall is per-clip.** MEASURED, n = 3,000:
  the v5 wide build requests **120.000°** and **260 clips (8.67 %) cannot deliver it horizontally**
  (238 rig B + **22 rig A**; pooled min **118.958°**). The build's own `_geometry.json` records
  **`"observed_frac": 1.0`** — true of the single clip it probed, false of the corpus (**0.911**).
  ⚠️ **The failure is at the frame's vertical CENTRE row**, not at the corners where one would look:
  as `|v|` grows, `ρ` grows faster than `r(θ)`, so the horizontal excursion is *largest* on the
  centre line. A "rows only" fix therefore CANNOT reach zero at 640 columns — at every height tested
  the residual stayed non-zero on **both** rigs.
  ⇒ **A geometry declaration must be computed over a POPULATION and reported PER STRATUM, and the
  builder must be able to abort on it** (`v2_compressed.py --require-fully-observed`, off by default).
  Sibling to **C2** (absence from a single probe) — same shape, applied to a *property* instead of an
  *existence*.
- **C40 — A DRIVER THAT MISLABELS ITS OWN FAILURE, AND A LOG THAT ERASES THE PROOF** *(new class, added
  2026-07-27)* ⇒ **a run can report success while its terminal condition was an external refusal, if the
  driver's exhaustion message covers both cases — and a `>` redirect can then destroy the only evidence
  that would have distinguished them.** MEASURED: the 2026-07-26 YouTube harvest ended with **650 of 650
  videos refused** (`Sign in to confirm you're not a bot`, **0 clips**) beginning **16:11:21 UTC**. The
  driver logged **`pool exhausted at 343 — proceeding`**. The report was finalised at **14:35 UTC** — before
  the block — and states *"Was it blocked? — NO."*
  ⇒ **It REFUTES that report's conclusion that "the binding constraint was not rate-limiting."** It was.
  ⚠️ **And the evidence nearly did not survive:** `run_scaleup_parallel.sh` opened `w*/harvest.log` with
  **`>`** each round, destroying rounds 5–8, so the block's onset was unrecoverable; only the archived
  round-9 logs proved it, and any re-run would have overwritten those too. **A one-character bug erased
  the record of the thing that actually stopped the harvest.** Fixed to `>>`.
  ⇒ **Two standing consequences.** (1) **A driver's "done" is not a verdict** — an exhaustion message must
  distinguish *supply exhausted* from *supply refused*, and a report written before a run ends cannot
  answer "was it blocked?". (2) **Append, never truncate, any log that is the sole record of an external
  interaction.** Sibling to C13 (a guard that cannot fail): here the *instrument* could not report the
  distinction it was being read for.
- **C41 — A REGISTRY ROW THAT RECORDED AN INTENT AND WAS NEVER ADVANCED TO AN OUTCOME** *(new class,
  added 2026-07-27)* ⇒ **the same row served as the pre-registration AND as the status, so the launch
  had nowhere to be written — and the ONLY quotable source for model facts went on asserting the
  arm did not exist while every experiment of the week measured on it.**
  **Retracted claim:** `MODEL_REGISTRY.md` §1.5.5 — *"`flagship-v4-fromscratch` — ✅ **READY, not
  launched** … CODE STAGED + VALIDATED, NOT LAUNCHED … **Zero GPU-day committed**"*, with Cost
  *"~53 h ESTIMATED"*.
  **MEASURED, re-read from the run's own artifacts on pod2 (2026-07-27), not from prose:**
  `metrics.json` **`final_step 29999`**; `supervisor.log` **launched 2026-07-23T21:54:44Z**, pid 108011,
  **restarts 0**, **`trainer exited rc=0`** 2026-07-26T09:01:37Z; `wallclock_s` **212 544.6 = 59.04 h**
  on pod2/A40; `config.json` `from_scratch true`, `trunk.ckpt null`, parity key
  `physicalai-train-e438721ae894` / `f09e44db`; `ckpt.pt` + four milestone checkpoints on disk;
  `Sayood/flagship-v4-fromscratch` on HF since 2026-07-26T09:12:06Z.
  **What it cost / would have cost:** the row was false for **4 days**, and not on a dormant arm — it is
  **the substrate of every selection experiment of the week** (Bar A, T3, E-V5-1, the fan work,
  E-GOAL-1→4; the `0.4907` re-scoring ceiling is measured on *this* checkpoint at 29 999). Anyone
  auditing those results was told by the authoritative source that their substrate had never run, and
  the "zero GPU-day committed" line understated real spend by **≈2.5 GPU-days**.
  ⚠️ **The failure is not that nobody noticed — four program reports, `LOOP_STATE.md` and two
  `RETRACTION_LOG` entries all recorded the run correctly.** Every *fast* surface was right and the
  *authoritative* one was wrong, which is the inverse of C1 and is why no reader caught it: the rule
  says trust the registry over the prose.
  ⇒ **Three standing consequences.**
  (1) **A pre-registration row and a status row must not be the same row.** A planned arm carries
  `Status: PLANNED` plus a separate `Outcome:` field that starts empty — an empty field is visibly
  unanswered; a stale sentence is not.
  (2) **THE CHECK THAT WOULD HAVE CAUGHT IT, and it is mechanical:** every `Location:` path in the
  registry is a real directory. **Any row whose status says NOT LAUNCHED while its run directory
  contains a `metrics.json` with a `final_step` is a contradiction a script can find** — the same
  nightly sweep as `stack/scripts/pod_git_drift.py`, pointed at experiment dirs instead of code.
  Equivalently, at the *repo* end: a committed eval JSON naming `ckpt_step 29999` for an arm the
  registry calls unlaunched is the same contradiction, findable without touching a pod.
  (3) **Registry refresh is already in `AGENT_OPERATING_STANDARD.md` ("whenever a model version is
  created, retired, or re-measured") — the gap is that LAUNCH and COMPLETION are neither.** Add both:
  a row is refreshed on *launch*, on *completion*, and on *first decision-grade eval*.
  Sibling to **C4** (inherited without re-verification) inverted — here the primary source was the
  stale one — and to **C13** (a guard that cannot fail): a row that can only ever be written once
  cannot report a state change.
- **C42 — AN ANCHOR THAT DOES NOT REPRODUCE ON A SECOND SUBSTRATE** *(new class, added 2026-07-27)* ⇒
  **a repair measured on one corpus slice can move a headline statistic there and NOT move it on the
  arm's own held-out data — and the honesty conditions can reproduce in SHAPE while the headline does
  not reproduce in MAGNITUDE, which makes the failure look like agreement.** MEASURED: the comma
  heading repair's anchor is **yaw R² +0.0114 → +0.3308** (n = 2992). Re-scored on `idm_head_v1`'s
  **own** held-out comma clips (30 clips / 4,140 windows, `episode_id`-disjoint):
  **R² +0.000048 → −0.000421.** ⇒ **the R² lift does not happen. Pasting +0.3308 into the card would
  have published +0.33 where the measurement is ≈ 0.**
  ⚠️ **What makes this treacherous: MAE fell 33.3 % (separated), medAE barely moved, normalised medAE
  got worse, and ρ stayed flat — every honesty condition reproduced exactly as briefed.** An analyst
  checking the caveats would have found them all confirmed and concluded the anchor transferred.
  **Measured cause** (one hypothesis tested and refuted first): **one wholly-stationary clip** — 300
  frames, **zero** observable frames, `v_max` 0.039 m/s. The repair deliberately leaves such a segment
  alone, so its 138 windows keep **84 impossible labels up to 15.28 rad/s**, which pin R². **Raising
  `v_min` cannot help** — 84–85 survive at 1.0 / 2.0 / 4.0 m/s.
  ⇒ **Two standing consequences.** (1) **An anchor is a property of its substrate until re-measured on
  the one you are about to quote it on** — the honesty conditions transferring is not evidence that the
  headline did. (2) ⭐ **A REPAIR AND AN ADMISSIBILITY DECISION ARE DIFFERENT THINGS.**
  `hold_heading_through_standstill` returns an `observable` mask that **no caller uses**; applying it
  removes *every* impossible label and collapses the label's own std **0.938 → 0.046 rad/s**. Repairing
  a label does not make an unobservable segment admissible, and conflating the two is how a stationary
  clip ends up governing a rotation statistic.
  **⤷ FORWARD POINTER 2026-07-27 (`anchor-settlement`; this row is NOT rewritten):** consequence (2)
  is now **implemented** — `comma2k19.yaw_rate_from_heading` derives the label *with* its
  admissibility and its DEFAULT (`"nan"`) makes a silent number impossible; `"keep"` needs a flag
  **and a written reason**; 19 tests. And consequence (1) turned out to be **stronger than stated**:
  the anchor did not merely fail to transfer — **it was measured partly IN-TRAIN.** See **C43**.
- **C43 — A HEADLINE MEASURED PARTLY IN-TRAIN, WITH ZERO NAME-LEVEL EVIDENCE OF IT**
  *(new class, added 2026-07-27)* ⇒ **two caches of the same corpus, built by different samplers with
  different segment counts, can share episodes BY CONTENT while no filename, tag index or id is
  comparable — and a train/eval overlap of 2 of 22 episodes can carry an entire published headline.**
  MEASURED: the comma heading-repair anchor **+0.0114 → +0.3308** (deployed head, n = 2 992, 22
  episodes on `comma2k19-val-76b6e94a97a1`) was compared to `idm_head_v1`'s own 40 comma TRAINING
  clips on `comma2k19-val-61c46fca8f7f` by **sha256 of the raw `poses` float32 bytes AND of the raw
  `frames_u8` sensor bytes**. **2 of the 22 are bit-identical** to 2 of the 40. Remove them and the
  same head, same predictions, same protocol reads comma yaw **R² −0.746 (CI [−1.574, −0.177])**.
  ⛔ **`+0.3308` is WITHDRAWN — not reduced, not "partially valid".**
  ⚠️ **Three things make this the instructive case, not just an error:**
  (a) **the honesty conditions were useless as a detector** — MAE and Spearman ρ move essentially
  identically on the contaminated and the clean subset, so checking the caveats confirmed nothing;
  (b) **the interval already said so and nobody read it** — the published CI was **[−1.2982,
  +0.7047]** and the OFF→ON contrast measures **+0.3194, CI [−1.262, +0.6425], NOT separated.** The
  disqualification was lifted on a point estimate whose interval spanned zero *before* the leak was
  known;
  (c) **the mechanism is composition, not memorisation alone** — the 2 leaked clips carry **4× the
  yaw variance** of the other 20 (`gt_std` 0.108 vs 0.025) and R² is variance-weighted, so 9.1 % of
  the windows moved the headline by **1.08**. The same fragility hits arms with **no leak at all**:
  every one of the 18 persisted v3 arms drops **0.36–0.58** R² on the clean 20, `R0` (= the shipped
  `V3F`'s rotation head) from **+0.6791** to **+0.3038 [+0.054, +0.479]** — still separated, so
  `+0.679` is **not** withdrawn, it is *half the size and conditional on 2 of 22 episodes*.
  ⇒ **Three standing rules.** (1) **Verify train/eval disjointness BY CONTENT — hash the raw bytes —
  before quoting any number that crosses two caches.** Names have been 600/600 wrong here before.
  (2) **A pre-registration that treats "the claim" as one object mis-handles a claim resting on two
  numbers with different provenances** — withdrawing `+0.679` alongside `+0.3308` would have been an
  error in the other direction. Separate them by provenance first. (3) **Report R² with the
  variance composition of its episodes**, or a leave-k-out; a variance-weighted statistic over
  heterogeneous episodes is not a property of the model alone.
  Record: `…/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`; raw
  `anchor_overlap.json` / `anchor_resettlement.json` / `arms_resettlement.json`. Sibling to **C42**
  (an anchor is a property of its substrate) and to **C5** (a variance-weighted metric dominated by
  a handful of rows).
- **C43 — A CONTAMINATED ANCHOR, AND THE HONESTY CONDITIONS COULD NOT DETECT IT (twice)** *(new class,
  added 2026-07-28; the completion of C42)* ⇒ **RETRACTED: "the repair lifts comma's yaw
  disqualification — deployed +0.3308."** Settled **by content** (sha256 of raw `poses` float32 bytes
  **and** raw `frames_u8` sensor bytes, both hosts, 6 hash families agreeing, 11 cross-cache matches):
  **2 of the anchor's 22 comma evaluation episodes are BIT-IDENTICAL to episodes the model TRAINED on.**

  | subset | legacy | repaired |
  |---|---:|---:|
  | `cm_ALL22` (the anchor) | +0.011430 | **+0.330822** |
  | **`cm_CLEAN20`** | −0.001230 | **−0.745999 [−1.574, −0.177]** |
  | `cm_INTRAIN2` | +0.856185 | +0.856185 |

  **The 2 leaked clips carry 61 % of the split's sum-of-squares from 9.1 % of its windows**, and read
  identically under all three protocols — the repair does nothing to them, they are simply memorised.
  ⚠️ **And the anchor was never separated in the first place: the published CI [−1.2982, +0.7047]
  spans zero, and the OFF→ON contrast is +0.3194 [−1.262, +0.6425], NOT SEPARATED.** It was quoted as
  a point estimate — by me, to the PI.
  ⛔⛔ **THE HONESTY CONDITIONS FAILED AS A DETECTOR FOR THE SECOND TIME.** MAE, medAE and ρ behave
  **identically on the contaminated and the clean subsets**. In C42 their reproducing wrongly implied
  the headline transferred; here their reproducing wrongly implied the split was sound. ⇒ **Caveats
  are not a leak test and are not a transfer test. Only content hashing is.**
  ⭐ **And the symmetric error was avoided:** the claim rested on **two** numbers with **different**
  provenances. `+0.679` is `R0`, the shipped rotation head, trained on a **content-disjoint** split —
  **no leak, NOT withdrawn** — though it reads **+0.3038 [+0.054, +0.479] (separated)** on the clean
  20, and all 18 persisted v3 arms lose 0.36–0.58 the same way while ρ barely moves. **Withdrawing it
  too would have been an error in the other direction.**
  ⇒ **Standing consequence: an "episode-disjoint" claim resting on `episode_id` is not evidence.**
  A 78 % leak (62/79) has already been measured once on a sibling val cache, and
  `physicalai-val-0c5f7dac3b11 × physicalai-train-e438721ae894` has **never been checked by content**.
- **C44 — A DIFFERENT ESTIMATOR ON THE IMPORT PATH** *(new class, added 2026-07-28)* ⇒ **a stale tree
  can supply not just old code but a DIFFERENT STATISTICAL ESTIMATOR, and the guard built to catch
  stale trees reports `ok: true` because it probes capability, not identity.** MEASURED:
  `/root/taniteval/taniteval/ci.py` is **`ef925f06…`** where HEAD's is **`c92618a0…`**, and
  `idm2_lib.py:19` / `idm3_a0.py` insert that path **unconditionally**. ⇒ **every published v3 interval
  came through it.** ⇒ **Assert the md5 of the `ci.py` actually loaded, not the one you intended to
  load** — and treat "the guard says ok" as a statement about capability only.
- **C45 — THE SECOND ONE-SIDED CLAMP: THE CLOSED-LOOP PRIMARY REWARDS LATERAL DEGRADATION** *(new
  class, added 2026-07-28)* ⇒ **a bounded term that FLOORS on most rows cannot be charged for further
  harm, so an injection that helps a minority RAISES the mean — and the metric pays for the failure it
  exists to catch.** MEASURED on `cv_holdv0`: a **2 m constant lateral offset** moves
  `PSS@twosided_v2` **+0.0581 [+0.0473, +0.0691] SEPARATED**; a **5° heading error +0.0747**; and a
  **ZERO-MEAN jitter (σ = 1 m), which cannot re-centre a bias, +0.0303 SEPARATED**. **8 of 8 injections
  separated in the WRONG DIRECTION, on both arms tested.**
  ⚠️ **Scale: the entire published gap between `cv_holdv0` and the best learned arm is −0.0090, so 2 m
  of injected lateral error is worth 6.4× the headline gap, BACKWARDS.**
  **Mechanism:** `recovery = clamp(1 − xt_end/xt_hold, 0, 1)` is floored on **55.65 % (`cv_holdv0`) to
  92.19 % (`refc_xl_produced`)** of defined rows, and the **median unclamped ratio exceeds 1.0 for
  EVERY arm**.
  ⇒ ⛔ **This is the over-travel blindness repeated in the OTHER weight-5.0 term — the one that was
  never audited.** Both halves of the composite were one-sidedly clamped; fixing `ego_progress` last
  night left the twin live. **Any v5 gate rendered before `recovery` is fixed is gated on a metric that
  rewards the failure mode it was built to detect.**
  ⇒ **Standing consequence: for every bounded term, report the FLOOR/CEILING FRACTION beside the
  score.** `discriminative_range` already **computed** `floor_frac` and **never used it** — a gate
  testing one end of a two-ended quantity. A term saturating on the majority of rows is not a metric;
  it is a constant with noise.
- **C46 — A COMFORT TERM THAT REWARDS NOT DRIVING** *(new class, added 2026-07-28)* ⇒ **a bound
  calibrated from literature rather than from the corpus can fail the ground truth itself, at which
  point it scores restraint rather than quality.** MEASURED: **the human's own logged path fails the
  comfort bounds on 16.60 %** of the same windows, while **`cv_holdv0` and `stand_still` — the two arms
  that do the LEAST — both score a perfect 1.0000** and **every learned planner floors** on the jerk
  clause. ⇒ weight set to **0.0**, measurement retained as a diagnostic. ✅ **Provable no-op: 16
  published `@clamp_v1` composites reproduce at max|diff| = 0.000000.**
  ⇒ **Validate any threshold against the ground truth before it carries weight.** If the humans fail
  it, it is not measuring what you named it.
- **C47 — "NEVER SATURATES" IS NOT THE FIX; A SOFT FLOOR IS STILL A FLOOR** *(new class, added
  2026-07-28)* ⇒ **replacing a hard clamp with a smooth unbounded form can score WORSE than the defect
  it replaces, because what matters is not whether the term saturates but whether its CHARGE RATE
  COLLAPSES WHERE THE DATA ACTUALLY LIVES.** MEASURED while fixing `recovery`: the intuitive
  unsaturating share form `xt_hold/(xt_hold + xt_end)` **floors on 0.0000 of rows — and scored 0/8 on
  the acceptance test.** Its charge rate decays like **r⁻²** while the **median row sits at r = 1.18**,
  so it **rewards a near-perfect row 4.3× harder than it charges a typical one.**
  ⚠️ **And the parameter that fixed the sibling term could not be ported.** An even split (`q = 0.5`) —
  *verbatim the argument that fixed `ego_progress`* — scored **7/8**, because **`recovery`'s ratio tail
  is much heavier**. ⇒ **A fix that worked on one bounded term is a hypothesis about the next one, not
  a solution.**
  ⇒ **Standing consequence: choose the shape against the DENSITY OF THE DATA, not against the
  algebra** — and prove it with an injection suite the candidate can fail. Two of three candidates here
  were refuted by their own acceptance test, which is the only reason the third is trustworthy.
  ⚠️ **AMENDED 2026-07-28 — C47 IS BOUNDED, NOT GENERAL, and the bound was measured by a stream that PRE-REGISTERED THE OPPOSITE PREDICTION AND WATCHED IT FAIL.** The `share` form — reward bias 3.332, **0/8 on `recovery`** — scores **10/10 on `lat_heading`**. And **`q = 0.5`, the parameter that FAILED on `recovery`, is the one that WINS on `lat_heading`.** ⇒ **the reward-bias proxy only predicts the outcome when the median row sits ABOVE the anchor** (`recovery` 1.181, 75.4 % past; `lat_heading` 0.9103, 46.3 %). **The rule stands as “judge the charge rate where the data lives”; it does NOT stand as “the share form is bad” or “q = 0.5 is wrong.”** A class stated one level too general is a trap of its own.
  ⭐ **Also established, and shipped as a proof-test: a strict refinement was IMPOSSIBLE.** Any bounded
  `g` agreeing with `1 − r` on `[0, 1]` must be constant above 1 — **i.e. it IS the defect.** So the
  choice was necessarily a range-budget with a free parameter, and pretending otherwise would have
  hidden a judgement call inside an apparently mechanical fix.
- **C48 — THE PRESCRIBED CURE WAS THE DISEASE** *(new class, added 2026-07-28)* ⇒ **a remedy can name
  the very artifact that carries the defect, and survive review because everyone reads the sentence as
  an instruction rather than a claim.** MEASURED: `MODEL_REGISTRY` R8 §2.2, `taniteval/registry.py:85`,
  `Paper/TANITAD_PAPER.md` §7.2 and `DOC_CORRECTION_SWEEP.md` **all instruct the reader to
  "re-evaluate on the CLEAN `f1b378` val"** — a split now content-verified as **77.5 % leaked**. The
  remedy for a leak prescribed the leaked corpus. **R4 F6 flagged it on 2026-07-25 and it is still in
  the paper.**
  ⇒ **When a document names a specific artifact as the fix, verify the artifact, not the sentence.** An
  instruction inherits no evidence from the correctness of the problem statement around it.
- **C49 — A CONFOUND ASSUMED SYMMETRIC** *(new class, added 2026-07-28)* ⇒ **"the leak inflates both
  arms equally, so the ordering is conservative" is a claim about the leak's DISTRIBUTION, and it needs
  measuring on each arm separately.** RETRACTED: the registry's argument that Branch-B's *worse*
  ordering was safe under contamination. **flagship-v1 is the parity-trained control and PROVABLY saw
  those episodes; Branch B's overlap was never measured** — so a memorisation advantage for the control
  is a live explanation of the very gap being reported. Findings 1 and 2 survive; **Finding 3, the
  paired ΔR², is CONFOUNDED.**
  ⇒ **Symmetry of a confound is an empirical claim, not a default.** Measure the contamination per arm
  before arguing an ordering survives it.
- **C50 — "HELD-OUT" WITH 40 % LITERAL TRAIN-ON-TEST** *(new class, added 2026-07-28)* ⇒ MEASURED on
  `idm_head_v1`'s published card, labelled *"val metrics below are held-out"*: **32 of 40 clips /
  2,815 of 3,517 windows (80.0 %) are bit-identical to the frozen encoder's training corpus, and 16 of
  40 clips / 1,407 windows (40.0 %) are bit-identical to clips the SCORING HEAD ITSELF trained on.**
  Only 8 clips (20.0 %) are content-disjoint.
  ⭐ **The matched clean counterpart already existed in the repo** — same head md5, same encoder, same
  protocol, n = 3,521 vs 3,517: **ADE@2s 2.703 → 3.856 (+42.7 %)**, speed MAE **+43.1 %**, and
  **`long_accel` R² FLIPS SIGN, +0.0811 → −0.1847.** All six metrics move the same way.
  ⚠️ **And the gap had already been explained away**: `VALIDATION.md` attributed it to *"clip
  selection… ADE scales with speed"* — written without knowing **80 % of the comparison set was
  memorised.** ⇒ **A plausible mechanism for a discrepancy is not a substitute for checking whether the
  split is clean.** *(Honest bound: the two 40-clip sets share only 8 clips, so +42.7 % is an upper
  bound, not an isolate — reported by the auditor against its own case.)*
- **C51 — A GUARD THAT COULD ONLY KILL, NEVER REPORT** *(new class, added 2026-07-28)* ⇒ **a threshold
  with no reporting channel converts a diagnostic into an outage, and a guard whose message names a
  cause that is IMPOSSIBLE BY CONSTRUCTION sends the reader hunting a bug that cannot exist.**
  MEASURED: the PI's geometry validation lost **both** wide arms to `flagship_v4.py:233`
  (`seam_fail` hard-wired at 1.5) — `B_wide` pre-clamp **1.760**, `C_v5` **1.511**, both at ~step 350,
  both at `λ_plan 0.833`, ~**2.7 GPU-h** burned. Four separable defects:
  1. ⚠️ **THE MESSAGE IS FALSE BY CONSTRUCTION.** It reads *"the in-graph clamp is not holding, i.e. a
     code fault."* The clamp is `scale = seam_clamp / ratio.clamp_min(seam_clamp)` (`:241`), which
     **cannot fail to bound the ratio at `seam_clamp`**. The check is on the **PRE**-clamp ratio, so it
     fires on precisely the condition the clamp then handles correctly. There was no code fault.
  2. **IT IS NOT A DIVERGENCE GUARD.** On **matched steps** both wide arms sat *at or below* the 51.4°
     control on total, `wm`, `plan_ade` and `oracle_ade` — and **`C_v5` tripped it at the LOWEST total
     (9.834), `wm` (4.242) and `plan_ade` (1.509) of its entire run.** The trip is uncorrelated with
     training health. *(My own first reading — "the wide arms are diverging" — was RETRACTED: it came
     from comparing A@1499 against B@350. The matched-step table inverts it.)*
  3. **THE KILL CRITERION IS A BATCH MAX.** `ratio.max()` — **one sample of 64 sets it.** The robust
     population statistic (`seam_clamp_bound_frac`) is computed **two lines below** and was used for
     nothing.
  4. **THE INSTRUCTION WAS UNFOLLOWABLE.** The raise tells the operator a sweep *"must raise seam_fail
     explicitly and record that it did"* — and `seam_fail` was **exposed nowhere**: no CLI flag, no
     config key. The codebase demanded an action it provided no way to perform.
  ⭐ **AND THE GUARD'S ONLY OUTPUT CHANNEL WAS A FATAL EXCEPTION.** `_factor_grafts` computes five seam
  numbers every forward pass and `v4_loss_step` already merges them into `log` (`:205`) — the
  **row-writer tuple dropped every one**. MEASURED: **0 seam keys in all three arm logs**, so *"how
  close did the control get to 1.5?"* is **unanswerable from arm A's log**. The module's own NAMED TRAP
  comment (`:245-248`) says a λ read is invalid without `_preclamp_mean` and `_bound_frac` — neither
  was ever written.
  ⭐ **The suite had been telling us for months:** **six existing tests set `seam_fail` to
  100.0 / 1e6 / 1e9 / 1e12** to get their measurements done. A default that every test disables is a
  default that is wrong.
  ⇒ **FIXED, and the fix is shaped by the class:** `seam_fail` exposed (`--seam-fail`, default
  **unchanged at 1.5** so no existing arm moves, recorded in `config.json`) and the seam telemetry now
  **written every log step**. Re-run cost avoided: `seam_fail` appears **only in the raise**, never in
  a computed value, so moving it cannot alter any forward result — arm A completed without raising,
  therefore raising it is a **provable no-op for A**, the A-vs-B contrast stays matched, and **A did
  not need re-running (3 h 47 m of A40 saved)**. Pinned by
  `test_seam_fail_is_a_pure_guard_and_changes_no_computed_value`.
  ⚠️ **Not yet fixed, and deliberately left for the PI:** the guard still kills on a batch max rather
  than on a persistent population condition, and the false "code fault" wording still stands in the
  message. Changing a fail-loud's *semantics* is a scientific decision, not a cleanup.
- **C52 — A PUBLISHED REMEDY IMPORTED WITHOUT ITS PRECONDITION** *(new class, added 2026-07-28)* ⇒
  **citing a result correctly is not the same as establishing that it applies. A method's stated
  precondition is part of the claim, and it must be MEASURED on our setting before the method is
  trusted to decide anything.**
  RETRACTED: my own E1d hypothesis that WiSE-FT-style weight-space interpolation would find a point
  on the REF-C-base → CL-SFT segment carrying the closed-loop gain without the open-loop regression.
  The citation was accurate (Wortsman et al. 2022 — interpolating a fine-tune with its base
  dominates early stopping) and the reasoning from it was sound; **the precondition was never
  checked.** WiSE-FT's frontier holds *when the fine-tune stays in the base's basin*. MEASURED on our
  arms it does not: `dep_overall` is **separated-WORSE at five consecutive interior points**
  (α = 0.20 → +0.1107, 0.30 → +0.1387, 0.40 → +0.1492, 0.50 → +0.1199, 0.60 → +0.0759; every CI
  excludes zero, paired episode-cluster bootstrap, n = 43 clusters). ⇒ **the interpolation path
  crosses a region worse than BOTH endpoints — the two solutions are not linearly mode-connected for
  this metric.**
  ⭐ **The probe was still worth running, and that is the class rather than a softening of it:** it
  cost ~1 h of an idle pod, carried a bit-identical reproduction control (α=1.00 reproduces E1c row
  4000 exactly), and returned a **stronger** negative than "no good α was found" — plus the
  decomposition that actually redirects the program (`E1D_RESULT.md` §3: junction recovery is cheap
  and monotone, overall-corridor recovery is expensive and barrier-crossing).
  ⇒ **Before importing a method, write down what it assumes and price the probe that tests the
  assumption.** A pre-registered cheap probe turns an unchecked precondition into a measurement.
- **C53 — A WORKLOAD PRICED BY ITS NAME, NOT BY ITS STAGES** *(new class, added 2026-07-28)* ⇒
  **"download" named the cheapest stage of a pipeline whose expensive stage was CPU-bound, and the
  name is what I costed.** RETRACTED: my own judgement that running a small YouTube harvest on pod3
  "is network/disk, light" and therefore safe alongside a training job.
  MEASURED: the harvest ran at **477–483 % CPU (≈5 cores)**, load average **24.39**, and starved the
  trainer's four `pt_data_worker` processes. `e1c_clsft` step rate:
  **1.0 s/step measured over a clean 150-step window after the harvest was killed, versus a 5.64
  s/step cumulative average while it ran.** *(Honest bound: the 5.64 is a cumulative average that
  includes process startup, so the contention cost is **at least ~4×**; 5.6× is the upper reading.)*
  The expensive stage was never the network — it was **decode + face/plate/body Haar cascades over
  every FULL-RES frame**, which the pipeline's own docstring describes and I had read.
  ⚠️ **The aggravating detail: I quoted the rule while breaking it.** I wrote "pod3 is training
  E1e-A — never add GPU load to a training pod" in the same message, reasoned that a download is not
  GPU load, and stopped there — as if GPU were the only contended resource. **CPU and dataloader
  workers are contended resources too, and the rule's purpose is throughput, not a specific device.**
  ⭐ **AND KILLING IT CREATED A SECOND, WORSE PROBLEM:** the pipeline deletes the source mp4
  *after* decode, so terminating it mid-decode **left a raw 137 MB YouTube video on disk** —
  a privacy-protocol violation manufactured by the remedy. Found by probing for it explicitly
  (`find -name "*.mp4"`) rather than assuming the kill was clean; deleted, verified 0 media files
  remaining, directory down to 1.5 KB.
  ⇒ **Price a job by its STAGES on the contended resource, not by the verb in its name — and when
  you kill a pipeline mid-flight, audit what its deferred cleanup never got to run.**
  ⚠️ **CORRECTION 2026-07-28 (see C54): C53's measured claims all stand** — 477–483 % CPU, load 24.39,
  1.0 s/step clean vs 5.64 s/step cumulative under contention, and the raw 137 MB mp4 left on disk.
  **But its "0 clips were produced" implied the harvest had had time to produce some. It had not:**
  `ps` shows that run lived ~6 minutes and was still decoding its FIRST video. The contention finding
  is unaffected; the insinuation that the harvest was also failing to work is withdrawn.
- **C54 — ELAPSED TIME INFERRED AGAINST A CLOCK I NEVER RECORDED** *(new class, added 2026-07-28)* ⇒
  **a duration computed from "when I think I last looked" is not a measurement, and I produced four
  false claims from it in one session — every one contradicted by the first direct check.**
  MEASURED instances, all mine:
  1. *"E1e-A has slowed to 4.4 s/step"* — derived by differencing step 2225 against step 1775 using my
     own recollection of when I read the earlier value. A 240 s window measured **1.2 s/step**. There
     was no slowdown.
  2. *"1.0 s/step is the recovered rate"* — measured in a 150 s window immediately after killing a
     competing job, i.e. during a **dataloader catch-up burst**. Later windows put steady state at
     1.0–1.2; the burst was real but was not the steady state I called it.
  3. *"The harvest has run ~35 min and produced 0 clips"* — its own `ps` elapsed was **~6 min**.
  4. *"The second harvest has run ~27 min with 0 clips — that's a pattern, not bad luck"* — the pod
     clock said **310 s**. It had downloaded one 138 MB video and was decoding it. Normal progress.
  ⭐ **The tell is identical every time: I compared a log timestamp against an assumed "now".** The
  cure is mechanical and costs one extra field — **read the remote clock (`date -u`) and the process's
  own `etimes`/`lstart` in the same probe that reads the log**, and quote only differences between two
  values I actually observed.
  ⚠️ **Why this is worth a class rather than a shrug: two of the four became alarms** (a phantom
  slowdown, a phantom broken pipeline), and one of those triggered a kill that left a raw video on
  disk (C53). **An inference error that manufactures an incident is not a rounding error.**
  ⇒ **Never quote an elapsed time that is not the difference of two observed clock readings.**
- **C55 — A DECOMPOSITION MEASURED UNDER ONE MANIPULATION READ AS A PRESCRIPTION FOR ANOTHER**
  *(new class, added 2026-07-28)* ⇒ **an asymmetry observed while MOVING BETWEEN two trained models is
  not a statement about what to TRAIN ON, and treating it as one costs a full arm.**
  RETRACTED: the inference that launched E1f — that because E1d measured junction recovery as *cheap
  and monotone* and overall-corridor recovery as *expensive and barrier-crossing* under α-interpolation,
  the expensive half must be **dragging down** the cheap one, so restricting supervision to junctions
  would isolate the good part.
  MEASURED, it does the opposite. Best `dep_junction`: **E1c (full buffer, 3,537 records) −0.4982** at
  open-loop +0.2026, versus **E1f (junction-only, 733 records) −0.2108** at +0.0555. **Training on
  junctions alone HALVES junction recovery.** Had the overall half been interfering, removal should
  have left junction recovery at least equal; instead E1f delivers **42 % of the junction gain at 27 %
  of the open-loop cost, and zero overall-corridor gain** — more efficient per unit cost, strictly
  smaller in absolute terms. A scaled-down arm, not a targeted one.
  ⚠️ **The decomposition itself was not wrong** — E1d's α-frontier is reproduced and stands. What was
  wrong is the *transfer*: E1d characterised the geometry of a **path between two models trained on
  everything**, and that says nothing about the loss surface reached by training on a subset.
  ⭐ **The pre-registration is what limited the damage.** Outcome C was named in advance — "P2 improves
  while P1 degrades" — so the result was read as the pre-committed reading rather than argued into a
  win, and the ~2 GPU-h arm returned a clean refutation instead of an ambiguity.
  ⚠️ **Honest residual, stated not resolved:** the junction subset spans **102 episodes vs 362**, so
  this arm cannot separate "junction-only is the wrong target" from "733 records is too little". The
  refutation is of the *interference* hypothesis, not of junction supervision in general.
  ⇒ **Before turning a measured asymmetry into a training prescription, ask what manipulation produced
  it — interpolation, ablation, or training — and require the prescription to be tested under the
  manipulation it will actually be used in.**
- **C56 — AN OPERATIONAL IMPOSSIBILITY INFERRED FROM ONE BLOCKED *METHOD*, GENERALISED TO THE
  *CAPABILITY*** *(new class, added 2026-07-28)* ⇒ **"we cannot do X" earned by one failed technique
  is a statement about the technique, not about X — and when it hardens into a standing rule it
  silently taxes every future operation.**
  RETRACTED: **"pods cannot SSH each other"** — a `CLAUDE.md` trap, repeated verbatim in
  `E1B_RESULTS.md:431`, `VAL_CEILING_AND_S3_DECISION_GRADE.md:165`, and the R6 chief-scientist
  review, which forced every multi-GB move onto either the **~1 MB/s dev-box relay** or the HF
  fast-path.
  **MEASURED 2026-07-28: pod→pod direct SSH runs at 42 MB/s CROSS-DATACENTER** (US-TX-1 →
  ca-mtl-1): `flagship-v2corpus-30k/ckpt.pt`, **3,415,808,330 B in 77 s**, size-exact.
  **42× the relay**, and independent of HF.
  **Root cause:** the original blocker was that **copying a private key between pods is
  classifier-blocked** — which is correct and should stay blocked. From that single blocked method
  I concluded the *capability* was absent, and never probed the standard alternative: **generate a
  keypair ON the destination and authorise its PUBLIC key on the source.** No secret ever moves, so
  nothing is blocked. A second contributor: the RunPod **proxy** (`ssh.runpod.io`) really cannot
  transfer files (sftp → `subsystem request failed on channel 0`; `scp -O` → exit 2), and I had
  generalised that true limitation of the proxy into a false one about pods. The **direct** mapping
  (`$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`) was never tried.
  **Cost, measured, not estimated:** a 22 GB pod3→pod1 move at **1.38 MB/s (~2 h)**; a 66 GB move
  written off in-doc as *"18 h — unusable"*; and R6 lists this constraint as **simultaneously
  blocking the formal 8-metric gate, a REF-C arm, and checkpoint backup** while HF sat 403.
  ⚠️ **This is the SAME class as Operating-Standard rule #2** ("absence found at ONE location is not
  absence") — applied to a *capability* rather than a file. The rule existed; I did not think to
  apply it to an ops constraint, only to files and features.
  ⚠️ **Scope, stated honestly:** proven for **this pod pair**. It does not prove every pod exposes a
  direct port 22, and 42 MB/s is a *cross-DC* figure — same-DC should be faster, unmeasured.
  ⇒ **Before recording an ops impossibility as a standing rule, name the METHOD that failed and
  probe one alternative method. A capability claim needs a second probe exactly like an absence
  claim does.**
- **C57 — A DESCRIPTIVE GAP BETWEEN TWO UNPAIRED NUMBERS READ AS A TEST THAT THE SMALLER IS ZERO**
  *(new class, added 2026-07-28)* ⇒ **"large vs tiny" is a description; "tiny vs zero" is a
  hypothesis test, and only the second licenses the word ONLY.**
  RETRACTED: *"the goal producer damages ONLY the longitudinal axis"*, published in
  `V4_30K_GATE_RESULTS.md` §1.1 off the unpaired per-arm means (along +0.4260 vs cross +0.0273).
  **MEASURED, paired episode-cluster bootstrap** (B=2000, same 881 windows / 40 episodes,
  `driving.frenet()` at 2 s, self-checked to **d = 0.0000** against `driving.py`'s own output):
  `long_abs_2s_m` **+0.4260 [+0.3227, +0.5420]** and `lat_abs_2s_m` **+0.0274 [+0.0061, +0.0533]** —
  **BOTH SEPARATED.** The asymmetry is real and large (**15.5×**); the word **ONLY** is not.
  **Root cause:** I had two single-arm columns, differenced them by eye, and treated the small
  difference as indistinguishable from zero. Nothing in that procedure is a test — no interval was
  ever computed for the small quantity. The paired machinery the program already mandates was
  available and cheap (no GPU, it reads persisted windows), and I simply did not apply it to the
  axis I had already decided was uninteresting.
  ⚠️ **This is the same family as "never quote an interval without its estimator", one step
  earlier: here there was no interval at all, and its absence was silently read as a result.**
  ⭐ **The correction carries new information rather than just a hedge:** the paired test also shows
  **both SIGNED components overlap zero**, so the producer adds error *magnitude* without a
  *directional* bias — symmetric scatter, not early braking or a drift. That points at a **noisy**
  goal estimate rather than a **mis-calibrated** one, and those need different repairs.
  ⇒ **Before writing ONLY / no effect / unchanged about a measured quantity, compute its interval.
  If it was not worth an interval, it is not worth a claim.**
- **C58 — A STRUCTURAL ZERO READ AS AN EMPIRICAL ZERO** *(new class, added 2026-07-29)*
  ⇒ **a count of zero produced by a MASK is not evidence about the model. Before calling a zero a
  defect, check whether the training objective could ever have produced a non-zero.**
  RETRACTED: *"`ROUTE_UNKNOWN` (class 3) occurs 154 times and is predicted **0** times"*, published
  in `V4_30K_GATE_RESULTS.md` §1.5 as a symptom of goal-head collapse.
  **It is the intended behaviour.** `v4_curriculum.py:40`: `IGNORE_INDEX = -100` is *"the masked
  target sentinel — the labeler writes this for every window whose LAT/LON/DIST/route mode is
  `unknown`; `F.cross_entropy` **skips it**"*; `goal_modes.py` declares
  `N_ROUTE_CLASSES = 4  # left/straight/right + the v2.1 UNKNOWN sentinel`, `ROUTE_DROPPED = 4`.
  **The head receives no gradient for UNKNOWN and structurally cannot emit it.**
  **Second-order consequence, worth as much as the retraction:** the harness's
  `route_exact_agreement` = **448/881** carries those masked rows **in its denominator**, so the
  headline accuracy and the majority baseline were both understated. On the judgeable classes
  (left/straight/right, n=**727**): accuracy **0.6162** vs majority **0.5420**, margin **+7.4 pts**
  (not +6.1). ⇒ **an ignore_index silently changes what a reported ratio means; check the
  denominator whenever a metric coexists with a mask.**
  ⭐ **The underlying finding SURVIVED and got sharper** — which is the test of whether a retraction
  was about framing or about substance. Per-class recall: straight **394/394 = 100 %**, left
  **49/212 = 23.1 %**, **right 5/121 = 4.1 %**. *"Near-blind to RIGHT turns specifically"* is both
  more accurate and more actionable than *"90 % straight"*, and it points the repair at turn recall
  / class imbalance rather than at the planner.
  ⚠️ **Neighbour of C57, distinct mechanism.** C57: a quantity whose interval was never computed.
  C58: a quantity that *could not have been non-zero*. Both end in an unearned claim about zero.
- **C59 — SEARCHING BY THE NAME I EXPECTED, THEN REPORTING ABSENCE** *(new class, added 2026-07-29;
  FOUR instances in a single day, which is why it is a class and not an incident)*
  ⇒ **a search that matches on a name I chose tests my naming convention, not the world. Absence is
  only reportable after searching by SUBJECT MATTER.**
  RETRACTED: *"the situation classification for MoE camera usage — nothing measured, DoA 15 %,
  design only."* Told to the PI in answer to a direct question. **FALSE.** A complete
  pre-registered study exists — `…/incoming/2026-07-26-situation-classifier/` — with scripts,
  artifacts, checkpoints, and adjudicated verdicts:
  **LANE CHANGE** (153 held-out clusters) and **INTERSECTION** (264) both **A−**: the image arm is
  above chance (lane change ΔAP **+0.01987 [+0.01141, +0.02901]**, AUROC 0.703; intersection ΔAP
  **+0.04894 [+0.03735, +0.06277]**, AUROC 0.769), **anticipation demonstrated** at median lead
  **1.4 s / 2.0 s** — but **vision adds NOTHING over ego state** (`head_ego` CV-AP **0.0697** beats
  every image arm; shuffled control 0.0166). ROUNDABOUT **UNPOWERED** (26 clusters).
  **Root cause:** I searched `h2-sensor-attention/` — the folder named after the *hypothesis* — and
  concluded absence. The work lived under `situation-classifier/`, named after the *artifact*.
  ⚠️ **THE SAME ERROR, FOUR TIMES ON 2026-07-29:**
  1. `find -iname "*flagship-v4*"` missed **`v4fs_ckpt.pt`** → I declared a completed 30 k arm LOST.
  2. `find -name "anchors*.pt"` missed **`flagship_v4_anchors_dense.pt`** → "no anchors on the fleet".
  3. `ASSET_INVENTORY.md` inventoried what I had touched → **seven workstreams omitted** (Part 2).
  4. This one.
  ⇒ **PROCEDURE, not a resolution to try harder.** Before writing that something does not exist:
  **(a)** grep the hub's `incoming/` tree for the SUBJECT (`situation`, `anchors`, `idm`), never the
  name you assumed; **(b)** list the directory rather than glob it; **(c)** state the search you ran
  next to the claim, so the reader can see what was actually tested. A claim of absence that does not
  name its search is not evidence.
  ⭐ **The correction carries more than the retraction:** the measured verdict is a NEGATIVE for the
  MoE camera plan — the situations are anticipatable, but **from ego dynamics, not from the camera**.
  A sensor-request policy conditioned on front-camera situation classification has no measured
  signal to stand on today. That changes what to build, and I nearly left it unsaid.
- **C60 — A BASELINE THAT IS OPTIMISTIC BY CONSTRUCTION, REPORTED AS IF IT WERE DEPLOYABLE**
  *(new class, added 2026-07-29 — raised by the PI, who was right)*
  ⇒ **on LOGGED HUMAN data, ego kinematics encode the human driver's already-executed reaction to a
  scene the human already perceived. Using them as a baseline input measures "can I detect that a
  human started reacting" — which is NOT the task an autonomous system faces, because the AV must
  PRODUCE that reaction.**
  RETRACTED: my statement that **"the front camera adds no value for situation classification"**,
  and the framing *"vision adds nothing over ego state"* as a program-steering conclusion.
  **What is actually measured stands:** `head_ego` CV-AP **0.0697** does beat every image arm
  (img+ego 0.0525, img-only 0.0376), lane change ΔAP vs ego **−0.04361 [−0.07252, −0.01914]** and
  intersection **−0.02742 [−0.04895, −0.00620]**, both separated. **The measurement is not in
  dispute; the INFERENCE from it was.**
  **Why the inference was wrong:** the ego arm's advantage is partly a proxy for the *human's*
  perception, baked into the speed trace by the driver's anticipation. An AV following its own
  policy has no such trace unless its own perception created it. ⇒ the ego baseline is **not
  available in the same form at deployment**, so beating it is the wrong bar for the camera.
  **The deployable-relevant result was measured and should be the headline instead:** the image arm
  is **above chance and separated** — lane change ΔAP **+0.01987 [+0.01141, +0.02901]** (AUROC
  0.703), intersection **+0.04894 [+0.03735, +0.06277]** (AUROC 0.769) — with anticipation at
  **1.4 s / 2.0 s** median lead. That claim needs no ego comparison at all.
  ⚠️ **Two gaps the PI's framing exposes, neither of which the study covers:**
  1. *"Behind a slower vehicle"* is **object-level**; the classifier input is a 2048-d
     `SpatialGridReadout` scene summary, not a detection. `obstacle.offline` (3D agent tracks on
     **97.44 %** of the corpus) is the untouched substrate for a lead-vehicle / closing-speed feature.
  2. The PI's own discipline is the fix and should be written into the next pre-registration:
     **future ego motion is legitimate for GENERATING and VALIDATING labels, and illegitimate as an
     INPUT.** The study honours this for labels (privileged geometry) but then admits ego kinematics
     as an input arm — which is exactly where the comparison tilts.
  ⇒ **Before reporting "X beats the camera", ask whether X EXISTS in the same form at inference. A
  baseline built from a human's behaviour is not a baseline an autonomous system can stand on.**

---

## C61 — 2026-07-29 — quoting a decay SHAPE from an instrument that cannot separate the two causes

**Root-cause class: REPORTING A MECHANISM WHEN THE MEASUREMENT ONLY SUPPORTS A MAGNITUDE.**
(Sibling of the learning-exponent rule in `CLAUDE.md`: an exponent without its discriminating
control is not admissible — here the missing control is not the fit window, it is the *baseline*.)

**RETRACTED.** On 2026-07-29 I reported the pure-imagination sweep (v1, K=4/8/12/16/20, n=881 each)
as showing that **"decay ACCELERATES — near-linear early, near-quadratic by 2 s"**, from a local
exponent rising **1.03 → 1.63 → 1.87 → 1.91** (global OLS 1.456). The *numbers* stand and are
MEASURED. **The interpretation does not.**

**WHY IT IS WRONG.** ADE-vs-horizon confounds two distinct hypotheses that the sweep cannot separate:
- **(a)** predicting 2 s ahead is intrinsically harder than 0.4 s ahead → the acceleration is
  **task difficulty**, and no architecture change is justified;
- **(b)** the rollout compounds its own error → **compounding is real**, and the indicated fix is
  rollout-recovery *training*, not a larger horizon.

Every number I reported is equally consistent with both. I named a mechanism ("the rollout degrades
faster and faster") when the instrument measures only an envelope.

**WHAT WAS MISSING.** A **teacher-forced arm at the same steps** — i.e. SkyJEPA's compounding ratio
`CR_k = e_k,rollout / e_k,teacher-forced` (arXiv 2606.23444, verified 3-0), with the per-step growth
term `ER_k = E[e_k − e_{k−1}]`. Without the teacher-forced denominator there is no way to tell
"the task got harder" from "we compounded".

**BLAST RADIUS.** Anywhere the phrase "decay accelerates" / "accelerating decay" was used to argue
that the 2 s horizon cap is a *model* limitation:
`…/incoming/2026-07-29-imagination-horizon/IMAGINATION_HORIZON.md` §7, commit `6a99f98`, and the
chat reports of 2026-07-29. **The ADE table itself needs no correction.**

**CORRECTION IN FORCE.** Quote the sweep as *"ADE grows super-linearly with horizon over 0.4–2.0 s;
whether that is task difficulty or rollout compounding is UNRESOLVED pending CR_k"*. ⛔ Do not use
the exponent rise to justify an architecture change until E-CR reports.

**FIX, PRE-REGISTERED, ~0–6 GPU-h**: add a teacher-forced arm to `taniteval/taniteval/imagination.py`;
re-score the SAME 40 val episodes both ways; report CR_k/ER_k at k=4/8/16/20 with the **paired
episode-cluster bootstrap**. CR_k flat near 1 ⇒ (a), narrative FALSIFIED, no architecture change.
CR_k rising super-linearly ⇒ (b), rollout-recovery training indicated.

**SECOND-ORDER LESSON.** The confound was found by a literature survey, not by our own review — the
same class as C57–C60. A control that a published instrument treats as mandatory is worth probing
for **before** a result is written up, not after.

---

## C62 — 2026-07-29 — probing the fleet I REMEMBERED instead of the fleet the INSTRUCTION NAMED

**Root-cause class: SUBSTITUTING WORKING MEMORY FOR THE STATED ENUMERATION.**
(Distinct from C59's "searched by one name". Here the correct name was *supplied to me, verbatim,
every iteration* — and I never used it.)

**RETRACTED.** Across every fleet report on 2026-07-29 I stated the fleet as **three working GPU pods
+ pod1 blocked**. The PI corrected it: **there are four.** `tanitad-eval` is a **fourth A40**, and it
was **idle the entire session** — 0 % GPU, 0 MiB, zero python processes.

**HOW IT HAPPENED.** The AUTONOMOUS DRUMBEAT prompt says, in its own step (1), *"check the fleet
(**pod2/pod1/pod3/eval**)"*. `LOOP_STATE.md` likewise lists **four** corrected aliases plus the new
pod: `tanitad-pod` · `tanitad-pod2` · `tanitad-pod3` · **`tanitad-eval` 69.30.85.106:22073** ·
newpod `69.30.85.48:22192`. **I probed pod / pod2 / pod3 / newpod every iteration and `eval` zero
times** — I had built a mental fleet list from the pods I happened to be working on, and then
re-probed *that list* while believing I was checking the fleet.

**WHY IT SURVIVED SO LONG.** Every iteration my probe *succeeded* on 4 hosts and returned coherent
results, so there was no error to notice. **A complete-looking answer over an incomplete enumeration
is indistinguishable from a complete answer** — nothing in the output can reveal the missing row.
This is why the check must be against the WRITTEN list, not against whether the probe looked healthy.

**COST.** One A40 idle for the whole session, while I twice reported "pod3 idling is on me" and
"never idle" — the corrective attention went to the pod I could see. Also: **I wrote the LOOP_STATE
fleet table this session and it too listed only four rows.** A stale-fleet error copied *into* the
file whose job is to prevent stale state.

**CORRECTION IN FORCE.** The fleet is **FOUR A40s** (`newpod`, `pod2`, `pod3`, **`eval`**) **plus
`pod1` = 8× RTX A6000 blocked on missing `/dev/nvidia*`**. `tanitad-eval` was found **BARE** —
`/workspace` empty, no repo, no venv, no caches — but with system `torch 2.8.0+cu128`, CUDA
available, and 510 MB/s disk. Provisioning started 2026-07-29 01:09 UTC.

**RULE ADOPTED.** ⛔ **Enumerate the fleet from `~/.ssh/config` (or the drumbeat's own list) at the
START of each iteration and probe EVERY alias — never from memory of "the pods we are using".**
A pod that is doing nothing is exactly the pod least likely to be in working memory, and exactly the
one whose idleness costs most.

---

## C63 — 2026-07-29 — importing an instrument without checking OUR architecture meets its PRECONDITION

**Root-cause class: TRANSFERRING A PUBLISHED METRIC WITHOUT TESTING ITS ASSUMPTIONS ON OUR STACK.**
(Adjacent to C61, which this was supposed to fix. C61 was "reported a mechanism the measurement
could not support"; C63 is "built the fix on an assumption I never measured".)

**RETRACTED.** E-CR (`PREREG_deep_research_2026-07-29.md`) pre-registered SkyJEPA's
`CR_k = e_rollout / e_teacher-forced` as the control that resolves C61, and I implemented it on
DECODED DISPLACEMENT — the same surface as our published ADE. **CR on that surface is
MIS-SPECIFIED for this architecture, and the first values it produced (0.729 / 0.632 / 0.659 /
0.755) are an artifact.** They must not be cited as evidence about compounding.

**THE PRECONDITION CR NEEDS.** `CR_k` assumes the rollout and teacher-forced arms are
**exchangeable inputs to the same decoder**. Ours are not.

**MEASURED, three arms, same readout, same windows** (v1 step 29,999, 48 windows / 4 eps):
`A (z_hat,z_hat)` = 0.0449/0.0858/0.2917/0.5325 · `B (z_true,z_hat)` = 0.0616/0.1358/0.4424/0.7058 ·
⛔ `C (z_true,z_true)` = **1.5145/2.9449/5.4684/6.7631** at k=4/8/16/20.
**Arm C contains NO PREDICTION — only ground truth — and is 12–34× WORSE than the model's own
recursive rollout.** A general latent-pair decoder would score it near zero.

**WHY.** `cos(z_hat, z_true_next) = 0.98377` but `cos(z_hat, z_last_ctx) = 0.99872`, and
`cos(z_true_next, z_last_ctx) = 0.97980`. Consecutive latents are 0.98–0.999 similar: one frame of
motion is a TINY vector against the embedding magnitude, so the readout extracts displacement from a
tiny difference and is exquisitely sensitive to that difference's distribution.
`z_hat − z_last_ctx` lies on the predictor's learned manifold (the readout's training domain);
`z_true − z_last_ctx` is dominated by real frame content the predictor deliberately does not model.
⭐ **The 1-step prediction is closer to "stay put" than to the true next latent** — the predictor is
not trying to reproduce the encoder's output; it emits a state the READOUT can decode.

**WHAT I SHOULD HAVE DONE FIRST.** Run arm C — a two-true-latent decode — **before** building the
teacher-forced arm. It costs one batch, needs no new code path, and would have falsified the design
in minutes instead of after a full driver, two debug rounds and a smoke run.
⇒ **RULE: when importing an external metric, measure its precondition on our stack FIRST, as its own
pre-registered step.** "The published instrument is sound" says nothing about whether our
architecture satisfies it.

**ALSO NOTED — the pre-registration was incomplete.** It registered CR ≈ 1 (H-TASK) and CR > 1
(H-COMPOUND). **The observed result fell outside BOTH.** A pre-registration that cannot express
"the instrument is invalid here" will pressure the next reader to force an out-of-range result into
the nearest registered box. Future preregs must carry an explicit INSTRUMENT-FAIL branch — the same
lesson GATE_PROTOCOL §0.7 already encodes for `nonav_route_beats_majority`.

**REDESIGN (E-CR v2), not yet run:** move CR onto **latent error**,
`e_k = 1 − cos(z_hat_k, z_true_k)`, comparing the predictor's own outputs against the encoder's with
**no decoder in the path**. ⚠️ It answers the world-model question but no longer speaks in metres;
the link from latent error to ADE runs through the readout we have just shown is domain-sensitive.
**Do not convert one into the other.**

**STILL IN FORCE:** C61's retraction stands; the 1.03 → 1.91 exponent rise may not justify an
architecture change; **E-ROLL, rollout-recovery training and the Koopman lever remain BLOCKED** —
now because E-CR has produced NO admissible number, not because it came back flat.

---

## C64 — 2026-07-29 — a corpus re-selection that did not inherit the previous generation's VAL EXCLUSION

**Root-cause class: A MISSING CONSTRAINT IN A BUILD SPEC, not a coding error.**
(Distinct from C59/C63, which are stale-claim classes. Nothing here was ever asserted and later
falsified — the constraint was simply never stated, so nothing could enforce it.)

**FOUND BEFORE ANY NUMBER WAS PUBLISHED.** The v2corpus arm was at step 25,900 / 30,000 when the
pre-registered void check ran. **21 of the 40 canonical validation episodes are inside
`physicalai-v2bal-4b7eeeac222d`, v2corpus's TRAINING corpus** — 52.5 % of the surface the comparison
was going to be scored on. Artifacts: `…/2026-07-24-v2-corpus-50h-balanced/V2BAL_LEAK_FINDING.md`
and `v2bal_val40_leak_check.json`.

**WHY IT IS STRUCTURAL AND NOT A COLLISION ARTIFACT.** `episode_id = int.from_bytes(clip_id[:4])`
collides (9,000 clips → 8,391 distinct ids), so the intersection *could* have been false positives.
The base rate settles it: v2bal selected **9,000 of an 18,731-clip pool = 48.0 %**, and the observed
overlap is **21/40 = 52.5 %**. Those agree. A collision artifact would sit as a small excess on top
of a near-zero true rate, not land on the selection fraction itself. ⇒ **the selection simply drew
from the whole pool without excluding the incumbent val episodes.**

**HOW IT NEARLY COST A WRONG PROGRAMME CONCLUSION.** Scoring v2corpus on the full 40-episode surface
would have measured it on its own training data for half the episodes. The inflation is **one-sided**
— it would have **manufactured a "more data helps" result** for a corpus whose whole purpose is to
justify further corpus investment. That is the most expensive possible direction for a silent bias.

**WHAT SAVED IT.** The void check existed only because `PREREG_v2corpus_vs_v1.md` registered it as a
**mandatory first step**, written ~20 minutes earlier while the outcome was still unknown. The
pre-registration did not merely record a hypothesis — **it forced a check that a results-first
workflow would have skipped**, and it fired 12 hours before the checkpoint landed.

**RULE ADOPTED.** ⇒ **Any corpus re-selection MUST take the incumbent validation episode list as an
explicit EXCLUSION INPUT, and MUST emit the intersection count as a build artifact.** Had the v2
build printed `val_overlap = 21`, this was visible in July rather than on the eve of the comparison.

**STATUS — the contrast is NOT void, but it is materially changed:**
19 leak-free episodes remain (harness bar ≥ 8). ⚠️ **19 clusters gives a much wider paired
episode-cluster bootstrap interval than 40 — a tie on 19 is NOT equivalent evidence to a tie on 40.**
⚠️ The survivors are **not a random subsample**: they are what a manoeuvre-balanced selection left
behind, so they may skew toward lane-keeping — *against* the v2 arm — and that must travel with any
result. ⛔ **v1's published 0.4271 is a 40-episode number and is NOT the comparator on this surface;
v1 must be re-scored on the same 19.** Step 1 remains confirming the intersection at **clip_id**
granularity, because the number that voids an experiment should be exact.

---

## C65 — 2026-07-29 — v5 DIED at step 2000 on a MIXED TREE: new trainer, stale `taniteval`

**Root-cause class: A POD TREE ASSEMBLED FROM TWO GENERATIONS, WHERE THE STALE HALF IS ON A CODE
PATH THAT ONLY RUNS LATER.** (CLAUDE.md already warns that "a pod's `stack/` checkout drifts
silently and a launch from it resurrects fixed bugs" — this is that trap with a delay fuse: the
mismatch was invisible for 8 hours because the stale half is only reached at the first gate.)

**WHAT HAPPENED.** v5 (176×624 @ 117°) trained cleanly from 21:20 to 05:03 and died at **step 2000**,
the first firing of `--heldout-gate --heldout-every 2000`:
`heldout_gate.py:610 → ps.score_windows(pw, progress_term=…, recovery_term=…)` →
`TypeError: score_windows() got an unexpected keyword argument 'recovery_term'`.

**THE MIXED TREE, by mtime:**
| file | mtime | provenance |
|---|---|---|
| `stack/scripts/train_flagship_v4.py` | Jul 28 17:10 | scp'd at launch, **untracked** |
| `stack/tanitad/train/heldout_gate.py` | Jul 28 17:10 | scp'd at launch, **untracked** |
| `taniteval/taniteval/pseudosim.py` | **Jul 27 18:22** | **stale**, and `?? taniteval/` — the WHOLE tree is untracked |

pod2's `TanitAD` HEAD is **`0f93b98`**, **363 commits** behind. New gate code calling old
`taniteval`. ⚠️ **LOOP_STATE claimed pod2 and pod3 were both synced. It was wrong for BOTH** — pod3
was found stale earlier the same night (and re-synced), pod2 was never checked until it crashed.

⚠️ **A git sync would NOT have fixed this.** `taniteval/` on pod2 is **untracked** — a standalone
copy outside version control. Every "sync the pod" instinct in the runbook targets the git checkout
and would have left the actual offender untouched.

**WHAT DID NOT GO WRONG (checked, because the stale tree could have been much worse):**
`37ccfea` records that a stale import once produced **HFOV 120.0 where the pinned tree says 117.0,
exit 0, no warning**. The v5 config carries `176×624` and `117.0` ⇒ **the 8 hours trained at the
approved geometry.** Had it not, the run would have been silently void rather than merely stopped.

**THE FIX — one line, verified by import, no git state touched.** pod2 already carried a CURRENT
`taniteval` at `/workspace/tev/taniteval` (Jul 28 19:14, `recovery_term` present). `run_v5c.sh` set
`PYTHONPATH=/workspace/TanitAD/stack` only, so the stale in-tree copy won. `run_v5d.sh` sets
`PYTHONPATH=/workspace/TanitAD/stack:/workspace/tev/taniteval`. Verified BEFORE launching:
`pseudosim` resolves to the Jul 28 file, `score_windows` has `recovery_term`, `heldout_gate` imports,
and `hg._taniteval()` resolves to the same file. Relaunched 05:48:48Z; trainer **auto-resumes from
`ckpt.pt`** (step 1000), so **~1,000 steps ≈ 3.6 h lost, not 8 h**.

⛔ **A `git reset --hard origin/main` was attempted first and was CORRECTLY BLOCKED** by the
permission classifier. pod2 carries **317 modified tracked files** plus the two untracked trainer
files the run actually used — the reset would have destroyed the very trainer that produced the 8
hours. **The block prevented a real loss.** The minimal PYTHONPATH fix is strictly better than the
"clean sync" I reached for first.

**RULES ADOPTED.**
1. ⇒ **Verify the ENTIRE import surface before a long launch, not just the trainer.** A run that
   starts is not a run that is correctly wired: `--heldout-every 2000` means the gate's imports are
   unexercised for hours. **Import every module the run will EVENTUALLY touch, at launch time.**
2. ⇒ **"Sync the pod" is not a single action.** Enumerate every root on `PYTHONPATH` and check each
   independently — the offender here was outside git entirely.
3. ⇒ **Prefer the narrowest fix that a real import can verify** over a wholesale reset. The reset was
   more destructive AND would not have been verifiable without re-running everything.

**STATUS 2026-07-29 06:01 UTC — ⚠️ THE FIX IS NOT YET PROVEN.** v5 resumed from `ckpt.pt` and is at
step **1050**, climbing, stderr empty. **The gate fires at 2000**, so the fix stays unverified until
it passes that point (~950 steps ≈ 3.5 h at 13.1 s/step). *(The "step 2000" visible minutes after
relaunch was the OLD log line from before the crash, not the resumed run. Reading "the run is up" as
"the fix works" would be the same error as reading "a run that starts" for "a run that is correctly
wired" — which is the entire subject of this retraction.)*

**PREVENTIVE CHECK — rule 1 applied to the other live run, not left as a maxim.** v2corpus on newpod
is **NOT exposed to this class**: `taniteval` is not importable there at all, `/workspace/TanitAD` is
not a git repo on that host, and the v2corpus command carries **no `--heldout` flags**. Different
trainer path entirely; the mixed-tree defect cannot reach it. **MEASURED, not assumed.**

### ⚠️ STANDING HAZARD — pod2's tree is in a state no runbook covers

Recording this because it survived the incident unchanged and is the *precondition* for a worse one:

- **317 modified tracked files** under `/workspace/TanitAD`.
- `taniteval/` is **entirely untracked** — outside version control, invisible to every sync recipe.
- ⚠️ **The trainer that is running RIGHT NOW is itself untracked** (`train_flagship_v4.py` and
  `heldout_gate.py`, both `??`). It exists on exactly one disk.

⇒ **pod2 is one careless `git` command away from losing the v5 trainer.** A `git clean -fd`, a
`checkout .`, or the `reset --hard` the classifier blocked would each delete it. The recovery this
session worked *because* nothing was reset — not because the tree is sound.

⛔ **Do NOT reconcile this during an incident.** The correct time is **after v5 finishes**, as a
deliberate operation: copy the untracked trainer files off first, diff the 317 modifications to
separate real work from drift, then decide what to commit and what to discard. Attempting it while a
4.6-day run is in flight trades a known-working state for an unknown one under time pressure.

*(This is the same shape as the stranding rule — "an artifact on one disk is NOT done" — but applied
to the tooling rather than the results, which is why it went unnoticed for so long.)*

---

## C66 — 2026-07-29 — MY C65 FIX DID NOT WORK, and "verified by import" verified the wrong thing

**Root-cause class: VERIFYING A FIX THROUGH A PATH THE FAILING CODE DOES NOT USE.**
(C65's own rule said *"verify the entire import surface before a long launch"*. I did verify an
import — **just not the one the gate performs.**)

**WHAT HAPPENED.** v5 was relaunched 05:48 with `PYTHONPATH=…/stack:/workspace/tev/taniteval`.
At **09:21** it reached step 2000 and died with the **byte-identical** error:
`TypeError: score_windows() got an unexpected keyword argument 'recovery_term'`.
**A second ~3.5 h burned on the same 1000→2000 stretch (~7 h total).**

⛔ **WHY THE FIX WAS DEFEATED — `heldout_gate._taniteval()` IGNORES `PYTHONPATH`:**
```python
def _taniteval():                                   # heldout_gate.py:196
    repo = Path(__file__).resolve().parents[3]      # -> /workspace/TanitAD
    for p in (repo / "taniteval", repo / "stack"):
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)                   # ← POSITION 0 BEATS PYTHONPATH
```
It derives the repo from **its own file location** and force-inserts `/workspace/TanitAD/taniteval`
— the **stale Jul-27 tree** — at the **front** of `sys.path`. My `PYTHONPATH` entry was never
consulted. **The environment cannot fix a path the code hard-derives.**

⚠️ **AND MY VERIFICATION WAS VACUOUS.** I ran a bare `import taniteval.pseudosim` under the new
`PYTHONPATH` and saw the Jul-28 file — true, and **irrelevant**, because a plain import is not what
the gate does. I never called `_taniteval()` itself. **A fix verified through a different code path
than the failure is not verified.**

**THE ACTUAL FIX.** Replace the tree **at the path the resolver forces**. The two trees are not
interchangeable file-for-file — current has **46** modules, stale **43**, and `pseudosim.py` is
94,413 B vs 48,548 B — so swapping one file could break imports the newer file needs.
- `mv /workspace/TanitAD/taniteval → taniteval.stale-20260729-C65` (**PRESERVED, not deleted**)
- `cp -a /workspace/tev/taniteval → /workspace/TanitAD/taniteval`
- `taniteval/` on pod2 is **untracked**, so no git state was disturbed.

✅ **VERIFIED THROUGH THE FAILING PATH THIS TIME — two checks, not one:**
1. **The resolver:** `hg._taniteval()` → `/workspace/TanitAD/taniteval/taniteval/pseudosim.py`,
   `score_windows` signature **contains `recovery_term`**.
2. ⭐ **The CALL SITE:** invoked `ps.score_windows(pw, progress_term=term, recovery_term=rterm)`
   with `term/rterm` read from `heldout_gate` itself. **The `TypeError` is GONE** — it now fails with
   `KeyError: 'traj'`, which is my synthetic `pw` lacking a real trajectory field. **Reaching the
   inside of the function is the proof.**

**RULES ADOPTED.**
1. ⇒ **Verify a fix by exercising THE FAILING CALL SITE, not an analogue.** "The import resolves" is
   not "the gate's import resolves"; "the module loads" is not "the function accepts these kwargs".
2. ⇒ **Suspect hard-derived paths before blaming the environment.** `Path(__file__).parents[N]` and
   `sys.path.insert(0, …)` silently outrank `PYTHONPATH`. **Grep for them before assuming an env var
   will land.**
3. ⇒ **When a fix fails, re-diagnose from zero.** I re-applied the same class of fix (a path change)
   rather than asking *why* the path change had not taken. The second failure was avoidable.

⚠️ **STILL UNPROVEN IN PRODUCTION.** v5 relaunched **09:23:17Z**, resumed from `ckpt.pt` @1000,
running 5 procs / GPU 100 % / stderr empty. **The gate fires again at step 2000 (~3.5 h).** The
call-site test raises confidence sharply but is **not** the production probe. ⛔ Do not record C66 as
closed until step 2000 is passed.


## C67 — 2026-08-01 — I KILLED v5 AND NEVER RELAUNCHED IT; IT SAT DEAD ~4 DAYS

**What I did.** To remove `--heldout-gate` as the PI instructed, I killed v5 by explicit PID at
`2026-07-29T09:55:23Z` (`rc=143`, SIGTERM). The plan was kill -> edit the launcher -> relaunch.
The edit **failed** (`sed: -e expression #1, char 11: unterminated s command`), leaving
`run_v5e.sh` still carrying the gate flag. The weekly API limit hit before I noticed, and
**the relaunch never happened.** pod2 was idle 2026-07-29 09:55 -> 2026-08-01 22:14 UTC, about
**3 d 12 h**.

**What was NOT lost.** `ckpt.pt` survives at step 1000. No model state was destroyed — the loss is
GPU-days, plus the 1000->2000 stretch for the third time.

**ROOT-CAUSE CLASS: destructive action taken before its replacement was verified ready.**
A kill is irreversible; an edit can fail. Ordering them kill-then-edit means any failure in the
edit leaves the pod dead with nobody watching. This is the same family as C65/C66 (acting on an
unverified fix) but with the irreversible step FIRST, which is strictly worse.

=> **RULE: never stop a running job until its replacement launcher is written AND verified.**
Prepare, verify, then kill-and-launch as a single action.

**Second, compounding error in the same episode.** I checked the new launcher with
`grep -c heldout` and got `1` — which was **my own explanatory comment**, not a flag.
=> **`grep -c <keyword>` is not a check for a flag.** The check is *"does any NON-COMMENT line
carry it"*. Same family as C66: a verification that does not exercise the thing it claims to verify.

**Fixed 2026-08-01 22:14Z.** Launcher rewritten via heredoc (no `sed` surgery), verified three
ways — non-comment `heldout` lines = 0, `bash -n` clean, and the emitted python arg block read
back in full — then launched. v5e stepping, stderr 0 B. Removing the gate also eliminates the
C65/C66 failure mode by construction: `heldout_gate._taniteval()` is never imported, so its
hard-derived `sys.path.insert(0, ...)` (which ignores `PYTHONPATH`) can no longer force-load a
stale tree.


---

## 2026-08-02 — FOUR retractions from ONE adversarial-verification pass (12 agents, 6 streams)

⭐ **All four were found by adversarially verifying MY OWN reports, not by the runs that produced
them.** Each report had passed my own review. The verification pass is now the load-bearing step,
not a formality.

### R-2026-08-02-a — REF-C's Thor four-family numbers: scored at the WRONG RASTER
**Root-cause class: SILENT SHAPE ACCEPTANCE — a model that validates nothing, and an eval that
assumed the model would.**

REF-C trains at 256 px square -> `grid_shape (8,8)` = 64 tokens. The eval fed a 176x624 sub-frame
= 120 tokens. **XL crashed loudly** (it has `graft_imagination=True`, which reshapes to the declared
grid). **base returned numbers silently** (`graft_imagination=False`, and `feat_proj` accepts any
token count). I read the crash as an XL-specific defect and the silent output as a valid result.
**It was the reverse: the crash was the instrument working.**

⇒ **RULE: assert the fed geometry against the checkpoint's declared shape before every scoring run.
When two arms share a defect and only one crashes, the crash is the honest signal.**
⇒ **RULE: an implausible magnitude IS the finding.** `speed_mae 3.06 m/s` for an arm with registry
ADE@2s 0.4728 was in the published table; I caveated *n* and provenance instead.

### R-2026-08-02-b — "the collector did not report route provenance": FALSE
**Root-cause class: `.get()` CONVERTED A SCHEMA MISMATCH INTO A PLAUSIBLE `None`.**

`refc_eval.py:177-190` always stamps `nav_provenance`. My caller read
`win.get("route_input_exercised")` at **top level**, where the key has never existed. The tell was
in the artifact: `nav_note: null` — a key that has never existed at top level under *any* mode.

⇒ **RULE: read a required stamp with `[]`, never `.get()`. A `None` from a BOOLEAN-valued field is
a read-path bug until proven otherwise — never evidence about the thing being measured.**
⚠️ Second-order: `route_input_exercised` = `nav_mode != "follow_constant" and len(hist) > 1`
conflates **exercised** with **varied**. Read `fed_command_hist`.

### R-2026-08-02-c — v5_guard's strategic follow-rate was deflated by an INVISIBLE 4th CLASS
**Root-cause class: CLASS-VOCABULARY MISMATCH between the label pipeline and the instrument, hidden
by a histogram that nobody checked summed to n.**

`v5_guard_5k.json` published `n_windows: 881` with `cmd_distribution {0:212, 1:394, 2:121}`.
**212+394+121 = 727.** 154 windows (17.5 %) were route class **3 = `ROUTE_UNKNOWN`** (unjudgeable,
`refb_labels.py:536`), which `classify_route` can **never** emit -> **deterministic misses in both
arms**, folded into `follow_true`, `follow_shuffled` and the bootstrap. The `ROUTE_LIVE` verdict
survives; the point estimate and CI were deflated ~1.21x. **The exact value is NOT recoverable from
the JSON** (`run_guard` returns only aggregates) — a re-run is required.

⇒ **RULE: any published class histogram MUST sum to the scored n, and the instrument must state
what it excluded.** A silently dropped class looks exactly like a model failure.
⇒ **RULE: never default an unjudgeable label to a real class** (`refb_labels.py:511` already said
"NEVER DEFAULT TO STRAIGHT"); exclude it and report the count.

### R-2026-08-02-d — the Thor "precision gate PASS / error does not compound"
**Root-cause class: MEASURED ON A RANDOMLY-INITIALISED MODEL FED GAUSSIAN NOISE — the exact defect
the paper's own §7.10 blockquote exists to prevent, repeated.**

None of the five Thor scripts call `torch.load` or `load_state_dict`; inputs are `torch.randn`.
- **Latency survives** — it is weight-independent. 272.56 -> 51.2 ms (5.33x), the K-sweep, batch
  scaling, RSS and thermals remain admissible **as architecture reads**.
- ⛔ **The precision gate does NOT survive.** Quantisation error is a function of the *trained*
  weight/activation distribution; outlier channels are the whole difficulty and a random network
  has none.
- 🔴 **We measured the OPPOSITE on real weights.** Paper §7.10: post-pool `readout_head` collapses
  to cosine 0.566 under W+A INT8, costing **+0.0215 m ADE@2s — past the pre-registered 0.02 m
  falsifier — with degradation growing 27x from 0.5 s to 2 s.**
- ⚠️ The CUDA-graph "bit-exact" row is near-tautological: a *static* input replayed through a graph
  must reproduce itself. An aliasing test needs **varying** inputs. UNVERIFIED as a hazard test.

⇒ **RULE: a precision/quantisation claim is inadmissible unless it was measured on TRAINED weights
and REAL inputs.** Latency may use random weights; numerics may not.
⇒ **RULE: state the five conditions (hardware, precision, corpus, n, WEIGHTS) or do not publish the
number.** The paper already carries this rule in §7.10 and it was still repeated.


## 2026-08-03 — R-2026-08-03-a: PSNR was not a valid metric for the NuRec render check
**Root-cause class: QUOTED A METRIC BEFORE CHECKING IT COULD DISCRIMINATE — on a corpus where it
provably cannot.**

I reported the first gsplat render of a NuRec scene as "PSNR 16.758 dB, 20.689 after affine colour
fit". The negative control (our ONE render scored against the correct reference frame AND four wrong
ones) shows those numbers certify nothing:

| ref frame | PSNR | NCC | grad-NCC |
|---|---|---|---|
| **0 CORRECT** | 16.758 | 0.704 | **0.2719** |
| 150 wrong | **17.457** | 0.767 | 0.1737 |
| 450 wrong | 17.073 | **0.782** | 0.1163 |

**A WRONG frame beats the correct one on PSNR (17.457 > 16.758) and on plain NCC (0.782 > 0.704).**
Every frame of the clip is a dark night street, so ~17 dB measures "both images are dark".

✅ **grad-NCC discriminates correctly** — argmax = frame 0, margin 0.0806 (0.2719 vs 0.1913, ~1.42x).
The mapping IS validated; it is validated by STRUCTURE, not by photometry.

⇒ **RULE: on a low-dynamic-range corpus (night, fog, tunnel, snow), run the negative control FIRST
and let it CHOOSE the metric.** A metric that cannot separate the right frame from a wrong one cannot
certify anything, no matter how reasonable its value looks. Same family as the 2026-08-02
retractions: an instrument fed the wrong thing that did not complain.

⚠️ Second finding the same pass: switching the sky env-map on made the render WORSE
(mean 0.240 -> 0.391 vs reference 0.266; PSNR 16.76 -> 15.32) because it fills the ~49% of pixels no
gaussian covers. "The obvious missing piece" is not automatically an improvement — measure it.


## 2026-08-03 — R-2026-08-03-b: "pod2 is the only live pod" and "not a memory problem" — BOTH WRONG
**Root-cause class: SURVEYED ONLY THE HOSTS I ALREADY KNEW, and READ A LIMIT AT THE WRONG SCOPE.**

**(i) The fleet.** I probed the 4 hosts in `~/.ssh/config`, found 3 refused, and reported *"pod2 is
the entire GPU fleet"*. The PI corrected me: a second pod was training. It is
**`69.30.85.48:22192`** (RunPod `interesting_gray_ant` / `v9ni8rpan3qyn3`), an A40 running
`flagship-v1arch-v2bal-30k` — **at step 9750 with `g_op_fwd_ade_m` 0.0898**, started 2026-08-01
23:34 UTC. It was simply not in my ssh config. Grepping the repo for connection strings found it in
one command.
⇒ **RULE: an ssh config is a cache of what I happened to write down, NEVER the fleet inventory.**
Enumerate from the provider or from repo-wide connection strings before any "we have N pods" claim.
Now registered as `tanitad-pod4` / `tanitad-v2arch`.

**(ii) The memory.** Diagnosing v5f's death I ran `free -g`, saw **503 GB total / 487 GB available**,
and wrote *"not memory"*. That is the **HOST**. The container cgroup limit is
`memory.limit_in_bytes = 49,999,998,976` — **50 GB**, and usage was **48.9 GB**. RunPod's own console
showed an *"Out of Memory (OOM) Detected"* banner and `Memory 50 GB` the whole time, and
`memory.oom_control` records **`oom_kill 6`** — the container has been OOM-killed **six times**.
⇒ **Exactly the class CLAUDE.md already warns about for disk** (*"Never judge pod disk with `df`" —
it shows the cluster, not the per-pod quota*). **The same trap exists for RAM: `free` shows the host,
not the cgroup.** Read `/sys/fs/cgroup/memory/memory.limit_in_bytes`, never `free`.

⚠️ **What survives:** v5f's specific crash today WAS `torch.OutOfMemoryError: CUDA out of memory`
(GPU, 44.42 GiB) — a traceback, not an inference. GPU-OOM and container-RAM-OOM are different
resources and both are real here. The mitigations applied (batch 16->8, accum 4->8,
expandable_segments, v2-lru 8->6) address both. Current state: 38.9 GB of the 46.4 GB usage is
**reclaimable page cache**, only 6.4 GB is RSS, `under_oom 0`, and v5f is stepping — so this is
tight, not imminent. `drop_caches` is DENIED inside the container (read-only `/proc/sys`).

⚠️ **Two operational risks the console shows and no probe of mine would have found:**
1. **"We have detected a critical error on this machine"** — RunPod advises backing up and
   recreating the pod.
2. **Scheduled maintenance 2026-08-06 21:00 → 2026-08-08 21:00 MESZ**, server down. v5f will not
   survive it in place; it must be checkpointed and migrated, or restarted after.
⇒ **RULE: the provider console carries state the pod cannot self-report. Check it before planning
multi-day runs.**


## 2026-08-03 — R-2026-08-03-c: every four-family RATE the programme has published is wrong by 5x-25x
**Root-cause class: A HAZARD DOCUMENTED NEXT TO ONE CALLER INSTEAD OF FIXED AT THE SHARED FUNCTION —
plus NO NUMBER WAS EVER COMPARED AGAINST A PHYSICAL QUANTITY THE DATA ALREADY CARRIED.**

`taniteval/four_families.py` hard-coded `DT_S = 0.1`, but `all_families` reads `win["pred"]`, which
for **both** `rollout.collect` and `refc_eval.collect` is the **SPARSE 4-waypoint view at
`WP_STEPS = (5,10,15,20)` — a 0.5 s grid**. Every derivative was divided by the wrong dt.

**MEASURED (Thor, 859 real held-out windows, `thor_c4_rescore_corrected.json`):** the ego's own
recorded speed `poses[:,3]` averages **12.4565 m/s**; the instrument returned **62.9789 m/s** —
ratio **5.0559**. At the true dt = 0.5 it returns **12.5958** (1.011x truth).

| corrections | factor | mechanism |
|---|---|---|
| `speed_*` | **/5.00** | 1/dt |
| `accel_*` | **/25.00** | 1/dt^2 |
| `yaw_rate_*` | /6.48 | 1/dt **and** the validity mask |
| `curvature_*` | **/8.36** | ⭐ the **mask alone** — curvature is dt-INVARIANT |
| `heading_*` | /1.90 | the mask alone |
| `along_*`, `cross_*` | **x1.00** | dt-invariant, always correct |

⭐ **The mask half is the bigger trap.** `MIN_DS_M = 0.05 m` is a *distance* gate meant to drop steps
where the vehicle barely moves (curvature = dtheta/ds **explodes** as ds->0). On a 0.5 s grid it is
5x too permissive: **excluding 38 more steps out of ~2465 moves `curvature_mae` by 8.4x.** Two
metrics that are dt-invariant by construction were still wrong, through the mask.

✅ **What survives:** every CROSS-ARM comparison, rank, and paired delta — the factor is common to
both arms. ⛔ **What does not:** every ABSOLUTE rate quotation, and every comparison against a
physical or external bar. Including the 2026-08-02 REF-C Thor panel (`speed_mae 3.0609 ->
0.6122 m/s`). `19.25 m/s^2` was never a physically possible acceleration and nobody noticed.

🔴 **THE ROOT CAUSE, and it is the transferable part: the trap was ALREADY DOCUMENTED — in a FORK.**
`stack/tanitad/eval/idm_families.py` says verbatim that feeding 4 waypoints at {5,10,15,20} to
`four_families` *"reads every speed and yaw-rate 5x too large"*, and its author responded by
**re-implementing the geometry with an explicit cadence for the IDM**. What nobody noticed is that
`rollout.collect` and `refc_eval.collect` emit **exactly that same shape** — so the trap was never
hypothetical, it was already live in the mainline instrument.
⇒ **RULE: when you find a hazard in a shared function, FIX THE SHARED FUNCTION. A warning written
beside your own caller protects only your caller, and it actively harms everyone else — a reader who
greps and finds the hazard documented concludes that someone has already handled it.**
⇒ **RULE: an instrument that derives a physical quantity must be checked against that quantity where
the data already carries it.** The episodes had `poses[:,3]` all along; one comparison would have
caught this at any point in the last months.

**FIXED + tested (staged):** `taniteval/taniteval/four_families.py` (dt derived from the window's own
`wp_steps`x`dt_s` contract via new `infer_dt`, never guessed silently; `prefer_dense=True` uses the
true 10 Hz path when present; `MIN_DS_MPS = 0.5 m/s` so the gate scales with cadence; every family
carries its `dt_s` and `all_families` emits a `_grid` provenance block) and
`taniteval/tests/test_four_families_dt.py` (12 tests, all passing).

---

## 2026-08-03 — R-2026-08-03-d: "REF-C's route pathway is INERT" — REFUTED by the experiment written to test it

**Class: an INFERENCE from true premises, carried as if it were a measurement.**

**Retracted claim.** That REF-C's 4-way `nav_cmd` is functionally dead — the reading behind
`PREREG_lan_refc.md` §1's *"the route input is degenerate at train time **and constant at eval
time**. A decoder in that position learns the marginal"*, and behind the LAN design that replaces it.

**What is MEASURED instead** (run dir `tanitad-thor:/home/nvidia/lan_e0/`, banked at
`…/Implementation/incoming/2026-08-03-lan-refc-e0/E0_refc-base_navcf_full.json`; REF-C-base
104.192 M, 859 windows / 39 val episodes, 256px SQUARE raster asserted, paired episode-cluster
bootstrap n_boot 2000):

- Sweeping `nav_cmd` moves the decoded trajectory by **0.2416 m** over the label-reachable commands
  {follow, left, right}, with the bit-identical-input control at **exactly 0.0** (tol 1e-6).
  **Verdict RESPONSIVE, not INERT.**
- ⇒ Every published REF-C number did not hold a *dead* input constant. It held a **live** one
  constant. The C6 confound is real and is now MEASURED — in the opposite direction from the one
  that was argued for a fortnight.

**And the pre-registered fallback is ALSO refuted.** §7 committed in advance that a RESPONSIVE
reading makes the cheap fix *"supply the label at eval and re-score every REF-C row … a bigger,
cheaper correction than LAN"*. Measured, assembled from the same decodes at zero extra GPU:

| arm vs `follow_constant` | ADE@2s Δ | LATERAL `cross_mae_m` Δ |
|---|---|---|
| **oracle** route (GT label, upper bound) | +0.0024 [−0.0107, +0.0147] **not sep** | **+0.0031 [+0.0001, +0.0063] SEPARATED WORSE** |
| **produced** route (model's own head) | **+0.0118 [+0.0011, +0.0227] SEPARATED WORSE** | +0.0028 [+0.0003, +0.0053] SEPARATED WORSE |

Handing REF-C the **correct** route degrades cross-track and curvature and buys nothing anywhere.
⇒ **Supplying the label at eval is NOT the cheap fix. Do not re-score the REF-C rows expecting a
gain.**

🔴 **THE ROOT CAUSE, and it is the transferable part.** Every fact in §1 was individually MEASURED
and individually true: the label thresholds net heading over 15–25 s; validity is 0.27; eval feeds a
constant. The *conclusion* — "therefore the pathway is unused" — was an **inference**, and it
travelled through six documents wearing the evidence class of its premises. **A chain of MEASURED
links does not make the conclusion MEASURED.** The experiment that could have settled it
(`nav_cmd_sensitivity`) is **four forward passes** and was available the whole time.
⇒ **RULE: when a design exists to fix a mechanism, the FIRST run is the one that shows the mechanism
is broken — not the one that shows the fix works.** §7 got this exactly right by pre-registering E0
as the cheapest discriminating experiment; the error was the fortnight before anyone ran it.

**Two secondary corrections banked in the same run:**

1. **A 4-way sweep of `nav_cmd` is really a 3-way sweep.** `refc_eval.ROUTE_TO_NAV = {0:1, 1:0, 2:2}`
   maps the 3-class route label onto nav **{0,1,2}**; index 3 is an embedding row no label or eval
   mode can emit. It dominates the raw sweep (`0v3` 1.5126 m vs `0v1` 0.2416 m) and would have
   overstated route sensitivity **7.3×**. Reporting it as a route response is an OOD probe
   mislabelled. `lan_probe.py` now reports the reachable-only sweep and takes its verdict from it.
2. **`PREREG_lan_refc.md` §6.2's PRIMARY OUTCOME READOUT IS NOT COMPUTABLE ON REF-C.** MEASURED:
   REF-C's decoder emits `traj` of shape **[B, 4, 2]** — 4 waypoints, **2.0 s**. `corridor_departure
   @ K=185 (18.5 s)` has no open-loop path to run on. E1a's K=185 result came from a **rollout**
   decoder; REF-C's is anchored one-shot. ⇒ **RULE: a pre-registration must state the readout is
   computable ON THE ARM, not merely that the instrument exists.**

**Near-miss caught before publication, logged because the mechanism generalises.** The per-window
reducers added for the bootstrap back-filled windows where every step failed the `MIN_DS` gate with
their **unmasked** row mean — importing exactly the crawling/stopped windows the gate removes:
`heading_mae` **4.0181°** vs `four_families`' **1.1486°**, `curvature_mae` **42.1365 1/m** vs
**0.00788** (**5,347×**). It would have published a separated lateral degradation
(+0.2241 [+0.0178, +0.5856]) that does not exist (true value +0.0185 [−0.0080, +0.0458], **not**
separated). Caught by a **component-vs-family self-consistency control** that now ships inside the
result JSON and runs on every invocation. ⇒ **RULE: when you compute per-window components to feed a
cluster bootstrap, assert they reduce to the family mean printed beside them. Same family as
R-2026-08-03-c: a derived quantity must be checked against the one the data already carries.**

## 2026-08-03 — R-2026-08-03-e: "TRT-fp16 flips 13.5 % of tactical selections" — CAUGHT BEFORE PUBLICATION
**Root-cause class: AN ENGINE COMPARED AGAINST A MODEL IT DOES NOT IMPLEMENT — a wrapper that
ACCEPTED an argument and IGNORED it. Third instance of the family behind runbook learnings #8
(identical-across-precisions error = wiring bug) and #9 (a gate compared the engine to a *different*
random model).**

Building the batch-9 predictor engine (P6), the first selected-candidate comparison read **48.3 %
agreement** against the 95.3 % bar, with a max score delta of **73.4** on scores whose decision
margin is **1.9**. The tempting headline — *"fp16 is not safe for the imagine-and-select path"* —
was wrong.

- **Three factors had changed at once.** Decomposed one at a time (200 windows / 23 episodes,
  episode-cluster bootstrap): fan **batching** alone **1.0000** [1.0,1.0]; **bf16 encoder** alone
  **1.0000**; the *engine* step **0.8650** (K=4) / **0.4000** (K=20).
- 🔴 **The engine step was MY OWN wiring bug.** The ONNX export was `(states, actions) -> z_next`,
  dropping the **D-030 tactical intent token**. `TacticalSelector` passes `intent=` as a KEYWORD, and
  a two-input wrapper takes it and discards it — no error, no warning. The engine arm was computing
  the *unconditioned* prediction, so the two arms were different models.
- ✅ **Rebuilt with `intent` as a third input** (`intent_is_live_rel_change = 0.0522`, so the seam is
  verifiably live): the fp16 batch-9 engine agrees on **200 of 200** selections with **exactly 0.0
  regret**. The intent-less control costs mean **0.131 m** regret, p95 **1.07 m**, max **4.76 m**.
- ⚠️ **A one-factor result would have been published as a precision verdict** and would have blocked
  fp16 deployment on a defect that does not exist.

⇒ **RULE: a runtime wrapper must REFUSE an input it cannot honour.** Implemented, not just written
down: `stack/scripts/build_predictor_trt.py::TRTPredictor.forward` raises when an `intent` token is
passed to an engine without an `intent` input **and** when one is withheld from an engine that has
it; `verify_engine` asserts the token changes the output. Silence was the only symptom this bug had.
⇒ **RULE: when an A/B differs in ≥2 respects, the FIRST run is the decomposition, not the verdict**
(class **C6**, and the decomposition here cost 6 minutes of GPU).
⇒ **RULE: verify an exported engine against the model WITH THE SAME CONDITIONING INPUTS.** Comparing
a conditioned reference to an unconditioned engine passes every shape and rel-err check that does
not exercise the conditioning.

**Two further corrections banked in the same run:**
1. 🔴 **`thor:~/trt/predictor_fp16.plan` — the "shipped" engine — was built from a RANDOMLY
   INITIALISED model** (`thor_trt.py` contains no `torch.load`/`load_state_dict`; second probe: its
   profile is static (1,8,2048)). It was never deployable, for a reason independent of its batch-1
   profile. Superseded by `thor:~/trt_deploy/` + `MANIFEST.md`.
2. ⚠️ **"Rebuild the engine at batch 9" was an INSUFFICIENT instruction.** `propose_and_score` loops
   over candidates, so the rebuilt engine driven by the unchanged caller measures **272.8 ms** —
   *worse* than the batch-1 engine's 265.7 ms. ⇒ **RULE: an optimisation stated as an artifact change
   must name the CALLER shape it requires**, or it silently buys nothing.

## 2026-08-03 — R-2026-08-03-f: "VISION ADDS NOTHING OVER EGO STATE" (`situations.py:19`) — RETRACTED
**Root-cause class: AN ARM RETIRED ON A COMPARISON IT WAS NEVER GOING TO WIN AND DID NOT NEED TO
WIN — the baseline was a signal the deployed system is not allowed to read, and the arm's OWN null
control was sitting in the same file, unrun. Sibling of C9/C13/C14 (an instrument structurally
unable to report the answer it is cited for) and of R-2026-08-03-e (an engine compared against a
model it does not implement).**

`stack/tanitad/data/situations.py:19-22` asserted, and ≥3 downstream documents repeated, that the
front camera *"has no measured signal to stand on"*, on the strength of `head_ego` CV-AP **0.0697**
beating `head_img_ego` **0.0525** and `head_img` **0.0376**.

- 🔴 **Defect 1 — the multimodal arm in that comparison was broken.** `sc_train.py:143` fuses by
  `np.concatenate([img, S["E"]], 1)`: a 16-dim PCA image block normalised by its own global mean-abs
  (`:130-131`) against a 3-dim ego block divided by a hand-set `EGO_SCALE = [10, 2, 0.5]` (`:38`),
  into one shared `nn.Linear`. Two unrelated normalisations, 5.3 : 1 dimensional imbalance. A broken
  fusion is not evidence about the modality it broke.
- 🔴 **Defect 2 — the baseline is not a legal inference input.** PI, 2026-08-03: *"for ground truth
  data of scenario classification you can use both ego and other label, for inference only vision."*
  Ego may DERIVE the labels; it may not be READ at inference. So "is vision better than ego" cannot
  decide the deployment question it was cited for. **The deployable question is whether vision beats
  CHANCE — and that comparison had never been run**, although `head_img_shuf` (the camera's own
  permuted-feature null) was banked in the same `.npz` from the start.
- ✅ **Run now, it separates.** `head_img` 0.03741 vs `head_img_shuf` 0.01715 on `lane_change`,
  ΔAP-lift **+1.1749 [+0.7930, +1.6890]**, paired episode-cluster bootstrap B=2000, 1,610 clip
  clusters (`…/incoming/2026-08-03-sitclf-fusion-wired/results_sitclf_vision_only.json`).
  **2.18× its own null.** The camera carries situation signal; it was never shown not to.

**MEASURED alongside, and it is why the old comparison was structurally unfair.** Label provenance,
two probes: every situation label is a pure deterministic function of the ego pose track
`[x, y, yaw, v]` — `stack/scripts/emit_situation_labels.py:54-62` reads only `d["poses"]`, and every
detector in `situations.py` (`:161`, `:210`, `:244`, `:284`) takes only `K = kinematics(P)`; the
emitter passes `cross=None`, so even `intersection` is the turn half alone. An ego-input head
observes the label's **generating process**; the camera must infer it from pixels.

⚠️ **Do NOT over-claim this as a leak — I checked and it is not one.** The head's window is
[t−0.7 s, t] (`sc_train.py:37`, offsets −7..0) and the label's evidence window is
[onset, onset+4 s] with onset > t: **disjoint, no future information**. The precise statement is
*same-source privileged access*, not leakage. One genuine boundary defect does exist and is new:
`omega_pre`/`alon_pre` are built on `np.gradient`, a **centred** difference, so they read **one frame
(0.1 s) past t** despite the source comment claiming "STRICTLY CAUSAL" — it bites only for onsets at
exactly t+1, but the comment overstates the guarantee.

⇒ **RULE: before retiring a modality, score it against ITS OWN NULL, not against a rival modality.**
A rival-modality comparison answers "which is better", never "does this work" — and if the rival is
not deployable, it answers nothing at all.
⇒ **RULE: a claim of the form "X adds nothing over Y" is inadmissible unless Y is a legal input to
the deployed system.** Check the deployment contract before the statistics.
⇒ **RULE: when a fusion is the mechanism under suspicion, no arm that passes through that fusion may
be quoted as evidence about its inputs.**

---

## R-2026-08-03-dtac1 — "REF-C's tactical head is INPUT-limited (blind to v0)" — REFUTED by my own pre-registered probe

**Root-cause class: A MECHANISM THAT IS REAL IN THE SOURCE IS NOT THEREBY THE BINDING CONSTRAINT.**
(Sibling of the "score it against its own null" class: I found a true structural defect by reading
code and promoted it to *the* cause without measuring how much of the failure it explains.)

**What I asserted**, in `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md` §6.3, registered
before running: *"My prediction, registered before running: INPUT-limited."* Grounds were sound and
MEASURED — `refc.py`'s `maneuver_head` genuinely reads `pooled` (the image embedding) alone while its
label is `dv = v(t+2s) − v(t)`, and REF-A's speed-input result (3.73 → 0.83 m) is real.

**What the measurement says** (`…/incoming/2026-08-03-dtac1-tactical-head/`, `refc-base` step 29999,
canonical val, 39 episodes / 1364 windows, `taniteval.ci.episode_cluster_bootstrap`):
`auc_lon_active` recovered from the EXISTING 5-way head = **0.7294** (shuffled control **0.4933**).
**The longitudinal information is already in the head that cannot emit a longitudinal class.**
The pre-registered threshold for this branch (≥ 0.65) was fixed in advance. A linear probe confirms
the input lever is real but *modest*: `pooled` 0.3833 → `pooled+v0` 0.4346 macro-recall (+0.051).

**And the correction has a second half the pre-registration also got wrong**, in the other
direction: its READOUT branch claimed F2+F3 would be *sufficient*. The τ frontier says F2 alone
(τ = 0) yields brake recall **0.072** / accel **0.045** — near-nothing — and that **`accelerate`
cannot be recovered at ANY τ** (peak 0.153) because the rarest class crowds it out as τ rises.
Separately, **9.68 % of windows have their longitudinal class destroyed by the 5-way LABEL**, which
no decode rule can undo.

⇒ **RULE: reading a defect out of the source establishes that it EXISTS, never that it DOMINATES.**
Before a structural claim decides a GPU-day, measure the fraction of the failure it accounts for.
⇒ **RULE: when a fix has separable levers, pre-register a probe that ORDERS them, not one that
confirms the favourite.** This probe cost minutes on an idle box and reversed the ordering
(F1 > F2 > F3 became F3 > F2 > F1), which is a full REF-C retrain's worth of scope.
⇒ **RULE: a single τ / threshold is a POINT on a trade-off, and the "principled" value is not
automatically the right one.** τ = 1 (the balanced posterior) maximises the prior correction, not the
metric: it took accuracy 0.705 → 0.427 and predicted `brake_stop` on 865 of 1364 windows against 153
true. Report the frontier; choose the operating point on train/dev data, never on val.

---

## 2026-08-03 — "the 20 s night clip contains no junction" — REFUTED. "…no junction-scale DECISION" — upheld, and now proved.

**Root-cause class: an ABSENCE ASSERTED FROM THE INSTRUMENT'S SILENCE, never probed against the
asset that owns the fact.** Same class as the Vulkan ICD (12 days) and `obstacle.offline`.

**What was asserted**, in `stack/experiments/alpasim-gsplat/cl_metrics.py` `families()` — and from
there in the STREAM C brief and in every report quoting the degenerate strategic block:
*"this 20 s clip contains no junction-scale strategic decision"*, offered as the explanation for
`route_head_eq_logged = 1.0000`. The grounds were real but indirect: `route_from_future_v21`
returned `ROUTE_UNKNOWN / road_following` on 100 % of windows. **Nobody opened `map.xodr`.**

**What the map says** (`stack/experiments/nurec-gsplat/results/junction_00040136.json`, two
independent sources agreeing to **rms 0.1053 m**, all three discrimination controls PASS):
the ego is **INSIDE a junction for 46 of its 202 poses (22.8 %)**, traversing **four** of them
(220, 222, 230, 239). Median clearance to the nearest junction is **19.963 m**, not "there isn't
one". NVIDIA's own `clipgt/intersection_area` independently labels **4** intersection polygons,
two of them entered, and **100 % of every polygon's vertices land on an xodr junction surface**.

**But the conclusion the false premise supported turns out to be TRUE for a different reason.**
At **every one of the four**, the ego's own lane has exactly **ONE** admissible continuation in the
junction's `<connection>` table, and the largest heading change through any of them is **2.58 deg**.
⇒ `route_from_future_v21` was **right**: road following really was the only option. The instrument
was never the problem; the scene has no branch.

⇒ **RULE: "the metric is degenerate here" is a hypothesis about the SCENE, and the scene has an
owner — go read it.** A label that abstains and an environment that offers no choice produce
identical output; only the map separates them. Had the survey been scoped as "find a scene with a
junction" instead of "find a scene with a BRANCH", it would have returned ~1265 of 1607 scenes and
almost all of them would have been just as degenerate.
⇒ **RULE: a degeneracy flag must name the QUANTITY that is degenerate.** `n_options == 1` is
checkable; "no junction-scale decision" is prose that survived because nothing could falsify it.

### Two corpus-level traps found while proving this (both now regression-tested)

**1. ⛔ In PhysicalAI-NuRec's OpenDRIVE, the reference line is NOT the driven line.** MEASURED on
scene `7c72937c`, road 35: `laneOffset a = 10.495 m`, so the **reference line sweeps 40.5 deg while
the lane centreline the car drives is straight to within 0.5 deg**. Computing a branch angle from
`planView` headings gave **+51.49 deg** for a manoeuvre driven at **+123.53 deg**. Any route,
heading or curvature quantity must come from the sampled lane centreline (reference + `laneOffset`
+ inner widths), never from a `<geometry hdg=…>` attribute. *This one was caught by the mandatory
component-vs-family self-consistency control, which fired on every left turn in the shortlist — the
second time in the programme that control has stopped a wrong number from being published.*

**2. Connecting-road centrelines overlap at a junction entry, so a nearest-lane snap picks the
wrong branch.** On `7c72937c` junction 149 the snap flip-flopped over ten internal roads; the modal
one was **15 (STRAIGHT)** while the ego actually drove **13 → 12** and turned 163 deg. Resolve the
branch topologically (the connector whose link lands on the road the ego is on when it leaves),
with polyline coverage as the cross-check: the correct branch scores **1.00 at 0.66 m mean**, the
next best **0.41 at 4.31 m**. A sibling of the same trap: an incoming connector **1.02 m long**
(road 189, scene 00040136) never wins a snap at all and made the option count read **0**, i.e.
"no continuation exists".

---

## 2026-08-03 — "brake_stop 0.026 → 0.503 with no retrain" — the MAGNITUDE survives an honest τ; the CLAIM does not survive precision.

**Root-cause class: A ONE-SIDED METRIC ON A RULE WHOSE ENTIRE MECHANISM IS MOVING THE DECISION
BOUNDARY TOWARD THE RARE CLASS.** Not the class everyone expected. The stream flagged
*"reading τ off this table is fitting on val"* against its own work, and that was the right flag —
but it was **not the load-bearing defect**. Same family as C9/C13/C14: an instrument structurally
unable to report the answer it is cited for.

**What was published** (`…/incoming/2026-08-03-dtac1-tactical-head/DTAC1_RESULTS.md` §2.3, §0.5, §3):
prior-corrected decoding lifts `brake_stop` recall **0.026 → 0.503** at τ = 0.5 with no retrain, and
`accelerate` peaks at 0.153. Recommended as a free read-out patch.

**What the honest re-run says** (run directory `…/2026-08-03-dtac1-tactical-head/`, output
`dtac1_tau_selection_refc-base-30k.json`, `DTAC1B_RESULTS.md`; τ AND the class prior selected
**leave-one-episode-out**, so every window is decoded by a rule fitted without its own clip):

* **The τ-on-val part was cheap.** Modal out-of-fold τ under macro-F1 is **0.50 in 36/39 folds** —
  the same τ. Cost of honesty **1.49–3.16 % relative**, and the paired episode-cluster bootstrap on
  out-of-fold vs val-optimal **includes zero in all 8 comparisons**.
* **The precision part was not.** Honest brake recall is **0.4248** (all 1364 windows) at precision
  **0.1711** — 380 fires against 153 true. **Precision appears nowhere** in the parent's results file,
  its probe JSON, its pre-registration, or `refc_tactical_probe.py` (adversarial R3, confirmed).
* ⛔ **On the 1232 windows the 5-way LABEL can represent, the patch is NOT separated from doing
  nothing:** Δmacro-F1 **+0.0107 [−0.0418, +0.0665]**, Δmacro-recall +0.0213 [−0.0256, +0.0726].
  The visible gain lives on the 132 windows the label destroys — the set the same report calls
  irrecoverable.
* **Optimising the published metric is self-defeating.** Selecting τ by macro-recall gives
  Δmacro-recall **+0.1069 [+0.0381, +0.1749]** (separated) for Δmacro-F1 **−0.0006
  [−0.0922, +0.0788]** and a *separated* accuracy loss of **−0.2757 [−0.3851, −0.1745]**. The recall
  is bought with an exactly offsetting precision loss, and macro-recall cannot see it.
* **The stated chance floor was wrong.** The full selection pipeline on **shuffled** logits scores
  macro-recall **0.3678**, not the nominal 0.3333 — so ~0.034 of every macro-recall here is extracted
  from the class prior by the procedure itself. (Third time this class has appeared: adversarial R4
  found the same on E-A2.)

⇒ **RULE: a decision rule that works by shifting a threshold toward a rare class MUST report
precision, and must be scored on the denominator where the label can actually carry the answer.**
Recall alone is monotone in exactly the knob being tuned, so it cannot falsify the tuning.
⇒ **RULE: an out-of-fold protocol beats a promise.** "Thresholds were fixed in advance" is
unverifiable self-report (adversarial R11 made that point against the parent's own pre-registration,
whose mtime is *after* its probe JSON). A leave-one-episode-out selection is checkable from the code:
a fold mechanically cannot see itself.
⇒ **RULE: report the empirical null, not the nominal one.** Run the whole pipeline — selection
included — on shuffled inputs and quote *that* as the floor.

**What is NOT retracted:** the retrain justification. It never rested on brake reporting. It rests on
the **9.68 %** of windows whose longitudinal class the 5-way label destroys, on `accelerate`, and on
the `lon_to_anchor` selection graft — and the representable-denominator null above **strengthens** it,
because it shows no decode rule reaches what the label already threw away.

### Second defect closed the same day — the F1 lever was not independently testable

`tactical_speed_input` raised `ValueError` unless `factored_maneuver` was also on, so the **shipped**
5-way head could never read the ego speed and the pre-registered arm set contained **no F1-only arm**.
F1's only estimate would have been `dtac1-full − dtac1-f2only`, two arms that also differ in the head
itself. **Root-cause class: a guard justified by reproducibility that was already guaranteed by the
flag defaulting OFF, and that silently removed an ablation.** Decoupled; `refc_f1only_config()` is
**+384 params (+0.000369 %)**, MEASURED, with the decoder bit-identical.
⇒ **RULE: before coupling two gated levers, ask which ABLATION the coupling deletes.** A conservative
guard that makes an effect unattributable is not conservative.

---

## R-2026-08-03-h — "a 2,049-parameter RIDGE beats the 2.17 M-parameter head" — BOTH parameter counts belong to OTHER experiments

**Retracted claim** (`…/2026-08-03-sitclf-fusion-wired/SITCLF_VISION_ONLY.md` §4 and §6.4, and the
BACKLOG **B4** brief derived from it): *"on `roundabout` the 2,049-parameter ridge probe (0.01056)
beats the 2.17 M-parameter transformer head (0.00721) on the same vision features."*

**Root-cause class: C4 (inherited without re-verification) in its sharpest form —
a CROSS-STREAM NUMERIC TRANSPLANT.** Both figures are correct *somewhere*; neither describes the two
arms actually being compared. They were carried across from two sibling streams whose arms have
similar names.

| number | where it is TRUE | what it actually is |
|---|---|---|
| **2,049** | `…/2026-07-26-situation-semantics/SITUATION_SEMANTICS.md:197` | a linear ridge on the **raw frozen 2048-d state at t** (2048 + intercept), target `NOT_T_seen` |
| **2.17 M** | `…/2026-07-26-h2-classifier/H2_CLASSIFIER.md:621-626` | **H2's** head at **d=256** — `head_img|trigger` = **2,173,187** params, a different stream, target and substrate |

**MEASURED — what the two sitclf arms really are** (primary artifacts, read at HEAD):

| arm | parameters | source |
|---|---:|---|
| `ridge_img` | **129** (16 PCA dims x win 8 + intercept) | `…/2026-07-26-situation-classifier/artifacts/train_summary.json` -> `selected.ridge_img.n_params` |
| `head_img` | **417,028** (`in_dim` 16, `win` 8, `d` 128) | `…/checkpoints/head_img.pt`, summed `state_dict` |

**The qualitative finding SURVIVES and gets stronger, the arithmetic does not.** The gap is
**129 vs 417,028 = 3,233x**, not the 1,059x implied. What is retracted is every sentence that
attaches "2,049" or "2.17 M" to a sitclf arm, and any inference that used the raw-2048 input as if
`ridge_img` had consumed it — it did not; it consumed the SAME PCA-16 x 8-frame window as `head_img`
(`sc_train.py:349-358`, `R_PCA = 16`, `window_flat`).

**What it would have cost:** B4 was briefed to sweep "~2k to ~2.17M". Taken literally that ladder
starts ABOVE the arm the finding is about — the 129-parameter floor would never have been built, and
the experiment would have been unable to reproduce its own premise.

**Guarded in code, not in prose** (`stack/tests/test_sitclf.py`):
`test_head_param_count_reproduces_the_deployed_head_img_checkpoint` asserts **417,028** against the
checkpoint, and `test_ridge_param_count_reproduces_the_banked_ridge_img_figure` asserts **129** and
**2049** side by side so neither can be reintroduced silently.
`tanitad.eval.sitclf.head_param_count` counts by CONSTRUCTING the module — a closed-form formula is
exactly what would have reproduced the error.

⇒ **RULE: a parameter count is a property of an ARTIFACT, so quote it from the artifact.**
`sum(p.numel())` on the checkpoint, or the trainer's own `n_params` field — never from a neighbouring
report, however similar the arm name. Arm names (`head_img`, `ridge_img`) repeat across streams with
different architectures behind them; the name is not the specification.
⇒ **RULE: when a claim compares two arms, name the ARTIFACT PATH of each.** This claim named neither,
which is why two numbers from two other experiments could sit in it unchallenged.

---

## R-2026-08-03-i — the IDM `long_accel` MECHANISM ("5× error amplification") — the CONCLUSION survives, the EXPLANATION is retracted

**Retracted claim** (`stack/scripts/idm_head.py`, `derive_long_accel` docstring, and
`…/incoming/2026-08-03-idm-derived-accel/IDM_DERIVED_ACCEL.md`): *"at R² 0.72 the speed channel still
has MAE 4.04 m/s, and a 0.2 s centred difference of two such predictions carries ~5× that error
against a target whose own MAE is 0.45 m/s²."*

**Root-cause class: an UNMEASURED INTERMEDIATE STEP inside a correct conclusion.** The "5×" is
`1/(2·dt)`, the gain differencing applies to *white* noise. Nobody measured whether the speed error
IS white at that lag — and it is not. The verdict the argument supported happened to be right, which
is exactly why the wrong reason survived: a confirmed conclusion stops anyone auditing its premise.

**MEASURED (mine)** —
`…/incoming/2026-08-03-idm-accel-recoverability/raw/speed_error_mechanism.json`, best speed arm
(ridge on the flattened latent window, R² +0.7145, MAE 4.7186 m/s), 17 held-out episodes:

| quantity | value |
|---|---|
| autocorrelation of the held-out **speed error** at 0.2 / 0.4 / 0.8 / 1.6 s | **0.9265 / 0.8451 / 0.6719 / 0.4241** |
| ⇒ variance of the error that SURVIVES a 0.2 s difference | **~7 %** (differencing cancels ~93 %) |
| std of the true speed difference / the predicted difference / the CAN label | 0.568 / **4.787** / 0.587 m/s² |
| derived accel vs the CAN label | R² **−65.89 [−112.09, −45.17]** |
| ORACLE — true speed difference vs the CAN label | R² **+0.9198 [+0.8771, +0.9447]** |

⇒ the route fails not because differencing *amplifies* the error but because the target's dynamic
range (**0.587 m/s²**) is an order of magnitude below what survives the cancellation (**4.787 m/s²**).
The error budget makes the requirement explicit: the speed track needs **σ ≲ 0.1 m/s**
(derived R² 0.549 at σ=0.1, 0.828 at σ=0.05) — a **~47×** improvement over the measured 4.72 m/s MAE.

**A second, softer retraction in the same docstring.** *"The channel carries no recoverable
information from the frozen v1 latents at this scale, so no reparameterisation of the head can repair
it"* was stated as an absolute from a **single-arm** null. It is now **much better supported** — 17
latent-input arms (6 closed-form kernel ridge, linear+rbf on four feature bases over the whole
regularisation path; 7 neural — transformers d64/L1→d512/L6, MLPs, a bi-GRU, an accel-only head;
2 at a 2.5 s context; 2 stationary-filtered) all at or below the empirical null of −0.0626, with a
**decisive capacity control**: the identical closed-form protocol on the TRUE speed window reaches
**R² +0.9262 [+0.8876, +0.9507]**. But it is **bounded by power, not absolute**: a planted signal on
a random carrier at SNR ≈ 7 is detected at true R² 0.3 and **missed at 0.1**, and on a high-variance
carrier a true R² of 0.6 hides. The admissible form of the claim carries that floor.

⇒ **RULE: when an argument has a mechanism, the mechanism is a CLAIM and needs its own evidence
class.** "Differencing amplifies error by 5×" is a measurable statement about the error's
autocorrelation; it was asserted from the arithmetic of `1/(2·dt)` alone.
⇒ **RULE: a NULL is only admissible with a DETECTION FLOOR.** "X is not recoverable" and "X is not
recoverable above strength S at this n" are different claims, and only the second is falsifiable.
The floor costs one extra arm — the same probe run on a latent with a KNOWN signal planted in it.

---

## R-2026-08-03-j — "ROLLING SHUTTER is the biggest render-quality lever" — the NUMBER stands, the CAUSE is RETRACTED

**Retracted claim** (`stack/experiments/alpasim-gsplat/RENDER_QUALITY.md`, "WHAT WORKED, IN ORDER OF
SIZE / 1. ⭐ Rolling shutter — biggest lever, 161× the cost"): *"All three move the right way at once,
and coverage rises 21 % — **the scanline sweep fills pixels a single pose leaves thin**"*, headlined as
**+35.1 % grad-NCC**.

**The measurement is NOT retracted.** Enabling gsplat's rolling shutter really does raise grad-NCC and
`mean_alpha`, and it really does cost ~two orders of magnitude. **What is retracted is the attribution
of that gain to the shutter.**

**Root-cause class: ATTRIBUTING AN EFFECT TO THE FEATURE THAT WAS TOGGLED, RATHER THAN TO THE CODE
PATH THE TOGGLE SWITCHED ON.** `rolling_shutter=…` + `viewmats_rs=…` does two things at once — it
sweeps the camera pose, *and* it enters a different projection branch. Only the first was in anyone's
head, so no arm was ever run that had one without the other. This is the same shape as the
`--v2` conflation in R-2026-07-3x: a single flag moving two things, and the gain booked to the
interesting one.

**MEASURED (mine)** — `~/rq_out/rs_sweep_chosen/report.json`, `~/rq_out/rs_cost_probe.json`, deployed
config, 12 frames over the 599-frame clip, grad-NCC with a paired frame bootstrap; repo copies under
`stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter/`:

| control | result | what it rules out |
|---|---|---|
| sweep run **BACKWARDS** (`native_swapped`) | **+0.0216 [+0.0131,+0.0312]** vs native's **+0.0210 [+0.0096,+0.0365]**; images **2.264/255** apart — closer to native than any other arm by 2.7x | a readout-motion correction **cannot be invariant to the readout direction** |
| the pose sweep done FAITHFULLY, 2→64 slices | monotonically **worse**: `s4` +0.0048 → `s16` +0.0004 → **`s64` −0.0095 [−0.0129,−0.0055]** | the sweep itself is worth **nothing**; the best sliced arm ties a **free** single-pose render |
| per-band Δ | native's biggest gains are at the frame **BOTTOM** (+0.0667, +0.0755) — where a TOP_TO_BOTTOM sweep and the shutter-END baseline are **the same camera** | a pose effect cannot appear where there is no pose difference (holds under either row→time convention) |
| gaussians surviving projection | production **759,404 / 614,538**; native RS **1,341,915 / 1,096,693** = **+77 % / +78 %** | a sweep **moves** geometry, it cannot **create coverage** |
| RS kernel at **ZERO motion** | **614,538 vs 614,538** — the same integer as production | the code path alone is not it either; the two must BOTH differ |
| `require_all_sigma_points_valid=False` on a plain **global** render | **+50 % / +52 %** more gaussians, no geometry change | ~two thirds of the effect is available **free** |
| `in_image_margin_factor` 0.1 → 2.0 | **+1 and +3 gaussians** | the obvious second candidate, dead |

**The actual mechanism, read from `gsplat/cuda/include/Cameras.cuh:357` and then COUNTED:**
`world_point_to_image_point_shutter_pose` returns the real `valid_start` on the GLOBAL branch but a
hard-coded `true` on the ROLLING branch; upstream, `require_all_sigma_points_valid` (default `True`,
and `gsplat.rasterization()` never exposes `ut_params`) culls any gaussian with a single invalid sigma
point. So **rolling shutter silently disables the cull.** The zero-motion control confirms the
predicted boundary exactly: with `q_start == q_end` an invalid point is invalid at both ends, the
function early-returns `false`, and the count matches production **to the unit**.

**A second, smaller retraction in the same section.** *"Off by default; `--rolling-shutter`"* was
presented with `render_probe.py --rs` as a working alternative. It never ran: `render_probe.py:204`
called `RollingShutterType.TOP_TO_BOTTOM`, which does not exist in gsplat 1.5.3 (`AttributeError`,
verified by reading the file **and** by querying the installed enum). Fixed — the member name is now
read from the calibration string.

⇒ **RULE: when a flag switches on a CODE PATH as well as a PHYSICAL EFFECT, the null arm is the same
code path with the physical effect set to ZERO.** Here that is one line (`viewmats_rs = viewmats`),
and it would have caught this on day one.
⇒ **RULE: a symmetry the claimed cause forbids is the cheapest falsifier available.** Rolling shutter
is directional; running it backwards costs one arm and it decided the whole question.
⇒ **RULE: `mean_alpha` (coverage) and grad-NCC (structure) moving TOGETHER is not "all three move the
right way at once" — it is a WARNING.** A geometric correction redistributes coverage; only a change
in *which primitives are drawn* raises it. "Everything improved" should prompt "what else did I
change?", not confidence.

---

## R-2026-08-03-C — "the closed-loop separation is ENTIRELY LATERAL, and ADE would have said no difference"

**What was asserted**, this morning, in the videos README, `README.md`, `PROGRAM_OVERVIEW.md` (C1) and
`Paper/TANITAD_PAPER.md` (§5.0.1 and the abstract-level summary): *REF-C base beats flagship v1
closed-loop on the NuRec reconstruction and the separation is **entirely lateral** —
`dist_to_gt` +1.171 [0.030, 2.244], heading +0.084, curvature +0.0050, yaw-rate +0.038 all separated,
**ADE +0.789 [−0.865, +2.728] NOT separated** — so an ADE-only table would have reported "no
difference".* It was quoted as the four-family doctrine's strongest single piece of evidence.

**What survives.** REF-C still beats flagship v1, and **every one of the four lateral separations
holds and widens.** That half is confirmed on a better render.

**What is RETRACTED: "entirely lateral", and with it "ADE would have reported no difference".** Re-run
on the shipped +23.4 % grad-NCC render (run dir `thor:~/cl_out_hq`, artifacts
`stack/experiments/alpasim-gsplat/results/closedloop-hq-render/`), on the same 437 paired windows,
same starts, same checkpoints, same scorer:

| | morning render | shipped render |
|---|---|---|
| `ade_0_2s` | +0.789 [−0.865, +2.728] **not sep** | **+7.164 [+5.265, +8.966] separated** |
| `abs_target_speed_err_ms` | +1.124 [−0.101, +2.566] not sep | **+6.397 [+5.000, +7.801] separated** |
| `along_track_ade_m` | +0.650 [−1.017, +2.590] not sep | **+7.153 [+5.240, +8.953] separated** |
| `route_corridor_departure_rate` | +0.204 [−0.002, +0.398] not sep | **+0.506 [+0.382, +0.629] separated** |

**Root-cause class: PUBLISHING A CLOSED-LOOP RESULT WITHOUT MEASURING THE POLICY'S SENSITIVITY TO THE
SIMULATOR IT WAS MEASURED IN.** The panel was treated as *an arm property observed through a
renderer*. It is a **joint** property of the arm and the renderer, and for one of the two arms the
renderer term dominates. flagship v1's driven path moves a **mean 9.05 m (max 37.78 m)** and its
commanded speed drops **12.96 → 7.05 m/s** under a render change that moves REF-C **0.43 m** and
**0.13 m/s** — a **21×** sensitivity ratio. At `k=0`, with zero accumulated drift, flagship's plan
already moves up to **9.09 m** from the render alone. Every "closed-loop ADE" for this arm was a
measurement of the pair, reported as a measurement of the arm.

**A second, smaller retraction in the same block.** The morning table listed `dist_to_gt_traj` (under
ADE) *and* cross-track (under LATERAL) as separate separated metrics. They are the **same number**:
`cl_metrics.py` builds them in one dict literal, `"cross_track": ct, "dist_to_gt": abs(ct)`. Verified
identical on 6/6 rollout sets, max |Δ| exactly 0. Four separated lateral metrics, not five.

**Why the attribution is admissible — the control was exactly zero.** Re-running the MORNING config
today reproduced the morning rollouts **bit-exactly**: 0.0 m driven path, 0.0 m plan, 0.0 on all 19
paired metrics, both arms, 450/450 windows (`CONTROL_<arm>_repro_vs_morning.json`). Every change is
the render. Each was additionally tested as a **difference-in-differences** on identical windows
rather than by comparing two CIs by eye; 9 of 10 separate.

**And the feature is identified.** A 2×2 over the two `empty`-road render changes: the **scale cull**
carries it (flagship ADE **+4.489 [+3.146, +5.999]**), the gated sky is null-to-helpful
(**−0.457 [−0.999, +0.118]**). The change that most improved fidelity to the reference is the one
that breaks flagship's speed control.

⇒ **RULE: a closed-loop number is not admissible without a render-perturbation sensitivity measured
beside it.** An arm whose plan moves 9 m under a fidelity improvement cannot be compared at 0.1 m,
and the sensitivity is itself the more transferable quantity.
⇒ **RULE: before attributing anything to a changed simulator, re-run the OLD simulator config and
require the old rollouts back.** Here it returned exactly 0.0 and licensed everything; on the
`objects` condition the same control **failed** (flagship 1.536 m mean / 7.266 m max, 18 of 19 paired
deltas moved) because an unrelated same-day code line only executes with actors attached — so no
`objects` morning-vs-HQ number was publishable, and that was caught by the control rather than by
review.
⇒ **RULE: "metric X did not separate" is a statement about the measurement conditions, not about the
metric.** ADE was blind here because the *renderer* had suppressed the effect that makes flagship
fail, not because ADE cannot see longitudinal failure. The four-family doctrine is untouched — the
argument for it is that ADE *can* hide the gap, and it did; what is retracted is this panel's use as
its showcase example.

---

## R-2026-08-03-k — every render-fidelity number on NuRec scene `00040136` was scored against a reference **6 FRAMES TOO EARLY**

**Retracted:** not one claim but a **class** — every **absolute** grad-NCC / MAE / PSNR ever quoted for
this scene. That includes `FINDINGS.md`'s original decode validation (*"correct frame 0.3802 vs best
wrong 0.2110"*), `RENDER_QUALITY.md`'s **0.2774 → 0.3424 "+23.4 %"** and **0.3747 "+35.1 %"** headlines,
panels `panel1`…`panel6_chosen`, and **every absolute number in my own
`…/results/2026-08-03-rolling-shutter/ROLLING_SHUTTER.md`.**

**Root-cause class: A NEGATIVE CONTROL WITH A BLIND SPOT BUILT INTO ITS OWN CONSTRUCTION — and nobody
checked the neighbourhood it excluded.** `render_quality.wrong_frames_for()` requires wrong candidates
to sit **`MIN_WRONG_GAP = 40`** frames away, with the comment *"a 'wrong' frame 5 frames away is nearly
the correct one"*. That is correct reasoning for the question it was built for ("is our decode real?")
and it makes the control **structurally incapable** of seeing a small index error. The control was
never wrong; it was answering a different question, and its passing was read as alignment.

**MEASURED (mine)** — `~/rq_out/rs_frame_offset_k10.json`, `~/rq_out/mp4_frame_count.json`; render held
at PRODUCTION settings (shutter-END pose, actors at shutter-END time), **only the reference index
varies**; 12 frames spread over the clip:

| reference offset | −2 | 0 | +2 | +4 | +5 | **+6** | +7 | +8 | +10 |
|---|---|---|---|---|---|---|---|---|---|
| mean grad-NCC | 0.2963 | **0.3114** | 0.3363 | 0.3806 | 0.4213 | **0.4911** | 0.4661 | 0.4077 | 0.3478 |

* **`argmax_histogram = {6: 12}`** — every frame, no ties, and the curve **turns over** at +7, so it is
  a maximum and not a scan boundary. *(An earlier ±3 scan of mine stopped at its edge still rising and
  reported "≥ +3"; it is superseded, and it is kept in the run dir precisely because reporting a
  boundary as an answer is the failure this entry is about.)*
* **+0.1797 grad-NCC, +57.7 %, free.** For scale, the rolling shutter of R-2026-08-03-j buys **+0.0210
  at ~90× the render cost**, and the honest pose-sweep effect is **+0.003–0.005**. The misalignment is
  **8.6×** the first and **~40×** the second.
* **Independent corroboration, same integer:** a **full sequential decode** (not the metadata estimate,
  though both agree) gives the mp4 **605** frames against the rig's **599** — **Δ = 6**.

**What survives and what does not.** Every arm in every panel shared the same wrong reference, so
**PAIRED DELTAS BETWEEN ARMS SURVIVE** — the rolling-shutter verdict, the layer/cull/sky A/Bs and the
"which metric may decide" analysis all stand. **ABSOLUTE values do not**, and the direction is
flattering-in-reverse: **the renderer is materially better than anyone measured.**

⚠️ **NOT established, and must not be "fixed" before it is:** WHICH side is off. Six extra frames in
the mp4 is consistent with leader frames, but I did not verify where they sit and I checked **one
scene**. `stack/experiments/alpasim-gsplat/rs_frame_offset.py` is the instrument; ~2 min/scene.

⇒ **RULE: a negative control certifies only the discrimination it was built to test.** This one proved
"our render matches THIS clip rather than a different part of it". It never claimed "…and the index is
right", and it was read as if it had. **State what a control excludes, next to what it includes.**
⇒ **RULE: before trusting any reference-based fidelity metric, scan the IMMEDIATE NEIGHBOURS.** The
hard negatives for an alignment error are `f±1, f±2, …`, i.e. exactly the frames a coarse control
deliberately excludes. It costs one render per frame.
⇒ **RULE: when two counts that should match do not (`605` vs `599`), that is a finding, not noise.**
The discrepancy was visible in `CAP_PROP_FRAME_COUNT` from the first day anyone opened the mp4.

---

## R-2026-08-03-l — flagship-v1's STRATEGIC route accuracy is the ECHO of an ORACLE INPUT, not a decision

> ⚠️ **ID corrected 2026-08-03 by the adversarial-verification pass.** This entry was appended as
> `R-2026-08-03-j`, which was **already taken** by the rolling-shutter retraction at the top of this
> file (and cited from `stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter/
> ROLLING_SHUTTER.md:704` and from R-2026-08-03-k). `-k` was taken too, so this entry is now `-l`.
> **The content below is unchanged and was independently reproduced** (see the verification note at
> the end of this entry). An append-only log whose standing rule is *"must be read before asserting
> in a known class"* cannot carry two entries under one citable identifier — check the last used
> letter before appending.

**Retracted claim.** `stack/experiments/nurec-gsplat/STRATEGIC_FAMILY.md` §(b) and
`results/closedloop_strategic_7c72937c.json`, carried into task #51's report: *"flagship-v1 /
empty **route_class_accuracy = 1.0000 (6/6)**, flagship-v1 / objects **1.0000 (6/6)**; paired
flagship − refc = **+1.000** (empty), **+0.800** (objects)"* — the option-set STRATEGIC family's
first numbers on a real branch scene.

**Root-cause class: A MODEL SCORED ON A TARGET IT WAS HANDED AS AN INPUT** — a *conditioning
echo*. New sub-class of the C6 family, and the one the existing guards were structurally unable to
catch. `discrimination_control` proves the **labels** carry entropy (its ORACLE copies
`route_gt_class`, a tautology). `BEST_CONSTANT` catches a head that always answers one class.
**Neither can catch an echo, because an echo is not constant and beats every constant.**

The harness derives the nav command from the ego's own logged future
(`closedloop_drive.py:348 nav_from_route` → `refb_labels.nav_command_v21`) and **feeds it to the
policy**; the flagship's `StrategicPolicy` then FiLM-conditions *every* causal block on
`nav_emb(nav_cmd)` and reads `route_head` off that stack (`models/fourbrain.py:58, 77-86`), while
its auxiliary route CE target (`route_target_v21`) is derived from **the same GT future**. The
shortcut — copy the FiLM condition — exists by construction.

**MEASURED (mine)** — run dir `stack/experiments/nurec-gsplat/results/2026-08-03-strategic-T1/`,
open-loop on the logged clipgt track with the real 4K reference camera, **116 map-derived decision
events over 77 NuRec T1 branch scenes, 4 745 poses**, episode-cluster bootstrap, cluster = scene.
The identifying move is a **MANIPULATION**: sweep the nav vocabulary with the **pixels held fixed**.

| quantity | flagship-v1 | refc-base |
|---|---|---|
| `nav_passthrough_rate` (argmax moves when only nav moves) | **1.0000** (n=4745) | **0.0000** (n=4745) |
| argmax under nav=`follow` / `left` / `right` | STRAIGHT 4745/4745 · LEFT 4745/4745 · RIGHT 4745/4745 | unchanged under all 4 navs |
| logit std across NAVS at a fixed pose | 9.65 (nav-to-image ratio **5.19**) | **exactly 0.0** (`HEAD_IS_NAV_BLIND`) |
| route_class_accuracy @ **navORACLE** | 0.8707 [0.8053, 0.9298] | 0.6983 |
| route_class_accuracy @ **navFOLLOW** (deployable; `follow` is **~75-79 % of training windows** and the standard eval value, `refs/refc.py:66-68`) | **0.1983 [0.1240, 0.2727]** — **BELOW** the best constant, separated | **0.6983 [0.6179, 0.7810]** — above it, separated |
| **navORACLE − NAV_ECHO** (a lookup table with **no image at all**) | **+0.0000 [+0.0000, +0.0000]**, not separated | −0.1724 |

⇒ under the oracle nav the flagship's route head is **indistinguishable from a nav lookup table on
every one of the 116 events**, and `NAV_ECHO` scores the identical 0.8707 with no model. Strip the
oracle and the arm falls to 0.1983, predicting STRAIGHT on 110 of 116 events (**precision 0.2091**,
zero LEFT and zero RIGHT emitted, `prediction_degenerate = true`) and naming a manoeuvre the map
does not admit on **34** of them. Head-to-head at the deployable setting **flagship − refc =
−0.5000 [−0.6053, −0.4017]**, separated, and **−0.5254 [−0.6897, −0.3823]** on the 39-scene
leak-free subset.

**Confirmed three independent ways** — this manipulation; a **closed-loop** observational check by
a sibling stream the same day (`strategic_conditioning_control.py`: flagship head an exact
bijection of nav, 369/369 and 81/81 over 450 ticks); and **source** (`fourbrain.py:77-86` vs
`refc.py:1130/1137/1140`, where REF-C's `route_head(pooled)` reads image features *before* the nav
one-hot is fused, i.e. nav-blind **by architecture**).

⇒ **RULE: before scoring a head, ask what was FED to it.** A metric is only a measurement of the
model if the answer is not already in the model's input. Enumerate every conditioning channel the
harness supplies from ground truth, and for each one ask whether the scored target is a function
of it.
⇒ **RULE: a degeneracy control must be run against the ARM, not only against the LABELS.** The
label-side control (ORACLE vs BEST_CONSTANT) passed on this scene set at
**+0.5641 [0.4601, 0.6667]** — the labels were fine. It was never capable of saying anything about
an arm, because its ORACLE is built by copying the label.
⇒ **RULE: identify a conditioning echo by MANIPULATION, never by an observational contingency
table.** A competent head and an echo agree whenever the command is correct; only holding the
observation fixed while moving the input separates them. Cost: one extra forward pass per
conditioning value.
⇒ **RULE: a PERMUTATION of the conditioning is not a substitute for a SWEEP.** MEASURED here: 50
of the 78 T1 branch scenes carry exactly ONE decision event, so a within-scene shuffle is the
identity — the control returns clean output and measures nothing.

**Now enforced in code, not in prose**: `taniteval.strategic_optionset.conditioning_echo_control`
+ `strategic_family(..., conditioning_sweeps=…)` → `STRATEGIC_SKILL_ADMISSIBLE`, threaded through
`four_families.strategic`. **With no sweep supplied the verdict is `None` (UNTESTED), never a
pass.** Regression tests: `test_an_echo_arm_beats_every_constant_yet_is_INADMISSIBLE`,
`test_echo_control_refuses_a_sweep_that_cannot_separate_anything`, +5 more in
`taniteval/tests/test_strategic_optionset.py`.

**INDEPENDENTLY REPRODUCED 2026-08-03 (adversarial-verification pass).** `aggregate_t1_strategic.py`
re-run on the banked `t1_route_ticks.json.gz` + `strategic_gt_t1.tar.gz` reproduces **every number in
the table above bit-for-bit**, on both the 77-scene and the 39-scene leak-free set. Passthrough
1.0000/0.0000 re-derived from the raw logits (4745 poses); `refc-base` logits **bit-identical across
all 4 navs**. Suites re-run green (stack 1851 / taniteval 903). **Three corrections that do NOT touch
the verdict but do touch what may be quoted from it:**

1. ⛔ **The mechanism is STRONGER than stated here.** `refb_labels.py:715` (`nav_command_v21`) and
   `:730` (`route_target_v21`) call the **identical** `route_from_future_v21(poses, t, horizon)` with
   identical arguments, and `:455 _ROUTE_TO_NAV` is a bijection on {LEFT, STRAIGHT, RIGHT}. The aux
   route CE target is not merely *"derived from the same GT future"* as the FiLM condition — on every
   `valid` window it **is a relabelling of it**. The CE is 100 % solvable from `nav` alone.
2. ⚠️ **The head-to-head is condition-dependent and the omitted contrasts flip sign.** Nine paired
   contrasts were computed and four were reported. Also separated: `flagship − refc @ navORACLE =`
   **+0.1724 [+0.0991, +0.2520]** (the flagship *wins*), `refc navORACLE − NAV_ECHO =` **−0.1724
   [−0.2520, −0.0991]** (the image-free lookup table *beats* REF-C), and `flagship − refc @
   navSHUFFLED =` **−0.3103 [−0.4202, −0.2000]**. The **direction** survives at navFOLLOW *and*
   navSHUFFLED; **−0.5000 is the largest of three defensible magnitudes**, and on these T1 turn
   scenes the true nav is left/right at **4265/4745 = 89.9 %** of scored poses, so navFOLLOW feeds
   the FiLM-conditioned arm a *wrong* command almost everywhere. Quote **navSHUFFLED (−0.3103)** when
   the claim must not depend on calling navFOLLOW "deployable".
3. ⚠️ **`nav_passthrough_rate = 1.0000` is a statement about the ARGMAX under 3 of 4 nav values.**
   At `nav=3` the flagship is **not** a lookup table (LEFT 288 / STRAIGHT 44 / RIGHT 4413), because
   `_ROUTE_TO_NAV` never emits `NAV_STRAIGHT=3` — that embedding row is **untrained**, making
   `flagship/navSTRAIGHT` an out-of-vocabulary probe rather than a condition. Harmless for the
   verdict (`nav_oracle ∈ {0,1,2}`, verified over all 4745 poses), but `flagship/navSTRAIGHT 0.3879`
   must not be read as a result.

---

## R-2026-08-03-nav — ⛔ "the closed-loop rollouts fed nav=0 everywhere because `nav_command_v21`
## needs 25 s of lookahead and the scenes are 20 s" is REFUTED

**Retracted claim (mine, stated to the PI):** *the scenes are ~20 s, `NAV_HORIZON_STEPS = 250`
(25 s), therefore the route label can never be judged and every tick was fed `NAV_FOLLOW` — the
model never received a turn command.*

**MEASURED 2026-08-03**, re-derived from the banked rollouts
(`stack/experiments/alpasim-gsplat/results/openloop-thor-2026-08-03/rollouts/`), all four arms
(flagship-v1 × REF-C-base × empty × objects), 190 scored ticks each:

| scene | frames | nav actually fed | `nav_valid` |
|---|---|---|---|
| **junction 7c72937c** | 199 (19.9 s) | **NAV_LEFT on 121 / 190**, FOLLOW on 69 | **161 / 190** |
| 00040136 (night) | 199 (19.9 s) | FOLLOW on 190 / 190 | 65 / 190 |

`route_from_future_v21` **clamps the horizon to the available future** — on the 199-frame junction
scene it integrates `arc = 125.1 m` and returns a turn, `valid=True`, at t = 0/5/10. The 25 s
default is a *cap*, not a *requirement*. Identical for all four arms, so this is a property of the
label function, not of an arm.

⇒ **The consequence is the opposite of what I reported.** On the junction scene the models *were*
given the correct turn command on 121 of 190 ticks and still tracked the road. **That makes the
missed exit a MODEL failure, not a nav-plumbing failure** — and it is therefore admissible evidence
about the strategic level, which the plumbing story would have thrown away.

**Root-cause class: I read the MECHANISM off a function's DEFAULT PARAMETER instead of measuring
its BEHAVIOUR on the actual data.** `NAV_HORIZON_STEPS = 250` and `T = 199` is a true pair of facts
that supports a false conclusion; one call on the real poses refutes it. Same family as
"a mechanism that is real in the source is not thereby the binding constraint" (R-2026-08-03-dtac1),
one level earlier: here the mechanism was not even real.
⇒ **RULE: never infer what a label function did from its signature. Read the value it actually
emitted — the rollouts record `nav` and `nav_valid` per tick precisely so this is a lookup, not a
derivation.**

### What survives, and is the real defect

⚠️ **`nav_valid` is 65/190 on the night scene and 161/190 on the junction scene, and the model
cannot see that bit.** `nav_command_v21` collapses `ROUTE_UNKNOWN` and `ROUTE_STRAIGHT` onto the
**same `NAV_FOLLOW`** token, so *"the road goes straight"* and *"I could not judge the route"* are
byte-identical at the model input. On the night scene **125 of 190 `FOLLOW` tokens are the
UNKNOWN sentinel**, not a route statement. Corroborated on the corpus at scale: of 3,179 windows
fed `follow`, **1,985 (62.4 %) are a collapsed UNKNOWN**
(`…/incoming/2026-08-03-refc-corpus-and-labels/`).
⇒ The fix is `nav_input_v22`'s **`(cmd, known)` pair** — already implemented in
`stack/scripts/refb_labels.py`, not yet wired into any trainer or driver.

### Second defect, separately confirmed here

`closedloop_drive.py:368` and `score_t1_strategic.py:392` pass `min_steps=10` to a v2.1 signature
that had dropped the parameter; a bare `except` swallowed the `TypeError`, so **the
scene-length-adapted short-horizon nav had never once executed** — every banked rollout row carries
`nav_short_err: TypeError(...)`. Fixed at HEAD (`refb_labels.py`, 18 tests in
`stack/tests/test_label_causality_and_nav.py`); re-verified live 2026-08-03: the call now returns
`valid=True`. Re-running it over the banked poses changes the short-horizon value on **7 of 22**
ticks of the night scene and correctly **declines to judge** on the junction scene (the 6 s window
cannot see a 125 m arc).
⇒ **RULE: a bare `except` around a label call converts a signature breakage into a plausible
default. Record the exception in the artifact — that is the only reason this was findable.**

---

## R-2026-08-03-v5f — ⛔ the "v5f IS GOING THE WRONG WAY" alarm is WITHDRAWN

**Retracted claim (mine, headlined in the 13:00 program report):** *v5f is degrading — it sat around
0.31 at steps 1800–2000 and is now 0.64–1.02; a decision is needed at the 5 k milestone.*

**MEASURED 2026-08-03T18:57Z**, `tanitad-new:/workspace/experiments/flagship-v5f-w120-30k/train_log.jsonl`,
53 metric rows, **500-step block medians** (n ≥ 4 per block):

| block | 1000–1500 | 1500–2000 | **2000–2500** | 2500–3000 | 3000–3500 | 3500–4000 |
|---|---|---|---|---|---|---|
| `g_op_fwd_ade_m` | 0.3522 | 0.2933 | **0.4191** | 0.1784 | 0.2389 | 0.1893 |

The alarm was the **2000–2500 bump**. The run has since printed values **better than anything before
it**. The HYPOTHESIS filed next to the alarm — LR warm-up under the changed `--batch 4 --accum 16`
regime — is what the data supports. **No restart.** Full row now at `MODEL_REGISTRY.md` §1.8.

**Root-cause class: a 3-point read inside an LR warm-up is not a trend.** This is the *fourth*
`g_op_fwd_ade_m` misread this programme, and the second in the "raised an alarm" direction (the
others read a lucky batch as a 73 % drop). The existing rule — *read ≥3 logged steps* — was
**followed and was still insufficient**, because 3 consecutive logged points span only 150 steps.
⇒ **STRENGTHENED RULE: on this metric, quote a BLOCK MEDIAN over ≥500 steps with its n, or do not
raise it. Never compare two blocks that straddle a warm-up boundary.**
⇒ **RULE: an alarm and its own hedge must be resolved by the same instrument that raised it.** The
13:00 report hedged correctly and recommended holding; the hedge is what saved the run, and it only
worked because the alarm was written with the counter-hypothesis attached.

### What the same probe found instead — and it is the more important result

`oracle_ade` improves monotonically after the bump (0.9450 → 0.5902 → 0.5663 → **0.5254**) while
`sel_gap = plan_ade − oracle_ade` **does not close at all** (0.4510 / 0.4878 / 0.5681 / 0.3980 /
0.3432 / 0.4715 over 2,650 steps), `rank_acc` sits at **0.000–0.375**, and
`frac_sel_2x_worse_than_oracle` at **0.25–0.50**. At step 3,650: `plan_ade` **1.0251** vs
`oracle_ade` **0.5254**.
⇒ **The fan is good and the SELECTOR is the defect — the arm would be ~2× better if it merely chose
correctly among candidates it already generates.** ⛔ Invisible in `g_op_fwd_ade_m`, which is why an
ADE-only read of this run reports "healthy". Third independent instrument now pointing at selection
rather than generation (with D-TAC1's within-`lane_keep` finding and the closed-loop TACTICAL row).
⚠️ Trainer-log numbers: a curve watch, **not quotable as a result**. The 5 k gate is adjudicated on
`stack/scripts/run_gate.py` over the four families with the paired episode-cluster bootstrap.

---

## R-2026-08-03-latent — ⛔ "our models keep only the LAST frame, so the cross-attended tokens are SINGLE-INSTANT and a single RGB frame cannot carry velocity" — the PREMISE is FALSE, and the CONCLUSION it supported is separately refuted

**Retracted** 2026-08-03. **Class C3 (mechanism instead of measurement)**, compounded by
**C4 (inherited without re-verification)**.

**What was asserted**, and used to brief two parallel streams as the load-bearing mechanism behind
the `long_accel` null, the sitclf capacity curve and the 88.7 % longitudinal gap:

> the model computes feature maps for all W frames and KEEPS ONLY THE LAST … the 64 tokens the
> anchor queries cross-attend are single-instant … a single RGB frame cannot carry relative
> velocity, closing rate, or TTC.

**The `[:, -1]` read is CORRECT** (`stack/tanitad/refs/refc.py:1683`). **Everything drawn from it is
not**, on three independent measurements:

1. **The kept tensor is NOT a single RGB frame.** `refc.py:241` — `in_channels: int = 9`,
   *"D-015 3-frame RGB stack (latest = `[-3:]`)"* — and the stack is **sliding**:
   `frames_u8[t][6:9] == frames_u8[t+1][3:6]` at **max |d| = 0.0**. One model "frame" already spans
   **300 ms**. Corroborated independently by the sitclf-temporal stream on a different corpus and a
   different file (`config.py:17`, `:360`; PhysicalAI cache `[199, 9, 256, 256]`).
2. **Keeping the other W−1 frames would not help.** `v1_window` — **all nine** latents, 18,432
   features — is **at the null** on `long_accel` (−0.0626, Δ vs its shuffled control +0.0000) while
   separating on `speed` (+0.7145, Δ +0.7197\*) in the same draw. The latent's frame-to-frame
   jitter along its own speed direction is **51.0×** the physical signal and correlates **+0.0061**
   with it.
3. **The channel is not recoverable from the video at all, at this n.** Pre-registered probe
   (`Project Steering/PREREG_TEMPORAL_LATENT.md`), **35 arms** over four substrates (frozen v1
   latent, raw frames, the D-015 sub-frame stack, full-resolution motion energy), linear and rbf,
   8–18,432 features: **zero** separate positive on `long_accel`; **six** separate positive on
   `speed`; the oracle (true speed window, 9 features) reaches **+0.9262**.

**And the finding that replaces it:** a **single static 32×32 grayscale frame** reads `speed` at
**R² +0.6642 [separated]** — **93 %** of the full 800 ms learned latent's +0.7145, and **1.75×** the
best motion-only arm in the panel (+0.3778). All **ten** LINEAR pure-difference arms sit at exactly
the null (−0.0052); their rbf counterparts reach +0.1449…+0.3778. ⇒ **on this corpus appearance
DOMINATES motion for reading speed**, and nothing in the pipeline was ever forced to learn motion.
⚠️ MEASURED on comma2k19 highway only — the magnitude elsewhere is UNKNOWN and is the top-ranked
follow-up.

**Root-cause class C3, in its most expensive form: a correct line of code was read, and a physical
consequence was inferred from it without measuring the tensor's shape or the channel's
recoverability.** The inference was reasonable, it was repeated in two briefs, and it would have
funded an architecture change (keep W frames — MEASURED cost: decoder MACs ×1.49 on REF-C-XL, peak
memory ×1.004, +4,097 params) that the data says cannot work for the channel it was proposed for.

⇒ **STRENGTHENED RULE: an architectural claim about WHAT A TENSOR CONTAINS must cite a measured
SHAPE and a measured RECOVERABILITY, not a line of code.** `in_channels=9` is four characters away
from the line everyone read. Related to **C15** (semantics from a name) — here the semantics were
taken from an *indexing expression* instead.

Evidence: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-latent-bottleneck/`
(`LATENT_BOTTLENECK.md`, `results_mechanism.json`, `results_temporal_falsifier.json`,
`results_precision_ladder.json`, `raw/temporal_kv_cost.json`).

---

## R-2026-08-03-hf — ⛔ "HF pulls at 93 MB/s" is RETRACTED (real: 23 MB/s, 4× slower)

**Retracted claim (mine).** The 2026-08-03 06:30 program report published a migration table with
`HF → new pod, 106 GB … 93 MB/s`, and I re-used 93 MB/s in agent briefs the same day as the number
to size transfer plans against.

**MEASURED 2026-08-03** on the Thor parity-corpus pull: the sustained rate is **23 MB/s**. Any plan
sized on 93 MB/s is wrong by **~2.5 h on a 278 GB corpus**. Upload (368–377 MB/s) is unaffected —
this retraction is about the **download** leg only.

**Root-cause class: a single leg of one transfer, measured once, promoted to a constant.** 93 MB/s
was real for that pod, that day, that file mix; it was never re-measured and it became a planning
input. Same family as *"the trainer's 13 s/step"*, which was the **cumulative mean** rather than the
marginal rate, and as the `225 ms/frame` render figure that turned out to be a **first-call** number.
⇒ **RULE: a throughput figure is only quotable with its date, its direction, and its endpoints.
Re-measure before sizing anything on it.** Upload and download are different numbers; a mean over a
run and its current marginal rate are different numbers.

⭐ **The important finding underneath it.** Before the pull, the raw parity corpus was probed for and
**not found on any live machine**: pod1/pod3/eval all `Connection refused`; pod4 and `tanitad-new`
hold no raw epcache (three probes each). It survives **only** on HF
(`Sayood/tanitad-physicalai-w120-256x640cyl` → `epcache-256px-phase0/`), where strict parity does
pass on the listing — uid `sha256 9877bef6…7386`, **2376/2376**, all 24 skip indices.
⇒ ⛔ **HF is currently the ONLY copy of the raw parity corpus**, and the E-SEL stream independently
escalated that the 256 px REF-C val raster it evaluates on has **one reachable copy**. Two streams,
two artifacts, same failure mode, found the same day. **Corpus durability is now the top
non-scientific risk in the programme.**

---

## R-2026-08-03-hor — ⛔ `--heldout-off-reason` (MINE, shipped today) had THREE latent defects

**What I claimed** when I added it earlier this session: a required reason string that is recorded
with the run, deliberately **not** a bare `--force` boolean, so an operator must state intent.

**What was actually true**, found only when another agent mirrored the flag's shape to build
`--parity-off-reason` and tested the mirror properly:

1. **Whitespace unlocked the guard** — a reason of `" "` satisfied the required-reason check, so the
   flag degraded to exactly the `--force` boolean it was designed not to be.
2. **The reason never survived `_staged_command`** — it was accepted, then dropped before the run
   record was written, so the intent it exists to capture was not persisted anywhere.
3. **It was never echoed**, despite the help text saying it would be.

All three are fixed for **both** flags, with 17 tests. Suite **1932 passed**, 12 skipped, 2 xfailed.

**Root-cause class: I tested that the flag EXISTS, not that it WORKS.** Every defect is downstream
of the argument parser, and my check stopped at the parser. A guard whose bypass is unlocked by a
space is not a guard, and a reason that is not persisted is not a record.
⇒ **RULE: for any guard, test the BYPASS PATH, not the happy path** — pass the degenerate value
(empty, whitespace, the sentinel) and assert it is refused; then assert the recorded artifact
actually contains the reason. Same family as *"a seam that is wired but cannot change the output is
decoration"*, one level down: a rail that can be stepped over is not a rail.
⇒ **RULE: mirroring an existing pattern is a free audit of the original.** This one found three
defects in code I had shipped hours earlier. When copying a shape, test the source too.

---

## R-2026-08-03-cite — ⛔ CITATION DRIFT: `refc.py:1112-1117` is the WRONG line range,
## and it propagated into every brief I wrote today

**What I circulated.** That `refc.py:1112-1117` *"computes feature maps for all W frames and keeps
only the last"*, and that this was the mechanism behind both the sitclf capacity ceiling and
`long_accel`'s unrecoverability. It went into **three agent briefs**, the chat summary, and a
program report.

**What is actually at those lines:** `_goal_along_prior` — the **anchor-endpoint prior**. Nothing to
do with feature maps. The correct location is **`refc.py:1688, 1691`**.

**And the claim the citation was carrying is itself refuted** (see the latent-bottleneck stream,
2026-08-03): REF-C's input is `in_channels=9`, a D-015 **3-frame stack**, so the kept map already
spans **~300 ms**. "Single-instant" was wrong in substance as well as in address.

**Root-cause class: a line number quoted from memory across a file that other streams were editing
concurrently.** Line numbers are the least stable identifier in a live repo, and I re-quoted mine
without re-reading. The claim looked verified *because it carried a precise-looking citation* — a
false precision that made it harder, not easier, to check.
⇒ **RULE: cite a SYMBOL, not a line range** — `RefCModel.forward`, `_goal_along_prior` — and re-read
before re-quoting. A line range is admissible only alongside the symbol name, so drift is detectable.
⇒ **RULE: a precise citation is not evidence the claim was checked.** Three agents accepted this one
because it was specific.

### Also retracted here: S6's registered conditionality

`refc.py`'s `refc_goal_config` docstring registered the S6 predicted-goal arm as **"conditional on
the sibling temporal-feature stream"**. **Retracted** — the two do **not** share an input path
(REF-C keeps one feature map; `sitclf.causal_window` stacks eight). A null in the temporal stream is
not evidence about S6, and the registration would have let an unrelated result **silently cancel a
lever that had never been tested**. Fixed in `refc.py`; S6 is an independent lever.
⇒ **RULE: an arm may be registered as conditional on another result only when the two share the
MECHANISM, not merely the topic.** Check the input path before writing "conditional on".

---

## R-2026-08-03-rho — ⛔ A RANK CORRELATION OVER THE CANDIDATE AXIS WAS USED TO SIZE A SELECTOR,
## and it is not a proxy for one — MEASURED on both REF-C arms

**What was circulated.** `PREREG_D-SEL…` §6.3 registered *"S3 LIVE ⇒ include S3 in the retrain
arm"*, triggered by Spearman `ρ(cons_i, −ADE_i)` over REF-C's whole candidate axis. E-SEL-1
measured **ρ = 0.6657** (base) / **0.6212** (XL) and the branch fired.

**What is NOT retracted.** E-SEL-1's ρ is correct and reproduces **inside its own CI** from an
independently decoded, **bit-identical** fan. E-SEL also flagged, correctly, that the statistic uses
the **future frame** `z_{t+5}` which the deployed path never sees, and refused to quote 0.65 as an
effect size. That refusal was right and is the reason this measurement exists.

**What IS retracted: the inference from a high ρ to a fundable SELECTOR.** MEASURED 2026-08-03
(`…/incoming/2026-08-03-s3-deployable/`, 881 windows / 40 episodes, both arms):

* the score with **ρ = 0.6657 — the one allowed to see the future — selects at 6.49 m ADE@2s**
  against a shipped 0.4728, i.e. **13.7× worse**;
* the **deployable** score selects at **20.23 m / 35.86 m**, i.e. **worse than the random control**
  (14.54 / 13.96 m);
* a **zero-parameter** score (distance to the constant-velocity baseline) reaches **ρ = 0.995** and
  still selects **worse than shipped** (0.815 m).

**Root-cause class: the statistic was computed over a population the decision cannot act on.**
**72–74 % of REF-C's fan is outside the reachable band and deleting it is MEASURED exactly inert on
ADE** — so a full-axis rank correlation is dominated by candidates **no selector ever picks**.
Restricted to the reachable survivors, ρ collapses: **oracle 0.6657 → 0.3008**, and the **deployable
score → −0.0286 [−0.0863, +0.0277], a CI that crosses zero**.

⇒ **RULE: a correlation only sizes a decision if it is computed over the candidates the decision can
actually choose between.** Report ρ on the actionable subset **beside** the full-axis ρ, or do not
quote it for a selector.
⇒ **RULE: convert a correlation into the decision's own units by MEASURING the decision** — here,
run the argmax and the gated graft — never by arguing from the correlation's magnitude.
⇒ This is the same family as the C6 confound and the REF-A I-JEPA leak one level out: not a leaked
*input*, but a **statistic whose support does not match the deployed choice**.

### Also logged: the branch-table defect reproduced ONE DAY after it was escalated

`PREREG_S3_DEPLOYABLE.md` §4 (written 2026-08-03) registered *"ρ_deploy **separated from**
C-ctxswap AND **separated from** C-cv"* as the FUND trigger. Both separations came back
**ADVERSE** — the controls beat the score — and the trigger fired anyway, exactly as
`PREREG_D-SEL…` §6.3's four S1 branches could not express E-SEL-0's adverse separation the day
before. **The escalation was read and the same defect was written again.**
⇒ **RULE: every "separated from a control" trigger carries a DIRECTION predicate** —
*separated **and** the delta favours the treatment*. A bare `separated` is satisfied by a control
that beats you.

---

## R-2026-08-03-rigpair — ⛔ "+0.930 → −2.465" PAIRS TWO DIFFERENT EXPERIMENTS

**What has been circulated** (`LATENT_BOTTLENECK.md` §5 RANK 1, the D-APPEAR brief, and several
summaries): *"the measured cross-rig collapse, frozen v1 speed R² **+0.930 → −2.465**"*, cited to
`…/incoming/2026-07-22-idm-proof/results.json`.

**What that file actually says**, read from source today:

| JSON path | value |
|---|---|
| `experiments/rigA_to_rigB/val/in_rig_heldout_rigA/r2/speed` | **+0.7863** |
| `experiments/rigA_to_rigB/val/cross_rig_rigB/r2/speed` | **−2.4654** |
| `experiments/physicalai_to_comma2k19/val/in_corpus_heldout_paival/r2/speed` | **+0.9297** |

⇒ The **+0.930 is the in-corpus baseline of a DIFFERENT experiment** (the PhysicalAI→comma2k19 arm).
The rig experiment's own in-rig baseline is **+0.7863**. The collapse is **+0.7863 → −2.4654**.

**Root-cause class: C4 (inherited without re-verification) compounded by C6 (confounded
comparison).** Both numbers live in one JSON under one `experiments` key, so a top-level grep for
"the biggest and the smallest speed R²" produces a pair that reads like a before/after and is not
one. Nobody re-opened the file because the pair had a plausible shape and an artifact path.
⇒ **RULE: a "X → Y" pair must name ONE experiment key, not one FILE.** Quote the JSON path of both
halves, not the filename; a shared artifact path is not evidence that two numbers are comparable.
⇒ **RULE: an in-domain baseline may only be paired with an out-of-domain number produced by the
SAME fit.** The cheapest check is that the two paths share every path component above the split.

### And the claim the pair was carrying does not reproduce

MEASURED today (`…/incoming/2026-08-03-appearance-shortcut-audit/results_p2_rig.json`), on the
principal-point-cropped episode cache, exact ridge, 116 rig-A + 120 rig-B episodes: the frozen v1
latent shows **no cross-rig speed drop at all** — A→A **+0.7052**, B→B **+0.7194**, B→A **+0.7127**,
A→B **+0.7011** — and the two rigs' horizon rows agree to **8 of 256** (a legacy geometric-centre
crop would be **~100 rows** off). ⚠️ Two things differ from the 2026-07-22 run — the cache geometry
**and** the head (a 2.9 M-param MLP that can extrapolate to R² −2.47 off-domain vs a ridge that
structurally cannot) — so the −2.4654 is **NOT attributed here**; it is only shown not to reproduce.
⇒ **RULE: "does not reproduce" and "is explained" are different claims.** Say which one you have.

---

## R-2026-08-03-appear — ⚠️ THE APPEARANCE SHORTCUT IS CORPUS-SPECIFIC (a pre-registered withdrawal)

**What was circulated**, from `LATENT_BOTTLENECK.md` §0.0: *"on this corpus `speed` is read mostly
from STATIC APPEARANCE — one static frame is 93 % of the full learned latent's speed accuracy, and
~1.75× the best motion-only arm"*, offered as *plausibly one fact* behind the 88.7 % longitudinal
gap and the cross-rig collapse.

**MEASURED off-highway** (`…/2026-08-03-appearance-shortcut-audit/results_p1_physicalai.json`,
same encoder `v1_speedjerk_ckpt.pt` step 29999, same recipe, 240 PhysicalAI episodes,
80 held out): the still frame reads `speed` at **−0.0025 = the empirical null**, against the latent
window's **+0.6752**. **RATIO −0.0037** vs comma2k19's **0.9296**. The ordering does not shrink,
**it reverses**: motion energy separates (`mot16_window_rbf` **+0.4124**) while every appearance
form is at the null.

⚠️ **This is NOT a retraction of a careless claim** — `LATENT_BOTTLENECK.md` labelled it a
HYPOTHESIS at programme scale, wrote the transfer test as its own RANK 1, and §7 named the exact
corpus property responsible. It is logged because the CLASS is worth having, and because the
withdrawal was pre-registered (`PREREG_APPEARANCE_SHORTCUT.md`, OUTCOME C) rather than argued after
the fact.

**Root-cause class: NEW — C16, EPISODE-DISJOINT MISTAKEN FOR DOMAIN-DISJOINT.**
The ladder (`results_p1b_mechanism.json`) shows the map exists on BOTH corpora and transfers on only
one: still-frame speed R² **within-clip +0.9825 → across-clip +0.6642** on comma2k19 (68 % retained)
against **+0.8023 → −0.0025** on PhysicalAI (0 % retained). comma2k19 val is **one driver, one
vehicle, one camera, one road class**, so holding out 17 of its 50 episodes does **not** hold out a
domain; PhysicalAI's 500 clips span cities, vehicles and two rigs and it does.
⇒ **RULE: state what the held-out unit is INDEPENDENT OF, not just that it is disjoint.** An
episode-disjoint split on a single-rig, single-road-class corpus certifies far less than the phrase
suggests, and a memorisation-shaped result will pass it.
⇒ **RULE (cheap and general): before believing any probe result, run it once with a random WINDOW
split as well.** If within-unit ≫ across-unit the arm is memorising; if the two agree the arm has
found something that travels. It costs one extra fit and it is what turned this null from
"the probe is broken" into a mechanism.

| # | class | recognition signal |
|---|---|---|
| **C16** | **Episode-disjoint mistaken for domain-disjoint** | a held-out split is described only as "disjoint", and the corpus has one rig / one vehicle / one road class |

---

## R-2026-08-03-mem — ⛔ "v5f is at 98–100 % of its container cap and OOM-looping" is RETRACTED

**What I claimed**, three times in one hour, and acted on each time: that the v5f trainer was pinned
against its 50 GB container cap, that the `rc=137` SIGKILL was a container OOM, and that my
`--workers 8 / --v2-lru 64 / --batch 8` change had caused it.

**MEASURED**, on `tanitad-new`, **with nothing running at all**:

| `/sys/fs/cgroup/memory/memory.stat` | |
|---|---|
| `cache` | **37.0 GB** ← reclaimable page cache |
| `rss` | **0.1 GB** |
| `usage_in_bytes` | 37.2 GB → **74 % of the cap, at idle** |
| **`memory.failcnt`** | **0** |

Under load: `usage_in_bytes` **98–100 %** while `rss` was **4.9 GB** and `failcnt` still **0**.

⇒ **`memory.usage_in_bytes` counts page cache and is not a pressure signal.** A cgroup that has
never hit its limit reports `failcnt 0`, and it did — throughout. **The container-OOM diagnosis is
refuted.** The `rc=137` remains **UNEXPLAINED**, and I am recording it as unexplained rather than
attaching it to the first plausible mechanism.

**Root-cause class: a counter that aggregates something RECLAIMABLE, read as pressure.** This is the
**third costume** of one trap already in the preflight twice — *"never judge pod disk with `df`"* (it
reports the cluster, hides the quota) and *"on Thor `free`/`tegrastats` show 106 GB used on an idle
box"*. I had written the Thor entry into `CLAUDE.md` **the same day** and still walked into it.
⇒ **RULE: before reading any usage counter as pressure, read it with the load REMOVED.** An idle
baseline separates "in use" from "accounted to us". It costs one measurement and it is the whole
diagnosis.
⇒ **RULE: prefer the counter that only moves on the event you care about** — `failcnt` /
`memory.events`, not `usage_in_bytes`; `torch.cuda.max_memory_allocated()`, not `mem_get_info`.

**Cost:** ~40 min of v5f training and three unnecessary restarts. v5f is back on its exact
known-good config (`--batch 4 --accum 16 --v2-lru 4 --workers 4`), resumed at step 4001 from a
checkpoint at 4000, `rss` 4.9 GB, `failcnt` 0.

**What SURVIVES and is still worth having.** The original diagnosis stands and is independent of
this error: **v5f is INPUT-BOUND** — GPU utilisation median **~39 %** on the shipped config against
v1arch's **79 %** at the same effective batch. And the `--batch 8` run MEASURED GPU utilisation at
**~94 %**. That gain is real and is still on the table; it is blocked only on explaining the
`rc=137`, which must be done before retrying rather than assumed away.

### Two ops mechanisms bought at the same time

1. **`supervise_run.sh` sources its manifest ONCE at startup, not per relaunch.** My first relaunch
   came back on the OLD command and looked successful. Correct order: edit manifest → kill the
   **supervisor** first → kill the trainer → start a fresh supervisor. **Verify by grepping flags out
   of the RUNNING process**, never by reading the manifest back.
2. **Restarting a supervisor immediately after killing the old one races its `flock`** — the new one
   exits with *"another supervisor holds …lock"* and **nothing runs**, while the log reads like a
   normal startup. Poll until both are gone; a lock with no holder in `/proc/*/fd` is debris.

⚠️ **And `pgrep -f <pattern>` self-matched my own ssh command THREE times in this sequence** — once
reporting a dead supervisor as alive, once listing my own shell as *both* the trainer and the
supervisor because the command string contained both literals. The preflight rule says "kill by
explicit PID"; the sharper form is: ⇒ **`pgrep`/`pkill -f` are inadmissible for STATE CHECKS too,
not just for killing. Use `kill -0 <explicit PID>`, or an `awk` filter that excludes your own
command.**

---

## R-2026-08-03-corpus — ⛔ "the raw parity corpus was not found on any live machine" is RETRACTED

**What I wrote**, in a program report and then into an agent brief as an established premise:
*"pod1/pod3/eval all `Connection refused`; pod4 and `tanitad-new` hold no raw epcache (three probes
each). It survives ONLY on HF."* I labelled corpus durability the programme's top non-scientific
risk on the strength of it.

**MEASURED:** `tanitad-thor` was holding it the whole time —
`~/epcache/epcache-256px-phase0/physicalai-train-e438721ae894` and
`~/valdata/physicalai-val-0c5f7dac3b11` with **40/40** val episodes present.

**Root-cause class: ABSENCE FOUND AT ONE LOCATION, REPORTED AS ABSENCE** — the rule that is already
first in the operating standard, with the 12-day Vulkan-ICD failure attached to it. The probes swept
`/workspace/...`, which is where the *pods* keep a corpus. **Thor keeps its corpus in `$HOME`.** One
probe shape, applied to a host with a different layout, and the answer inverts.
⇒ **RULE: a negative probe must vary the PATH SHAPE, not only the host.** "Not at `/workspace/X` on
four machines" is one probe repeated four times, not four probes.
⇒ **RULE: I inherited this from a sibling stream's report and promoted it to a brief premise without
re-deriving it.** That is exactly the INHERITED→MEASURED laundering the evidence-class rule exists to
stop, and it is worse in a brief than in a report, because a brief is read as settled ground.

**The risk survives the retraction, in a smaller and better-specified form** — Thor's *train* holding
is partial, so the 278.78 GB raw parity train epcache still has one non-durable copy plus HF. But it
had to be re-derived, and the sharper census is:

| artifact | size | copies | durable | status |
|---|---|---|---|---|
| raw parity TRAIN epcache 256 px | 278.78 GB | 1 | 1 (HF) | mitigation in flight |
| raw parity VAL epcache 256 px | 70.39 GB | 1 | 1 (HF) | mitigation armed, sequenced behind the train pull |
| **9 checkpoints on `pod4`** | ~28 GB | 1 | **0** | 🔴 **PI authorisation required** |

✅ **Closed:** the 256 px REF-C val raster — **40/40 episodes match the HF LFS sha256 bit-for-bit**
and `torch.load` cleanly, including the previously 21 %-truncated `ep_00028`. Size and exit code were
not accepted as evidence.

### Registry citation defects: the brace-expansion bug is NOT isolated

The `…_vs_refc-{base,xl}-30k.json` defect corrected earlier today is **one of 17**. Full sweep of
`MODEL_REGISTRY.md`: **252 citations — 115 EXISTS · 4 MISSING · 17 NOT_A_PATH · 22 pod paths
(uncheckable)**. All 4 MISSING are **malformed citations, not missing artifacts**. Five cited pod
paths are stranded with no repo counterpart.

⚠️ **Process note, recorded because it is a real exposure.** An agent command printed `Keys.txt` to
its tool output. **VERIFIED: `Keys.txt` is git-ignored (`.gitignore:44`), has never been staged, and
appears in no commit on any branch.** The exposure is confined to the session transcript. The HF
token was redacted by the agent; **the file's other keys were not**, so they should be treated as
disclosed and rotated. ⇒ **RULE: read the token programmatically (`grep -oE 'hf_[A-Za-z0-9]+'`) —
never `cat`, `head` or print the file, not even once, not even to check it exists.**

---

## R-2026-08-03-align — the reference-offset CORRECTION, and three claims made ABOUT it that do not hold

`R-2026-08-03-k` established that every absolute render-fidelity number on `00040136` was scored
against a reference **6 frames too early**. This entry closes it: the rule is now measured, the
absolutes are re-baselined, a gate is in the code — and **three statements made while establishing
the +6 are withdrawn.**

**THE RULE (MEASURED, mine, on 2 of 2 scenes that have both files):**

> `video_frame_index = rig_frame_index + (n_mp4_decodable − n_rig_frames)`
> **+6** on `00040136` (605 − 599) · **+5** on `7c72937c` (604 − 599).

Renderer neighbour scan, k = ±10, 12 frames per scene, PRODUCTION render, only the reference index
varying: `argmax_histogram` **`{6: 12}`** and **`{5: 12}`** — unanimous, **zero refusals**, bootstrap
mass **1.00** on each point (bootstrap over probed frames of `argmax(mean curve)`, B=2000; an
integer estimator has no meaningful SE, so the **mass function IS the interval**). Gain over the
shipped indexing: **+0.1797** and **+0.1699** grad-NCC, free.
⭐ **`7c72937c` had never been scanned by the renderer at all** — its +5 rested on arithmetic. It is
now MEASURED, and it agrees with the counting predictor exactly.
Artifacts: `…/incoming/2026-08-03-render-rebaseline/raw/rs_frame_offset_{00040136,7c72937c}_k10*.json`.

### 1. RETRACTED — "cross-correlating image motion against ego translation … peaks at exactly +6 and turns over at +7" as an INDEPENDENT CONFIRMATION

Source: `stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter-adversarial/ALIGNMENT_DIRECTION_GPUFREE.json`,
cited as one of *"3 independent probes incl. a renderer-free one"*.
**Literally true and NOT decisive.** MEASURED (mine, `raw/align_gpufree.json`): peak `r = 0.44884`
at +6; **best competitor outside ±1 is `r = 0.44341` at +8**; **prominence 0.0054** — a 1.2 %
separation across a ±2-frame span on a curve that ramps monotonically 0.383 → 0.449. It **agrees**;
it cannot **discriminate**. The genuinely independent confirmations are `count_delta` (the dataset's
own counts) and `leader_pad` (the frozen head block).

### 2. RETRACTED — `static_head_block_frames = 5` on `7c72937c` as evidence of a leader

Same file. That scene's own `rig_ego_speed_mps_first_9_frames` field reads **`0.0` for all nine
frames**. A stationary camera produces near-identical frames with no synthetic leader, so the block
length is **not identifiable from the video alone** there. `frame_align.leader_pad()` now REFUSES
(`ego_stationary_unidentifiable`) instead of reporting the number. The +5 is unaffected — it stands
on `count_delta` **and** the renderer scan.

### 3. SUPERSEDED — "rolling shutter buys +0.0210 at ~90× the render cost" (`R-2026-08-03-j`)

That delta was measured against a reference 5–6 frames out of alignment. Re-measured at the
corrected reference, same code, same frames, paired bootstrap over frames B=10000
(`raw/RS_MARGINAL_REBASELINE.json`), RS over the deployed config:

| scene / n | superseded | **corrected** | ratio | cost |
|---|---|---|---|---|
| `00040136` n=12 | +0.0179 [+0.0124, +0.0237] | **+0.1158** [+0.1046, +0.1285] | **×6.5** | ×93.5 |
| `7c72937c` n=12 | +0.0542 [+0.0332, +0.0782] | **+0.1615** [+0.1440, +0.1803] | **×3.0** | ×81.6 |

⚠️ **This does NOT vindicate rolling-shutter physics** — `R-2026-08-03-j`'s retraction of the CAUSE
is now *better* supported. An RS render sweeps the pose across a **30.559 ms readout ≈ 0.917 of a
frame**, so it spans the newly measured **sub-frame residual** (+0.232 fr / 7.75 ms on `00040136`,
+0.164 fr / 5.47 ms on `7c72937c`) by construction. A temporal-smear arm gaining most when a
temporal misalignment is removed is evidence the gain is TEMPORAL, not shutter-specific. **The
deployment verdict is untouched: 2498–3684 ms against a 36 ms budget. RS stays off.**

### AND THE HEADLINE THE PROGRAMME SHIPPED: "+23.4 % grad-NCC" SURVIVES AT ROUGHLY HALF SIZE, AND DOES NOT REPLICATE

Same code, same scene, same frames, same arms — **only `--ref-offset` differs**, and the offset-0 arm
reproduces the shipped `panel6_chosen` EXACTLY (0.2774 / 0.3424 / 0.3747), so this is not confounded
by a code change. Paired bootstrap over frames, B=10000 (`raw/REBASELINE_TABLE.json`):

| contrast | superseded | **corrected** |
|---|---|---|
| `00040136` n=5 (the shipped panel) | **+23.4 %**, Δ +0.0650 [+0.0422, +0.0923] sep | **+13.5 %**, Δ +0.0572 [+0.0323, +0.0892] sep |
| `00040136` n=12 | +22.1 %, Δ +0.0564 sep | **+8.0 %**, Δ +0.0362 [+0.0083, +0.0648] sep |
| `7c72937c` n=12 | +8.4 %, Δ +0.0233 [+0.0014, +0.0490] sep | **+4.4 %**, Δ +0.0199 **[−0.0097, +0.0521] NOT sep** |

⇒ Under the standing rule already written into `Q2_RENDER_FIDELITY_PLAN.md` §R4 (*improvement = CI
above 0 on ≥2 of 3 scenes*), the shipped configuration now clears **1 of 2**. Not a refutation — a
**loss of the evidence that justified it**, and the `cull=0.95` / `sky-gain=0.3` optima were both
chosen by maximising grad-NCC against the misaligned reference and must be re-swept.
✅ **What is untouched: every closed-loop number, including `R-2026-08-03-C`.** `cl_metrics.py` never
opens the reference video (no `load_refs`, no `VideoCapture`, no `.mp4`); the four families come from
rollout poses and `sequence_tracks.json`. The one channel by which the correction could still reach
them — the render CONFIG was selected on the biased criterion — is flagged as a follow-up, not a claim.

**Root-cause class: C14 — A SWEEP'S GRID END RE-LABELLED AS A MEASURED LIMIT, in its ESTIMATOR form:
an instrument structurally incapable of reporting anything but an answer.** `max(d, key=score)`
returns an offset on a monotone curve, on a flat curve, and on pure noise. In this work alone it did
so three times: a ±3 scan that stopped at its edge still rising and reported *"≥ +3"*; the
cross-correlation on `7c72937c` that rose monotonically −15 → +15 and argmaxed at **+15**; and the
gate's own first version, which classified a **flat** curve peaking at the window edge as
*"the residual is off-window"* and **blocked a CORRECT offset**.

⇒ **RULE: an estimator that cannot return "I do not know" is not an estimator, it is a formatter.**
Every alignment curve now goes through `frame_align.adjudicate()`, which refuses on `weak`,
`not_separated`, `boundary` or `no_turnover` — and the ORDER matters: signal strength is adjudicated
BEFORE window position, because a flat curve peaking at an edge is *no signal*, not *off-window*, and
those demand opposite responses.
⇒ **RULE: a control that the estimator passes by REFUSING proves nothing.** `motion_lag`'s
injected-shift controls all "fail" on real data purely because it refuses the base case; that is
stated in the report rather than scored as a pass.
⇒ **Shipped so it cannot recur:** `render_quality.py::assert_reference_aligned` runs BEFORE any arm
reports a number, is **on by default**, derives the offset **per scene**, and has three outcomes —
PASS / FAILED / **CANNOT CERTIFY** (no probe frame carries signal), the last explicitly *not*
"aligned". Demonstrated on the real scenes: it REFUSES the shipped configuration on both
(naming +6 and +5), REFUSES an **over**-correction of +10 (naming −4, so it cannot be satisfied by
pushing the offset until the number looks good), and PASSES at +6 and +5. 50 tests
(`stack/tests/test_frame_align.py`, `test_render_quality_alignment_gate.py`,
`test_ref_offset_repo_wide.py`), full suite **2031 passed**.

⚠️ **NOT established:** that the rule holds beyond **n = 2 scenes**. Only 2 of the 79 NuRec scenes on
Thor have `rig_trajectories.json` (the rest are mp4-only), both are 599-rig-frame clips from one
release, and there is **no fps field, no dropped-frame marker and no manifest field** that predicts
the delta — `data_info.json` says 599 on both. The offset must be **estimated per scene**, and that
estimate is part of every future render number.

---

## R-2026-08-03-appear — ⛔ "APPEARANCE DOMINATES MOTION" is CORPUS-SPECIFIC, and I reported it
## as a programme-scale finding

**What I reported to the PI**, as the finding that "reframes the programme": *a single 32×32
grayscale STILL FRAME reads `speed` at +0.6642 — 93 % of the 800 ms learned latent and 1.75× the
best motion-only arm; all ten linear pure-difference arms sit at the null; nothing in our training
ever forced the encoder to learn motion.* I attached to it the hypothesis that this re-explains the
88.7 % longitudinal gap and the cross-rig collapse.

**MEASURED on PhysicalAI-AV** (pre-registered OUTCOME C, `PREREG_APPEARANCE_SHORTCUT.md`, S/C/P/VOID
with every threshold fixed before any PhysicalAI number existed). Encoder-matched — same frozen
`v1_speedjerk_ckpt.pt` @ 29999, same recipe, 240 eps → 160/80, 4,787/2,390 windows matched to
comma's 4,554/2,346:

| arm | comma2k19 highway | **PhysicalAI-AV** |
|---|---:|---:|
| `pix32_centre_rbf` — one 32×32 grey still frame | **+0.6642** | **−0.0025 = the empirical null** |
| `v1_window` — 18,432 features, 800 ms | +0.7145 | **+0.6752** |
| **ratio** | **0.9296** | **−0.0037** [−0.0498, −0.0000] |

The run is **admissible, not VOID**: `v1_window` separates (+0.6777 [+0.6328, +0.7235]) and the null
arm reproduces the floor.

⇒ **The programme-scale claim is WITHDRAWN.** And it does not merely shrink — **the ordering
reverses**: on PhysicalAI every appearance form is at the null (1,024 / 9,216 / 9 features; linear,
rbf, single-instant, within-stack) while **64 features of motion energy separate** (`mot8_centre_rbf`
+0.3707, `mot16_window_rbf` +0.4124).

### The mechanism, established by a pre-registered ladder rather than argued

| corpus | still-frame arm | within-clip split (leaky) | across-clip split (real) | retained |
|---|---|---:|---:|---:|
| comma2k19 | `pix32_centre_rbf` | +0.9825 | +0.6642 | **68 %** |
| PhysicalAI | `pix32_centre_rbf` | +0.8023 | −0.0025 | **0 %** |

Both rival explanations were killed by measurement, not by assertion: the PhysicalAI substrate is not
degenerate (**higher** dynamic range than comma's, 0 constant features) and its speed distribution is
not narrower (**CV 0.621 vs 0.589**). The appearance→speed map **exists on both corpora and transfers
on one**, because **comma2k19 val is a single driver / vehicle / camera / road class**.

⭐ **Root-cause class C16: EPISODE-DISJOINT ≠ DOMAIN-DISJOINT.** A held-out split that separates
*episodes* can leave the *domain* fully shared, and a shortcut that memorises the domain then reads
as a capability. **Cheap general fix: run every probe once with a RANDOM-WINDOW split too.** The gap
between the two splits IS the shortcut, measured — 68 % on comma, 0 % on PhysicalAI.
⇒ **RULE: a probe on a single-domain corpus cannot support a programme-scale claim about
representations, however clean its held-out split.** I promoted a comma2k19-highway result to
"reframes the programme" in one step.

### Two further corrections that travel with it

1. ⛔ **"+0.930 → −2.465" pairs TWO DIFFERENT EXPERIMENTS.** 0.930 is one run's *held-out* read;
   −2.4654 is another run's *cross-rig* read, and **that run's own within-rig baseline is +0.7863**.
   The honest pair is **+0.7863 → −2.4654**. Fixed in `MODEL_REGISTRY.md`; the figure also appears in
   ~8 other documents which the registry now supersedes.
   ⇒ **RULE: a "X → Y" degradation pair must come from ONE experiment**, or the difference silently
   contains the between-run delta as well as the effect.
2. ⚠️ **The cross-rig collapse does not reproduce in the current cache**: A→A +0.7052, B→A +0.7127,
   paired **+0.0075 [−0.0318, +0.0502] NOT separated**. The −2.4654 is **NOT ATTRIBUTED** — cache
   geometry (the rigs' horizons agree on 8 of 256 rows here) and MLP extrapolation both remain live.
   **The appearance shortcut is NOT a third explanation for it** — it is at the null in all four rig
   cells.

### What the audit CONFIRMED rather than overturned

- **The scenario classifier is NOT threatened** on all three situations: the shortcut's first hop
  does not exist on this corpus (still→speed R² **+0.0102** vs latent **+0.6900**). This agrees
  independently with the sitclf stream's own still-frame control (motion worth ~70 % of the skill).
  ⚠️ Only `intersection` is properly powered; the agent said so rather than dressing up the other two.
- **`long_accel`'s unrecoverability is untouched** by this retraction — that result stands on its own
  17-architecture evidence.
- The **0-GPU latent screen is now promoted** to `stack/tanitad/eval/latent_screen.py` with 12
  contract tests, reproducing its reference through the promoted module and passing its oracle
  control. ⚠️ **New fact: the same encoder is 6.9× LESS jittery on PhysicalAI than on comma2k19**, so
  **the screen's thresholds are corpus-dependent and must carry their corpus.**

⚠️ The agent also found and reported **a defect in its own pre-registration** (AP-lift's chance level
is 1.0, not 0 — the raw form would have wrongly read THREATENED on `lane_change`) rather than
quietly using the corrected form.

---

## R-2026-08-04-failcnt — ⛔ THE REFUTATION IN `R-2026-08-03-mem` WAS ITSELF BUILT ON AN UNUSABLE COUNTER

**What `R-2026-08-03-mem` concluded**, and what has been carried since: *"A cgroup that has never
hit its limit reports `failcnt 0`, and it did — throughout. **The container-OOM diagnosis is
refuted.** The `rc=137` remains **UNEXPLAINED**."*

**MEASURED 2026-08-04 on the same pod** (`tanitad-new`,
`…/incoming/2026-08-04-v5f-sigkill/raw/counters_snapshot.txt`):

| counter | value |
|---|---|
| `memory.limit_in_bytes` | 49,999,998,976 (46.57 GiB) |
| `memory.memsw.limit_in_bytes` | 49,999,998,976 — **EQUAL** |
| cgroup `swap` / host swap | **0 / 0** |
| `memory.failcnt` | **0** |
| **`memory.memsw.failcnt`** | **28,908,911** → **29,219,916** 26 min later (**≈ 200 failures/s**) |
| `memory.max_usage_in_bytes` | 49,999,998,976 — **exactly the limit** |
| `memory.oom_control` | `oom_kill_disable 0  under_oom 0  ` **`oom_kill 1`** |

In cgroup v1, `try_charge()` charges **memsw first** and `page_counter_try_charge` increments
`failcnt` on the counter it exceeded. With `memsw.limit <= memory.limit` and no swap — this
container, and the ordinary Docker/RunPod default — **memsw absorbs every failure and
`memory.failcnt` is pinned at 0 for the container's life, however hard the cap is hit.** The data
prove it without the kernel source: a cgroup whose **peak usage equals its limit exactly** has
certainly hit that limit, and `failcnt 0` beside it is only explicable by the memsw-first path.

⇒ **The `rc=137` is EXPLAINED: a SIGKILL from the OOM killer.** The attempted
`--batch 8 --accum 8 --v2-lru 64 --workers 8` projects to **≈51.2 GiB of UNRECLAIMABLE memory
(`rss + shmem`) against a 46.57 GiB cap** — LRU **18.8 GiB** (measured 33.4 MB/clip × (8+1) × 64),
4 extra workers **+8.1 GiB**, transport **+15.1 GiB**, on a 9.90 GiB base. It died at **3 min**,
during DataLoader warm-up, which is exactly when that burst peaks. Full workings, and the seven
alternatives refuted one by one (operator `kill -9` path, `memwatch`, supervisor guards, GPU fault,
double-launch, CPU quota, MooseFS `Errno 5`): `…/incoming/2026-08-04-v5f-sigkill/V5F_SIGKILL.md`.

**What still stands from `R-2026-08-03-mem`:** `memory.usage_in_bytes` counts reclaimable page cache
and is **not** a pressure signal (37.2 GB of 50 at idle, `rss` 0.1 GB). That is correct, and the
idle-baseline rule it bought is excellent. **Only the `failcnt` half fails.**

⚠️ **Residual, stated rather than papered over:** `dmesg` is `Operation not permitted` in this
container and `/dev/kmsg` is unreadable, so the kernel's own OOM report cannot be read from inside.
`oom_kill` increments for a task in this memcg killed by **either** the memcg or the host's global
OOM killer, so the two are **not formally separated** — host-OOM is strongly disfavoured (503 GiB
host, 414 GiB available, zero swap) but not excluded. The RunPod console settles it in one look.

**Root-cause class: NEW — C17, A COUNTER THAT IS STRUCTURALLY FROZEN FOR THIS CONFIGURATION, READ
AS EVIDENCE OF ABSENCE.**
This is the sibling of C-`R-2026-08-03-mem`'s *"a counter that aggregates something RECLAIMABLE,
read as pressure"* — and it bit **the fix for that very error, in the same hour**. Note the shape:
the earlier retraction's own closing rule (*"prefer the counter that only moves on the event you
care about — `failcnt`"*) was **right**, and was then applied to a counter that **cannot move**. A
correct rule aimed at the wrong instrument produces a confident wrong answer, and it inherits all
the authority of the retraction that introduced it.
⇒ **RULE: before reading a zero as absence, establish that the counter is ABLE to be non-zero.**
A counter at 0 that *cannot* move and one that *did not* move are the same digit and opposite facts.
Cheap general check: find a sibling counter that IS non-zero (`memory.memsw.failcnt` here), or
induce the event once and confirm the counter responds.
⇒ This is CLAUDE.md's *"absence found at ONE location is not absence"* in a cgroup costume: the
generalisation is **absence found with ONE instrument is not absence.**

| # | class | recognition signal |
|---|---|---|
| **C17** | **Structurally-frozen counter read as absence** | a zero is load-bearing, and nobody has shown the counter can be non-zero in this configuration |

**Instrument shipped so this is not re-derived:** `stack/scripts/pod_kill_forensics.py`
(+ `stack/tests/test_pod_kill_forensics.py`, 21 tests). `live_failcnt()` decides which `failcnt`
can move and names the frozen one; `unreclaimable_bytes()` reports `rss + shmem` instead of
`usage_in_bytes`; `oom_window()` **refuses to return a bare `oom_kill` count** without the
container-start window (it read 6 then 0 on pod2 and was over-quoted); `decode_exit_code()` states
that 137 is SIGKILL and **never** a CUDA OOM, which exits 1 with a traceback.

⚠️ **Two corrections that travel with this.** (1) `V2CompressedCache`'s docstring said
"~2-4 MB/clip"; the 256×640 lossless-PNG caches MEASURE **33.36 MB/clip** (n=40, reproduced) —
**8-17× low**, and budgeting from it is what made the fatal config look affordable. Fixed in
`v2_dataset.py`. (2) The *"GPU median ~39 %"* premise for the speed-up is **unstable**: on the same
unchanged config, **median 33 % (n=20)** and, 20 min later, **median 99.5 % (n=12)** — utilisation
is bimodal across the 16-step accumulation cycle. **The size of that prize is currently unmeasured**
and should be re-measured with a step-synchronised instrument before a cutover is spent on it.

---

## R-2026-08-03-mem — ⛔ AMENDED 2026-08-04. Half of this entry is itself wrong.

That entry retracted the container-OOM diagnosis of v5f's `rc=137` on two grounds. **One stands, one
does not, and the `rc=137` is NOT unexplained.**

| the entry's grounds | verdict |
|---|---|
| *"`memory.usage_in_bytes` counts reclaimable page cache and is not pressure"* | ✅ **STANDS** — measured at idle: `cache` 37.0 GB / `rss` 0.1 GB |
| *"`memory.failcnt` was 0 throughout, so the cgroup never hit its limit"* | ⛔ **WRONG — that counter CANNOT MOVE on this cgroup** |

**MEASURED, and I re-verified it myself before amending:**

```
memory.limit_in_bytes        49,999,998,976
memory.memsw.limit_in_bytes  49,999,998,976   <- EQUAL, and swap is 0
memory.max_usage_in_bytes    49,999,998,976   <- EXACTLY the limit
memory.failcnt                            0   <- structurally frozen
memory.memsw.failcnt             29,660,004   <- ~200 failures/second
memory.oom_control            oom_kill 1
```

In cgroup v1, `try_charge()` charges **memsw first**. With `memsw.limit == limit` and no swap, memsw
absorbs **every** failure and `memory.failcnt` can never increment. **A cgroup whose peak usage
equals its limit has certainly hit it.**

⇒ **The `rc=137` WAS a container memory-cgroup OOM kill.** Seven alternatives were refuted with
artifacts — including my own tooling (my cutover script targeted only hardcoded PID 19412, sent a
plain SIGTERM 7.5 min earlier and had no escalation line; `memwatch.sh` has no kill path and was
created 6 min *after* the death), the supervisor guards (pod copy bit-identical to repo, md5
`0daf4be6…`, sends no signal), a GPU fault (0 uncorrectable ECC — and **SIGKILL is never a CUDA OOM,
which exits 1**), double-launch, CPU quota (**0.0 %** throttled) and MooseFS `Errno 5`.
⚠️ **Residual, not papered over:** `dmesg` is denied here, so the memcg OOM killer cannot be formally
separated from the host's global one. Host-OOM is strongly disfavoured but not excluded.

⭐ **And the mechanism closes arithmetically.** `V2CompressedCache`'s LRU is **per-process**, and the
measured mean payload is **33.36 MB/clip (n=40, reproduced)** — the class docstring claimed
**"2–4 MB", 8–17× low**. The attempted config projects to **~51.2 GiB unreclaimable (`rss+shmem`)
against a 46.57 GiB cap**: LRU 18.8 + 4 extra workers 8.1 + transport 15.1 on a 9.9 GiB base. It died
at **3 minutes — during DataLoader warm-up, exactly when that burst peaks.**

### Root-cause class: I REPLACED ONE UNVALIDATED COUNTER WITH ANOTHER

Correcting the `usage_in_bytes` error, I reached for the nearest alternative counter and **never
checked that IT could move**. `failcnt = 0` was not evidence of absence; it was **absence of
evidence**, and I published it as a refutation.
⇒ **RULE: a counter reading zero is evidence only if you have shown it CAN be non-zero.** Check an
instrument's dynamic range before using it as a negative — the cheapest form is to find the sibling
counter that *is* moving (`memsw.failcnt` here, at 200/s).
⇒ **RULE: when you retract a measurement error, the replacement measurement needs MORE scrutiny than
the original, not less.** A correction carries the authority of having just been careful, and that is
exactly when an unvalidated instrument slips through.

### Two operational facts that fall out of it

1. ⛔ **`--v2-lru 64` must never be retried on this cache.** MEASURED: under `shuffle=True` over
   410,202 windows / 2,400 clips it buys **~2.7 % hit rate for 18.8 GiB**.
2. ⚠️ **The "GPU utilisation ~39 %" premise is UNSTABLE and the prize is currently unmeasured.** On
   the *same unchanged config*: median **33 % (n=20)**, then **99.5 % (n=12)** twenty minutes later.
   Utilisation is **bimodal across the 16-step accumulation cycle**, so any single median is an
   artifact of when it was sampled. ⇒ **Re-measure with a step-synchronised instrument before
   spending a cutover on it.** (I quoted 39 % to the PI as though it were stable; it is not.)

---

## R-2026-08-04-briefs — ⛔ TWO PREMISES I PUT INTO AGENT BRIEFS WERE WRONG, one of them load-bearing

### 1. The anchor provenance — a synthetic rebuild would have silently invalidated every REF-C number

**What I briefed** as established fact: *"`refc_anchors_full.pt` is reconstructible — `build_refc_anchors.py` is in the repo and the config records `{n 256, pool 4096, seed 0}`."*

**MEASURED: that triple is `refc_xl_config()`'s UNUSED SYNTHETIC DEFAULT** (`MODEL_REGISTRY.md:1299/1309`), **not the file's provenance.** The real artifacts record **`pool_size 200000`** — the `--data-root` path. Corroborated three ways: `refc_anchors_small64.pt`'s own metadata, `flagship_v4_anchors_dense.pt`, and registry:1241's explicit *"not the synthetic default"*.

⇒ **A rebuild on my premise would have produced a COMPLETELY DIFFERENT VOCABULARY that still loads
with shape `[256, 4, 2]`** — and every REF-C comparison scored against it would have been silently
invalid. The agent caught it before building.
⇒ **RULE: a config default is not provenance.** A field present in a config object may be the value
that was *used*, or the value that was *never overridden*. Read the artifact's own recorded metadata,
not the constructor that could have made it.
⇒ **RULE: a shape check is not an identity check.** `[256, 4, 2]` loads for any 256 anchors; the
thing that would have failed loudly is the nesting relationship, which is why the rebuild was
deliberately staged as `refc_anchors_full_REBUILD.pt` and **not** under the scoring name.
*(The rebuild reproduces the same VOCABULARY, not the same BYTES: 59/64 rows bit-exact vs
`small64`, 5 differ by ≤7.63e-06 m (≤64 ULP), **selection order preserved**; a second Thor run is
`torch.equal` to the first, so it is cross-host float rounding, not nondeterminism. Architecture vs
torch version could not be separated — stated, not guessed.)*

### 2. The speed-stratification caveat I propagated is FALSE on the surface it was applied to

**What I briefed**, twice: *"20.7 % of lead windows sit at 0–1 m/s where the metric cannot
discriminate, and the 15+ m/s band is UNPOWERED (n=2)."*

**MEASURED on the canonical val40 — both claims INVERT:** the **15+ m/s band is the LARGEST
lead-bearing band (88 leads)**, and the 0–1 m/s crawl defect is **1.9 %, not 20.7 %**. Verified four
ways (max speed-source disagreement 0.002 m/s).

**Root-cause class: a caveat measured on ONE corpus surface (R0) quoted as a property of THE METRIC.**
It was true where it was measured and false where I applied it — and because it was a *caveat*, it
read as conservative rather than as a claim needing its own evidence.
⇒ **RULE: a stratification caveat carries its surface, exactly like a throughput number carries its
endpoints.** "n=2 in the top band" is a fact about a dataset, never about an instrument.
⇒ **RULE: conservative-sounding claims still need evidence classes.** A caveat that discourages work
can do as much damage as an optimistic claim that invites it.

### 3. Also fixed here — an instrument that could not certify ANY Thor run

Both Thor efficiency runs came back quarantined as `.CONTAMINATED-*`. **Not contention:** Tegra's
`nvidia-smi` returns `[N/A]` for `memory.used`, `float()` raised inside `_gpu_state`'s **single
shared `try`**, and the `--query-compute-apps` probe on the following lines **never ran**, leaving
`exclusive = null` — not `False`. `taniteval.efficiency` was **structurally unable to certify any
Thor run**, and would have reported that as contention forever. Fixed with a defensive `_num()` and
independent try blocks, +2 regression tests.
⇒ **RULE: one `try` around several probes converts a failure in the first into silence from all of
them.** Same family as the bare `except` that hid the `min_steps` TypeError and made a nav fallback
that had never once executed look like a working default.

---

## R-2026-08-04-vt — ⛔ A MEASURED ABSENCE THAT WENT STALE IN ONE DAY, and a caveat that was already retracted and reached me anyway

**Stream:** Architecture & Inference / D-VT1 (leak-guarded target speed).
**Artifacts:** `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-04-target-speed/`.

### 1. *"The parity train cache is not reachable from any non-training host"* — FALSE by 2026-08-04

I wrote this into an escalation, inheriting `PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md` §1.1, which
had established it on **2026-08-03 with five probes at four paths** — correctly, at the time.

**MEASURED 2026-08-04:** `tanitad-thor` holds
`/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894`, **2376 episodes**,
file-list sha256 `9877bef6…7386` — matching the parity-verified corpus uid. The guarded target-speed
labels for the whole train corpus were minted from it in **22 s**
(`raw/train_vtarget_guarded.npz`, 472 627 pose indices).

**Root-cause class: NEW — C2-adjacent but distinct. Call it C2b, a STALE ABSENCE.**
C2 is *"absence from a single probe"*. This was the opposite failure: a **thorough, multi-probe,
correctly-measured absence** that was re-quoted **one day later** without re-probing. The corpus
moved; the finding did not. A five-probe absence is *more* dangerous than a one-probe absence,
because its rigour makes re-checking feel unnecessary.
⇒ **RULE: an absence claim carries a TIMESTAMP as well as a probe count, and a claim that a
*resource* is missing expires the moment anything could have moved it.** Absence of a fact is
durable; absence of a *file on a machine* is a snapshot. Re-probe before quoting, however good the
original measurement was.
⇒ **What it would have cost:** the escalation would have told the REF-C DATA stream that the
train-corpus mint was still blocked. It was 22 seconds of work.

### 2. The 0–1 m/s / n=2 lead-window caveat reached a third document — the log did not stop it

`R-2026-08-04-briefs` §2 already retracted *"20.7 % of lead windows sit at 0–1 m/s … the 15+ band is
UNPOWERED at n=2"*. It arrived in my brief anyway and I copied it into a JSON before catching it.

**RE-MEASURED** directly from `val40_lead_block.npz` (`state` × `speeds`, 270 LEAD windows):
0–1 m/s = **32 (11.85 %)**; 15+ = **88 (32.59 %), the LARGEST lead-bearing band**. ⚠️ The genuinely
low-powered band is **10–15 m/s at n = 12**, which no document had flagged.

**Root-cause class: C4 with a twist — a RETRACTED claim propagating through briefs.**
The log recorded the retraction; the brief did not read the log. ⇒ **RULE: retraction is not
complete until the claim is removed from the briefs that carry it.** An append-only log corrects the
record but does not corrupt the copies already in flight — the entry needs a blast radius, like the
`overlapping_holdout_se` entry has.

## R-2026-08-05-crlf — "~2.7 KB of unrecovered Google Drive doc edits" — RETRACTED, it is CRLF

**Retracted:** the session handoff's **priority item 2**, *"recover ~2.7 KB of Google Drive doc
edits"* (`PROJECT_STATE.md` +395 B, `Paper/TANITAD_PAPER.md` +1,865 B, `README.md` +369 B,
`.gitignore` +51 B).

**MEASURED 2026-08-05: every delta equals that file's LINE COUNT, exactly.** The two large files
were byte-compared against the raw Drive bytes and are **identical after `\r\n → \n`** — 101,754 ==
101,754 and 157,043 == 157,043. `.gitignore` matches once measured at `7f34086`, the pre-session
state (910 + 51 = 961); it is 1,016 B at HEAD only because this session added `**/gotty_url.txt`.
**There are no unrecovered edits.** Detail: `Project Steering/DRIVE_DOC_DELTA_IS_CRLF_2026-08-05.md`.

**Root-cause class: a SIZE or TIMESTAMP difference read as a CONTENT difference.** This is the
**third** CRLF incident of the session — the v5 trainer pin went red on a file nobody had edited
(the CRLF hash reproduced the old pin, the git blob was identical on both refs), and the same class
sits in `CLAUDE.md` already. What made this one convincing was that *four* files agreed and both
signals pointed the same way: Drive was **bigger** and **later**. Both were true; neither implies an
edit.
⇒ **RULE: before sizing work from a byte delta, check whether the delta equals the newline count.**
One `wc -l` closes it. Normalise line endings before any cross-store diff.
⇒ **RULE: `read_file_content` on the Drive connector is NOT the file.** It returns a natural-language
representation — on `PROJECT_STATE.md` it gave **106,585 B for a 102,149 B file**, escaped the
markdown (`\#` for `#`) and added trailing spaces, so a diff built on it showed **293 of 395 lines
differing**. Acting on that would have written hundreds of phantom changes into the repo. Use
`download_file_content` (base64 of the real bytes) for anything byte-level.

## R-2026-08-06-yawgate — the TACTICAL κ ranking of Alpamayo vs flagship — RETRACTED, it was the gate

**Retracted:** from the four-family Alpamayo comparison committed at `340333d`
(`…/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md` §12,
`comparison/a2_four_families.json`), the claims that on 39 paired OOD-val clips our arm's
**executed-manoeuvre κ was 0.4968 against Alpamayo's 0.3333**, that Alpamayo's **declared** lateral
manoeuvre was weakly coupled at **κ 0.1488**, and that our arm **"drove 0 of 2 left turns"**.

**MEASURED 2026-08-06 (`comparison/a2_gate_audit.json`), two independent defects, both in the
instrument:**

1. **Net yaw was summed over steps where the ego was not moving.** At `v ≈ 0` the path tangent
   flips freely; one stopped window contributed a net yaw of **π**. Excluding steps below
   `MIN_DS_MPS = 0.5` — as `four_families._seq_geometry` already does — moved Alpamayo's executed κ
   from **0.3333 to 0.4882** on its own.
2. **`DIR_YAW_RAD = 0.15` is ~6.5× the typical turn on this horizon.** The **human's own** median
   |net yaw| over 2 s is **0.023 rad**, p90 **0.185**, and only **17.9 %** of windows exceed the
   gate. Nearly every window classifies as "straight" by construction.

**Corrected reading.** At gate **0.10** the two arms are **indistinguishable on executed manoeuvre
— κ 0.7263 (ours) vs 0.7292 (Alpamayo)**. The published ranking was an artifact. And the genuine
finding was invisible at 0.15: as the gate tightens the two **declarations move in opposite
directions** — Alpamayo's rises 0.196 → 0.466 while ours falls 0.440 → 0.116, i.e. its declaration
carries fine lateral information (the "nudges" its own Chain-of-Causation names) that our gate
discarded, and ours carries only coarse information. Gate-free sign-only agreement is **0.7143 for
both**, over **n = 21** declared turns for Alpamayo against **n = 7** for ours.

**Root-cause class: A THRESHOLD QUOTED AS IF IT WERE A MEASUREMENT.** A classifier gate is a free
parameter of the instrument. Reporting one κ at one gate, with no sweep, publishes the gate's
opinion as the model's. This is the same family as *"never quote a learning-curve exponent without
its fit window"* and *"never quote an interval without its estimator"* — a number whose value is set
by an unstated analysis choice.
⇒ **RULE: any classification threshold that decides a reported statistic must be SWEPT, and the
sensitivity published with the number.** If a verdict flips inside the plausible range of the
threshold, there is no verdict.
⇒ **RULE (second, and it is the third sighting of this class): a quantity undefined in a regime must
not be aggregated over that regime.** `df` reporting the cluster instead of the pod quota; curvature
at `v ≈ 0` returning 1.6 × 10⁶ 1/m; yaw at `v ≈ 0` returning π. Gate by displacement before summing
any path-tangent quantity.

**⛔ Blast radius — NOT confined to this document.** `DIR_YAW_RAD` is `taniteval/hierarchy.py:164`
and feeds `consistency.maneuver_vs_trajectory`, `commanded_route_vs_maneuver`,
`commanded_route_vs_trajectory` and every `*_turn_subset` in the hierarchy panel — **every published
manoeuvre-coherence κ in the programme**, all on the same 2 s horizon over the same corpus.

⭐ **RESOLVED for the deployed arm, 2026-08-06, and it lands in the panel's favour.** The
`flagship-v1arch-v2bal-30k` panel was re-run gate-swept (step 29999, **880 windows**, 40 OOD-val
q90 episodes, 141 s; `…/2026-08-06-v1-defect-triage/results/`). `verdict_stable = true` —
κ stays at or above the panel's own 0.2 coherence threshold at **every** swept gate
(`kappa_range [0.2038, 0.5787]`). **The published coherence call was NOT an artifact and is not
retracted.**

⚠️ **The number still does not travel.** 0.5787 is true only at 0.15; the same model on the same
windows scores 0.2038 at 0.01 — a **2.8×** span. ⇒ the VERDICT is quotable, the MAGNITUDE is not,
ever, without its gate. And `kappa_turn_subset` at the published gate is **0.2005** — sitting *on*
the threshold, so the comfortable headline is carried by the straight-dominated majority; on the
windows where a direction decision actually exists, coherence is marginal.

⚠️ **A second correction, by power rather than by error:** this retraction quoted the flagship's
declared-vs-driven κ as **0.3432** from 39 single windows. The 880-window panel reads **0.5787** at
the same gate. Quote the panel; the 39-clip figure was under-powered.

**Still open:** REF-B / REF-C / v2corpus panels carry unswept κ. They now report
`gate_sensitivity.status = UNAVAILABLE` with the reason in their own output, so the gap is visible
rather than silent. Re-running them is cheap but **not urgent — no live decision rests on an
unswept number now that the deployed arm's verdict has held.**

**How it was caught, because the route generalises:** the low κ was checked against the obvious
confound before being reported — Alpamayo's own CoC says *"Nudge left to pass the parked SUV"*, and a
nudge is not a 0.15 rad turn. ⇒ **Before reporting a coherence statistic as low, ask what scale the
declaration is about and whether the instrument can see it.**

## 2026-08-07 — v5f review: two claims corrected same-day

- **Claim:** "v5f has no goal/route conditioning of candidate generation." **FALSE** —
  `flagship_v15.py` defines cond_vtarget + cond_route generation tokens (ReZero-gated,
  goal-dropout 0.5). **Class: absence asserted from ONE file (the v4 wrapper) without
  probing the inherited class** — the CLAUDE.md rule-2 class, inside a review that cited it.
- **Claim:** "horizon = 2 s." **Incomplete** — dense 2 s operative + 5 s tactical knots
  (TACTICAL_HORIZONS). Class: config read from one function, not the whole config path.

## 2026-08-09 — eval frame-args "fix" was a duplicate: the flags already existed

- Added --frame-h/-w/-hfov/--f-ref/--projection to eval_flagship_v4 believing they were
  missing; they are provided by `tanitad/geometry.py:279`'s shared helper — the duplicate
  crashed argparse. **Class: absence asserted after grepping ONE file (the script) for
  add_argument, missing the shared helper** — rule-2, third instance this week. The real
  eval2 failure was my own invocation (passed --v2-subframe without the --frame-* flags
  that already existed).
## 2026-08-15 — "batch_00184 (8 clips) has no SAM3" was the visible 7 % of a 57 % gap

**Claim (runbook §6.11, written at pod stop):** the aug120 SAM3 gap is **`batch_00184`, 8 clips**,
whose SAM3 stage never produced a directory. **Incomplete by 14×.** MEASURED at fusion against the
far side of `Sayood/tanitad-ph0-aug120`: **115 of 201 clips (57.2 %) have no SAM3 record at all**,
in *every* batch, not one.

**Root cause of the underlying defect:** `aug120_pipeline.py` passed `--n` to the bridge and to the
VLM and **omitted it for SAM3**, whose default is **4** (`ph0_sam3.py:387`, consumed at `:411`
`[:a.n]`). Every batch got SAM3 on its first 4 clips — verified per file: 25 sam3.json × exactly 4
records, and those 4 are exactly the first 4 of that batch's v2 order. The stage printed
`SAM3_RC=0` throughout.

**Root cause of the WRONG CLAIM (the part that generalises):** `batch_00184` was found by a
**structural** probe — a far-side listing showed one batch prefix with 1 file where every other had
2. That probe can only see a **missing file**, never a **short file**. The 24 batches whose SAM3
stage silently covered 4 of 8 or 4 of 40 clips all had a sam3.json present, so they looked complete
at exactly the granularity the probe operated at.

⇒ **NEW CLASS — C18: a defect found by a probe is bounded by that probe's granularity.**

| # | class | recognition signal |
|---|---|---|
| **C18** | **Defect scoped by the probe that found it** | *"the gap is N items"*, where N came from listing CONTAINERS (files, directories, repos) rather than counting their CONTENTS against what was submitted |

**The check that settles it is a conservation count: `n_out == n_in` per stage, per batch.** Here it
was one line — `len(sam3.clips)` vs `len(v2.clips)` — and it turned 8 into 115.

⚠️ Sibling of **C2** (absence from a single probe) and a first cousin of the `df` / `tegrastats` /
`memory.usage_in_bytes` family, but distinct and worth its own number: those are probes reporting
the **wrong scope**; this is a probe reporting the **right scope at the wrong resolution**, which is
harder to spot because the answer it gives is *true* — just not the whole count.

⇒ **RULE: every pipeline stage writes its own coverage record (`n_in`, `n_out`, the ids it skipped)
into the batch it pushes.** An `rc=0` is not coverage. This is now the standing requirement for the
4,472-clip build (`…/incoming/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md §2`).

**Second, smaller correction found in the same pass:** the published val-600 fusion summary reads
`n_v2: 600, n_sam3: 596`, and under the fuser as it then stood those **4 clips were fused with a
silently empty perception layer** — no marker, and their `census_vs_scene` / `goal_evidence`
verdicts were computed from a detector that never ran. Same class as the `0/600` join defect pinned
in `2da0799`, one level down. The fuser now **refuses** a partial SAM3 leg unless the operator names
it (`--missing-sam3-ok REASON`) and stamps `perception.absent` per record; the SAM3-dependent checks
return `not_computable` instead of a fabricated verdict. **`fused_w120val/` on HF still carries the
4 unmarked records** — correcting them would re-baseline the published 175/41/56, so it is flagged,
not silently redone.

---

## 2026-08-16 — ⛔ NEW CLASS C68: A SHARED APPEND-ONLY ARTIFACT DOES NOT IDENTIFY ITS PRODUCER

Three errors in one hour, all mine, all while trying to answer the PI's simplest question of the
night — *"measure the speed of training on the Thor and compare it to the A40"*. One was asserted
and retracted; two were caught before they were quoted. They share a single root cause, and it is
not any of C2 / C18 / the `df` family.

| # | what I did | why it was wrong |
|---|---|---|
| 1 | claimed two `V6Stack` state_dicts were identical from the **md5 of `torch.save` output** | `torch.save` writes a zip container whose bytes are **not canonical**; a mismatch would have been a false alarm and a match proves the container, not the tensors. Correct method, then run: iterate keys and `torch.equal` — **405 keys, 0 differing**. |
| 2 | wrote a waiter filtering the resumed run's rows with `step > 6250` | `train_log.jsonl` is **APPENDED ACROSS PROCESSES**. The banked A40 run ends at step **6300** and the Thor resume starts at **6250**, so that filter admits a banked row and the "marginal s/step" becomes a delta **between two machines**. |
| 3 | wrote a measurement script that labelled the last log segment `THOR` | with the resume not yet having logged, the *only* segment present was the **A40's own**, and the script printed `THOR MARGINAL 17.45 s/step` — the A40's rate, under Thor's name. |

⇒ **C68 — the artifact you sampled does not carry the identity of the thing you are claiming about.**

| # | class | recognition signal |
|---|---|---|
| **C68** | **Shared artifact, unstated producer** | a rate, delta or identity computed across a file that **survives restarts** (`*.jsonl`, `metrics.json`, a serialized container), filtered by a key that **resets or overlaps** on resume — `step`, `epoch`, `idx` — rather than by something that identifies the *process* |

**The check that settles it is a producer discriminator, not a value filter.** Here it was already in
the file: `train_v6_staged.py:1406-1411` logs `step_s` as `(now − t0) / (step − start_step)` and
writes a `step_s_note` naming its own divisor. Segmenting on that divisor **resetting** separates the
two processes exactly; segmenting on `step` cannot, because step numbers legitimately repeat.

⚠️ **Sibling but distinct.** C2 is absence found at one location. C18 is the right scope at the wrong
resolution. The `df` / `tegrastats` / `memory.usage_in_bytes` family is the wrong *scope*. **C68 is
the right scope, the right resolution, and the wrong PRODUCER** — which is the hardest of the four to
see, because every value you read is real and internally consistent.

⭐ **AND IT ALREADY HAD A NEAR-MISS IN THIS PROGRAMME.** `step_s` was itself the subject of the
standing "÷ `--log-every`" trap in `CLAUDE.md` — the trainer author foresaw the *arithmetic* misread
and wrote a self-describing note against it, but not the *cross-process* one. A note that documents
its own divisor is exactly what made the fix one line; **the general rule is that any metric written
by a restartable process should record the identity of the process that wrote it.**

⇒ **RULE: never compute a delta across a restartable log without first segmenting on a producer
discriminator, and never quote a rate from a single process segment without saying which one.**
Applied here: the corrected instrument (`~/thor_measure_sstep.py`) refuses outright — exit 1, quoting
nothing — until a **second** segment exists, and reports marginal-vs-marginal rather than
marginal-vs-cumulative, because the A40's own end-of-run marginal (**19.68 s/step** over its last 300
steps) is **13 % above its lifetime mean (17.46)** and comparing across those two is a third way to
get a real number that answers a different question.

---

## 2026-08-16 — "the Alpamayo `trajectory` task's metric block never ran on 4,474 of 4,729 rows" — RETRACTED, they carry a differently-named metric

**Claim (`2026-08-15-2200-campaign-science-addendum.md` §2.5, and copied from it into the first
draft of `MODEL_REGISTRY.md` §11.1):** *"Only 255 of 4,729 trajectory rows carry `min_ade_m` (the
rest have `num_trajectory_samples: None`, i.e. the GT-dependent metric block is absent)."*

**MEASURED on re-verification** (`…/scratchpad/reverify_a2_counts.py` → `a2_reverify.json`, an
independently written probe over the same sha256-verified `records.parquet`): the `raw_json` column
holds **two disjoint schema variants**, and the 4,474 rows are **not** metric-free.

| variant | rows | GT-referenced error | waypoints |
|---|---|---|---|
| A | **4,474** (94.61 %) | **`ade_vs_gt_m` on 4,474/4,474, 0 null** | **`pred_xyz` = 64 points on 4,474/4,474** |
| B | **255** (5.39 %) | `min_ade_m` / `min_fde_m` on 255/255 | `pred_xyz_shape` only — ⛔ **no `pred_xyz`** |

`num_trajectory_samples` is **absent entirely** from variant A (255 hits over the whole column), so
*"the rest have `num_trajectory_samples: None`"* is false as stated. `ade_vs_gt_m`: **n 4,474 ·
mean 2.2584 m · median 1.5245 m**, corroborating variant B's `min_ade_m` (2.3469 / 1.5233) at
**17.5× the n**. ⚠️ Whether the two are the **same estimator** is 🟥 UNVERIFIED — do not pool them.

**What it cost / would have cost:** the claim **understated the usable trajectory data by 17.5×**
and was about to be published in the paper as a bound on what the augmentation set supports. Any
future A2 trajectory analysis would have been scoped to 255 clips when **4,729** are available
(4,474 with waypoints + 255 with the metric block).

**Root-cause class — C18-family, and worth naming as its own recognition signal:** a
**key-presence probe over a HETEROGENEOUS column**. The probe asked *"does `min_ade_m` /
`num_trajectory_samples` appear?"* and read *no* as *"the computation never ran."* A key-presence
probe can see a key MISSING; it **cannot see a differently-named key answering the same question**.
This is the same defect as `batch_00184` (a listing sees a missing file, never a short one) at the
level of JSON keys instead of files — the probe was operating at the right scope and the right
resolution, but over a column it wrongly assumed was **homogeneous**.

⇒ **RULE: before reading a key's absence as a fact about the world, census the SCHEMA VARIANTS of
the column.** One line — `Counter(frozenset(json.loads(x)) for x in col)` — separates "the value is
missing" from "this row is a different record shape." Sibling of **C2** (absence from a single
probe): here the second probe is not a second *location* but a second *shape*.

*(Same pass also confirmed, not retracted: 23,644 rows / 4,729 clips / one `NF4-…-UNVALIDATED`
quantisation arm / 78.36 wall-hours / 0 errors, the 356× dataset-card understatement, and the
stratification being delivered 100 % complete on all four labelled classes.)*

---

## 2026-08-16 — "no banked latents exist in the repo" — RETRACTED. `find -maxdepth 4`, files at depth 6

**What I asserted**, in `…/incoming/2026-08-16-selector-capacity-control/SELECTOR_CAPACITY_CONTROL.md`
and in an agent brief that then shaped a whole task: *"two independent probes (dev box + Thor) find
only REF-C latent dumps … E-WC2 now needs a GPU pass at a deliberate training pause."*

**What is true.** The REF-C latents are **in this repo**, at

```
TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
    2026-08-04-lambda-findability/raw/latents_refc-{base,xl}-30k.pt     39.5 MB / 26.7 MB
```

— **six directories deep.** My dev-box probe was `find . -maxdepth 4 -iname "*latent*"`. It returned
three unrelated hits and **could not have reached the files at all**. I then described it as one of
"two independent probes", which is how a *configuration limit* got promoted to a *corroborated
absence*.

⇒ **This is C14 in a search command.** C14's binding lesson is *"before recording a limit, ask
whether the instrument could have reported a LARGER value"* — there it was a sweep whose grid string
ended at 12°, here it is a search whose depth bound ended at 4. Both report the shape of the probe
and get read as the shape of the world.

⚠️ **And it defeats C2 while appearing to satisfy it.** C2 says *absence at one location is not
absence — use a second probe*. I ran a second probe (on Thor), it agreed, and **both were wrong in
the same direction for different reasons**: the dev-box one was depth-bounded, and the Thor one was
looking at a machine that legitimately has no repo checkout of that directory. **Two probes that
share a blind spot are one probe.** The check that would have caught it is trivial: an unbounded
`find` on a distinctive filename.

| # | class | recognition signal |
|---|---|---|
| **C69** | **A search whose own bound excluded the answer** | `find -maxdepth N`, `head -n`, `grep -m`, a glob that stops at one directory level, an API `limit=` — any enumeration reported as "not present" without first asking whether the enumerator could have reached it |

**What survives, and it is most of it:** *no frozen S-W latents have ever been dumped* is **still
true** — the S-W surface §5.2 names does need ~10–25 GPU-min at a deliberate pause. **What is
withdrawn is the cost verdict**: the REF-C route, which is the arm the 1.7/3.0 thresholds were
derived on, needs **no GPU whatsoever**. Its one missing input is the **val40 pose arrays** for the
6 s ground-truth endpoint — a pose-only backfill.

⇒ **The deeper root cause, and it is the reusable half:** §5.2 priced *"0 GPU, banked latents"* by
costing the **expensive** input (latents, a GPU dump) and never costing the **cheap** one (poses, a
file read). My retraction repeated the same shape from the other side — I re-priced the expensive
input and still never asked what else the estimator needed. ⇒ **RULE: when a plan says a step is
free, enumerate EVERY input it consumes and locate each one, not just the one that sounds costly.**

⇒ **RULE: never write "X does not exist" from an enumeration whose bound you have not stated.**
Quote the exact command, and prefer an unbounded search on a distinctive name over a bounded one on
a generic pattern.

## R-2026-08-16-stale-blockers — "our ingest reads 5 of 36 features" — RETRACTED, it is 6; and a whole agent was commissioned against a blocker that cleared 13 days earlier

**Retracted (two claims, one root cause):**
1. `CLAUDE.md` — *"our ingest reads **5** of 36 features"*. **It is 6.**
2. `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-76` — *"Until that lands, **arm
   evals will still report the family UNAVAILABLE**"*. **It landed 2026-08-03, the day the package was
   filed**, and has been measured through 2026-08-14.

**MEASURED 2026-08-16, from source and raw artifacts in this repo:**
`obstacle.offline` is a real read — `stack/scripts/build_obstacle_join.py:148`,
`stack/scripts/lead_state_gate.py:308-338`, `stack/tanitad/data/bev_raster.py`. The eval-path wiring
is `taniteval/taniteval/lead_source.py` + `taniteval/tools/build_lead_block.py`, and it was fed on
exactly the 40 val episodes the INTAKE named:
`…/incoming/2026-08-04-instrument-durability/raw/val40_lead_report.json` — **`n_episodes: 40`,
`canonical_881: true`, registration `n_ok: 40 / n_failed: 0`, `LEAD 270 / NO_LEAD 551 / NO_LABEL 60`**.
`Project Steering/MODEL_REGISTRY.md:1187-1190` had already struck ~~`distance_keeping` UNAVAILABLE~~.

⭐ **The two retractions are ONE event.** `obstacle.offline` becoming a read on 2026-08-03
invalidated both statements simultaneously. Neither was revisited. **What it cost:** a whole agent
commissioned on 2026-08-16 to build an instrument that had existed for 13 days.

**Root-cause class: NEW — C70, a STALE BLOCKER.**
C2 is *absence from a single probe*; C2b is *a correctly-measured absence re-quoted after the world
moved*. **C70 is the forward-looking sibling: a statement of the form "blocked on X" / "until X
lands" / "not yet fed", which is a claim with an EXPIRY DATE and no alarm attached.**

⚠️ **The mechanism is a ONE-DIRECTIONAL REFERENCE, and this is the reusable half.** The closing work
*did* cite the blocker — `lead_source.py`'s docstring names this INTAKE's open work item as its
reason for existing. **The successor knows about the predecessor; the predecessor never learns it has
been superseded.** So the corpus accumulates confident, well-evidenced, obsolete instructions. The
2026-07 orthogonality instrument (10 days unmerged, its merge request in a README nobody re-read) is
the same shape.

| # | class | recognition signal |
|---|---|---|
| **C70** | **A stale BLOCKER — a "blocked on X / until X lands / not yet" line never revisited when X landed** | any doc sentence whose truth depends on a *future* event, with no artifact named whose existence would falsify it. Danger scales with the doc's authority: an INTAKE or a protocol misdirects work; a dated report is only history. |

⇒ **RULE: a "blocked on X" line MUST NAME THE ARTIFACT whose existence clears it** — a path, not a
prose description. Then the check is `test -f`, and a sweep is mechanical rather than a re-reading.
⇒ **RULE: before commissioning work against a blocker statement, re-probe the blocker.** The
statement's own rigour is not evidence it is still true — that is exactly C2b's lesson pointed
forwards.
⇒ **RULE: an empty ORCHESTRATOR VERDICT block is NOT a rejection.** The distance-keeping package
landed and is in production use with its verdict block still blank; reading blank as "not accepted"
is a second way this class misdirects.

**A fourth-order twist worth recording.** The feature count had *already* been retracted twice
("2 of 36" → 4 on 2026-07-26 → 5 on 2026-08-16) **inside the rule that warns about stale
absence-claims**, and the 2026-08-16 entry itself prescribed the fix — *"pin it so the suite fails
instead of a document rotting"* — and then did not do it. ⇒ **A retraction that names its own remedy
and stops short of implementing it will be re-retracted.** The remedy now exists:
`stack/tests/test_physicalai_feature_readset.py` (9 tests, green) asserts the counts, the exact
feature names, the 36 denominator (from the MEASURED `pai_features.csv` probe), and carries a
drift detector *proved able to fail*. Its failure messages name the documents to update.

⚠️ **And the number was ambiguous, not merely wrong — which is why hand-editing kept failing.**
"Our ingest" was never defined. It is **2** for `physicalai_r0.py` (clip selection), **5** for
`physicalai.py` (the episode build), **6** program-wide including the pod-side join. ⇒ **RULE: state
the LAYER with the count, never the bare phrase "our ingest".**

**Blast radius (this entry needs one, per the R-2026-08-04 rule):** the stale count reached **14
documents (17 sites) plus a code docstring** (`stack/tanitad/data/bev_raster.py:12` — *a stale count
in code outlives one in prose, because nobody greps docstrings*). Highest-authority carriers:
`Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md:143` (a **protocol**, and doubly stale — wrong
count *and* cleared blocker), `Project Steering/V6F_PLANNER_DESIGN.md:536`,
`Project Steering/Gates/flagship-v5-retrain.PREP.md:58` ("32-of-36" → 30-of-36).
Full sweep + verdict tables: `…/incoming/2026-08-16-stale-blocker-sweep/STALE_BLOCKER_SWEEP.md`.

### C70 — what the full-corpus sweep added (same day, 3 parallel streams, 67 claims verified)

**C70a — A STALE POSITIVE STATUS. The inverse failure, and it is worse.** `BACKLOG.md` C2 /
`BOOST_PROGRAM.md` S-3 / `MODEL_REGISTRY.md` §1.7 all carry v2corpus as **"🟢 RUNNING"** with an ETA
of **2026-07-29 — 18 days past** — and no completion row. **A stale blocker makes you rebuild
something; a stale "running" makes you WAIT FOREVER**, and it never trips the "is this still true?"
instinct because it reads like good news. ⇒ **RULE: a status of RUNNING carries an ETA, and an
expired ETA is a DEFECT, not a delay. Probe or downgrade it.**

**C70b — A WRONG PATH MAKES A BUILT INSTRUMENT LOOK UNBUILT.** `MODEL_REGISTRY.md:1583,1644` cite the
distance-keeping instrument as `tools/build_lead_block.py`. That path does not exist (three probes);
the file is `taniteval/tools/build_lead_block.py`. ⇒ **This is a SECOND, INDEPENDENT mechanism aimed
at the SAME instrument as the stale blocker above.** Today's wasted commission had two sufficient
causes, either of which alone would have produced it. ⇒ **RULE: a citation is a claim. `test -f` it.**

**C70c — A BLOCKER CAN BE STALE THE DAY IT IS WRITTEN.** Two cases, both **same-day**:
`ROADMAP.md:68,112,297` said `stack/tanitad/scena/` *"does not exist"* — it was committed the same
day (`9ebfb09`); `REPO_TRIAGE_2026-07-20.md:67,287` said *"`tools/` does not exist in HEAD at all"* —
16 files + 10 tests landed the same day (`c4d8451`, `1e13e3a`). ⇒ **Expiry is not proportional to
age. Never treat a recent doc as safe.**

**C70d — the survival curve is BIMODAL, which refutes a prior conclusion.** The 2026-07-26 program
harvest concluded *"most of today's stranding is same-day"*. **REFUTED: 9 of its 12 open items are
still open 21 days later.** ⇒ **What clears, clears fast; what does not clear that day tends never to
clear.** So the triage question is not *"how old is it?"* but *"has anything touched it since day
one?"* ⇒ **RULE: an item still open at day 2 needs an OWNER, not a re-read — it will not self-clear.**

**C70e — a HALF-MERGE defeats the existence probe.** `--nav-known-channel` parses at
`stack/scripts/refc_train.py:1221,726` but `:402` never passes `nav_known=`, so
`stack/tanitad/refs/refc.py:2005` raises on the first forward pass. *"Does the flag exist?"* returns
**yes**; the thing is broken. ⇒ **RULE: verify a merge by RUNNING the path, not by finding the symbol.**
*(Same family as the sibling lesson that green-by-import is not green-by-execute.)*

⚠️ **And a refuted PREMISE does not clear the CLAIM it supported.** `ROADMAP.md:104` blocks a CARLA
dry-run on "a graphics-capable pod"; the Vulkan premise died with C2 and AlpaSim ran bare on an A40 —
but **no CARLA run exists**, so the blocker stands. Verdict **PARTIAL**, not CLEARED. Recording it as
CLEARED would have been a stale "it works now", which is the same defect pointed the other way.

**Sweep totals (MEASURED 2026-08-16):** 3 streams · 48 `INTAKE.md` + 121 `Project Steering/*.md` +
~507 `incoming/**` docs · **67 claims adjudicated, 30+ CLEARED and annotated in place, 8 still-open
integration requests surfaced** (oldest **24 days**, one of which had its gating condition — "the next
v4.x launch" — voided by the programme reaching v6).

---

## 2026-08-16 — ⛔ NEW CLASS C71: A GUARD THAT FIRES FOR A REASON UNRELATED TO WHAT IT CHECKS

> ⚠️ **RENUMBERED C70 → C71, same day.** Two agents appended a "new class C70" within minutes of
> each other — this one and the **stale-blocker** class above, which had already lettered five
> sub-classes (C70a–e). Renumbering the un-lettered one was the smaller edit. The commit that
> introduced it (`545c98d`) still says "C70" in its message; **this header is the authority.**
> ⇒ *The append-only log has no allocator, so concurrent classes can collide. Grep the file for the
> next free number immediately before appending, and re-grep if the append is not immediate.*

Executing the five v6 ladder edges surfaced **three defects on the resume path**, and the priority
one is a guard that had been "working" for reasons that have nothing to do with the thing it
appeared to protect.

**What appeared to be true.** A cross-stage `--resume auto` was impossible: point S-T at an S-W
checkpoint and it errors out. **What was actually true:** `load_resume` did a strict load and adopted
`ck["step"]` with **no stage check at all**. The load *succeeds* — every stage saves the whole
`V6Stack` — and the stage label was **already in the file** (`_run_config` writes it) and simply
never read. The error came from `torch.optim` complaining about param-group sizes, which is worthless
three ways:

1. it names nothing actionable — the operator is sent to the optimiser for a lineage problem;
2. it holds **solely because trainable-tensor counts happen to differ** (MEASURED: S-W **240** ·
   S-T **80** · S-S **54** · S-J **374**), so **one `STAGE_GROUPS` edit makes it pass silently**;
3. it is **SKIPPED ENTIRELY when the checkpoint has no `opt` key** — which is exactly the shape of
   `ops/ckpt_fp16_snapshot.py`, the documented pod-handover artifact. On the handover path the guard
   did not exist at all.

And it fired **after** the corpus build and the O4 saliency pass over every window.

| # | class | recognition signal |
|---|---|---|
| **C71** | **Guard fires for an unrelated reason** | a check believed to protect X, whose actual trigger is Y — where Y merely *correlates* with X today. Signals: the error message names a subsystem the operator did not ask about; the guard's mechanism is never stated anywhere; nobody has tested it with X true and Y false |

⇒ **The test that separates them is the one the fix now carries:** construct the case where the
*incidental* trigger is DEFEATED but the real condition still holds. Here that is a checkpoint
labelled S-T whose optimiser was built over S-S's trainable set — the shapes collide, the accidental
barrier is gone, and the new stage check refuses anyway **because it never looks at an optimiser**.

⚠️ **Family, and it is a large one:** `df` reporting the cluster instead of the pod quota; Thor's
`free`/`tegrastats` moving 596 MB for 60 GB of unified allocation; cgroup `usage_in_bytes` counting
reclaimable page cache. All are signals that track the quantity of interest **until they don't**.
C71's variant is nastier because it is a *guard*: the others merely misinform, this one manufactures
confidence that a dangerous path is closed.

⚠️ **Distinct from C13** (*a guard that cannot fail*). C13's estimator saturates below its own
threshold and can never fire. **C71 fires reliably — for the wrong reason** — so it looks like
positive evidence the check works. C13 produces confident silence; C71 produces confident noise.

**Two silent siblings found in the same pass, both now fixed:**
- **A provenance lie.** `train()` loads `--init-from` then `--resume`, so the resume overwrites
  everything while `config.json` kept the *init's* md5 — MEASURED `fbce009a…` recorded against
  `326034884…` actually in the model, **with no warning**. Refusing the flag pair would be wrong
  (`supervise_run.sh` replays its captured command, so a resuming relaunch necessarily still carries
  `--init-from`), so the report is **demoted**, not deleted.
- **An error message pointing at the wrong subsystem.** `--init-from` on an fp16 snapshot refused
  with a **400-key geometry mismatch** — sending the operator to the *architecture* for a container
  that was simply never opened (state under `"model"`, reader looking for `"stack"`). ⇒ **RULE: when
  a load fails, distinguish "I could not READ this container" from "the CONTENTS disagree" before
  the message names a cause.**

⇒ **RULE: for any guard that matters, write down its MECHANISM, then test it with the real condition
true and the incidental trigger removed.** A guard nobody has seen fire for the stated reason is a
hypothesis, not a protection.

---

## 2026-08-16 — ⛔ NEW CLASS C72: AN EXACTNESS CHECK WHOSE EXACTNESS IS A PROPERTY OF THE MACHINE, NOT OF THE CLAIM

*(Next free number re-grepped immediately before appending — C70/C71 collided earlier today.)*

The E-WC2 endpoint backfill carries a gate that refuses unless the recomputed 2 s endpoint is
**bit-identical** to the banked one, because without it every latent would regress onto a
*neighbour's* endpoint and return an inflated σ that looks like a measurement. Correct instinct.
**The gate then refused a CORRECT backfill.**

The dumps were produced on **Thor (aarch64)**; the backfill ran on **x86**. `cos`/`sin` differ in the
last bit. MEASURED: **825/881 rows bit-identical**, max disagreement **7.63e-06 m = 1.118 ULPs** —
while the failure the gate exists to catch, a **±1-row shift, is 0.5123 m ≈ 4e5 ULPs**. The two are
separated by **67,139×**. `torch.equal` cannot see that gap; it sees only "not equal".

| # | class | recognition signal |
|---|---|---|
| **C72** | **Exactness that belongs to the machine, not the claim** | a `torch.equal` / `==` / md5 gate on **recomputed floating point**, especially across architectures, compilers, or BLAS builds. Signal: the check's tolerance is 0 while the defect it guards against is orders of magnitude larger |

⇒ **The fix is NOT a loosened tolerance.** A bare bit-identity check also passes **VACUOUSLY** on a
degenerate block where every shift happens to match — so relaxing it would trade one blind spot for
two. The replacement is **last-bit agreement PLUS a ±1-row POSITIVE CONTROL that did not exist
before**: the gate must demonstrate it can still *detect* the misalignment it was built for. That
control is the part worth copying; the tolerance is incidental.

⚠️ Sibling of **C13** (a guard that cannot fail) and **C71** (a guard that fires for an unrelated
reason). C72 completes the set: **a guard that fires for the RIGHT reason, at the WRONG resolution.**
All three are fixed the same way — state the mechanism, then test the guard with the real condition
true and false.

---

## 2026-08-16 — ⚠️ C69 AMENDED: the val40 cache is NOT single-copy-on-Thor

C69 (above) withdrew a cost verdict after `find -maxdepth 4` missed files at depth 6. Its **residual
claim** — that reaching the val40 poses required either a Thor-side read or a multi-GB pull — is now
**also corrected, and by measurement**:

**A `torch.save` file is an uncompressed zip.** Range-reading only the `poses` member of the 256 px
val40 cache mirrored on HuggingFace moved **18.4 MB instead of 4.70 GB — 255× less — in 81 s**, with
**40/40 sha256 bit-identical** to the committed `manifest_EVALPOD_val40.json`. **Thor was never
contacted.**

⇒ **RULE: before pricing a remote artifact by its total size, ask what the container format lets you
read.** Our banked `.pt` files are zips; a member-level range read is often three orders of magnitude
cheaper than the pull, and it leaves a training machine untouched. This is the *cheap-input* half of
C69's own lesson — I re-priced the expensive input twice and never asked how the cheap one was
stored.

---

## 2026-08-16 — G1/G4 were decided by a statistic that NEVER LOOKED AT 7 OF THE 40 VAL EPISODES

*(Next free number re-grepped immediately before appending — C70/C71 collided earlier today.)*

`taniteval/taniteval/planner_p2.py` computed `g1_delta` with `_jack_paired` — the banned
`overlapping_holdout_se` family — **and that decided `G1_pass`.** CLAUDE.md's own measurement says
that estimator distorts paired deltas by up to **×−4.15 including a SIGN FLIP**, so the gate could
have carried the wrong sign for 21 days.

✅ **IT DID NOT. NEITHER VERDICT FLIPS**, and that is MEASURED on banked per-window data, CPU-only:

| gate | banned estimator | decision-grade bootstrap | flips? |
|---|---|---|---|
| **G1** head − planner ADE@2 s | +2.2572 ± 0.3292 → PASS | PASS (3/4 arms exact) | **no** |
| **G4** planner closed-loop | 1.0375 < 1.6852 → PASS | **0.9799 [0.7456, 1.2312] < 1.7318** → PASS | **no** |
| **G4 paired** *(new — first ever computed)* | −0.6873 ± 0.2191 | **−0.7375 [−0.9362, −0.5295]**, p(δ>0)=0.0000 | **no** |

**The comparison is against what actually ran:** recomputing the *banned* estimator from the banked
windows reproduces the published numbers **bit-exactly at 4 dp** (3.1501±0.3472 / 0.4522±0.0312 /
0.8248±0.1035), so this is not a re-derivation that happens to agree.

⇒ **NEW CLASS C73: A WEIGHTED MEAN THAT SILENTLY DROPS PART OF THE POPULATION.**
The banned estimator is not merely a bad *variance* estimator — it is a weighted mean that assigns
**7 of the 40 val episodes weight EXACTLY 0** (17.5 %; ids 1, 9, 22, 23, 27, 28, 34) while
over-weighting its most-drawn episode **2.60×**. **G1 was decided by a statistic that never looked at
seven episodes.**

| # | class | recognition signal |
|---|---|---|
| **C73** | **Weighted mean silently drops part of the population** | an estimator built from overlapping splits/folds/bootstrap draws where **no one has printed the per-unit weight vector**. Signal: `n` is quoted but the *effective* n is not; some units appear in no split |

⚠️ **This is a WRONG-POPULATION problem, not a precision problem**, which is why "the interval is a
bit narrow" was always the wrong summary of it. ⇒ **RULE: for any weighted or resampled estimator,
print the per-unit weight vector once and check for zeros before it decides anything.**

**What the correction DOES move, even though no verdict does:** point estimates **−6.9 % to +6.8 %,
bidirectional within a single artifact**; intervals **1.17×–2.17× too narrow**; the divergence rate —
the *safety-shaped* number — by **+20.3 %** (8.7 % → 7.2 %); and ⭐ **the G4 THRESHOLD ITSELF was a
legacy heldout mean, 2.69 % low (1.6852 vs 1.7318) — the old gate was HARDER than the honest one.**

⚠️ **G1's re-decision is PARTIAL and is stated as such:** `plan_wp` (the open-loop CEM arm) was never
dumped per-window (three probes). A flip would require a **−73.6 %** error on that one arm against a
measured envelope of **−6.9 % to +5.9 %** on identical windows and split structure. ~400 s of GPU
closes it; the script needs no changes.

⭐ **AND THE GUARD HAD TO BE AN AST WALK, QUANTIFIED:** of 228 Python files whose *text* mentions the
banned family, **176 name it only in prose declaring they do NOT use it** (e.g. `h2c_stats.py:1`
*"never overlapping_holdout_se"*). A regex guard would have fired **≥176 false positives on its own
documentation** — the `pgrep -f` self-match trap in a third costume, at a ratio of **176 : 0**. The
shipped guard is an AST taint-propagation walk with six negative controls (including an inlined
`bool(_jack_paired(...)["mean"] >= 0.2)`, a 3-variable laundering chain, an import alias, and a
not-yet-invented `_jack_*` sibling) and three false-positive controls.

**Also found, unquantified and owed:** `planner_p2.py`'s CEM is **unseeded**, so every P2 number
carries a sampling component nobody has bounded (measured drift 0.019 % — small, but *measured* is
not the same as *bounded*).

---

## 2026-08-16 — ⛔ NEW CLASS C74: PRESENCE FOUND AT ONE LOCATION, READ AS THE WHOLE SET

**C2 is *"absence at one location is not absence"*. It has a MIRROR, and until today the log had no
rule for it.** Two instances, hours apart, both in operator-facing material:

1. ⛔ **The launch-fatal one.** `V6_GO_PACKAGE.md` §2.0(a) lists the md5s an operator must verify
   before shipping files to a pod. It names **three** `scripts/` files. The trainer's **import-time
   closure is four** — `train_stage_a` and `stage_a_probes` are module-level imports at
   `train_v6_staged.py:114` / `:117`. A file-ship following that list dies with
   `ModuleNotFound: train_stage_a`. ⭐ MEASURED two independent ways (a clean-child `sys.modules`
   probe and a module-level AST walk) — **and `git show 2b8d09e:…` proves the list was WRONG THE DAY
   IT WAS WRITTEN.** It is not staleness. **One dependency was measured; the CLOSURE was assumed.**
2. **The inherited one.** The chain stream reported §2 as *"stale in five ways"*. A direct
   re-derivation against the code found **eleven**. The five were all real — that is exactly what
   made the count persuasive. The author had been reading the *chain*, not the *runbook*, so the six
   they never looked at were invisible.

| # | class | recognition signal |
|---|---|---|
| **C74** | **Presence at one location read as the whole set** | a LIST asserted complete — dependencies, call sites, consumers, defects, affected docs — where each entry was verified but **the enumeration itself never was**. Signal: nobody can say what procedure would have found a *missing* entry |

⇒ **The check is the CLOSURE, not the members.** For a dependency list, walk the import graph. For a
consumer list, sweep by every exported symbol *and* by artifact, not just by module name — the
`bev_raster` audit found **9** consumers where the brief said 2, and its third sweep caught the one
that **imports nothing and restates the geometry inline**, which no import search could ever reach.

⚠️ **And a verified list is more dangerous than an unverified one**, because each entry survives
spot-checking. C2's fix is a second probe *at another location*; **C74's fix is a probe of a
different KIND** — the second location shares the first one's blind spot (see **C69**: two probes,
one bounded by depth and one on a machine with no checkout, agreeing and both wrong).

⭐ **The sweep's own reframing, which generalises past lists:** across 11 runbooks, *"how stale is
it"* turned out to be the wrong question — **a runbook's danger is whether its host is still alive.**
Nine pointed at released pods and therefore fail safe. `THOR_DEPLOYMENT_RUNBOOK.md` was **the least
stale by flag count and the most dangerous**: its §4 TensorRT build targets the box that is training
right now, with no precondition. ⇒ **Rank operational docs by BLAST RADIUS ON A LIVE HOST, not by
edit age.** Ten of §2's defects make a command *fail*; the eleventh — *"S-W is the only stage that
can start tomorrow"*, written before S-W was live — makes it **SUCCEED**, starting a second trainer
on the fleet's only GPU. **A stale instruction that fails is an inconvenience; one that succeeds is
an incident.**

---

## 2026-08-16 — ⛔ NEW CLASS C75: A GUARD WHOSE REFERENCE IS A MOVING POINTER BECOMES A SELF-COMPARISON

*(Next free number re-grepped immediately before appending.)*

The v6 factored-goal work needed the strongest guard in the programme: **the default build's
`state_dict` must stay bit-identical to the architecture the LIVE S-W checkpoint was built from**,
because a broken strict resume kills a 7-day run. The test compared against **`git show HEAD:…v6.py`**.

**Then I committed.** `a558b79` — a whole-index commit for an unrelated runbook fix — **swept the
agent's in-progress `v6.py` into HEAD**. From that moment `git show HEAD:…v6.py` returned **the
agent's own file**, the test found them identical, and it **skipped itself** with *"matches HEAD
byte-for-byte."*

⇒ **IT WOULD HAVE PASSED FOREVER, AND IT WOULD HAVE PASSED LOUDEST EXACTLY WHEN THE ARCHITECTURE HAD
DRIFTED MOST** — because the more of the change that reached HEAD, the more perfectly the comparison
matched.

| # | class | recognition signal |
|---|---|---|
| **C75** | **Guard referenced to a moving pointer** | a check whose baseline is `HEAD`, `latest`, `main`, "the current file", `newest mtime`, or any symbol that the act of working advances. Signal: **the guard gets quieter as the change gets bigger** |

⇒ **THE FIX IS TO RESOLVE THE REFERENCE BY CONTENT, NOT BY POSITION.** The guard now walks
`v6.py`'s own history for the newest revision **lacking the change marker** — which resolves to
`b12c190`, and that is *also* the architecture the live S-W checkpoint was built from. A content
anchor cannot be advanced by committing.

⭐ **AND THE NEGATIVE CONTROL EARNED ITS KEEP AGAIN, in a way worth copying.** Two controls were
run: a *registered* stray `Linear` fires every check — unsurprising. But a **DISCARDED** `Linear`
(one RNG draw, no `state_dict` key, no parameter) **passes the param-count check AND the flag-flip
check**, and is caught **only** by the per-tensor `torch.equal` comparison. ⇒ **The cheap checks are
not substitutes for the expensive one**, and that is now demonstrated rather than asserted — the
same reason a file-level md5 of `torch.save` output is inadmissible (**C72**).

⚠️ **MY PROCESS ERROR, STATED PLAINLY, BECAUSE IT FIRED FOUR TIMES TONIGHT.** `git commit` takes the
whole index and `git commit -- <pathspec>` segfaults on this repo, so I have been running
`git add -A <dir>` and committing. That sweeps **live agents' in-progress files**, and tonight it:
swept a writeup mid-edit, swept an unverified test file, unstaged-then-re-swept another agent's work,
and here **silently disarmed the strongest guard in the programme**. ⇒ **RULE: `git add -A` is
banned while any agent is live. Stage only paths a completed agent has named in its manifest**, and
if a live agent's file is already in the index, **wait** — the index is not a queue and a commit is
not a checkpoint.

---

## 2026-08-16 — ⛔ NEW CLASS C76: A GATE THRESHOLD NOBODY CHOSE, ON AN ESTIMATOR NOBODY SIZED

*(Next free number re-grepped immediately before appending: C75 was the highest.)*

The S-W gate's O6 row read `"O6_rank_retention": ">= 0.8x effective rank across phases"`. The
number it compares is `spectrum_report(...)["effective_rank"]` from ONE training batch — on the
live v6F S-W run **48 rows of d=2048**, because `--batch 8 --window 6` and `--eps-per-batch 4`.

⛔ **A centred covariance from 48 rows has rank ≤ 47.** The banked mean of **15.13 is 15 of 47**,
not 15 of 2048 — and that misreading was available because **the emitted record did not carry
its own ceiling**.

MEASURED (`…/incoming/2026-08-16-sigreg-gate-power/`, CPU, seeded):

- at n=48 an **isotropic** d=2048 population (true effective rank 2048) reads **46.86**; one
  collapsed **7.3× to true rank 281** still reads **22.76**;
- pooling the **same healthy population** to 1536 rows moves the reading **14.02 → 121.57** —
  ⇒ **~15 was a CEILING ARTIFACT, not collapse**;
- the `>= 0.8x` criterion fires when NOTHING changed between **11 %** (model null) and **38 %**
  (the run's own banked spread), with **power 0.145** against a 1.43× true collapse. Likelihood
  ratio **1.32** — very nearly no test at all.

| # | class | recognition signal |
|---|---|---|
| **C76** | **Threshold nobody chose, on an estimator nobody sized** | a gate constant with no stated false-positive rate on the estimator that will actually be used, and no stated `n`. Signal: **the number is inherited, so it cannot be wrong — and therefore never gets checked** |

⇒ **RULE: any gate threshold ships with the false-positive rate it achieves on the estimator that
will actually be used, and the estimator ships with its own ceiling stamped in the record.**

⭐ **THIS IS C13's FAMILY INVERTED.** C13/C14 are instruments structurally unable to report the
answer they are cited for — a guard that **cannot fail**. This is a guard that **fires when
nothing happened**. Both are unfalsifiable in practice; only the direction differs.

⚠️ **AND THE MECHANISM WAS AN INHERITED TRADE, NOT A COINCIDENCE.** Both samplers group few
episodes × many windows on purpose — `train_v58f_unicycle_head.py:353` says verbatim *"The mild
within-batch correlation is an accepted, stated trade"*, taken to cut MooseFS cold loads ~8×.
That is fine for a **loss** (it only needs an unbiased gradient) and fatal for a **rank
estimator** (it needs independent directions). ⇒ **When an I/O or batching trade is accepted for
one consumer, list the OTHER consumers of the same tensor before inheriting it.**

⭐ **A ZERO NEEDS A COMPANION ROW OR IT SHOULD NOT BE BELIEVED.** A sibling stream hypothesised
that SigReg's slice resampling caused part of the spread. Refuted: varying only the SigReg
generator moves `effective_rank` by **sd 0.000000** — but that alone is indistinguishable from
*"the knob was never connected"*. The **same contrast moves the `o6` loss** (sd 0.019896, range
0.083679), which converts a suspicious zero into a demonstrated one. ⇒ **Every null result
carries the positive control that proves the manipulation was live.**

⚠️ **A CI NOBODY CHECKED IS A DECORATION — and the house estimator was the wrong one here.**
Coverage of the finite-n estimand, 60 datasets, nominal 0.95: percentile cluster bootstrap
**0.250 / 0.000**, pivotal cluster bootstrap **0.300 / 0.000**, leave-one-cluster-out jackknife
**0.850 / 0.867**. Bootstrap-with-replacement duplicates blocks and duplicated rows are exactly
rank-deficient, so it is biased **down** for a RANK functional. ⇒ **Narrow, evidence-backed
carve-out from the paired episode-cluster-bootstrap rule FOR RANK ESTIMANDS ONLY — it does not
generalise, and any other estimand needs its own coverage check first.**

---

## 2026-08-16 — "the 115-clip SAM3 gap is CLOSED" — RETRACTED. 115 records, ZERO detections

*(Next free number re-grepped immediately before appending.)*

I verified the first headless Colab production run far-side and reported the gap closed. **It is not
closed.** MEASURED over 25 randomly sampled clips (seed 0): **total detections 0, clips with zero
25/25**, every concept zero. Each record carries its own cause, per concept, per frame:
`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`.

SAM3 ran, raised on every concept, and the pipeline **faithfully recorded the failure**. Files real,
schema valid, `n_frames_run` 5–7 exactly as claimed, counts exactly matching the fixture — and the
content is 115 clips of nothing.

**What I checked, and why it all passed:** record count vs fixture (115/115) · zero-byte scan (0) ·
3-clip round-trip asserting `clip_id == filename` and that `frames` was populated. **I never asked
whether a single detection existed.** The probe needed was one expression, `n_det_total > 0`, and the
error string was inside the payload I had already downloaded and printed the *keys* of.

| # | class | recognition signal |
|---|---|---|
| **C77** | **A pipeline that records its own failure faithfully passes every structural check** | the artifact is well-formed, complete, correctly named, correctly counted — and its payload is an error census. Signal: verification enumerated *containers and identity* but never evaluated the *quantity the artifact exists to produce* |

⇒ **This is C18 in the reviewer's seat.** C18 is "a defect found by a probe is bounded by that
probe's granularity". Here the probe was at the right scope (far side, records not files — I even
said so) and the wrong RESOLUTION (existence, not content).

⚠️ **And it is the aug120 gap itself, repeated one layer up, by me, on the run that was fixing it.**
That gap was *"25 sam3 files were all PRESENT but each held exactly 4 records"*. I then verified its
repair by counting files and identities. **The failure mode I was correcting is the one I used to
declare it corrected.**

⇒ **RULE: for any artifact that exists to PRODUCE something, the completion criterion is a
non-trivial count of the produced thing — never the presence, naming, or size of the container.**
For perception: `n_det_total > 0`, per-concept totals, and an **error-string census** (an `error` key
inside a payload is a first-class signal, not decoration). For labels: non-default token counts. For
evals: n per family. Concretely, the perception criterion is now *"detections exist AND are visible
on video"* — a renderer that draws nothing is itself the failing test.

⚠️ **Second, smaller retraction inside the same report:** I read `n_frames_run` 5–7 as evidence the
re-run "filled the hole with the coverage the stage should have produced" — a satisfying inversion of
the original `--n`-defaults-to-4 defect. It shows only how many frames were **ATTEMPTED**.
**Attempts are not coverage**, and a number that flatters the story deserves more scrutiny, not less.

---

## 2026-08-16 — the SAM3 backfill's DELIVERABLE MANIFEST — 4 of 6 rows point at nothing

*(Next free number re-grepped immediately before appending: C77 was the highest.)*

Landing the C77 fix, I went to read the retracted run's own evidence and **four of its six
manifest rows name files that do not exist** —
`…/2026-08-16-sam3-backfill-run/raw/{exec_run1.log, SAM3_BACKFILL_115_output.ipynb, hf_count.py,
bootstrap.py}`. MEASURED: `raw/` is an EMPTY directory (`ls -la` → `total 0`), and
`git ls-files <pkg>` returns **exactly one path**, the report itself. The commit that added the
package (`ee7795c`) staged the `.md` and nothing else.

⇒ **The exec log that would have shown the dtype traceback in real time, the executed notebook,
and the far-side verifier are all gone** — the C77 root cause had to be re-derived from scratch on a
fresh T4 rather than read off the banked log.

⚠️ **The manifest is the artifact whose whole job is to make stranding visible**
(`AGENT_OPERATING_STANDARD.md` rule 2, *"every artifact and where it lives"*). Here it did the
opposite: it **certified** four locations nobody looked at. A row asserting a path is a CLAIM, and
`git ls-files <path>` is a one-command falsifier that was never run.

| # | class | recognition signal |
|---|---|---|
| **C78** | **A manifest row asserted, never resolved** | any "artifact → where it lives" table published without `git ls-files` (or an equivalent existence probe) run against **every** row. Signal: the row's path was *typed from intent* — the file the agent meant to save — rather than *read back from the index* |

⇒ **RULE: a deliverable manifest is verified by resolving every row before the report is filed, and
the verification command is stated in the report.** `git add` exit codes are already inadmissible
here (CLAUDE.md's silent-no-op trap); an unresolved manifest row is the same failure one level up.

⭐ **THIS IS C77's OWN SHAPE, ONE LAYER OUT.** C77 = an artifact that passes every *structural*
check while its payload is empty. C78 = a *manifest* that passes every *editorial* check while its
referents are empty. Both are documents that describe containers and are believed about contents.
The same sentence fixes both: **resolve the thing the row is about, not the row.**


---

## 2026-08-16 — "encode-once changed the detections", then "SAM3 is not reproducible across VMs" — BOTH RETRACTED. I enumerated two of three variables

*(Next free number re-grepped immediately before appending: C78 was the highest.)*

Fixing C77 I found the SAM3 engine encoding each frame **once per concept** — 44 ViT-trunk passes
per clip where 7 were needed (banked `wall_s` 97–98 s for a 6-frame clip). I made it one encode per
frame, then checked the re-run against the record the previous VM had banked: **60 vs 64**
detections, `pedestrian` 4→7, `traffic light` **0→2**.

**Wrong reading #1 — "the optimisation changed the science."** I was one edit from reverting a
correct fix. Refuted by a same-session control (`…/2026-08-16-sam3-dtype-fix/raw/eq3_whole_clip.json`,
clip `0089a096`, all six run frames):

```
A   per-concept encode   89.3 s   tot=64   per-frame {0:10, 8:11, 16:6, 19:5, 24:13, 32:19}
B   encode-once          21.2 s   tot=64   per-frame {0:10, 8:11, 16:6, 19:5, 24:13, 32:19}
A2  per-concept again    89.8 s   tot=64   identical to A
```

**A == B on every per-concept AND per-frame count**, A == A2 ⇒ the optimisation is a pure refactor
worth **4.21×**, and the 60↔64 gap lives somewhere else.

**Wrong reading #2 — "SAM3's detection count is not reproducible across Colab T4 VMs."** I wrote
that entry, with a rule attached ("bank a corpus in ONE session"), and it was refuted **twenty
minutes later by my own re-run**: the fixed pipeline on VM #2 reproduced VM #1's record for that
clip **exactly** — `det=60`, `car:38 bus:7 pedestrian:4 traffic sign:11`, byte length 64 081 both
times. SAM3 *is* reproducible across VMs.

⇒ **THE ACTUAL THIRD VARIABLE: THE VIDEO FILE.** The pipeline re-bridges each clip from its w120
shard (`bridge_batch` → `v2_to_pilot`) into a scratch mp4 it then deletes; my experiment pulled the
**pre-bridged** `bridged_w120train_2400/videos/<cid>.mp4` from HF instead. Same clip, same frame
indices, different bytes — and ~7 % of detections sit close enough to
`Sam3Processor(confidence_threshold=0.5)` to flip on re-encode noise.

| # | class | recognition signal |
|---|---|---|
| **C79** | **The control arm differed in MORE ways than were enumerated** | an A/B whose baseline is an artifact produced *earlier or elsewhere*. Signal: the diff list was written from the change the author was thinking about (here: the code path), and the arms were never run back-to-back with **every** input pinned — data included |

⇒ **RULE: an optimisation is validated against the SAME arm re-run in the SAME session on the SAME
INPUT BYTES, and the baseline re-run against itself (A vs A2) is part of the evidence.** Where the
baseline can only be a banked artifact, the difference bounds *code + machine + data* together and
may not be attributed to any one of them.

⚠️ **AND NOTE HOW THE FIRST RETRACTION HID THE SECOND.** Retracting "the code changed it" felt like
the end of the investigation — I had a same-session control, a clean 4.21×, and a satisfying
"nondeterminism" story that explained the leftovers. **A residual that a plausible story absorbs is
still a residual.** What settled it was a measurement I was already running for another reason.

⭐ **CONSEQUENCE THAT STANDS, AND IT IS THE USEFUL ONE: A PERCEPTION OVERLAY MUST BE DRAWN ON THE
BYTES THE MODEL SAW.** ~7 % of detections flip on re-encode alone, so a video rendered from a
different copy of "the same" clip is showing boxes computed on frames that are not the ones on
screen. Any renderer that pulls its own copy of the video states that, or re-derives the frames the
way the pipeline did.

⚠️ **The liveness control is unaffected and that is not luck** — `road`/`sky` score far from 0.5, so
they do not flip on re-encode noise while the agent concepts near the threshold do. **A control is
only a control if it sits far from the decision boundary.**

---

## 2026-08-16 — "the suite is green at 3532/0/17/2" — the 17 SKIPS were 10 tests that could not fail

*(Next free number re-grepped immediately before appending: C79 was the highest.)*

The dev box lacked `torchvision`, `av` and `imageio`, so several test modules were
`pytest.importorskip`-ed at import. Installing those three (needed for this package's video work,
`--no-deps`, torch verified untouched) moved the suite from **17 skipped to 7** — **10 tests started
executing for the first time in months** — and exactly one of them failed:

```
tests/test_v2_dataset.py::test_manifest_is_cached_and_reused
    assert m2["version"] == 2      # v2_dataset.MANIFEST_VERSION has been 3 since v3
E   assert (3 == 2)                # added per-clip image_h/image_w
```

The manifest format was bumped 2 → 3 and its own regression test **could not object**, because the
whole module was skipped on the only box that runs the suite.

| # | class | recognition signal |
|---|---|---|
| **C80** | **A green suite whose skips were never counted** | a pass/skip line quoted as health where nobody has asked WHICH tests skipped, on WHICH box, and for how long. Signal: the skip count is stable release after release — i.e. the same guards have been inert the whole time |

⇒ **RULE: a suite headline carries its skip list, not just its skip count**, and any test skipped
for a MISSING OPTIONAL DEPENDENCY on the box that runs CI is treated as *not covered* rather than
*passing*. Where the dependency is installable, install it.

⇒ **Fixed at the root, not at the symptom:** the assertion now reads
`m2["version"] == MANIFEST_VERSION` — **a version literal in a test is a stale claim waiting to
happen**, and the constant cannot drift from itself. *(Suite after the fix: recorded in
`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-sam3-dtype-fix/SAM3_DTYPE_FIX.md`
§7 — measured, not projected.)*

⭐ **THIS IS C13/C14's FAMILY — an instrument structurally unable to report the answer it is cited
for.** A skipped test is a guard that cannot fail, and a skip count is exactly the kind of number
that looks like accounting while hiding the thing being accounted for. Same shape as C77 one level
down: the container (a green run) was checked; the content (which assertions actually executed) was
not.

---

## 2026-08-16 — a banked record whose `live: False` contradicted its own counts

*(Next free number re-grepped immediately before appending: C80 was the highest.)*

The C77 liveness control stored a derived verdict, `liveness.live`, beside the counts it was derived
from. Its rule changed **mid-corpus** — `all(control > 0)` → `any(control > 0)`, after clip
`24b6948f` returned `road 2 · sky 0` under an underpass and was flagged dead while carrying **22
real detections**. Every reader was updated to recompute, so no reported number was ever wrong.

**The artifact was not updated, and that is the defect.** MEASURED over all 115 far-side records
(`…/2026-08-16-sam3-dtype-fix/raw/strip_stale_live_flag.json`):

```
carried the derived field   81
DISAGREED with own counts    1   24b6948f  stored live=False · recomputed True · {road: 2, sky: 0}
```

One record sat on disk asserting the opposite of the data next to it. Any consumer reading the flag
rather than the counts — the `aug120_pipeline` batch gate, the overlay's liveness row, a future
re-fuse, a human six months out — would have scored a healthy clip as **the one dead-engine failure
that blocks a PASS**.

| # | class | recognition signal |
|---|---|---|
| **C81** | **A derived field written down beside its own inputs** | a stored verdict (`is_valid`, `passed`, `live`, `status`) whose inputs are banked in the same record. Signal: **the rule that produces it has changed at least once**, and nothing rewrites the artifacts when it does — so the field and the data can disagree, and the field is what gets read |

⇒ **RULE: where the inputs are banked, do not store the verdict.** Compute it at read time, in ONE
named function, and let every consumer call that. **A field that cannot be stale beats a field that
must be kept in sync.**

⇒ **FIXED BY DELETION, NOT CORRECTION** — correcting the one wrong flag would have left the same
trap armed for the next rule change. `live`/`all_fired` are gone from the schema
(`ph0_sam3.liveness_probe`), `ph0_sam3.is_live()` is the only derivation, and every banked record was
rewritten to drop the field — **no GPU, no re-detection**, per-file far-side verified.
`is_live()` also IGNORES the field on any older record it meets.

⚠️ **AND NOTE WHERE IT WAS CAUGHT.** Not by the code — my readers already recomputed and were right.
It surfaced only because an independent census **compared the stored flag against a fresh
recomputation** instead of reading one or the other. ⇒ **When a record carries both a derivation and
its inputs, audit them AGAINST EACH OTHER at least once; agreement is the only evidence that the
cache is not rotten.**

⭐ **SAME FAMILY AS C77, INVERTED ONE MORE TIME.** C77: the artifact was well-formed and its payload
empty. C78: the manifest was well-formed and its referents absent. **C81: the record is well-formed
and internally CONTRADICTORY.** Each time, the thing that looked authoritative was a summary of
something nobody re-derived.


---

## C82 — I RETRACTED A **CORRECT** CLAIM, ON ONE AGENT'S UNCORROBORATED REPORT, INSIDE A RETRACTION ABOUT DOING EXACTLY THAT (2026-08-16, orchestrator)

**RETRACTED: the first draft of C82 itself.** It declared *"Suite 3572 passed / 0 failed"* to be a
false, INHERITED claim, credited a subagent with catching it, and told two live agents their baseline
was wrong. **All of that was mistaken. The suite was green.**

**WHAT SETTLED IT — three independent BANKED full-suite artifacts, none of which I had read when I
wrote the retraction:**

| artifact | result |
|---|---|
| `…/2026-08-16-seam-instrument/raw/pytest_baseline_pre_edit.txt` | `3574 passed, 7 skipped, 2 xfailed` — **exit 0** |
| `…/2026-08-16-sam3-dtype-fix/raw/pytest_full_suite.txt` (different stream, minutes later) | `3574 passed, 7 skipped, 2 xfailed` — **exit 0** |
| `…/2026-08-16-seam-instrument/raw/pytest_after_final.txt` | `3658 passed, 7 skipped, 2 xfailed` — **exit 0** |

The contradicting figure — *"9 failed / 3563 passed / 7 skipped / 2 xfailed, 3572 collected"* — is
corroborated by **no artifact**, and **does not close arithmetically**: 3563 + 9 + 7 + 2 = **3581**,
not the 3572 it called the collected total. The inconsistency was visible in the report itself and I
did not check it.

⇒ **ROOT-CAUSE CLASS: INHERITED-QUOTED-AS-MEASURED — the SAME class the first draft was written to
log, applied to the CORRECTION rather than to the original.** A retraction is a claim. It carries the
identical evidence bar as the thing it retracts, and I gave it a *lower* one: I demanded artifacts of
myself, then overturned my own verified position on a single unverified sentence because it was
phrased as a correction. **A report that says "you were wrong" is not self-authenticating.**

⚠️ **AND IT WAS ASYMMETRIC IN THE DANGEROUS DIRECTION.** I applied more scepticism to the claim that
things were FINE than to the claim that things were BROKEN. Alarm feels like diligence, so it gets
waved through; that is how a healthy system gets "fixed" into a broken one. **Bad news requires
exactly as much evidence as good news** — the pod-monitor trap in this same file is the same shape:
a false failure invented by a filter, acted on as real.

⇒ **RULE: before retracting, READ THE PRIMARY ARTIFACT.** Banked suite output, raw JSON, the log
itself. Three such artifacts existed in the repo the whole time and answered it in one command. And
⇒ **when two agents disagree, do not adjudicate between their SUMMARIES — go to what each measured**,
then say which is corroborated.

⭐ **WHAT SURVIVES, and it is worth keeping.** The observation that provoked all this is still true:
**a suite count is only admissible against a QUIESCED tree or a NAMED COMMIT.** With ≥4 agents
editing, counts genuinely drift (3574 → 3658 here, purely from tests being ADDED), an 8-minute run
samples a moving tree, and my own mid-run "1 failed" was a file edited at 20:26:26 *during* the run —
that test passes in isolation. ⇒ Quote it as **"N passed at `<sha>`"**, and gate per agent on
**"my change adds N passing tests and introduces zero NEW failures"**, never on a global total.

⇒ **THE CORRECTED FACTS:** the suite is green and has been throughout. `fa5c73b`'s commit message
(*"3572 passed / 0 failed"*) is accurate to within the ±2-test drift of concurrent test additions and
**needs no correction**. The two agents I told otherwise have been told again.

### ⭐ THE MECHANISM, MEASURED — the phantom failures were the APPARATUS, not the code

Run on the same tree, minutes apart, differing in **one environment variable**:

```
PYTHONUTF8=1  python -m pytest -q   ->  3658 passed, 0 failed, 7 skipped, 2 xfailed
              python -m pytest -q   ->  3656 passed, 4 FAILED, 7 skipped, 2 xfailed
```

The four are encoding-sensitive and nothing else:
`test_bev_consumer_fov.py::test_figure_main_never_writes_outside_its_output_dir` ·
`test_bev_consumer_fov.py::test_figure_caveats_the_missing_frame_when_none_is_recorded` ·
`test_e_wc2_sigma_star.py::test_cli_print_contract` (TypeError) ·
`test_ff_v58f.py::test_tool_REFUSES_the_biased_estimator_by_name`.

⇒ **On Windows this suite requires `PYTHONUTF8=1`, and without it reports failures that are
properties of the SHELL, not of the code.** An agent reporting "N failed" without stating its
invocation is reporting its own environment. *(Independently discovered the same day by the
lane-change stream, logged as **C84**: it filed four failures as "pre-existing, not mine", then found
they were its own apparatus — its words, and they are the right words: **"a probe reported my
apparatus, and I read it as the subject"**.)*

⚠️ **This is what makes the asymmetry above concrete.** The uncorroborated "9 failed" was almost
certainly this — a run without the variable. I did not ask *"what command produced that number?"*,
which is one question and would have ended it. ⇒ **A test result is not a number; it is a number PLUS
its invocation.** Quote the command or the count means nothing — the same discipline this file already
demands for an exponent (fit window, R², n) and for an interval (its estimator).

⭐ Note which test is among the four: **`test_tool_REFUSES_the_biased_estimator_by_name`** — the guard
against `overlapping_holdout_se`. A false red on the very test that enforces our estimator rule is the
strongest possible argument for stating invocations: an apparatus artefact landing on a load-bearing
guard is exactly how a real guard gets waved away as "known flaky".

---

## C84 — I REPORTED FOUR TEST FAILURES THAT WERE MY SHELL'S ENCODING, AND BUILT A THEORY ON TOP OF THEM (2026-08-16, arch-inf agent)

**What I claimed.** *"The suite is NOT green. I measured the tree myself before my first edit:
**4 failed · 3570 passed**. The four failures are pre-existing, in files I did not touch, and are another
agent's blast radius."* Named: `test_bev_consumer_fov.py` (×2), `test_e_wc2_sigma_star.py::test_cli_print_contract`,
`test_ff_v58f.py::test_tool_REFUSES_the_biased_estimator_by_name`.

**What is true.** ⛔ **ALL FOUR WERE MINE — an artifact of MY SHELL.** MEASURED:

| invocation | result |
|---|---|
| `pytest -q <the 4 tests>` | **4 failed** — `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f` |
| `PYTHONUTF8=1 pytest -q <the same 4>` | **4 passed** |

They read UTF-8 fixtures; without `PYTHONUTF8=1` this Windows shell decodes cp1252 and dies on the first
non-ASCII byte. The suite IS green — matching the banked artifacts (`…/2026-08-16-seam-instrument/raw/
pytest_baseline_pre_edit.txt`, `…/2026-08-16-sam3-dtype-fix/raw/pytest_full_suite.txt`, both **3574 passed,
0 failed**) and the orchestrator's original figure.

⇒ **ROOT-CAUSE CLASS: A PROBE REPORTED A PROPERTY OF THE MEASURING APPARATUS AND I READ IT AS A PROPERTY OF
THE SUBJECT.** This is the `df`-on-a-pod family that opens CLAUDE.md — `df` reporting the cluster instead of
the pod quota, `free`/`tegrastats` on Thor, `memory.usage_in_bytes` counting page cache: *"a probe that
reports the wrong scope is worse than no probe, because it looks like an answer."* Test **runs** join test
**counters** in that family. I verified I had run the command; I never verified the command measured the tree.

⇒ ⛔ **THE AGGRAVATING HALF, and the part worth keeping: I EXPLAINED THE DISCREPANCY AWAY INSTEAD OF
INVESTIGATING IT.** My number disagreed with the brief's, so I wrote a confident narrative — *"three
incompatible baselines in one session; four agents are live; the tree moves under every measurement"* — which
was plausible, partly true in general, and **false here**. It converted a signal into a story and closed the
question. **A disagreement with a banked artifact is EVIDENCE, not noise to be rationalised.**

⇒ **TWO TELLS I IGNORED**, both cheap:
1. the four failures sat in files with **no plausible relationship to each other** — a real regression has a
   mechanism, and "four unrelated files" is a signature of an *environmental* cause;
2. `UnicodeDecodeError: 'charmap'` is an **encoding** symptom, not a logic one. The traceback named the class.

⇒ **THE RULE.**
1. **When your measurement disagrees with a banked artifact, suspect YOUR APPARATUS FIRST.** Re-run one
   failing case under a different environment before concluding anything about the subject. Cost: ~10 s.
2. **A failure set with no common mechanism is an environment hypothesis until disproved.**
3. **On Windows, `pytest` in `stack/` REQUIRES `PYTHONUTF8=1`** or it reports four failures that do not exist.
   Quote a suite result with the invocation that produced it, not just the number.
4. ⚠️ **This was load-bearing, not cosmetic.** Under my wrong baseline I would have shipped with *"4 failures,
   not mine"* as an accepted state — i.e. I had **disabled the green-suite gate for myself** and would have
   waved through any genuine regression that landed in those files.

*(Companion to C82, which is the same disease in the orchestrator's direction: there, an inherited number was
trusted over artifacts; here, a self-measured number was trusted over artifacts. **Measuring it yourself is
necessary and NOT sufficient — the apparatus is part of the claim.**)*

---

## C85 — THE WHOLE SAM3 CORPUS'S `rle_rows` IS FLATTENED AND CANNOT REDRAW ITS OWN MASK — AND EVERY CHECK WE RAN AGREED IT WAS FINE (2026-08-16, arch-inf agent)

**What was banked.** 2 496 detections across 115 clips, each carrying `rle_rows` documented as
per-row `[row, start, end)` runs — *"enough to redraw the mask exactly"*. It is not. Every run in the
v1 corpus reads `[0, flat_start, flat_end)`.

**MEASURED**, clip `0089a096`, a `car` at `box_xyxy [54.4, 82.1, 66.0, 94.1]` on a **448**-wide frame:

```
banked : [[0, 36794, 36800], [0, 37240, 37250], [0, 37688, 37698], …]   12 runs, ALL on row 0
correct: [[82, 58, 64],      [83, 56, 66],      [84, 56, 66],      …]
```

**Root cause.** `Sam3Processor.set_text_prompt` returns `masks` of shape **`[N, 1, H, W]`**, not
`[N, H, W]` — MEASURED on a live T4, never asserted anywhere in our code. `_rows_rle` did
`for r, row in enumerate(np.asarray(mask))`, which on a `(1, H, W)` array yields **one** item whose
`row` is the entire `(H, W)` plane, so `np.flatnonzero` returned **flattened** indices.

⇒ **ROOT-CAUSE CLASS: A SERIALISER THAT ACCEPTED A SHAPE IT NEVER ASSERTED, SITTING NEXT TO A REDUCER
THAT WAS SHAPE-AGNOSTIC AND THEREFORE AGREED WITH IT.** `mask_area_px` is `m.sum()` — correct on any
shape. So the two derivations of the same object *corroborated each other* while one of them was
wrong: **the run lengths summed to exactly `mask_area_px`**, which is the check a careful reviewer
would run, and it passes.

⇒ ⛔ **WHY NOTHING CAUGHT IT, spelled out, because the list is the lesson.** The JSON was well-formed.
The field was present and non-empty. The count was right. The lengths summed to the banked area. The
liveness control was live. The error census was empty. C77's own fix — *"a perception backfill is
complete when detections exist and are visible on video"* — was satisfied on the first half and
**never actually exercised on the second**: the overlays were rendered, banked, and looked plausible
because `draw_masks` does `over[r, a:b]` with `a` ≈ 36 794 on a 448-wide array, and **numpy silently
clips an out-of-range slice to nothing**. ⇒ *"visible on video"* degraded to *"the boxes are visible
on video"*, and nobody noticed the fill was missing. **A renderer that draws nothing looks like a
sparse scene.**

⇒ ⭐ **AND THAT LAST SENTENCE IS MEASURED, NOT REASONED** (`…/2026-08-16-sam3-extraction-v2/`
`raw/c85_overlay_proof.json`, produced by `code/c85_overlay_proof.py`). Banked v1 record
`sam3_backfill/0089a096…json`, frame 448×179, 60 detections, **962 RLE runs — every one on row 0,
maximum column index 63 539, and only 2 of 962 landing inside the frame**. Rendering those exact
detections through the real `ph0_rich_overlay.draw_camera` twice — once as banked, once with
`rle_rows` deleted — the two images differ by **0 pixels**. The mask layer contributed **nothing**.
*(Proving it against the same function with the field removed, rather than against a hand-rolled
expectation, is what makes the 0 mean "the mask layer drew nothing" instead of "my reimplementation
agreed with itself" — boxes, labels and scaling are identical on both sides and cancel.)*

⇒ ⭐ **WHAT DID CATCH IT, and this is the transferable part: THE STRICT READER FOUND THE PERMISSIVE
ONE'S BUG.** Adding contours introduced `crack_loops`, which begins `if m.ndim != 2: return []`. It
produced **no contour at all** on real masks while producing a perfect contour on a synthetic 2-D
test — a loud, immediate, unmissable disagreement. `_rows_rle`, handed the same input, produced
plausible garbage without complaint. ⇒ **When two derivations of the same object differ in
strictness, believe the strict one. Prefer a function that RAISES to one that COPES** — a permissive
reader does not remove the defect, it removes the evidence.

⇒ **THE DATA IS RECOVERABLE; THE SCHEMA WAS THE LIE, NOT THE PIXELS.** A v1 run decodes as
`row = start // W`, `col = start % W` (a run never straddles a row boundary). **No re-detection is
needed** to rescue v1's masks — which also means the cost of this retraction is a decoder, not a
GPU day.

⇒ **BLAST RADIUS, stated per claim rather than per document:** `mask_area_px`, `box_xyxy`, `score`,
`per_concept_hits`, `n_det_total` and the road/sky liveness control are **unaffected** — every number
in `SAM3_CONCEPT_RELIABILITY.md` and `SAM3_DTYPE_FIX.md` stands. What is void is (a) `rle_rows` read
as documented, and (b) **every `*_rich.mp4` overlay rendered from a v1 record, which shows boxes and
NO mask fill**. ⚠️ Anyone comparing a v1 overlay against a v2 one will read that as a regression in
the wrong direction.

⇒ **THE RULE.**
1. **Assert the shape you are about to iterate.** `as_2d_mask()` now squeezes leading singleton axes
   and **raises** on anything it cannot interpret; `_rows_rle` and `read_outputs` go through it.
2. **Bank the frame the encoding refers to.** Every detection now carries `mask_hw`, which makes the
   encoding self-checking — 36 794 is a legal column index in *some* frame, and is obviously not one
   in a 448-wide frame.
3. **A round-trip is the only real test of a serialiser.** *Rebuild the object from the record and
   compare it to the original* — `code/v2_integration_check.py` does exactly that (163/163 exact) and
   would have failed on day one against v1. Summing the lengths is not a round-trip; it is a checksum
   that both the correct and the broken encoding satisfy.
4. ⚠️ **"Visible on video" is only a completion criterion if someone LOOKS at the layer in question.**
   A silent-clip renderer turns a data defect into a plausible picture.

*(Same family as C77 — a well-formed artifact whose content is wrong — and C18 — a check scoped to
the container. The new half is that here the container check was **quantitative and passed**: this
is the first one where the wrong data satisfied a numeric invariant.)*

---

## C86 — I HANDED AN AGENT A MECHANISM THAT WAS ADJACENT TO THE FAILURE, AND IT WAS WRONG (2026-08-16, orchestrator)

> ⚠️ **RENUMBERED C85 → C86 on 2026-08-17: this entry COLLIDED with another C85.** Two agents
> appended to this append-only file within the same hour and both claimed `C85` — mine (this one)
> and the SAM3 `rle_rows` finding above. Neither was wrong to append; **nothing allocated the
> identifier**. ⇒ *An append-only log shared by concurrent writers needs the NEXT NUMBER to be read
> and claimed atomically, or it silently produces two records with one name and every later citation
> becomes ambiguous.* Caught by `grep -n "^## C8[0-9]"` before either was cited anywhere; the
> content of both entries is unchanged.

**RETRACTED:** my briefed diagnosis of `test_v6_ladder_edges.py::…[S-J]` — *"it seeds the model
but draws its batch from global RNG, which is a plausible mechanism"* — and the framing that went
with it, *"intermittent"*.

**WHAT IT ACTUALLY IS — `KeyError: 'interp'`, and nothing to do with randomness of data:**
`_grad_census` built its dict with `setdefault`, so a **zero-parameter group vanished from the
census entirely**; `STAGE_GROUPS["S-J"] is MODULE_GROUPS` includes `interp`, which is empty at the
default build; and the assertion `any(census[g] … for g in want)` iterates a **hash-randomised
`set`** and **short-circuits** — so the missing key is only ever reached when `interp` happens to
sort first.

⇒ **Not flaky. DETERMINISTIC, keyed on `PYTHONHASHSEED`:** seeds 3 and 12 FAIL, seeds 0 and 5 PASS.
Rate **3/25 = 12.0 %** measured over separate processes, against **43/400 = 10.75 %** predicted
from a hash-seed sweep. *(And `mk()` calls `torch.manual_seed` — a `KeyError` on a STRUCTURAL
lookup is unreachable by any data value, so my mechanism could not have produced this failure at
all.)*

⇒ **AND IT DID NOT PRE-DATE THE CHANGE, which I had marked UNVERIFIED and was the useful question.**
The test file is byte-identical at `ee02ff7` and HEAD, and at `ee02ff7` it PASSES at the exact seeds
that deterministically fail at HEAD. `interp` entered `MODULE_GROUPS` in `06b8782` at 20:57 —
**18.5 h after the test was written**. The test was correct when written and was broken by a model
change.

⇒ **ROOT-CAUSE CLASS: A MECHANISM ADJACENT TO THE FAILURE, ADOPTED BEFORE IT WAS TESTED AGAINST THE
FAILURE'S OWN SHAPE.** "Seeded model, unseeded batch" is a real property of that test and a real
cause of real flakes — it simply was not THIS one, and I never checked it against the actual
symptom. A `KeyError` naming a specific group is a **structural** signature; nondeterministic DATA
cannot produce it. **The error message contained the refutation and I did not read it before
theorising.** Sibling of C84 (*a probe reported my apparatus, and I read it as the subject*).

⇒ **THE CHEAP DISCRIMINATOR, and the standing rule (the agent's own words, adopted):
MAKE THE MECHANISM PREDICT A RATE, THEN MEASURE THE RATE.** A hypothesis that cannot name a
number is not yet a hypothesis. Here the real mechanism predicted 10.75 % and delivered 12.0 %;
mine predicted nothing and would have been "confirmed" by any retry that happened to pass.

⚠️ **AND THE BUG WAS HIDING A SECOND, WORSE ONE** — which is why "retry until green" would have
been expensive. With `agent_slots=True`, `interp` owns **62 params**,
`apply_stage_freeze(·, "S-J")` marks **all 62 trainable**, and the S-J loss reaches **0**: verbatim
the failure mode `v6.py`'s own docstring claims to prevent — a module advertised as training while
nothing trains it. Pinned `xfail(strict=True)`; **the `v6.py` S-J group tuple must be fixed before
the first `agent_slots=True` run.**

⚠️ **One more instance of the self-matching-filter trap, same night:** `grep -l failed` over the
post-fix logs matched **35 of 35** runs — because pytest prints `xfailed`. The opaque `ZZ…ZZ`
marker is what kept the 40/40 figure honest. **Never grep an output stream for a word that is a
substring of the stream's own vocabulary.**

---

## C87 — AN INSTRUMENT DEFECT READ AS A PROPERTY OF THE THING MEASURED, TWICE, IN OPPOSITE DIRECTIONS (2026-08-17, arch-inf agent + orchestrator)

**RETRACTED:** the standing framing that SAM3's `traffic sign` class *"is ~⅔ garbage on `w120val`
but ~88 % precise on `aug120`"*, and every version of the sentence *"neither number transfers"*
— including the careful one I wrote into `NEXT_4472_BUILD_INPUTS.md` yesterday, which said
**"not on aug120"** rather than "G1 was wrong". **All of them located the disagreement in the
CORPUS. It was never in the corpus.**

**WHAT SETTLED IT — three arms, blind, on identical detections:**

| arm | what changed | "no sign at all" |
|---|---|---|
| **A** `w120val`, uniform draw, box **outlined** | — | **9.4 %** [3.0, 17.4], precision 0.852 |
| **B** G1's own clips, G1's max-area rule, box **outlined** | selection | **2.7 %** |
| **C** G1's own banked tiles, re-read blind | **nothing — the same JPEG bytes** | **88.9 %** [77.6, 98.2] |

⇒ **2.7 % vs 88.9 % ON THE SAME DETECTIONS. The only difference is whether the box is drawn.**

⭐ **THE CROPPER WAS THE VARIABLE.** Of G1's 54 tiles: **0** are a tight crop of the box they are
attributed to, **45** are padded to a ~96 px floor, **5 are the ENTIRE 640×256 native frame**, the
median tile is **4.05×** the tight-crop area, and **none carries a box outline**. A human shown a
wide street scene and asked *"is there a sign here?"* answers about the scene, not about the
detection. **G1's crops were unreadable by construction** — and G1 read them correctly. A blind
re-read reproduces its verdict and is *more* severe. **The adjudicator was never the variable.**

⚠️ **AND THE EARLIER "REFUTATION" OF THIS VERY HYPOTHESIS WAS ITSELF SCOPED WRONG.** The 2026-08-16
reliability study reported rendering **REFUTED** — but it tested a **REIMPLEMENTATION** (`r7` crops
exactly the box, so the sign fills the tile). **The defect lives precisely in the difference between
the reimplementation and the original.** ⇒ *Re-implementing a step and finding it sound does not
test the step that ran; it tests your version of it.* To refute a rendering hypothesis you must read
**the bytes the original produced**, which is what arm C finally did.

⚠️ **A SECOND, QUIETER ERROR RODE ALONG: the corpora were never two corpora.** G1's pilot-50 is a
**strict subset** of the `w120val` leg (overlap **50/50**), and box geometry and score
distributions are indistinguishable across all three legs (median sign box 68.9 / 70.9 / 70.9 px²).
Both the study and `NEXT_4472_BUILD_INPUTS.md` attribute G1 to *"w120val, 600 clips"*; the primary
sources say **the 50-clip pilot** — and the corpus is **596**, not 600. ⇒ **A "corpus difference"
was invented to explain a number, then cited as the reason the two could not be compared.** The
comparison was available the whole time.

⇒ **ROOT-CAUSE CLASS: AN INSTRUMENT DEFECT ATTRIBUTED TO THE SUBJECT.** The same family as reading
`df` on a pod, `free` on Thor, `memory.usage_in_bytes` under a cgroup, and C85's flattened RLE that
summed to the right total — but inverted: here the broken instrument made the subject look **worse**,
and the correction makes a channel usable that had been written off. **Both directions cost the same
amount of truth.**

⇒ **RULES THIS EARNS:**
1. **When two measurements of the same thing disagree, suspect the two INSTRUMENTS before you
   suspect two populations** — a population difference is the more interesting story and therefore
   the more seductive one.
2. **Adjudicate on the artifact the original produced, never on a faithful-looking rebuild.**
3. **A human-adjudication protocol must show WHAT WAS DETECTED, not the region around it.** An
   outline is not cosmetic; it is the difference between asking about a detection and asking about
   a scene.

⇒ **CONSEQUENCE — the sign channel is RELEASED, with limits stated:** admissible as a **per-clip
presence flag at 0.5** on the val side; **≥0.70 if per-detection** (precision 0.920, retains 46.4 %)
but **NOT tuned** — 8 FPs, non-monotone bands. ⛔ **KIND and TEXT stay forbidden**, and the reason is
sharper than a rate: the two highest-scoring false positives are a **dashboard `30` roundel
(0.927)** and a **commercial hoarding (0.778)**. ⇒ **A threshold removes the harmless errors and
keeps the harmful ones** — the very failures that would corrupt a navigation claim score ABOVE the
true signs.

---

## C88 — OUR TWO GIT RULES CONTRADICT EACH OTHER UNDER CONCURRENCY, AND THE "SAFE" ONE IS THE ONE THAT SWEEPS (2026-08-17, orchestrator)

**NOT a retraction of a claim — a retraction of a PROCEDURE.** `CLAUDE.md` says two things that
cannot both be followed while agents are live:

1. *"when the index contains other agents' work, commit with an explicit pathspec"* — because
   `git commit` takes the **whole index**, and this has swallowed siblings' work twice (`60265d3`,
   `3d41bd0`);
2. *"⚠️ `git commit -- <pathspec>` SEGFAULTS on this repo and is NOT usable as the default … prefer
   a pathspec-free `git commit -F <msgfile>`"*.

⇒ **The only commit form that works is the only commit form that sweeps.** It happened a **third**
time tonight: `c98aadb` — titled *"The SAM3 sign confound closes on the INSTRUMENT…"* — also
contains **12 files of an unrelated manoeuvre-label fix** (and, earlier, `06b8782` carried 12
tactical-label files under a lane-change title, and `b2f8bc9` carried 10 sign-adjudication files
under a `DIR_YAW` title).

⚠️ **AND "CHECK THE INDEX FIRST" DOES NOT FIX IT.** I checked, and printed, the index immediately
before every one of those commits. **Listing is not excluding.** Worse, the index moves *between*
the listing and the commit — agents bank incrementally by design, which is behaviour we asked for.

⇒ **ROOT CAUSE: a rule written for a SEQUENTIAL repo, kept unchanged after the programme went
CONCURRENT.** With N agents staging continuously there is no moment at which the index contains
only one stream's work; "commit your own files" stopped being an available action, and the rule
never noticed.

⇒ **THE PROCEDURE THAT ACTUALLY HOLDS, replacing the contradiction:**
1. ⛔ **The commit MESSAGE must enumerate every stream the commit contains**, not just the one that
   motivated it. A sweep that is *named* costs a reader nothing; a sweep that is *silent* makes the
   swept work unfindable — which is the real damage, not the mixing.
2. **Commit at the MOMENT a stream lands**, not in batches. The sweep window is the time since the
   last commit; shrinking it is the only lever that does not fight the tooling.
3. **A swept-in stream must be cross-referenced from its own package** — the manoeuvre-label fix
   lives in `c98aadb` and its report must SAY SO, or `git log --grep` for it returns nothing.
4. ⚠️ **Never "fix" a sweep by rewriting pushed history.** Splitting a pushed commit is the PI's
   call, and the cure is worse than a mis-titled commit.

⭐ **CREDIT: the swept agent caught this, verified nothing was lost** (all 12 files
`HEAD:<path> == git hash-object <worktree>`), **and escalated instead of quietly re-staging** — which
is exactly why the damage is bounded to a title rather than to the work.

⚠️ **A second measurement defect surfaced in the same report, and it is mine:** the baseline I have
quoted in every brief tonight — *"3750 passed"* — is **`stack/` only**. There is a **second suite,
`taniteval`, 1092 tests**, which no brief of mine mentioned and which caught a **real back-compat
break** in this work. ⇒ **A suite total that silently omits a suite is the C82 family again:** a
number whose SCOPE is narrower than the claim it is used to support. Health claims must name which
suites ran.

---

## 2026-08-17 — ⛔ **`D1` IS WITHDRAWN: THE F-18 SLOT PROBE FAILS ITS OWN POSITIVE CONTROL**

**What is retracted.** The pre-registered finding **D1** — *"the encoder does not carry agent
geometry"* — as reported by `…/incoming/2026-08-16-slot-probe-run/SLOT_PROBE_RUN.md` and hardened by
`…/incoming/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md`, **together with its two strongest
derived sentences**: *"the trained v6 latent serves this readout no better than random vectors"* and
*"640 raw patch tokens at 240× the `cells` surface does not rescue it, so the loss is at the ENCODER,
not the readout."*

⚠️ **This is NOT a claim that the v6 latent DOES carry agents.** It is the claim that **the
instrument cannot answer the question**, which is a different and much weaker state than the one the
programme has been reasoning from. The pre-registered **D1 DROP must not be executed on this
evidence**, at 30 k or at any step.

**The measurement that forces it** (`…/incoming/2026-08-17-probe-positive-control/`): the *identical*
probe (`sp2_probe.py`, md5 `aabbee36fce5f164d47a555fad369cbd`, byte-identical), *identical* 2 721
windows / 70 episode clusters / split / estimator, fed a memory tensor that is **a direct encoding of
the frame's own GT boxes**, scores **10.175 m [9.182, 11.168]** against a constant's **5.133 m** —
K1 **+5.042 [+4.080, +6.065]**, separated, FAILED — **at all three seeds** (K1 +5.042 / +4.946 /
+1.946, every one positive and separated). ⇒
**The apparatus ranks PERFECT INFORMATION (10.18 m) BELOW RANDOM VECTORS (5.95 m).**

⭐ **The sharpest form, and it removes every remaining rejoinder:** on a memory that puts the GT lead
at a **fixed, known address**, a **ridge regression recovers it at 1.016 m with r = +0.979**
(K1 −4.116, separated, PASS) while **the identical slot probe on the identical tensor scores
6.319 m and still LOSES to the constant** (+1.178 [+0.313, +2.131], separated) — **6.2x worse than a
linear map on the same numbers.** With that perfect representation AND the readout rule repaired,
the apparatus ceiling is a **TIE** with a constant (+0.522, not separated). ⇒ **This instrument
cannot emit a K1 PASS at all, so its "K1 fails" verdicts carry no information about the
representation.** The information is present and even linearly available; what fails is the 74-slot
decoder plus its `pred_lead` readout.

**⛔ ROOT-CAUSE CLASS — this is the class, not the number.**

> **AN INSTRUMENT VALIDATED ONLY BY NEGATIVE CONTROLS.** Five controls were run across two studies
> (C-CONST, C-SHUF, C-EPMEAN, C-SHUF-XEP, matched random-latent) and every one is NEGATIVE: they
> prove the probe is not **cheating**. **Not one proves the probe can SUCCEED.** A null from such an
> instrument is *unattributable*: "X does not carry the signal" and "this apparatus cannot read the
> signal from anything" are indistinguishable.

**Why the negative controls were structurally unable to catch it — MEASURED, not argued.** The
incumbent readout selects `argmax presence` over in-corridor slots **with no range cap**, while the
GT selects the **nearest** in-corridor agent **within 30 m** (`sp2_probe.py:141` vs `:99`) — and the
decode runs to 60 m. On the oracle arm **15.98 %** of selected slots are beyond 30 m; on the
random-latent null **0.00 %** are, because a degenerate head emits ~20 m on every frame. ⇒ **The
defect is INVISIBLE on a broken arm and only bites on a working one.** No negative control can
expose a bug that only a functioning representation triggers.

**Sibling instances of the same class already in this log / in `CLAUDE.md`:** C13 — *a guard that
cannot fail* (K3 pinned at ≈0.50 by construction: a head trained on pure noise "passes"); C9/C14 —
*instruments structurally unable to report the answer they are cited for*; the `df` / Thor `free` /
cgroup `usage_in_bytes` family — *a probe that reports the wrong scope is worse than no probe,
because it looks like an answer.* **This one is the sharpest form yet: the probe reported a
confident, well-controlled, five-times-replicated answer to a question it could not answer.**

**⭐⭐ AND THE REPAIR IS IDENTIFIED, SO THIS IS NOT A DEAD END.** The same probe on the same oracle
cache with **`--n-queries 16` instead of 74** scores **2.982 m, median 0.816 m, K1 −2.186
[−3.165, −1.192] separated — the FIRST K1 PASS ANYWHERE IN F-18** (the constant on the retained
windows is unchanged, 5.167 vs 5.133, so it is not an abstention artefact). **74 was fitted from the
in-grid AGENT-COUNT p99 — correct for SET PREDICTION, catastrophic for the LEAD functional the
metric scores**, because it puts ~13 slots in the 3.5 m corridor for `argmax presence` to choose
between. ⚠️ One seed; the same run MEASURED 3.096 m of K1 spread across three seeds at 74 queries,
so this must be replicated at ≥3 seeds before "16 is the fix" is quotable.

⭐ **AND THE REPAIR CARRIES ITS OWN NEGATIVE CONTROL, so it is not this log entry's own error class
one level up:** the **window-matched random-latent null at the same `n_queries` 16 FAILS K1 by
+4.808 [+4.112, +5.482], separated**, with K2 and C-SHUF-XEP correctly unseparated. ⇒ **At 16
queries the instrument passes on the answer (−2.186) and fails on noise (+4.808) — a ~7 m
separation, and the first configuration of this probe that does both.**

⭐⭐ **AND THE REAL ARM AT THAT REPAIRED POINT IS THE FIRST F-18 READING FROM A WORKING INSTRUMENT:**
`v6F-SW-30k@11250` @ 16 queries scores **8.331 m, K1 +3.217 [+2.310, +4.246] — still FAILS**, with
**both anti-echo controls UNSEPARATED (K2 +0.205 ns, XEP +1.225 ns), like noise and unlike the
oracle** (whose K2/XEP separate by 3.1 / 6.1 m). ⇒ **D1 points the same way it always did — but now
on evidence that can bear it.** ⛔ **IT IS STILL NOT RESTORED, AND MUST NOT BE QUOTED AS RESTORED:**
one seed against a measured 3.096 m seed spread; arm-vs-arm ordering that is MARGINAL, not paired
(window sets 2 408 / 2 577 / 2 665, each arm's K1 paired only against its own C-CONST); and an
early-read at 37.5 % of training. ⭐ **The path to a valid D1 is now short and specified:
{oracle, latent, null} @ 16 queries × ≥3 seeds, re-run at 30 k — ~9 fits, no trunk compute.**

⛔ **A SECOND CRITERION FALLS WITH IT: `K3` IS ANTI-CORRELATED WITH QUALITY.** The parity run
escalated K3 as vacuous (a noise head scores 0.5002 and "passes `K3 ≥ 0.50`"). The positive control
shows worse: the nq-16 arm — **median error 0.816 m, K1 PASS** — scores **0.4432 and "fails"**. ⇒
**K3 ranks a noise head above the only arm that has ever passed K1. Remove it from the KEEP gate;
do not re-threshold it.**

**⇒ THE STANDING RULE THIS EARNS.**

> ⛔ **NO FROZEN-LATENT PROBE IN THIS PROGRAMME MAY REPORT A NULL WITHOUT A PASSING POSITIVE
> CONTROL** — a representation constructed to contain the answer, run through the *identical* fit,
> windows, split, seeds and estimator. If the positive control fails, the null is **withdrawn**, not
> softened. The cost is one cache rewrite (seconds) plus one fit (~20 min):
> `…/2026-08-17-probe-positive-control/code/pc1_oracle_cache.py` + `code/chain_a.sh` do it end to end,
> and the move generalises — **replace the memory with an encoding of the label, change nothing
> else.**

**Also corrected in the same package** (details in `PROBE_POSITIVE_CONTROL.md`):
* ⚠️ **`build_o4_weights` is a SAMPLER, not a loss term** (`stack/scripts/train_v6_staged.py:745`,
  consumed at `:2470` by `InteractionSampler`). It has no gradient; "the cosine of O4's gradient"
  is not a quantity that exists. Any brief asking for it must be redirected to a sampler analogue.
* ⚠️ **`lead130_agents.jsonl` carries NO image-space box** — `{cx, cy, yaw, l, w, occ, track_id,
  cls}` only. Pixel extents must be DERIVED from `f_ref` and are **ESTIMATED**; height is absent, so
  pixel AREA is not derivable at all.
* ⭐ **The "small agents" worry does not apply to this metric:** the GT lead is median **37.8 px
  wide ≈ 2.4 ViT patches**, with only **4.34 %** of scored windows below one patch. The failure is
  worst where the agent is **nearest and biggest**, and the random-latent null reproduces the entire
  stratum profile — so the profile belongs to the label distribution, not to the latent.

---

## C89 — WE READ THE TACTICAL LABELS AT THE HORIZON THAT MAXIMISED AGREEMENT, AND CALLED IT THE ARCHITECTURE'S BAND (2026-08-17, caught by the PI)

**RETRACTED:** *"Both axes peak at 2.0 s — exactly the v6 tactical band"* (tactical-label
validation), and my own relay of it into the review-sheet brief: *"Read at the 2.0 s horizon …
2.0 s is the v6 tactical band."* **The whole 40-clip review sheet the PI was asked to adjudicate is
built at the wrong horizon.**

⛔ **2.0 s IS NOT THE TACTICAL BAND. IT IS THE SEAM.** From `v6.py` §4b, the binding horizon spec:

```
PLAN_STEPS = 60 · DT = 0.1 · HORIZON_S = 6.0
OP_BAND_S  = (0.0, 2.0)   # operative
TAC_BAND_S = (2.0, 6.0)   # TACTICAL
```

2.0 s is where operative authority **ends** and tactical **begins** — the single least
representative point in the plan for a tactical claim.

⇒ **AND THE SWEEP HAD THE RIGHT NUMBER ALL ALONG** (`a4_horizon_sweep.json`, 201 clips, MEASURED):

| horizon | LON κ | LAT κ |
|---|---|---|
| **2.0 s** (seam, chosen) | **0.3655** | **0.4694** |
| 3.0 s | 0.3444 | 0.4622 |
| 4.0 s | 0.2924 | 0.4386 |
| **6.0 s** (`TAC_BAND_S` end) | **0.2331** | **0.4040** |

**−36 % longitudinal and −14 % lateral** between the horizon we reported and the horizon the
architecture defines. The better number was not discovered; it was **selected**.

⇒ **ROOT-CAUSE CLASS: A MEASUREMENT WINDOW CHOSEN BY ITS RESULT, THEN JUSTIFIED BY A SPEC IT DOES
NOT MATCH.** This is the exponent rule (`CLAUDE.md`) in a new costume — *"the same log gives
−0.387/−0.505/−0.564/−0.621/−0.738 depending on the window"* — but worse, because here a **binding
constant existed** (`TAC_BAND_S`) that settles the question in one grep, and nobody ran it.

⚠️ **The ridge check was sound and still did not save us.** The report correctly distinguished a
ridge from a fitted spike (16 of 72 cells within 0.05 κ of the peak) — but the ridge spans
**0.5–3.0 s**, i.e. the OPERATIVE band plus the seam. ⇒ *A robustness check tells you the peak is
real; it cannot tell you the peak is in the right place.* Robust and relevant are different
properties, and we verified only the first.

⚠️ **The pre-registered hypothesis was also confirmed and still misused.** H_HORIZON (*"agreement
rises as the horizon shortens"*) was upheld — but "agreement peaks short" does **not** license
reading a TACTICAL label short. The architecture defines the band; the data does not get a vote on
which quantity we are measuring.

⇒ **RULES THIS EARNS:**
1. **A horizon, window, or band is a SPEC LOOKUP, never an argmax.** If a binding constant defines
   it, quote the constant and its file:line beside the number.
2. **When a sweep is run, report the value AT THE DEFINED POINT first**, and the argmax second and
   labelled as such. We had 6.0 s in the same JSON and led with 2.0 s.
3. **State the horizon on every tactical artefact** — the sheet did say "2.0 s", which is why the
   PI caught it. *Labelling the choice is what made the error findable; the error was making it.*

⇒ **CONSEQUENCE:** the review sheet must be **rebuilt over (2.0, 6.0] s** before the PI's
adjudication means anything, and every tactical κ quoted this session (0.3655 / 0.4694) must be
restated at the band — ~~**0.2331 / 0.4040**~~ — or explicitly marked as seam values.

### ⛔ C89b — APPENDED 2026-08-17 BY THE REBUILD AGENT: **THIS RETRACTION'S OWN CORRECTION WAS ALSO WRONG.**

⛔ **`0.2331 / 0.4040` ARE NOT BAND VALUES. Do not quote them as such** — they are what C89 above
told the next reader to use, so this is the highest-propagation-risk number in the log.

`…/2026-08-16-tactical-labels/code/tac_a4_horizon_sweep.py:140-148` anchors **every** horizon at
`t0`:

```python
k  = min(int(round(t0 + H * POSE_HZ)), poses.shape[0] - 1)
dv = float(v[k] - v[t0])          # <-- anchored at t0, at EVERY horizon
```

⇒ the row labelled `2.0` is the net change over **(0.0, 2.0]** — i.e. exactly `OP_BAND_S`, the
**operative** band — and the row labelled `6.0` is **(0.0, 6.0]**, the **full horizon with the
operative band inside it**. ⇒ **NO ROW OF THAT SWEEP IS THE TACTICAL BAND**, so no restatement was
available from the banked JSON at all; the band quantity had to be computed fresh (anchored at
`t0+20`, read across `t0+21 … t0+60`).

⭐ **THE ACTUAL BAND VALUES** — MEASURED, 201 clips, episode-cluster bootstrap (`taniteval/ci.py`),
PRODUCTION thresholds, statistic `mean_band` (mean in-band deviation from the band start):

| window | LON κ [95 % CI] | LAT κ [95 % CI] | n (LON/LAT) |
|---|---|---|---|
| ⭐ **`TAC_BAND_S` (2.0, 6.0]** | **0.1428** [0.0540, 0.2250] | **0.1777** [0.0658, 0.2953] | 201 / 193 |
| ⚠️ seam (0.0, 2.0] = `OP_BAND_S` | 0.3270 [0.2289, 0.4192] | 0.3132 [0.1973, 0.4323] | 201 / 193 |
| ⚠️ full horizon (0.0, 6.0] *(the "0.2331/0.4040" row)* | 0.2210 [0.1165, 0.3167] | 0.3806 [0.2587, 0.4911] | 201 / 193 |

Paired band−seam: **LON Δκ −0.1843** [−0.2746, −0.0961] · **LAT Δκ −0.1354** [−0.2707, −0.0162],
**both CI-separated**. The true band is **worse than either number C89 offered**.

⇒ **ROOT-CAUSE CLASS: A CORRECTION ADOPTED FROM THE NEAREST NUMBER ALREADY IN THE ARTIFACT, WITHOUT
RE-DERIVING THE QUANTITY THE NEW NAME DEMANDS.** C89 correctly identified that the *name* was wrong,
then reached for the row whose *label* matched the corrected name — never checking that the row
computed the corrected quantity. **The sweep's column header said "horizon", and a band is not a
horizon.** *(Same family as C89 itself, one level down; and a sibling of C91 — a value taken from
what was written about an artifact rather than from what the artifact computes.)*

⇒ **RULE: when a retraction supplies a replacement number, state the LINE OF CODE that computes it.**
If the replacement comes from an existing artifact, quote the expression — not the column label.
A corrected name over an uncorrected quantity is a second error wearing the fix's clothes.

⚠️ **A third argmax was in the same artifact and neither C89 nor the sheet caught it:** the 2.0 s
sheet's thresholds (`Δv 0.75 / Δyaw 0.05`) were *also* selected as the κ-maximising cell, described
in its own source as *"the thresholds that maximised κ at this horizon"*. **The horizon and the
threshold were both argmaxes; C89 caught one.** ⇒ When you find one window chosen by its result,
**enumerate every other free parameter in the same measurement** before declaring the fix complete.

**Artifacts:** `…/2026-08-16-tactical-review/code/tacrev_band_agreement.py` ·
`…/raw/b1_band_agreement.json`. Rebuilt sheet: `…/review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html`.

---

## C90 — A CORRECTION WAS WRITTEN INTO A DOCUMENT AND NEVER APPLIED, FOR THREE WEEKS (2026-08-17, registry sweep)

**MEASURED:** `CLOSEDLOOP_RERUN.md:389` states, in its own text, that divergence must go
**22.2 % → 23.5 %** and that this **"needs edit"**. Dated **2026-07-26**. **It was never done.** The
banned-estimator value stood in every downstream reader for three weeks while the document that
knew it was wrong sat beside it.

⇒ **ROOT-CAUSE CLASS: C70 EXACTLY — a correction recorded in a place with no mechanism to apply
it.** *"Needs edit"* is not a task; it is a note to a reader who never came. This programme has now
lost an orthogonality instrument for 10 days, a merge request for 12, and a numeric correction for
21 — all to the same shape: **the fix was known, written down, and had no owner.**

⇒ **RULE: a correction that cannot be applied in the same turn must become a TRACKED ITEM, not a
sentence.** If it is worth writing *"needs edit"*, it is worth a task chip or a failing test. A
document is where you record that something WAS fixed, never where you record that it SHOULD be.

⚠️ **And the sweep that found it was unbounded on purpose.** A bare numeric regex over 4,939 files
returned **229 files** — mostly coincidental floats; context-gating cut it to **76 lines / 23
files**. *The self-match family again* (`pgrep -f`, `grep failed` matching `xfailed`): a filter
matched on the shape of the thing rather than its meaning.

---

## C91 — WE TOOK A VERDICT INVENTORY FROM THE HEADLINE INSTEAD OF THE ARTIFACT (2026-08-17, registry sweep)

**RETRACTED:** the standing understanding that the banned estimator put **two** verdicts on the P2
path (G1, G4) and that both had been re-decided.

**MEASURED:** enumerating **every boolean** in `planner_p2_flagship-30k.json` returns **FIVE**, not
two. The third that matters is **`planner_beats_cv`**, and unlike G1/G4 **its flip is REACHABLE**:
it is banned on *both* sides, the corrected CV floor (**0.8377**) is *higher* — moving the
comparison **toward** the planner — and flipping needs **+6.59 %** against a measured local upper
edge of **+5.877 %** and a programme-wide **+11.69 %**.

⇒ ⛔ **`planner_beats_cv` is UNDECIDED, not "no flip".** Materially unlike G1's −73.6 % margin. It
**must not be quoted in either direction** until the re-drive; the same ~400 s GPU job settles it
and G1 together.

⇒ **ROOT-CAUSE CLASS: an inventory of what a result CLAIMS, taken from what was WRITTEN ABOUT it
rather than from the artifact.** The two verdicts we knew about were the two the headline mentioned.
Nobody opened the JSON and counted. ⇒ **RULE: before declaring an artifact's verdicts re-decided,
ENUMERATE every verdict field in the artifact itself.** A verdict nobody listed cannot be
re-decided, and its silence reads exactly like agreement.

⚠️ **Two more of the same shape in the same sweep:**
* **All three** closed-loop baseline numbers were banned, not just the threshold (**FDE 3.5296 →
  3.6190**, **divergence 0.2216 → 0.2350**). All moved the **same way** — the v1 closed-loop failure
  was **UNDERSTATED** — so every P2-vs-head margin **widens** (drift reduction 38 % → **42.9 %**;
  divergence 2.5× → **3.2×**). *A bias that flatters the thing you are arguing against is the last
  one you notice.*
* The registry **publishes the same paired delta twice, 800 lines apart**, with intervals differing
  **3.31×** — REF-A `[2.447, 2.798]` (banned, unlabelled) vs `[2.0945, 3.2570]`. That ratio sits
  **above the top of the documented 1.107–3.100× narrowing band**, on a paired delta.

⭐ **AND THE MOST CONSEQUENTIAL SITE WAS NOT IN THE REGISTRY AT ALL:** `Paper/TANITAD_PAPER.md:2847`
carried banned-estimator bars **inside the PRE-REGISTERED GATE LIST** (G4 **1.69 → 1.7318**, G5
**0.452 → 0.4271**). ⇒ **A defective statistic was defining the bar that FUTURE arms must clear** —
and the old bar is **HARDER** than justified, so arms were being held to a threshold the estimator
had inflated. *A wrong number in a results table misreports the past; a wrong number in a gate
misdirects the future.*

⚠️ Also fixed, and worth naming as its own shape: `MODEL_REGISTRY.md` §1.3 read **"Cite 0.628
(heldout)"** — **not a stale number but a STANDING INSTRUCTION to prefer the banned statistic**.

---

## C92 — I QUOTED A POSITIVE PROBE WITHOUT A TRIVIAL-PROXY CONTROL (2026-08-18, O2/O3/O4 research)

**RETRACTED — my own claim**, published in `Project Steering/Reports/2026-08-17-2319-program-report.md`
§3 EFFICIENCY and repeated to the PI: *"the linear-readout finding (**~1.8 m better than the random
null, r +0.159**) says the information is present and weak."*

**MEASURED:** that margin is an **EGO-SPEED PROXY**. **Ego speed ALONE beats the entire 2048-dim
latent** (K1 **−1.562 PASS** vs **+1.580 FAIL**), and partialling `v0` out of the latent readout
leaves **r = +0.052**. The latent was not being read; `v0` was leaking through it.

⇒ **ROOT-CAUSE CLASS: A POSITIVE PROBE QUOTED WITHOUT A TRIVIAL-PROXY CONTROL.** We have an
elaborate negative-control discipline — C-CONST, C-SHUF, C-EPMEAN, C-SHUF-XEP, random-latent nulls —
and every one of them asks *"does the instrument cheat on noise?"*. **None asks the opposite
question: "is a one-number baseline already enough?"** So a real margin over noise was read as
capability when a scalar the model is HANDED explains it.

⚠️ **This is the SAME SHAPE as C79 (the D1 withdrawal), one week apart and in the opposite
direction.** There, five negative controls could not see an instrument that failed its **positive**
control. Here, the negative controls could not see a positive result that failed its **triviality**
control. ⇒ **RULE: every probe carries BOTH — a positive control (can it read the answer when
handed the answer?) and a trivial-proxy control (does a scalar already in the input do as well?).
A margin over a random null is not evidence until both are reported.**

⭐ **AND IT IS THE ECHO TEST AGAIN, IN A THIRD COSTUME.** Flagship v1's route head scored 1.0000 by
echoing its own nav input; the open-loop lateral skill was an action echo (97.9 % vs 0.0 %
hold-action); this readout echoed `v0`. **Three times now the programme has measured a model
returning its own input and called it a capability.**

⚠️ **SEPARATE INSTRUMENT DEFECT, SAME PACKAGE — `pc6_linear_readout.ridge_fit` PENALISES ITS OWN
INTERCEPT.** A no-signal arm is therefore driven toward **zero**, never toward the **mean** — so it
scores worse than a constant predictor by construction. **This taints every prior "K1 FAIL" produced
by that module**, and the repair currently lives in one run's `code/` rather than in the module. ⇒
Propagate it, then re-read the affected FAILs; a FAIL from a biased floor is not a finding.

---

## C93 — "O2 IS AN INDEPENDENT OBJECTIVE" IS FALSE BY CONSTRUCTION, AND THE CODE SAID SO (2026-08-18)

**RETRACTED:** the standing framing of **O2, O3, O4 as three comparable, independently weightable
loss directions**, and with it the treatment of the measured **O2↔O5 cos +0.870** as an empirical
near-collinearity awaiting explanation.

**MEASURED, by derivation from our own source:** `V6Stack.cells()` (`stack/tanitad/models/v6.py:3710`)
is a **PURE RESHAPE**, so O2 and O5 score the **identical tensor**, and the algebra closes exactly —
**O2 = O5's step-`j` term + Cov_c(w, err)**. The +0.870 is **forced by the code**. It was never a
finding; it was a restatement.
⇒ O2's **unique** content over 7 banked rows is **0.45–3.33 %, median 1.81 %, and sign-unstable
(4 −, 3 +)**. *(Initialisation-time rows; the live-log read is the zero-GPU instrument `E-O2-A`.)*

⇒ **And O4 is a SAMPLER, not a loss** — `|jerk| + |decel| + steering reversals` **from ego actions
alone**, with no perception content and no gradient. It cannot distinguish lead-vehicle braking from
stop-line braking. **`InteractionSampler` is a misnomer**; O4 must stop being described as an
interaction lever.

⇒ **ROOT-CAUSE CLASS: WE DEBATED THE WEIGHTS OF TERMS WHOSE ALGEBRAIC RELATIONSHIP WE HAD NEVER
DERIVED.** A weighting question presupposes independent directions. One `reshape` settles it, and
the trainer had been **logging both halves all along** (`o2_unweighted` *is* O5's step-`j` term).
⇒ **RULE: before tuning the relative weight of two loss terms, derive their relationship from
source. If one is a reparameterisation of another, no weight fixes it.**

⭐ **THE FINDING THAT SUBSUMES ALL THREE, AND IT IS NOT A LOSS PROBLEM AT ALL:** every latent
objective sits behind a **40:1 average pool** — `AvgPool2d((4,10))`, each of 16 cells the mean of
**40 ViT tokens (64×160 px)**. The median GT lead vehicle is **37.8 px**. ⇒ **An object smaller than
one cell is averaged away before any objective sees it.** That is an **architectural bottleneck of
the POOLING kind — which destroys individuation — not of the COMPETITION kind, which no
re-weighting of O2/O3/O4 can address.**
⭐ **Two independent methods, one answer:** the linear ladder's rung profile — **agent count highest
(r² 0.076), relative motion exactly 0.0000** — is precisely what a 40:1 pool predicts, and it was
derived without reference to the pooling analysis.
⇒ **This explains D1 without any appeal to the world model's competence.** *(⚠️ It also refutes my
own earlier "sub-patch resolution limit" hypothesis in its details while vindicating its instinct:
the lead is LARGE in patches — 2.4 — and small only relative to the POOL.)*

⛔ **"The objectives just need more training" is REFUTED at three independent sources** (a ~8k-step
plateau; locality DEGRADING with longer training; *"capacity scaling cannot recover the missing
structure when the objective fails to encode it"*). ⚠️ And the one published case of object structure
emerging from our objective class **failed on a narrow corpus** — which our 2376 driving episodes
are. **No weight, term, or architecture change addresses that**; it is a corpus-composition question.

---

## C94 — THE TACTICAL SELECTOR WAS UNREACHABLE FROM **BOTH** SIDES, AND WE ONLY EVER LOOKED AT ONE (2026-08-18, E4 resolution)

**MEASURED:** SEL-1's **pre-registered reopening path** — the only mechanism that can lift the
selector refusal — **could not return `FUNDED` for any measurement whatsoever.**
`read_sw_admission` looked for a **top-level `sigma_2s_m`**; `e_wc2_sigma_star.py` writes
`references_and_ratios.sigma_perax_2s_m`. **The name AND the nesting level both differed**, so the
join never resolved.

Proved by executing the real estimator on a dump with a **planted σ = 0.30 m** — deep inside the
`FUNDED` band (≤ 0.80). The estimator recovered **0.3026**; the reader still returned
`verdict: null`, and `assert_selector_admissible` **REFUSED the launch.**

⇒ ⭐ **THIS IS THE EXACT MIRROR OF E4.** E4 is a gate that **could not report PASS**; this is a gate
that **could not report FUNDED**. One blocked the finish, the other blocked the start — and
**together they made the PI's *"eventually we need a tactical selector"* unreachable in both
directions at once.** We spent days debating which repair E4 deserved while the *other* end of the
same path was equally dead and **nobody had probed it**, because a refusal that never lifts looks
exactly like a refusal that is correct.

⇒ **ROOT-CAUSE CLASS: A FIXTURE THAT MODELS THE CONSUMER'S EXPECTATION INSTEAD OF THE PRODUCER'S
OUTPUT.** `test_v6_chain.py` hand-wrote `{"sigma_2s_m": …}` — the shape the reader *wanted* — so the
**join was never exercised** and a green suite certified a connection that did not exist. Same family
as `touch_ancestor`'s config-less `stage_gate.json`.
⇒ **RULE: a fixture standing in for another component's output must be GENERATED BY THAT COMPONENT,
or the test proves only that the consumer agrees with itself.** The pin now **runs the real
estimator**.

⚠️ **The one remaining blocker is a SCRIPT, not a decision.** Step 1 of the four-step selector recipe
**does not exist** — nothing dumps v6 S-W latents in E-WC2's contract (MEASURED at three probes;
`refc_dump_latents.build_model` builds a `RefCModel`). Steps 2–4 run today. It is now **emitted by
`v6_chain.py admission` flagged `⛔ NOT BUILT`** rather than left in a docstring — because a missing
step recorded only in prose is C90. ⏳ **Its only cheap window is the S-W→S-T boundary, ~5 days out.**

---

## C95 — THE MECHANISM IN BOTH PRIOR E4 REPORTS WAS WRONG, AND THE FIX THEY IMPLIED WOULD HAVE CHANGED NOTHING (2026-08-18)

**RETRACTED:** both prior E4 write-ups name `if w.w_select:` as the reason the S-T gate cannot read
`sel_gap`.

**MEASURED:** that line is real, and it is **not the emitter the gate consumes.** `run_stage_gate`
reads only `--gate-probes`, `X3_isolation`, `spectrum`, `x4_spectra`.
⇒ ⛔ **Option (a) — enabling `--selector goal --w-select 1.0` — would have made the T0 log key
appear and left the gate EXACTLY as `INCONCLUSIVE`.** We would have spent **3.15 GPU-days** and a
deleted pre-registration to move a number the gate never reads.

⇒ **ROOT-CAUSE CLASS: A PROBE READ AT THE WRONG SCOPE — two same-named quantities living at two
different eval tiers.** The verdict was right and the mechanism was wrong, which is the most durable
kind of error because the conclusion keeps confirming it. **Same family as every scope trap in the
trap list** (`df` on a pod, `free` on Thor, cgroup `usage_in_bytes`, the per-worker RAM figure): a
reading that is TRUE SOMEWHERE, quoted where it does not apply.
⇒ **RULE: before fixing a gate, read what the GATE consumes — not what the trainer logs under the
same name.**

⚠️ **AND THE AGENT'S OWN FIRST REPAIR INVERTED THE DEFECT, IN ITS OWN WORDS:** *"I set out to fix a
gate that couldn't report PASS and briefly built one that couldn't report FAIL."* Excluding a
not-applicable criterion **unconditionally** discarded an incumbent test's **planted FAILING**
`sel_gap`, and the gate read **PASS on a rung that had FAILED**. ⇒ **A supplied verdict now always
beats the predicate**, with disagreements surfaced in `applicability_conflicts` rather than silently
resolved. **Trilemma proved on constructed inputs: PASS/FAIL/INCONCLUSIVE × both arms = 6/6**, plus
vacuous-gate refusal and FAIL-never-softened.
⭐ *The general lesson: **when you loosen a gate, test the direction you were not trying to fix.**
Every loosening is a candidate FAIL-suppressor.*

⇒ **RESULT:** S-T with the battery folded in now reads **PASS** instead of `INCONCLUSIVE` **forever**.
`sel_gap` is **not deleted** — it stays `required` and binds on any arm with a scorer (resolved from
`stack.cand_score`, **not** the flag), so **S-S is solved rather than moved**. Launch line
**diffed, not asserted**: byte-identical, md5 `5381f2ea…` both sides, `--v2-lru 64` carried.

---
## C96 — C93's NUMBERS ARE OVERTURNED BY THE LIVE LOG: O2's UNIQUE CONTENT IS **34 %, PERFECTLY SIGN-STABLE, AND GROWING** (2026-08-18)

**RETRACTED — the numeric half of C93, logged by me hours earlier.** I wrote that O2's unique
content is *"0.45–3.33 %, median 1.81 %, and sign-unstable (4 −, 3 +)"*, and that this made O2 not
separately weightable.

**MEASURED on the LIVE run** (v6F-SW-30k, steps 50–12 650, **n = 254** training rows, log md5
`370e778b0b7f79917c94302337f142c1` verified both ends):

| | banked (O234 §2.1a) | **live** |
|---|---|---|
| source | dry ladder, **steps 1–2, batch 2** | steps 50–12 650 |
| n | 7 | **254** |
| `|Cov|/unweighted` | median **1.81 %** | **median 34.23 %** (4.18–60.10) |
| sign | unstable, **4 −/3 +** | ⭐ **254 −, 0 + — perfectly stable** |
| trend | — | 16.2 % → 46.4 %, **late/early 2.59×** |

⇒ **The pre-registered kill criterion — *"if it stays sign-unstable, O2 is not separately
weightable"* — DOES NOT FIRE.** Cross-checked by running the O234 package's **own unmodified
instrument** on the live log: identical to every digit.

⚠️ **WHAT SURVIVES AND WHAT DOES NOT.** The **algebraic identity stands** — `cells()` is a pure
reshape, so O2 = O5's step-`j` term + `Cov_c(w, err)` is a *derivation*, not a measurement. What
dies is the **conclusion drawn from it**: the covariance term is not a rounding error, it is a
**third of the objective and rising.**

⇒ **ROOT-CAUSE CLASS: A DECISION-GRADE NUMBER TAKEN FROM A DRY LADDER AND READ AS THE LIVE RUN.**
The banked n = 7 was narrower even than its own caveat admitted — identical `w` extrema across all
four files means **one replayed dummy batch, 2 windows, `o5_k` 12 against the live 60.** ⛔ **And the
other leg of "O2 is redundant" is the same class:** the `cos(g_O2, g_O5) = +0.870` came from a probe
whose **own metadata reads "SYNTHETIC CPU build — NOT the live checkpoint"**, at **732,541 params
against the live 336,542,025.** ⇒ **BOTH LEGS WERE INITIALISATION-TIME.** Same family as
operating-standard rule 1: a claim that decides a GPU-day must be MEASURED on the thing it decides
about.

⭐ **AND THE SIGN IS THE REAL FINDING, not the magnitude.** `Cov < 0` on **all 254 rows** means O2
**systematically DE-EMPHASISES the high-error far field.** So O2 is not a redundant term doing
nothing — it is a **substantial term aimed away from the agents**, and it is doing *more* of that
as training proceeds. That is a stronger reason to change it than redundancy ever was.

⇒ ⛔ **DO NOT quote the 1.81 % / sign-unstable figures again.** `MODEL_REGISTRY.md` and any O2
weighting decision use the live-log values.

---

## C97 — THE C92 RIDGE REPAIR OPENED THE MIRROR-IMAGE DEFECT, EXACTLY AS C95 WARNED (2026-08-18)

**MEASURED:** under the repaired ridge (`intercept_col=-1`), a fully-shrunk fit **is** "predict the
train **mean**" — while `C-CONST` is the train **MEDIAN**. On a skewed target K1 therefore
degenerates into **mean-vs-median**, and a **pure `torch.randn` null now "PASSES" `n_agents_all`
at −1.884**, with `pred_sd` **0.715** against `gt_sd` **46.459**.

⇒ **C92 made no-signal arms FAIL by construction. This makes them PASS by construction.** The same
criterion, biased in the opposite direction, by its own repair.

⇒ ⛔ **A `pred_sd/gt_sd` degeneracy guard is REQUIRED before any repaired PASS is quotable.**
`lead_gap` is unaffected (its target is not skewed that way), so the re-read verdicts below stand.

⇒ **ROOT-CAUSE CLASS: EXACTLY THE ONE C95 NAMED HOURS EARLIER — *"when you loosen a gate, test the
direction you were not trying to fix."*** There it was an E4 repair that briefly built a gate which
could not report FAIL. Here it is a floor repair that lets noise report PASS. **Every correction to
a criterion is a candidate bias in the opposite direction, and the repair is not finished until both
directions are tested.**

### What the re-read actually returned (all 5 incumbent refits reproduce banked BIT-EXACTLY, so only the intercept penalty moved)

| arm | before | after | verdict |
|---|---|---|---|
| `v6F@11250` | +1.580 | **+0.736** | separated — **FAIL SURVIVES** |
| `v6F@9000` | +1.811 | **+1.291** | separated — **FAIL SURVIVES** |
| `RANDOM-NULL` | +3.401 | **+0.043** | **not separated — DIES** |
| `GT-ORACLE-DIRECT` | 1.016 m | **0.580 m**, r +0.9932 | **PASS survives and STRENGTHENS** |

⇒ **2 of 3 FAILs survive an unbiased floor; the third was 98.7 % instrument.** Mechanism measured:
with the bias penalised, the noise arm had to **load features** to reach y's 15 m level, giving
`pred_sd` **8.468** against `gt_sd` 6.200; repaired, it collapses to **0.131** — a constant.

⭐ **AND C92 IS SHARPER THAN I STATED IT.** O234 §3.4's *"the latent beats a random null by
1.6–1.8 m"* does not merely shrink — **it INVERTS, separated in both directions** (−1.821 → **+0.691**;
−1.590 → **+1.246**). An alpha sweep bounds the truth: at the **eval-optimal** alpha — cheating in the
arm's own favour — the latent reaches **5.113 m** against a constant's **5.133** and noise's **5.175**.
⇒ **~0.02–0.07 m, not 1.6–1.8 m: a 25–90× overstatement, and NO alpha anywhere reaches a PASS.**
The arm beats the null on the **inner split** and loses on **held-out episodes** ⇒ **episode-level
overfitting, not agent geometry.**

⚠️ **STILL OUTSTANDING, ZERO-GPU:** enumeration by opening artifacts (C91) found **24 files, 214
verdict rows, 170 still standing on the biased solve — 90 of them separated FAILs.** The ladder
shares the same `ridge_fit`. Those 90 are not yet findings.

⭐ **A REASSURING MEASUREMENT, recorded because the alarming version was the natural guess.** A
repo-wide audit found **15 ridge implementations**, of which 5 use the `np.eye` idiom. **All four
others already handle the intercept correctly** — each carries `A[-1,-1] -= lam` with a comment
reading *"do not penalise the intercept"*, several of them weeks OLDER than `pc6`. ⇒ `pc6` was
**not the norm; it was the sole regression from an idiom the repo had already written down four
times.** *(And my own "there is exactly ONE copy of the module" was true of the FILENAME and false
of the DEFECT — I searched for `pc6_linear_readout*` rather than for `ridge_fit`. Absence found by
one search pattern is not absence.)*

---

## C98 — C94'S DEFECT HAD A SECOND DOOR, AND THE OBVIOUS READING OF THE TASK WALKS STRAIGHT INTO IT (2026-08-18, S-W latent dumper)

**MEASURED while building step 1 of the selector-admission recipe:** `sigma_perax_2s_m` is written
**only inside `if refs.available`**, and `fan_references` requires `fan`/`gt`.

⇒ ⛔ **A latent-only dump — which is exactly what *"dump the frozen S-W latents"* sounds like —
produces an admission artifact with NO σ AT ALL, and the gate stays dead.** Not an error, not a
crash: a complete-looking artifact missing the one field the whole path exists to carry.
⚠️ **`DUMP_CONTRACT` filed those keys under `required_for_the_ratios`**, which reads as *optional
extra precision* rather than *required or the output is empty*. That wording is why it was easy to
miss. Fixed: executed, pinned, contract text corrected, `fan` made **non-optional**.

⇒ **ROOT-CAUSE CLASS: C94 EXACTLY — a producer→consumer join that no test exercised end-to-end —
recurring within HOURS of C94 being logged, one door further along the SAME path.** C94 was the
reader looking at the wrong address; this is the writer conditionally not writing at all. **One
repair of a broken join does not make the rest of the join sound**, and the second failure is
harder to see because it produces a *plausible* artifact rather than a missing one.
⇒ **RULE, strengthened: for any producer→consumer path, enumerate EVERY field the consumer needs
and prove each one is written under the conditions the caller will actually run in** — not merely
that the two agree on a schema.

⭐ **THE ROUND TRIP NOW PASSES ON PLANTED VALUES, END TO END** (881 windows / 40 episodes; producer
→ real estimator → real chain reader → `assert_selector_admissible`):

| planted σ | recovered | error | chain verdict | selector launch |
|---|---|---|---|---|
| 0.30 | **0.3046** | +1.53 % | **FUNDED** | **ADMITTED** |
| 1.10 | **1.1274** | +2.49 % | INCONCLUSIVE | REFUSED |
| 2.00 | **1.9856** | −0.72 % | REFUSED | REFUSED |

⭐ **And the plant is non-circular by construction**, which is the part that answers C94's class: the
corpus is built in **two passes** so σ is planted against the **encoder's own latents**, and pass 2's
latents are asserted **bit-identical** to pass 1's. **No test hand-writes a dump.**

⚠️ **THREE THINGS THAT MUST TRAVEL WITH THIS, or the next reader misdiagnoses a healthy run:**
1. ⛔ **Expect `NO_VERDICT` from the estimator on the REAL run — that is NOT a failed dump.** The
   live arm has `selector: null`, so there is no `sel` and σ/ADE is uncomputable; the chain's
   admission verdict (absolute metres) is still real. ⭐ **Fabricating `sel` was correctly refused** —
   candidate 0 is arbitrary and argmin is the **oracle**; either manufactures the denominator. *A
   missing denominator is a fact to report, not a gap to fill.*
2. ⚠️ **The live run's `--max-horizon 60` must NOT be inherited by the dump grid** — it would
   re-select windows and **break parity**. The grid stays canonical (WINDOW=8 / K_MAX=20).
3. ⛔ **The script is NOT on Thor** (MEASURED: `dumper_shipped_to_thor: false`). It needs an
   md5-verified **file-ship**, never a `git fetch` — on a pod that hangs, and a failed fetch
   followed by a checkout **resets the tree**.

⭐ **VISION-ONLY IS MEASURED HERE, NOT ASSERTED** — and the control has teeth. The producer re-runs a
batch with `v0` **and** `actions` permuted and requires the latent blocks **bit-identical**, and a
test proves the control **reports failure on a `v0`-reading stand-in** rather than passing
vacuously. That is the standard C92 exists to enforce, applied prospectively for once instead of
retrospectively.

---

## C99 — I HAND-LISTED A SHIP SET, VERIFIED EVERY md5, AND THE CODE STILL COULD NOT RUN (2026-08-18, dumper ship to Thor)

**MEASURED.** I shipped the three files I had changed — `v6_dump_sw_latents.py`, `v6_chain.py`,
`e_wc2_sigma_star.py` — to Thor and verified **all three md5s byte-identical on both sides**
(`4cdc09c2…` / `771b337e…` / `8132c767…`, sizes matching). Then the real-import probe returned:

```
ImportError: cannot import name 'K_MAX_GRID' from 'refc_dump_latents'
```

Thor's `refc_dump_latents.py` was **11,629 bytes with ZERO occurrences of `K_MAX_GRID`**; the repo's
is **30,089 bytes with it at line 96** — a **2.6× smaller, badly stale** dependency that I never
listed because **I had not changed it.**

⇒ **ROOT-CAUSE CLASS: A SHIP SET DERIVED FROM "WHAT I EDITED" INSTEAD OF FROM THE IMPORT CLOSURE.**
⛔ **And this lesson was already in the log, from the S-T sync EARLIER THE SAME NIGHT**, in that
agent's own words: *"the closure was derived by IMPORTING it, not by listing: 76 files, not 13. A
hand-listed dependency set is a guess about what a launch touches."* **I then hand-listed three.**

⇒ **RULE: the ship set is the IMPORT CLOSURE, computed, never the diff.** A file you did not edit can
still be the one that is stale on the target — staleness is a property of the *target*, not of your
*changes*.

⚠️ **AND THE SHARPER HALF: md5 AGREEMENT PROVES TRANSFER, NOT FUNCTION.** All three files were
present and byte-perfect, which is exactly what a "sync verified" report would have claimed. **A
green checksum on the wrong file set is a confident wrong answer** — the same shape as the `df`,
`free`, and cgroup traps: a probe that answers a *different question* than the one asked.
⇒ **Only a real import (or a real forward) is evidence that a pod can run the code.** Presence,
md5, and `git log` are all necessary and none is sufficient.

⭐ **WHAT WORKED, AND IT IS WORTH KEEPING:** the failure surfaced in **2 seconds** because the dumper
carries a **preflight import probe at startup** — required in its brief precisely because an
analysis-time import that fails *after* the rollout once destroyed a complete 2-arm, 40-episode run
whose compute was already paid for. **The gate an agent built caught the orchestrator's mistake.**

**Resolution:** Thor's stale copy backed up to `/home/nvidia/_thor_backup_2026-08-18-dumper/`
(md5 `a2c52bf5…`), the current file shipped (`e6203a4a…`, 30,089 B, `K_MAX_GRID` ×4), and
re-verified: `[swdump] preflight OK`, and `v6_chain.py admission` now emits **zero** `NOT BUILT`
lines on Thor. **The live run was unaffected — step 12,700 → 12,800 across the operation with
`step_s` unchanged at 26.474.**

---

## C100 — OF 87 BANKED SEPARATED-FAILs, **ONE** IS A SUBSTANTIVE FINDING (2026-08-18, K1 degeneracy guard + re-read)

**RETRACTED:** the standing body of K1 FAIL verdicts from the linear-readout ladder. Re-read across
**165 rows** with the C92 intercept repair **and** the new C97 degeneracy guard:

| of 87 banked separated-FAILs | n |
|---|---|
| **die at the repair** (C92 — the biased floor made them fail) | **23** |
| **killed by the guard** (C97 — degenerate, constant-offset) | **42** |
| **flip to PASS** | **11** |
| survive both | **11** |

⛔ **And 10 of the 11 survivors are `ego_yawrate` at K1B +0.0000 [+0.0000, +0.0000] — two of them on
RANDOM-LATENT NULLS.** ⇒ ⭐ **Exactly ONE substantive finding remains: `ll_s09000 lead_gap`, K1B
+0.748 [+0.002, +1.624].** The `@11250` FAIL that the previous stream reported as *surviving* is
**DOWNGRADED** — K1B **+0.404, not separated.**

⇒ **ROOT-CAUSE CLASS: A BODY OF "FINDINGS" THAT WAS OVERWHELMINGLY INSTRUMENT.** Two defects in
opposite directions (C92's biased floor, C97's degenerate ceiling) between them account for **65 of
87**. ⚠️ **`LATENT_LINEAR_LADDER.md` must be re-read before ANY of it is quoted — 93 of its 165 rows
change.**

⭐ **THE GUARD HAS NO FREE PARAMETER, WHICH IS WHY IT SETTLES THINGS.** With `c_own = mean(pred)`:

> **K1 = [MAE(pred) − MAE(c_own)] + [MAE(c_own) − MAE(C-CONST)] = K1B + K1C**

**K1B is the latent-attributable part and is ALGEBRAICALLY INVARIANT to the choice of C-CONST**
(pinned); K1C is a pure which-constant contest. And **`|K1B| ≤ pred_mad ≤ pred_sd` is a THEOREM**
(reverse triangle inequality, then Jensen) — so `pred_sd < |K1|` **proves** a constant-offset
component **with no refit and no bootstrap, from fields already banked.**

⇒ **THE mean-vs-median QUESTION DISSOLVES RATHER THAN BEING DECIDED, and the median STAYS.** MAE's
optimal constant **is** the median, so switching would *weaken* the baseline and **manufacture
PASSes** — C97's own failure mode a third time. The gap is now **reported**
(`mean_minus_median_const_gap`) instead of being an invisible route to a PASS. **No default flips.**

⚠️ **THE AGENT'S OWN FIRST DRAFT WAS THE REJECTS-EVERYTHING KIND**, and it says so: letting
`sd_ratio` short-circuit K1B would have **rejected a genuine readout on a heavy-tailed target**
(`n_agents_all`: gt_sd 46.5, median 34). ⇒ **C95's rule fired correctly for once, prospectively:
both directions were pinned (29 tests) before the guard shipped** — a degenerate arm is caught, and
a strong arm, a **weak-but-genuine** arm, and a signal-strength sweep are not.

⛔ **THE TRIVIAL-PROXY CONTROL THEN KILLED THE GUARD'S OWN BEST NEW RESULT — exactly as it should.**
Under the repair `n_agents_all` PASSes on **all 15 arms**, and the guard rejects **precisely the two
containing zero information** (SNR sweep monotone in injected noise, −2.219 → −1.044). **But the v6
arm (K1B −2.785) beats the SINGLE EGO-SPEED SCALAR (−2.243) by 0.012 gt_sd**, and on
`n_agents_grid` **the scalar PASSES while the 2048-dim latent does not separate.**
⇒ ⛔ **"The v6 latent reads scene density" is NOT SUPPORTED — it is ~80 % `v0`.** *(C92's discipline
applied prospectively rather than retrospectively, for the first time.)*

⚠️ **A SEPARATE STALENESS, SAME CLASS AS C99 AND FOUND THE SAME NIGHT:** the **scratch** copy of
`pc6_linear_readout.py` was **pre-C92 with no `intercept_col` at all** — **a launch from it would
have silently re-run the defect the repair removed.** Staleness is a property of the *target*; the
repo being correct proves nothing about the copy the job actually loads.

⚠️ **AND ONE MEASURED-AND-RETRACTED IN THE SAME PACKAGE:** the claim that `ll1`'s two repair routes
agree to *"~1e-12"* on the inner split. The **full fit** agrees to 5e-14; the **inner split** differs
by up to **0.74 MAE — eleven orders out** — enough to flip near-tied alpha choices (K1 **+0.4274 →
+0.0317**). ⇒ **The two routes' numbers must never be pooled.** *An agreement measured on one path
and asserted for another is the scope error again.*

---

## C101 — `planner_beats_cv` IS AN **OPEN-LOOP** VERDICT, THE ZERO-GPU PATH CANNOT REACH IT — AND THE QUESTION IT STOOD FOR IS NOW SETTLED AGAINST THE PLANNER (2026-08-18)

**RETRACTED — my own framing in the re-drive brief.** I wrote that `planner_beats_cv` is the
closed-loop planner-vs-CV comparison and that the banked `p2win_flagship-30k.pt` should therefore
settle it with zero GPU.

**MEASURED from source:** `planner_beats_cv` is computed inside **`analyze_openloop`**
(`planner_p2.py:621`, fn at `:555`) from `collect_openloop`'s `plan_wp`/`cv_wp` — **OPEN LOOP,
n = 881 windows / 40 episodes, stride 8.** The banked dump is the **CLOSED-loop** collection
(**221 windows / 20 episodes, stride 16**). ⇒ ⛔ **Different tier, different windows, different
episode count. The banked path does not reach this verdict.** Absence confirmed at **three probes,
including an exhaustive walk of every `.pt` in the repo: no open-loop CEM planner arm is banked
anywhere** — independently confirming `JACK_IN_GATES.md` §3.1.

⇒ **ROOT-CAUSE CLASS: I ASSUMED A VERDICT'S TIER FROM ITS NAME.** *"planner beats CV"* sounds like
one comparison and is another. **This is C95 again** — two same-named quantities at two eval tiers —
and it is the reason the tier stamp is binding. ⚠️ **The `--analyze-only` instinct was still right as
discipline** (it once recovered a complete 2-arm run for free); it was the *premise* that was wrong,
and **the agent checked rather than assumed**, which is the only reason this cost nothing.

⇒ **THE OPEN-LOOP VERDICT REMAINS UNDECIDED, AND NO ANSWER WAS MANUFACTURED.** Reproduction gate
passes — `cv` and `open_grnd` **bit-exact**, `closed_bike` drift **0.0193 %** (⭐ *the documented
unseeded-CEM residual — an independent confirmation that the `cem_seed` fix was addressing a real
effect*). Corrected arms reproduce `JACK_IN_GATES` to 4 dp: CV **0.8377**, operative **0.4271**,
head **3.3839**. Flipping needs the planner to drop **6.589 %** against a local envelope of
**−6.909 %…+5.877 %** and programme-wide **+11.69 %** — above the local edge, inside the
programme-wide one. ⚠️ **No bound closes it**: the banned estimator gives **7 of 40 episodes weight
exactly 0**, so those windows are unconstrained by the published mean.

### ⭐ BUT THE QUESTION IT STOOD FOR IS NOW ANSWERED, AT THE PRIMARY TIER

The published G4 compared planner vs **head**, never vs **CV** — so this had **never actually been
asked**. Computed from banked data, **paired**, **T1**:

> **planner − CV = +0.2585 m [+0.0869, +0.4309], CI-SEPARATED, p(δ>0) = 0.9975**
> ⇒ **the CEM planner is 35.8 % WORSE than constant velocity, closed-loop.**

⭐⭐ **AND IT LOCALISES THE DEFECT: operative-under-true-actions BEATS CV (−0.3151 m).**
⇒ **THE LOSS IS IN THE ACTION SEARCH, NOT IN THE WORLD MODEL.** The WM rolls out well when handed
true actions; the CEM cannot find actions that exploit it.

⛔ **AND THE PLANNER LOSES ON THE FAMILY IT IS DESIGNED FOR.** Per family, never pooled:
**LONGITUDINAL 1.9062 vs 1.6705 m**, speed error **0.9431 vs 0.7607 m/s**, bias **+0.2737 vs
−0.0995 m/s**. The **lateral** loss its own scope note predicts; **the longitudinal loss it does
not.** *(This is exactly why ADE-only reporting is banned: a scalar would have shown "worse" without
showing that it is worse **where its cost function is aimed**.)*
**TACTICAL** and **STRATEGIC** are genuine **N/A with reasons and n** — the CEM emits no manoeuvre
class, and the P2 cost carries **no route/goal term**. Distance-keeping/TTC is uncomputable: no
lead-agent track.

⚠️ **A real data defect caught in passing:** curvature on **11 stopped-ego windows** gave GT `|κ|`
**mean 34.83 1/m against a median 0.00081** (max **23,004**). **Masked, not clamped**, with the
excluded n published — a clamp would have silently kept fabricated geometry in the mean.

### ⚠️ C91's OWN COUNT WAS IMPRECISE — the class it exists to name, applied to itself

C91 says the artifact holds **five** verdicts. Enumerated properly: **14 boolean instances across 6
distinct names** — C91 **collapsed the 9 `beats_head` grid entries**. The verdict *"enumerate every
field in the artifact"* stands; **the number in it was itself taken one level too coarse.**
*(`beats_head` is settled by a **4.686× margin**, ~30× any measured estimator error, so its
unrecoverable estimator is irrelevant. **No other verdict moves.**)*

### ⏳ BLOCKED ON THE PI, AND CORRECTLY SO

The open-loop re-drive is feasible here — checkpoint **local** (`flagship4b-speedjerk-30k_ckpt.pt`,
step 29999), harness imports, `cem_seed=0` present, the 4060 free, Thor untouched. The **only**
missing input is the val cache: **HF `epcache-256px-phase0/physicalai-val-0c5f7dac3b11/`, 40
episodes, 4.70 GB.** ⭐ **The agent did NOT download it, on the grounds that an agent brief is not
user consent — which is right, and is the standard.** One job then closes `planner_beats_cv` **and**
G1's fourth arm; parity is provable afterwards by bit-exact reproduction of the model-free `cv`.

---

## C102 — "THOR IS SYNCED" WAS TRUE OF THE THREE FILES C99 NAMED AND FALSE OF THE LAUNCH (2026-08-18, closure audit)

**RETRACTED — the resolution line of C99 itself.** C99 closed with Thor's stale `refc_dump_latents.py`
replaced, the dumper's preflight green, and `v6_chain.py admission` emitting zero `NOT BUILT` lines.
That evidence was real. **The implied claim — that Thor's `stack/` was now in step with the repo —
was not measured, and is false.**

**MEASURED (`…/incoming/2026-08-18-thor-closure-audit/`, `raw/*.json`).** Import closure computed by
AST from the v6 entry points: **120 files** where C99 shipped 3. Against Thor:
**3 DRIFT + 1 absent**, including **`train_v6_staged.py` itself** (234,845 B / 4,076 ln vs the repo's
252,691 B / 4,374 ln, missing `GATE_APPLICABILITY`, `probe_applies`, `arm_record`,
`SEL_GAP_TIER_NOTE`, `UNMEASURED_BY_CONSTRUCTION` — precisely the five S-T launch-path fixes), and
`tanitad/data/anchor_goal.py` **not present at all**.

⇒ **ROOT-CAUSE CLASS: THE SAME ONE C99 NAMED, APPLIED ONE LEVEL UP — I FIXED THE HAND-LISTED SHIP SET
AND KEPT A HAND-LISTED *ENTRY-POINT* SET.** Widening the entry points from the 7 of the ladder to 14
(adding `t1_eval.py`, `eval_four_families.py`, `seam_probe.py`, `t1_summary.py`, `run_spectral.py`,
`refc_dump_latents.py`, `v5_guard.py`) grew the closure to **134** and found **3 MORE stale files** —
`four_families.py` (missing `kappa_verdict`, `_anti_echo`), `hierarchy.py` (missing `PER_WINDOW_KEYS`),
and **`v0_antiecho.py`, 46,905 B, absent from Thor entirely**. **7 stale files in total.**
**A hand-listed set of entry points is a guess in exactly the way a hand-listed ship set is.**

⚠️ **AND THE MEASUREMENT ALMOST LIED THREE TIMES, EACH TIME CLEANLY AND PLAUSIBLY:**
1. **MSYS argument mangling** rewrote `--remote-root /home/nvidia/TanitAD` into
   `C:/Program Files/Git/home/nvidia/TanitAD`, and the audit reported **120/120 MISSING_REMOTE**
   while the trainer was executing a file from that very path. ⇒ `MSYS_NO_PATHCONV=1`, plus a
   demangler in the tool so it cannot recur silently.
2. **A `ps` filter containing the words `supervise` and `train_v6_staged` matched my own ssh command
   line**, inventing a supervisor and a second trainer — the PTY-echo trap in a `ps` costume.
   ⇒ tokens assembled from `chr()` codes; corrected result **0 supervisors, 1 trainer**.
3. **`pytest -q` reported 4 failures** in files I never touched; all four are
   `UnicodeDecodeError: 'charmap'` reading a subprocess's `⛔ ⚠️ ⭐` output under **cp1252**, and all
   four pass under `PYTHONUTF8=1`. ⇒ **a "green suite" claim on this dev box is meaningless without
   the encoding stated.**

⭐ **THE ONE THAT WOULD HAVE BEEN A FABRICATED FINDING:** the real-import probe reported
`lead_state_gate: No module named 'pandas'` as a blocker. **It is not** — its only closure import
site, `probe_latent_state.py:117-124`, is a `try/except ImportError` with documented fallback
constants. A direct `import_module()` **bypasses a guard the launch relies on**. The tool now detects
`try/except ImportError` and classifies failures BLOCKING vs GUARDED, so an audit whose entire
subject is fabricated findings cannot produce one.

**Resolution:** all 7 shipped LF-normalised, Thor's originals backed up to
`/home/nvidia/_thor_backup_2026-08-18-closure/`; final state **DRIFT 0 / MISSING_REMOTE 0** and
**131 of 134 modules import for real** on Thor's train venv — the 3 remaining are absent
`pandas`/`pyarrow`, one guarded and two on data paths the ladder does not take (nothing installed:
`uv pip install` has twice replaced torch with a wheel the driver cannot run). The live v6F S-W run
advanced **12,750 → 12,800 → 12,850** with `step_s` **26.4735 → 26.4745** throughout.
**The instrument is `stack/scripts/launch_closure_audit.py` with 19 tests in
`stack/tests/test_launch_closure_audit.py` — the next launch runs it instead of hand-listing.**

⭐ **AND ONE TURN SHARPER THAN C99.** C99's lesson was *md5 agreement proves transfer, not function*
— it failed loudly, in 2 seconds, on an `ImportError`. **Row 1 would not have failed at all.**
MEASURED by AST over Thor's own backed-up original: the stale `train_v6_staged.py` **still exported
`assert_stage_precondition`** (45 top-level functions vs 47), so `v6_chain.py`'s import of *"THE
adjudicator"* would have **SUCCEEDED** — while containing **0 references to `probe_applies`** against
the current file's **6**. The S-T gate would have run with **no applicability filtering**, scored arms
against criteria `GATE_APPLICABILITY` marks not applicable, and emitted a normal-looking verdict.
⇒ **A SUCCESSFUL IMPORT IS NOT EVIDENCE OF CURRENCY, whenever the stale file still exports the names
the caller asks for.** Presence, md5, `git log` and *a green import* are each necessary and none is
sufficient; only a content comparison over the COMPUTED closure catches this.

---

## C103 — C100's SURVIVING v6 PASS REVERSES AT 3 SEEDS, AND THE "ZERO SEED SPREAD" THAT MADE ONE SEED LOOK SUFFICIENT WAS ITSELF A PRODUCT OF THE DEFECT (2026-08-18)

**RETRACTED:** C100's one surviving v6 PASS. On `n_agents_all`, **seed 0** had the latent beating the
ego-speed scalar by **0.012 gt_sd**. Across **three seeds the SCALAR WINS** — K1B **−1.689 vs
−1.477**. ⇒ ⛔ **On 10 of 11 rungs the SINGLE EGO-SPEED SCALAR matches or beats the 2 048-dim latent
on the 3-seed mean;** the 11th favours the latent by **0.00002 gt_sd** on a degenerate rung.

⇒ ⭐ **THE MECHANISM IS THE REPAIR ITSELF, AND IT IS THE GENERALISABLE PART.** The C92 intercept
defect had **frozen the alpha sweep** — a shrunk-to-zero fit made the inner-split MAE insensitive, so
alpha selection barely moved. **Repairing it UN-TRUNCATES the sweep**, and the arm's own K1B now
moves **2.516 across seeds**.
⇒ ⛔ **This FALSIFIES the ladder's own load-bearing methodological claim — *"seed spread is exactly
zero on 8 of 11 rungs, so ≥3 seeds supply no uncertainty here"*.** That claim was **measured under
the defect**, and it is precisely what licensed reading a single seed.

⇒ **ROOT-CAUSE CLASS: A STABILITY CLAIM MEASURED UNDER A DEFECT IS NOT INHERITED BY THE REPAIRED
INSTRUMENT.** *"This quantity doesn't vary, so one seed is enough"* is a property of the **instrument
as it was**, not of the question. **Every variance/stability/convergence claim taken before a repair
must be re-measured after it** — a repair changes the estimator's sensitivity, not only its bias.
⚠️ **Practical consequence: C100's 165-row inventory rests on seed 0 and MUST be re-run at 3 seeds**
(one flag, ~45 min CPU, zero GPU). Its *direction* is probably safe — 65 of 87 died to two
independent mechanisms — but **no individual row is quotable until then**, and at least one row is
known to reverse.

### What survives, and it is the part that matters

⭐ **THE HEADLINE SURVIVES AND STRENGTHENS.** Repaired, the latent's ego-speed readout **TIES a
constant** (K1 **+0.032 [−0.532, +0.508]**, not separated) while the **ego-oracle at 10× noise earns
a guarded PASS** (K1 **−1.604**, r **+0.828**). ⇒ The comparison is now **PASS-vs-TIE**, not
FAIL-vs-FAIL — a cleaner statement of the same finding.
⭐ **And lead gap INVERTS rather than shrinking:** the latent is **0.694 m WORSE than the
random-latent null** (5.869 vs 5.175), and partialling `v0` out flips its correlation to **−0.107**.

### Route non-pooling, now MEASURED rather than argued

44 paired rows: **2 alpha choices differ, 0 verdicts differ** — but `ego_v0`'s K1 differs by
**0.3957** (A +0.0317 / B +0.4274) and its **K1B by a factor of 8**. ⇒ The verdicts are robust to the
route; **the numbers are not**. Pooling them remains banned.
*(The refits used route B deliberately, because §7's table was rendered from it — swapping routes
mid-document would have been the pooling error itself.)*

### ⚠️ Stale citations created BY this in-place rewrite — the C90 hazard, inverted

Correcting a document in place **invalidates every line-number citation into it**.
⛔ **`POOLING_BOTTLENECK_R1R2.md` §1.5 and `2026-08-17-O234-DESIGN-RESEARCH.md` §3.4a both now quote
stale ladder numbers** — all three r² values moved, the `n_agents_all` half is killed by the
trivial-proxy control, and **"relative motion exactly 0.0000" is now 0.0013** *(a figure I repeated
in C93 and in three commit messages)*. `MODEL_REGISTRY.md` was checked and quotes no ladder number.
⇒ **RULE: an in-place correction must be followed by a citation sweep of the documents that cite it.**
Fixing the source and orphaning its citations trades one stale claim for several.

### On the pooling thesis, stated carefully

The rung **profile** is **not** among the 65 that died — it is an r²-vs-null statement and the
**ordering held** (same top three, same bottom four). What died is the **K1-FAIL side**: all four
confirming rungs are now `DEGENERATE-CONSTANT` or a verdict the random null earns identically.
⚠️ **But the profile never localised the loss**, and §11's localisation **INVERTS** (cells **+0.320** >
tokens **+0.263**). ⇒ **R1/R2 must be decided on E-R1-0's own evidence, not on this ladder.**
*(⚠️ A report that E-R1-0 "has since dropped R1" is **INHERITED** from this agent and **NOT verified
here** — that stream has not yet reported to the orchestrator. Do not act on it until it does.)*

---

## C104 — ⛔ THE 40:1 POOLING BOTTLENECK IS **REFUTED**. THE INFORMATION SURVIVES THE POOL; THE ENCODER IS THE CONSTRAINT (2026-08-18, E-R1-0)

**RETRACTED — the load-bearing claim of C93, which I called *"the finding that subsumes all
three"*:** that `AvgPool2d((4,10))` averages agents away before any objective can see them, and that
this explains D1 without appeal to the world model's competence.

**MEASURED, pre-registered, on frozen `v6F-SW-30k@11250`** — 1,302 train / **1,507 eval windows in
70 episode clusters**, four pooling ratios (40:1 deployed / 10:1 / 4:1 / 1:1) differing **only in the
kernel**, each forced to exactly 2,048 features by a **fixed random projection**, **5 seeds**,
`intercept_col=-1` throughout:

> ⛔ **On the four rungs the hypothesis was built to explain, REMOVING THE POOL ENTIRELY moves r² by
> |Δ| ≤ 0.0002, with the CI containing zero on all five seeds.**
> (`lead_closing` Δ = **+0.00001 [−0.00597, +0.00504]**.)

⇒ **`R1 IS DROPPED` by its own pre-registered criterion.** No rung meets `R1 PROCEEDS`.
*(`lead_gap` rises 15× — 0.0050→0.0750 — but collapses to 0.0060→0.0087 once `v0` is partialled out;
`ego_v0` rises 2.4× and separates on only 2 of 5 seeds.)*

⭐ **THREE CONTROLS MAKE THE NEGATIVE ADMISSIBLE RATHER THAN CONVENIENT** — this is what a negative
result has to carry, and what D1 lacked when it was withdrawn (C79):
1. **POSITIVE CONTROL `PC-2OBJ`** — two *opposing* plants inside one deployed cell — is a clean step
   **0.0000 → 0.9998.** ⇒ **The ladder demonstrably HAS the power to see a pooling-destroyed
   signal.** It saw nothing because there was nothing.
2. **NO-PROJECTION SUPPLEMENT** — exact dual ridge on **all 491,520 features** (gated equal to the
   primal solve to 1.4e-06 of pred sd): `lead_closing` **0.00058 → 0.00045 → 0.00022 → 0.00002**.
   ⇒ It **FALLS** as the pool is removed. **The random projection was hiding nothing.**
3. **TRIVIAL-PROXY `C-V0`** — ego speed *alone* beats **all four** arms on `lead_gap` (0.467 vs
   0.075), `nearest_any`, and `n_agents_grid`.

*(Matched-random null floors at r² ≈ 0.001–0.002; **three of the four rungs sit BELOW their own
null.**)*

### ⭐⭐ AND THE DISCRIMINATOR RETURNED THE LARGER RESULT

DINOv3 is **gated `manual` and our token gets 403** (three probes; metadata public, weights not).
Substituted **`facebook/dinov2-base` at 224×560**, which at patch 14 tiles into **exactly our 16×40
grid at an identical 0.4000 aspect** (asserted at runtime).

**Through the SAME deployed `AvgPool2d((4,10))`, on the SAME windows:**

| rung | DINOv2-B/14 | **ours** |
|---|---|---|
| `lead_closing` r² | **0.01713** (partial-`v0` r +0.129) | **0.00000** |
| `lead_gap` r² | **0.44997** (partial r² 0.120) | **0.00496** |
| `ego_v0` r² | **0.71733** | **0.05240** |

⛔ **And it is NOT a capacity gap: DINOv2-B/14 is 86 M params against our encoder's 87.3 M.**

⇒ ⭐ **THE INFORMATION IS IN OUR IMAGES, IT SURVIVES THE POOL, AND NEITHER POOLING NOR CORPUS
NARROWNESS IS THE CONSTRAINT — THE ENCODER/OBJECTIVE IS.** Removing the pool on an encoder that
*does* carry the signal is worth **+14 % / +30 %**; **the encoder gap is 91×.**
⇒ And DINOv2's `lead_present` **satisfies §7.1's PROCEED criterion in full** — ⭐ **proving the rule
is satisfiable and that ours simply fails it**, which is the difference between a hard problem and a
broken component.

⇒ **ROOT-CAUSE CLASS: A MECHANISM INFERRED FROM AN ARCHITECTURAL DIAGRAM AND A PLAUSIBLE ARITHMETIC
COINCIDENCE, NEVER TESTED BY REMOVING IT.** *"40 tokens per cell, 64×160 px per cell, 37.8 px median
vehicle"* is a compelling story and it is **wrong**. ⚠️ **Two independent methods appearing to agree
did not save it** — the ladder's rung profile was cited as confirmation, but a profile that is
*consistent with* a mechanism is not evidence *for* it when both are also consistent with a weak
encoder. ⇒ **RULE: an architectural bottleneck claim is not established until the bottleneck is
REMOVED and the metric moves.** Ablate the mechanism, don't narrate it.

### Two corrections that travel with this

⛔ **The "68.5 % of the stack sits downstream of the pool" figure — which I repeated — is a SCOPE
ERROR.** It was computed from `V6Config()` **defaults**; the **live checkpoint** is `d_model 768`,
**336,559,305 params** ⇒ `predictor_op` **55.9 %** (not 68.5), `encoder` **25.9 %** (not 17.4).
**The total was understated 3.8×.** *(The `df`/`free` family again: a real number from the wrong
scope.)*

⚠️ **The live S-W run is 336.6 M — 12.2 % over the programme's "Sub-300M" headline.** **Not a silent
breach** (launched with `param_budget: 350000000`, so the assert passed by design), but it
**contradicts `CLAUDE.md`'s opening line.** ⏳ **PI decision: restate the claim or rescope the model.**

⚠️ **The C97 guard has a hole, found by its first external user:** `SD_RATIO_FLAT_FLOOR = 0.05`
**does not flag C97's own headline case** (null `n_agents_all`, ratio **0.091**). ⇒ Our
`n_agents_all` K1 PASS is **not quotable as latent-attributable** until layer 2 runs.

### ⇒ WHAT THIS CHANGES

**E-R1-1 is dropped and E-R2-0 promoted** (the 2×2's *placement* axis is now measured inert, leaving
the *target* axis — R2-cells, 16,899 params). ⭐ **But ENCODER experiments now outrank both**: a
frozen-external-encoder readout arm, and a **DINOv2 token-distillation `aux` loss** — both cheaper
than R1, and aimed at the component the evidence actually indicts.
⏳ **A human must accept the DINOv3 licence** for the stronger arm; the agent correctly did not, and
did not use a mirror.

---

## C105 — C99 FIXED THE HAND-LISTED **SHIP SET** AND KEPT A HAND-LISTED **ENTRY-POINT SET**; AND A GREEN IMPORT IS NOT EVIDENCE OF CURRENCY (2026-08-18, Thor closure audit)

**MEASURED.** C99's lesson was *"the ship set is the import closure, computed, never the diff"* — and
I then briefed the audit with a **hand-listed set of 7 entry points**. The computed AST closure of
the ladder is **120 files**; widening the entry-point list to **14** (adding `t1_eval.py`,
`eval_four_families.py`, `seam_probe.py`, `t1_summary.py`, `run_spectral.py`,
`refc_dump_latents.py`, `v5_guard.py`) grew it to **134 and found three MORE stale files.**

⇒ ⛔ **SAME CLASS, ONE LEVEL UP.** C99 computed the closure but hand-listed its **roots**. A closure
is only as complete as the set it is closed *over*. ⇒ **RULE: derive the entry points too — from
what the runbook and chain actually invoke — and state the root set, because it is the assumption
the whole audit rests on.**

**7 files were genuinely stale** — 5 drifted, **2 absent entirely**, including `taniteval/v0_antiecho.py`
at **46,905 B**.

### ⭐⭐ THE MOST DANGEROUS ONE WOULD NOT HAVE FAILED LOUDLY

Thor's `train_v6_staged.py` (**234,845 B vs 252,691 B**) **still exported `assert_stage_precondition`**
— so `v6_chain.py`'s import of *"THE adjudicator"* would have **SUCCEEDED** — while containing
**0 references to `probe_applies`** against the current **6**.
⇒ ⛔ **The S-T gate would have run with NO applicability filtering and emitted a normal-looking
verdict.**

⇒ **This is the third rung of one ladder, and each rung looked sufficient at the time:**
| check | proves | does NOT prove |
|---|---|---|
| file **present** | it was transferred | that it is the right bytes |
| **md5** matches (C99) | the bytes are right | that the code can run |
| **import** succeeds | the module loads | ⛔ **that it is CURRENT** |
⇒ **RULE: a stale file that still exports the names its caller asks for passes every structural
check and is wrong. Verify SYMBOL-LEVEL CURRENCY — the presence of what the caller will actually
use — not merely that the import returns.**

### ⚠️ CRLF QUANTIFIED, AND IT IS WORSE ON BIGGER SETS

**47 of 50 apparent-drift rows were artifacts — 94 % FALSE**, against **70 %** measured on 08-16.
**The false-positive rate RISES with the size of the compared set.** Verified arithmetically:
`rollout.py`'s 373-byte raw delta equals its **373 lines** exactly. ⇒ Normalise before comparing,
always, and report artifacts as their own category.

### ⚠️ THREE PROBES RETURNED CLEAN, PLAUSIBLE, WRONG ANSWERS — all in one audit

1. **MSYS rewrote `--remote-root`** into `C:/Program Files/Git/home/nvidia/TanitAD` ⇒ the audit
   reported **120/120 MISSING_REMOTE** *while the trainer was executing a file from that path.*
2. **A `ps` filter containing `supervise` and `train_v6_staged` matched its own ssh command line**,
   inventing a supervisor and a second trainer. Re-probed with `chr()`-assembled tokens: **0
   supervisors, 1 trainer.** *(The self-match trap, for the fourth measured time.)*
3. **`pytest` reported 4 failures** in untouched files — all `UnicodeDecodeError: 'charmap'` under
   cp1252, **all passing under `PYTHONUTF8=1`.** *(C82/C86's mechanism, confirmed again.)*

⭐ **And a fourth, caught before it was filed:** `lead_state_gate` appeared to be a blocking import
failure, but is **guarded by `try/except ImportError` with documented fallbacks**
(`probe_latent_state.py:117-124`) — the probe had bypassed the guard. The tool now classifies
**BLOCKING vs GUARDED** rather than counting raw failures.

### FINAL STATE

**DRIFT 0 · MISSING_REMOTE 0 · 131/134 modules import for real** on Thor's train venv, cross-checked
against an **independent digest** (Thor's `sha256sum` binary vs local `hashlib`): **7 match, 0
differ.** The 3 remaining failures are environment, not drift, and none blocks S-T.
**Live run untouched: 12,750 → 12,900, `step_s` 26.4735 / 26.4742 / 26.4745 / 26.4742.**

### ⛔ AND ONE DEFECT THAT WOULD HAVE KILLED S-T — FOUND, FIXED, PINNED

`train_v6_staged.py` imported `SeamDumpError` **inside the `try` whose `except` names it.** If that
import fails, Python evaluates `except SeamDumpError`, finds it unbound, and **the resulting error
propagates OUT OF THE WHOLE `try` STATEMENT — the broad `except Exception` below is never reached.**
**Proven in 6 lines:** it escapes as `UnboundLocalError: cannot access local variable`.
⇒ The block sits **immediately before `_save_ckpt`**, so it **kills the trainer at a checkpoint
boundary**, and it fires exactly when `taniteval` is off `PYTHONPATH` — **the live run's own
configuration.** S-W escaped only because the chain emits `--dump-seam-plan` on **S-T/S-S/S-J and not
S-W**. ⇒ **S-T would have hit it.** Repaired (guarded import + sentinel) and pinned by **7 tests**,
the first of which reproduces the language rule directly so it fails against the old shape.

⚠️ **The test's own locator was wrong THREE TIMES before it was right** — first matching the
enclosing checkpoint block, then the inner sentinel check, then an unrelated guard clause in another
function — **each returning something plausible instead of failing.** ⇒ **A locator must be pinned by
a conjunction only the intended node satisfies, and must ASSERT UNIQUENESS rather than take a first
match.** *The scope error, three times, inside a helper written to catch a scope error.*

### Two standing items

⛔ **`stack/scripts/pod_git_drift.py` is stale doctrine** — all four `DEFAULT_PODS` are dead
machines, it matches by **basename repo-wide**, and it does **no CRLF normalisation**, so it would
report drift on roughly half of what it inspects. ⏳ **`pandas`/`pyarrow` in Thor's `tanitad-train`
is a PI call**; the safe recipe (`--no-deps`, torch last from the pinned index, verified with a real
`conv2d`) is in the audit report. **The agent installed nothing** — correct, beside a 5-day run.

===============================================================================

## C106 — ⛔ C104's PHRASING IS TOO WEAK AND MIS-PLACED: OUR ENCODER'S **RANDOM INITIALISATION** BEATS ITS **TRAINED** SELF 3.6× ON C104's OWN METRIC — THE OBJECTIVE IS NOT FAILING TO ADD, IT IS SUBTRACTING (2026-08-18, encoder-experiments falsifier battery)

**AMENDED — C104's conclusion, *"the encoder/objective is the constraint"*.** The direction was
right and the statement was **too weak, and located one level too high**.

**MEASURED, pre-registered, zero training**, on the **same** frozen `v6F-SW-30k@11250` windows,
the **same** rows, the **same** deployed `AvgPool2d((4,10))`, the **same** command,
`--proj-seeds 0 1 2`, `intercept_col=-1`, 1 302 train / 1 507 eval windows in **70 episode
clusters** — `…/Research/2026-08-18-encoder-experiments/raw/falsifier_summary.json`:

| arm | width | raw feat | `ego_v0` r² | `lead_gap` r² | K1 PASS |
|---|---:|---:|---:|---:|---:|
| **ours — trained S-W @11 250** | 768 | 12 288 | 0.05207 | 0.00490 | **0/3** |
| `dino1f` — DINOv2 **pretrained**, ONE sub-frame | **768** | **12 288** | **0.70593** | **0.42743** | **3/3** |
| `dinorand` — DINOv2 arch, **untrained**, 3 frames | 2304 | 36 864 | 0.02660 | 0.01843 | 0/3 |
| **`randenc` — OUR arch, RANDOM INIT (3 seeds)** | 768 | 12 288 | **0.1894** [0.1736, 0.2011] | **0.0176** [0.0160, 0.0203] | 0/9 |

### ⭐ THREE THINGS THIS SETTLES, EACH BY REMOVING ONE VARIABLE

**1. C104's headline was a 4-variable comparison; the two biggest non-encoder variables are now
MEASURED INERT.** The DINOv2 arm differed from ours in pretraining, architecture, **input
format** (3 × 3-channel sub-frames vs one 9-channel tensor) and **token width** (2304 vs 768).
At **identical width and identical feature count**, ONE sub-frame recovers **98.4 %** of the
3-frame `ego_v0`, **95.0 %** of `lead_gap` and **99.7 %** of `lead_closing`. ⇒ **the
concatenation and the 3× width explain essentially NONE of the gap.** C104 is *strengthened*.

**2. Pretraining does the work, not architecture.** Removing only the pretrained weights
collapses the teacher: `ego_v0` **27×**, `lead_gap` **24×**, `lead_closing` **46×**.

**3. ⛔⛔ AND THE ONE THAT AMENDS C104 — OUR TRAINED ENCODER RANKS BELOW ITS OWN RANDOM
INITIALISATION, ON BOTH RUNGS, ON ALL THREE SEEDS: `ego_v0` 0.05207 vs 0.1894 (3.6×),
`lead_gap` 0.00490 vs 0.0176 (3.6×).**

⭐ **The mechanism makes this the honest RAW-PIXEL control, which is what gives it force.**
`ViT5Encoder` uses LayerScale at `ls_init = 1e-5` (`stack/tanitad/models/encoder.py:303`,
`:307`, `:312`; residual `x + ls * f(x)` at `:315-316`), so at initialisation all 12 blocks
contribute ~1e-5 and the encoder is approximately `RMSNorm(patch_conv(x) + pos)` — **a fixed
random LINEAR map of raw patch pixels.** ⇒ the finding in its strongest form: **raw pixels
through a random linear map read `ego_v0` and `lead_gap` ~3.6× better than our trained
encoder's tokens.** *(It also explains `dinorand`, which would otherwise look inconsistent:
vanilla ViT has no LayerScale, so an untrained one is a genuinely deep random net, and deep
random nets destroy linear readability — 0.027, below even ours. Near-linear random map 0.19,
deep random net 0.027, pretrained 0.71: the ordering is coherent.)*

### ⚠️ THREE LIMITS, STATED BEFORE THE FINDING TRAVELS

1. ⛔ **NEITHER OF OUR ARMS PASSES K1 — only DINOv2 does** (0/3 ours, 0/9 `randenc`, 0/3
   `dinorand`, **3/3** `dino1f`). The admissible claim is *"on **r²**, the metric C104 quoted,
   our trained encoder ranks below its own random init"* — **NOT** *"the random encoder works"*.
2. ⚠️ **Our arm's readout is itself a flat line.** `pred_sd/gt_sd` **0.0141–0.0222** (vs 0.89–0.92
   for `randenc`), with the ridge choosing `alpha = 1e7` **at the grid edge**. Under the C97
   guard's own layer-3 screen (`SD_RATIO_FLAT_FLOOR = 0.05`) our arm is flagged.
3. ⚠️ **T0-DIAGNOSTIC throughout**, on the **130-episode probe corpus** (2 809 windows, 70
   clusters) — not the 40-episode val set, and never driving performance.

### The step trend: the deficit is EARLY, not accumulating

A re-encode from the three local checkpoints (**gate first**: `trained@11250` reproduces the
banked cache to Δ 0.0001 / **0.00000** / 0.00000) gives 9 250 → 10 000 → 11 250:
`ego_v0` 0.05573 → 0.05453 → 0.05197 (**falling**) while `lead_gap` 0.00470 → 0.00477 → 0.00490
(**rising**). ⛔ **INCONCLUSIVE as a trajectory and NO exponent was fit** — a 2 000-step window is
**6.7 %** of the run and the two rungs move in opposite directions. ⇒ **the 3.6× deficit was
already fully established by step 9 250**: readability is gone **early** and then stable.

⇒ **ROOT-CAUSE CLASS: A COMPARISON THAT DIFFERED IN FOUR WAYS, READ AS EVIDENCE ABOUT ONE OF
THEM — AND THE MISSING CONTROL WAS THE CHEAPEST ONE IN THE BUILDING: THE MODEL'S OWN
INITIALISATION.** C104 correctly demanded a positive control and a trivial-proxy control, and
had both. What it lacked was a **BASELINE-OF-SELF**: *what does this exact architecture read
before any training?* ⭐ **RULE: whenever an arm is compared to a foreign model, also compare it
to ITSELF AT INITIALISATION.** It costs one forward pass, it is width- and format-matched by
construction, and it is the only control that can detect a training objective making a
representation **worse** — which no external comparison can, because an external model being
better is equally consistent with ours merely being weaker.
*(Same family as C104's own lesson — "ablate the mechanism, don't narrate it" — applied one
level in: **ablate the TRAINING, not only the component.**)*

### ⇒ WHAT THIS CHANGES

**E-XENC-2's cheap path is now the WEAKER arm on mechanistic grounds, not merely the cheaper
one.** If readability is lost before step 9 250, a distillation term warm-started late repairs
after the fact instead of preventing. ⇒ the pre-registration records that a failure of the
short warm-started arm is **NOT** evidence against distillation from step 0. ⏳ The decisive
missing measurement is a **step-0-to-9 250 sweep**, blocked on checkpoints that are not local —
an artifact-retrieval item, not a compute one.

⚠️ **A param-count correction that travels with this:** C104 states the live checkpoint is
**336 559 305**. Two independent sources say **336 542 025** — the checkpoint's own
`_meta.config.param_report.total`, and a fresh `V6Stack` instantiated from that same
checkpoint's `v6_config`. **Δ = 17 280.** C104's substantive point is unaffected (336.5 M is
still 12.2 % over the "Sub-300M" headline); only the digits need fixing.

Full pre-registration, controls, kill criteria and manifest:
`TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-experiments/PREREG_ENCODER_EXPERIMENTS.md`

---

## C88 — **RESOLVED 2026-08-18.** The whole-index sweep was two correct rules colliding, and a temporary index routes around the segfault

**Appended to the C88 entry rather than opening a new class, because nothing is retracted here — a
known defect finally has a mechanism instead of a warning.**

**The diagnosis came from an agent escalating it to me as the committer, and it is right: this was
never carelessness.** Two rules the programme follows, both correctly, collide —
`AGENT_OPERATING_STANDARD.md` tells agents to **stage as they go** so nothing is stranded in one
context; `CLAUDE.md` tells committers `git commit` takes the **entire index**. **Obey both and an
agent's incremental `git add` is GUARANTEED to be swept by whatever commit lands next.** ⇒ *"Be more
careful"* could not have worked, and the mitigation we had (enumerate the foreign streams in the
message) **documents** the sweep without **preventing** it.

**MEASURED tonight:** one agent's **44-file** deliverable landed across **three** commits, **41 of
them under subjects about unrelated work.** The sweep has now fired **at least five times**.
*(Both swept commits DID enumerate their foreign streams with exact counts, so provenance is
recoverable from the bodies — the damage is archaeological, not lost work.)*

⇒ **THE FIX: `stack/scripts/scoped_commit.py`.** `GIT_INDEX_FILE` points git at a scratch index
seeded from HEAD; only the named paths are added; the commit is created through **plumbing**
(`write-tree` / `commit-tree` / `update-ref`). ⭐ **That never enters the porcelain partial-commit
code path — which is the one that SEGFAULTS on this repo** (exit 139 under MSYS git, `0xC0000005`
under native Windows git, and three separate root-cause theories for it were each falsified). **The
real `.git/index` is never opened for writing.**

⭐ **Two guards make it worth using rather than merely slower:** it **REFUSES** if the resulting tree
touches any path that was not named — without that assertion it is just a slow `git commit` — and it
**CHECKS**, rather than assumes, that the shared index kept its other entries.

⭐ **VERIFIED BY USING IT ON ITSELF**, with **55 foreign files from four live streams staged**: the
commit changed **exactly one path**, and the shared index went **63 → 64** across it — *agents kept
staging while the commit landed, and nothing was lost.* Committed as `3a3c26e`.

⇒ **ROOT-CAUSE CLASS, and it is the useful part: WHEN A RULE IS VIOLATED REPEATEDLY BY PEOPLE
FOLLOWING IT, THE RULE PAIR IS THE DEFECT — NOT THE DISCIPLINE.** Five recurrences and a documented
warning did not help, because the warning asked for behaviour the other rule forbade. ⇒ **Before
re-issuing a rule that keeps being broken, check whether obeying a second rule makes breaking the
first unavoidable.**

⚠️ **It does NOT replace reading `git diff --cached --name-only` first.** When foreign work genuinely
belongs in a commit, take it and **name it** — a recorded sweep is recoverable, a silent one is not.

---

## C107 — AT THREE SEEDS THE LADDER'S SUBSTANTIVE COUNT IS **ZERO**, ON BOTH ROUTES (2026-08-18)

**RETRACTED:** C100's one substantive survivor. `ll_s09000 lead_gap` (K1B +0.748 [+0.002, +1.624])
is **SEED-UNSTABLE** — buckets `survive_both / die_at_repair / die_at_repair`, K1B **+0.748 /
−0.022 / +0.145**, alpha **1e3 / 1e7 / 1e5**. **It separates on one seed of three.**

**Reproduction gate PASS — 3465/3465 fields identical** to the banked route-A seed-0 rows, so this
is an *extension* of C100's run, not a different one.

| of the 87 banked separated-FAILs | route A | route B |
|---|---|---|
| dead on all three seeds | **58** | **58** |
| flip to PASS (the positive controls, unanimous) | 11 | 11 |
| survive all 3 seeds — **all `ego_yawrate`**, `\|K1B\|/gt_sd` ≤ 0.013, **one on a random-latent null** | 9 | 9 |
| survive on some seeds but not all | 9 | 9 |
| ⭐ **SUBSTANTIVE** | **0** | **0** |

⇒ **C100's DIRECTION is confirmed and its exception is gone.** 22 rows are SEED-UNSTABLE (23 on
route B). **The two routes return the same inventory.**

### ⭐ The mechanism, now measured on all 165 rows instead of 4 arms

The **defective** instrument picks the same alpha on all 3 seeds for **132 of 165** rows; the
**repaired** one for **42** — a **3.1× drop in seed-stability CAUSED BY A REPAIR**, confirming
C103's account at scale.
⚠️ **And the agent published the counter-column rather than only the supporting one:** max K1 seed
spread is *larger* on the **incumbent** (4.239 vs 2.812). ⇒ The honest claim is **"the repair unfroze
the majority"**, not *"the incumbent had no variance"*. *That distinction is the difference between a
finding and a slogan.*

### ⚠️ MY OWN C103 ROUTE REASSURANCE WAS ALSO A ONE-SEED CLAIM

I wrote in C103: *"44 paired rows, 2 alpha choices differ, **0 verdicts differ**."* At **3 seeds over
176 rows**: **2 verdicts and 11 guard verdicts differ**, and max `|ΔK1|` grows **0.396 → 0.721**.
Neither verdict difference is a PASS↔FAIL flip (both are stable-vs-unstable), and both routes still
return the same inventory — **but the reassurance did not inherit.**
⇒ **C103's own root-cause class, applied to C103: a stability claim measured at one seed is not
inherited by the multi-seed result.** I logged the rule and then quoted a single-seed reassurance
inside the same entry.

### ⭐ A MISSING CONTROL WHOSE ABSENCE WAS INVISIBLE

The re-read's only `C-V0` sat on the **cells** cache ⇒ **33 of the 165 rows — every tokens-window
arm — had NO trivial-proxy control at all.** Nothing reported a gap; the column simply was not there.
Added `proxytok`. Result: of **154 paired rows the ego-speed SCALAR matches or beats the latent on
120**; on the signal-bearing rungs, **13–14 of 14**.
⛔ **The v6 latent's only 3-seed-stable guarded PASSes are `n_agents_all` at four checkpoints — the
scalar wins ALL FOUR, and seed 0 is the outlier on every one.**
⇒ **RULE: a control must be enumerated PER ARM, not per study.** "We ran the trivial-proxy control"
was true and covered 80 % of the rows.

### ⚠️ A VERIFICATION THAT FALSELY PASSED — and the rule it earns

The agent's first staging check **word-split space-containing paths into 360 fragments and compared
empty-to-empty**, reporting *"360 files, 0 mismatches"*. Redone with `-z`: **72 files, 0 mismatches.**
⇒ ⭐ **"A CHECK THAT CANNOT FAIL IS NOT A CHECK."** This repo's paths contain spaces
(`TanitAD Research Hub`), so any unquoted/word-split verification degenerates to a tautology that
reports success. **Same family as the pooling story and the fixture-shaped test: a green result whose
green means nothing.**

### ⚠️ TWO OPERATIONAL FACTS WORTH KEEPING

⛔ **NEVER GATE ON A SUITE WHILE A MULTI-PROCESS CPU JOB IS LIVE.** A `taniteval` run concurrent with
the ladder chains returned **22 spurious FAILs** (subprocess timeouts under CPU contention); run
alone, the same files pass **34/34**. *A red suite caused by your own concurrency looks exactly like
a regression.*
⚠️ **The alpha grid is still binding** — the chosen alpha sits at a grid **edge** on **78 / 94 / 82
of 176** rows. Until the grid is widened, no ridge result here has a bracketed optimum.

### ⏳ Still stale, and the replacements have MOVED AGAIN

`POOLING_BOTTLENECK_R1R2.md` §1.5 (lines 111–119) and `2026-08-17-O234-DESIGN-RESEARCH.md` §3.4a
(line 427) + E-PROBE-A (line 1015) must be re-quoted from the **3-seed** column:
`n_agents_all` **0.076 → 0.1613**, `lead_closing` **0.0000 → 0.0009**, `r_pv0` **+0.052 → −0.107**.
*(The citation-sweep hazard from C103, one iteration on: the numbers moved again before the sweep
was done.)*

---

## C108 — THE DRIFT TOOL COMPARED THE **WORKING TREE, NOT HEAD** — `--ship` WOULD HAVE PUSHED ANOTHER AGENT'S WORK-IN-PROGRESS ONTO A BOX RUNNING A 5-DAY JOB (2026-08-18)

**MEASURED.** `launch_closure_audit.py` reported `stack/tanitad/models/v6.py` as **DRIFT** while
Thor's copy was **byte-identical to HEAD**. The dev box had moved, not Thor: a sibling agent's
**uncommitted `FROZEN_EXTERNAL_*` work** was in the working tree.

⇒ ⛔ **`--ship` would have copied that work-in-progress onto the training box.** Every row now
carries `local_dirty_vs_head` / `remote_matches_head`, and `--ship` **holds dirty files back**
unless `--ship-dirty`.

⚠️ **I RAN `--ship` ON THIS TOOL TONIGHT.** It was safe only by accident of ordering — I had
committed the seam fix (`14623d7`) *before* shipping, so the file was clean against HEAD. **Had I
shipped first and committed after — the more natural order — I would have pushed uncommitted code
onto a box 43 % through a 30 000-step run.**

⇒ **ROOT-CAUSE CLASS: A SYNC TOOL WHOSE REFERENCE IS THE WORKING TREE ANSWERS "IS THE REMOTE LIKE MY
DESK?" WHEN THE QUESTION IS "IS THE REMOTE LIKE THE REPO?"** With several agents editing
concurrently, the desk is *never* the repo. **The reference for a ship must be a COMMIT.** *Same
family as every scope trap this week — a true answer to the wrong question — and the first one that
would have written to a live training box.*

### ⭐ The entry points are now DERIVED, and the coverage gap is a NUMBER

Entry points are read out of **launch sources** — files that emit or document launch command lines
(`v6_chain.py`, `V6_GO_PACKAGE.md`, `GATE_PROTOCOL.md`) — as a **fixed point**, since a script named
by a launch source is itself one. **That matters concretely: `train_v6_staged.py` SUBPROCESSES
`eval_four_families.py`, `seam_probe.py` and `t1_eval.py` — argv, not imports, and INVISIBLE to any
import walk.**

**120 files (7 entries) → 134 (14) → 161 from 52 entries**, a strict superset, pinned by a test
asserting `ladder7 ⊂ fixed14 ⊂ derived`. Every run now prints its **root set** and a **coverage
gap** — *"54/158 launchable scripts covered, 104 not"* — with `--entry-mode executable` giving the
ceiling (298). ⇒ **The assumption the whole audit rests on is now visible in every run.**
⭐ **And the derivation did not narrow coverage:** four entries no launch source names
(`watch_gates.py`, `t1_summary.py`, `run_spectral.py`, `v5_guard.py`) are **kept as a FLOOR** and
printed `FLOOR ONLY`. *A derivation that narrowed coverage would be a regression wearing principle.*

Re-run against Thor: **4 more stale/absent files both hand-lists had missed** — correctly **not
shipped**, none is on the S-T path and a mid-run ship was not required.

### `pod_git_drift.py` — REPAIRED, not deleted, and the reason matters

⭐ **Deletion would have lost a capability nothing else has.** `launch_closure_audit.py` only
inspects files **inside the repo's closure**, so a **box-only file** is invisible to it *by
construction* — and that is exactly the REF-B v2 / TanitEval failure mode. **Two probes found zero
automated callers: it was doctrine that never ran**, which is why four defects accumulated in it.

A fourth defect the audit had not listed: **the index walked `.claude/worktrees/`** — 14 stale repo
copies, **8,079 indexed files against the repo's 2,132**, driving ambiguous basenames from **5.6 % →
89.4 %**. On the live fleet it would have printed **361 DRIFTED of which 293 (81.2 %) are
artifacts**; on the live checkout alone **227 of 251 (90.4 %)**.

⛔ **AND IT FOUND 45 GENUINELY STRANDED TanitAD FILES ON THOR** — `nurec_work/` 16, `/home/nvidia`
19, `rq_out/` 4 — **each existing in exactly one place on earth**, unwatched since the RunPod fleet
was released, because the nightly checker had been pointed at **four dead pods**. Plus **24 DRIFT
rows** in Thor's live checkout outside the launch closure, **direction unknown**. Filed as
`BACKLOG.md` **A11–A13**. *This is operating-standard rule 3 — "an artifact on one disk is NOT
done" — with a measured count at last.*

⚠️ **A sixth instance of the check-that-cannot-fail class, found while testing:** `--hosts` with no
values **scanned nothing and printed `TOTAL: 0`, exit 0.** Now a hard error.

⚠️ **Suite honesty worth copying:** `stack` **3892 / 0**, and the agent states the delta does **not**
fully reconcile (+16 its own, +9 a sibling's untracked file, **−1 unexplained against a baseline it
never measured**) ⇒ **it quotes the absolute, not the delta.** An unexplained −1 against an INHERITED
baseline is exactly the shape of C82/C86.

---

## C109 — C106 IS **HALF RIGHT**: the `ego_v0` half strengthens with a real interval, the `lead_gap` half dies, and the "3.6×" must go (2026-08-18, adversarial verification)

**Attacked five ways, with refutation as the default posture. Reporting both directions.**

### ✅ WHAT SURVIVES — and it is now properly estimated

**Random init really does read `ego_v0` better than the trained encoder.**
⚠️ **C106's bracket `[0.1736, 0.2011]` was the PROJECTION-SEED SPREAD, not an estimator** — a
dispersion quoted where a confidence interval was implied. The pre-registered **paired
episode-cluster bootstrap** on Δr²c (eval-window identity **asserted, not assumed**) gives
**+0.150 [+0.055, +0.226], p(Δ>0) = 1.000** in C106's own cell, and **positive in 27/27 cells**
across 3 init × 3 projection × 3 ridge seeds.

⭐ **And C106's mechanism is now verified FROM THE WEIGHTS, not only from source.** Random init has
residual fraction **0.0002**, cos **1.0000** against its own linear path — **it IS the raw-pixel
linear map.** ⚠️ **C106 asserted that half and never checked the other:** the trained arm **has
moved** (LayerScale **70× init**, residual **0.38**), which is what makes the comparison the one
C106 assumed. *An interpretation resting on two premises had verified one.*

### ⛔ WHAT DIES

* **`lead_gap`: 0 of 27 cells CI-separated**, p(Δ>0) only **0.71–0.76**, and **the sign flips in
  9/27.** ⇒ *"3.6× on both rungs"* becomes *"on `ego_v0`"*.
* ⛔ **THE RATIO ITSELF.** It compares a **near-constant** predictor (`pred_sd/gt_sd` **0.014**) to a
  **live** one (**0.89**). And re-drawing the **ridge inner split** — **the seed C106 held fixed,
  and worth 10× more variance than the projection seed it did vary** — moves it to **2.8× / 2.0×**.
* **C106's own "our readout is a flat line" caveat is one inner-split draw**: at `ridge_seed=2` the
  same arm reads `pred_sd/gt_sd` **0.95**.
* ⭐ **An attack that FAILED, reported because it failed:** widening α to **1e13** changed nothing
  (≤ 0.0008). **That favours C106**, and closes C107's standing alpha-grid concern for these rows.

### ⭐ THE FINDING THAT REFRAMES BOTH

⛔ **Our trained arm is NOT CI-separated from its own matched-random null** — `lead_gap` **0/9**,
`ego_v0` **3/9 and not in C106's own cell** — while **the random arm IS (9/9)**.
⇒ **The honest claim is SIGNAL vs NO-SIGNAL on one rung, not a ratio.** **A ratio whose denominator
is not separated from noise is not interpretable**, however many seeds it holds up across.

### ⛔ AND IT REACHES BACK INTO C104's EVIDENCE

**`PC-2OBJ` — the positive control C104 and C106 both cite — is INERT AT THE DEPLOYED POOLING RATIO
BY CONSTRUCTION**: two *opposing* plants inside one cell **cancel**. Run at p40 it **reproduced the
un-planted arm to 5e-05.** `PC-LOCAL`/`PC-DIST` are the controls that actually fire (our own trained
tokens through the deployed pool: **0.0596 → 1.0000, K1 9/9**).
⇒ ⚠️ **C104's "the ladder had full power to see a pooling-destroyed signal" rested on a control that
is silent at exactly the ratio in question.** *C104's conclusion is not overturned — PC-LOCAL/PC-DIST
do fire, and the 40:1→1:1 null result stands — but its headline control was the wrong one, which is
the D1/C79 shape (an instrument validated by a control that could not have failed).*

### ⭐ NEW, AND MONITORABLE FROM STEP 0

**The trained token field is RANK-COLLAPSED** — **97.6 % of token-channel variance in ONE
direction**, effective rank **1.22 against 67–68**; design-matrix rank **6.7 vs 16.4**. Confirmed on
the banked cache. PCA-whitening lifts *both* arms ~3× and closes nothing ⇒ **a co-symptom, not the
explanation** — but unlike *"the objective subtracts geometry"* it is **observable from step 0**, and
the existing `z_op` spectrum monitor demonstrably does not cover it.

⚠️ **CORRECTION — the step-0→9250 sweep is UNAVAILABLE, not merely unretrieved:** a whole-filesystem
probe of Thor found **nothing before ≈ step 9100**. But the programme's own `z_op` ladder has a
**step-2000** point, re-read from raw JSON: `ego_v0` **0.1346 → 0.0801** and `lead_gap` **0.0123 →
0.0059** by step 9000, **then flat**, while `nearest_any` stays flat throughout. ⇒ **There is a real
decline over 2000→9000**, so C106's direction has support — on `ego_v0`, over that window, and not
as a ratio.

⏳ **OPEN:** the frozen-external guard is built and pinned both ways (9 tests; catches the 86.6 M
un-freeze and **cannot be satisfied by freezing everything**) but is **NOT YET CALLED** by
`train_v6_staged.py` — deliberately, since the arm it protects does not exist yet. **It must be wired
in the same change that introduces E-XENC-1**, or it is a guard that never runs, which is the
`pod_git_drift.py` failure mode (C108) in advance.

---

## C110 — "45 STRANDED FILES" WAS AN UNDERCOUNT PRODUCED BY THE INSTRUMENT'S OWN FILTER — IT IS **102** (2026-08-18)

**RETRACTED — my own figure in C108**, and in `BACKLOG.md` A11: *"45 genuinely stranded TanitAD
files on Thor."*

**MEASURED.** `pod_git_drift.py:91` is `SUFFIXES = (".py", ".sh")` ⇒ **every stranded result JSON,
run log, `.md`, `.yaml` and `.bak` was invisible to it BY CONSTRUCTION.** A content-hash sweep over
the same directories found **102 files present on Thor and absent from the repo by content — the
tool could see only 47 of them.** *(It also under-reported one directory it could see: A11 records
`/home/nvidia` root as 19; the tool's own output prints **21**. The other seven counts are exact.)*

⇒ **ROOT-CAUSE CLASS: A COUNT QUOTED AS COMPLETE WHEN IT IS THE INSTRUMENT'S FILTER SPEAKING.** I
published "45" hours after logging C108's own lesson that a green checksum on the wrong file set is a
confident wrong answer — **and then took a number from a tool whose scope I had not read.** ⇒ **RULE:
before quoting a count, read the instrument's inclusion rule. A census is a claim about the FILTER
until proven otherwise.** *Same family as `df`, `free`, the cgroup counter, and the closure's
hand-listed roots: a real number answering a narrower question than the one asked.*

### ⭐ AND THE MOST CONSEQUENTIAL FIND IS A **WRONG** FILE, NOT A MISSING ONE

The repo's banked `…/incoming/2026-08-02-thor-deployment-profile/thor_profile.py` **cannot have
produced its own co-banked `thor_profile.json`**: the JSON carries `"frame": "176x624 hfov 117.0"`
and **the banked script never assigns `out['frame']`**. Thor's copy adds exactly that, plus
`speed_input=True` / `action_dim=3` via `resolve_v2_frames` ⇒ **a DIFFERENT MODEL WAS PROFILED than
the banked script describes.**
⛔ **And `pod_git_drift.py` saw this file and downgraded it to `NAME_ONLY` — "weak evidence, not
drift".** That classification is precisely how it escaped. Filed as **A15**; **not acted on** —
which side is canonical belongs to that package's owner.
⇒ **A banked script that cannot produce its banked result is worse than a missing one: the pair looks
like provenance.**

### What was actually rescued

**117 files, 807,347 B, 117/117 sha256-verified on BOTH sides, 0 mismatches**, pulled by
`ssh -n 'tar -czf - <explicit list>'` streaming to stdout — **nothing written on Thor.**

⭐ Highest-value: `nurec_work/` **37 files** (the msgpack walk, rig/pose derivation, the `.ckpt`
pickle probe, the whole PPISP/CRF recovery including the quantile-matching method, 11 scoring
`report.json`) · **`_s1_backup/refc_pre.py`** — 115 KB of REF-C model source **as it stood before the
S1 patch**, which is the REF-B-v2 failure mode exactly · 28 Thor bring-up/probe scripts ·
`parity_verify/make_prefix.py`.

⭐ **Deliberately left, with reasons** — `get-pip.py` (third-party), two regenerable NuRec extracts,
one byte-identical duplicate, and ≈3.7 GB of PNG/npz/pt outputs. **The 17 run logs were KEPT**: raw
measurement transcripts, not noise. *Judging by content rather than extension is the whole point,
given that an extension filter is what caused the undercount.*

⚠️ **`*.mp4` resolved rather than skipped:** none in the eight A11 directories; of 26 elsewhere on
Thor, **18 are already banked byte-identical (md5-checked, not name-checked)** and the other 8 are
superseded lower-quality render passes ⇒ **no `git add -f` needed.** *(Name-checking would have
missed the distinction — the same trap as `NAME_ONLY` above.)*

**Beyond A11:** zero `.md`/`.py`/`.sh` stranded in the AlpaSim result directories — **those
conclusions are safe.** 19 gate/summary JSONs banked; 24 bulk rollout dumps (9.6 MB) recorded with
sha256 but not pulled, filed as **A14**.

⏳ **Two instrument fixes escalated, correctly not made by the finder** (shared instrument):
**widen `pod_git_drift.py:91` beyond `.py`/`.sh`** — it missed **46 of 102 (45 %)** — and **raise
`NAME_ONLY` severity for program-authored basenames.**

⚠️ **This closes the backlog item, not the hole:** the nightly checker had been pointed at four dead
pods since 2026-08-15, which is why any of it accumulated.

**Trainer advanced across the pull, not merely alive:** step **13,150 → 13,200**, `step_s` **26.4716 →
26.4707** (−0.003 %), PID 25477 `kill -0` alive at both ends.

---

## C111 — ⛔ **A LIVE HF TOKEN SAT IN A PLAINTEXT LOG ON THOR, AND I ALMOST PUSHED IT TO GITHUB** (2026-08-18)

**MEASURED.** The Thor rescue banked `rescued/rq_out/logs/contention.log`, whose **line 11 contains a
Hugging Face User Access Token**. I committed it locally. **GitHub push protection rejected the push
(GH013)** and named the file and line.

⇒ ⛔ **NOTHING WAS PUSHED. The secret never left this machine.** The commit was undone
(`reset --soft`), the token redacted in place, and **the entire 117-file package re-scanned: exactly
one file was affected, the other 116 are clean.**

⛔ **THE UNBLOCK URL WAS NOT USED.** GitHub offers a one-click "allow this secret" link. Using it
would have published a live credential — **the remedy for a blocked secret is to remove it, never to
authorise it.**

### Why this happened, and it is not the rescue's fault

⚠️ **The invariant we hold is about `Keys.txt` — "read tokens in place; never copy, print, or write
them to args."** ⛔ **This token was in neither `Keys.txt` nor a script: it was in a RUN LOG, because
some earlier command carried it on a command line and the log captured stdout.** The invariant
protects the *source* file and says nothing about the *exhaust*.

⇒ **ROOT-CAUSE CLASS: WE GUARDED THE CREDENTIAL STORE AND NOT THE ARTEFACTS THAT RECORD ITS USE.**
A token in `Keys.txt` is protected by `.gitignore`; the same token echoed into
`contention.log` is protected by nothing. ⇒ **RULE: any bulk import from a machine — rescue, pull,
backup — is SCANNED FOR CREDENTIAL PATTERNS BEFORE IT IS STAGED, not after a remote rejects it.**
The scan is seconds; it belongs in the rescue procedure.

⚠️ **AND THE RESCUE'S OWN "JUDGE BY CONTENT, NOT EXTENSION" PRINCIPLE IS WHAT EXPOSED IT** — the 17
run logs were deliberately kept as *"raw measurement transcripts, not noise"*, which was the right
call for provenance and is exactly why the scan is required. **Keeping logs is correct; keeping them
unscanned is not.**

### ⏳ TWO PI ACTIONS, and the first is time-sensitive

1. ⛔ **ROTATE THE TOKEN.** It is **still present in plaintext on Thor** at
   `~/rq_out/logs/contention.log:11` — redacting our copy does nothing to the machine's. Treat it as
   **exposed**: it sat in a world-readable log on a networked box for an unknown period, and it is
   the token with **WRITE access to the `Sayood/` HF namespace.**
2. ⚠️ **The pre-redaction blob still exists in local git objects** (the undone commit
   `ab77da96`), unreachable and never pushed. It disappears on the next `git gc --prune=now`, which
   was **not** run now because several agents are live and it is not urgent while the ref is
   unreachable and the token is being rotated anyway.

⭐ **Worth stating plainly: the control that caught this was GitHub's, not ours.** Our own procedure
staged it, committed it, and would have pushed it. **That is the finding.**

---

## C112 — ⛔ **201 ALPAMAYO CLIPS ARE ALREADY IN THE PARITY TRAIN CORPUS** — any eval split built on it is 4.3 % contaminated (2026-08-18, concurrency pilot)

**RETRACTED:** the standing claim that the Alpamayo augmentation set does **not** overlap the parity
corpus.

**MEASURED: 201 of the 4,729 Alpamayo clips are already in the parity TRAIN corpus**, and **the live
trainer is reading exactly that cache** — `--v2-cache …e438721ae894-w120-256x640cyl`, read from
`/proc/25477/cmdline`, not from a doc. **mtimes prove they are genuine selection members, not later
contamination.**

⇒ ⛔ **Any eval split built on this corpus is 4.3 % train-contaminated — the REF-A I-JEPA leak class**
(where ~80 % of val sat inside train and made the number unusable). The **201-id exclusion list is
banked**; it must be applied **wherever an Alpamayo eval split is defined**, not once.

⇒ **ROOT-CAUSE CLASS: A NON-OVERLAP ASSUMED FROM PROVENANCE ("different source ⇒ disjoint") RATHER
THAN COMPUTED FROM IDS.** Parity is sacred precisely because it is checkable; **this one was never
checked.**

### ⛔ AND MY OWN ABORT CRITERION WAS STRUCTURALLY UNABLE TO FIRE

I briefed a **+5 % `step_s` abort**. `train_v6_staged.py`'s `step_s` is a **CUMULATIVE MEAN over
every step since process start** — its own `step_s_note` says so. ⇒ At the intended trip point
(27.7 s/step) **the cumulative mean never reaches 28.0 at ANY duration**; even at a catastrophic
**40 s/step it needs 9 hours**, and the pilot ran ~2 h. **It would have reported "safe" no matter
what happened.**

⇒ **This is the third gate this week that could not return the answer it existed to give** — E4 could
not report PASS, SEL-1 could not report FUNDED, and this could not report ABORT. ⭐ **The agent
first-differenced the series AND UNIT-TESTED THE TRIP LOGIC ACROSS THE BOUNDARY before trusting it** —
which is the step that turns a written criterion into a working one.
⚠️ **Corollary: the reference band "26.47–26.66 all day" is NOT variation — it is a CONVERGING MEAN,
strictly non-increasing over 100 points.** A quantity that cannot rise is not a monitor.

### The answer, with the effect reported honestly rather than rounded away

| phase | n | median s/step | Δ vs BEFORE | p |
|---|---:|---:|---|---|
| BEFORE (steady) | 108 | 26.3591 | — | — |
| **DURING** | 5 | **26.4993** | **+0.532 %** [+0.282, +0.785] | **0.00064** |
| **AFTER** | 4 | **26.3474** | −0.045 % [−0.095, +0.384] | 0.713 |

⇒ **RUN THE EXTRACTION CONCURRENTLY.** The effect is **real — it must not be rounded to "no
effect"** — but it is ~6× below the threshold: **40 minutes of training bought against 5.3 days of
calendar.** ⭐ **The AFTER phase is what makes it causal rather than coincidental** — the slowdown
appears with the load and disappears with it. *And that mattered: the first clean after-point was
26.4600 (+0.38 %), which alone would have suggested the effect was not the extraction; three more
points settled it.* GPU utilisation was indistinguishable (98 % vs 97 %); `gnorm`/`loss` stayed
inside baseline variation. **All 10 chunks completed, 476/476 clips verified BY CONTENT, no abort,
trainer and snapshot daemon alive at all 141 polls.**

### Three more findings

⭐ **Thor's own throughput is 11.76 MB/s — 4.7× the 2.5 MB/s dev-box figure** I published as
*"MEASURED but n=1"* (707.8 MB sustained over 60 s; a second host agrees at ~13.6 MB/s; it self-limits
to 8.58 MB/s under concurrent extraction). **The w120 sizing was built on the wrong number and was
4.7× pessimistic.**

⚠️ **A second contaminated baseline, found and excluded:** 18 of 126 pre-load points were a
**post-resume warm-up transient** (~900 steps at 27.1 s/step), **consecutive rather than periodic** —
a checkpointing explanation was tested and **refuted**. Excluding it *strengthened* the result;
leaving it in would have **hidden** the effect. *It also explains the cumulative-mean decay above.*

⛔ **A launch-path defect that would have produced a SILENTLY WRONG CORPUS:** `build()` died **after
paying for a 536 MB download** because nothing creates `<root>/r0/r0_selection.parquet` — and that
path also drives **per-clip intrinsics**, so the fallback would have cropped **~215 px wrong for rig
B**. *The crash was the good outcome.*

⚠️ **Full-extraction cost, since the corpus is brutally density-skewed (median 2 clips/chunk):**
50 % of clips = **5.2 h** of download; 100 % = **41.8 h and 1.73 TB**. Cap by density — **while
recording that this is a dataset-composition decision, not merely a cost one.**

⚠️ **The 476-clip pilot corpus (18.33 GB) lives ONLY on Thor** (`/home/nvidia/w120pilot/out/`) — too
large for the repo, with a fully staged rebuild recipe (2 h 19 m). **A recorded risk, not a stranded
artifact** — the distinction being that the recipe is banked.

---

## C113 — C112's headline rate was **4.3 %; the buildable rate is 78.2 %** — and the leak I was sent to close is the *less* dangerous of the two directions (2026-08-18, eval-contamination closure)

**CORRECTED, not retracted:** C112 is right that 201 of 4 729 Alpamayo clips are in
`physicalai-train-e438721ae894`. Three things it could not see:

**1. ⛔ The rate that matters is 78.21 %, not 4.25 %.** A split can only contain clips that EXIST.
**Only 257 of the 4 729 have w120 video built**, and **201 of those 257 are parity-train** —
`raw/contamination.json::buildable_eval_split_today`. So the Alpamayo eval split buildable today is
**REF-A-I-JEPA scale (~80 %)**, not "the same class at smaller scale". ⇒ **ROOT-CAUSE CLASS: A
CONTAMINATION RATE QUOTED OVER THE CATALOGUE RATHER THAN OVER THE BUILDABLE SET.** The denominator
that flatters is the one that is easy to count. Same family as `df` reporting the cluster.

**2. ⛔ The 201 do not *coincide with* the aug120 perception corpus — they ARE it, exactly.** C112
read the matching count as a coincidence. **MEASURED: `fused_aug120_v2_index.jsonl` and
`fused_aug120_v3_index.jsonl` both hash to `80632f17…`, byte-identical to the exclusion list —
201/201 = 100 % inside the parity train corpus.** The mechanism is one line,
`aug120_pipeline.py:53`: `todo = (records ∩ w120_corpus) − done`, and the w120 corpus **is** the
parity geometry sibling. **The cohort was SELECTED FROM the train corpus.** ⇒ *A matching count
between two sets is a prompt to test set EQUALITY, not a coincidence to note.*

**3. ⭐ THE OTHER DIRECTION IS WORSE, AND NOBODY WAS LOOKING AT IT: 6 of the 40 canonical val
episodes (15.0 %) are inside the Alpamayo record set.** Not "an eval split contains train clips"
but **"a train corpus is about to swallow the deployed val"** — the set behind EVERY published
open-loop number (881 stride-8 windows). **Blast radius today is ZERO** (nothing trains on those
labels) and the trigger is already scheduled: the 4 472-clip build. **No existing guard would fire**
— `parity.py` §9 checks a cache against ITS OWN corpus digest, and an augmentation corpus is a
different corpus by construction. ⇒ **Whoever runs that build must call
`parity.filter_train_clips()` first.**

### Why the assumption was made — it was UNANSWERABLE, not lazy

The manifest carries `clip_id_sha256_sorted`: a digest of the **whole sorted list**, which is a set
IDENTITY and cannot test one element. The ids are gated-confidential and live only on pods. So
*"is clip X in the parity train split?"* **had no answer on any other host** — and an unanswerable
question gets answered by provenance. ⇒ **The fix is the missing ORACLE, not a reminder:**
`parity.py` §10/§10b + a committed per-clip `sha256` set (`parity_train_clip_digests.json`,
`deployed_val40_clip_digests.json`). Membership is exact, enumeration stays impossible, §9's
confidentiality rule holds. ⭐ **The mint REFUSES to write unless its source reproduces the
committed corpus digest** (the `register_v2_geometry_sibling` contract), and the whole chain —
banked listing → manifest digest → per-clip digests → committed file — re-walks inside the repo on
every test run, with **no pod access**.
⚠️ **Derived, never hand-listed** (C99/C105): the question asked is *"is this clip in the parity
train split?"*, so the next 4 472 clips need no list update.

### ⚠️ A15, settled the same day and the same class

**`thor_profile.py` cannot have produced its co-banked `thor_profile.json`** — and the gap is not
the missing `out["frame"]` line. As written it builds a **256×256** positional embedding
(`flagship4b_config()` is `image_width=None`) and feeds it 176×624 → **263.44 M** params vs the
JSON's **263.58 M**; and only **`action_dim=3`** — the SPEED channel, default 2 — reproduces
263.58 M at width 624, so **as written it profiles the no-speed variant**. Its docstring also gives
the sub-frame the PARENT's 120° when the 176×624 slice is **117.0°**. **The JSON is canonical**; the
script now says so in its own first lines. ⇒ *A banked script that cannot produce its banked result
is worse than a missing one, because the pair looks like provenance.*

**Package:** `…/incoming/2026-08-18-alpamayo-parity-exclusion/` · pin:
`stack/tests/test_eval_contamination.py` (17 tests; each guard neutered in turn and required to go
red — 4, 8 and 1 failures respectively).

---

## C114 — A FULL-SUITE RUN TAKEN WHILE SIBLING AGENTS EDIT THE TREE MEASURES A **TORN SNAPSHOT**, NOT THE CODE (2026-08-18)

**MEASURED.** A full `stack` run returned **`3 failed, 3988 passed, 7 skipped, 2 xfailed`** in 605 s
— **while reporting exit code 0.** *(The "exit codes are not evidence" trap firing on the very run
that was meant to verify a fix. Reading the output is the only reason the failures were seen.)*

**All three failures were a sibling stream's mid-edit state, and the mtimes settle it in the OPPOSITE
direction to the obvious guess:**

| file | mtime | vs the suite window (04:45:12 → 04:55:17) |
|---|---|---|
| `scripts/train_v6_staged.py` — holds `STAGE_MAY_INTRODUCE` | 04:44:45 | **BEFORE** — already carried `t2_head.` |
| `tests/test_v6_stage_init_introduction.py` | **04:58:33** | **AFTER** — still the old 5-tuple during the run |

⇒ **The suite ran a SOURCE file that had already grown the allowance against a TEST not yet updated
to expect it.** The natural reading — "a test was updated ahead of its source" — is backwards.
Re-verified three ways: **10 pass in isolation, 51 alongside the new guard files, 62 alongside the
sibling's own T2/T5 tests.**

⇒ **ROOT-CAUSE CLASS: A NON-ATOMIC READ OF THE WORKING TREE.** This is the *same family* as the
contention rule (*22 spurious failures from CPU starvation alone*) but a **DISTINCT MECHANISM** —
pytest walks 223 files over ten minutes while agents rewrite them underneath, so the run reports on a
state that **never existed as a coherent commit.**
⇒ ⭐ **RULE: attribute a full-suite failure to FILE MTIMES before attributing it to your own change.**
A red suite under live concurrency is a claim about *when the reader arrived*, not about the code.

⚠️ **Consequence for the numbers I published this session:** `stack` **3893 / 0** was measured at
`6784455` and was valid then; the tree has since grown (**3,988 collected**). **A suite count is
stamped to a commit, not to a session** — quoting last hour's total after five streams have landed is
the same class as quoting a dry-ladder measurement as the live run (C96).

⭐ **AND IT VALIDATES A DECISION MADE INDEPENDENTLY.** The same unstaged dependency that produced the
torn snapshot is why `train_v6_staged.py` was **held back** from commit `3f2287b`: its staged form
imports `T2_AUGMENTATIONS` **3×** while HEAD's `v6.py` has **0** and the worktree's copy is
**unstaged**. ⇒ **One concurrency defect surfaced twice — once as a false test failure, once as a
would-be broken HEAD** — and both were caught by checking *what the index actually holds* rather than
what the working tree shows.

---

## C115 — `z_tac` HAS NO TEMPORAL MIXING: IT IS A FUNCTION OF THE LAST FRAME ALONE, SO HALF OF CATALOG T2 IS NOT EXPRESSIBLE (2026-08-18, F-7/F-8 build)

**MEASURED, two independent ways, while building F-7:**
1. **Freezing every non-last frame leaves `z_tac` BIT-IDENTICAL.**
2. **`z_tac(time_reverse(x))` equals `z_tac` of a window made entirely of `x[:,0]`.**

**Mechanism:** `encode_window` flattens `[B, W]` into the **batch** axis, so **no frame ever sees
another**. The tactical latent is a per-frame quantity that happens to be read at the last tick.

⇒ ⛔ **HALF OF CATALOG T2 IS NOT EXPRESSIBLE ON THIS ARCHITECTURE.** A *time-reversal* augmentation
does not produce "the reversed manoeuvre" — it produces **a frame W ticks earlier**, which is a
different *sample*, not a different *manoeuvre*. ⇒ **Using it as a hard negative would directly
OPPOSE T5/F-8**, whose whole content is that consecutive windows should agree. Built, and
**excluded from the default negative set** — a catalog/diagram decision, escalated rather than taken.

⇒ **ROOT-CAUSE CLASS: A SPEC WRITTEN AGAINST AN ASSUMED REPRESENTATION.** The catalog describes
manoeuvre contrastives as though `z_tac` integrated a window; it does not, and nothing in the catalog
is wrong *about the objective* — the mismatch is with the **representation the objective would act
on**. ⇒ **RULE: before implementing an augmentation, prove the representation is SENSITIVE to the
thing being augmented.** A contrastive term over an invariant is a no-op wearing a loss function's
clothes.

⚠️ **This bears on more than F-7.** Any claim that the tactical layer "integrates a window", or that
its latent carries manoeuvre *dynamics* rather than an instantaneous read, must now be checked
against this fact rather than assumed. *(It does not touch the operative path, which rolls out
explicitly, nor the four-family results.)*

### Three more, each caught by a control rather than by a run

⛔ **F-8 is DEGENERATE ALONE:** a flat plan scores **exactly 0**, and the emission is **exactly zero
at init**, so the term starts **at its own global minimum** and has nothing to descend. Guarded in
**both** `v6_loss_step` and `preflight` — not one of them.
⚠️ **A defect in the agent's OWN trivial-proxy control, caught by its own test:** it issued verdicts
at **any n**, and at **n = 4** the null ratio spans **0.397–3.361**. Now refuses below **n = 32** and
reports SEM. *(A control that returns a verdict on four samples is C107's "a check that cannot fail",
inverted — a check that cannot be trusted.)*
⭐ **The preflight guard itself had an `AttributeError`** — `a.shared_encoder` does not exist; the
flag is `--per-layer-encoders`. **Found by an assertion, not by running it.**

⛔ **AND THE FULL `stack` RUN FAILED 3 TESTS WHILE EXITING 0** — the allowlist pin firing correctly
on `t2_head.`. **Had the exit code been trusted, three failures would have shipped.** *(Eighth
instance this session.)*

**Costs, MEASURED by building, not estimated:** F-7 **+164,225 params / +5 keys**; F-8 **+0 / +0**
(zero-parameter). **Default build unchanged at 87,893,449 / 405** — verified twice, top and tail.
**Earliest legal insertion:** F-7 at **S-T** (`STAGE_MAY_INTRODUCE["S-T"] += ("t2_head.",)`) — **no
fresh S-W run needed**, since S-W does not train `layer_tac` anyway; **F-8 needs no insertion point
at all** and can be enabled over any existing checkpoint.
⭐ **The `06b8782` class does not apply:** the entry went into the **existing** `layer_tac` group via
`_GROUP_PREFIXES`, so `MODULE_GROUPS` is untouched — and `STAGE_GROUPS["S-J"] is MODULE_GROUPS`
verified **False**.

---

## C116 — C115 IS A **DEFECT, NOT A DESIGN**: the temporal pool the paper mandates ALREADY EXISTS AND IS SIMPLY NOT WIRED — and `φ_str` has the same hole (2026-08-18)

**C115 REPRODUCES INDEPENDENTLY**, by two mechanisms its author did not use: the gradient from
`z_tac` to history frames is an **exact structural zero** — `[0.0, 0.0, 0.0, 0.0637]` — while the
**positive control `ẑ_op` on the same graph is nonzero at every frame** (`[0.047, 0.022, 0.021,
0.131]`). History-splicing leaves `z_tac` bit-identical; swapping the **last** frame moves it **0.779
of its own scale**. Holds in `train()` mode.

### ⭐⭐ THE UPGRADE: it is a DEFECT, and the proof is POSITIVE rather than an absence

**`PhiTac` — the causal-TCN temporal pool that the design docs, the backlog AND `TANITAD_PAPER.md`
all mandate — ALREADY EXISTS** (`tactical.py:99`), **is tested, and was trained** (registry §1.13b).
⛔ **`v6.py` references it ZERO times.** Git provenance verified in-repo: the flattening arrived in
**`0c30a0f`**, a commit about a **selector verdict** whose message mentions encoder, window and
temporal extent **zero times**.

⇒ **ROOT-CAUSE CLASS: A CORRECT COMPONENT EXISTS, IS TESTED, AND IS NOT CALLED — and the change that
orphaned it was about something else entirely.** This is not "we never built it" and not "the spec
was wrong": **the spec and the implementation agree with each other and disagree with the wiring.**
⇒ **RULE: a component's tests passing says nothing about whether anything CALLS it.** The
`pod_git_drift.py` class (C108 — doctrine that never ran) in an architecture costume, and the
sharpest form yet: **a green test suite over dead code.**

### ⭐ NEW, AND C115 DID NOT MENTION IT: `φ_str` HAS THE SAME DEFECT

`φ_str` is specified as a pool over a **`z_tac` window**; `uplink_str` reads a **single tick**.
**MEASURED: `∂z_str/∂frames` is also EXACTLY ZERO at every history frame.** ⇒ **A second retraction
line, on the strategic layer** — and it means the hierarchy is instantaneous at *two* of its three
levels.

### ⛔ THE PAPER'S FORMAL CONTRACT IS FALSE TODAY

`TANITAD_PAPER.md:686` states **`z_T = φ_T(sg[z_O(t−3..t)])`** — inside a section that calls these
properties **"checked, not assumed"** — and **`assert_isolation` checks only the OTHER property.**
34 sites inventoried and split invalidated / needs-ruling / unaffected. **False under either outcome
of the pending experiment**, so it must be corrected regardless.

### ⭐ THE OVER-CORRECTION GUARD, AND IT IS LOAD-BEARING

**The programme has THREE tactical implementations, and TWO genuinely integrate the window.**
MEASURED: `TacticalStage0` does so **under the same probe that fails on v6** (nonzero gradient at
every slot). ⇒ ⛔ **"The tactical layer is fake" is WRONG — this is a fact about `V6Stack` alone.**
**Untouched:** the operative path (it rolls out explicitly), all four metric families, and every
v1-flagship tactical result **including the `ctx→tactical` seam**.

### ⚠️ THREE INSTRUMENT HAZARDS — two inside the agent's own probe

⛔ **`z_tac.sum()` IS AN IDENTICALLY-ZERO SCALAR.** `adapter_tac` ends in LayerNorm, so `.sum()` has
**exactly zero gradient for ANY input**. The first run therefore returned all-zero gradients
**including the last frame** — *a stronger, FALSE version of C115.* **Caught by its own positive
control**, fixed with a random projection, and **pinned as a negative control**. ⇒ *An instrument
that confirms your hypothesis harder than the truth is the most dangerous kind of agreement.*
⚠️ **At the default config the autograd probe is uninformative** — `isolate_uplink=True` severs the
graph and **mimics the finding by a different mechanism.**
⛔ **`v6.py` GREW 4,914 → 5,154 LINES DURING THE READ** (`adapter_tac` moved 3737 → 3977). Every
citation is now stamped to `sha256 d1cd69d7…` and the probe re-run against that exact state. ⇒
**C114's torn snapshot IN A CITATION COSTUME — and the more dangerous form, because a stale line
number still looks authoritative.**

### The fix, and the experiment that decides whether to make it

**Legal insertion is S-T ONLY** (the sole non-empty `STAGE_MAY_INTRODUCE`), as a **new key prefix,
never a widened shape**, **default-OFF** — `load_resume` is hard `strict=True`, so any default-build
change **kills the live v6F resume**. ⇒ **No fresh S-W run needed.** *(Fixing it inside
`encode_window` instead would cost a full S-W run.)*

⚠️ **`PREREG_TEMPORAL_LATENT.md` ALREADY EXISTS AND ALREADY RAN** — it fired **OUTCOME V** for
`long_accel` (**0 of 35 arms**, including a 9-frame window basis), so *"just give the readout more
frames"* is **already measured insufficient for that channel**. ⛔ **But it explicitly does NOT
retire the thesis for the MANOEUVRE DECISION**, which is the target here.
⭐ **The decisive arm is `A-pad`: `PhiTac` over the last frame repeated W times — identical
parameters, ZERO temporal information.** That separates **information** from **capacity**, which is
the only way the result is interpretable either way.

⏳ **Three rulings owed:** whether "temporal hierarchy" claims the **state** or the **clock**; the
`TANITAD_PAPER.md:686` correction; and opening the `φ_str` line as its own retraction.

---

## C117 — ⛔ I REPORTED THE UNIFIED PERCEPTION CORPUS AS "PUBLISHED GATED-PUBLIC". IT IS ON ONE DISK. (2026-08-18)

**RETRACTED — my own claim**, in the 2026-08-17 program report (§1 *"Published gated-public
(`sam3_unified_201_v2/`, `fused_aug120_v3/`) with a 3/3 byte round-trip against md5s committed to
git"*) and repeated in later summaries.

**MEASURED, by me, after two agents disagreed:** `pushed_to_hf: **false**` in **both**
`…/2026-08-17-perception-floor-unify/raw/floor_homogeneity_manifest.json` and
`…/raw/f7_run86_local.json`. A repo-wide sweep of every banked manifest returns exactly **one**
`hf_repo` — `Sayood/flagship-v4.2b`, **a MODEL repo, not the perception corpus.**

⇒ ⛔ **THE UNIFIED 201-CLIP PERCEPTION CORPUS EXISTS ON ONE DISK ON THE DEV BOX.** Only the **older
115-clip leg** is on HF, and it is in a **PRIVATE** repo — not gated-public.

⇒ **TWO ERRORS IN ONE SENTENCE, and the second is the dangerous one:** I named the **wrong
artifact** (the 115 leg's push, reported as the unified 201) **and the wrong visibility** (private,
reported as gated-public). ⭐ **A push that really happened, attributed to a thing that was not
pushed, is far harder to catch than an invented one** — every part of the sentence is individually
true of *something*.

⇒ **ROOT-CAUSE CLASS: "DONE" INHERITED ACROSS A SUPERSEDING ARTIFACT.** The 115-clip leg *was*
finished and pushed; the 201-clip corpus **superseded** it; the completion status did not get
re-derived for the new artifact. **This is exactly the failure operating-standard rule 3 exists to
prevent — *"an artifact on one disk is NOT done"* — and I published it as done.**
⇒ **RULE: when an artifact supersedes another, its published/banked status is RE-DERIVED, never
inherited.** The superseding step is precisely where "done" silently detaches from the thing it was
true of.

⏳ **PI DECISION, and it is the most urgent item in the data layer:** push the unified corpus off one
disk. *(The programme has already lost REF-B v2's architecture, the TanitEval harness, the pod ops
bundle and 486 lines of TanitResim to this exact shape.)*

### ⚠️ Two more corrections to briefs I wrote

⛔ **I briefed `DATA_STRATEGY.md` as "a month stale". It was ONE DAY stale** — already at **v3.0
(2026-08-17)**, not the v1.0 of 2026-07-06. **The "month stale" phrasing is a CORRECT 2026-08-15
observation that I inherited forward PAST ITS OWN REPAIR.** ⇒ Same class as the above: a true
statement about a superseded state, carried into the present. *(The refresh was still warranted —
v3.0 was overtaken inside 24 h — but a reader would have expected a very different starting point.)*

⭐ **The stranded-file counts RECONCILE — all three are correct, for three different quantities:**
**102** stranded by content hash → **98** pulled (4 left with recorded reasons) **+ 19** gate JSONs
found beyond the A11 roots = **117** banked. `BACKLOG.md` A11's "98" is a fourth correct number for a
fourth quantity. *Independently re-counted from the banked tree.* ⇒ **Not a contradiction — four
denominators.** Which is C110's lesson again: **name the denominator or the number is unquotable.**

### ⛔ C111's RULE IS STATED AND UNIMPLEMENTED

**MEASURED at three probes** (scripts/tests by name; repo-wide grep for `detect-secrets`/`trufflehog`
and similar; `pre-commit` + `.github/workflows/` + the operating standard): **NO CREDENTIAL SCANNER
EXISTS.** C111 mandated that any bulk import be scanned **before** staging, and nothing implements
it. **The token is still plaintext on Thor.** ⇒ *A rule with no mechanism is C108's "doctrine that
never ran", written yesterday.*

⚠️ **And the space-splitting trap caught ME during this very verification** — an unquoted `for f in
$(grep -rl …)` tore this repo's `TanitAD Research Hub` paths into fragments and reported
`No such file or directory` for each. Redone with `-rlZ` + NUL-delimited read. **Third logged
instance, first one in my own hands.**

---

## C118 — A DERIVER THAT UNDERCOUNTED THREE TIMES, AND A NEUTER-MATRIX THAT LEFT A FILE BROKEN ON DISK WHEN KILLED (2026-08-18, build parity guard)

### ⭐ The guard now RUNS, and finding its doors required three attempts at the derivation

C113's escalation was a sentence — *"whoever runs that build must call `parity.filter_train_clips()`
first"* — i.e. **doctrine with no mechanism**, the C108 class. It is now `parity.py` **§10c
`guard_corpus_build()`**, wired into **six doors** and pinned by **24 tests whose population is
derived from source rather than hand-listed.**

⛔ **A hand-list would have shipped with TWO HOLES**, and neither appears in any runbook description
of "the build":
* **`rebuild_pai_rolling.py`** writes `ep_%05d.pt` through `mixing.save_episode` **directly**,
  bypassing `epcache.build_episodes_cached` — **gating the obvious writer would have missed it**;
* **`slice_v2_cache.py`** emits a whole **new** `.v2ep.pt` corpus **from an existing one** — no
  source video, no HF, no `--sel`.

⚠️ **AND THE DERIVER ITSELF UNDERCOUNTED THREE TIMES IN A ROW — 4 → 3 → 11 modules — as it was
tightened for precision.** ⇒ **ROOT-CAUSE CLASS: C110 again, one level up. A derivation rule is
itself a filter, so its output is a claim about the RULE until the rule is shown complete.** ⇒ The
committed rule therefore **deliberately OVER-reports into a census and justifies each exclusion**,
rather than narrowing until the list looks clean. *A short list is the shape both a precise rule and
a broken one produce.*

⭐ **The design hinges on a fact measured FIRST, not assumed:** parity-train digests (2400) ∩
deployed-val digests (40) = **0**. That is what lets the gate default to the dangerous-direction
check **while leaving the canonical train build untouched** — ⭐ **a guard that fired on the
legitimate case would be switched off within a week.** *Designing for the guard's own survival is
the difference between a control and a nuisance.*

⛔ **Two doors genuinely CANNOT be gated, and saying so is the right answer:** `epcache_to_pilot.py`
and `lake/view.py` key on **positional `ep_%05d`/`episode_id`, not clip ids**, so a gate there
**would miss every lookup and print a reassuring `0 contaminated` FOREVER.** Both classified with
reasons; one now says so at run time. *A guard that cannot see its subject is worse than none — it
manufactures assurance.*
⭐ **Gating the WRITER rather than the CALLER means `build_pai_cache.py` is gated without being
touched.**
⚠️ **And the test found a defect in the gate itself:** with the audit stamp placed **after** the
disjointness shortcut, `role="audit"` over a coincidentally-clean set returned
`decision_grade: True` — **the waiver silently did not apply.** Fixed.

### ⛔ NEW OPS HAZARD: a neuter-matrix killed mid-case LEAVES THE FILE BROKEN ON DISK

The C107 neuter matrix disables each guard in turn and requires the suite to go red — **all 8 went
red** (12/1/2/3/3/2/2/2 failures). ⚠️ **But the first matrix run was killed by a tool timeout
mid-case and left a source file NEUTERED ON DISK.** ⛔ **`try/finally` is no defence against
SIGKILL.**

⇒ **RULE: any test that deliberately damages a tracked file must be recoverable WITHOUT its own
cleanup path running** — take a pre-run backup and **md5-compare every touched file afterwards**,
which is what caught it here (**all 7 identical**). *A destructive test whose restore lives only in
`finally` is one timeout away from committing sabotage.*

### Status

**Suites:** `test_build_parity_guard.py` **24 passed**, `test_eval_contamination.py` **17 passed**
(both **in isolation** — the attributable evidence); `stack` **4084 passed / 0 failed**, `taniteval`
**1136 passed** *(full runs are a torn snapshot per C114 — a sibling ran pytest concurrently)*.
⭐ **No third plaintext copy of the ids: all 13 files scanned against the 4,729-id list, ZERO hits —
verified, not asserted.** C113's escalation #3 (the 3 `test_v6_stage_init_introduction` failures) is
**resolved**; that stream landed.

⏳ **Four items for the PI:**
1. **A re-run of the parity VAL cache build now REFUSES** until passed `--corpus-role val`
   (`…/2026-07-28-wide-val-build/code/launch_val_build.sh:30`). Intended; the banked launcher was
   **not** edited.
2. **The 4,472 build is now one flag** — `--exclude-parity-overlap` drops the 6 deployed-val
   episodes and records it in `_geometry.json`. **Without it, it refuses: there is no silent path.**
   ⚠️ The *other* half of `NEXT_4472_BUILD_INPUTS.md` §2 item 4 — **separate corpus vs parity
   extension** — remains a decision **no guard can make.**
3. **A bounded gap NAMED rather than closed:** the label stages write `{cid}.json`, too generic a
   shape to derive on. Safe **by position** (their clip set comes from the now-gated bridge) and
   `s2_labels` already refuses on `role="eval"`. Residual exposure is a **hand-assembled** label set
   or epcache; closing it needs records carrying their own provenance — a schema change.
4. **C113's plaintext id lists are now LOad-BEARING:** this work depends on them as test evidence,
   so **deleting them breaks the pin.** That raises the cost of the §9 "digests only" decision.

---

## C119 — "INTERACTION ENTROPY" IS MAXIMAL ON AN EMPTY ROAD, AND F-11's SPECIFIED HORIZON IS ARITHMETICALLY UNREACHABLE (2026-08-18, F-9/F-11)

### ⭐⭐ The curriculum would have driven training BACKWARDS

**MEASURED:** the naive reading of *"multi-agent kinematic entropy"* — bare spatial entropy over the
agent raster — reads **0.9649 on an EMPTY road against 0.2500 on a DENSE one. Inverted, 3.9×.**

⇒ ⛔ **An interaction curriculum built on it would have up-weighted EMPTY SCENES** — the exact
opposite of its stated purpose, and **it would have looked like it was working**: the score rises,
the sampler follows it, and nothing in any log says the scenes are getting emptier.

⇒ **ROOT-CAUSE CLASS: A PROXY WHOSE EXTREME IS THE DEGENERATE CASE.** Entropy is maximal at
*uniformity*, and an empty raster is perfectly uniform. **The metric was never wrong about entropy —
it was wrong about which end of it we wanted.** ⇒ **RULE: before a score drives sampling, EVALUATE
IT AT THE DEGENERATE INPUT.** *(Same family as C115's contrastive-over-an-invariant and F-8's
zero-at-init: a term that is silently satisfied by the thing you were trying to avoid.)*

The shipped mass-gated functional reads **0.0064 / 0.1863 / 0.4886** (empty / one agent / four
agents), and **an exactly-empty raster scores exactly 0.**

### ⛔ F-11: the mechanism is expressible; the SPECIFIED HORIZON is not — and this is arithmetic

`t_max = frames − window − max_horizon`, and a K-tick roll needs `max_horizon = K · stride_str`, so
windows/episode is **114 − 20K** on the 120-frame cache:

| K | horizon | cost |
|---|---|---|
| 4 | 8 s | **−64 % of windows** |
| 5 | 10 s | **−85 %** |
| **6** | **12 s** | ⛔ **ZERO windows** |

**The catalog asks for 4–15 ticks; only 4 and 5 exist. 30 s is longer than a 12 s episode.**
⭐ The **K=1 row (94 windows/ep) reproduces an independently MEASURED figure** in
`PI_DECISIONS_2026-08-12.md` §D4 — the input is corroborated, not assumed.
⛔ **And past `max_k` this is a PARITY BREAK, not an error:** short episodes drop to **zero windows
SILENTLY**. The guard therefore refuses on the shortest episode **and** on any non-zero drop-out
count. ⏳ **Diagram-owner/PI decision: amend the horizon band or change the cache.**

### ⭐ The C115 question was ANSWERED, not assumed

**The invariance is in the LATENT; the temporal structure F-11 needs is in the PREDICTOR.**
`predictor_str` is a genuine `z(t) → z(t+stride)` map, so a multi-tick roll is that map **composed
with itself** — the same shape `o5_rollout` already uses one layer down. **Nothing asks a latent to
integrate a window.** ⇒ Expressible, and built. *(The C115 brief was written expecting F-11 to be
the casualty; the sensitivity check moved the finding to F-9 instead. That is the check working.)*

### ⛔ F-9's own gate row is not computable today

T3's gate is *"P7 calibration ρ ≥ 0.3 held on **interaction-rich strata**, not just pooled"*. **Two
probes:** `w7_roll_rerank.py` has **zero** stratification support, and a repo-wide sweep across
`stack/` and `taniteval/` finds none. ⇒ **A per-stratum P7 is a PREREQUISITE for adjudicating any T3
arm** — a separate work item, and until it exists the cell cannot be gated even if trained.

### Construction facts

**Default build unchanged: 87,893,449 params / 405 keys** — measured at top and tail, and **by
building through the real `build_stack_from_args` path four ways** (default / F-9 / F-11 / both):
**delta (0, 0) every time.** Both cells are **structurally zero-parameter**, so the live
tensor-strict v6F resume is untouched.
⭐ **Neither needs an insertion point** — the F-8 case, not the F-7 case: zero keys means nothing for
`STAGE_MAY_INTRODUCE` to adjudicate, and `test_v6_stage_init_introduction.py` needed **no**
extending. **The `06b8782` class cannot apply** — neither touches `MODULE_GROUPS` or
`_GROUP_PREFIXES`, and a zero-parameter loss has no mechanism to move a stage's trainable set.

⚠️ **F-11's degenerate solution is the IDENTITY**, and the control is sharper than a threshold: on
slow-drift targets the HOLD rollout **beats** the untrained roll, and against **`z + noise` the
identity is BAYES-OPTIMAL and no predictor can win** ⇒ **the control is only informative once the
latent has learnable dynamics**, which is stated rather than assumed. Both controls **refuse below
n = 32.**
⚠️ **A defect in the agent's own wiring, caught and fixed:** F-9's curriculum refresh was placed
**after** the draw, so every step sampled under the **previous** step's exponent — **invisible in any
log.** Pinned against the source.
⚠️ **The full `stack` run was done TWICE and only the second counts** (4082 → **4084**): the first
predated the ordering fix. *Reporting the run that preceded your own fix is the torn-snapshot class
(C114) in a self-inflicted form.*

**Suites:** `stack` **4084 passed / 7 skipped / 2 xfailed**, `taniteval` **1136 passed**, new suites
**40 (F-9) + 29 (F-11)**. **Neither cell has been trained — no claim is made that either improves
anything.**

---

## C120 — `assert_isolation` MEASURES BACKWARD GRADIENTS; ADMISSIBILITY IS A FORWARD QUESTION — AND EVERY GOAL PORT IS `detach()`ed (2026-08-18, goal-provenance audit)

### ⭐⭐ The isolation guarantee is structurally blind to the ruling it is cited for

`V6Stack.assert_isolation` measures **backward gradient edges**. The PI's binding
goal/situation ruling is a **forward information** question. ⛔ **And `V6Stack.forward` routes EVERY
downward goal port through `_cut()` = `detach()`** (`v6.py:4341`, applied at `:4698-4700`, `:4750`,
`:4757`, `:4766`, `:4773`).

⇒ ⛔ **A situation-classifier output spliced in behind any of those leaks COMPLETELY and is
CERTIFIED CLEAN by the gradient probe.** **Demonstrated on a deliberately-wired arm, not argued:**
`forward_information_path = True`, `backward_gradient_path = False`.

⇒ **ROOT-CAUSE CLASS: A GUARANTEE STATED IN ONE DIRECTION AND RELIED ON IN THE OTHER.** `detach()`
is *designed* to sever gradients while passing values — so the very mechanism that makes the
isolation matrix meaningful is what makes it **blind here**. Nothing was broken; the check answers a
different question than the one the ruling asks. *Same family as `df` on a pod and `free` on Thor —
a correct instrument, wrong scope — but this one guards a **binding admissibility rule**.*

⇒ **The repair is an INTERVENTION probe** — replace a node's value mid-forward — which is
**detach-transparent** and distinguishes **`DIRECT_PATH` from `COMMON_ANCESTOR`**. ⭐ **A
correlational test cannot make that distinction, and it is exactly the shared-trunk disclosure the
ruling demands.**

### Measured, per arm (6 `V6Config` arms) — the answers are good

* **Every goal head is a function of `frames` alone.** `actions` reaches **no** goal node; `v0`
  reaches **exactly one** — the unicycle `emission`. ⭐ **This VERIFIES BY COMPUTATION the prose
  claim at `v6.py:62`** that `v0` *"enters ONLY the unicycle"*. One allowlisted edge, **pinned
  against widening**.
* ⭐ **The REVERSE direction — never checked before — is CLEAN.** The situation classifier's entire
  input is `sub["img"]` (`sitclf_train.py:160`, re-read first-hand, not inherited). It is an offline
  scorer with **zero imports** in `models/` or any trainer.
* **Shared trunk: a shared CLASS, not a shared INSTANCE.** The situation substrate comes from a
  **frozen v1 checkpoint's** encoder; v6 trains its own. **Common ancestor, no path** — which is the
  disclosure the ruling asks for, stated rather than assumed.
* ⚠️ **`tac-goal-cond` is UNPOWERED** — `cond_tac_dyn` is zero-init, so nothing can move it, and
  **the gate exits 3 rather than calling that clean.** *Refusing to certify an untestable arm is the
  C79 lesson applied prospectively.*

### ⚠️ THE INSTRUMENT ALREADY EXISTED, WITH ZERO CALL SITES

`stack/tanitad/eval/goal_admissibility.py` — **242 lines, 12 tests, flagged 12 days ago in the
stale-blocker sweep, never called.** ⛔ **And its provenance clause intersected two HAND-WRITTEN
LISTS OF SYMBOL NAMES** — the C112/C113 class exactly: **a non-overlap assumed from provenance
rather than computed.** ⇒ The gap was real; it was **not where the brief placed it**. *C116's shape
again — a correct component that nothing calls — twice in two days.*

### ⚠️ Two defects in the agent's own probe, both caught and pinned

⛔ **The input probe SHIPPED INERT** — it perturbed the input and then ran the model on the
**unperturbed** batch. ⭐ **It was caught by an IMPLAUSIBLE READING (a vision-derived goal path that
did not depend on vision), NOT by the positive control** — which covered only the other code path.
⇒ **ROOT-CAUSE CLASS: A CONTROL POWERS ONE CODE PATH, NOT THE MODULE.** *C107's per-arm lesson at
finer grain: a positive control is scoped to the branch it exercises.*
⚠️ **The dead-node rule called DISJOINTNESS AN ARTEFACT** — in a genuinely disjoint pair neither node
moves the other **by definition**, so both read as "dead". ⇒ **Deadness must be judged against a
source that SHOULD move the node.**
⚠️ And `stack/scripts/goal_provenance.py` **already existed** (a different question — goal *source*).
The agent probed `tanitad/eval/`, concluded the name was free, **and says so: the
absence-at-one-location rule, broken and reported.**

### ⛔ A NEW MECHANISM FOR THE EXIT-CODE TRAP — the PIPE's status, not pytest's

A suite run reported **"exited with code 0" WHILE A TEST FAILED**, because the reported status was
**the pipe's**, not `pytest`'s. *(Tenth instance this session, and a mechanism distinct from every
earlier one — bad flag, collision error, 0-byte output, `printf` truncation, MSYS rewriting.)*
⚠️ **STATUS AT THE TIME: the full suite had not finished**, and the agent explicitly refused to
upgrade the claim on a run it had not seen end — correctly.

✅ **RESOLVED SHORTLY AFTER: `TRUE_PYTEST_RC=0` — 4112 passed, 7 skipped, 2 xfailed (522 s)**, green
against the live worktree, contention and all.

⛔ **AND SETTLING IT CORRECTED TWO MISREADINGS, ONE OF WHICH I HAD ALREADY WRITTEN INTO THIS ENTRY.**
The **"839 passed / 1 failed"** I recorded above **was never a suite size**: that run used **`-x`**,
so 839 is simply **where it stopped at the first failure**. The real suite is **4112** — ⇒ **quoting
it as a total would have understated coverage by ~5×.**
⇒ **ROOT-CAUSE CLASS: A COUNT READ OFF A RUN THAT WAS CONFIGURED TO STOP EARLY.** `-x` makes the
number a statement about *when the run halted*, not about the suite. **C110's "a count is a claim
about the filter" — here the filter is a FLAG rather than a file-type**, and it is invisible in the
output line itself.
⚠️ The same run's wrapper reported **"exited with code 0" while a test had failed** because the
pipeline's status was **`tail`'s**, not pytest's. ⇒ The final run **wrote pytest's own `$?` to the
log BEFORE any pipe** — which is the general fix for this whole family.
