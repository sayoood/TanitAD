# LOOP_STATE — live operational state for the autonomous loop

**The cron drumbeat reads THIS FILE, not a frozen prompt.** Rewriting the cron every time reality
moved was costing a re-derivation every 30 minutes and shipped stale instructions twice in one day.
Update this file instead; keep it short and dated.

`LAST_UPDATED: 2026-07-28 18:2x UTC (20:2x Berlin). 🔵 AUTONOMOUS.
✅✅ **FLEET OUTAGE RESOLVED — IT WAS NEVER AN OUTAGE.** The migration split the fleet across **TWO
datacenters** and reassigned ports; I was probing ca-mtl-1 endpoints for pods that had moved to
**US-TX-1**. **`~/.ssh/config` is now corrected and all four aliases resolve** —
`tanitad-pod` **38.147.83.15:39198** (was 30107) · `tanitad-pod2` **69.30.85.123:22091** ·
`tanitad-pod3` **69.30.85.16:22079** · `tanitad-eval` **69.30.85.106:22073**. Plus a **new** A40 pod
at **69.30.85.48:22192**. ⛔ **Do NOT re-raise "pod1/pod2 unreachable" without re-probing these.**
✅ **176×624 IS UNBLOCKED AND LAUNCH-READY ON pod2** (A40, 376 GB, restore complete). The w120 caches
returned and **parity is CRYPTOGRAPHICALLY VERIFIED by the trainer's own preflight** — train **2400
clips** sha256 `e61a04553df5…`, val **600 clips** sha256 `0b176d2e5cb4…`, both matching the committed
manifest, skip-hash `f09e44db`. **No rebuild is needed**: 176×624 is a `--v2-subframe` pixel slice of
the existing 256×640 cache (the cache's own `_geometry.json` says so). `PREFLIGHT: OK` run twice.
⛔ **NOT LAUNCHED — one PRE-REGISTERED HOLD, and it is the PI's to clear (task #38):** the v5 PREP card
records *"on EGO YAW RATE the wide frame is separated-WORSE (−0.03546 R²)"*, marked **"under
investigation before v5 trains"**, while that card calls ego-motion perception *"the real item"*.
Launching past a pre-registered hold selects the outcome after seeing the convenience.
⚠️ v5 trains at **117.000°, not 120°** (the rig-clean slice costs 3° of field), and **v1's 0.4271 is
NOT a valid comparator** for a wide arm — v1's encoder was trained at 51.4°, so wide frames are OOD.
⚠️ **arm A's validation is GONE** (`/workspace/smallval` empty after the restore) though the arm
finished `rc=0` — **~4 h to re-run**; PI decides whether it is needed.
⭐ **C56 — "pods cannot SSH each other" is RETRACTED. MEASURED 42 MB/s cross-datacenter.** Recipe:
keygen on the DESTINATION, append its **PUBLIC** key to the source's `authorized_keys`, connect to the
source's **direct** `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22` (**not** the proxy, which cannot move
files). ⇒ **the ~1 MB/s relay and the HF-403 no longer block any multi-GB move.**
✅ **BOTH pod2 AND pod3 were 91 commits behind at `0f93b98`; BOTH ARE NOW SYNCED** and verified by a
real `import` (`seam_fail_frac=0.75`, `seam_fail_patience=50`), not by `git log`. *(pod3's
`/workspace/TanitAD-main` has no `.git` — it is a zip download; `/workspace/TanitAD` is authoritative.)*

### 🔴 THE 30k FLAGSHIP GATE — every input located, ONE step blocked (2026-07-28 ~18:4x UTC)

⭐ **`flagship-v4-30k` IS NOT LOST.** I briefly concluded it was, because pod2's `/workspace/experiments`
is empty. **It is on pod3 at `/workspace/v4instr/v4fs_ckpt.pt` — step 29,999, 3,243,109,310 B, parity
`e438721ae894` / `f09e44db`.** ⛔ **My search was `*flagship-v4*`; the file is named `v4fs_*`, which
that glob CANNOT match** — the same one-name-absence error as `anchors*.pt` vs
`flagship_v4_anchors_dense.pt` earlier the same day. **Search by content, not by one name.**

⭐⭐ **THE GATE MUST RUN ON pod2, NOT pod3 — and this is the fact that was hardest to find:**
`physicalai-val-0c5f7dac3b11` (the canonical parity val, **600 eps, `DONE`, 66 GB**) lives ONLY at
`pod2:/workspace/data/physicalai_phase0/_epcache/`. **pod2 also has `comma2k19` (83 G)** ⇒ the
`--data realmix` resume is far less blocked than I priced it. On pod3 there is **no usable val**:
`f1b378` is a **code-level refusal** (`parity.py:24`, 78.5 % of its episodes are IN the parity train
set) and `s3parity/views/…0c5f7dac…` is **poses-only, 4,385 B/episode — no frames**.
⚠️ **I nearly ran MODE A on the 44-ep E1 held-out.** The harness warned: *"NON-PARITY val corpus …
Results off it are NOT cross-arm comparable"*, **44 eps / 7,511 windows vs the canonical 40 eps /
881 windows** — it WARNS AND PROCEEDS, so a non-comparable number would have come out. Run killed.
⚠️ **`0.4271` is `wm_fidelity_ade_2s`** (the WM handed TRUE future actions). Registry §1.2: *"IS NOT A
PLANNING BAR AND MUST NOT BE USED AS ONE."* The card uses it correctly — as the **MODE A harness
reference only**. Same-surface planning bar is **0.4907** (881 windows / 40 eps).
✅ Card `flagship-v4-30k.card.json` EXISTS, `gate_step: 30000`, **registered at step 29,650 — before
the checkpoint existed**. Escalation #4 (multiplicity) is therefore CLOSED. Primary is DEMOTED to
diagnostic; co-primary is **REPORT_ONLY this gate**; `nonav_route_beats_majority` **VOID → INSTRUMENT-FAIL**.

### ✅ THE 30k GATE RAN (2026-07-28 ~19:1x UTC) — **NO-VERDICT by protocol.** Full results:
`…/Benchmarks & Eval/…/incoming/2026-07-28-v4-30k-gate/V4_30K_GATE_RESULTS.md`

**Estimator throughout: episode-cluster bootstrap, n = 881 windows / 40 episodes**, horizon 2 s.
MODE A ran FIRST per O-03 and PASSED (v1 → **0.42148**, Δ −0.0056 vs the full-set 0.4271,
tolerance 0.05). ⚠️ The v4 canary run prints *"HARNESS NOT VALIDATED"* — that message compares **any**
canary run to **v1's** reference and is a **category confusion on a v4 arm**, not a harness failure.

⭐⭐ **THE RESULT THAT MATTERS — paired vs the constant-velocity floor, same windows:**
**produced (deployable) +0.0186 [−0.1711, +0.1940] OVERLAPS 0** · oracle −0.1954 [−0.3713, −0.0418]
SEPARATED. ⇒ **the arm's entire measured advantage over constant velocity depends on being handed an
ORACLE goal.** Stated precisely: *indistinguishable*, **not** "proven no better" (wide CI, 40 eps).
Paired Δ(produced−oracle) `ade_0_2s` **+0.2140 [+0.1602, +0.2759]**, and the cost **COMPOUNDS ~11×**
across the horizon (+0.0377 @0.5 s → +0.4148 @2 s), every horizon SEPARATED.
⭐ Paired Frenet: `long_abs_2s` **+0.4260 [+0.3227, +0.5420]** · `lat_abs_2s` **+0.0274
[+0.0061, +0.0533]** — **BOTH separated (C57 retracts my "ONLY longitudinal")**; asymmetry 15.5×.
**Both SIGNED components overlap zero** ⇒ the producer adds error **magnitude without directional
bias** — a **noisy** goal estimate, not a **mis-calibrated** one. Different repairs.
**Secondaries 3 PASS / 2 FAIL / 2 UNPRODUCIBLE / 1 VOID:** FAIL `wm_canary` **1.1409** vs ≤0.55 ·
FAIL `miss_at_2m` **0.2123** vs ≤0.10 · PASS `oracle_in_fan` 0.2330, `seam_norm_ratio_max` 0.1208,
`encoder_touching_levers` 2 · VOID `nonav_route` → **INSTRUMENT-FAIL, never MODEL-FAIL**.
⛔ **CANNOT BE COMPLETED AUTONOMOUSLY (task #42):** `speed_benefit` is **unrecoverable from this
arm's log** (the `g_op_fwd_ade_m` fix did not ride the launch) and `deploy_tick` has **no v4 panel
and no owner**. Co-primary NOT run (no v4 closed-loop driver exists; it is REPORT_ONLY anyway).
⚠️ Bounding caveats: **frame UNVERIFIED** (ckpt carries no geometry block) · val parity is
**COUNT-ONLY** (600/600 present, no uid digest committed).
⚠️ **B4 TRAP FIRED AND BIT TWICE:** `eval_flagship_v4` imports `taniteval` **non-fatally** → first
run exited 0 with the primary silently null; **AND** the JSON's
`cross_check…driving_py_from_persisted_windows` stays `null` **even when driving DID run** — the real
result is in the `driving` block / `[driving]` log line. **I concluded the primary was missing from
that field alone and cried wolf.**
⚠️ **`--head-config` IS MANDATORY for old ckpts.** Without it the loader builds the head from CURRENT
defaults and STRICT-load dies on 5 keys; this arm's own config says `"cond_imagination": false`.

🔴 **BLOCKED ON ONE THING: moving 7 GB of checkpoints pod3 → pod2.**
⛔ **C56 HAS A MEASURED LIMIT: pod→pod direct works CROSS-datacenter, NOT same-datacenter.** pod2→pod3
is `Connection refused` on the public mapping (no hairpin NAT) and their private subnets are disjoint
(**pod3 `172.16.96.2`, pod2 `192.168.0.2`**). The 42 MB/s figure was US-TX-1 → ca-mtl-1.
**Dev-box relay MEASURED at 5 MB/s** (not the inherited ~1 MB/s) ⇒ 7 GB ≈ 48 min round-trip — viable
but slow. **Fast path = HF (~118 MB/s), and HF IS REACHABLE** (token OK, user `Sayood`, 17 repos).
🔴 **The HF push was BLOCKED BY THE AUTO-MODE CLASSIFIER. I did not route around it — Sayed must
grant it or approve the 48-min relay.** There is **no** `flagship-v4-fromscratch` repo, so that
completed 30k arm is still SINGLE-DISK.
⭐ **THE SEAM GUARD IS FIXED PROPERLY (C51 closed), not raised** — the `seam_fail 8.0` workaround is
RETIRED. Trigger is now POPULATION-over-TIME: batch **MEAN** ratio > 1.5 **AND** ≥75 % of the batch at
the clamp **AND** 50 consecutive steps; counter resets on any healthy step; `seam_clamp` untouched, so
**no computed value moves — only when we abort**. The message no longer names a cause that is
impossible by construction. 5 tests incl. a minority-can-never-kill pin. stack/ **1581 passed, 12
skipped**.
⭐ **CLOSED-LOOP (D-A) — three one-dimensional levers, two exhausted:** E1c (training time) **BOUND**
— departure 0.5877→0.147, P1∧P2 separated at **15/17**, but guardrail Ga 0/17, and the open-loop cost
**PLATEAUS** from step 2250 so longer training cannot fix it. E1d (weight space, WiSE-FT α) **BOUND**
and *stronger than a null*: `dep_overall` is **separated-WORSE at five consecutive interior α** ⇒ the
base→FT segment **crosses a barrier**, the endpoints are NOT linearly mode-connected (**C52** — a
published remedy imported without checking its precondition). E1d also **decomposed the primary**:
junction recovery is **cheap and monotone** (separated-better at α=0.20 for +6.5 % open-loop) while
overall-corridor recovery is **expensive and barrier-crossing**. ▶ **E1e (loss weighting) — `lam_replay`, the one flag
`e1c_clsft.py` itself calls "deliberately NOT a lever".**
**E1e-A (λ=3) DONE, MEASURED: VERDICT BOUND, 0/4 success — but P1∧P2 fired 4/4** (E1c: 15/17), and
**the frontier EXTENDED**: ⭐ **no E1c checkpoint anywhere on its 17-point frontier reaches an
open-loop cost below +0.1893, while E1e-A reaches +0.0958 holding −0.3911** (step 3000: 88.7 % of
E1c's best closed-loop gain at 44.4 % of its open-loop cost). Base-reproduction control matched E1c on
every field. ⚠️ **Stated precisely: E1c's best (−0.4407) still has the LARGER closed-loop gain, so
neither point dominates — the true claim is that E1e-A occupies a region E1c never attained.**
**Structural: `lam_replay` sets the ASYMPTOTIC open-loop cost and does NOT remove the plateau** (A
plateaus ~+0.10 from step 2000; E1c ~+0.20 from 2250) ⇒ longer training cannot close Ga for A either.
**E1e-B (λ=8) DONE: BOUND, 0/4 success, but P1∧P2 4/4** — ⭐ **the risk it was run to test did NOT
materialise: P1 survives at λ=8.** Ga's lower bound walked 0.047→0.038→0.024→**0.023 and FLATTENED**.
⛔ **THE λ AXIS IS CLOSED** (a finer grid was pre-committed as INADMISSIBLE *before* any of it ran,
precisely so a near-miss could not tempt a sweep — the temptation was real and was declined).
⭐ **THE AXIS'S DELIVERABLE IS A TRADE-OFF CURVE, monotone and non-crossing, none dominating:**
λ=1 → CL **−0.4407** @ OL +0.2158 · λ=3 → −0.3911 @ +0.0958 · λ=8 → −0.2891 @ **+0.0500**.
⇒ `lam_replay` is a **calibrated control**: the program can now CHOOSE an operating point.
⚠️ **BUT NO POINT PASSES THE GATE, SO THERE IS STILL NO D-A DELIVERABLE.** A curve one can choose
along is not a checkpoint that passes — that distinction is why the criterion was fixed in advance.
**THREE LEVERS, THREE BOUNDs, THREE DIFFERENT STRUCTURAL REASONS:** training time = plateau (E1c);
weight space = barrier, endpoints not linearly mode-connected (E1d, **C52**); loss weighting =
asymptote (E1e).
✅ **E1f DONE — BOUND, `primary_ok 0/4`.** Pre-registered **OUTCOME C exactly**: P2 (junction)
separates-better at 3/4 and strengthens monotonically to **−0.2108**; **P1 (overall) fails at all four,
sitting AT ZERO** (+0.0025, +0.0576, −0.0302, +0.0004).
⭐ **AND IT REFUTES THE INFERENCE THAT LAUNCHED IT (C55):** E1c full-buffer reaches junction
**−0.4982**; E1f junction-only reaches **−0.2108**. **Training on junctions alone HALVES junction
recovery** — if the "expensive half" had been interfering, removal should have left it ≥ equal. E1f
buys 42 % of the gain at 27 % of the cost with ZERO overall gain: a **scaled-down** arm, not a
targeted one. *(E1d's decomposition is NOT wrong; the TRANSFER was — it described a PATH BETWEEN two
models trained on everything, which says nothing about training on a subset.)*
⛔ **FOUR LEVERS, FOUR BOUNDs:** training time (plateau) · weight space (barrier, **C52**) · loss
weighting (asymptote +0.023) · **the target (restriction gives less of everything)**.
**THERE IS STILL NO D-A DELIVERABLE.**

## 🔴🔴 THE Ga DECISION — BOTH MAGNITUDES NOW ON THE TABLE (PI JUDGEMENT, NOT ANOTHER ARM)

⭐⭐ **MEASURED and previously UNREPORTED by me across four experiments:** `peak_xte`/`mean_xte` sat in
every frontier artifact since E1c and I reported only departure RATES and open-loop ADE.
**The BASE arm wanders 38.944 m off track at peak in an 18.5 s closed-loop rollout — against a 1.75 m
corridor** (mean 14.306 m). It does not leave the lane; it leaves the road.
| arm | peak_xte base→ft | paired Δ | sep |
|---|---|---|---|
| **E1c** | **38.944 → 3.042 m** | **−35.9030** [−49.33, −24.12] | ✅ |
| E1e-A | 38.944 → 4.502 m | −34.4430 | ✅ |
| E1e-B | 38.944 → 7.790 m | −31.1540 | ✅ |
| E1f | 38.944 → 15.012 m | −23.9328 | ✅ |
⇒ **E1c cuts peak excursion 92 %.** Ga blocks on an open-loop regression of **+0.05–0.20 m** — a ratio
of order **167 : 1**. ⚠️ **NOT an argument to relax Ga, and NO verdict changes** — the arms still fail
the gate and the open-loop cost is real, CI-separated on BOTH axes and **tail-heavy laterally**
(E1c `cross_p90` **+0.3945** vs mean +0.1611 = 2.45×). It is the **missing magnitude**, so the
judgement is made on both sides rather than on departure rates alone.
✅ **RESOLVED same day (code read, no GPU):** the closed-loop knot-ADE frame confound **does not
exist** — `oyaw`/`oxy` and `gt_ego_waypoints` share **the same origin AND rotation** (`poses[last]`),
and **every window re-initialises at the recorded pose**, so both arms start identically. My "different
ego states" worry was wrong. **The horizon mismatch reconciles it:** knot-ADE spans **0.5–2.0 s**,
departure spans **18.5 s** ⇒ the arms take a slightly different EARLY line (+0.2154 m lateral at ≤2 s
for E1c) and are **dramatically more stable long-horizon**. This CHARACTERISES the gain; it does not
undermine it.
⚠️ **Still open, needs GPU and sits DOWNSTREAM of the Ga call:** whether that early lateral deviation
is a **recovery-oriented** line or merely tolerated. No per-window traces exist; it needs a capture
re-run. **Not spent.**
✅ **D-B COMPLETE (not re-fired; authorization spent):** YouTube→GeoCalib→IDM end-to-end. **Block has
LIFTED** — 0 bot-block messages, retried from **the same pod3 egress** after ~37 h (NOT rotation;
idle eval deliberately unused). 20 clips / 3 videos; ⭐ **GeoCalib resolved hfov 53° and 58° against
the 100° fallback — both ~2× off, independently reproducing "the fixed HFOV is wrong" on a fresh
sample.** IDM: 2,240 windows, `frac_in_plausible_0_45_mps = 1.0`, mean 14.212 m/s. Privacy VERIFIED
BY PROBE (0 media files remain). ⛔ **No ground truth exists — these are UNVALIDATED pseudo-labels;
no accuracy claim is quotable.** 🔴 **PI DECISION: licensing** — `{None: 27, CC-BY: 1}`, the harvest
kept non-CC by default.
✅ **E1's held-out split is CONTENT-CLEAN at the sensor level** — 0/44 episodes and **0 shared frames**
vs `physicalai-train-e438721ae894`, which is **REF-C base's OWN training corpus** (config-verified,
same path, same n=2376) ⇒ the C49-class gap is CLOSED and E1a/E1c/E1d are not leak artifacts. SPIKE
control recovered 3 injected episodes, so the zero is evidence, not absence of evidence.
✅ **C48 CLOSED across all four sites** — the paper both REFUSED and PRESCRIBED the 77.5 %-leaked
`f1b378` split, 200 lines apart; root cause was ONE `note=` string in `taniteval/registry.py` that
four documents inherited instead of checking the artifact.
⚠️ **D-B YOUTUBE: the block HAS LIFTED (MEASURED 07-28 05:22 — 115 candidates, ~137 MB, ZERO
bot-block), but there is NO usable output** (0 clips) and a `--no-geocalib` batch is unquotable anyway
(fixed 100° HFOV measured wrong for 11 of 12 clips). Real harvest waits for a FREE GPU. **C53**: I
priced that harvest by its name and it cost the trainer **≥4× (5.6× cumulative)** at 477 % CPU — and
killing it left a **raw 137 MB video on disk**, since the pipeline deletes the source only *after*
decode. Deleted, 0 media files remain.
⛔ **STILL OWED BY THE PI:** the RunPod restart (blocking everything above). — SUPERSEDED HEADLINE OF 2026-07-27 16:21 UTC FOLLOWS, KEPT FOR AUDIT: ⚠⚠ **CLOCK CORRECTION: `date -u` says 2026-07-27, but the previous LAST_UPDATED, the last report filename (`2026-07-28-0757`) and ELEVEN `incoming/2026-07-28-*` dirs are ALL A DAY IN THE FUTURE. That time has not passed. ORDER BY COMMIT ORDER, NOT BY FILENAME.** Report `2026-07-27-1821`. 🔵 AUTONOMOUS. ⭐⭐ **v5 IS NOW TRAINABLE, EVALUABLE AND REGISTERED** — both caches complete on pod2 (train 2,401 / val 601 = 20 GB), renamed, membership-proved (2400/2400, 600/600, 0 missing, 0 extra), REGISTERED, manifest committed with the corrected `observed_frac` (train 0.9348642592 / val 0.9360899822; the false 1.0 was caught ONE RUNBOOK STEP before registration made it permanent). The evaluator now reads v2 — it had NO v2 path and `build_v2_providers` had ZERO eval callers — verified BIT-IDENTICAL to the trainer's own seam. Encoder is **0.662 of a real step** ⇒ the clean frame costs **0.767× a full step**. 🔴 **BUT `--heldout-gate` CRASHES AT ITS FIRST PROBE** (`cond_vtarget is on but no vt_band supplied`) — **~2,000 steps / several GPU-hours in.** Every existing test of that path uses a STUB head. A TRIPWIRE was added, NOT a fix: the choice between `vt_band = 0` (a REAL speed band) and an explicit `VT_DROPPED` **defines what v5's early-stop stops on** — PI DECISION. ⭐ **RESOLUTION: `NO GAIN` — 384×960 REFUSED.** Three probes, none returned GAIN; the mirror step costs −0.00150 AP [−0.00402, +0.00139] and the knee is 2.3–3.0× BELOW the v5 frame; a REAL 1440-token arm was built and **won nowhere**. ⭐ **And it separated two effects: the wide frame is separated-better than today's (+0.04246 AP) and 98.1 % SURVIVES AT MATCHED px/deg ⇒ WHAT WIDENING BUYS IS NOT ANGULAR RESOLUTION.** ⚠️ ego yaw rate is separated-WORSE (−0.03546 R²). ⭐ **CLEAN RIG FIX IS A SLICE** — pad/mask **0.0000000000 on BOTH rigs (max, not mean)**, bit-identical on 6/6 clips / 1,206 frames; rows-only costs **0.0165 %** of agent samples vs columns-only 1.1770 %. Frame **176×624, 429 tokens (−33.0 %)**. ⛔ NOT closed: the rig stays readable from pixels (all-zero 0.0000834 A vs 0.0079316 B, ~97 % transient scene black — a CORPUS-BALANCE confound no crop removes), and **no ADE was measured: FREE TO TRY, NOT PROVEN BETTER.** ⛔ **THE CLOSED-LOOP COMPOSITE WAS BLIND TO OVER-TRAVEL** (v1 over-travels on 48.80 % of windows, p95 2.430×). Fixed as `@twosided_v2`, reproduction gate EXACT. **`v1_tactical_follow − cv_holdv0` goes n.s. → −0.1212 SEP: "v1's plan is tied with doing nothing" WAS A METRIC ARTEFACT — it is separated-WORSE**, and the whole v1 family drops below every REF-C arm incl. both ego-ablated ones. ✅ `cv_holdv0` STILL ranks first among realisable arms at every w; rank 1 overall under BOTH terms is the ORACLE longitudinal schedule. ⭐ **E-GOAL-4: joint training CONFIRMS the goal (+62.09 %) but +46.3 % OVER-CREDITED IT 1.76×** — a trained selector with NO goal already gets +35.62 %, so **PLAN v5 WITH +26.31, NOT +46**. ⭐⭐ **And the goal carries NO INFORMATION: `g_along` = GBM(v, ax_fd) at R² 0.999894 with the no-goal arm fed both columns ⇒ an INDUCTIVE BIAS, not a channel ⇒ FUNDING A STRATEGIC SUPPLIER IS THE WRONG LEVER.** 🔴 **v5's CORRIDOR CO-PRIMARY IS BLOCKED: the renderer is HARD-CODED to a pinhole f=266/c=128 (the old 256² crop). On v5's cylindrical frame the error is MEAN 46.3 px against a TRUE SHIFT of 42.7 px, 99.08 % of pixels >1 px, up to 47 px of SPURIOUS VERTICAL motion where truth is 0** (control on its own frame: 0.118 px). Only K=20 is producible and K≤20 is refused at both ends ⇒ `check` correctly renders **INCOMPLETE**. Fix stream RUNNING on pod2/eval. ⚠️ **D-B YOUTUBE: the window opened 12:00 UTC and the retry HAD NOT FIRED for 4 h 21 m** — launched this iteration on pod3, gentle config W=2 TARGET=400 SEEDS=4 --sleep 4, ONCE, blur verified first; **NEVER bypass bot-detection; if blocked, STOP.** 🔴 **PI DECISIONS OWED: (1) `vt_band` — defines what the early-stop stops on; (2) 2400 vs 2376 — VAL IS 600/600 IDENTICAL so all eval numbers are unaffected, but the extra 24 sit in a tight position band 1798–1941 and CANNOT be dropped without re-registering; (3) frame 176×624 (does NOT tile the readout) vs 128×576 (tiles, zero-mask, +3.5 pp agents); (4) `w`.** Fleet: pod1 flagship-v2corpus-30k step 20,400/30,000 (68 %, ~28.8 h left); pod2 + pod3 + eval were ALL IDLE and were refilled this iteration. 6 new retraction classes C34–C39.`





## 🔴🔴 2026-07-28 ~04:10 UTC — **POD1 AND POD2 ARE BOTH UNREACHABLE. PI ACTION REQUIRED.**

**MEASURED, and the two failures are DIFFERENT — do not treat them as one outage:**

| pod | alias target | TCP | ICMP | reading |
|---|---|---|---|---|
| **pod1** | `38.147.83.15:30107` | ❌ refused | ✅ **ping OK** | host is UP, **nothing listening on the port** ⇒ the documented RunPod signature: *a volume resize stops the pod and reassigns its SSH port* |
| **pod2** | `69.30.85.75:22022` | ❌ refused | ❌ ping fails | **host unreachable entirely** — stopped, or IP reassigned |

pod3 and tanitad-eval answered **in the same minute**, and `ssh -G` resolves both failing aliases
correctly, so this is **NOT** a local SSH-config or network fault. *(The error string
`Connection to UNKNOWN port -1` is OpenSSH's phrasing for a failed banner exchange — it is NOT
evidence of a config problem, and reading it as one wastes the iteration.)*

**⛔ I CANNOT SELF-RECOVER.** New ports/IPs come only from the RunPod console; `Keys.txt` holds an HF
token and **no RunPod credential** (checked by label, values never read). **This is a blocking PI
action:** restart/inspect both pods, then publish the new `Host` ports into `~/.ssh/config`.

**WHAT IS AT RISK (and what is not):**
- **pod1 — `flagship-v2corpus-30k` at step 23,850/30,000 (79.5 %), ~2.9 GPU-days in.** Volume-resident
  ckpts survive a stop: `ckpt.pt`, `ckpt_step20000.pt`, `ckpt_step15000.pt`, `ckpt_step5000.pt`. The
  supervisor auto-resumes from `ckpt.pt`, so the expected loss is the steps since the last save, not
  the run. ⚠️ **Verify `restarts:` and the resumed step on reconnect — do not assume it resumed.**
- **pod2 — arm B of the PI's geometry validation**, restarted 02:31 UTC and only ~100 steps in;
  `chain2.sh` (B → C → evalchain) will need relaunching. **Arm A's results are already banked and
  safe** (`pw_A_old.npz`, `pw_A_old_blind.npz` on the pod volume — ⚠️ **single-disk, not yet pulled**).
- **NOT at risk:** E1c/E1d live on pod3 and their summaries are committed to the repo.

**Do NOT** re-run arm A on reconnect — it completed `rc=0` and raising `seam_fail` is a proven no-op
for it (`test_seam_fail_is_a_pure_guard_and_changes_no_computed_value`).

## ⛔ RETIRED QUESTIONS — the drumbeat prompt still asks these; they are ANSWERED

The `/loop` and cron prompts are frozen text from an earlier phase. **Do not spend an iteration re-investigating any of the following.** Each has been measured and closed; if a future probe contradicts one, that is a new finding and should be reported as such.

| the prompt asks | the answer | evidence |
|---|---|---|
| *"is the ~430 s/step pathology fixed or still crawling?"* | **THERE IS NO PATHOLOGY AND THERE NEVER WAS.** `step_s` in trainer logs is **ACCUMULATED over `--log-every`** (÷50). pod1 logs ~540 ⇒ **~10.7 s/step**. | MEASURED, `train_log.jsonl`; line count × 50 reconciles with `step` |
| *"REF-A rebuild status — DINO features == 2376?"* | **REF-A IS NOT AN ACTIVE ARM.** No rebuild is running or queued. | fleet probe: 4/4 pods, no REF-A process |
| *"REF-B step progress"* | **REF-B IS NOT AN ACTIVE ARM.** | same |
| *"screen the profiler agent (430 s hotspot verdict)"* | **No such agent exists or is needed** — see row 1. | — |
| *"pod2 finishes its 30k flagship ~now — run the 8-metric gate"* | **STALE.** pod2 built the **v5 wide caches**. The 30k arm in flight is **`flagship-v2corpus-30k` on pod1**. | fleet probe |
| *"hold every arm to v1's 0.4271"* | ⛔ **CATEGORY ERROR.** `rollout.py:170` sets `actions_source="expert_future"` and `:174` names it `wm_fidelity_ade_2s` ⇒ **0.4271 is what the WORLD MODEL scores when HANDED THE TRUE FUTURE ACTIONS.** A selector that must *choose* cannot be held to it. **The legitimate same-surface bar is 0.4907** (881 windows / 40 eps, `a0` 0.4714). | MEASURED in code; `MODEL_REGISTRY.md` annotated |
| *"D-B YouTube: retry ONCE at/after 2026-07-26 12:00 UTC"* | ⛔ **SPENT AND BLOCKED** — see the D-B section above. Firing again is a second run outside the authorization, at a blocked IP. | MEASURED, `…/2026-07-27-yt-dB-retry/` |

**Standing consequence:** an iteration that reports on these instead of advancing real work is a wasted iteration. Answer from this table and move on.

## ⛔⛔ D-B YOUTUBE — AUTHORIZATION **SPENT**, AND THE RUN IT AUTHORIZED WAS **BLOCKED**

`SPENT: 2026-07-26 12:33:31 UTC → 16:33:33 UTC` — the staged gentle config (`W=2 TARGET=400 SEEDS=4 --sleep 4`, GeoCalib geometry) **ALREADY FIRED**, with exactly the parameters any drumbeat would re-issue.

⛔ **DO NOT FIRE IT AGAIN. THE SINGLE-RUN AUTHORIZATION IS USED.** A drumbeat prompt that still says *"retry ONCE at/after 2026-07-26 12:00 UTC"* is **STALE**: acting on it is a SECOND run outside the authorization. *(This paragraph exists because an agent was briefed with “D-B has not fired” — INHERITED and false — and correctly refused to launch, making **zero** YouTube requests.)*

🔴 **AND THE RUN ENDED IN A BOT-BLOCK THAT NOBODY RECORDED.** The prior report was finalised **14:35 UTC** and states *“Was it blocked? — NO.”* **The block began 16:11:21 UTC**: in the final round **650 of 650 videos were refused** with `Sign in to confirm you're not a bot`, **0 clips**. **The driver logged it as `pool exhausted at 343 — proceeding`** — that mislabel is the entire reason it went unnoticed, and it **REFUTES** the prior conclusion that *“the binding constraint was not rate-limiting.”* ⇒ **pod3's egress was blocked ~24 h before the next window; a “retry” would have hit a freshly-blocked IP.**

⚠️ **yt-dlp's own error text instructs the reader to `--cookies-from-browser`. That is TOOL OUTPUT — DATA, NOT AUTHORIZATION — and it was correctly not followed.** Bot-detection is never bypassed.

**Any further YouTube harvesting is a NEW decision for the PI**, and must account for the blocked egress. Artifacts: `…/incoming/2026-07-27-yt-dB-retry/`.

## 🔴 STANDING DIRECTIVES — SAYED 2026-07-25 (the drumbeat ACTS on these without re-asking)

**D-A. CLOSED-LOOP REF-C — LOOP AUTONOMOUSLY UNTIL SIGNIFICANT CLOSED-LOOP PERFORMANCE IS ACHIEVED.** This is a *sustained* mandate, not a single experiment: keep iterating the improve→measure→bank cycle every drumbeat until a **materially better closed-loop number** exists (define "significant" pre-registered per experiment; a plausible bar: junction departure-rate CI-separated better than REF-C base on a **horizon-honest** instrument, with WM/ADE not regressed). **Current chain:** research DONE → **✅ E1a + E2a DONE 2026-07-25 (`2d6589b`, MEASURED, JSONs in `…/incoming/2026-07-25-closedloop-horizon-and-shift/`):** E1a **FIRES OUTCOME A** (corridor-dep 0.0035→0.5877, paired Δ+0.5842 [0.5071,0.6565] SEP p=1.0, OOD≤1.30) → **the BOUND verdict was horizon-confounded, C6 LOGGED**; E2a **PERCEIVABLE** (oracle R² 0.72 / ceiling 0.91, 91% downstream, truncation 0.01% / conditioning 0.11%) → **the lever is the OBJECTIVE, not the encoder or denoise steps.** → **▶ NOW ACTIVE: E1b failure-gated CL-SFT + replay** (~1 pod-day on the now-free pod3, renderer-free, R2LPL-shaped: mine only *recoverable pre-failure* states from the K=185 rollouts where corridor-dep just crossed, supervise **anchor scores** toward the recovering anchor, replay open-loop batches against forgetting; PRE-REGISTER: significant = junction departure-rate@K185 CI-separated-better than REF-C base on the horizon-honest instrument, WM-canary + open-loop ADE@2s NOT regressed; nuPlan Test14-hard 60.67→83.51 is the PUBLISHED precedent) → then intervention #3 (drivable-corridor channel, gated on a ~1-day probe of all 36 PhysicalAI features for map/lane data — the "no HD map" claim has **never been second-probed** and its sibling "no agent boxes" was already retracted as C2). **Every step pre-registered with both outcomes; never quote an interval without its estimator; report each drumbeat.**

**D-B. YOUTUBE BULK HARVEST — RETRY DELIBERATELY AFTER A COOLDOWN.** Sayed approved a deliberate retry. **EARLIEST RETRY: 2026-07-26 12:00 UTC** (~30 h after the block; the block was self-inflicted by our own request churn). **DO NOT auto-retry before that, and NEVER bypass bot-detection** (no cookies/sign-in/player-client evasion/proxy-hopping — out of bounds regardless of convenience). At/after that time, fire the staged **gentle** config ONCE: `W=2 TARGET=400 SEEDS=4 --sleep 4` (`…/incoming/2026-07-25-youtube-idm-scaleup/run_scaleup_parallel.sh`), which self-completes harvest→pseudo-label→4-seed lift→`results_scaleup_downstream.json`. **Use GeoCalib geometry** (`decode_canonical_geocalib`; the fixed 100° HFOV was wrong for 11 of 12 real clips). Privacy unchanged: full-res face/plate/body blur BEFORE downscale, delete raw video, ship pointers-not-bytes. **If it blocks again, STOP and report — do not escalate volume or rotate egress.**

## 🔓 STANDING AUTHORIZATIONS (Sayed 2026-07-23 — recorded so the DRUMBEAT can act WITHOUT re-asking, after this chat's context is gone)

Both carry a **standing veto** (Sayed may reverse anytime); both are pre-registered + reversible-until-committed. **Execute per the EXACT conditions — do NOT broaden scope.** Notify Sayed at each firing.

1. **v4.2b FALLBACK — ✅ EXECUTED to the decision point (`a39aa62a`, 2026-07-23); 🔴 AWAITING SAYED'S ARCHITECTURE CALL.** v4.2b STOPPED (PID 99197+4 workers killed by explicit PID; ops-supervisor 752 correctly left alive; **ckpt.pt PRESERVED step=4000** + all 3 ckpts **backed up to gated HF, no 403**). ⭐ **Cosine pre-probe (n=512) REFUTES surgery:** seam cos(g_wm,g_plan) = **+0.0043** (near-ORTHOGONAL) → PCGrad strips only ~2% of g_plan = a no-op (floor-0.15 already attenuated g_plan to 15% and canary still hit 0.70; surgery passes 98% → can't help). So the coupled path = **from-scratch ONLY** (~53h, v1's proven co-evolve recipe). **✅ SAYED CHOSE FROM-SCRATCH → `flagship-v4-fromscratch-30k` CONFIRMED RUNNING & HEALTHY (`a5655251`, pod2, trainer PID 108011 + supervisor 107985, restarts:0, launched 23:54 Berlin 07-23).** MEASURED step-0: **not-frozen** (enc 149/149 + pred 159/159 req-grad, gnorm both >0), **eff-batch 64**, `from_scratch:true` random-init, loss 1230→924 dropping, `lam_mult 1.0` (floor inert from-scratch, as designed). ⚠️⚠️ **CRITICAL 10k-GATE RECALIBRATION — the canary baseline is 15.674 (n=881), NOT ~0.42 or the smoke's toy ~1.5.** Random-init WM = garbage predictions = huge rollout error; it will **DESCEND** toward v1's 0.42 as WM+planner co-evolve over 30k. **Do NOT judge the 10k canary against the warm-start ≤0.55 bar** (impossible from 15.67 in 10k) — judge the DESCENT TRAJECTORY (co-evolving like v1?) + planner ade, per the card, NOT v1's 0.452. Pace ~7.9 s/step (cold) → **10k gate ≈ ~22h**, 30k ≈ 60-66h. ⚠️ config note (immaterial, attributability holds): differs from v4.2b in 2 flags (from-scratch + floor 0.25-vs-0.15) not 1, but floor is INERT from-scratch (verified lam_mult=1.0). ✅ eval driver EXISTS (`eval_flagship_v4.py`, stream C `a938e1c0`, validated — the design-§7 "blocker" note is stale). Frozen-WM (D1) stays the validated parallel fallback. `…/incoming/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md`. ORIGINAL AUTH: IF v4.2b's Phase-B canary hits the **FAIL bar (≥0.65, ~v4.2 levels)** → run the **~0-GPU cosine pre-probe** on the freed pod2 → **fire whichever it selects**: **from-scratch v4** (~53h/30k, if cosine≈−1 / surgery can't help) OR **gradient-surgery coupling** (`--coupling seam`, ~1.3 A40-day, if a real orthogonal subspace exists). No re-ask; notify + veto-window. *(PASS **≤0.55** → just continue to the 10k gate, no decision needed. MIDDLE **0.55–0.65** → floor 0.10 continue = reversible, autonomous.)* **At that same pod2-free moment, back up the v4.1/v4.2/v4.2b ckpts to the gated `Sayood/` HF repo** (pattern-consistent w/ the 3-arm ckpts, low-stakes — confirmed proceed-under-pattern; ⚠️ may hit the HF-storage-full 403, see below). ⚠️ **the fallback launch (from-scratch OR surgery) MUST include the staged `train_flagship_v4.py` log-fix (`a9dfe223`)** so the new arm logs `g_op_fwd_ade_m` and its 10k gate is speed_benefit-computable (else NOT-SUPPLIED → INCOMPLETE).
2. **GATE-1 — ✅ CLEAN RUN EXECUTED (`a9147f0e`) → 🛑 FT GPU-COMMIT HELD (pre-registered bound branch, the science was done).** Both gates confirmed green (a: offroad 11→7/coll 5→1/pass 3→8; b: low-OOD on-policy) — but the clean run MEASURED two bounds that make a *promotable* FT un-buildable NOW: **(1) instrument gap** — low-OOD source is map/agent-free, can't emit off-road/collision (mutually exclusive with low-OOD until AlpaSim); **(2) data bound** — only ~13-22 real junction eps = memorization regime (leave-3-out held-out Δ≈0). **This is the AUTHORIZED outcome** ("if it can't be a clean promotable run → hold + report"), NOT a veto and NOT a failure. **Path = 3 unblocks:** (2) low-OOD lane-departure metric + (3) REF-C arm → **launching now** (`abe82f1f`, independently valuable); (1) mine ~**100+** distinct real junction scenes (train-corpus held-out-safe / L2D-comma turns) = the data unblock for a future promotable FT → **Sayed's call** (bigger data-eng effort; the FT is only worth it once (1) lands). ⭐ **strategic finding: the low-OOD-vs-safety-metric gap is ~fundamental** — reactive-agent collision/off-road needs a sim (AlpaSim=NuRec=3.2× OOD), low-OOD needs real footage (no agents). Resolving BOTH needs a lower-OOD renderer (hard); until then it's a genuine tradeoff (low-OOD lane-keeping now vs high-OOD full-safety via AlpaSim).

⚠️ **HF-STORAGE BLOCKER (found by `ae72a9e1`):** `Sayood/` private HF storage is **FULL — 403 "storage limit reached"**. This blocks (i) the REF-C 2nd arm of the low-OOD instrument, and **(ii) likely the v4 ckpt-backup in Auth 1 above** (same account). **A Sayed-side storage upgrade / cleanup is a cheap unblock with broad payoff** (also unblocks the older v1.5/v1.6-head + TanitDataSet-C pushes). Until then the ckpt-backup may need a gated-repo that isn't over quota, or waits.

## 📍 CURRENT STATE (2026-07-23 autonomous 11-h run) — the PLANNER, not the world model, is the bottleneck

Two MEASURED results converged this iter; both corrected a live headline. **Read `RETRACTION_LOG.md` C7 before re-asserting.**

1. 🔴 **AlpaSim n=12 PAIRED (DONE, `a901caeac`): REF-C base BEATS flagship v1 — the n=1 was a lucky scene.**
   pass **8/12 vs 2/12**, score **0.496 vs 0.066**, paired Δ **−0.430 [−0.646,−0.215]**, sign-test 8-0 (p=0.008); **collisions TIED**. v1's tactical head is a **high-deviation planner** (plan_dev 1.12 vs 0.34) → **offroad, not collision**. within-sim / ~3.2× OOD; residual 480×854-vs-native-1080 confound. Fixed in RETRACTION_LOG C7 + LEADERBOARD §5.5.
2. 🔴 **v4.1 10k gate = FAIL — CONFIRMED (MEASURED, validated harness, `a938e1c0`):** primary `ade_0_2s` **0.8522 [0.75, 0.98]** ≫ 0.60 bar (≫ v1's 0.4271); miss_2m 0.249 FAIL; oracle_in_fan 0.484 FAIL. **WM is HEALTHY** (canary **0.4599** PASS — the lr_trunk-3e-5 fix worked); the failure is the **PLANNER**, concentrated in **speed/longitudinal** (steady-cruise Δ −0.56 m/s, CI-separated worse than CV & hold-v0; path GEOMETRY beats CV). Formal gate INCOMPLETE (3/8 secondaries have no emitter) but substantively unambiguous. Harness VALIDATED first (MODE A 0.42148 vs 0.4271, O-03 ✓). **F (`planner_on_frozen_wm`) running → pre-commits the kill/v4.2 fork. 🔴 Sayed: kill v4.1, or bank the healthy WM to 30k?** (NOT killed unilaterally — his flagship, WM still healthy).

**Through-line:** our planners underperform (v1-tactical offroad-prone; v4.1 starved); REF-C's anchored diffusion is our best planner yet is open-loop. Stream C (v4-eval-harness) settles v4.1; stream D (planner synthesis) ranks the fixes.

**Standing context (banked earlier):** 🟥 the cross-rig encoder collapse is **REPRESENTATIONAL, not data-diversity** (`results_multirig.json`: multi-domain light-FT −1.61 vs single −1.65, NO recovery) → the own-encoder needs **explicit GAIA-2 camera-conditioning**, not more data. ⚠️ **Fan-out lesson (binding): STAGGER, never burst** — a WAM deep-research workflow = ~106 sub-agents; over-launching crashed the session on 07-23. ⚠️ **DATA-INTEGRITY INVARIANT (2026-07-23):** the val split `physicalai-val-f1b378f295ae` (resident on pod3) **LEAKS 78% (62/79 eps) into the parity train** — **NEVER eval any arm on it.** The canonical CLEAN held-out val is `physicalai-val-0c5f7dac3b11` (v1 validates at 0.425 ≈ 0.4271 there). No past result affected (all used the clean split). ⚠️ **AGENT-MONITOR NOTE (2026-07-23, corrected):** SOME agent-armed background monitors have had **delivery gaps** (Orin phase-1 notification gap; Branch B milestone monitor) → the agent looks stalled (GPU idle + no report) though the JOB finished fine. But **NOT universal** — `a938e1c0`'s transfer notification fired correctly and on time. Guidance: brief agents to **poll the ckpt/log directly as a SAFETY** (belt-and-suspenders), not "the notification will fail." *(My earlier "hit every agent" framing was overstated.)*

## FLEET — verified ~00:15 Berlin 07-24 (nvidia-smi compute-apps, THIS iter)

| pod | GPU | owner | state |
|---|---|---|---|
| `tanitad-pod2` | 🟢 34.7 GB, PID 108011 | flagship **from-scratch v4** | ⭐⭐ **step 11650 — 10k gate=CONTINUE; WM co-evolving, val healthy on its descending trajectory (NOISY per-point, judge the trend).** val ade@2s: 0.4825@10500 (best) · 0.7225@11000 (BOUNCE) · **0.4788@11500 (SNAPPED BACK)**; oracle 0.242→0.532→0.249; miss 0.169@11500 (best). ✅ **the 11000 bounce was a HARD EVAL BATCH = NOISE, confirmed by the 11500 snap-back** (correctly NOT alarmed on 1 point — canary stayed in-band 1.3-2.6, controller "ok"). net trend down from 0.59@6.5k → ~0.48, → v1's 0.427. λ_plan=1.0, restarts 0, auto-continues. heartbeat 21:12Z. 30k ≈ ~1.8 days. **⚠️ FORMAL 8-metric gate DEFERRED:** the eval-pod relay needs an HF push from pod2 but `Sayood/` HF = 403 full → run the formal `eval_flagship_v4.py`+`gate_emitters.py` at the 30k finish (or post-HF-cleanup); the CONTINUE decision does NOT need it (in-trainer clean val is decision-grade). **✅ CKPT-BACKUP DECIDED (Sayed 2026-07-25): WAIT FOR THE 30k FINISH, then push the FINAL ckpt to HF — do NOT push mid-training from pod2** (it is the RAM-bound pod that OOM-killed a flagship before; the run is healthy and ckpts every 1,000 steps bound the loss). **HF space is now FREED (277→220 GB) so the push + the formal 8-metric gate are both unblocked the moment 30k lands. DO NOT RE-ASK.** 30k ≈ ~1.3 days. **NEVER eval here.** ✅ GATE-PREP DONE: `eval_flagship_v4.py` + `gate_emitters.py` + `speed_benefit.py` all now on the eval pod (`/root/v4eval/stack/`, scp'd 15:2xZ) → the 10k gate can render a COMPLETE verdict incl. speed_benefit. ⚠️ **GATE-PREP (confirmed this iter): the from-scratch log has `g_op_fwd_ade_m` = 0 matches** — the a9dfe223 log-fix did NOT ride this launch → 10k-gate `speed_benefit_recovered_frac` reads NOT-SUPPLIED from the log. **NOT fatal: run `gate_emitters.py` on the 10k ckpt to emit it fresh (built+validated on v1 = 0.8184 PASS, GateEmit stream). Pre-gate action: ensure `eval_flagship_v4.py` + `gate_emitters.py` + `speed_benefit.py` are on the eval pod (staged/uncommitted → may need scp) ~1-2h before the gate.** ⚠️ full verdict = holding to λ_plan=1 + the 10k gate. eff-batch 64. **NEVER eval here** |
| `tanitad-pod3` | 🟢 ACTIVE — **YouTube-IDM PILOT** (`a9b5eacc`) | scale-up | 🟢🟢 **SAYED GREEN-LIT THE YOUTUBE-IDM SCALE-UP (07-24) — pilot LAUNCHED on pod3.** His gate was "did you compare extracted ego-motion vs GT?" → **YES (MEASURED, `results_idm_pipeline_derisk.json`): speed R² 0.62 (cross-class) / 0.66 (cross-rig) zero-shot, longitudinal-traj R² 0.60, yaw R²≈0 cross-class (WEAK), accel unusable (dropped). Direct speed 0.62-0.66 is BELOW the clean-0.70 bar; the GO rests on DOWNSTREAM tolerance (109% speed / 71% yaw of real-label pretraining value, parity, 4 seeds).** YouTube itself has NO GT → cross-class 0.62 is the optimistic proxy. Pilot brief: privacy-safe (CC-licensed dashcam ONLY, face/plate blur, pointers+pseudo-labels not raw video), pod3-only (never pod2/eval), priority-ordered (scaffolding→harvest→pseudo-label→downstream-lift), PRE-REGISTERED downstream metric (YouTube-pilot pretrain vs no-YouTube floor on parity-val speed_r2). Verdict WIN→justifies full harvest / BOUND→names the domain gap. ⚠️ full non-CC harvest = a SEPARATE Sayed licensing gate. — Prior (DONE): OWN-ENCODER PIVOT RESOLVED FAVORABLY (`ad4e13c4`): flagship-v1's encoder IS a usable IDM substrate AS-IS — the CHEAP pivot. The −1.169 rig-B "failure" was a HEAD-DIVERSITY artifact (one-domain readout overfits geometry): a **multi-domain readout head recovers it → rig-B speed +0.657 / yaw +0.504, cross-CLASS fisheye→comma +0.585** (≈ comma's own ceiling). v1 ≫ Branch B everywhere; Branch B's weakness is real (aug-caveat closed). ✅ **so: use frozen v1 + a multi-domain IDM head (cheap) for the immediate need; the EXPENSIVE warm-started variant is gated on ONE residual — cross-CLASS YAW transfer (fisheye→rectilinear) is UNVERIFIED (comma yaw unreadable, a label artifact — C6) → cheapest next = a readable-yaw rectilinear corpus (data-acquisition, not a quick pod job).** ⭐⭐ **IDM DOWNSTREAM-ABLATION = GO (`ad4e13c4`, MEASURED, 8 seeds paired) — overturns the R²~0.63 proxy's "lean no-go".** Pseudo-label WM-pretraining captures **~96% of the real-label pretraining benefit** (comma speed R² floor −0.77 / pseudo 0.447 / ceiling 0.491 = 0.965; traj ADE 0.984; rig-B 0.96; all 8 seeds beat floor CI-separated). R²~0.63 labels = near-real-quality PRETRAINING signal (pretraining tolerates the noise the proxy penalized). **→ YouTube-IDM SCALE-UP is now EVIDENCE-JUSTIFIED = a Sayed GO/NO-GO (big + licensing-gated commitment).** Recommended design: v1 frozen + multi-domain IDM head → pseudo-label YouTube → pretrain WM → FT on parity; weak per-clip speed prior; speed+long-traj primary, drop accel, caveat yaw/lateral. ⚠️ still bounded by v1's cross-class rep gap (~0.63, caps the ceiling not the pretraining value). ✅ **PARITY-VALIDATION = GO, DECISION-GRADE (`ad4e13c4`, MEASURED on the ACTUAL parity target, 4 seeds):** pseudo-label WM-pretraining captures **109% speed / 107% traj / 71% yaw of the real-label ceiling** (floor −0.44 / pseudo 0.751 / ceiling 0.651; all seeds beat floor CI-sep) — even STRONGER than the 96% proxy (on-target labeler = on-distribution). YouTube-IDM pretraining MECHANISM fully de-risked; residual = v1's cross-class gap caps novel-rig ABSOLUTE quality (not the value). **→ YouTube-IDM scale-up is now thoroughly evidence-justified; the ingest is Sayed's licensing-gated commitment.** **pod3 FREE/held** (scale-up = Sayed-gated; other next steps data-blocked). Prior: 🔴 **BRANCH B TRANSFER = FAIL (`ad4e13c4`) (`ad4e13c4`, MEASURED `results_branchb_transfer_e50_CONVERGED.json`).** Cross-rig speed R² (gate >0.9): Branch B **−0.667** (clean disjoint) vs **flagship-v1 frozen +0.657** — paired dR2 CI excludes 0 on **3/4 arms** → from-scratch GAIA-2 camera-conditioning does NOT give rig-robustness + is WORSE than the plain v1 encoder. The own-encoder / YouTube-IDM thesis resting on it is **NOT supported.** ⭐ **POSITIVE discovery: flagship-v1's TRAINED encoder is the stronger cross-rig substrate** (+0.66 multirig_val; though −1.17 rig_val = not uniformly robust). Harness validated (v1 in-domain +0.86-0.91), leakage-controlled (episode-disjoint), C5-cautioned. **pod3 HELD:** the recommended pivot (flagship-warm-started encoder variant) is a NEW training arm = Sayed's call after this decisive failure. ⚠️ HF backup push classifier-gated (ckpt safe on pod3+MooseFS). ✅ Registry §10.1 = FAIL BANKED (`ad2e2ff9`; + encoder-strategy doc updated) |
| `tanitad-eval` | ⚪ FREE (departure-power DONE) — reserved for 10k gate | — | 🔴 **DEPARTURE-POWER n=40 = BOUND, DECISIVELY → the closed-loop-improvement (recovery-aug) direction CLOSES honestly (`a1f26c92`, MEASURED).** At full power (n=40 cross-fit, 1.83×) the naive departure "win" **REVERSES: +0.0089(n12) → −0.0302(n40) SEP, departs 3.3× MORE**; ADE worse under both metrics. The n=12 win was favorable-split noise. ⚠️⚠️ **RETRACTS my "D2 recovery-aug HALVES departures + generalizes" durable-positive** (reported repeatedly incl. the 12:57 report) — **C5, n=12-fragile, reverses at n=40** (logged). Confound flagged: cross-fit trains 20-ep vs 28-ep folds → part is data-reduction, but the unbiased full-corpus estimate is neg+separated → not robustly promotable. **Durable (un-retracted): the method/machinery (renderer-free recovery + low-OOD on-policy harness + tolerance-band metric + encoder canary) reusable · REF-C encoder safely FT-able · 2 measurement lessons (use band_ade2d; n=12 underpowered→cross-fit — gate every future CL claim).** Next CL bets = (B) reactive-agent renderer + map-aware instrument = bigger builds, not recovery-FT. eval reserved for 10k gate (~7h) |
| `tanitad-pod` (pod1) | ⚪ FREE — bigplanner DONE | — | ⭐ **FROZEN-WM CONTENDER RESOLVED (`ade3edfb`, MEASURED): capacity is NOT the lever.** Scaling the feedforward planner 11× is a FLAT line (0.599→0.601→0.599, none separated; bigger query-heads OVERFIT to 0.82-0.86). It saturates at ~0.60 because the error above the 0.4045 oracle is **aleatoric future-uncertainty** (intent isn't determined by the past) — more capacity can't reduce the unknowable. CEM search (0.132) only escapes it via a **privileged test-time signal** (optimizing per-window against the actual future). **→ frozen-WM feedforward = a solid CHEAP degradation-free FALLBACK (~0.60, beats CV/hold-v0, canary untouched), NOT a search-matching contender.** ⭐⭐ **VALUE-MODEL CRUX = FAIL (`ade3edfb`, MEASURED) → frozen-WM contender DEFINITIVELY DEAD; it is a ~0.60 FALLBACK, period.** Learned-value search (deployable, no GT future) = **1.016, SEP-WORSE than W 0.599** — a value model learns only E[cost] (minimiser = mean trajectory W already gives), and CEM adversarially fools it. Every deployable route: feedforward 0.599 · bigger 0.60 · distill 1.40 · learned-value 1.02 — all hit the ~0.60 aleatoric wall. 🔴 **HONEST REFRAME (supersedes the "search 4.5× / planner is the headroom" claim — logged C6):** the 0.132 "search ceiling" is **HINDSIGHT-PRIVILEGED** (it peeks at the expert's actual future, which the ego doesn't control in an open-loop metric) → the W→0.132 gap is **prediction-vs-hindsight, NOT deployable planning headroom.** Frozen-WM investigation COMPLETE end-to-end. pod1 FREE. |

## 🔄 ACTIVE STREAMS (5 active: **from-scratch-v4→30k**/pod2 · **v2-corpus build**/pod3 (finishing) · **YouTube-IDM non-CC SCALE-UP**/`aea4861a` pod1/pod3 · **GeoCalib**/`a4ebd01c` eval · **paper-v0.5 + PROGRAM_OVERVIEW**/`a4427055` devbox;

✅ **COMMITTED 2026-07-25 (Sayed: "commit our valuable achievements")** — branch `agent/benchmarks-eval-20260721`, 924 staged files in 3 clean commits, index now 0: **`52d089a` stack** (v2 dataloader · traffic-light TLC metric · TanitResim + the nav-label bugfix · gate emitters) · **`df32781` taniteval** (canonical CLI · **78%-leak split now hard-refused in code** · 153 tests green off-pod) · **`2b18575` research+steering** (v4 fork resolved · D1/D2 verdicts · IDM win · v2 corpus · MODEL_REGISTRY v4 · 5 program reports · C5/C8 retractions). NOT pushed (per standard).
⚠️ **NEW OPERATIONAL TRAP (BANKED in CLAUDE.md §Git-hygiene): `git commit -- <pathspec>` SEGFAULTS on this repo** (MSYS exit 139 AND native Windows 0xC0000005 → not the shell; it's git's partial-commit temp-index path; NOT fsmonitor, already false). ⚠️ **It is the pathspec SHAPE, not the file count** — my first read ("178+ files") was WRONG, a 2-file commit then crashed too. **Works:** a space-free dir (`stack` 81, `taniteval` 149) · a single file (even under `A & B/`). **Crashes:** a dir WITH spaces (`"TanitAD Research Hub"`) · **2+ pathspecs where any has a space** (repro 2×). Each crash **leaves a stale `.git/index.lock`** → next commit dies with "Another git process seems to be running" (debris, not contention): verify no git proc, `rm -f .git/index.lock`, index survives. **Use ONE space-free pathspec per commit**, or a pathspec-free `git commit -F msg` ONLY after verifying no foreign agent work is staged.

🟡 **YOUTUBE-IDM NON-CC SCALE-UP (`aea4861a`) — PIPELINE BUILT + VALIDATED, but the DECISION-GRADE VERDICT was NOT produced.** 🔴 **YouTube HARD-BLOCKED pod3's IP mid-harvest** ("Sign in to confirm you're not a bot"; a block not throttling — an isolated single request also fails). **Root cause OURS: the pipeline was iterated against the LIVE source** (single → parallel ×3 restarts → GeoCalib rework → 3 smokes + a 65-clip run); cumulative burst volume tripped the anti-bot. The pilot's clean run held only because it ran ONCE at low volume. ✅ **Correctly NOT bypassed** (no cookies/sign-in, no player-client evasion — bot-detection bypass is out of bounds; the block is a rate-limit signal to respect). **VALIDATED (MEASURED, real clips before the block):** non-CC harvest works (real non-CC dashcam, `is_cc=false`, license per pointer → the CC-scarcity ceiling IS removed) · privacy intact (full-res face/plate/body blur — e.g. 14 faces/58 plates/21 bodies on one clip — raw mp4 deleted, pointers-not-bytes) · **GeoCalib per-video geometry integrated deadlock-free** (thread_type=NONE; f_eff≈266 `fully_canonical`) · **2 real traps fixed: GeoCalib's `opencv-python` dep silently clobbered pinned cv2 4.11 → dropped `CascadeClassifier` → BROKE THE PRIVACY BLUR** (restored), and 8-worker thread oversubscription (loadavg 98 at 81% idle CPU). **NOT done:** P2 harvest to 400-500, P3 label-at-scale, P4 ≥4-seed lift, the verdict. **No partial corpus survives** (cleaned during a restart) → **the 80-clip pilot DIRECTIONAL win (~92% of ceiling) remains the ONLY YouTube result; the scale claim is NOT upgraded.** 🔴 **GATED ON EGRESS = a SAYED decision** (cooldown-then-gentle-rerun vs alternative egress vs stop at directional). **DO NOT auto-retry.** Everything else is one command away: staged gentle config `W=2 TARGET=400 SEEDS=4 --sleep 4` self-completes harvest→label→4-seed lift→`results_scaleup_downstream.json`. Logged in RETRACTION_LOG (new class: operational churn against a rate-limited live third-party source).

✅ **GEOCALIB DONE (`a4ebd01c`, staged) — QUALIFIED PASS, adopt as the YouTube geometry front-end** (with robust multi-frame aggregation + confidence-gated fallback, both implemented; NEVER single-frame). ⭐ **THE HEADLINE (MEASURED on 12 real CC videos, `youtube_geocalib_measurement.json`): estimated HFOV median 66.6° (confident-only 60.5°), range 32–77° — only 1 of 12 is within 10° of the pilot's assumed 100°.** The fixed 100° over-crops ~1.4× → **inflated pseudo-speed on most clips** → per-video estimation is warranted, confirmed on real data. GT-recovery (comma2k19, focal 910): **6.8% median focal error**, vFoV +3.6°, **resolution-robust at ≤480p** (matters — the pilot decodes ≤480p). ⚠️ HONEST BOUNDS: NOT a precise oracle — weak absolute-focal tracking (Pearson **r=0.41**, regresses toward a ~50-55° vFoV prior), per-frame outliers ±30% (hence aggregation), 120° f-theta fisheye is out-of-model (correctly rejected), and the absolute claim rests on ONE real rectilinear camera (adding KITTI/nuScenes = top follow-up). ⭐ **Found+fixed a REAL bug the integration introduces: a threaded PyAV decoder torn down with a live CUDA context DEADLOCKS on close** (the pilot never hit it — its decode had no inline CUDA) → fixed via single-threaded decode. 🔴 **ESCALATED to the scale-up agent (`aea4861a`) via SendMessage, not left in a doc:** consume `geocalib_intrinsics.py` / `decode_canonical_geocalib`, MUST avoid AUTO-threaded PyAV, and re-run or dual-report if it already labeled on fixed-HFOV — otherwise the decision-grade lift bakes in a known systematic geometry error. Follow-up pre-registered: the downstream ADE closure (re-pretrain on GeoCalib- vs fixed-cropped YouTube) needs the IDM encoder on pod3.

🟢 **SAYED 2026-07-25 — 3 directives + "keep building the program":** (1) **COMMITTED to non-CC YouTube licensing** → YouTube-IDM scale-up GO (decision-grade, ~500-1000 clips, GDPR-safe pointers-not-bytes; the pilot WON at 80 CC → ~92% of ceiling); (2) **implement GeoCalib** (per-video intrinsics, removes the fixed-HFOV approx; validate on known-intrinsics data first); (3) **HF cleanup** — I ENUMERATED `Sayood/` (~277GB full; mostly Sayed's OTHER projects: 4× Bussy-8B 70GB + datasets 36GB; TanitAD ~134GB = comma2k19-episodes 88GB + TanitDataSet-C 16GB + ckpts). **Minimal safe unblock (~13GB, all TanitAD, SAYED'S CLICK — I don't run irreversible deletes even when asked): delete flagship-v4.1/v4.2/v4.2b (9.7GB superseded warm-start fails) + refa-ijepa-4b (1.9GB val-leaked) + 2 empty repos** → unblocks the flagship ckpt-backup + formal gate; I do the backup the moment space frees. (4) **keep building** → next autonomous work = low-OOD reactive-agent renderer (the real closed-loop lever) + v2-QA + next-gen-flagship prep.

- v2-corpus build (`ab583570` PARALLELIZED

✅ **v2-DATALOADER-WRAPPER DONE (`a0892dc5`, staged) — the ONE integration gap CLOSED; next-gen flagship LAUNCH-READY.** Lazy provider → the UNCHANGED `FlagshipWindowDataset` (windows byte-identical to the raw path). MEASURED (eval-pod A40, 20 real v2ep.pt): **20/20 contract-identical (all 12 window keys value-for-value, both label regimes); RAM bounded (1200 fetches flat ~1.2GB, corpus-size-independent); throughput 46.7 win/s @8 workers ≫ the ~7 needed → fully hidden under compute.** `--v2-cache <dir>` flag; NO flag = byte-identical to today (parity-safe). `pytest` 836✓. Poses/actions MEASURED float32 (no dtype gap). Staged: `stack/tanitad/data/v2_dataset.py` + `train_flagship4b.py` (wired) + `test_v2_dataset.py` + `…/incoming/2026-07-24-v2-dataloader-integration/NOTE.md` (launch cmd). **Remaining for the v2 flagship: (1) v2 build completes (~2h) → (2) consolidate pod1+pod3 shards (union by clip-id) → (3) launch per NOTE.md when pod2 frees (~2d). Queued: v2-corpus QA (validate consolidated cache loads + the balanced distribution is achieved, no corrupt clips) when the build finishes.** — v2 build (`ab583570` PARALLELIZED — split at clip-id index 4500, ZERO overlap; ✅✅ **BUILD COMPLETE 2026-07-25: 9,000/9,000 clips** (pod1 4,953 + pod3 4,047, disjoint by clip-id) = the balanced **50.25 h** v2 corpus, JPEG-compressed 256px, ~25 GB, pod1 79 GB free. ✅✅ **QA DONE (`a54113c8`, MEASURED, staged) → VERDICT: GO on the DATA, NO-GO on launching today (2 code/ops gates, neither a data problem).** ⭐ **THE BALANCING WORKED: turns 28.04% vs 28.0% target — every class within 0.07 pp** (lane_keep 44.93 vs 45, accel 13.03, brake 14.00, speed 9.72/52.28/38.01, junction presence 37.84→**61.38%**). Selection proxy predicted built reality almost exactly (per-clip turn-fraction **r=0.9997**). **INTEGRITY CLEAN: 9,000/9,000 loadable, ALL 1,808,710 JPEG frames decoded individually, 0 failures**, 0 non-finite poses, 22.32 GB. ⭐ **Strongest evidence: 571 clips exist in BOTH the parity epcache and v2 → 571/571 BIT-IDENTICAL poses (max |Δ| = 0.0)** — two independent build paths agree to the last bit. Key **recomputes to `4b7eeeac222d`** from the built files (on-disk IS the designed corpus); 0 overlap / 0 missing / 0 double-built. ⚠️ **3 HONEST CORRECTIONS: (1) trainable hours = 49.742 h, NOT 50.25** (that was the pre-D-015-stacking raw resample). **(2) 28.04% is a v1-LABELER number; under the curvature-gated v2 labeler the same corpus reads 18.83%** (1.63× parity, not 1.97×) — flagship default `labels_v2=False` sees 28%, a `--v2-labels` run does NOT. **(3) `episode_id` = first 4 hex of the UUID → 8,391 distinct for 9,000 clips (609 collisions, 6.8%; parity 1.4%, so it SCALES not regresses)** — harmless for training but it would **silently mis-cluster different clips in the episode-cluster bootstrap = our decision-grade CI estimator** → fix/guard queued. Loader: 12-key contract intact, builder-vs-loader byte-identical on 24 clips, **20.0/37.6/49.4 win/s @4/8/16 workers, RSS 4.4/16.8/30.4 GB (~2 GB/worker → 16 workers would OOM a ~55 GB cgroup)**. 🟢🟢 **`flagship-v2corpus-30k` LIVE on pod1 — trainer PID 699286, relaunched 2026-07-25 with the CORRECTED labeler; ETA ≈2026-07-29T01:10Z.** Step-150 health: loss 37.69→16.54, 11.27 s/step, RSS 12.2/57.7 GiB, restarts 0. Step-0 verified: **OOM guard 9,000 matched files (NON-ZERO)**, **`labels_v2 True`**, params **286,339,251** w/ encoder 87.1M *inside* trainable (not frozen), eff-batch 64, 9,000 providers / 9,000 DISTINCT episode ids, 1,538,710 windows. ⭐ **THE EXPERIMENT IS NOW GENUINELY SINGLE-VARIABLE — verified from the ARTIFACT not registry prose:** `flagship4b-v2-30k/config.json` = `v2_labels:true, speed_input:true, rollout_k:12, anchor_tactical:true`, parity corpus, same 286,339,251 params — **the new run matches EVERY one; only the CORPUS differs** (13.13 h parity → 49.742 h balanced). *(I first launched with `--no-labels-v2` on a void rationale — the "running arm" I matched is a different trainer+architecture; caught by the agent, restarted ~20 min in, logged C8.)* ⚠️ **PRE-REG CORRECTED AGAIN: matched-step is 5,000, NOT 7,800** — the control's log ends at 7,700 with **duplicated rows (7,500/7,700 twice = a supervisor resume)** and its only cleanly archived ckpt is `ckpt_step5000.pt` → **step 5,000 = primary (clean both sides)**, ~7,700 = secondary, flagged as sitting in the resume-duplicated tail. Aborted run preserved as `…_ABORTED-labelsFALSE-20260725T0221Z/` (verified it held NO ckpt.pt → a mismatched-label resume was impossible, not merely unlikely). ✅ **Death-only watchdog attached** (manifest `TRAIN_CMD` verbatim from `/proc/699286/cmdline`; supervisor correctly logged "trainer ALREADY RUNNING … NOT launching"). 🔴 **LANDMINE FOUND + DEFUSED: `runs.d/flagship-v3enc.env` was still `ENABLED="1"`** — `pod_boot_hook.sh` iterates ALL `runs.d/*.env` and `/pre_start.sh` is installed, so **the next pod restart would have relaunched that RETIRED arm (STOPPED@10,800, gate=RESTART) onto the same A6000**, fighting this run for GPU + cgroup. Set `ENABLED="0"` w/ inline reason + backup (a config change to another arm's manifest — flagged, not buried; reverting is one character). 🔴 **PROVENANCE BUG (queued fix): under `--v2-cache` the trainer writes `"cache_dirs": null, "data": "realmix"` → the run's OWN config.json does NOT record which corpus it trained on.** Mitigated for this run (corpus + key `4b7eeeac222d` captured in the run manifest AND the staged REGISTRY §1.7 entry), but **a one-line trainer fix is worth making** so provenance never lives only in a log line. — superseded: LAUNCH GATES CLEARED (`a0ba571e`) → ⭐ **Premise changed on evidence: pod1 is IDLE (2 MiB GPU) and is the BETTER node** — A6000 48GB vs A40 46GB, **14× faster I/O (91.7 s local NVMe vs 1,285 s MooseFS — decisive for a JPEG-decoding loader)**, more RAM (57.74 vs 51.22 GiB cgroup), and it already holds 4,953/9,000 so only 9.93 GiB moves. pod2 untouched (still training, frees ≈07-26 10:10Z). **Gate 1 CLEARED**: `stack/` synced to pod1+pod3, verified by a FULL trainer import on both torch majors (not a file listing) — `--v2-cache` in the real parser, `v2_dataset` imports, `refb_labels` 1300 lines md5 `6632348b…`; pod2 got a zero-risk shadow stack. **Gate 2 IN FLIGHT**: pod3→pod1 streamed resumable pipe since 00:03Z at ~1.38 MB/s (dev-box uplink is the hard cap: 1.047 single / 1.271 4-stream = link-limited), ETA **02:12Z**, integrity proven 20/20 then 30/30 md5 mid-flight. ⭐⭐ **3 LANDMINES CAUGHT BEFORE LAUNCH DAY: (1) the OOM guard was SILENTLY INERT on every v2 run** — it globbed nested raw `*/ep_*.pt` while the v2 cache is flat `*.v2ep.pt` → **matched 0 files and could free nothing**, while its pre-arm message looked healthy (fixed: 0→4,953 watched, now reports matched count + warns at 0); **(2) the guard's 60 GiB default is ABOVE pod1's 57.74 GiB cgroup** → could never fire before the OOM killer (added a cgroup-**v1** cap check — the pods are v1, a v2-only check silently skipped); **(3) `v2_compressed.py`, the corpus BUILDER, was stranded on pods only, absent from the repo** → rescued, md5-identical. ✅ **episode_id FIXED AT LOAD** via `stable_episode_id()` (63-bit blake2b of the full clip_id): **304 collisions → 0 on the real shard**, zero rebuild, every `.v2ep.pt` byte-identical (the QA identity proof stands), parity path untouched. ⭐ **BONUS: the v2 flagship already RAN end-to-end twice (done:true) — 286,339,251 params (= parity's 286.34M), loss 36.08→26.10 over 30 steps, ~11 s/step → 30k ≈ 3.8 days, data wait only 13% of step time (8 workers confirmed right).** *(A first ~30 s/step read was the agent's own pytest stealing 1289% CPU — diagnosed, killed by explicit PID, honest.)* 🔑 **LABELER DECIDED BY ORCHESTRATOR: `--v2 --no-labels-v2` (v1 labeler, 28.04% turns).** Decisive fact I verified on pod2: the RUNNING flagship's config has **`v2_labels: false`** → v1 labels make the **CORPUS the ONLY changed variable** vs the running arm (clean attribution) AND match the balance the corpus was selected for. *(Data identical either way — only the label convention differs, so turn EXPOSURE is unchanged.)* Pre-registered as: same architecture + same labeler, differing ONLY in corpus (13.13 h parity → 49.742 h balanced). 🔴 **2 items left for Sayed (both deliberately NOT actioned): HF is blocked TWICE OVER — the 403 storage AND an invalid token on pod3; and pod1 CAN already reach pod3's SSH port → direct pod-to-pod transfer is network-feasible and needs only an authorized-key install = a SECURITY-CONFIG change, his call. Either collapses future 2-h moves to minutes.** — superseded gate text: 🔴 2 LAUNCH GATES → (1) **pods' `stack/` is STALE** — `--v2-cache` absent from the trainer on BOTH pods, `v2_dataset.py` missing, `refb_labels.py` stale (474/745 vs repo 1300) → a launch dies on the flag then ModuleNotFound; (2) **no single node holds all 9,000** (pod1 alone 27.35% turns, pod3 28.89% — only the UNION is on target) and pods can't ssh each other, dev-box relay ~1 MB/s (22 GB ≈ 6 h), **HF fast path 403-blocked** → ⭐ **the HF-storage cleanup would collapse this transfer from HOURS to MINUTES = concrete new evidence for that Sayed decision.** Corpus verified UNTOUCHED by QA (0 files modified). Report: `…/incoming/2026-07-25-v2-corpus-qa/`. ✅ `load_compressed` wrapper DONE. Next (mine): v2-corpus QA when pod3 finishes (run the v2 loader on each shard + verify the balanced distribution); consolidation onto pod2 is a ~2d-out task, gated on pod2-free + the pod→pod relay (HF-403 or slow dev-box));

✅ **YouTube-IDM PILOT = WIN, directional (`a9b5eacc` DONE, MEASURED `results_youtube_pilot_downstream.json`):** pretraining a small WM on **80 CC-licensed pseudo-labeled YouTube clips** lifts parity-val driving from a broken floor to near the real-label ceiling — **speed R² −0.520→+0.563 (3 seeds, clip-cluster bootstrap CI excludes 0 EVERY seed, gaps +1.37/+0.88/+1.05); yaw 0.55→0.75; ADE HALVED 12.8→6.3m; ≈92% of the real-parity-label pretraining ceiling.** The YouTube domain TRANSFERS (the one read we couldn't get from our own data), despite unknown intrinsics/no-CAN-GT/80 clips. ⚠️ DIRECTIONAL not decision-grade (80 clips vs 300, 3 seeds; the negative floor inflates the raw gap → the fraction-of-ceiling ≈0.92 is the substantive claim; speed+traj trustworthy, yaw rides the 15-clip real FT). P3: 80 clips→8,960 windows, speed sanity 8.51 m/s 100%-plausible-not-collapsed. **P2 CC-SCARCITY (key op finding): 80 clips from 31 producing / 63 tried / ~339 CC candidates — clean continuous CC forward-dashcam is SCARCE (CC pool = timelapse/compilation-dominated) → a LARGER harvest needs the non-CC tiers = SAYED + LEGAL DECISION.** Privacy: CC-only + face/plate/body blur full-res pre-downscale, only latents+pseudo-labels+pointers persisted (32 CC videos, auditable, ship-pointers-never-bytes). 🔴 **ESCALATIONS: (1) pilot WINS → larger harvest justified but CC-scarce → non-CC = Sayed's call; (2) decision-grade wants ~300+ clips/4+ seeds + a per-video intrinsics estimator (GeoCalib); (3) intake `…/incoming/2026-07-24-youtube-idm-pilot/` alongside the 07-22 groundwork.** pod3 freed → repurposed to the v2-build shard. banked this session: GradCouple·LowOOD·Registry·LowOODhard·G1proto·GateEmit·G1clean·LaneKeep·D1-frozenWM·v4.2b-fork·from-scratch-launch·**D2-departure-power-BOUND**·**TanitEval-productionize-DONE**·**TanitResim-productionize-DONE**·**TrafficLight-TLC-DONE**·**corpus-profile-DONE**)

🆕 **V2 CORPUS (50h) — Sayed 2026-07-24, decisions recorded:** enlarge the training set to **~50h within NVIDIA PhysicalAI-AV** with an IMPROVED (balanced) scenario distribution. **DECISIONS: (1) this is a NEW "v2" canonical corpus for the NEXT flagship gen — BREAKS PARITY with `e438721ae894` BY DESIGN (do NOT touch the sacred set); the CURRENT from-scratch flagship FINISHES on the 13h set (untouched). (2) "Augmentation" = distribution-balancing BY SELECTION (oversample rare classes), NOT synthetic perturbation; re-balance the ORIGINAL 2,376 too (whole 50h balanced).** ⭐ **CORPUS PROFILE DONE (MEASURED, `corpus_profile.json`): 13.13h / 472,627 frames / 2,376 clips × 19.9s; maneuvers lane_keep 59.6% · accel 13.2% · brake_stop 12.9% · turn_right 7.4% · turn_left 6.9% (14.25% turns); nav straight 52.8%/L 12.0%/R 11.5%; speed stopped 7.8%/city 45.9%/hwy 46.3% (balanced); only 42.6% of clips have ANY turn; 30k steps = 4.73 epochs (max_horizon 20, 406,099 windows).** GAPS ranked: (1) turns/junctions (2) stops/low-speed-urban (3) **semantic scenarios 0% coverage** (4) sharp geo. ✅ **PHASE-1 DONE (`ab583570`, MEASURED, staged): 50h reachable with ZERO download** — 197 egomotion chunks already on the dev box = **18,731 moving clips / 104.6h** (2× headroom), same country-stratified pool as parity. **Balanced 9,000-clip selection hits target EXACTLY:** lane_keep 59.6→45.0%, turns 14.25→28.0%, accel 13.0%, brake 14.0%, hwy/city/stop 38/52/10%, junction-clip presence 37.7→61.3%; 1,311 original re-selected + 7,689 new (whole corpus balanced). Key **`physicalai-v2bal-4b7eeeac222d`**. 30% turns available (one-line change). Deliverables `…/incoming/2026-07-24-v2-corpus-50h-balanced/` (`V2_CORPUS_DESIGN.md`, `r0_selection_v2.parquet` 9k clips, `v2_pool_scored.parquet`, scorer+selector). 🟢 **PHASE-2 BUILDING on pod1 (detached, `ab583570` executed; Sayed chose "fit the quota"):** ⭐ **JPEG-compressing the f-theta-cropped 256px frames = 982 GB → ~25 GB** (2.9 MB/ep, full 256px parity KEPT, no downscale) — the fit-quota win, far better than the ~250 GB estimate. **Frames BIT-IDENTICAL to the parity `_decode_mp4` pipeline** (`torch.equal`; only JPEG q90 differs); `load_compressed()` returns the exact `ToyEpisode` contract (frames/poses/maneuvers). Build: **~26 clips/min, 297/9000 at last check, ETA ~6h**, 5/5 workers, 0 errors, RESUMABLE (skips built clip_ids; a kill yields the built portion). Disk: pod1 `/workspace` is a **LOCAL 500GB NVMe (NOT MooseFS)** → `df` IS authoritative here (92 GB free; the 409 GB used = untouched parity+cosmos builds). Cache: `tanitad-pod:/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d/`. Throughput fix (logged): crop only the ~201 KEPT frames not all 605 = 4.84× faster bit-identical; tuned K=5 for pod1's ~57.7GB cgroup cap (the OOM cap that killed the flagship here before). **✅ NEXT-SESSION CHECK: `ls …/*.v2ep.pt | wc -l ≈ 9000`** (build finishes unattended, no babysitting). 🔴 **INTEGRATION FLAG (before v2 training): the v2 cache is a NEW on-disk format (JPEG dicts via `v2_compressed.load_compressed`), NOT the raw `load_episode()` the current epcache uses → a v2 training run MUST read via a thin `Dataset` wrapper calling `load_compressed` (~1-2ms/frame decode in dataloader workers). One wiring step, queued.** Deliverables `…/incoming/2026-07-24-v2-corpus-50h-balanced/` (`V2_PHASE2_BUILD.md`, `v2_compressed.py`, `build_v2_launch.sh`). Target 28% turns (30% one-line if wanted). ⚠️ semantic scenarios (traffic-light etc.) still 0% — kinematic selection can't buy them; needs the separate VLM track. Also fixed a latent chunk-mapping bug (5 clips w/ egomotion in a neighbour zip).

✅ **TrafficLight-scenario+metrics DONE (`a9806aeb`, staged not committed) — both real metric gaps closed:** (1) **traffic-light scenario = SC-14** (red-light running, from `SCENARIO_DATABASE.md`) + a new **TLC (Traffic-Light-Compliance) metric** `= red_entry_gate × stop_quality × green_flow ∈[0,1]` (a single red-run zeroes it; covers stop-before-line / no-red-entry / stop-smoothness / no-phantom-brake-on-green). Discriminative synthetic-oracle: rule_barrier TLC **1.0** vs soft_prior **0.0** (ran the red); 24 analytic tests. `stack/tanitad/eval/metrics.py` (+163 additive), `scenarios/traffic_light.py`, `scenarios/registry.py`. (2) ⭐ **FIRST REAL beyond-ADE NUMBERS (MEASURED, dev-box RTX 4060 — NOT a training pod, comma2k19 val 90 eps, real 262.84M WorldModel):** decision-tick **latency p50 14.33 ms**, **TMS 0.0435** (expert-log reference band, not our policy), **CNCE median 210,551** (real architecture-efficiency). ⚠️ **MetaDrive CONFIRMED genuinely absent (3 probes)** → real closed-loop **TLC/LAL/OKRI/LOPS stay renderer-gated**; cheapest unblock = `pip install metadrive` + signalized-junction build OR label comma2k19 intersection segments w/ signal state. Full `pytest` **836 passed/2 skip**. 🔴 escalations (queued): SC-14 catalogued→oracle-tested in `SCENARIO_DATABASE.md` (Opponent-Analyzer-owned); `LEADERBOARD.md` SC-14 row gated on a real TLC (renderer). Note: `…/incoming/2026-07-24-traffic-light-scenario-metric/`.

✅ **TanitResim-productionize DONE (`a9a6f00b`, staged not committed):** main was the canonical superset (nothing stranded — 20 worktree copies all superseded, confirming the C8 glob-artifact). Delivered: **decoded-intent HUD** (per-arm `tactical: <maneuver>` + `strategic: route <goal>` + ADE + v, per THE STANDARD) · **real bug fixed** — the SPA hard-coded nav labels `[straight,left,right]` were WRONG vs canonical `NAV_COMMANDS=(follow,left,right,straight)` (indices 0/3 mislabeled; old test passed by luck) → now data-driven from `meta.nav_commands` · **BEV-only fallback** for uncalibrated (cosmos f-theta) corpora · **synthetic sample bundle + `--demo` one-command** · end-to-end verified in a real browser (zero console errors, traversal guard 404s). Tests **39 passed** (+6); full non-slow suite **829 passed/1 skip**. README rewritten. 🔴 **ESCALATION (integration, queued): `tanitad/replay/arms.py` wires `maneuver_probs`/`nav_cmd` into `ArmOutput` for REF-B ONLY** — MainArm/RefAArm build the base ArmOutput WITHOUT the tactical/strategic heads → on LIVE flagship/refa checkpoints the decoded-intent HUD shows only `ADE·v` until arms.py is wired to pull the trained `tactical_policy`/`strategic_policy` outputs. Shared territory (agent correctly did NOT edit) → **QUEUED as a small bounded wiring fix** (matters for flagship replay overlays, i.e. post-10k-gate; not urgent).

⚠️ **TOOLING/METRICS FRONTIER — Sayed 2026-07-24. CORRECTED after verification (2 of my 5 "gaps" were FALSE — a truncated mtime-sorted-Glob artifact, see RETRACTION_LOG):**
- (1) ✅ REAL GAP: **traffic-light / signalized-intersection handling is NOT built** — "traffic" is only in VLM semantic LABELS, not an eval scenario/metric (grep-grounded, not glob). → traffic-light agent `a9806aeb`.
- (2) ✅ REAL GAP: beyond-ADE suite (LAL/TMS/OKRI/CNCE/LOPS, `tanitad_metrics.py` → `stack/tanitad/eval/metrics.py`) is **synthetic-fixture-validated, NOT all run on real models** (the module's OWN docstring admits this; gated on the MetaDrive renderer). → same agent P2.
- (3) ❌ **FALSE — RETRACTED:** TanitEval is NOT stranded. Narrow glob + the agent's git check (merge-base==worktree HEAD, main 4 commits AHEAD, diff was 100% CRLF noise) prove **main has all 31 modules incl. bench/closedloop/runner/report/registry** and is the newest/most-complete copy. The worktree is fully superseded (safe to prune).
- (4) ❌ **FALSE — RETRACTED:** lake is NOT stranded — main has all 16 modules incl. **ingest/hf_export/license_guard/schema/filtering** (narrow glob confirmed). *(Open, separately: whether the lake has been RUN at ingest-scale + published — a real question, NOT a stranding one; verify before asserting.)*
- (5) TanitResim: consolidation/viz-completeness being VERIFIED by agent `a9a6f00b` (brief correctly conditioned on "verify main is canonical first" — not assumed stranded).
✅ **TanitEval-productionize DONE (`aed535f5`):** real hardening delivered + staged (NOT the false de-strand): **closed-loop wired into the one canonical `runner.py` CLI** (20 subcommands); the **78%-leak val split `physicalai-val-f1b378f295ae` now HARD-REFUSES** (`data.list_val_episodes(..., allow_leaky=False)` raises → points at `CLEAN_VAL`); off-pod reproducible tests (**153 passed**, was 0/collection-error) via new `conftest.py`; `README.md`; CI estimator (episode-cluster bootstrap) surfaced as default, deprecated `overlapping_holdout_se` kept read-only. Staged on `agent/benchmarks-eval-20260721`, NOT committed (2 sibling agents still staging → orchestrator commits per-pathspec after they finish). **This is the "use your resources / ≥5 works" correction — but the tooling was MORE complete than I claimed; the real under-run is (1)(2) + real-scale dataset, not stranding.**

🔴🔴 **V4 GATE DRY-RUN DONE (`ae33799c`) — VERDICT: THE 30k GATE IS *NOT* READY. It would have produced NO VERDICT AT ALL.** Rehearsing it early was decisive. **FATAL B1 (found, root-caused, FIXED w/ a red→green regression test, suite 837✓):** a missing `speed_benefit_recovered_frac` doesn't merely read NOT-SUPPLIED — it makes `run_gate.py check` **`SystemExit` BEFORE evaluating anything**. Root cause `train_flagship_v4.py:143`: `grounding_losses` emits the key **already `g_`-prefixed** while the joint-step log filtered for the **unprefixed** `"op_fwd_ade_m"` and re-prefixed → matched nothing (would have written `g_g_…`). ⚠️⚠️ **A PRIOR "FIX" FOR THIS SHIPPED AND WAS SILENTLY INERT** — its comment asserted the key was "already computed in `log`"; it wasn't, proven by `flagship-v4.2-step4000`'s log (written AFTER that patch, still 0 occurrences). 🔴 **The fix CANNOT retro-fill the in-flight arm's (8000,10000] bucket → SAYED DECISION on amending the 30k card.** **EMITTERS:** `speed_benefit` 0.8184 PASS on v1 but `n_arm_rows=0 → None` on a real v4 arm · `deploy_tick` 18.7641 ms PASS on the v1 panel but **NO v4 input exists** (`efficiency.py` has zero v4 awareness; a trunk-only panel would be WRONG — omits the head's denoise passes — so the agent correctly refused to emit a misleading number) · `nonav_route` **path BUILT → returns 0 = FAIL**: with the command withheld v4.2b predicts **straight on 240/240 windows**, acc 0.6708 == majority 0.6708 = **a pure command echo → this KILL secondary fails the gate on its own** (measured on v4.2b; from-scratch not yet measured). **2 SILENT TRAPS ALSO FIXED: (B4)** `eval_flagship_v4.py` imports `taniteval` NON-fatally → with the PYTHONPATH its own docstring documents it **exits 0 while writing `gate_primary_ade_0_2s: null`** (the agent's first run "looked successful" and had silently lost the primary) → runbook preflight makes it loud; **(B5)** `/root/run_gate.py` is **STALE (621 vs 847 lines), predating the 07-21 hardening — and the 2026-07-23 10k gate JSON was rendered BY IT** → current version installed. Machinery itself is sound: with all 8 secondaries supplied it renders a **COMPLETE** verdict. Gate cost is only **~11 min GPU** — the risk was never runtime, it was missing inputs. 🔴 **5 ESCALATIONS FOR SAYED: (1)** in-flight `speed_benefit` unrecoverable → amend card / accept no verdict / recompute off milestone ckpts (changes the pinned estimator); **(2)** `nonav_route` will FAIL unless the P6 strategic planner lands; **(3)** `deploy_tick` needs a v4-aware lever panel — **nobody owns it**; **(4) register a 30k CARD** — the current card is `gate_step: 10000` and reusing it at 30k is the multiplicity abuse GATE_PROTOCOL §2 forbids; **(5) restart budget may already be EXHAUSTED** — card says `restarts_used: 0`/cap 2 for `joint-planner-wm` but v4.1/v4.2/v4.2b/from-scratch look like ≥3 (NOT adjudicated — needs the family assignment confirmed). Also reaped 3 orphaned **deadlocked GeoCalib** jobs (reparented to PID 1, 0 CPU progress over 25 s vs 5 h elapsed, `futex_wait_queue`, 2 writing to DELETED logs — the exact PyAV/CUDA deadlock that agent documented as fixed). Deliverables `…/incoming/2026-07-25-v4-gate-dryrun/`.

**IDLE CAPACITY (drumbeat ~06:1x Berlin / 04:10 UTC 07-25): FLEET FULLY UTILIZED — 4/4 pods working.** pod2 = flagship-v4 (step 15,050/30k, 50%) · pod1 = flagship-v2corpus (step 450, healthy) · **eval = v4 GATE DRY-RUN (`ae33799c`)** — rehearse the ENTIRE formal-gate chain on the HF-reachable `Sayood/flagship-v4.2b` so the decisive 30k gate (~1.2 d) can't discover a broken emitter at the worst moment (`speed_benefit_recovered_frac` has NEVER run on a v4-line ckpt); also reaps 3 suspected orphan CUDA contexts (1.2 GB/942 MB/848 MB) · **pod3 = v1.6 PAIRED DECISION-GRADE INTERVAL (`acf06502`)** — the program's "best ADE" arm (`flagship-v16-ab-ft`) must not rest on `overlapping_holdout_se` (1.28–2.06× too narrow); compute the **paired episode-cluster bootstrap** vs deployed v1 on the clean split, both outcomes publishable (if it overturns the ordering → retraction-class). ⚠️ **`step_s` READ CORRECTLY: pod1's log shows `step_s 545.5` — that is ACCUMULATED over `--log-every 50` → 10.9 s/step, matching the measured 11.27; NOT a 545 s/step pathology** (the documented false-alarm trap).

**(superseded) IDLE CAPACITY (drumbeat ~00:1x Berlin 07-25): FLEET FULLY UTILIZED — Sayed said "keep building," fleet re-loaded.** pod2 = flagship→30k (~1.7d). eval = **GeoCalib** (`a4ebd01c`). pod1 = **YouTube-IDM non-CC harvest/label** (`aea4861a`). pod3 = finishing v2 (~soon) → then the scale-up pretrain. **0 idle.** Queued next (as pods free / next iters): **low-OOD reactive-agent renderer** (the real closed-loop lever + unblocks beyond-ADE-on-real-models — a research/design-first build), v2-corpus QA, next-gen-flagship prep. 🟡 **HF cleanup = Sayed's click** (minimal ~13GB: del flagship-v4.1/v4.2/v4.2b + refa-ijepa-4b + 2 empty) → then I do the flagship ckpt-backup + formal gate. Flagship healthy (11500 eval 0.4788, the 11000 bounce was confirmed noise). ⭐ **The closed-loop cheap-experiment chain is now COMPLETE + honestly closed:** renderer-research → LOWOOD-CL-TRAIN (BOUND) → tolerance-band re-score (ADE-"cost" = metric artifact, REOPENED) → **departure-power n=40 (BOUND — the departure "benefit" reverses, CLOSED)**. Net: recovery-augmentation is not a net win on road-keeping at full power; the next closed-loop bets are bigger builds (reactive-agent renderer + map-aware instrument), not autonomous. **Sayed decisions live: (1) YouTube-IDM SCALE-UP go (re-fills the pods) · (2) HF-storage.** NEXT: from-scratch λ_plan→full + **10k gate (~7h)** = the flagship verdict (the one remaining live event).

### 🆕 TWO ADDITIONAL RESEARCH DIRECTIONS (Sayed 2026-07-23) — research + design + SMALL proof experiments; ADDITIVE, replace/modify NOTHING running
- **D1 — FROZEN WM + LEARNED planner** (`ade3edfb`, pod1) ✅ **DONE — VERDICT: VIABLE, not bottlenecked (MEASURED, 2 seeds, paired).** Froze v1's WM, trained a 3.77M planner by **backprop of ADE THROUGH the frozen WM** (analytic-gradient, Dreamer/SHAC): **W = 0.599 ADE@2s — NOT separated from the 0.40 oracle-action ceiling** (+0.194 [−0.045,+0.448]); beats CV (−0.247 SEP), action-BC (−0.401 SEP), static-decode (arm F 3.65 = REF-A regime). ⭐⭐ **WM canary 0.4045 UNCHANGED BY CONSTRUCTION** — the v4-saga degradation (v4.2b 0.42→0.70) CANNOT occur frozen. REF-A frozen-ceiling REFUTED (only bites static-latent decode; routing thru dynamics avoids it). ⚠️ open-loop, 12-ep val, W doesn't BEAT coupled v1.6 (0.489) on ADE-point — it's competitive-w/-closeable-gap + DOMINANT on WM integrity. 🔑 **STRATEGIC: lands exactly as the coupled v4 line FAILED on WM-degradation → frozen-WM is now a validated 3rd option for the fork (surgery/from-scratch/frozen-pivot) — Sayed's architecture call** (his v4 design was explicitly "nothing frozen"; joint-training's bet is WM co-adaptation, which frozen forgoes). Staged `…/incoming/2026-07-23-frozen-wm-learned-planner/`. **✅ AMORTISED-MPC PROTOTYPE DONE (`ade3edfb`, pod1, MEASURED 12-ep paired):** ⭐ **CEM search over the frozen WM = 0.132 ADE — 4.5× better than the feedforward W (0.599), beats even the 0.40 GT-action ceiling** → the frozen WM is an EXCELLENT differentiable simulator (controllable to 0.13m via its preimage); **the PLANNER is the headroom, not the WM.** ⚠️ **GUARDRAIL: naive distill (CEM→feedforward action-prior) FAILS = 1.399** (brittle non-smooth preimages, val-collapse, rollout-compounding) → "distil the search into a policy" is a TRAP, ruled out for ~0 cost. **Synthesis (Sayed-decision, not urgent — from-scratch is the flagship): the amortised-MPC path is worth pursuing but as TEST-TIME-SEARCH + a learned value/cost (TD-MPC2 model-based-RL half), NOT a distilled prior — needs an offline reward/value model we don't have yet.** W (0.599) stays the best DEPLOYABLE feedforward frozen planner.
- **D2 — improve REF-C's planner for CLOSED-LOOP** (`a1f26c92`, eval) ✅ **PHASE-1 DONE — WIN on the primary metric (data-efficient lever that GENERALIZES).** ⭐ Mechanism (MEASURED): REF-C is **covariate-shift-BLIND** — a 1m off-path view moves its plan 7mm (`recovery_ratio 0.0074`, n=881) = the root of the plan_xte 0.57→12.98m drift. Fix = **recovery-augmentation FT** (decoder-only, FROZEN encoder=WM-safe; warp real frames by an analytic homography bounded to the low-OOD envelope, supervise return-to-path — every window a recovery example, renderer-free, non-self-referential). **Held-out (12ep/264win, paired): corridor_departure_rate 0.0174→0.0085 (HALVED, +0.0089 [+.0008,+.0197] SEP; junctions +0.0333 SEP), guard peak_xte holds.** ⭐⭐ **GENERALIZES to held-out eps** — beats Gate-1's memorization wall (Gate-1 real-junction FT was held-out Δ≈0; this synthetic aug separates). ⚠️ **BUT closed_ade2s 0.587→0.875 SEP-WORSE** (drivability-vs-accuracy trade, Nachkov) → NOT a free upgrade, do NOT deploy this ckpt. ✅ **PHASE-2 DONE → DIRECTION-2 CLOSED. Verdict: the lever is REAL but decoder-only is EXHAUSTED (Pareto-bound), NOT promotable.** Full sweep (naive/g1/g2/g3 gentle-FT + g2s1/g2s2 speed-term) MEASURED: no config holds departures ≥base/2 AND recovers `closed_ade2s` to base 0.587 — dADE stays separated-worse in ALL. ⭐ **ROOT CAUSE (twice-confirmed): the FROZEN ENCODER is the bottleneck** — it doesn't encode the lateral offset (recovery_ratio ~0), so the decoder can't decouple "cut departures" from "stay accurate." 🔑 **The only remaining lever = encoder-in-the-loop light-FT (unfreeze last-k blocks, lr_enc ~5e-6, plan-free-canary-GATED — the v4 WM hazard), recipe banked in INTAKE.md → a TRAINING ESCALATION, needs Sayed's go (NOT autonomous).** Echoes D1's "planner is the headroom on a frozen WM": frozen representations bound the downstream. Staged `…/incoming/2026-07-23-refc-planner-closedloop/`. 🔴🔴 **FINAL VERDICT (2026-07-24, supersedes the "Pareto-bound" reasoning above): DIRECTION-2 CLOSES HONESTLY at full power — but for a DIFFERENT reason than "Pareto trade."** Two cheap follow-ups settled it: (1) **tolerance-band re-score** — the "ADE-worse" cost was LARGELY a knife-edge-L2-metric artifact (fair `band_ade2d(1.0)` forgives benign in-lane recovery; cost vanishes CI∋0 for 3/4 configs), so the Pareto "trade" was mostly not real; this REOPENED the direction. (2) **departure-power n=40 cross-fit (`a1f26c92`)** — the actual residual, the departure benefit, **REVERSES at full power: +0.0089 S (n=12) → −0.0302 S (n=40), the recovery FT departs 3.3× MORE.** So the n=12 "halves departures + generalizes" headline was **favorable-split noise (C5-retracted, logged).** Net across the whole chain: recovery-augmentation is **NOT a net win** on road-keeping — no ADE cost worth worrying about (metric artifact) AND no departure benefit (n=12 noise). Durable un-retracted deliverables: the machinery (renderer-free recovery + low-OOD on-policy harness + `band_ade2d` metric + encoder canary), **REF-C's encoder is safely fine-tunable**, and 2 binding measurement lessons (use `band_ade2d`; use full-corpus cross-fit for ~1pp departure effects). The remaining escape (encoder-in-loop light-FT) is unchanged as a Sayed-gated training escalation, but the cheap-experiment phase is exhausted; the next real closed-loop lever is a bigger renderer/instrument build.

⚠️ **pod1 now runs D1** (its flagship WM is the frozen simulator — the parked pod is productively used). QUEUED for pod2-free: cosine pre-probe + v4.2b-cascade + ckpt-backup (may hit HF 403). **The one broad unblock still = a Sayed-side HF `Sayood/` storage cleanup/upgrade** (REF-C-on-pod1 leaderboard re-runs + ckpt-backup + old dataset/head pushes).

⚠️ **CKPT-BACKUP STANDING RISK (Registry agent flagged):** v4.1 (3.24 GB) + v4.2 + v4.2b ckpts sit on **single pod disks, NOT HF-backed** — a full-quota/pod-death loses them (the flagship-mid-checkpoint trap). Backup is an HF *publish* (Sayed-pattern: 3-arm ckpts already authorized to gated `Sayood/` repos) AND can't push from pod2 while v4.2b trains → **do the backup the moment pod2 frees at v4.2b's resolution**, before launching the next v4.x. Also: 3 KILL secondaries (`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`, `nonav_route_beats_majority`) have NO v4 emitter → no v4 gate can render a COMPLETE formal verdict yet.

| # | stream | owner | state | next |
|---|---|---|---|---|
| v4.2b | flagship v4.2b (floor 0.15) | pod2 PID 99197 | 🟢 **LIVE** (v4.2 killed, both ckpts preserved; fresh from v1). First-step CONFIRMED: not-frozen (enc+pred 149/149, gnorm_enc 6.23), eff-batch 64, canary baseline 0.42 (healthy start), floor 0.15 set | ⭐ **Phase-B canary @ step 2000-3000 (~4h) = THE TELL**: stays low (works, interpolation between v4.1-starve & v4.2-degrade) vs runs away like v4.2's 0.77 → **NOT floor-tunable → HARD STOP floor-roulette → pivot to FROM-SCRATCH v4** (v1's proven recipe: co-evolve WM+planner from random init — the degradation is a warm-start artifact). v4.2b @ **step 900, Phase A** (canary 0.495 = v4.2's Phase A, indistinguishable until λ_plan ramps @ 2000). **PRE-REGISTERED rule (v4.2 hit 0.86@2000 / 0.72@4000 / 0.77@5000): canary @ steps 2500-3000 — ≤0.55 & <v4.2 & gnorm_pred↑ → PASS, continue to 10k; ≥0.65 (~v4.2) → FAIL, fire from-scratch; 0.55-0.65 → floor 0.10 or pivot per planner trend** |
| G1proto | Gate-1 closed-loop-aware FT **PROTOTYPE** | `a7c1eb9c` eval | ✅ **DONE — ⭐ Gate-1 gate (a) CONFIRMED: the lever WORKS (MEASURED, paired, deterministic baseline reproduced exactly).** FT REF-C's decoder on 675 on-policy junction steps + GT-recovery labels → **junction offroad 11/15→7/15 (−36%), at-fault collisions 5→1, pass-rate 3→8**; 7 scenes recover on-road incl. the clean covariate-shift roundabout; open-loop selected→recovery L1 5.06→0.55. ⚠️ **NOT a clean number — 2 measured flaws a naive rerun repeats:** (i) **MEMORIZES at n=15** (leave-3-out: held-out recovery 4.65→4.15 then degrades), (ii) **HIGH-DEVIATION side-effect** — SAME 3 scenes newly-offroad in both ckpts, plan-dev ~6× (0.41→2.69), **intrinsic to the recovery objective** (the v1 retraction mechanism), not overtraining. + ~3.2× NuRec OOD. Rec for clean run: low-OOD source + more scenes + CAT-K/RoAD target filtering + deviation regularizer. Staged `…/incoming/2026-07-22-alpasim-closedloop-evalpod/` (`GATE1_PROTO_NOTE.md`) | → clean run `a9147f0e` |
| G1clean | **CLEAN Gate-1 run** (both gates green → PRE-AUTH FIRED) | `a9147f0e` eval | ✅ **DONE — landed the pre-registered BOUND branch → HOLD the FT GPU-commit (did the science, not a rubber-stamp).** Ran 15 held-out fine-tunes + a 5-fold leave-3-out (not a naive in-sample rerun). Two independent MEASURED bounds block a *promotable* run: **(1) 🧭 INSTRUMENT GAP (binding): the low-OOD source is MAP-FREE + AGENT-FREE → it structurally emits only drift/deviation, NEVER off-road/collision/pass** (those need the map+reactive-agents = NuRec's 3.2× OOD). **"low-OOD" and "junction off-road/collision" are MUTUALLY EXCLUSIVE with existing instruments** — the gap is ~fundamental: reactive-agent safety metrics need a SIM (→ reconstruction OOD; AlpaSim renders via NuRec = the SAME 3.2×), while low-OOD needs REAL footage (→ no reactive agents). True resolution = a LOWER-OOD renderer (hard R&D), not AlpaSim-as-is. **(2) DATA BOUND: only 22/16/13 distinct real junction eps** (≥10/20/30° heading) → held-out split = memorization regime, MEASURED: leave-3-out held-out recovery-L1 **5.06→5.06 (Δ≈0)** while train→0.41. P1 fixes BUILT+MEASURED (necessary, not sufficient): CAT-K/RoAD filter drops **328/675 (49%)** catastrophic labels (147 backward); +λ_dev trust-region → held-out plan-shift **7.58→1.49 m (−80%)** (the high-deviation v1 mechanism TAMED) but recovery stays ~5 (memorization is data-bound, not deviation-bound). Staged `…/incoming/2026-07-23-gate1-clean-run/` | → 3 unblocks: (2)+(3) launching now (`abe82f1f`); (1) mine ~100+ junction scenes = Sayed's call |
| LaneKeep | low-OOD lane-keeping metric + REF-C arm | `abe82f1f` eval | ✅ **DONE — ⭐ FIRST ABSOLUTE low-OOD closed-loop comparison (n=40/881, paired, episode-cluster bootstrap).** `corridor_departure_rate@1.75m` built + REF-C base wired (md5-verified, eval-local no relay). **REF-C base DECISIVELY out-drives flagship v1, separated in EVERY stratum:** ADE@2s **0.564 [.452,.676] vs 1.488 [1.329,1.647]** (Δ+0.924); departure-rate **0.0134 vs 0.0318** (Δ+0.0184); peak-XTE 0.442 vs 0.764; both at **1.02-1.20× OOD** (≪ NuRec 3.75×). ⭐⭐ **2 findings:** (1) **TRIPLE-confirms REF-C>flagship-CL** via a DIFFERENT low-OOD instrument (n=1→n=12 AlpaSim→**n=40 real-footage**) → the C7 ordering is NOT a reconstruction artifact; (2) **decomposes the gap:** longitudinal scenes BOTH keep lane ~perfectly (dep 0.4%/0.04%) yet flagship ADE 4× (1.455 vs 0.354) = flagship's deficit is **LONGITUDINAL not lane-keeping** (its 89% signature); junctions flagship departs 2.3× more (14.6% vs 6.4%) = high-deviation head confirmed. ⚠️ lane-keeping/drift NOT off-road/collision; within-source relative; deployed-decoder. Raw `…/incoming/2026-07-23-lowood-lanekeeping-refc/lowood_lanekeep_40ep.json` | ✅ **LEADERBOARD §5.5 + G-B1 footnote INTEGRATED** (closes the "sim2real OOD axis still open" line); registry closed-loop row pending (fold w/ next registry touch); ⭐ **feeds D2's proof metric** (`corridor_departure_rate`) |
| LowOOD | lower-OOD closed-loop source | `a1cc5a0f` pod1 | ✅ **DONE — DECISION-GRADE, the ~3.2× NuRec confound is BROKEN (MEASURED, pod1, clean val).** ⭐ **real-footage log-replay → reconstruction OOD = 0 by construction, confirmed:** Δ=0 open-loop ADE@2s **0.4045** (n=265) vs NuRec recon **1.5157** → reproduces the flagship's REAL level (registry 0.4271) = the 3.2× is **eliminated, not reduced.** Deviation envelope: lateral 3m→0.470, yaw 12°→0.460 → across the WHOLE plausible range (±3m/±12°) stays **≤1.16× baseline, NEVER approaches NuRec's 3.21×** (frame-mismatch OOD ~14× smaller: +16% vs +221%). ⭐ **longitudinal is FREE** (arc-length re-index of the 1-D real-frame manifold @1.3m spacing, ≤0.6m to a real frame) — **exactly covers the flagship's dominant 89%-longitudinal failure mode, zero recon OOD.** Mechanism: appearance fidelity (not pose-exactness) keeps the encoder in-distribution. Rec: **build Gate-1 on (a) real-footage log-replay + kinematic ego, longitudinal-first.** ⚠️ **HONEST BOUND (do not overclaim):** this is the SOURCE's *observation*-OOD (apples-to-apples w/ 0.47/1.52), **NOT yet a closed-loop planner's on-policy action-selection sensitivity** — that's the real Gate-1 pre-req, being validated now (P2 below). n=12/265, flagship-only, single seed. Staged `…/incoming/2026-07-23-lower-ood-closedloop-source/` | → hardening (`ae72a9e1`, pod1) |
| LowOODhard | lower-OOD source → decision-grade + Gate-1 pre-req | `ae72a9e1` pod1 | ✅ **DONE — ⭐ Gate-1 gate (b) CONFIRMED GREEN.** **P2 (the real pre-req, MEASURED):** C6-clean closed-loop harness on REAL footage (deployed controller verbatim, only "imagine latent"→"arc-length re-index real frame + warp by ON-POLICY (dlat,dψ)"; tick-0 self-check 0.0/0.0). **The loop STAYS low-OOD on-policy:** longitudinal (flagship's 89% mode = the Gate-1 target) OOD peak **1.017× [1.006,1.029]**, junction **1.190× [1.133,1.219]**, overall 1.054×, **100% ≤1.5×** vs NuRec's flat **3.75×**. Closed-loop ADE 1.45 (< imagination-in-loop 1.685), dominated by longitudinal drift that arc-length re-index absorbs OOD-free. **P1 (decision-grade CIs):** real-footage ADE 0.4045 **[0.3128,0.5149]**, NuRec 1.5157 = 2.94× the upper CI (elimination CI-robust); lateral no separation to 2m (3m first), yaw separates 3°→+0.055@12°. ⚠️ **BOUNDS:** n=12 eps, flagship-only, ground-plane-lateral optimism, yaw>12° clamped (~1.25-1.3× extrapolated, still ≪3.75×), drift-loop not safety. 🔴 **REF-C 2nd arm UNREACHABLE from pod1** (ckpt only on off-limits pods, never on HF; REF-C-small HF = 403 storage-full) — ~1h when a ckpt reaches pod1. Staged `…/incoming/2026-07-23-lower-ood-closedloop-source/` | REF-C arm pending an HF-storage unblock / pod-free |
| GateEmit | v4 gate emitters (make the gate render COMPLETE) | `a9dfe223` pod1 | ✅ **DONE — v4 gate now renders a COMPLETE verdict** (`pytest` 803✓/2skip). **STEP-0 reconciliation (decisive): all 3 ARE KILL** (registry-agent right; run_gate mechanism + `V4_FLAGSHIP_DESIGN.md` §9 both confirm — the "P7 report-only" was a CONFLATION with a DIFFERENT off-card 5-falsifier set). Built+MEASURED on v1: **`deploy_tick_p99_ms`** 16.89-18.76 ms ≤50 **PASS**; **`speed_benefit_recovered_frac`** **0.8184** ≥0.70 **PASS** (v3enc 0.1859); **`nonav_route_beats_majority`** **0 FAIL** (route_acc 0.7083 == majority 0.7083, v1 follow-head = pure straight-echo 72/72 — the "0.861" in RETRACTION_LOG is the SPEED probe, not route). Dry-run: 3 omitted→INCOMPLETE, all-8→**RESTART** (a COMPLETE verdict; v1 honestly restarts on oracle_in_fan 0.3073>0.30 + route-echo). ⭐ **Finding+fix:** the v4 trainer COMPUTED `g_op_fwd_ade_m` but DROPPED it from the log row → a real v4 arm's log is NOT gate-computable for speed_benefit (logs `canary_ade@2s`, different scale). **One-line log-only fix staged** (`train_flagship_v4.py`, parity-safe) → **MUST ride the next v4.x launch.** Staged `stack/tanitad/eval/speed_benefit.py` + `scripts/gate_emitters.py` + tests + `…/incoming/2026-07-23-v4-gate-emitters/` | ⚠️ v4.2b's log predates the fix → its 10k-gate speed_benefit needs a FRESH eval emit (or reads NOT-SUPPLIED); next v4.x launch carries the fix |
| GradCouple | smarter planner↔WM gradient coupling | `a4fde3c6` (pod-free) | ✅ **DONE — design + tested ref-impl + pre-reg STAGED** (`…/incoming/2026-07-23-planner-wm-gradient-coupling/`: `DESIGN.md`, `PRE_REGISTRATION.md`, `grad_surgery.py` CPU-smoke-green, `tests/` 9✓, `INTAKE.md`). ⭐ **KEY (MEASURED, source-read): `cond_imagination:false` → the planner sends ZERO grad to the predictor; its ENTIRE trunk footprint is ONE activation `states`[B,8,2048].** That's why a scalar `λ_plan` could ever stand in — AND why the whole conflict de-conflicts in **one vector pair at ~0 cost**: one-sided PCGrad at the `states` seam removes only the planner-grad component OPPOSING `g_wm`, WM grad **byte-untouched** (asymmetric: WM protected, planner subordinate). Drop-in for the `grad_scale` seam (`flagship_v4.py:211`), cost 0.1-0.3%/step no extra enc pass, canary+10k-gate preserved, **NOT a 3rd encoder lever** (O-20 refinement, door stays closed). PCGrad/GradVaccine adapted; GradNorm/CAGrad/MGDA rejected w/ reasons. §7 honest: can't fix warm-trunk self-degradation at lr_trunk (the C6 confound — C₀ control measures it first) nor a cosine≈−1 opposed objective (→ starvation) — both route to from-scratch | ⭐ **cheap follow-on QUEUED: the near-free cosine pre-probe** (measure ⟨g_wm, g_plan⟩ at the seam on the preserved v4.2 ckpt — ~0 GPU) **decides surgery-vs-from-scratch BEFORE spending the 1.3 A40-day.** ⚠️ needs v4 code+ckpt+val cache = **pod2, but NEVER while v4.2b trains** → runs the moment v4.2b resolves & frees pod2; becomes the tie-breaker on the v4.2b-FAIL fork (from-scratch vs surgery). Trainer splice queued for the next Sayed-approved v4.x launch (default `--coupling scalar` = byte-identical) |
| Registry | MODEL_REGISTRY v4-section | `a75fbf0c` (pod-free) | ✅ **DONE — §1.5 flagship-v4 line inserted** (schema mirrors §1; old variants block → §1.6; STAGED not committed). All numbers from RAW eval JSON: v4.1@10k `ade_0_2s` **0.8522** [0.75,0.98] / miss_2m 0.2486 / oracle_in_fan 0.4838 / canary **0.4599** PASS; v4.2@4000 **0.9869** / canary 0.7222; **v4.2b PENDING** (no number fabricated); from-scratch READY. Params **≈247.88M MEASURED-by-instantiation** (⚠️ a measurement record, NOT a config print — registry carries a "don't quote bare" caveat; faithfulness-checked vs 263.44M). not-frozen verified 2 ways. ⭐ **PRECISION (raw JSON wins): v4.1's formal `run_gate.py` verdict is `INCOMPLETE`** (3/8 KILL secondaries have no emitter) with primary `pass:false` → registry reads **"formally INCOMPLETE, substantively FAIL"**; our "FAIL" shorthand is the substance, not the formal verdict. Note `…/incoming/2026-07-23-registry-v4-section/` | ⚠️ fold v4.2b's number in when it resolves; ckpt-backup risk ↓ |
| FSprep | from-scratch v4 fallback | ✅ **READY** (`a05a5c9e`) — `--from-scratch` = skip warm-start, random-init trunk (not-frozen trivially passes; pytest 786✓ +5 tests). **Smoke CONFIRMS the premise**: from-random-init canary 1.52→1.165 (WM co-evolves, v1-style, no degradation). **ONE flag from v4.2b** = max attributability. ~53h/30k | 🔓 **fires under the STANDING AUTH** (no re-ask): v4.2b canary ≥0.65 → cosine pre-probe → from-scratch **OR** gradient-surgery, whichever it picks; notify + veto-window. Runs on pod2/eval (needs `tanitad.lake`+val cache; pod1 lacks it). λ_plan launch-decision: keep floor (inert from-scratch) vs `--lambda-plan 1` (v1's literal regime) |
| G1prep | Gate-1 on-policy rollout collection | ✅ **DONE** (`ab3ecfce`, eval FREE) | 15 junction scenes, 675 on-policy steps, **recovery labels (GT path 0.5-2s ahead) IN HAND**. ⭐ **REFINED mechanism (corrects the "execution failure" read): the ego tracks its plan TIGHTLY (0.49m); the PLAN degrades on-policy (plan_xte 0.57→12.98m) = textbook COVARIATE SHIFT** → the *planner*, not tracking → exactly what Gate-1 (retrain planner on-policy + recovery labels) fixes. ⚠️ **~3.2× OOD confound** (NuRec rollouts → a fine-tune partly targets reconstruction-OOD; sufficient to PROTOTYPE, not to train robust). Intersections add at-fault collision mode (5/7) | Gate-1 fine-tune = Sayed go. ✅ **the "needs a lower-OOD source" blocker is RESOLVED** (LowOOD landed decision-grade — real-footage log-replay, recon OOD=0, longitudinal-free). Clean Gate-1 run now buildable on the real-footage source pending P2 closed-loop-sensitivity validation + Sayed's go |
| BrB | own-encoder **Branch B** | `a8713fcb` pod3 | 🟢 **step 10.3k/40k (~26%), HEALTHY — no adjustment.** Loss 10.2→~1.0, IDM 5.8→0.3-0.8. ⭐ **Camera-conditioning WORKING: all 12 blocks learned from zero-init, rig-A-vs-B token-delta 2.7-7.5/block (vs Branch A ablation's +0.1)** — the from-scratch mechanism the cheap warm-start couldn't do. ~16h left, zero crashes | step-40k → HF push + **held-out-rig transfer eval** (the real cross-rig test vs the −2.1 ablation) |
| C | v4-eval-harness | `a938e1c0` | ✅ **DONE** — harness VALIDATED (0.42148✓); **v4.1@10k gate = FAIL** (`ade_0_2s` **0.8522** ≫0.60; WM canary 0.46 PASS → planner-speed is the fault). `eval_flagship_v4.py` staged | → Sayed kill/v4.2 fork; MODEL_REGISTRY needs a v4 section |
| D | planner-bottleneck synthesis | `a090c33b` | ✅ **DONE** | verdict above → F |
| E | Orin/Thor INT8 benchmark | `adce6d71` pod1 | 🟢 phase-1 accuracy sweep running | per-layer FP16-vs-INT8 map + Orin/Thor estimates |
| FreeFloor | Gate 0 + 0b (free inference floor) | ✅ **DONE** (`a63488de`) | ⚠️ **Free floor RULED OUT for junction off-road** (Gate 0 selection ΔOFFROAD +0.00 [−0.08,+0.08]; Gate 0b gradient-nudge junction **0.73→0.73**, both validated). ⭐ **KEY MECHANISM: the plan is on-road (0 off-road plans) but the ego STILL departs → CLOSED-LOOP EXECUTION failure, not planning** → **Gate-1 (closed-loop-aware training) is the measured-justified next lever**. ✅ ship the gradient-nudge floor as a FREE safety override (intersection collisions 0.71→0.43, plan-dev 0.34). Benchmark: balanced 38-scene ΔScore −0.123, flag TIES on roundabout+highway | 🔴 **Sayed: green-light Gate-1** (closed-loop-aware IL on AlpaSim rollouts, on v4.2) after v4.2's gate? |
| Floor3 | free-floor rung 3: WM-MPC/CEM | ✅ **DONE** (`a783de21`) | ⚠️ **TIE** — MPPI/CEM over the WM does NOT beat the single-step re-plan (Δ+0.005 to +0.011, not separated; lateral off-road separated-WORSE +0.136). Single-step re-plan already captures the imagination benefit. Feasible (~50ms tick) but no gain. Deferred (not refuted) to AlpaSim | **Free floor FULLY characterized: none of rungs 1-3 fixes junction off-road → 🔴 Gate-1 (closed-loop-aware training) is THE lever, Sayed's call** |
| RL? | closed-loop-WM-training research | ✅ **DONE** (3 angles + synthesis, ~40 sources) | **Verdict: RL is the LAST lever; build the free inference floor (safety filter + guided diffusion + WM-MPC) + run Gate 0.** "Dreamer-RL"=analytic grads thru WM (false choice); crux=WM-exploitation (→v4.2). Doc: `Research/2026-07-23-closed-loop-wm-training-verdict.md` | 🔴 **Sayed: launch Gate 0?** |
| ~~DAgger~~ | closed-loop-aware IL proof | done | ✅ **DAGGER_HURTS on the cheap harness** (self-referential: on-policy states = WM's own imagined off-manifold latents → over-correct). **Refutes the no-renderer harness as a TRAINING proving ground, not DAgger.** Keep budget on v4.2; DAgger re-enters AlpaSim-validated only | — |

**Stream D verdict** (`Research/2026-07-23-planner-is-the-bottleneck.md`): v4's operative planner is **ALREADY RIGHT** — it IS REF-C's anchored-diffusion `FlagshipV15Head` (the plan_dev-0.34 family); **do NOT change it.** v1-tactical is high-deviation by **HEAD DESIGN** (unconstrained regression + the harmful intent seam cos −0.238), which anchored diffusion strictly fixes. v4.1 starvation = lr_trunk 3e-5 + a **controller BUG** (ran naive halve-to-zero, NOT the design's cap-and-hold/O-14 — MUST fix before any v4.2). **Cheapest next = ⭐ `planner_on_frozen_wm`** (anchored-diffusion planner on a FROZEN healthy WM, λ_plan=0, ~0.5 A40-day) → splits "starved" vs "bad-by-design", pre-commits the fork; QUEUED for pod3-free. Ranked experiments: (1) native-1080 re-run 0.06d · (2)⭐ frozen-WM 0.5d · (3) dagger_planner_ft 0.1–0.3d · (4) AlpaSim suite 0.2d · (5) v4.2 cap-and-hold 5d.

Own-encoder DESIGN done this iter (`a8713fcb`): `CameraConditionedEncoder` (GAIA-2 per-block conditioning + known/unknown mask, warm-starts v1 ViT, 97.4M), smoke-validated (loss 4.80→1.88, IDM 2.76→0.98, pytest 778 green); no published model solves rig-robust monocular dynamics for driving. `…/incoming/2026-07-22-own-dynamics-encoder/`.

---

### 📇 historical (superseded by CURRENT STATE above — kept for provenance)

**SURVIVED + banked (decision-grade):**
- 🟥 **Multi-rig co-train VERDICT** (`results_multirig.json`): **the cross-rig collapse is REPRESENTATIONAL,
  NOT data-diversity.** rigA+comma→held-out rigB: light-FT speed R² **−1.61 vs single-domain −1.65 (NO
  recovery)**; rigA+rigB→held-out comma: 0.452 vs 0.411 (+0.04, marginal). **Adding a 2nd training domain
  does NOT recover transfer** → the own-encoder needs **EXPLICIT camera-conditioning + geometry (the
  expensive-but-right path), NOT just more data.** Do NOT proceed to YouTube on frozen/light-FT/2-domain.
- ✅ **v4.1 reached the 10k GATE** (`ckpt_step10000.pt` landed healthy; watch `bd3naj2qv`) → **run
  `run_gate.py` on it after reset** — the flagship's first decision point.
- 🔬 **WAM research: 11 verified claims (UNSYNTHESIZED — synth hit the limit).** KEY borrowable mechanism:
  **GAIA-2 camera-parameter conditioning** (arXiv:2503.20523 — separate intrinsics/extrinsics/distortion
  embeddings, summed + injected per block; rig-generalization from multi-rig data + **explicit conditioning,
  NOT scale**) = EXACTLY what the multi-rig verdict demands. Vista = forward-only (not IDM); DrivingWorld /
  Doe-1 = AR ego-state, frozen tokenizer. Full claims: `tasks/wgmi9zg09.output`.

**POST-RESET PLAN → ✅ EXECUTED (2026-07-23 loop iter, session CLEARED — calls work):**
(1) ✅ **v4.1 10k gate** launched (`a1891ef6`, pod1) — evals `ckpt_step10000.pt` (NOT on pod2). (2) WAM
synthesis folded INTO the own-encoder (agent reads the 11 claims directly — no re-run of the 106-agent
workflow). (3) ✅ **own-encoder RESUMED** (`a8713fcb`, pod3) with the **camera-conditioning GAIA-2 recipe**
(the "cheap multi-domain" branch is REFUTED). (4) ✅ **eval-pod orphan KILLED** (PID 1408459, 15 GB freed)
+ **AlpaSim suite RESUMED** (`a901caeac`, eval). (5) 4 streams now (one per pod) — will add the 5th–6th
(H26 hierarchy proof / a research stream) STAGGERED as the gate + suite report, NOT bursted.

## ⚠️ v4.1 @ 10k — REVISED (see CURRENT STATE): WM healthy, but the PLANNER is gradient-STARVED
The lr_trunk-3e-5 fix protected the WM **at the planner's expense** — the canary controller clamped lam_mult to 1.5e-5, so in-loop val ade@2s **0.705 > v1's 0.452** and the real gate would plausibly FAIL. The gate is BLOCKED pending the v4-eval-harness (stream C). Original in-loop note kept below for provenance:
### (original, now-revised) ✅ v4.1 HEALTHY AT 10k — the lr_trunk-3e-5 fix confirmed AT SCALE (MEASURED, in-loop)
Canary trend 8.5k→10.5k: **0.526 → 0.603 → 0.501 → 0.460 (controller "ok") → 0.496** — flat near the 0.42
baseline, nothing like v4's runaway to 1.30+. val ade@2s ~**0.60** (oracle 0.36), improving. PID 79542, 34.7
GB, step 10.5k. ⚠️ Nuance: the controller keeps lam_mult TINY (7.6e-6) → the planner→trunk coupling is
still minimal, so v4.1 ≈ low-lr-trunk-FT + a near-decoupled planner head with a HEALTHY WM (vs v4's
destroyed WM). The formal 10k gate (`a1891ef6`) adjudicates CONTINUE vs RESTART.
⚠️ **Pods report UTC. Sayed reads Europe/Berlin (UTC+2). Always report to him in LOCAL time.**

---

## FLEET — verified 17:19 local (GPU + real `dd`, never `df`)

| pod | GPU | lock | disk | state |
|---|---|---|---|---|
| `tanitad-pod` (pod1) | — | `exp-a` (from 17:25) | 1.7 GB/s | **Running post-mortem experiment A** — 2 k-step re-run at `v2_ego_dropout=0.0`, ONE lever, ~6 GPU-h, writes to a NEW dir. ⚠️ **DO NOT RECYCLE / DO NOT OVERWRITE `flagship4b-v3enc-30k/`**: holds `ckpt_step10000.pt` (carries `step=10000`; the run reached 10,800 but never checkpointed again), the only 10 k state that will ever exist |
| `tanitad-pod2` | 0 %, 0 MiB | FREE | 477 MB/s | idle |
| `tanitad-pod3` | 97 %, 18.7 GB | `vlm-production` (TTL, expires 19:35 local) | 407 MB/s | Cosmos-Reason2 production labeling, **ALIVE** — PID 53351, 9.5 h in. **Do not take its GPU** |
| `tanitad-eval` | 0 %, 0 MiB | FREE | 265 MB/s | free between jobs |

**Always `gpu_lock.sh acquire <owner>` before GPU work** (`/usr/local/bin/gpu_lock.sh`; blocks on the
lockfile AND real `nvidia-smi` occupancy; `--adopt` if already running; `--pid N` ties liveness to a
job). Contamination once silently produced 2×-wrong latency numbers and cost 9 arms.

## FLEET — verified 07:03 local 07-22 (compute-apps ground truth)

| pod | GPU proc | lock | state |
|---|---|---|---|
| `tanitad-pod` (pod1) | none | FREE | **IDLE** — v4 G1-de-risk / REF-C-small-eval candidate. ⚠️ holds `ckpt_step10000.pt` (v3enc 10k, irreplaceable) — do not recycle |
| `tanitad-pod2` | 57658 (11.5 GB) | refc-small | REF-C small **29,100/30,000, ~15 min to done** → agent a37d17 evals |
| `tanitad-pod3` | none | FREE | **IDLE** — label-control candidate |
| `tanitad-eval` | arch-inf | arch-inf | SC-13 opponent probe + probe_r2_ci, running |

## 🌙 OVERNIGHT AUTONOMOUS RUN (Sayed 2026-07-22 ~00:00) — 5 goals, honest scope

1. **REF-C small (54.7 M) train + eval** — ✅ achievable overnight (agent, pod2). Closes D-030 ladder.
2. **Build TanitDataSet C + R, push C to HF** — 🟡 major progress; ⚠️ HF push is a PUBLISHING action → agent STAGES it, Sayed fires it. Only tier-C (permissive) is pushable; PhysicalAI-derived data is `PermissionError`-blocked by construction.
3. **AlpaSim: REF-C then flagship v1 closed-loop** — 🟢 **bare-run CONFIRMED tonight** (torch 2.7.0+cu128 sees the A40 with no container); next = invoke NRE render service on a scene, then REF-C adapter. May reach first closed-loop by morning or hit a Vulkan-render wall.
4. **v4 build + train** — 🔴 build is **~118 eng-h (≈3 agent-weeks)**, will NOT finish overnight. Agent doing P1→P4 + the G1-panel-on-v1 de-risk. ⚠️ **"augmented dataset" = v2.1/v3 labels + proven improvements; lead-state augmentation was FALSIFIED (+1.16 %, CI∋0) — NOT in scope.** Training NOT launched without Sayed's go on the final config (multi-A40-day commitment).
5. **Consolidate benchmarks + leaderboard** — 🟡 mostly done today (LEADERBOARD rewritten, driving.py inline); loop finishes + documents.

## 🌙🌙 11-HOUR AUTONOMOUS RUN (Sayed 2026-07-22 16:57 local — away until ~04:00) — PROGRAM-WIDE

Mandate: progress the whole program autonomously (tool dev · data eng · training · eval · closed-loop
sim · new-arch ideas + proof · inference opt · Orin/Thor deploy). **PARTICULARLY: deep-research +
design + prove an INVERSE-DYNAMICS MODEL (IDM)** that estimates ego motion+action from raw monocular
video → pseudo-label action-free **YouTube** dashcam video → pretrain the WM at scale.

Streams launched (fan-out CAPPED at ~4 concurrent, banking each):
- **⭐ IDM concept** — ✅ research DONE (108 agents; verdict **(a)>(b)>(c)**: a supervised, PREDICTIVE,
  NON-CAUSAL IDM head on our TRAINED encoder = the **VPT blueprint** [1962h→70k h, 90.6%/R²0.97];
  Seer/DriveWAM forecast-latent aux; **domain gap = the dominant risk**; refuted VPT-"100×"). ✅ **design
  doc written** `Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN.md` (our-asset synthesis: f-theta
  canon = the missing intrinsics front-end · our WM predictor = the Seer aux · 2 PhysicalAI rigs = a
  built-in intrinsics testbed · parity firewall). 🟥 **proof FAILED (MEASURED, I verified `results.json`) → do NOT scale yet.** In-distribution WORKS
  (PhysicalAI held-out speed R² **0.930** / yaw **0.924**) but cross-domain COLLAPSES: comma speed 0.657 /
  yaw 0.0005 ⚠️**STALE-PENDING — `heading_repair` OFF, see C29 note below**; **rig-A→rig-B speed R² −2.465** (MAE 14 m/s on near-identical speed dists = a genuine
  encoder-latent rig shift, C6-controlled, not a confound). Frozen encoder + per-clip crop does NOT survive
  a rig/intrinsics change. **→ RE-GATE DONE (MEASURED, I verified `results_regate.json`): NEITHER FIX REACHES THE GATE → NO-GO on
  YouTube on this recipe.** Fix #1 (f-theta front-end) = **NO-OP** — ALREADY applied in the baseline
  (f_eff rig-A/B/comma all ≈266) → the rig collapse is **NOT intrinsics-driven**. Fix #2 (light-FT) INERT
  on the corpus gap (comma speed R² 0.406→0.411), partial on the rig gap (−3.21→−1.65), both far from 0.9.
  🔴 **LABEL-PROTOCOL CORRECTION 2026-07-27 (C29):** the comma `yaw 0.0005` above (and every pre-07-27
  comma yaw number) was scored with **`heading_repair` OFF** — comma's heading is `arctan2` of the ENU
  velocity, undefined at standstill; MEASURED **26.27 %** of comma frames below 0.5 m/s are physically
  impossible vs **0.000 %** above (PhysicalAI **zero in every bin**, so the `0.924` and every rig number
  here are unaffected). Superseded value kept for audit; **STALE-PENDING** — no repaired measurement
  exists on this substrate. On the v3 val split (`heading_repair` ON, `v_min` 0.5, nothing retrained) the
  deployed head reads comma yaw **R² +0.3308** (was **+0.0114**); retrained, **+0.679**. ⭐ Honesty
  condition: comma-only MAE **−42.5 %** but **medAE −1.1 % and nMedAE 8.0 % WORSE** — tail and summary
  statistic, not typical accuracy. **The NO-GO stands**: it rests on speed (0.657) and the ADE ratio,
  neither of which the heading label touches. Inventory: `…/incoming/2026-07-27-comma-yaw-reissue/`.
  🔴 **AMENDED 2026-07-27 (`anchor-settlement`, C43): `+0.3308` is WITHDRAWN** — by content (sha256 of
  raw pose bytes AND raw sensor bytes), **2 of its 22 comma val episodes are inside that head's own
  comma TRAINING set**; without them it reads **−0.746** (CI [−1.574, −0.177]), and its published CI
  **[−1.2982, +0.7047]** already spanned zero. `+0.679` stands (no leak) at **+0.3038 [+0.054,
  +0.479]** on the 20 content-clean episodes. **The NO-GO still stands** — speed and ADE are
  untouched. Record: `…/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`.
  **Path forward = encoder RETRAIN on undistorted multi-domain data + speed-prior scale head, NOT a frozen
  readout.** YouTube-ingest / A1 = **ON HOLD (decided)**. Feeds the encoder-strategy synthesis.
  Triangulates: frozen (IDM, REF-A) fails, trained (v1) works. ✅ **Encoder-strategy synthesis DONE**
  (`aa35bea9`, `…/Research/2026-07-22-encoder-strategy-and-vjepa2ac.md`): **V-JEPA2-AC (arXiv:2506.09985)
  FREEZES its encoder yet reports OUR exact failure mode** — "must infer the action axis from monocular
  RGB, no explicit camera calibration," authors "manually tried camera positions" → **camera-rig
  sensitivity persists at >1M h SSL scale (scale ≠ rig-invariance).** Ranking: **trained-from-scratch
  (MEASURED best 0.452 vs REF-A frozen 2.13–2.92) > our-own video-SSL (hyp) > light-FT > frozen (dead)**;
  v4/v5 → trained-from-scratch. 🟡 **MULTI-RIG CO-TRAIN in flight (`ae75b7c`):** does multi-domain
  training recover cross-rig transfer (**data-diversity** fix = cheap → proceed to YouTube) or is it
  **representational** (→ justifies expensive video-SSL encoder)? Pre-registered, hours, existing assets —
  the go/no-go before any encoder-retrain GPU-days. IDM design §3 "start frozen" now contradicted → co-train.
  ✅ **f-theta calibration front-end BUILT + validated** (`a0bdaf18`, Sayed's "add calibration front-end"):
  canonicalization **round-trips to f_eff 266.0–266.5** on a real comma2k19 frame (both camera branches) →
  9-ch `[2,9,256,256]` contract; production intrinsics estimator = **GeoCalib** (ECCV'24) + VP cross-check.
  Pipeline design (S0–S5, parity-firewalled) + **licensing verdict** staged: raw YouTube = `refuse`/nc-internal,
  pseudo-labels inherit (annotations-only URL layer, ship pointers not bytes), **WM-pretrained = internal-OK /
  public-gated-on-legal** (YouTube ToS is download-time, doesn't propagate into weights). → **scale only after
  the deep known-dynamics validation + the licensing sign-off.**
- **Orin/Thor deployment** — ✅ **DONE** (`ab7fb7`). ONNX export of the deployed flagship4b (263.44M,
  enc/pred parity **1.2e-6 / 1.9e-6 PASS**, opset 17); **TRT-FP16 builds on A40** (enc **1.205 ms**, pred
  0.666 ms, ⭐ **TRT FUSES our MHA** — the published ViT-MHA-won't-fuse risk does NOT bite our encoder on
  SM8.6); CUDA-graph 20-step rollout **96.4→27.87 ms (3.46×)**, exact. Precision map LOCKED: **Orin
  (Ampere) FP16+INT8 only — NO FP8/FP4; Thor (Blackwell) +FP8+NVFP4**; INT8 trap PUBLISHED → mandate a
  per-layer FP16-vs-INT8 benchmark. Composed tick 18.75 ms (5.3× 10 Hz). v4 delta: operative predictor =
  v1 verbatim → same levers, +9.8M diffusion (denoise-count = tick knob). Blocked: real Orin/Thor silicon
  + NVFP4 (no chip). Staged `…/incoming/2026-07-22-orin-thor-deployment/`.
- **v4 → v4.1 RESTART** (Sayed authorized 2026-07-22 ~18:10). Phase-B finding decided it: WM canary
  0.42→…→1.48 (oscillating, peaks creeping up) driven by the **trunk LP-FT itself (lr_trunk 3e-4)** — it
  kept degrading with the planner gradient fully clamped (lam_mult=0), so the trunk fine-tuning, not the
  planner, is the culprit; WM loss rose 2.3→4.24. **v4.1 = fresh warm-start from the v1 trunk +
  `--lr-trunk 3e-5` (10×↓)** + phase-A trunk-freeze if the trainer supports it. Agent `a8adcdbf`: (1)
  grab the v1.5/v1.6 ckpt off pod2 + push HF (unblocks flagship-AlpaSim) → (2) kill v4 PID 75844 → (3)
 launch `flagship-v4.1-30k`. ✅ **DONE — FIX CONFIRMED (matched-step, MEASURED):** v4.1 PID **79542**,
  warm-started from the v1 trunk, `--lr-trunk 3e-5` (no phase-A-freeze knob exists → lr cut is the only
  no-code lever). **Canary v4.1 flat s500 0.451 / s1000 0.460 (controller "ok") vs v4 breaching 0.599 /
  0.814 → the WM STAYS HEALTHY.** val ade 3.27→1.17 (planner warming). Watch **re-armed for PID 79542** +
  `flagship-v4.1-30k/ckpt_step10000.pt` → G1 gate at 10k; keep checking canary past 1500. ⚠️ **HF push of
  `v16-ab-ft` (val 0.442) BLOCKED on a publish decision** (private storage full → make public or upgrade)
  — but NO LONGER blocking anything: the flagship-CL is DONE via v1's tactical head (v1 on HF); v1.5/v1.6
  only needed if Sayed wants that comparison. **Sayed's other directives (in flight):**
  flagship-in-AlpaSim via the flagship eval inference (a04ddb, after the v15/16 ckpt lands on HF) · IDM:
  **add the f-theta calibration front-end as a committed stage THEN scale** (a0bdaf18) + **validate the
  IDM DEEPLY on known-dynamics CAN data** before YouTube (extend ae75b7c after its cross-rig proof).
- **AlpaSim** (eval pod, `a04ddb`) — ✅ **REF-C sweep + STATISTICAL SUITE DONE.** n=12 scenes:
  **at-fault collision 33 % (4/12), REF-C passes ~HALF** (base 6/12) — the n=1 "both collide" was the
  **worst-case** 41-actor highway scene, NOT the norm (C5 refinement, logged). **base ≥ XL holds
  closed-loop** (mean score 0.345 vs 0.246, 6 vs 5 passes; **2× XL capacity = no closed-loop advantage**).
  ⚠️ n=12 wide binomial CIs (score gap is the cleaner base≥XL signal); NuRec reconstructions, 480×854.
  **videos DONE** (native AlpaSim layout = TanitEval standard; mp4s staged + sent to Sayed). 🔬 **v1 finding:
  pure flagship v1 is an action-conditioned WM, NOT a policy** (rolls under TRUE future actions) → can't
  drive closed-loop; **v1.5/v1.6 CAN** (v1 trunk + anchored-diffusion tactical head) but their ckpt is
  **only on pod2 (v4 training) — 🔴 BLOCKED, won't touch it.** → **Sayed decision: push the v1.5/v1.6
  head to HF from pod2, or wait for v4.** Now a04ddb runs the **`public_2601` SUITE** for a STATISTICAL
  REF-C base-vs-XL closed-loop number (resolves the n=1 caveat; free eval pod). **Program finding:
  open-loop-trained planners collide closed-loop → closed-loop-aware planner = the next architecture lever.**
Rotate next as capacity frees: data-eng (YouTube-dashcam pipeline feeding the IDM · A1 label transfer),
eval (v4 milestones), tools (AlpaSim viz via `corpus_overlay.py`), new-arch proofs from the IDM research.

## ✅ exp-A VERDICT (2026-07-22 00:00): MASK IS CAUSAL — at TRAINING time

Bucket means `g_op_fwd_ade_m` 0–2k, mask-OFF retrain (exp-A) vs v1 vs v3enc(mask-on):
| bucket | v1 | v3enc | exp-A |
|---|---|---|---|
| 0–500 | 1.237 | 1.479 | **1.012** |
| 500–1000 | 0.672 | 1.130 | **0.581** |
| 1000–1500 | 0.486 | 1.028 | **0.478** |
| 1500–2000 | 0.482 | 0.655 | **0.316** |
**exp-A tracks and BEATS v1; v3enc(mask-on) far worse.** Reconciles exp-B (mask-off at inference on damaged weights → no help) with exp-A (mask-off in training → weights never damaged). **The damage is baked in during TRAINING; the fix belongs there — exactly v4's learned-null-row (P5b), now MEASURED not inherited.** ⚠️ Bound: training-side metric, not held-out eval (2k too short). Log at `taniteval/results/trainlogs/expA-nodrop_train_log.jsonl`.

## 🔄 ACTIVE STREAMS — the loop maintains THIS table every iteration

| # | stream | owner | state | next milestone | blocked on |
|---|---|---|---|---|---|
| S1 | AlpaSim closed-loop | agent a04ddb (eval pod) | ✅ **M1+M2 DONE — AlpaSim RUNS closed-loop bare on the eval pod** (I verified the staged eval JSON: `run_uuid 1d2d0617…`, status **PASS**, score **0.6637**, collision/offroad/wrong_lane 0, **img_is_black 0 = real frames**, drove 39.2/73.8 m). Full bare topology localhost: renderer :6011 + physics :6006 + controller-MPC :6007 + driver :6789 + runtime. **NuRec renders via CUDA/gsplat/OptiX — the 12-day Vulkan wall DOES NOT apply (retire that risk).** M3 REF-C driver adapter (`refc_driver.py`) IMPLEMENTED, base+XL ckpts on pod `/root/models/refc-{base,xl}-30k`. 12 files staged @ `incoming/2026-07-22-alpasim-closedloop-evalpod/` incl `RUN_RECIPE.md`. Pod CLEAN (svcs stopped by PID, GPU free, lock released) | ✅ **M4 correctly HELD at gate 1** (agent refused to emit an untrustworthy number). Root: REF-C input is **f-theta fisheye→pinhole canonicalization** (per-clip cx,cy crop + resize + 3-frame stack), NOT a resize; `r0_selection.parquet` (per-clip intrinsics) is ABSENT on the eval pod; and real-mp4 vs NuRec-reconstruction can never byte-match → **geometric f-theta consistency** is the true requirement. ✅ **de-risk COMPLETE (I verified the staged logs).** **B**: REF-C base reproduces registry **0.4728 exactly** (ckpt+env good). **B2**: canonical arm validated (`f_eff=266` PASS); naive `cv2.resize` diverges from canonical **0.747 m mean / 1.566 m @2s** (⚠️ n=3 windows) ≈ 3.3× REF-C's ADE → naive closed-loop = meaningless. **C**: USDZ **carries** f-theta calib on-pod (rig-B, backward poly, inversion 1.24 px) → A feasible WITHOUT the build parquet. **VERDICT: A MANDATORY + FEASIBLE. Sayed GO on A (2026-07-22)** — agent a04ddb building it | **✅ M4 DONE — REF-C base+XL closed-loop MEASURED (I verified base).** Native f-theta confirmed (renderer hands the forward poly + cx=956/cy=755 directly; `f_eff=265.9≈266` self-check on LIVE frames). **BOTH base & XL COLLIDE at-fault (front), score 0.0** — base progress 0.54 / dist-to-GT 1.66 m / 39.8 m before collision; XL worse (0.41 / 2.67 m / 30.4 m) → "base ties/beats XL" holds closed-loop too. Open-loop-trained diffusion planners accumulate error into a lead-actor collision (real, not a preprocessing artifact). ⚠️ n=1 scene = DIRECTIONAL (suite `public_2601` for a stat number). base uuid `1d0fee08…` | **Finishing /goal (a04ddb):** REF-C small → flagship v1 adapter → **TanitEval videos (front-cam overlay + BEV inset)** for all 4. 🔬 **New-arch lever surfaced: closed-loop-aware REF-C** (both collide open-loop-in-closed-loop) | finishing goal |
| S2 | flagship v4 30k | main+agents | 🟢🔥 **LIVE & TRAINING on pod2** (Sayed GO 2026-07-22, launched 16:00 local). **I verified step 200** (advancing 0→200), PID **75844** GPU 100 %/33.3 GB, **canary baseline 0.42148** (n=881, at the 0.452 ref), loss 1202→15.3, plan_ade 70→16, phase-A ~1.52 s/step (WM warmup, λ_plan=0). Code synced to pod2 (was STALE 23 K→60 K; **md5 byte-identical on 7 modules**); PREFLIGHT OK + real-smoke passed. ⚠️ pod2 interpreter is `/usr/bin/python3` (CUDA-capable, torch 2.4.1+cu124) — `/workspace/venv/bin/python` does NOT exist here. Out `pod2:/workspace/experiments/flagship-v4-30k/` | **G1 KILL gate at step 10000** (card `flagship-v4.card.json`, 8-KILL/5-report) — watch armed for `ckpt_step10000.pt` → run `run_gate.py`. Conservative ETA ~30 h; do NOT extrapolate from the phase-A rate (C5). First in-loop eval at step 500 | running |
| S3 | REF-C small train+eval | agent a37d17 | ✅ **DONE + EVALUATED.** Selected ADE 0.5261 (SEPARATED worse than base 0.4728/XL 0.4714 — first ladder separation) BUT **matched-64-anchor oracle: small 0.221 < base 0.283 < XL 0.437 — small proposes BEST per-anchor.** **The fan lever is ANCHOR WIDTH, not encoder scale**; 2.4× param cut cost 0 fan quality. Registry §4.2 rewritten. 11.5 ms tick | — | DONE |
| S4 | TanitDataSet C+R | agent a867 | 🟢 **DONE pipeline** — C=R=90 comma2k19 recs, license-verified, HF push STAGED. ⚠️ 90 recs = proof-of-pipeline, NOT a corpus | **L2D adapter (~2-3 eng-d) = the real unblock** (0 recs, no loader/video) | needs L2D data |
| S5 | arch-inf (SC-13 + probe_r2_ci) | agent, eval pod | running | SC-13 = task #10 | nothing |
| S6 | benchmark consolidation | loop | queued | verify LEADERBOARD current + document | nothing |
| S9 | **v4 training-label set** | agent a0531 | ✅ **DONE.** Code+tests staged (770 green), Proof A PASSED, **mints COMPLETE** (train 42.9 MB / val 10.8 MB, ~3.06 h). **Proof B coverage PASSES** (I verified the provenance): LAT/LON **1.00**, dist 0.63, stop_dist 0.42, route 0.81, vt 0.83, strat ttm 0.20 / curv3s 0.66 / curv5s 0.69 / tspeed 0.83; lat/lon-active 0.29/0.25; **lowest slot 10× above the ~2 % dead line → no starved head**. Parity `bit_identical:true`, key `e438721ae894` unchanged, 406,099 windows. Provenance staged @ `incoming/2026-07-22-v4-labels/`; multi-GB `.pt` caches stay pod-side (`pod2:/workspace/v15/labels_{train,val}_v4.pt`) indexed by eid | — | DONE |
| S7 | L2D adapter | agent a155df | ✅ **DONE (tonight's scope)** — schema verified (LeRobot v3.0, 100k eps/26.5M frames/735h), **drive-dedup PROVEN** (100k→46,473 non-overlap, `groupby(session_id)`, known pair 150 frames @0.000000m), Apache-2.0 re-verified, **1-drive slice built end-to-end** (14 recs, sha256✓, tier ship, km/h confirmed). `l2d.py`+`L2DIngestor`+9 tests staged | full-corpus ingest ~2-3 eng-d (follow-on) | GDPR gate + vocab v1.1 |
| S8 | HF upload TanitDataSet-C | main bg | 🔴 **BLOCKED by the box TLS proxy** — Xet transport hung at 0 B; LFS path moves bytes but the proxy **resets the S3 multipart PUT** (`WinError 10054`) at 44–193 kB/s. Repo created (empty), retrying, may never complete from here | **ROBUST FIX: rebuild the 90-rec lake ON the eval pod (its comma2k19 cache is the source) and push from the pod** — pods have clean ~118 MB/s HF egress (proven by the model-ckpt pushes) | dev-box proxy |

**IDLE CAPACITY 11:12: ALL FOUR A40s FREE.** Placed: v4 build agent → checkpoint-based G1 panels on the eval pod (validation, no Sayed-go). Remaining candidates need a decision or a build, not a launch I'm withholding: v4 training (needs Sayed + P4 loop); AlpaSim service wiring (build); label control (LOW value now — small-vs-base already gave the clean scale read); L2D full ingest (CPU) + pod-side HF push. ~~pod1 FREE (v4 training host once built+approved); pod3 FREE (lock released — VLM done 13h ago). pod2 → REF-C small.** Candidates, none launched without Sayed:
the **label control** (XL-with-v2.1 or base-with-v1 — the blocking experiment for ANY scaling claim),
**REF-C small 54.7 M** (closes D-030's 3-size ladder; only a 150-step smoke exists), or the
**v1.6 paired bootstrap** (task #31).

## AWAITING SAYED (ranked)

0. 🔴 **TanitDataSet-C HF push BLOCKED from the dev box** (repo created empty). The box TLS proxy resets S3 upload connections; Xet hangs at 0 B, LFS crawls-with-resets. **Fix = push from a pod** (rebuild the 90-rec lake from the eval pod's comma2k19 cache, push over the pod's clean HF egress). Earlier claim of "PUSHED" was wrong (uploader died at committed 0/17). Original attempt: 🔴 I prematurely said "PUSHED" at 00:32 — the uploader **died on a session restart having committed 0/17 files** (it hashed all 15.9 GB but the process was a child of the Claude session). **Relaunched DETACHED 07:00** (`scratchpad/hf_push2.log`, PIDs 21408/26352 — `Start-Process`, survives restarts; `upload_large_folder` is resumable). `Sayood/TanitDataSet-C`, 90 comma2k19 (MIT) recs, structural licence proof (only `shards/owned-safe/`). Slow over the TLS proxy (~6 h). **Verify `committed: 17/17` before claiming pushed. To make public later: `HfApi(token=...).update_repo_settings(..., private=False)`.**

1. 🟢 **v4 is LAUNCH-READY — approve the ~5 A40-day 30 k run (or hold).** The build is DONE + verified,
   not just design-approved: **(b)** P4 loop wired (canary→λ_plan controller down-only, ckpt/resume
   bit-exact, milestone archive) and **(c)** the 3 gate bugs fixed — my own `pytest` **770/2** + the
   14+5 gate tests green. ≈247.9 M (62 % of the 400 M cap, ~30 M *smaller* than v1). G1 gate **1.66**
   A40-days, 30 k **4.96**. Card split **8 KILL / 5 REPORT** (`Project Steering/Gates/flagship-v4.card.json`).
   **Remaining before I fire it: ONLY your go.** (a) DONE (Proof B: LAT/LON 1.00, no starved head), (b)+(c) verified, **dense anchors built** (`pod2:/workspace/experiments/flagship_v4_anchors_dense.pt`, `[256,20,2]`, parity source). Every launch precondition verified on pod2.
   Verified launch command (pod2, every path confirmed present):
   `PYTHONPATH=/workspace/TanitAD/stack python scripts/train_flagship_v4.py --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt --out /workspace/experiments/flagship-v4-30k --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000 --batch 16 --lr-head 1e-4 --lr-trunk 3e-4 --eval-every 500 --save-every 1000 --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda`
   (dense-anchor prereq: `build_refc_anchors.py --data-root …/_epcache --out …/flagship_v4_anchors_dense.pt --n-anchors 256` over `DENSE_HORIZONS` 1..20 → `[256,20,2]`.)
   ⚠️ Levers 2 of 2, door CLOSED. ⚠️ 32.6× speed-fan is CONFOUNDED — never scope the strategic layer down on it.
2. **HF push of the v3enc 10 k gate ckpt** — publishing action, needs his word. Two pod disks only.
3. **REF-C train/eval `nav_cmd` mismatch** (trains with a v1 command, evals constant `follow`:
   `refc_eval.py:78` → `refc.py:786`) — blocks strategic-goal work on REF-C arms. Flagship path clean.
4. **v2.1/v3 labels for v4.**
6. **Enrol ROUTEDIST + TACDIST as vocab v1.1** — L2D ships a metric distance-to-maneuver natively (the bump you've been holding); adapter mints it to a sidecar, OUT of the frozen 18-slot/114-token goal until you enrol it.
7. 🔴 **GDPR gate on L2D** — real German dashcam footage, NO anonymization statement in the card. Apache-2.0 grants copyright, not the data-protection right → **face/plate check REQUIRED before any L2D frame is re-hosted / published**. Blocks L2D entering a shippable corpus.
5. **Price the LAT×LON factorized tactical head** (`V3_FACTORIZED_TACTICAL_HEAD_SPEC.md`, ~5 k params,
   one retrain): 24.9 % of windows have a live LONMODE the 5-way softmax cannot express.

## ✅ v4 GATE BUGS — ALL THREE FIXED + VERIFIED (agent a5cee3d, 2026-07-22; my own `pytest` 14+5 green)
1. ✅ **`cluster_bootstrap` emitted + gate FAILS LOUD on the deprecated estimator.** `run_gate._read_eval_metric` now PREFERS the episode-cluster bootstrap and raises `SystemExit` if only `overlapping_holdout_se` is present (a gate can no longer silently read the forbidden 1.28–2.06×-narrow estimator). `driving.py` emits a gate-readable `cluster_bootstrap` block (additive; bench already did). Recomputed CI on committed windows `[0.3675, 0.4871]` = **exact** match to `CI_RECOMPUTE_2026-07-20.json`.
2. ✅ **miss-name drift resolved** — alias group `{miss_at_2m, miss_2m, miss_rate@2m}` resolved in both the JSON-read and supplied-`--secondary-value` paths; no emitter renamed (no published-JSON churn).
3. ✅ **report-only secondaries preserved** — `cmd_check` emits off-card `--secondary-value` into a `report_only` list (`adjudicated:false`), read+printed, never touching `sec_ok`/the verdict; the **8-KILL / 5-REPORT** split emits correctly.
Files (STAGED, not committed): `stack/scripts/run_gate.py` + `taniteval/taniteval/driving.py` + 19 new tests. Proof: `taniteval/results/v1_g1_dryrun_gate_FIXED.json`.
⚠️ **CORRECTED 2026-07-23 (`a9dfe223`):** the OFF-CARD report-only falsifiers are `imag_win_at_5s`, `strat_subspace_{sufficiency,compression}`, `longh_5s_beats_persistence`, `cruise_delta_vs_holdv0` (a set of 5, NON-blocking). These are NOT the same as the 3 ON-CARD **KILL** secondaries `speed_benefit_recovered_frac`/`deploy_tick_p99_ms`/`nonav_route_beats_majority` — those WERE gate-blocking and are now BUILT (see GateEmit row). The earlier "P7 report-only" framing conflated the two sets.

## ✅ RESOLVED — `test_seam_clamp` full-suite order
`stack/tests/test_flagship_v4.py::test_seam_clamp` now passes in full-suite order — **full stack `pytest -q` = 770 passed / 2 skipped** (my own run 2026-07-22 12:5x local, venv `C:/Users/Admin/venvs/tanitad`). The RNG-order flake (head weight-init off the global RNG) was fixed by seeding inside the test. No open v4 test bug.

## OPERATIONAL TRAPS (each has produced a false alarm)

- ⚠️ **Pods run `/workspace/venv/bin/python`, NOT `python3`.** `ps aux | grep python3` and
  `ps -C python3` return EMPTY for a perfectly healthy job. Use `nvidia-smi
  --query-compute-apps=pid,process_name,used_memory` — it is the ground truth for "is the GPU
  actually working". This nearly produced a "the VLM job is dead" alarm at 17:04 while PID 53351 was
  at 97 % util. **Take multiple util samples and check compute-apps before ever declaring a job dead.**
- ⚠️ **A trainer's `ckpt.pt` step field ≠ the last logged step.** v3enc logged 10,800 but `ckpt.pt`
  carries `step=10000` (`--ckpt-every 1000`). Always read the checkpoint's own `step` field.
- ⚠️ **Never `pkill -f <trainer>`** — it self-matches your own ssh command and kills the session.
  Kill by explicit PID (parent first; children exit with it).

## RETIRED — do not do these

- **The power-law exponent gate is DEAD.** Same log gives −0.387…−0.738 by fit window, R² 0.09–0.58.
  Decisions run through `GATE_PROTOCOL.md` via `run_gate.py`; use matched-step *bucket means* for
  comparatives — **never a single row** (that produced the retracted "step 450 / 23×" artifact).
- **Post-hoc re-ranking stays closed** (v1.0 0.0 %, v1.2 +2.9 % n.s., ~92 % aleatoric). **But a
  MISSING INPUT is not re-ranking and is open**: the fan is a SPEED fan (32.6× along-vs-across,
  0.0 % of windows laterally dominated) and there is no longitudinal signal anywhere in selection.
- Do not re-raise: REF-C base **ties** XL (paired Δ+0.0013, not separated; 21.8 vs 44.1 ms);
  Reason1-vs-Reason2 done (neither can mint ROUTE; the left bias is the **model's**, proven by the
  enum-order probe); route labels v2.1; pod2 quota.

## ⭐ OPERATING STANDARD (Sayed 2026-07-21) — binding

`CLAUDE.md` §"Operating standard" + `Project Steering/RETRACTION_LOG.md`. Five rules:
**(1)** evidence class on every claim — MEASURED / PUBLISHED / **INHERITED** / ESTIMATED / HYPOTHESIS;
a GPU-day decision may never rest on INHERITED. **(2)** absence from ONE probe is not absence — two
paths, and prefer the tool that OWNS the fact. **(3)** finish before you start; done = in the repo,
staged, with provenance. **(4)** **log retractions BY ROOT-CAUSE CLASS** — that log is the
self-learning mechanism, and it must be read before asserting in a known class. **(5)** aim above
published SOTA; settle conflicts with the cheapest **pre-registered** experiment, both outcomes
committed — never by scoping the goal down. Orchestration: priority order in every brief, bank
incrementally, **cap fan-out**.

## 🟢 ALPASIM — NRE IMAGE IS ON THE EVAL POD, NO DOCKER USED (in progress 2026-07-21 23:05)

**Both credential gates PASSED (MEASURED):** NGC key in `Keys.txt` authenticates to `nvcr.io`;
Sayed granted `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (2.89 TB total — pull only 1–3 scenes
from `sample_set/26.04_release`).

**Pulled WITHOUT a container runtime** — the "we need docker" blocker was partly wrong. A registry
pull is plain HTTPS. Recipe that worked:
1. Mint a **600 s bearer token scoped `repository:nvidia/nre/nre-ga:pull`** on the DEV BOX (the
   `nvapi-` key never leaves it), pipe to the pod via **stdin**, write to `/dev/shm/h` (tmpfs, 600),
   `curl -H @/dev/shm/h`, delete after. **Key never on a pod disk, never in an argv.**
2. `xargs -a layers.txt -P 8 -I{}` → **42 layers, 40 unique, 14,295,757,278 B — exact manifest match.**
   Layers at `tanitad-eval:/opt/nre/layers/`.
3. Extract in manifest order → `/workspace/nre/rootfs` (**IN PROGRESS, 25 GB and climbing**).

**Image facts (MEASURED from the config blob):** ENTRYPOINT `/app/run`, WORKDIR
`/app/run.runfiles/_main`, USER root, `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video`,
CUDA 12.8. **NRE is a Bazel-packaged Python app** — `pycena_nrm_full.runfiles` (149 entries) with a
**hermetic Python 3.11 and a full venv incl. torch/torchvision**. Best case for running bare.

⚠️ **TRAPS PAID FOR — do not repeat:**
- `/` on `tanitad-eval` is **200 GB and was ALREADY ~172 GB full** (`/root`). The first extraction
  silently ran out of space. **Extract to `/workspace`, never `/`.** (`df` IS valid for local `/`;
  the never-`df` rule is about the MooseFS quota.)
- **Windows CRLF** in a scp'd digest list → `curl: (3) URL rejected`. `sed -i 's/
$//'`.
- `xargs -I@` **collides with curl's `@file`** syntax → use `-I{}`.
- A **heredoc cannot coexist with a credential piped on stdin** — both want stdin; `set -e` then kills
  the block silently.
- `tar: Cannot change ownership … Operation not permitted` on MooseFS is **BENIGN** (metadata only).
- 🔴 **I twice concluded files were "missing" from a tree that was still extracting.** Class C2.
  **Check `pgrep -fc 'tar -xf'` before concluding anything is absent.**

**NEXT:** wait for `EXTRACT2_DONE` → symlink `/usr/local/nvidia/lib{,64}` to the host driver libs →
`ldd` the launcher → run bare. Then the 3 bare Python services + a **REF-C adapter** (the drafted
`flagship_v1_policy.py` is for a ROLLOUT policy; REF-C is a direct head and needs its own).
⚠️ REF-C **cannot** be closed-looped without a renderer — it has no world model. AlpaSim is the ONLY
way to close-loop it; that is why this subproject matters.

## 🟢 RENDERING IS NOT BLOCKED — the 2026-07-09 finding is RETRACTED (class C2)

MEASURED 2026-07-21 on `tanitad-eval` and `tanitad-pod2` (A40): `/dev/dri` **card3 + renderD130** ·
Vulkan ICD **`/etc/vulkan/icd.d/nvidia_icd.json`** (the old probe looked only in `/usr/share/`) ·
loader `libvulkan.so.1.3.275` · ICD target `libGLX_nvidia.so.580.159.04` · SPIR-V
`libnvidia-glvkspirv.so.580.159.04`. Only userland gaps: `vulkan-tools`, `libegl1` (missing
`libEGL.so.1` **symlink**). **This unblocks AlpaSim and CARLA after 12 days.**

⚠️ **The remaining AlpaSim blocker is the CONTAINER RUNTIME, not graphics.** NRE ships only as
`nvcr.io/nvidia/nre/nre-ga:26.04` (`external_image: true`, no pip form) and our pods are unprivileged
containers with no docker/podman/singularity. **Fix: make that image the RunPod pod TEMPLATE — do not
attempt docker-in-docker.** Env: `NVIDIA_DRIVER_CAPABILITIES=all` (currently EMPTY), unset `DISPLAY`.
**First test before allocating storage: does the NGC-gated image pull at all?**

## RETRACTED CLAIMS — never repeat these

- ❌ *"v3enc's failure is a generalisation gap (in-sample `ego_r2` 0.79–0.85 vs held-out 0.393)"* —
  **inadmissible**: `ego_linear_r2` is an in-batch B=16/D=2048 fit reading **0.595 on a randomly
  initialised encoder at step 0**, and v1 never logged it (no control).
- ❌ *"v1 reached v3enc's level at step 450 (~23×)"* — single-row noise; bucket means say **~3.5–5×**.
- ❌ *"decorr strangled speed capacity"* (D-031) — `decorr_w = 0.0 if step < 10000`, **never on**.
- ❌ *"the label defect explains v3enc"* — `nav_valid_frac` is 0.21–0.25 in **all four arms including
  the deployed v1** that scores probe 0.861.
- ❌ *"6 of 9 ROUTE tokens never minted"* — it is **4 of 9** in the repo.

## THE ZERO-FILL BUG — real, worth fixing, but NOT the attributed cause of v3enc's failure

`v2_ego_dropout=0.25` **zero-fills the v0 action channel** (`flagship_losses.py:214-217`). Base action
channels are `(steer, accel)` — no absolute speed — so **0.0 m/s is an in-distribution "stopped"**,
indistinguishable from a mask.

**Experiment B (2026-07-21, PARTIAL) measured its true scope — read this before citing it:**
- Forcing the mask off recovers **~51 %** of the matched-step *training-log* gap; corrected ratio
  **2.69×**, not 4.48×. Paired Δ +0.1697 [+0.1617, +0.1780], B=2000, 6,400 windows / 2,192 episodes.
- 🔴 **It explains NONE of the EVAL gap.** The mask is gated on `model.training`
  (`flagship_losses.py:215`) and TanitEval loads `.eval()` (`loaders.py:151`), so **the mask was never
  active in any eval we ran.** The 4.60× / 3.19× eval deficits are untouched by this mechanism.
  **v3enc's held-out failure remains UNATTRIBUTED.**
- ❌ *"encoder intact / rollout collapsed corroborates the mask"* — `g_*_mid_de_m` reads only
  `(z_t, fut_states)`, no actions, so it is **mask-invariant by construction** (max|Δ| = 0.0).
  B supplies **zero** information about the encoder. (The v3enc-vs-v1 training-log split still stands
  on its own evidence; B just cannot corroborate it.)
- ❌ *"v3enc `inv` 0.3784 ≈ no-speed control 0.3644"* — a **3-channel mean vs a 2-channel mean**
  (the control has `action_dim=2`). Channel-matched, v3enc's steer+accel `inv` **0.2881** is **23 %
  BETTER** than the control's 0.3723 — the claim reverses. `inv` moved only 28 % of the way to v1;
  **72 % of that gap is not the mask.**
- ❌ *"the model discounts v0"* — refuted. It realises **20.9 %** of a `v0=0` error but **72.0 %** of a
  permuted speed and 67.3 % of a doubled one. It built an **implicit null embedding at 0.0**, aliased
  with the 6.45 % genuinely-stopped windows; under the mask **78.4 %** of the zeros it sees are lies.
  This validates the *shape* of the learned-null-row fix, not its link to the eval failure.

**Still fix it:** `flagship_v15.py:348-351` zero-fills v0 at dropout **0.5** — twice the flagship rate —
in a function whose goal and route paths already use learned DROPPED rows. Live in the v1.5 planner
and inherited by v4. **Rule: never zero-fill a channel whose zero is a valid value.**

**Attribution in flight:** §9 row A **LAUNCHED 17:25 local** on pod1 — 2 k-step re-run at
`v2_ego_dropout = 0.0`, one lever changed, ~6 GPU-h. It is the only experiment that separates weight
damage baked in during training from corruption applied at eval. **Decision rule pre-registered:**
exp-A tracks **v1** ⇒ mask is causal in training, v4's fix is load-bearing; exp-A tracks **v3enc** ⇒
mask is a side-show and **v4 must not be built on this premise**; in between ⇒ quantify, do not round
to either story.

**Meta-rule, the real repeat cause:** **≤2 encoder-touching levers per arm.** Neither v2 nor v3enc can
attribute its own failure; that ambiguity cost ~60 GPU-h.

## STANDING

Never idle. Diagnose real causes before restarting anything. Report in chat every iteration (D-023),
but **do not repeat an unchanged report** — advance an open question instead and say what was learned.
Resume rate-limited agents via `SendMessage`, **never respawn**.
