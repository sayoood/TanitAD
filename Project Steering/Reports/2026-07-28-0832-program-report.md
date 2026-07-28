# TanitAD — program report, 2026-07-28 08:32 Berlin (06:32 UTC)

**Previous report on disk: `2026-07-28-0757`.** Interval covered: the **20 commits** from `f4847da`
(the commit that last touched that file) to `4cdf205`.

⚠️ **TWO NAMING/CLOCK FACTS, FLAGGED RATHER THAN SILENTLY RESOLVED.**
1. **The `0757` slot was already occupied.** `2026-07-28-0757-program-report.md` exists and claims
   05:57 UTC — but `git log` shows it was last committed **2026-07-27 10:56 +0200**, ~20 h *before*
   its own claimed timestamp, and it sits **20 commits behind HEAD**. It therefore cannot cover this
   morning's work. This is the known narrative-clock drift. This report is named for the **real**
   clock (`date -u` = 2026-07-28T06:32:33Z) and does not overwrite it.
2. **"Sequential-numbered, not ISO"** — all **13** program reports on disk are date-prefixed
   (`YYYY-MM-DD-HHMM-`). I follow the on-disk convention rather than break it on one file; the
   sequential scheme belongs to the *weekly* report, a different artifact.

⚠️ **"Hold every arm to v1's 0.4271" is SUPERSEDED BY MEASUREMENT and I am not applying it.**
MEASURED in code: `taniteval/rollout.py:170` sets `actions_source="expert_future"`, `:174` names the
metric `wm_fidelity_ade_2s`. **0.4271 is what v1's world model scores when HANDED THE TRUE FUTURE
ACTIONS** — a fidelity number, not a planning bar. Holding a *selector* to it compares a model that
must choose against one that was given the answer. The legitimate same-surface bar is **0.4907**
(881 windows / 40 episodes, in-sample, `MODEL_REGISTRY §1.2`).

---

## 1. Fleet — half of it is down, and that is the headline

| pod | state | evidence |
|---|---|---|
| **pod1** | 🔴 **UNREACHABLE.** `flagship-v2corpus-30k` frozen at **step 23,850 / 30,000 (79.5 %)** | MEASURED: TCP 30107 refused; ICMP *flapping* (present, absent, present across 6 checks) |
| **pod2** | 🔴 **UNREACHABLE**, worse failure — IP answers on **nothing** | MEASURED: TCP + ICMP both dead on 22022 and 22 |
| **pod3** | 🟢 running E1e-A, **step 3900/4000**, **1.2 s/step** | MEASURED over three independent windows (150 s, 240 s, 180 s) |
| **eval** | 🟢 alive, idle | MEASURED |

**Diagnosis, completed this interval (MEASURED):** port 22 on pod1's IP answers with a generic
`SSH-2.0-OpenSSH_8.2p1 Ubuntu` banner — that is the **RunPod host node, not our container** (ours was
the forwarded port). No auth was attempted against it. `ssh.runpod.io:22` is reachable but requires
the pod ID *and* a RunPod account key; **`Keys.txt` holds an HF token and no RunPod credential**
(checked by label; no value read or printed). ⇒ **There is no recovery path from the dev box.**

**Checkpoints are safe, not lost:** pod1 holds `ckpt.pt` + `ckpt_step{20000,15000,5000}.pt` on the
volume and its supervisor auto-resumes, so expected loss is steps-since-last-save. ⚠️ **Verify the
resumed step and `restarts:` on reconnect — do not assume it resumed.**

### What each pod produced this interval
- **pod1** — nothing; frozen mid-run.
- **pod2** — nothing; it took arm B of the geometry validation down with it (~100 steps in).
  ⛔ **It also holds the ONLY copy of the w120/256×640 wide cache AND arm A's completed validation
  results** (`pw_A_old.npz`, `pw_A_old_blind.npz`). Verified absent on pod3 and eval.
- **pod3** — E1c frontier read, E1d run end-to-end, the E1 sensor-level leak check, and E1e-A.
- **eval** — nothing.

---

## 2. Closed loop (D-A) — three one-dimensional levers, two exhausted, one mid-flight

**Estimator for every number below: paired episode-cluster bootstrap, `taniteval/ci.py`, B=2000,
resampling the 44 held-out episodes (43 clusters at K=185). `overlapping_holdout_se` used nowhere.**

### E1c — BOUND, but the primary fired at 15/17 (MEASURED, `e1c_frontier_result.json`)
- Corridor departure **0.5877 → 0.147 at step 2750 (Δ −0.4407)**; junction **0.8414 → ~0.40
  (Δ −0.4414)**. **P1 ∧ P2 CI-separated LOWER at 15 of 17 checkpoints**, multiplicity-robust at all 15.
- **Guardrail Ga (open-loop ADE@2s not separated-higher) held at 0/17.** So did Gb1, Gb2.
- ⭐ **"Train longer" is ruled out by the data:** the open-loop cost falls 0.5048 → 0.2197 over steps
  500–2250 then **PLATEAUS** — 0.2083, 0.2158, 0.2133, 0.1893, 0.2026, 0.1969, 0.1947. Eight
  consecutive points inside ±0.02, no trend.

### E1d — BOUND, and a *stronger* negative than a null (MEASURED, `e1d_alpha_result.json`)
WiSE-FT weight-space interpolation, α ∈ {0.10…1.00}. **Control passed:** α=1.00 reproduces E1c
frontier row 4000 exactly (−0.4274 / −0.4270 / +0.1947 [+0.1415,+0.2522]), so every row is quotable.
- **`dep_overall` is SEPARATED-WORSE at five consecutive interior α** — 0.20 (+0.1107), 0.30
  (+0.1387), 0.40 (+0.1492), 0.50 (+0.1199), 0.60 (+0.0759); every CI excludes zero.
  ⇒ **The base→FT path crosses a barrier; the endpoints are NOT linearly mode-connected** for this
  metric — which is exactly WiSE-FT's precondition. **C52.**
- ⭐ **And it decomposed the primary:** `dep_junction` is better at **every** α and never
  separated-worse, separated already at α=0.20 for **+0.0308 open-loop cost (6.5 % of base, vs +41 %
  at the endpoint)**; `dep_overall` only turns good past α≈0.72.
  ⇒ **Junction recovery is cheap and monotone; overall-corridor recovery is expensive and
  barrier-crossing.** This is the finding that redirects the program.

### E1e — RUNNING (pod3), pre-registered before launch (`a7a2781`)
Lever: `lam_replay` 1.0 → 3.0 — the one flag `e1c_clsft.py`'s own header calls *"deliberately NOT a
lever here"*. Everything else byte-identical to `run_e1c.sh`.
**INTERIM MEASURED** (in-training gate, same 44 held-out episodes, same estimator): open-loop cost cut
**46.9 %–67.2 % at every one of 15 matched steps**, e.g. step 3250: **+0.1902 → +0.0891**.
⭐ **Structural result:** E1e-A plateaus at **~+0.10 from step 2000**, exactly as E1c plateaued at
**~+0.20 from step 2250** — same curve, half the height ⇒ **`lam_replay` sets the ASYMPTOTIC open-loop
cost; it does not remove the plateau.**
🔴 **NOT YET A RESULT:** Ga still fails at every step (best CI [+0.050, +0.132], nowhere near 0), and
**the closed-loop side is UNMEASURED until the frontier eval.** A halved open-loop cost is worth
nothing if P1/P2 died buying it. **E1e-B (lam_replay 8.0) is STAGED, NOT LAUNCHED** (`4cdf205`) — the
pre-registered skip rule requires seeing A's frontier first.

### ✅ The whole chain now stands on a content-verified split (MEASURED, `E1_heldout44_x_train.json`)
`heldout-79d4e3d2d4c6` × the 2,376-episode parity train corpus: **0 overlap on `poses`, `poses_xy/yaw/v`,
`actions` AND `frames_sha256`**, and **0 of 44 val episodes share even ONE frame** with any train
episode. **SPIKE control:** 3 real val episodes injected into an in-memory copy of train were recovered
exactly, by poses *and* frames ⇒ **the matcher provably detects a leak, so the zero is evidence.**
⭐ **And the C49-class gap is CLOSED:** `refc-diffusion-base-v21-30k/config.json` records its training
data as `/workspace/pai_epcache/physicalai-train-e438721ae894`, n=2376 — **character-for-character the
corpus measured** ⇒ clean against **the base arm's own training data**, not merely "some parity split".
⚠️ Two flags not glossed: `maneuvers_sha256` = 10 (a low-cardinality label sequence; collisions are
structural, no pixel information) and **`filename_overlap` = 44** — names collide completely while
content does not.

---

## 3. Other streams — material change only

- **Wide-FOV / geometry (PI-directed).** Frame **176×624 (117°) DECIDED by the PI**. ⛔ **Cannot
  start: pod2 holds the only copy of the cache.** The validation has now been killed twice — first by
  the seam guard (fixed, below), then by the outage.
- **4-brain dominance / v5.** No material change this interval; blocked behind the same cache.
- **IDM.** No material change. The 3-seed ensemble re-ship remains outstanding.
- **Datasets / D-B YouTube.** ⭐ **The block has LIFTED — MEASURED 05:22 UTC: 115 candidates, ~137 MB
  downloaded, ZERO bot-block messages**, from the **same pod3 IP** that was blocked on 07-26 (a real
  cooldown retry, *not* egress rotation). ⛔ **But 0 clips were produced and nothing from it is
  quotable** — it ran `--no-geocalib` at a fixed 100° HFOV, already measured wrong for 11 of 12 real
  clips. Real harvest waits for a free GPU.
- **H2 · Orin/Thor · AlpaSim.** **No material change this interval.** Not padded.

---

## 4. Retractions logged this interval (root-cause classes)

- **C51 — a guard that could only kill, never report.** The factorised-seam fail-loud fired on
  `ratio.max()` (one sample of 64) and its message named *"a code fault"* that is **impossible by
  construction** — the clamp `seam_clamp / ratio.clamp_min(seam_clamp)` cannot fail to bound. It cost
  the PI's geometry validation **both** wide arms (pre-clamp 1.760 and 1.511) on arms training **at or
  below** the control on every loss term; `C_v5` tripped at the **lowest** total/wm/plan_ade of its run.
- **C52 — a published remedy imported without its precondition.** My own E1d hypothesis. WiSE-FT holds
  when the fine-tune stays in the base's basin; MEASURED, it does not.
- **C53 — a workload priced by its name, not its stages.** I ran the YouTube harvest beside a trainer
  calling it "network/disk, light". MEASURED **477–483 % CPU**, load 24.39, trainer throughput
  **≥4× worse** (5.6× cumulative). ⭐ **And killing it left a raw 137 MB video on disk** — the pipeline
  deletes the source only *after* decode — a privacy violation manufactured by the remedy. Deleted;
  0 media files remain.

**Fixed properly this interval (PI instruction):** the seam guard now triggers on a **population
condition over time** — batch **mean** ratio > 1.5 **AND** ≥75 % of the batch at the clamp **AND** 50
consecutive steps; counter resets on any healthy step; `seam_clamp` untouched so **no computed value
moves, only when we abort**. The `seam_fail 8.0` workaround is **retired**. 5 tests incl. a
minority-can-never-kill pin. Suites: **stack/ 1581 passed, 12 skipped; taniteval/ 773 passed** — zero
new skips.

---

## 5. Decisions owed by Sayed

1. 🔴 **RESTART pod1 AND pod2 (RunPod console).** Blocking everything. **pod2 is binding** — sole copy
   of the wide-FOV cache *and* arm A's completed validation.
2. 🟡 **HF storage is full** (403 on push). pod1's 15k milestone checkpoint is **single-disk**.
3. 🟡 **Any further YouTube harvesting** — the one-shot authorization is spent; the block has lifted,
   so this is a fresh decision.
4. ⚪ Parked: closed-loop metric weights (`w`, `q`, the `r ≤ 0` blind spot, `lat_heading` 0.83); the
   wheelbase fix ("measure first"); the 30-pod-day X2 run; 2400 vs 2376 episodes.

**Already decided and actioned:** 176×624 · guard fixed properly · E1e approved · cleanup done.

---

## 6. Blocked, and on what

| item | blocked on |
|---|---|
| 176×624 wide training | **pod2** (sole cache copy) |
| `flagship-v2corpus-30k` + its 8-metric gate | **pod1** |
| Arm A/B/C geometry contrast | **pod2** |
| GeoCalib YouTube harvest + IDM pseudo-labels | **free GPU** (pod3 busy; serialized deliberately after C53) |
| Checkpoint backups | **HF storage quota** |

---

## 7. Next steps, priority order

1. **E1e-A frontier** (~1 h) → the pre-registered call. Success point ⇒ D-A deliverable and B follows;
   BOUND ⇒ three one-dimensional levers exhausted and the **junction-restricted buffer** becomes the
   experiment — targeting *what* is supervised rather than how heavily, per E1d's decomposition.
2. **On pod restart:** verify pod1's resumed step, then the formal 8-metric gate.
   ⚠️ `nonav_route_beats_majority` is **VOID BY CONSTRUCTION** — adjudicate **INSTRUMENT-FAIL**, never
   MODEL-FAIL. **Do not re-run arm A** — it completed `rc=0` and the `seam_fail` change is a proven
   no-op for it.
3. **Relaunch the geometry validation** (arms B and C) on pod2 with the redesigned guard.
4. **GeoCalib-correct YouTube harvest + IDM**, on a free GPU, sequentially.
