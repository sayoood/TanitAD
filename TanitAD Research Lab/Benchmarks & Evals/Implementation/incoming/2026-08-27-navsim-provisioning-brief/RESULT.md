# NAVSIM PROVISIONING SCOUT — RESULT

`TanitAD_EvalFlyWheel, 2026-08-27. Commissioned by the Master Mind on PI approval.
⛔ ZERO-COST scouting only: nothing downloaded beyond documents already banked, no GPU, no
HF job, no spend. Evidence class on every claim. Sources banked in the Library.`

**Evidence classes:** `PUB-PAPER` (read by me from the banked PDF) · `PUB-DEVKIT` (the navsim
repo/docs at the pinned SHA) · `INHERITED` (another stream's extraction, not re-verified by
me) · `ESTIMATED` (my arithmetic, assumptions stated) · `UNVERIFIED`.

---

## 0. Headline for the PI

**GO for preparation, NO-GO for a full download.** The cheap, decision-grade path is
`navhard_two_stage` — **31 GB**, not the 223 GB `navtest` — and it is the split the current
SOTA papers actually report EPDMS on.

⭐ **The comparability worry in the brief is smaller than assumed, and I can now say so with a
number.** The brief anticipated that our single 256×640 front camera would be incomparable
against multi-camera + LiDAR entries. **It is not.** Drive-JEPA's own headline is produced in a
**perception-free, front-camera-only** setting, and there is a published ladder of exactly that
class to sit in.

⛔ **And one correction to the commissioning brief**, stated up front because it would otherwise
propagate: **Drive-JEPA's v1 figure is 93.7 PDMS, not 93.3** — and neither number is the one we
should be measured against (§1.2).

---

## 1. Harness — the version the SOTA tables use

### 1.1 The pin

| what | value | class |
|---|---|---|
| repo | `github.com/autonomousvision/navsim` | PUB-DEVKIT |
| pinned SHA used for our protocol spec | **`0a380a9`** (2025-10-27) | PUB-DEVKIT |
| **v1 PDMS** (`navtest`) | ⛔ the **`v1.1` branch** — NOT `main` | PUB-DEVKIT |
| **v2 EPDMS** | `main`, **v2.2** (2025-04-28, the AGC2025 devkit) | PUB-DEVKIT |
| licence | research use; nuPlan/OpenScene terms flow through | INHERITED |

⛔ **The single most expensive mistake available here:** `main` is NavSim **v2**, and its scorer
is the **EPDMS** scorer. Computing "v1 PDMS" from `main` silently produces a different metric.
NavSim v1 and its `navtest` leaderboard live on the **`v1.1` branch**.

⚠️ **A post-SOTA bugfix exists and matters.** 2025-09-29 (issue #151): `multiplicative_metrics_prod`
and `weighted_metrics` *were not correctly excluded by the human filter*. Both papers below are
2026 preprints and therefore **postdate** it — but neither states a devkit SHA, so which side of
the fix they are on is **UNVERIFIED**. ⇒ **Pin our own SHA and publish it beside our number.**

### 1.2 ⛔ What the SOTA numbers actually are — read from the primaries, not the summaries

**Drive-JEPA** (`2601.22032`, banked; I extracted the text myself — **PUB-PAPER**):

> abstract & §1, twice: *"achieves **93.7 PDMS** on v1 and **87.8 EPDMS** on v2"*

The **93.3** in the commissioning brief appears **only in the paper's own NeurIPS checklist**,
where it misquotes its own abstract. **The paper's claim is 93.7.** The brief's 87.8 EPDMS is
correct. ⇒ Quote 93.7, and note the internal inconsistency if the figure is ever load-bearing.

**Latent-WAM** (`2603.24581`, banked, PUB-PAPER): **89.3 EPDMS on NavSim v2**, "perception-free".
Confirmed verbatim from the PDF. The brief's figure is correct.

### 1.3 ⭐ The row we would actually be compared to

Drive-JEPA's 93.7 is its **full framework**: input **1024×256, stacking front + left + right**
cameras (PUB-PAPER). That is *not* our configuration. Its **perception-free** configuration —
*"only the front camera at 512×256"*, no LiDAR — is, and it has its own published table:

**PUB-PAPER, Drive-JEPA Table 1 — perception-free planners, front-view camera only:**

| method | encoder | pretrain data | PDMS |
|---|---|---|---|
| LAW | **21 M** | ~20 h | **83.8** |
| World4Drive | **21 M** | ~20 h | **85.1** |
| Epona | 1.1 B | 128 h | 86.1 |
| **Drive-JEPA** | 307 M | 330 h | **89.0** |

⭐ **This is the ladder TanitAD belongs on**, and it is a good place to be:
- it is **front-camera-only** — our modality, not a multi-sensor rig;
- it is **perception-free** — *"supervised solely by human trajectories without relying on
  perception annotations"* (PUB-PAPER), which is our setting;
- the entry rung is **21 M parameters at 83.8 PDMS**, far inside our sub-300 M budget, and the
  top rung is **307 M**, i.e. our own scale class.

⇒ **A front-camera-only TanitAD row is a legitimate published-class entry, not an asterisked
outlier.** It must still be **LABELLED** front-camera-only, perception-free, with its devkit SHA
— but it is comparable *in kind*, which is what the brief asked us to establish.

⚠️ **The leaderboard server does not record modality** (PUB-DEVKIT: only `TEAM_NAME`, `AUTHORS`,
`EMAIL`, `INSTITUTION`, `COUNTRY` are carried). Every "camera-only" claim on a leaderboard row is
the authors', not the server's. Our labelling obligation is therefore ours alone to keep.

---

## 2. Data — sizes REPORTED, nothing downloaded

**PUB-DEVKIT** `docs/splits.md`. Splits are *scene filters* over downloadable *dataset splits*.

| group | split | logs | sensors | needed for |
|---|---|---|---|---|
| OpenScene | `trainval` | 14 GB | **> 2000 GB** | training only |
| OpenScene | `test` | 1 GB | 217 GB | parent of navtest |
| OpenScene | `mini` | 1 GB | 151 GB | — |
| NavSim | `navtrain` | 14 GB | **445 GB** (300 GB without history) | training only |
| NavSim | **`navtest`** | 983 MB | **223 GB** | **v1 PDMS headline** |
| NavSim | **`navhard_two_stage`** | 892 MB | **31 GB** | **v2 EPDMS** |
| Competition | `warmup_two_stage` | 27 MB | **1.2 GB** | ⭐ smoke run |
| Competition | `private_test_hard_two_stage` | 14 MB | 11 GB | AGC2025 submission |

**Counts:** `navtrain` 103k samples · `navtest` 12k samples (PUB-PAPER) · `navhard` **450 Stage-1
+ 5462 Stage-2 observations** (PUB-PAPER).

⚠️ **`navmini` does not exist as a published split.** A "396 scenarios" figure circulates in
secondary summaries and is in neither `docs/splits.md` nor either paper — **UNVERIFIED, do not
quote.** The real smoke split is **`warmup_two_stage` at 1.2 GB**.

⚠️ `navtrain` sensors ship as eight `navtrain_{current,history}_{1..4}.tgz` archives with
**published MD5s — use them**; missing files on `navtrain` are a reported recurring failure.

---

## 3. The wrap — our stack to their agent API

Our adapter already exists: **`taniteval/adapters/navsim.py`** (60 KB, **50 synthetic tests
green**, no dataset required). What remains between it and a real submission:

| # | mismatch | ours | theirs | cheapest adapter | cost |
|---|---|---|---|---|---|
| 1 | **camera set** | 1× **256×640 cylindrical** front | 8× 1920×1080 pinhole (`cam_f0…cam_b0`) | consume `cam_f0` only; **reproject pinhole → our cylindrical** using their published intrinsics | ~1 d |
| 2 | **projection** | cylindrical, column linear in azimuth, `f_ref` 305.577, **120° FOV** | pinhole | ⛔ the pinhole FOV formula is WRONG on our projection (92.6° vs true 120°) — reprojection must be explicit, never a resize | in #1 |
| 3 | **LiDAR** | none | merged 5-sensor `(6,n)` | decline it; `SensorConfig` supports per-stream flags | 0 |
| 4 | **output form** | **(a, κ) unicycle** | 41 poses `(x, y, heading)`, ego frame, 4 s @ 10 Hz | integrate our unicycle → 41 poses. ⚠️ **their scorer asserts `states.shape[1] == num_poses + 1`** | ~0.5 d |
| 5 | **horizon** | our tactical horizon | **4 s**, data 2 Hz, simulation 10 Hz (interpolated) | resample; state the horizon in the artifact | in #4 |
| 6 | **history** | our window | `num_history_frames = 4` (⚠️ 1.5 s elapsed, docs say "2 s") | map our window; **state the span you mean** | ~0.5 d |
| 7 | **ego status** | ⛔ **we forbid it at inference** | `ego_velocity`, `ego_acceleration`, `ego_pose` always populated | ⛔ **there is NO switch that removes ego status, and nothing verifies an agent declined to read it** — our vision-only claim must be enforced and evidenced **on our side** | design, not code |
| 8 | **driving command** | predicted goal | 4-dim one-hot (⚠️ paper says 3; **implement 4**) | admissible as a *route* signal; ⛔ must not carry situation-classifier output | — |
| 9 | **estimator** | episode-cluster bootstrap | scene token, not episode | ⛔ **UNRESOLVED — our bootstrap does not transfer.** The adapter emits the point estimate and an explicit `UNAVAILABLE` interval rather than inventing one | **open** |
| 10 | **four families** | binding | NavSim reports none of them | refuse each inline with reason + n; **an EPDMS number alone does not satisfy the four-families rule** | done in adapter |

⛔ **Two protocol traps already encoded in the adapter:** read **`score`, never `pdm_score`**
(the latter masks Extended Comfort and divides by 14; true EPDMS has denominator 16 and is
assembled downstream in `run_pdm_score.py::compute_final_scores`); and **submitted velocity and
acceleration are discarded** — the devkit notes *"velocity and acceleration ignored by LQR +
bicycle model"*, so output is poses only.

⚠️ **Open leak question, load-bearing and UNVERIFIED:** whether nuPlan's `route_roadblock_ids`
— the source of the driving command — is itself derived from the expert's driven path. The
devkit reads it from the nuPlan DB with no derivation exposed. This is our own *"a supplied route
is optimistic by construction"* rule in their costume. **Check it before quoting any
route-conditioned NavSim result as route-following skill.**

---

## 4. Compute estimate — ESTIMATED, no jobs run

Assumptions stated so they can be corrected: the agent is queried **once per scene**
(PUB-PAPER), so inference is one forward pass per scene, not per simulated tick; our measured
Thor throughput is **12.3–14.1 windows/s**, flat across a 6× batch range (20 SMs saturate at
batch 8); metric simulation is **CPU-bound and dominates**.

| split | scenes | Thor inference | RTX 4060 inference | metric sim (CPU) | wall-clock |
|---|---|---|---|---|---|
| `warmup_two_stage` | ~small (1.2 GB) | **< 2 min** | < 5 min | ~10–20 min | **~20 min** |
| `navhard_two_stage` | 450 + 5462 obs | **~7–8 min** | ~20–25 min | ~1.5–2.5 h | **~2–3 h** |
| `navtest` | 12 000 | **~15 min** | ~45 min | ~2–4 h | **~3–4 h** |

**Cross-check (PUB-DEVKIT):** the official leaderboard quotes **≈2 h evaluation turnaround** per
submission, which brackets the `navhard` row and says the CPU metric stage, not the GPU, is the
cost centre. ⇒ **GPU is not the constraint. Disk and CPU are.**

**HF Pro:** ⛔ **not estimated as a plan.** The quota is a hard ceiling and CPU-heavy metric
simulation is the wrong shape for metered GPU. **Recommendation: do not use HF GPU for this.**
Local (Thor / dev box) is unmetered and sufficient.

---

## 5. GO / NO-GO cost table

| option | disk | download | wall-clock | what it buys | verdict |
|---|---|---|---|---|---|
| **A — `warmup_two_stage` smoke** | **1.2 GB** | minutes | ~20 min | end-to-end proof the adapter runs a real NavSim scene; shakes out all 10 mismatches | ⭐ **GO — recommend first** |
| **B — `navhard_two_stage` (v2 EPDMS)** | **31 GB** | ~1–2 h | ~2–3 h | **a real EPDMS number comparable in kind to Latent-WAM 89.3 and Drive-JEPA 87.8** | ⭐ **GO after A** |
| **C — `navtest` (v1 PDMS)** | **223 GB** + 1 GB logs | ~8–15 h | ~3–4 h | the v1 PDMS ladder (LAW 83.8 → Drive-JEPA 89.0) | ⚠️ **defer** — 7× the disk of B |
| **D — `navtrain` (training)** | **445 GB** (300 without history) | ~1 day | — | training on their data | ⛔ **NO-GO now** — not needed to be measured |
| **E — leaderboard submission** | — | — | ≈2 h/submission, 1/day | external ranking | ⛔ **NO-GO** — needs a PI decision + 5 metadata fields |

**My recommendation: A now, B on the PI's word, C only if the v1 ladder is judged worth 223 GB.**
Total to a real, publishable external number: **32 GB and roughly half a day of local compute.**

⚠️ **Two things that are NOT resolved by spending the 32 GB**, and should be settled first
because they change what the number *means*:
1. **The estimator** (§3 row 9). Scene tokens are not episodes. **Do not report a NavSim CI until
   this is settled** — a point estimate with an honest `UNAVAILABLE` interval is admissible; an
   invented interval is not.
2. **Ego-status enforcement** (§3 row 7). Their framework cannot switch ego status off. Our
   vision-only compliance is ours to enforce and evidence.

---

## 6. Deliverable manifest

| artifact | location | state |
|---|---|---|
| **this file** | `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-08-27-navsim-provisioning-brief/RESULT.md` | written |
| NavSim protocol spec (900 lines, the evidence base) | `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md` | staged, branch `claude/zen-bose-eb35e9` |
| NavSim adapter + 50 tests | `taniteval/adapters/navsim.py`, `taniteval/tests/test_navsim_adapter.py` | staged, 53 tests green |
| NavSim primary, tagged `navsim`, citing this file | `2406.15349` in `Library/library.json` | staged, sha256-verified |
| Pseudo-simulation primary (v2/EPDMS), tagged `navsim` | `2506.04218` | staged, sha256-verified |
| Library integrity | `kb_add.py --verify` → **44 entries, 0 orphans, 0 problems** | verified by content |

**Read for this scout but NOT re-banked** (they are already banked on `agent/arch-inf-20260803`,
and re-banking would attribute another stream's work to mine): `2601.22032` Drive-JEPA,
`2603.24581` Latent-WAM. I extracted their text from that branch to verify §1.2 and §1.3 myself.

⛔ **Nothing committed, nothing pushed. Nothing downloaded. No GPU, no HF job, no spend.**

⚠️ **INTEGRATION — already escalated to the Master Mind in a separate message, repeated here so
it is not lost:** my 25 Library files are staged under **`TanitAD Research Hub/`**, a path that
**no longer exists** on `agent/arch-inf-20260803` (it is `TanitAD Research Lab/`). A naive merge
resurrects a second parallel library tree, and `library.json` is whole-file JSON: **yours 78
entries · mine 44 · shared 24 · only-mine 20 — including `2406.15349`, the NavSim paper, which is
absent from your index.** Merge by **key union**, never file-replace, and re-path my files first.

---

## 7. ADDENDUM 2026-08-27 — the two blockers are now ENFORCED, not pending

§5 closed with two things that 32 GB does not fix. They are no longer a promise in a message:
they are **blocking gates in `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.3.0**, each with a
regression arm in `tools/tests/test_criteria_check.py` (**55 tests green**; full `tools/` suite
**362 passed** on the merged tree).

| gate | blocking rule |
|---|---|
| `navsim.estimator_unit` | a NavSim artifact MUST declare `estimator.cluster_unit`; until the unit is settled and pre-registered, the ONLY admissible interval is `{status:UNAVAILABLE, reason, n}`. ⛔ A bare point estimate, or an episode-cluster CI, FAILS. |
| `navsim.ego_enforcement` | MUST name the **mechanism** by which the agent did not read ego status, plus its evidence. ⛔ *"we did not use it"* is an assertion, not enforcement, and FAILS. |
| `navsim.modality_label` | MUST declare `protocol.sensor_set` and `protocol.setting`. ⛔ Comparing our row against a multi-camera or perception-based row without both FAILS. |

⭐ **Why the estimator gate is phrased around overlap, not just units:** NavSim scenes are
explicitly allowed to overlap (PUB-DEVKIT `docs/splits.md`: *"NavSim splits contain overlapping
scenes"*). Resampling overlapping units as if independent understates variance — the same family
as `overlapping_holdout_se`, which biases the POINT ESTIMATE bidirectionally up to a sign flip.
Inventing a CI here would reproduce a retracted defect on a borrowed benchmark.

The corrected reference numbers (93.7 not 93.3; the perception-free ladder LAW 83.8 → Drive-JEPA
89.0; the `v1.1`-branch pin) are recorded in the same registry block under
`published_reference_numbers`, with a test asserting the ladder's entry rung stays inside our
parameter budget — so a future report cannot quietly drift to the flattering multi-camera row.
