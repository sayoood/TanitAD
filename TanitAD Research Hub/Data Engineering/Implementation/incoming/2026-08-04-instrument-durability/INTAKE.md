# INTAKE — instrument durability: distance-keeping off-Drive, REF-C anchors rebuilt, §6 latency

- **Date:** 2026-08-04 · **Discipline:** Data Engineering · **Status:** PENDING orchestrator triage
- **Branch:** `agent/arch-inf-20260803` · **Hosts:** dev box + `tanitad-thor` (idle, aarch64 NVIDIA Thor)
- ⛔ No training pod was touched. `tanitad-new` (v5f) and `tanitad-pod4` (v1arch) were never contacted.

## HEADLINE

1. **Distance-keeping NOW RUNS ON THOR.** MEASURED: the canonical **881** val40 windows reproduce
   exactly off-Drive, **270 LEAD / 551 NO_LEAD / 60 NO_LABEL**, registration 40/40, and
   `four_families.longitudinal(..., lead=)` returns real headway / time-gap / min-TTC instead of
   `UNAVAILABLE`. It needed **29.36 MB**, not the 2.4 GB obstacle corpus.
2. **The rebuilt REF-C anchors REPRODUCE — but NOT bit-exactly.** 59/64 rows are bit-identical to
   the committed `refc_anchors_small64.pt`, the other 5 differ by **≤ 7.63e-06 m (≤ 64 ULP)**, and
   the **selection order is preserved**. Same vocabulary; not the same bytes.
3. ⛔ **THE BRIEF'S ANCHOR PREMISE WAS WRONG AND THE ERROR WAS LOAD-BEARING.** The cited
   `{n 256, pool 4096, seed 0}` is the model preset's **unused synthetic default**, not the anchor
   file's provenance. Building synthetically would have produced a **completely different
   vocabulary** and silently invalidated every REF-C comparison. See §2.
4. **§6's six latency figures: UNRESOLVED, and deliberately not replaced.** See §3 — a Thor number
   is not an A40 number, and no idle A40 exists.
5. ⭐ **A REAL BUG FOUND AND FIXED en route:** `taniteval.efficiency`'s contamination check was
   **structurally unable to certify any run on Thor** (Tegra `[N/A]` telemetry aborted the probe
   before the exclusivity check ran), so it quarantined every Thor result on an idle GPU. Fixed,
   two regression tests added, and the re-run now certifies `valid: true`. §3.3.
6. ⭐ **INDEPENDENTLY CORROBORATED.** A parallel Benchmarks & Eval stream built the same val40 lead
   join by a different code path and got **881 / 270 / 247** — identical. §4. It also means **two
   builders and two "CV floors" now exist**; that needs an orchestrator decision, not another doc.

---

## 1. P1 — `lead_source.py` is no longer single-checkout

### 1.1 What was mirrored

| file | dev box | Thor before | Thor after |
|---|---|---|---|
| `taniteval/taniteval/lead_source.py` | 21,428 B | **ABSENT** | present |
| `taniteval/taniteval/lead_metrics.py` | 20,109 B | **9,998 B (STALE — no `distance_keeping_by_speed`)** | 20,109 B |
| `taniteval/taniteval/four_families.py` | 33,487 B | 32,154 B (stale) | 33,487 B |
| `taniteval/taniteval/ci.py` | 16,601 B | 16,601 B | unchanged |
| `stack/scripts/lead_state_gate.py` | — | stale | synced |
| `stack/tanitad/refs/refc.py` | `c490bf3a…` | **`23c89f24…` DRIFTED** | `c490bf3a…` ✅ |

⚠️ Thor's `lead_metrics.py` was **half the size** of the dev box's — the `by_speed` wiring
`four_families.py:255` calls had never reached it. A `four_families` run there would have raised
`ImportError` on `distance_keeping_by_speed`, not degraded gracefully.

### 1.2 The transfer was 29.36 MB, not 3.1 GB — and how

The 40 val40 episodes touch **37 obstacle chunks**. Shipping those chunk zips plus their egomotion
siblings is **3,145.8 MB ⇒ ~25 min** at the measured 2.1 MB/s. But those 37 chunks hold ~3,700
clips and val40 needs **40**. `code/build_val40_lead_bundle.py` extracts only the per-clip members:

- `val40_lead_bundle.zip` — **29.36 MB**, `obstacle/<clip>.parquet` ×39 + `egomotion/<clip>.parquet` ×40
- transferred in **14.3 s at 2.05 MB/s** (MEASURED — corroborates the 2.1 MB/s LAN figure)
- **sha256 `ef60310a0a2512e85919bb919a5fb2139e639b40a8665f22f066411bed3e03b7` verified identical on
  both ends**, and then verified by *loading* every member — never by size or exit code.

### 1.3 ⭐ How a banked episode is re-joined to its corpus clip (reusable)

The val40 episode `.pt` carries **no clip UUID** — only `episode_id`, and `poses/actions/frames`.
The join is in `stack/tanitad/data/physicalai.py:740`:

```python
ep_id = int.from_bytes(clip["clip_id"].encode()[:4].ljust(4, b"\0"), "big")
```

⇒ **`episode_id` IS the first 4 characters of the clip UUID, packed big-endian.** MEASURED: all 40
val40 prefixes are **unique**, and each resolves to **exactly one** clip in
`r0/phase0_selection.parquet` (40/40, zero ambiguity). This is the general recipe for re-joining any
banked episode dump to `obstacle.offline` / `egomotion` without a clip list.

### 1.4 THE PROOF — reproduced numbers, not a file-exists check

`code/val40_lead_run.py` on Thor, `raw/val40_lead_report.json`:

| quantity | expected | MEASURED on Thor | |
|---|---|---|---|
| total windows | **881** (canonical) | **881** | ✅ |
| LEAD | ~270 | **270** | ✅ |
| NO_LEAD | — | **551** | — |
| NO_LABEL | 61 *(per brief)* | **60** | ⚠️ **off by one — see §1.6** |
| episodes with no `obstacle.offline` | `ep_00037.pt` | **`ep_00037.pt`** (22/22 NO_LABEL) | ✅ |
| registration | all | **40/40 OK, 0 failed** | ✅ |
| `poses_sha256` vs committed manifest | all match | **40/40 match** | ✅ |

- **Registration residual:** median ≤ **0.0098 m** across all 40 (sub-centimetre on 39).
- **Grid spacing recovered: 0.100496 – 0.101006 s.** ⚠️ This is the `int(span·10)` truncation the
  module warns about — **it is not 0.1 Hz⁻¹**, and it was FIT, not assumed. Assuming 0.1 would drift
  ~0.13 s over a 199-step episode ≈ **1.8 m of lead displacement** at 13.6 m/s.

### 1.5 The family actually turns ON — and the before/after that proves it

`code/val40_distance_keeping.py` → `raw/val40_distance_keeping.json`, on Thor, 270 LEAD windows:

| | `distance_keeping` status | mean headway (m) | mean time-gap (s) | mean min-TTC (s) | n |
|---|---|---|---|---|---|
| **GT** | **OK** | 28.9447 | 3.1655 | 25.0713 | 247 |
| **CV (hold-`v0`)** | **OK** | 28.0998 | 3.2272 | 23.1545 | 240 |
| **control: `lead=None`** | **`UNAVAILABLE`** | — | — | — | — |

⇒ **That last row is the proof.** The identical call without a lead block still reports
`UNAVAILABLE`; with it, real headway / time-gap / min-TTC come back. The family is fed, not merely
importable.

**D-LEAD-1 sign reproduction — 2 of 3, and I am not calling that a pass.**

| Δ (GT − CV) | val40 (this run) | D-LEAD-1 reference | sign |
|---|---|---|---|
| headway (m) | **+0.8449** | +0.9769 | ✅ |
| min-TTC (s) | **+1.9168** | +1.7474 | ✅ |
| time-gap (s) | **−0.0617** | +0.1641 | ❌ **FLIPPED** |

⚠️ **Most likely an estimator artifact of MY comparison, not evidence against the instrument** —
and I did not resolve it. GT and CV have **different n** (247 vs 240), so these are **unpaired
means over different window subsets**, while D-LEAD-1 used a **paired** episode-cluster bootstrap on
matched windows. CLAUDE.md's standing rule is explicit that two arms on the same windows take the
**paired** estimator. There is also **no CI here** and the surface is 270 windows vs 14,027.
⛔ **This table is NOT decision-grade and must not be cited as a refutation of D-LEAD-1.** The
correct follow-up is the paired episode-cluster bootstrap on the intersection — a work item, named.

### 1.6 ⚠️ NO_LABEL is 60, not the 61 the brief states — MEASURED, with full attribution

| episode | NO_LABEL windows | `obstacle.offline` | label span (s) |
|---|---|---|---|
| `ep_00016.pt` | 2 | present | 2.73 – 19.99 |
| `ep_00028.pt` | 9 | present | 0.04 – 12.19 |
| `ep_00034.pt` | 8 | present | 7.45 – 19.95 |
| **`ep_00037.pt`** | **22 (all)** | **ABSENT** | — |
| `ep_00038.pt` | 19 | present | 16.06 – 19.94 |
| **total** | **60** | | |

The 60 is fully attributed and internally consistent (2+9+8+22+19). The brief's **61** is
**INHERITED and not reproduced here**; I did not chase the one-window difference. It changes neither
LEAD = 270 nor the 881.

---

## 2. P2 — `refc_anchors_full.pt` rebuilt, and what the rebuild actually proves

### 2.1 ⛔ FIRST — the brief's premise was wrong, and the error was the dangerous kind

> *"`build_refc_anchors.py` is in the repo and the config records `{n 256, pool 4096, seed 0}`."*

`pool 4096` in `MODEL_REGISTRY.md:1299`/`:1309` is the **`refc_xl_config()` preset field**, i.e. the
size of the *synthetic* pool `build_refc_anchors.py` would use **if no `--data-root` were given**.
It is **not** the provenance of `refc_anchors_full.pt`. Three independent probes settle it:

1. `refc_anchors_small64.pt`'s own metadata: **`pool_size = 200000`** (the `max_pool` cap on the
   REAL-data path), `source = "base128[:64] == full256[:64] (nested FPS prefix, seed 0)"`.
2. `flagship_v4_anchors_dense.pt`: `pool_size 200000`, `source /workspace/v4run/anchsrc/physicalai-train-e438721ae894`.
3. `MODEL_REGISTRY.md:1241` states it in words for REF-B v2: *"Anchors are **FPS over real GT
   trajectory targets** built from the dataset at launch, **not the synthetic default**."*
   And `taniteval/registry.py:211` calls XL's anchors *"externally built"*.

⇒ **A synthetic `--pool-size 4096` rebuild would have produced a different vocabulary that still
loaded, still had shape `[256, 4, 2]`, and would have silently invalidated every REF-C number.**
This is exactly the failure the brief warned about, arriving through the brief itself.

### 2.2 The rebuild

Run on Thor against the **parity** cache — and the parity guard fired and passed:

```
[parity] *train* cache: physicalai-train-e438721ae894 VERIFIED — 2376 episodes,
         uid sha256 9877bef64da3… matches the committed manifest (skip-hash f09e44db).
```

`--data-root /home/nvidia/epcache/epcache-256px-phase0 --n-anchors 256 --horizons 5,10,15,20
--seed 0 --max-pool 200000` → `[256, 4, 2]`, `pool_size 200000`, `source physicalai-train-e438721ae894`.

### 2.3 THE VERDICT — same vocabulary, different bytes

`raw/anchor_reproduction.json`:

| check | result |
|---|---|
| `torch.equal(rebuild[:64], small64)` | **False** |
| rows **bit-identical** | **59 / 64** |
| differing rows | 4, 19, 37, 40, 63 |
| global max abs diff | **7.63e-06 m** (7.6 µm) |
| global max ULP distance | **64** |
| max relative diff | 7.05e-06 |
| `allclose(rtol=0, atol=1e-5)` | **True** |
| **selection ORDER preserved** (row *i* ↔ row *i*) | **True** |
| **Thor rerun vs itself** | **`torch.equal` True, identical sha256** |

**Reading.** FPS selected the **same 256 trajectories in the same order**; 5 of the first 64 carry
accumulated float32 rounding differences of up to 64 ULP. Physically these are the same anchors to
7.6 micrometres.

Three facts constrain the cause:
1. **Not nondeterminism** — a second identical Thor run is `torch.equal` to the first, same sha256.
2. **Not a different selection** — the 5 differing rows appear **nowhere** in the rebuilt 256 as
   exact matches (set-membership test: 59/64), i.e. they are *rounded versions* of the same
   trajectories, not substitutions.
3. ⇒ The pool itself differs in its last bits, so `waypoint_targets`' rotation arithmetic rounded
   differently on the rebuild host.

⚠️ **What I CANNOT separate:** the rebuild host differs from the original in **both** architecture
(aarch64 vs amd64) **and** toolchain (**torch 2.13.0+cu130** here; pod3's version is not recorded in
any artifact I found). I state the cause as **cross-host float32 rounding** and do **not** claim it
is architecture specifically — that would need an amd64 host carrying the parity cache, which the
programme does not currently have idle.

### 2.4 ⚠️ CONSEQUENCE — do not swap this in as the scoring vocabulary yet

`refc_scale_ab.analyze`'s nested-vocabulary control uses **`torch.equal`** (per `provenance.json`:
*"torch.equal(small64, base128[:64]) AND …"*). A Thor-rebuilt `full256` **fails that check** despite
being the same anchor set. So `refc_anchors_full_REBUILD.pt` is admissible as a **provenance
reconstruction and a citation target**, and is **NOT** admissible as a drop-in replacement until
either (a) a bit-exact **amd64** rebuild is produced, or (b) the nesting check is changed from
`torch.equal` to a stated tolerance — **a decision for the PI/orchestrator, not for me.**

⛔ I did **not** rename it `refc_anchors_full.pt`. It is staged as
`refc_anchors_full_REBUILD.pt` precisely so it cannot be mistaken for the original.

---

## 3. P3 — the six §6 latency figures: **UNRESOLVED**, and not replaced

### 3.1 What I independently confirmed (MEASURED, ours)

Re-read the committed artifacts directly:

| arm | §6 prose (fp32/tf32/amp16) | committed JSON `plan_step.p50_ms` | JSON `env.gpu` |
|---|---|---|---|
| flagship-30k | 103.42 / 93.76 / 104.49 | **97.3199 / 97.6981 / 123.8325** | **NVIDIA A40** |
| refc-xl-30k | 44.28 / 27.84 / 26.12 | **44.0647 / 27.7808 / 20.9993** | **NVIDIA A40** |

⇒ The prose figures are in **no** committed artifact. Confirmed, not inherited.

### 3.2 ⛔ Why I did NOT repoint them to a Thor measurement

**Every committed efficiency artifact records `env.gpu = "NVIDIA A40"`. Thor is aarch64 / NVIDIA
Thor.** Writing a Thor number into the A40 row would repeat, exactly, the defect §6 already
documents at length — the "11.16 ms deploy tick" that was quoted against a 103.42 ms planning tick,
where §6's own correction reads: *"The two figures differ in **five** dimensions at once and are not
comparable."* Substituting hardware silently would manufacture the defect it is meant to remove —
the same reason the citation-fixing stream deliberately left these six alone.

**And no A40 is AVAILABLE — MEASURED by me 2026-08-04, not inherited from the brief.** ⚠️ Note the
precise claim: A40s **exist**; they are **busy**. This is re-measurable later, not lost.

| host | probe result |
|---|---|
| `tanitad-pod4` | **NVIDIA A40, util 100 %**, compute PID 9076 holding 13,656 MiB → **TRAINING** |
| `tanitad-new` | **NVIDIA A40, util 59 %**, compute PID 1654683 holding 16,054 MiB → **TRAINING** |
| `tanitad-pod` / `tanitad-pod3` / `tanitad-eval` | `Connection refused` (all three) |
| `tanitad-pod2` | TERMINATED (per ssh config) |

⛔ Both live A40s are training, and CLAUDE.md forbids evaluating on a training pod — an efficiency
benchmark there would be contaminated by construction *and* would perturb a running job.
*(Probes were read-only `nvidia-smi` queries. ⛔ No `pgrep -f` / `pkill -f` — those self-match the
ssh command.)*

⇒ **Recommendation: mark the six figures UNRESOLVED in §6** with the reason *"no surviving artifact
from the original session; both A40s are occupied by training as of 2026-08-04"*, and quote the
committed JSONs (**97.32 / 97.70 / 123.83** and **44.06 / 27.78 / 21.00**, NVIDIA A40) wherever a
number is needed. The ranking they support is unaffected.
⭐ **This is a schedulable work item, not a dead end:** re-run `taniteval.efficiency` on
`tanitad-pod4` or `tanitad-new` the moment its training finishes, and §6 can be repaired with a real
A40 number. ⚠️ I did **not** edit `MODEL_REGISTRY.md` — that is the orchestrator's call.

### 3.3 ⛔ A REAL BUG FOUND AND FIXED: the contamination check is inoperative on Thor

The first Thor run came back with **both arms quarantined** as
`eff_<key>.CONTAMINATED-<ts>.json`. That looked like GPU contention. **It was not.**

```
"contamination_check": {"gpu_exclusive_before": null, "gpu_exclusive_after": null, "valid": false}
"gpu_state_before":    {..., "error": "ValueError: could not convert string to float: '[N/A]'"}
```

**`null`, not `False`.** Root cause, MEASURED: Tegra/Jetson `nvidia-smi` returns `[N/A]` for
`memory.used` and `clocks.sm` —

```
$ nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw ...
NVIDIA Thor, 0, [N/A], [N/A], 39, 2.77
```

— so `float(f[2])` raised inside `_gpu_state`'s **single shared `try`**, aborting the function
**before** the `--query-compute-apps` probe on the next lines ever ran. `exclusive` stayed `None`,
and `run_and_save` quarantines anything not strictly `True`.

⇒ **`taniteval.efficiency` was structurally unable to certify ANY run on Thor**, on any GPU state.
And the probe that would have answered works perfectly there: `nvidia-smi --query-compute-apps`
returned **empty, exit 0** — the GPU genuinely *was* exclusive. This is the C9/C13/C14 class: an
instrument that cannot report the answer it is cited for.

**FIXED** in `taniteval/taniteval/efficiency.py`:
- new `_num()` parses a telemetry field to `None` instead of raising on `[N/A]`;
- the two `nvidia-smi` probes are now in **independent `try` blocks**, so descriptive telemetry can
  never suppress the exclusivity decision;
- a non-zero `nvidia-smi` exit now raises rather than being read as "no processes";
- missing fields are surfaced as `telemetry_unavailable` rather than silently null.

Two regression tests added to `taniteval/tests/test_efficiency.py`: a Tegra-`[N/A]` case that must
still read `exclusive True`, and a real-neighbour case that must still read `False` with the A40
parse unchanged. **`pytest taniteval/tests/test_efficiency*.py` → 40 passed.**

⚠️ **This is an integration item, not just a fix:** any *past* Thor efficiency artifact carrying
`CONTAMINATED` in its name may have been perfectly clean. The name is not evidence.

### 3.4 What I DID measure — a new, separately-labelled Thor row

Thor is the **edge deployment target** and §6 has no Thor planning tick at all, so the measurement
is worth having on its own terms. `code/thor_planning_tick.py` → `raw/THOR_PLANNING_TICK.json`,
re-run after the §3.3 fix so the exclusivity is *certified* rather than assumed. Protocol identical
to the A40 harness (batch 1, warmup 30, 200 iters, `torch.cuda.Event`, synchronize-bracketed).
Registry ckpt paths point at the dead eval pod's `/root/models/...`; Thor has no sudo, so they are
overridden **in-process** — no checkout file was edited.

⚠️ Reported as p50 **and** p95/p99 over 200 warmed iterations, never a single call — the
"224.98 ms render" lesson.

**Hardware, stated explicitly:** `NVIDIA Thor`, aarch64, compute capability **11.0**, 125,772 MiB,
**torch 2.13.0+cu130 / CUDA 13.0**. `contamination_check.valid = **true**`, `other_compute_procs 0`
before and after, on both arms.

**PLANNING TICK — NVIDIA Thor (MEASURED 2026-08-04, ours, `raw/THOR_PLANNING_TICK.json`)**

| arm (step 29,999) | fp32 p50 / p99 | tf32 p50 / p99 | amp16 p50 / p99 |
|---|---|---|---|
| **flagship-30k** | **183.72** / 188.87 | **81.58** / 85.06 | **73.01** / 74.57 |
| **refc-xl-30k** | **164.88** / 165.52 | **55.73** / 56.00 | **46.42** / 47.48 |

- **10 Hz budget (100 ms):** both arms MISS it in fp32; both MEET it at **p99** in tf32 and amp16.
- **Reproducibility:** an independent second run gave refc-xl fp32 **164.72** vs **164.88** (0.1 %),
  tf32 55.58 vs 55.73, amp16 46.54 vs 46.42 — so these are steady-state, not first-call.
- Peak `torch.cuda.max_memory_allocated` **1466.4 MiB** (⚠️ the only admissible memory read on Thor —
  `free`/`tegrastats`/`VmRSS` all lie there).

⭐ **The REF-C-over-flagship ranking DIRECTION holds on Thor, but the MARGIN collapses:**
**1.11× / 1.46× / 1.58×** (fp32/tf32/amp16) versus the A40's 2.2–4.6×. Worth a look — but it is one
host, one batch size, and **not** something I chased further.

⛔ **This row does not replace §3.1 and must never be quoted as an A40 number.**

---

## 4. ⭐ INDEPENDENT CORROBORATION — and a duplication to resolve

While staging I found **another agent's work already in the index**:
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-04-distance-keeping-arms/`
(incl. `code/build_val40_lead_block.py`, `code/thor_extract_poses.py`). It builds the **same**
val40 → `win["lead"]` join, independently.

**It agrees with this package exactly on the load-bearing numbers:**

| quantity | this package | `distance-keeping-arms` |
|---|---|---|
| windows rebuilt | **881** | **881** |
| LEAD windows | **270** | **270** |
| GT-oracle windows retained | **247** | **247** |

⇒ Two independent implementations, two different code paths, identical 881 / 270 / 247. That is far
stronger evidence than either alone, and it is the reason I state P1 as settled rather than likely.

⚠️ Their `cv` retains **236** where my hold-`v0` retains **240**. **Not a conflict — a different CV.**
Their own `_cv_note` records it: the dump's `cv` is `baseline_waypoints()['constant_velocity']`
(finite-difference last-step velocity, which carries a lateral component), while mine is
`[v0·t, 0]` straight-ahead on the egomotion speed at t0. They quantify the gap at
`cv_max_abs_err_m_vs_holdv0 = 1.3434 m`. **Two "CV floors" are live in the programme
simultaneously** — that is a retraction waiting to happen and it should be named canonically.

⛔ **ESCALATION — duplicate instrument.** Two val40-lead builders now exist. One should be canonical
and the other should call it. I did **not** merge them: that is an orchestrator decision, and
silently deleting a sibling's staged work is exactly what the git-hygiene rule forbids.

## Parity / privacy

- **PARITY UNTOUCHED.** Read-only over `labels/*.zip` and `r0/phase0_selection.parquet`; no clip
  re-selected. The anchor rebuild's own guard VERIFIED `physicalai-train-e438721ae894`, 2376
  episodes, skip-hash `f09e44db`.
- **PRIVACY.** PhysicalAI-AV is gated. The bundle (raw gated parquets) stays on Thor and the dev
  box and is **NOT** committed. The staged index `raw/val40_lead_index_ANON.json` carries
  `clip_<sha256[:8]>` only — no UUID.

## Tests

| suite | result |
|---|---|
| `tools` | **246 passed** in 130.5 s ✅ (matches the expected 246) |
| `taniteval/tests/test_efficiency.py` + `test_efficiency_levers.py` (incl. **2 NEW**) | **40 passed** ✅ |
| `taniteval/tests/test_lead_source.py` + `test_lead_metrics.py` | **32 passed** ✅ |
| `stack` (2043) | ⚠️ **NOT COMPLETED — see below** |
| `taniteval` full (~810) | ⚠️ **NOT COMPLETED — see below** |

⚠️ **MY OWN PROCESS ERROR, logged rather than hidden.** I launched the `stack` and `taniteval`
suites **concurrently** on the dev box **without `OMP_NUM_THREADS`**. That is precisely the trap
CLAUDE.md documents — *"torch spawns ~113 threads PER PROCESS, and concurrent arms then make NO
PROGRESS — it looks exactly like a hang"* — and it presented exactly as documented: `stack` froze at
**27 %** for many minutes with no failures and no movement. MEASURED: 16 python processes, 8 of them
holding 19–84 threads each (~440 threads).

I stopped my own tasks rather than `kill` broadly, **because a sibling agent is working concurrently
in this repo and I could not attribute every PID.** ⛔ Killing an unattributable process is not worth
a test count.

**Nothing here indicates a real failure:** no test failed in either suite before the stall, `stack/`
is **untouched by this package** (my only code changes are in `taniteval/`), and the tests that
actually cover those changes are the 40 + 32 above, all green. **The two full-suite counts are a gap
in this report, and I am stating it rather than quoting a stale number.**

## DELIVERABLE MANIFEST

⚠️ **Nothing produced here lives in only one place**, except the two gated-corpus items marked ⛔
(which are deliberately not committed).

| artifact | where it lives |
|---|---|
| `INTAKE.md` (this file) | `repo:…/incoming/2026-08-04-instrument-durability/INTAKE.md` |
| `refc_anchors_full_REBUILD.pt` (10,215 B, `[256,4,2]`) | `repo:…/2026-08-04-instrument-durability/` + `tanitad-thor:/home/nvidia/leadwork/refc_anchors_full_rebuild.pt` |
| `raw/anchor_reproduction.json` | `repo:…/raw/` |
| `raw/val40_lead_report.json` | `repo:…/raw/` + `tanitad-thor:/home/nvidia/leadwork/` |
| `raw/val40_distance_keeping.json` | `repo:…/raw/` + `tanitad-thor:/home/nvidia/leadwork/` |
| `raw/THOR_PLANNING_TICK.json` | `repo:…/raw/` + `tanitad-thor:/home/nvidia/leadwork/eff_thor/` |
| `raw/eff_flagship-30k.json`, `raw/eff_refc-xl-30k.json` (Thor, certified) | `repo:…/raw/` + `tanitad-thor:…/eff_thor/` |
| `raw/val40_lead_index_ANON.json` (clip_sha8 only) | `repo:…/raw/` |
| `code/build_val40_lead_bundle.py` | `repo:…/code/` |
| `code/val40_lead_run.py` | `repo:…/code/` + `tanitad-thor:/home/nvidia/leadwork/` |
| `code/val40_distance_keeping.py` | `repo:…/code/` + `tanitad-thor:/home/nvidia/leadwork/` |
| `code/thor_planning_tick.py` | `repo:…/code/` + `tanitad-thor:/home/nvidia/leadwork/` |
| **FIX** `taniteval/taniteval/efficiency.py` (`_num`, split probes) | `repo:` + `tanitad-thor:/home/nvidia/TanitAD/taniteval/taniteval/` |
| **FIX** `taniteval/tests/test_efficiency.py` (2 new tests) | `repo:` |
| ⛔ `val40_lead_bundle.zip` (29.36 MB, **gated PhysicalAI parquets**) | `tanitad-thor:/home/nvidia/leadwork/` + dev-box scratchpad — **NOT committed, by policy** |
| ⛔ Thor `taniteval` mirror (`lead_source/lead_metrics/four_families/progress/lateral/driving/ci`) | `tanitad-thor:/home/nvidia/TanitAD/taniteval/taniteval/` — copies of repo files, no new content |

## ESCALATIONS (do not leave these in a README)

1. ⛔ **`refc_anchors_full_REBUILD.pt` must not be renamed to `refc_anchors_full.pt`** until the
   `torch.equal` nesting check is either satisfied by a bit-exact **amd64** rebuild or relaxed to a
   stated tolerance. **PI/orchestrator decision.**
2. ⛔ **Duplicate val40-lead builder** vs `2026-08-04-distance-keeping-arms` — pick one canonical.
3. ⛔ **Two different "CV floors"** are in simultaneous use (hold-`v0` vs finite-difference),
   differing by up to **1.3434 m**. Name one canonical before a comparison is published on the wrong one.
4. ⚠️ **§6's six latency figures**: recommend marking UNRESOLVED with the reason. I did **not** edit
   `MODEL_REGISTRY.md`.
5. ⚠️ **Past `.CONTAMINATED-*` efficiency artifacts from Thor may be clean** — re-check, don't discard.
6. ⚠️ **Open work item:** paired episode-cluster bootstrap for the val40 GT-vs-CV distance-keeping
   deltas (§1.5), to settle the time-gap sign flip properly.
