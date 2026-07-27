# D-B YouTube retry — 2026-07-27

**Agent:** `yt-dB-retry` (2026-07-27) · **Pod:** `tanitad-pod3` (A40) · pod1/pod2/eval untouched
**Repo HEAD at start:** `a7ba1b2` · branch `agent/benchmarks-eval-20260721`
Evidence classes used throughout: **MEASURED** (ours + artifact path) · **PUBLISHED** · **INHERITED**
(not re-verified) · **ESTIMATED** · **HYPOTHESIS**.

---

## ⛔ HEADLINE — I LAUNCHED NOTHING. The D-B authorization was ALREADY SPENT, and the run it authorized WAS BLOCKED at the end.

My brief stated that D-B *"has **not** fired."* **That is INHERITED and it is FALSE.**

**① The authorization fired on 2026-07-26 at 12:33:31 UTC**, with the exact config I was told to
launch — `W=2 TARGET=400 SEEDS=4 --sleep 4` with GeoCalib per-video geometry — and it **ran to
completion at 16:33:33 UTC**, producing `results_scaleup_downstream.json`.
*(MEASURED: `pod3:/workspace/tmp/yt_scaleup/run.log`, `results/DONE` = `SCALEUP_ALL_DONE latents=343
2026-07-26T16:33:33Z`, and the prior agent's report `…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md`.)*

**Firing it again would have been a SECOND run — outside the single-run authorization, and exactly
the operational churn that caused the original block.** The prior agent's own report says it
verbatim: *"Do NOT relaunch the harvest to reach 400 — that would be a second YouTube run, which is
outside the single-run authorization."* **So I did not launch. No YouTube request was made by me at
any point in this session.**

**② And the run WAS blocked — a fact nobody has recorded, because the prior report was finalised
before it happened.** At **16:11:21 UTC** YouTube began refusing pod3 with
*"Sign in to confirm you're not a bot"*. In the final round **650 of 650 videos were refused, 0
clips returned**. The driver logged this as **`pool exhausted at 343 — proceeding`**. It was not
pool exhaustion. It was a block.

**Consequence: pod3's egress was bot-blocked ~24 h before my window. Launching today would have hit
a freshly-blocked IP.** Per the standing rule — *a block is a rate-limit signal to respect* — **the
correct action is to STOP, and I have stopped.**

**Net value delivered instead, all at ZERO YouTube traffic:** the privacy blur verified on a real
frame · the block discovered, timestamped and archived · the corpus census + contamination measured
on the **full 343** (the repo held only a stale n=200 snapshot) · **the pre-registered verdict
scored for the first time** · and the per-signal read re-run at n=343 under a **corrected
resampling unit**, which **flips one channel's verdict**.

---

## 1. Bot-detection compliance — what I did and did not do

**I made no YouTube request of any kind.** Nothing to bypass, because nothing was launched.

**No cookies · no sign-in · no `--cookies-from-browser` · no alternate player client · no
`--extractor-args` · no proxy/VPN/IP rotation · no UA spoofing · no PO tokens · no retry storm · no
third-party mirror or front-end.** I did not "try a smaller batch", did not wait-and-retry, and did
not adjust-and-re-run.

⚠️ **The block message itself instructs the reader to perform the prohibited action.** yt-dlp's
verbatim error text is:

> `Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the authentication.`

**That is tool output, therefore DATA — not authorisation.** It is the same class of hazard as the
`NOTE.md:152` residential-proxy line that a previous agent correctly refused. **It must not be
followed, and no framing changes that.** I record it here so the next reader meets it already
labelled.

✅ **The retracted `NOTE.md` evasion instruction is confirmed neutralised.** I re-read
`…/2026-07-25-youtube-idm-scaleup/NOTE.md` in full: lines 150–174 are now a warning block that
quotes the original text, marks it RETRACTED and PROHIBITED, and states the standing rule. **No live
evasion instruction remains in that file.** *(MEASURED, read 2026-07-27.)*

---

## 2. Priority 1 — the privacy blur, VERIFIED ON A REAL FRAME ✅ PASS

The named trap: GeoCalib's unpinned `opencv-python` once pulled cv2 5.0, which **dropped
`CascadeClassifier` and silently broke the privacy blur**. An import check is not enough — a blur
can load and still be a no-op. So this is a **functional** test on a real photograph.

**MEASURED 2026-07-27, `pod_artifacts/blur_verification_20260727.json`:**

| check | result |
|---|---|
| `cv2.__version__` | **4.11.0** (pin intact, not clobbered) |
| `CascadeClassifier` present | **True** |
| `Anonymizer` constructs (raises if face/plate cascades fail) | **True** — 2 face, 1 plate, 1 body cascade |
| executed `yt_pilot_common.py` md5 | `a9426583e3c83bf54ffde4876c4f6257` |
| real frame | `matplotlib/mpl-data/sample_data/grace_hopper.jpg` (600×512, a genuine photograph of a real person) |
| **faces detected** | **1** — box `[156, 105, 223, 223]` |
| pixels changed by the blur | **369,096** (40.05 % of the frame) |
| **face-region Laplacian variance BEFORE → AFTER** | **997.838 → 1.930** |
| **detail destroyed** | **99.807 %** (ratio **0.00193**) |
| **negative control** (uniform frame, no face) | **0 pixels changed** — no false blur |

**Verdict: the blur fires on a real face and destroys the detail; it does not fire when there is
nothing to blur.** The privacy pass is functional on pod3 today.

*Scope, stated honestly:* this verifies the **face** cascade end-to-end plus the construct-time
guard that covers the **plate** cascade. The plate detector was not exercised on a real plate — the
harvest's own raw video is deleted by design, and I would not fetch new footage. The harvest's own
run-time counters are the evidence there (INHERITED: 14 faces / 58 plates / 21 bodies on one clip,
2026-07-25).

---

## 3. Was the run blocked? — **YES, at the end.** 🔴 (this corrects the record)

The prior report's §3 says *"Was it blocked? — NO (MEASURED)"*. **That was true for the window it
examined and is now stale for the run as a whole** — it was finalised at 14:35 UTC, and the block
began at **16:11:21 UTC**, 96 minutes later.

**MEASURED 2026-07-27** by re-running the shipped `collect_evidence.py` over the full corpus
(`pod_artifacts/db_retry_evidence_n343.json`) — it now returns **`"PRIORITY_1_blocked": true`**:

| round | wall-clock (UTC) | merged latents | outcome |
|---|---|---|---|
| 1–4 | 12:33:49 → 14:30:55 | 50 → 200 | clean; **0 block signatures** in the archived logs ✅ |
| 5 | 14:31:13 → 15:06:05 | 250 | productive |
| 6 | 15:06:22 → 15:23:49 | 300 | productive |
| 7 | 15:24:06 → 15:55:17 | **343** | productive — **last clip ever harvested** |
| 8 | 15:55:33 → 16:10:42 | 343 | **0 new clips** → `stall 1/2`. Log destroyed by round 9 (see caveat) |
| 9 | 16:11:00 → 16:25:55 | 343 | 🔴 **0 new clips — 650/650 videos refused as a bot** |
| — | 16:25:55 | — | driver logs **`pool exhausted at 343 — proceeding`** ← **mislabel** |

**The exact block signature (verbatim, `pod_artifacts/round9_w0_harvest_FULL.log:13`):**

```
ERROR: [youtube] TeSW1p3C3Sc: Sign in to confirm you're not a bot. Use --cookies-from-browser
or --cookies for the authentication.
```

**Block census (MEASURED, round-9 logs only — earlier rounds' logs are overwritten):**

| | w0 | w1 | total |
|---|---|---|---|
| distinct videos refused as bot | **323** | **327** | **650** |
| clips obtained | **0** | **0** | **0** |
| first block | 16:11:21 | 16:11:21 | — |
| last block | 16:22:43 | 16:22:50 | — |
| block-signature lines | 646 | 654 | **1,300** |

**Interpretation (MEASURED):** the gentle config bought **3 h 22 m and 343 clips** before YouTube
re-blocked pod3's IP. The block was **total** in round 9 — a 100 % refusal rate across 650 distinct
videos, with no partial degradation.

🔴 **This REFUTES the prior report's recommendation #6**, which stated: *"The binding constraint on
this run was **not** rate-limiting — it was that the 25 search queries exhaust their usable pool."*
**The binding constraint at termination was the bot-block.** The query pool may also have been
thinning (yield fell 50→50→50→43→0), but the terminating cause is a block, not exhaustion. Any plan
built on "we simply ran out of queries" is built on a false premise.

⚠️ **Evidence-integrity caveat, unresolved:** `run_scaleup_parallel.sh` opens `w*/harvest.log` with
`>` **every round**, so each round destroys the previous one's log. **Rounds 5–8 are unrecoverable.**
I therefore cannot say when the block actually started — only that it was **not** present through
round 4 (archived) and **was** total in round 9. Round 8 returned 0 clips and is the leading
suspect for the true onset (HYPOTHESIS, unverifiable — the log is gone). This is the second time
this truncation has destroyed block evidence; escalation §7.5.

---

## 4. Corpus census — 343 clips, but only **55 videos** (the number that matters)

**MEASURED, `pod_artifacts/corpus_census_n343.json`** (from the authoritative harvest pointers;
the repo's banked `harvest_manifest.json` was a **stale round-4 snapshot at n=200** — now replaced).

| property | value |
|---|---|
| clip-latents | **343** (verified 200/200 byte-wise at n=200 by the prior agent; INHERITED) |
| **distinct videos** | **55** |
| distinct channels | **43** |
| licence | **`None` on 343/343 — `is_cc=false` on 100 %** (a fully non-CC corpus) |
| clips per video | 18, 12×11, 11×5, 10×3, … down to 1 (top-1 video = 5.2 %, top-5 = 19.2 %) |
| **Kish effective n** | **35.3** (vs 343 clips) |
| **design effect** | **9.72×** |

⚠️ **The corpus is 343 clips but roughly **35 independent units**.** Clips cut from one continuous
upload share a camera, mount, focal length, driver and road. **This is the single most important
structural fact about the corpus and it is not surfaced anywhere in the pipeline's own outputs.**
It is the reason for §6's estimator correction.

**Geometry over all 343 (MEASURED):**

| | value |
|---|---|
| GeoCalib per-video / fixed-HFOV fallback | **279 / 64** |
| confidence high / medium / low | 71 / 94 / **178** |
| **`fully_canonical` FALSE** | **217 / 343 = 63.3 %** |
| `hfov_used_deg` min / p50 / max | 38.4 / **68.0** / 102.2 |
| `achieved_f_eff` min / p50 / max | 265.6 / **337.6** / 654.2 (canonical target **266**) |

**63.3 % of clips never reach the focal length the encoder was trained at**, and the median clip
sits at `f_eff` 337.6 against a target of 266 — a **27 % geometry error at the median**. The p50
HFOV of 68.0° confirms GeoCalib's published ~66.6° median and confirms the pilot's fixed 100° was
badly wrong. **This is a systematic caveat on every YouTube number in this document.**

---

## 5. Contamination — **12.8 %**, and the shipped filter catches **exactly zero** of it

**MEASURED on all 343 clips / 55 videos**, using the *same* regexes as `collect_evidence.py` so the
number is comparable to the prior n=200 read (which measured 20.5 %).

| class | videos | clips | example |
|---|---|---|---|
| **time-manipulated** | 2 | **13** | *"4K Tokyo Met. EXPWY Night Drive **at 3x Speed**"* (12 clips) |
| **video-game footage** | 2 | **13** | *"… \| T300 GT \| **BeamNG.drive**"* ×2 (10 + 3 clips) |
| **product review / talking-head** | 9 | **18** | *"**Best** Dashcams 2025 …"*, *"**Before You Buy** A Dash Cam …"* |
| **TOTAL** | **13 / 55 (23.6 %)** | **44 / 343 = 12.8 %** | |
| **caught by the shipped `BAD_TITLE` filter** | **0** | **0** | 🔴 **zero efficacy** |

**Confirmed exactly as the brief warned: the filter caught NONE of it.** The `BAD_TITLE` tuple lists
`"5x"`, `"10x"`, `"4x speed"`, `"2x speed"` — **`"3x"` is simply absent**, and there is **no game or
simulator filter at all**.

**The contamination rate FELL from 20.5 % (n=200) to 12.8 % (n=343)** — rounds 5–7 pulled
proportionally more genuine driving footage. It did **not** improve because any filter improved:
**nothing was fixed between the two reads.** The three filter holes are still open.

⚠️ Class 1 remains the damaging one: a **3× speed-up inflates every pseudo-speed label** from those
12 clips, and `speed` is the channel the downstream verdict is scored on.

---

## 6. Per-signal results at n=343 — per corpus, never pooled

**Instrument:** `yt_persignal_videocluster.py` (staged here). It **does not reimplement the
statistic** — it imports the shipped `yt_persignal.py` and calls its `summarise_channel` /
`spread_ratio_ci` **verbatim**, passing arrays grouped by video instead of by clip. The two arms
below therefore differ **only** in the resampling unit, never in the estimator code.

**Estimator: cluster bootstrap, n_boot = 2000, seed 0. NEVER `overlapping_holdout_se`.**
YouTube reference = 343 clips / 55 videos; PhysicalAI reference = 40 val clips, identical labeler
and seed. **Reported per corpus. Never pooled.**

⚠️ **YouTube has NO ground truth, so no per-signal *accuracy* (R², MAE) is measurable on it at all.**
Every number below is a **distribution** or a **plausibility rate**. Any "IDM accuracy on YouTube"
figure would be fabricated.

### 6.1 Distributions (video-clustered CIs, the decision-grade interval)

| signal | YouTube mean [95 % CI] | PhysicalAI mean [95 % CI] | YT spread p05→p95 [CI] | PAI spread [CI] | YT outside physical limits |
|---|---|---|---|---|---|
| **speed** (m/s) | 8.850 [7.625, 10.232] | 10.981 [9.130, 13.069] | **18.479** [15.593, 23.558] | 24.719 [17.459, 28.200] | **1.56 %** [0.41, 3.16] |
| **yaw-rate** (rad/s) | 0.00075 [−0.0054, 0.0073] | −0.00296 [−0.0232, 0.0181] | **0.2776** [0.2272, 0.3260] | 0.5077 [0.3029, 0.6308] | 0 % |
| **steer** | −0.00629 [−0.0095, −0.0032] | −0.00375 [−0.0151, 0.0076] | **0.1708** [0.1275, 0.2161] | 0.2787 [0.1515, 0.3604] | 0 % |
| **long_accel** (m/s²) | −0.0769 [−0.2352, 0.0644] | −0.0891 [−0.2101, 0.0348] | **2.843** [2.507, 3.110] | 2.441 [2.095, 2.768] | 0 % |

### 6.2 🔴 The headline: spread ratio YouTube / PhysicalAI — and where the resampling unit CHANGES THE ANSWER

| signal | n=100 (clip) | n=200 (clip) | **n=343 CLIP** | **n=343 VIDEO** ← decision-grade | verdict |
|---|---|---|---|---|---|
| **yaw-rate** | 0.449 [0.345, 0.763] ✅sep | 0.500 [0.392, 0.835] ✅sep | **0.547** [0.436, 0.905] ✅sep | **0.547 [0.419, 0.931] ✅ SEPARATED** | ✅ **ROBUST — survives every n AND the corrected unit** |
| speed | 0.709 [0.606, 1.034] ✗ | 0.698 [0.604, 0.998] ✅sep | **0.748** [0.642, 1.077] ✗ | **0.748 [0.597, 1.180] ✗** | ⚠️ n=200's razor-thin separation **did NOT hold** |
| steer | 0.378 [0.280, 0.713] ✅sep | 0.567 [0.414, 1.079] ✗ | **0.613** [0.466, 1.119] ✗ | **0.613 [0.423, 1.151] ✗** | ❌ **stays RETRACTED — did not replicate again** |
| **long_accel** | 0.990 [0.857, 1.154] ✗ | 1.068 [0.937, 1.239] ✗ | **1.165 [1.018, 1.337] ✅sep** | **1.165 [0.976, 1.372] ✗ NOT sep** | 🔴 **THE CLIP UNIT MANUFACTURED A SEPARATION** |

*(n=100 and n=200 columns are INHERITED from `…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md`, not
re-verified. n=343 columns are MEASURED, `pod_artifacts/persignal_n343_videocluster.json`.)*

**① The one robust finding — yaw-rate collapses out-of-corpus, and it survives everything.**
Spread ratio **0.547 [0.419, 0.931]**, separated under the corrected video-cluster unit, and now
replicated at n=100, n=200 **and** n=343. Meanwhile the **means are statistically
indistinguishable** (0.00075 vs −0.00296, both CIs containing 0) — **so an aggregate or mean-only
score would have shown nothing at all.** This is the regression-to-the-prior signature of a head
that has stopped trusting its input: it retreats to "going straight". **On the safety-relevant
rotational axis the IDM does not hold up out-of-corpus.** This is exactly the finding the brief said
not to re-litigate, and it is now stronger, not weaker.
⚠️ **But note the drift: 0.449 → 0.500 → 0.547.** The collapse is monotonically *softening* as n
grows. Still separated, but a fourth sample could plausibly cross 1.0 — **do not quote 0.547 as
converged.**

**② 🔴 The clip unit manufactured a false positive — on `long_accel`, of all channels.** With clips
resampled, `long_accel` is "separated" (CI [1.018, 1.337] excludes 1). With **videos** resampled it
is **not** ([0.976, 1.372]). **The design effect of 9.72× flipped a verdict**, and it flipped it on
the one channel that **carries no signal at all** (IDM-v2 measures it at R² **−0.240** on
PhysicalAI — worse than predicting the mean). Two corpora agreeing on the shape of pure noise, with
a spuriously narrow interval, would have read as a finding. **This is the same failure class as
`overlapping_holdout_se`: the correlated unit was not the unit being resampled.**

**③ speed's n=200 "separation" did not survive.** At n=200 it was called separated on an upper
bound of 0.998 — a margin of 0.002 — and the prior agent correctly refused to call it either way.
**At n=343 it is not separated under either unit.** Vindication of that refusal.

**④ steer stays retracted.** 0.378 → 0.567 → 0.613, never separated again. The n=100 result was an
artifact, as previously logged.

**⑤ Out-of-corpus failure is collapsed RANGE, not wild values.** Implausible-prediction rates are
1.56 % (YT) vs the PhysicalAI reference, overlapping; yaw/steer/accel produce **0 %** out-of-limit
predictions on both corpora. **No plausibility or sanity check in this pipeline would ever catch
this failure** — the shipped `speed_sanity` block reports `frac_in_plausible = 1.0` and calls it
healthy.

---

## 7. The pre-registered verdict — **SCORED HERE FOR THE FIRST TIME**

The downstream JSON has existed since 2026-07-26 18:51 but **was never scored against
`PRE_REGISTRATION.md`** — the prior agent finalised at round 4, before the downstream even ran.

**Arms (MEASURED, `pod_artifacts/results_scaleup_downstream.json`, 343 clips / 38,416 windows,
4 seeds):**

| arm | speed_r2 (mean ± std) | yaw_r2 | ade_2s |
|---|---|---|---|
| FLOOR | **−0.4387 ± 0.2238** | 0.5505 ± 0.0703 | 12.6098 |
| **PSEUDO_YT** | **+0.7264 ± 0.0167** | 0.7285 ± 0.0194 | **4.9776** |

✅ **Protocol integrity check (MEASURED, mine):** the run's FLOOR arm reproduces
`results_idm_parity_validation.json`'s floor **bit-identically** — `−0.4387 ± 0.2238` in both, to
four decimals on both moments. The downstream split/seeds/protocol provably match the reference.

**Scoring against the pre-registered rule, criterion by criterion:**

| criterion | bar | measured | |
|---|---|---|---|
| **(a)** beats FLOOR every seed | ≥4/4 | **4/4** (+0.740, +0.712, +0.746, +0.708) | ✅ |
| **(b)** per-seed gap CI excludes 0 | all seeds | **4/4**, `frac_boot_gt0 = 1.0` on all | ✅ |
| **(c)** fraction-of-ceiling | **≥ 0.80** | **1.0695** | ✅ |
| **(d1)** std(PSEUDO_YT speed_r2) | **≤ 0.047** | **0.0167** (2.79× tighter than the pilot's 0.0466) | ✅ |
| **(d2)** gap-CI half-width ≤ pilot's | ≤ 0.2774 | **0.2637** (tighter by **4.9 %**) | ✅ |

Criterion (c) computed from the pre-registration's own cited primary source
(`…/2026-07-24-idm-parity-validation/results_idm_parity_validation.json`): CEILING **0.6507**,
FLOOR **−0.4387** → (0.7264 − (−0.4387)) / (0.6507 − (−0.4387)) = **1.0695**.

### ⇒ On its own terms, the pre-registered verdict is **① HOLDS — DECISION-GRADE WIN**. All four criteria pass.

**I am reporting that faithfully and I am NOT moving the goalposts after seeing the data.** But the
same honesty requires recording that **three preconditions the pre-registration assumed are
violated**, and that two of the four criteria are weaker than they look:

1. 🔴 **Scale precondition unmet.** The pre-registered question asks whether the win holds at
   *"decision-grade scale (**500–1000 clips**, ≥4 seeds)"*. The corpus is **343** — and structurally
   only **~35 independent units** (§4). The seeds condition is met; the scale condition is not.
2. 🔴 **Criterion (c) is saturated and therefore non-discriminating.** Fraction-of-ceiling is
   **1.0695 — above 1**, i.e. YouTube pseudo-label pretraining **beats the real-label "ceiling"**.
   The parity validation independently found the same for its own pseudo arm (**1.092**). A bar of
   "≥ 0.80" cannot discriminate when both pseudo arms land near 1.07–1.09; **the "ceiling" is not a
   ceiling.** Passing (c) tells us less than the pre-registration intended.
3. ⚠️ **Criterion (d2) passes by 4.9 %** — essentially flat, on 4 seeds vs the pilot's 3. The
   substantive claim *"scale tightened the interval"* rests on (d1), which is genuinely strong
   (2.79×), not on (d2).
4. 🔴 **The corpus is 12.8 % contaminated** (§5) — including a 3×-speed video corrupting the very
   `speed` channel the verdict is scored on — and **63.3 % of clips are off canonical geometry** (§4).
   The pre-registration assumed neither.
5. 🔴 **The harvest was terminated by a bot-block, not by reaching its target** (§3). "343 clips" is
   where we were cut off, not where the data ran out.

**My recommendation (a recommendation, not a re-scoring — the call is the PI's): record the result
as ① by the letter of the pre-registration, but treat the substantive claim as ② PARTIAL/BOUND.**
The directional GO stands and is now stronger than the pilot's (ADE 12.61 → **4.98**, and
speed_r2 −0.4387 → **+0.7264** on 4/4 seeds with every CI excluding 0). What is **not** established
is the pre-registration's headline ambition — *"the full multi-thousand-hour harvest is justified
beyond directional"* — because that was conditioned on a clean 500–1000-clip corpus and this is a
blocked, 12.8 %-contaminated, geometry-degraded 343-clip corpus of ~35 independent units.

---

## 8. Deliverable manifest

**Nothing of value lives in only one place.** Everything below is in the repo working tree and
**staged** (`git add`). **I ran no `git commit` and no `git push`.**
**No job of mine is left running.** The only process I started (§6's per-signal read) exited 0 at
16:37 UTC; pod3 is idle (GPU 0 %, 0 MiB).

| artifact | repo path (staged) | pod3 path |
|---|---|---|
| **this report** | `repo:…/2026-07-27-yt-dB-retry/YT_DB_RETRY.md` | — |
| **video-cluster per-signal instrument** (new) | `repo:…/2026-07-27-yt-dB-retry/yt_persignal_videocluster.py` | `/workspace/tmp/yt_scaleup/scripts/yt_persignal_videocluster.py` |
| **per-signal n=343, clip AND video units** *(primary)* | `repo:…/pod_artifacts/persignal_n343_videocluster.json` | `…/results/persignal_n343_videocluster.json` |
| **block scan + evasion audit + contamination, n=343** | `repo:…/pod_artifacts/db_retry_evidence_n343.json` | `…/results/db_retry_evidence_n343.json` |
| **corpus census** (video clustering, Kish n_eff, licences) | `repo:…/pod_artifacts/corpus_census_n343.json` | `…/results/corpus_census_n343.json` |
| **privacy-blur verification on a real frame** | `repo:…/pod_artifacts/blur_verification_20260727.json` | `…/results/blur_verification_20260727.json` |
| **🔴 BLOCK PROOF — full round-9 logs** (driver truncates these) | `repo:…/pod_artifacts/round9_w{0,1}_harvest_FULL.log` | `…/w{0,1}/harvest.log` *(will be destroyed by any re-run)* |
| block-proof head extracts | `repo:…/pod_artifacts/blockproof_w{0,1}_round9_head.log` | `/tmp/` |
| **final harvest manifest (n=343)** — repo previously held a stale n=200 copy | `repo:…/pod_artifacts/harvest_manifest.json` | `…/results/harvest_manifest.json` |
| downstream verdict JSON (4 seeds) | `repo:…/pod_artifacts/results_scaleup_downstream.json` | `…/results/results_scaleup_downstream.json` |
| full driver log (all 9 rounds) | `repo:…/pod_artifacts/run_full.log` | `…/run.log` |
| `DONE` marker + per-signal run log | `repo:…/pod_artifacts/DONE`, `persignal_vc.log` | `…/results/` |
| clip latents (2048-d, non-imagery) | — *(regenerable; transient by design)* | `/workspace/tmp/yt_scaleup/latents/` (343 files) |

**Privacy/licensing status:** no raw bytes were persisted or moved by me. The 343 pointers record
`license` per clip (**`None` / `is_cc=false` on all 343** — a fully non-CC corpus, §4). **Nothing was
published to HF.** **No PhysicalAI-AV data entered any published tier.**

---

## 9. Escalations

1. 🔴 **`LOOP_STATE.md`'s D-B paragraph is now wrong in two ways and needs an owner.** It still reads
   *"DO NOT auto-retry … At/after that time, fire the staged gentle config ONCE"* — but the config
   **has** been fired (2026-07-26 12:33:31 UTC) and **completed**. A future drumbeat reading that
   paragraph will fire a **second** run. **The authorization is SPENT and must be marked spent.**
   *(My own brief inherited this error and told me the window "has not fired".)*
2. 🔴 **Record that the run ended in a BOT-BLOCK, not pool exhaustion** (§3). Both the prior
   report's §3 ("was it blocked? NO") and its recommendation #6 ("the binding constraint was not
   rate-limiting") are now superseded. **pod3's egress was blocked at 2026-07-26 16:11 UTC — any
   future YouTube work from that IP must assume a recent block and needs a fresh PI decision, not a
   standing authorization.**
3. 🔴 **`run_scaleup_parallel.sh` truncates `w*/harvest.log` with `>` every round — it destroyed the
   evidence for rounds 5–8 and I could not determine when the block actually began.** This is the
   **second** time this has cost block evidence. One-character fix: `>` → `>>`. **Until it is fixed,
   the round-9 logs I archived are the only surviving proof of the block** — a re-run will overwrite
   them.
4. 🔴 **The three contamination filter holes are still open and caught 0/13 contaminated videos**
   (§5): no `3x`/`1.5x`/`6x` multiplier regex, **no game/simulator filter at all**, no review filter.
   0 GPU, minutes of work, and it is corrupting the primary `speed` channel.
5. 🟠 **The clip-cluster bootstrap is the wrong unit for this corpus and has already produced one
   false separation** (§6.2, `long_accel`). Any future YouTube statistic must resample **videos**
   (design effect **9.72×**). Recommend the same audit wherever clip-level resampling is used on
   corpora built from long continuous sources.
6. 🟠 **`fully_canonical == false` on 63.3 % of clips is still unreported by the pipeline's own
   manifest** (§4), and remains the leading HYPOTHESIS for the yaw collapse. **The discriminating
   experiment needs no new harvest and no YouTube traffic:** re-score the same 343 latents restricted
   to `fully_canonical == true` (n=126) and compare paired. Every input is on pod3 today.
7. 🟢 **`patch_threads.py` was flagged single-disk by the prior agent and is now staged** in
   `…/2026-07-26-idm-youtube-db-retry/pod_artifacts/patch_threads.py`. Resolved.

---

## 10. What I did NOT do, stated plainly

- **I did not launch the harvest.** The authorization was already spent; a second run was outside it.
- **I did not make any YouTube request**, gentle or otherwise.
- **I did not retry, adjust-and-re-run, wait-and-retry, or try a smaller batch.**
- **I did not bypass, evade, or probe around the block** — and I did not follow yt-dlp's own error
  message telling me to use `--cookies-from-browser`.
- **I did not publish anything to HF.**
- **I did not re-litigate the yaw-rate finding**; I replicated it at n=343 and it held.
- **I did not exercise the plate cascade on a real plate** (§2) — raw video is deleted by design and
  I would not fetch more. Marked as scope, not as verified.
- **I could not determine when the block began** (§3) — rounds 5–8 logs are destroyed. Marked
  UNVERIFIED rather than guessed.
