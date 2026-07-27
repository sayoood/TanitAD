# D-B — the ONE authorized gentle YouTube retry

**Date:** 2026-07-26 · **Agent:** `db-youtube-retry` · **Pod:** `tanitad-pod3` (A40; pod1/pod2/eval untouched)
**Status at finalisation (2026-07-26 14:35 UTC):** the single authorized run is **still
harvesting round 5**. Everything below is MEASURED on the 200 clips banked through round 4.

> **① Priority 1 — the run was NOT blocked.** YouTube served pod3 normally from 12:33 UTC.
> Discovery, metadata and downloads all succeeded; **0 bot-block signatures** across every
> archived log. The 2026-07-25 hard block cleared during the cooldown, and the gentle
> config did not re-trip it over 2 hours of harvesting. **No evasion of any kind was used
> — and one instruction in the program's own handoff note telling the next agent to use a
> residential proxy was refused and is escalated (§1).**
>
> **② Priority 2 — yield is 200/400 (50 %) and NOT complete.** All 200 verified byte-wise,
> shape-wise and duration-wise: 200/200 load, 0 truncated, 0 short, 24.8 s each. Reported
> as 200/400, not as "the pipeline works".
>
> **③ Priority 3 — the per-signal answer to Sayed: only `yaw_rate` gives a robust
> out-of-corpus finding, and it is bad.** Its predicted range collapses to **half** the
> in-corpus range (spread ratio **0.500 [0.392, 0.835]**, separated at n=100 *and* n=200)
> **while its mean is indistinguishable** — an aggregate score would have shown nothing.
> **`steer`'s collapse did NOT replicate from n=100 to n=200 and is retracted here;**
> `speed` sits on the significance boundary; `long_accel` transfers "perfectly" because it
> carries no signal at all (R² −0.240 in-corpus).
>
> **④ The corpus is 20.5 % contaminated** by three classes the filters miss entirely — a
> 3×-speed video, **two** BeamNG video-game videos, and seven dashcam *product reviews*.
> **Removing them does not recover the yaw collapse**, so it is a domain property, not a
> data-cleanliness artifact.

---

## 1. Bot-detection compliance — what I did and did not do

**I did not bypass, evade, or probe around any block.** Concretely:

- **No cookies, no `--cookies-from-browser`, no sign-in, no account.**
- **No alternate player clients** (`player_client=android/ios/tv/web_embedded`), no
  `--extractor-args`.
- **No proxy, no VPN, no IP rotation, no User-Agent spoofing, no PO tokens, no
  `--force-ipv4`, no third-party mirror/front-end.**
- **No retry storm.** The one authorized run was launched **once**. Videos that failed
  were marked `dl_fail` and abandoned, never retried.

**Pre-run audit (MEASURED).** I read the runner before executing it and grepped both the
repo copy and the **on-pod copy that actually runs** for every evasion token:

```
grep -niE 'cookie|player_client|extractor_args|proxy|user.?agent|http_headers|
           po_token|force.?ipv4|source_address|geo_bypass|innertube|invidious|piped'
```

→ **zero hits in every file that touches YouTube**: `harvest_scaleup.py`,
`run_scaleup_parallel.sh`, `yt_pilot_common.py`. Being exact rather than rounding to
"zero everywhere": the automated audit reports **one `proxy` hit each in
`pseudo_label.py` and `run_youtube_pilot_downstream.py`** — both are the English word in
a comment ("identical recipe to the parity/**proxy** runs"), neither is an option, and
neither file makes a network request. Full per-file hit list with line context is in
`db_retry_evidence.json → evasion_audit_of_executed_scripts`.

The complete yt-dlp option set actually used is `quiet, no_warnings, skip_download,
extract_flat, noplaylist, noprogress, format, max_filesize, outtmpl, sleep_interval,
max_sleep_interval, sleep_interval_requests`. Nothing in it disguises, authenticates, or
reroutes the client; `sleep_interval*` only *lowers* our request rate.

The audit was run against the **on-pod** copies, and `harvest_scaleup.py`'s md5 is
identical on pod and in the repo (`085cd11219a3f4b34c7fc4f7796b5a66`) — so the audited
file is provably the executed file.

⚠️ **One instruction found in program data was NOT followed.** The prior agent's handoff
note (`…/2026-07-25-youtube-idm-scaleup/NOTE.md:152`) proposes, verbatim:

> "use a different egress (residential proxy / different pod IP / a machine YouTube
> hasn't flagged)"

**I did not do this and it must not be done.** A residential proxy is precisely the
bot-detection evasion the standing rule forbids. Flagged here so it is not quietly
inherited by the next agent. **Recommend striking that line from the NOTE.**

---

## 2. Exact configuration and command

Pre-registered config, run **exactly once**, unmodified: `W=2 TARGET=400 SEEDS=4 --sleep 4`
with **GeoCalib per-video geometry**.

```bash
ssh tanitad-pod3
PYTHONPATH=/workspace/TanitAD/stack W=2 TARGET=400 SEEDS=4 SLEEP=4 \
  setsid nohup bash /workspace/tmp/yt_scaleup/scripts/run_scaleup_parallel.sh \
  > /workspace/tmp/yt_scaleup/run.log 2>&1 < /dev/null &
```

Launched **2026-07-26 12:33:31 UTC** (14:33 Europe/Berlin). Authorization window opened
2026-07-26 12:00 UTC; I verified the pod clock read **12:29 UTC** before launching, so
the run started **inside** the window (not early).

**One change to the driver was required and is staged.** `run_scaleup_parallel.sh` had no
way to pass `--sleep`; the prior NOTE said to "add `--sleep 4` to the harvest call" by
hand. I made it a first-class, **gentle-by-default** knob instead:

| line | change |
|---|---|
| config block | `SLEEP=${SLEEP:-4}` — default is now 4 s, previously effectively 0 |
| harvest call | `--sleep "$SLEEP"` added |
| start banner | logs `sleep=4s` so the pacing is visible in `run.log` |

The repo copy was also **newer than the pod copy** (an inline-GeoCalib comment fix that
never shipped); I synced repo → pod so the audited file is the executed file.
`harvest_scaleup.py` was **not modified** — repo and pod md5 both `085cd11219a3f4b34c7fc4f7796b5a66`.

**Preflight (MEASURED, pod3, 2026-07-26 12:32 UTC):** baseline merged latents **0** (clean
yield accounting, no leftovers); real `dd` 8 GB write **530 MB/s** (`df` is invalid on
MooseFS); GPU **0 MiB / 46068 MiB**, no live jobs; `yt_dlp 2026.07.04`; **`cv2 4.11.0` with
`CascadeClassifier` present** — i.e. the privacy Haar blur is functional, not silently
broken by the 2026-07-25 opencv-5 dependency clobber.

---

## 3. Was it blocked? — NO (MEASURED)

| probe | result | evidence |
|---|---|---|
| search discovery (`ytsearch`) | **OK** — 150 / 158 candidates for w0 / w1 in ~7 s | `w0/harvest.log`, `w1/harvest.log` 12:33:56–12:34:01 |
| metadata `extract_info` | **OK** on every candidate | per-video log lines |
| media download | **OK** | clips produced from 12:37 onward |
| bot-block signature scan (all archived logs, 2 h of harvesting) | **0 hits** for `sign in to confirm` / `not a bot` / `HTTP 429` / `too many requests` | `db_retry_evidence.json`, `logs_archive/` |

**Two per-video `HTTP Error 403: Forbidden` events occurred** — `jum8PGur1PU` (12:55:42,
w1) and `g8L5PJKTyRk` (13:37:44, w0). Neither is a bot-block: videos immediately before
and after each downloaded normally from the same worker and the same IP, the two are
81 minutes apart, and both were dashcam-*review* uploads of the kind that commonly carry
embed/region restrictions. The harvester recorded `dl_fail` and moved on — **neither was
retried**, which is the correct behaviour. **2 restricted videos out of 32 harvested is
an ordinary rate and shows no throttling trend.**

⚠️ **Evidence-integrity note:** the driver opens `w*/harvest.log` with `>` at the start of
every round, so each round destroys the previous round's log. The first 403 was already
gone before I started archiving; it is quoted here from the live session transcript, and
the second is preserved in `logs_archive/`. Archiving was added mid-run (§9 item 5).

---

## 4. Yield — 200/400 (50 %), every clip byte-verified

*Presence is not completeness.* `verify_yield.py` does not count files: it byte-sizes
every latent, `torch.load`s it, checks tensor shape and finite-ness, and joins each to its
pointer to confirm frame count → duration.

**MEASURED at 2026-07-26 14:31 UTC, after round 4 —
`pod_artifacts/yield_verification.json`:**

| check | result |
|---|---|
| merged clip-latents | **200** |
| **verified OK** (byte-sized, `torch.load`s, shape-checked, finite, non-empty) | **200 / 200** |
| bad / zero-byte / non-finite / truncated | **0** |
| total latent bytes | 204,782,800 (~1.024 MB/clip, uniform) |
| pointers written | 200, **0 duplicates** |
| full-length clips (248 frames = **24.8 s**) | **200 / 200**, **0 short** |
| **YIELD vs TARGET** | **200 / 400 = 50 %** |

**The run had not reached TARGET when this report was finalised — it was still
harvesting round 5.** So the honest headline is **200 / 400 at 2 h 00 m elapsed**, not
"the pipeline works". Rounds landed at 50 → 100 → 150 → 200, i.e. **~50 clips per
~30-minute round and flat**, so TARGET would need roughly **4 more rounds / ~2 more
hours** — assuming the candidate pool holds, which §5 shows it does not.

`248` frames is the correct full length (`--clip-frames 250` stacked with `N_STACK=3`
→ 250−3+1 = 248). **My first verifier pass hard-coded 250 and flagged every clip as
short — a false alarm of exactly the kind this check exists to prevent, caught and
corrected before any number here was quoted.**

**Geometry (200 clips):** GeoCalib per-video on **157**, fixed-HFOV fallback on **43**
(low confidence). Confidence: 46 high / 73 medium / 81 low.
⚠️ **`fully_canonical` is FALSE on 120 / 200 clips (60 %)** — the crop never reached the
canonical `f_eff ≈ 266` the encoder was trained at. Measured HFOVs run **40–89°**, nowhere
near the fixed 100° the pilot assumed (consistent with GeoCalib's published ~66.6° median),
so switching to GeoCalib was right — but **the majority of clips still sit off the
canonical geometry.** This is a systematic caveat on every YouTube number here, it is
**not surfaced by the pipeline's own manifest**, and it is the leading suspect for the
yaw collapse in §6.

---

## 5. 🔴 Corpus contamination — three classes the filters miss (MEASURED)

Found by auditing harvested titles against the shipped `BAD_TITLE` reject list.
**41 of 200 clips (20.5 %) came from 10 of 32 videos (31 %) that should never have been
harvested — and the shipped filter caught NONE of them.**

| # | class | videos | clips | why it is contamination | why the filter missed it |
|---|---|---|---|---|---|
| 1 | **time-manipulated** | `Kocl0ZOUejc` *"4K Tokyo Met. EXPWY Night Drive **at 3x Speed**"* | **12** | every pseudo-speed and trajectory label from it is inflated ~3× | `BAD_TITLE` lists `"5x"`, `"10x"`, `"4x speed"`, `"2x speed"` — **`"3x"` is simply absent**, as are `1.5x`, `6x`, `8x`, `x3` |
| 2 | **video-game footage** | `7hvoPyb-mlY`, `lAFzDVzsk24` — both *"… \| T300 GT \| **BeamNG.drive**"* | **13** | synthetic game footage, not real dashcam video — it defeats the entire point of an *out-of-corpus real-video* validation | there is **no simulator/game filter at all** |
| 3 | **product-review / talking-head** | `g8lL-_IwomM`, `gDLib8egfR0`, `T4ppy6MlL4w`, `3iBpLlQlW-8`, `zVg48PoeClk`, `u8k5NjSk1cE`, `ss2xteImDfg` | **16** | studio review video, not continuous forward driving | the `dash cam` queries surface merchandise reviews; the shot-cut filter catches most (0 clips from several) but not all |

Class 1 is the damaging one: it corrupts **speed**, the channel the pipeline labels
`primary` and the one the downstream verdict is scored on. The `BAD_TITLE` tuple is a
hand-written substring list with an obvious hole; a regex over `\b\d+(\.\d+)?\s*x\b`
catches `3x` and every other multiplier.

Class 2 got **worse** as the run progressed — a second BeamNG video appeared in round 4,
so this is a systematic query-pool property, not a one-off.

*(Anonymiser counts corroborate class 3 independently: the review videos blurred 1643 /
472 / 431 faces-plates per clip versus 0–174 on genuine driving footage — talking heads
and product close-ups, not road scenes.)*

⚠️ **One false positive in my own detector, disclosed:** `ss2xteImDfg` *"Ditch Your Old
Dashcam! I Replaced Mine with an **Insta360 X4**"* was flagged `time_manipulated` because
my regex matched `X4` — a product model number, not a speed multiplier. It is genuine
contamination (it is a review video, class 3) but for the wrong reason. A production
filter needs the multiplier pattern anchored to a speed word.

---

## 6. Per-signal IDM results

**Structural finding, reported before any number: the shipped YouTube pipeline cannot
produce the four-signal breakdown Sayed asked for, and never could.**

- `pseudo_label.py` persists only `speed_mean/p05/p95`, `yaw_rate_abs_mean_CAVEAT` and
  `long_disp_2s_mean`. Its own metadata declares `dropped: ["long_accel", "steer"]`.
- `run_youtube_pilot_downstream.py` emits only `{speed_r2, yaw_r2, ade_2s}`.
- **Steering and acceleration are discarded before anything reaches disk.**

**And a harder limit: YouTube has no ground truth, so no per-signal *accuracy* — no R²,
no MAE — is measurable on it at all.** Any "IDM accuracy on YouTube" number would be
fabricated. What is measurable without GT is the per-signal *distribution* and
*physical-plausibility rate*; per-signal accuracy comes only from GT-bearing corpora.

`yt_persignal.py` (staged here) re-runs the same labeler over the persisted YouTube
latents and emits **all four channels** with a **clip-cluster bootstrap** (n_boot 2000,
resampling unit = clip; **never `overlapping_holdout_se`**).

### 6.0 Per-signal read on out-of-corpus video (MEASURED)

Primary artifact: `pod_artifacts/persignal_final_n200.json` (**n = 200 clips / 22,400
windows**). Replication artifact: `persignal_interim.json` (**n = 100 clips**), run
earlier on the first half of the harvest. PhysicalAI reference = **40 val clips / 3,517
windows**, identical labeler and seed. Estimator on **every** interval:
**clip-cluster bootstrap, n_boot = 2000, seed = 0, resampling unit = clip.**
**Reported per corpus. Never pooled.**

| signal | YouTube mean [95 % CI] | PhysicalAI mean [95 % CI] | YouTube spread p05→p95 [CI] | PhysicalAI spread | outside physical limits |
|---|---|---|---|---|---|
| **speed** (m/s) | **8.495** [7.988, 9.058] | 10.981 [9.130, 13.069] | **17.26** [16.32, 18.13] | 24.72 | **2.01 %** [0.92, 3.49] |
| **yaw-rate** (rad/s) | **−0.00042** [−0.0060, 0.0048] | −0.00296 [−0.0232, 0.0181] | **0.2539** [0.2262, 0.2878] | 0.5077 | 0 % |
| **steer** | **−0.00674** [−0.0101, −0.0037] | −0.00375 [−0.0151, 0.0076] | **0.1581** [0.1338, 0.1840] | 0.2787 | 0 % |
| **long_accel** (m/s²) | **−0.0143** [−0.0733, 0.0429] | −0.0891 [−0.2101, 0.0348] | **2.608** [2.468, 2.746] | 2.441 | 0 % |

### 6.0.1 ⚠️ The n=100 → n=200 replication, and what did NOT survive it

I ran the same statistic at n=100 and again at n=200. **Two of the four channels changed
their verdict.** I am reporting this rather than quoting the n=100 table that told a
tidier story.

**These two samples are NESTED, not independent** — the n=200 set contains the n=100 set
plus rounds 3–4. That makes the instability *more* concerning, not less: adding 100 clips
to an existing 100 was enough to move `steer`'s point estimate by 0.19 and flip its
separation. A truly independent replication would very likely move it further. **Treat
every per-signal number here as provisional at n=200 except `yaw_rate`.**

**Spread ratio YouTube / PhysicalAI** — independent clip-cluster bootstrap of *both*
corpora, statistic `(p95−p05)_YT / (p95−p05)_PAI`:

| signal | n=100 ratio [CI] | sep? | **n=200 ratio [CI]** | **sep?** | verdict |
|---|---|---|---|---|---|
| **yaw-rate** | 0.449 [0.345, 0.763] | yes | **0.500** [0.392, 0.835] | **yes** | ✅ **ROBUST — separated at both n** |
| speed | 0.709 [0.606, 1.034] | no | **0.698** [0.604, 0.998] | **yes** | ⚠️ point estimate stable (~0.70), separation **borderline** — CI touches 1 from either side |
| steer | 0.378 [0.280, 0.713] | **yes** | **0.567** [0.414, 1.079] | **no** | ❌ **DID NOT REPLICATE** — point estimate moved 0.38 → 0.57 |
| long_accel | 0.990 [0.857, 1.154] | no | **1.068** [0.937, 1.239] | no | ✅ robustly NOT different |

**① The one robust finding: yaw-rate collapses on out-of-corpus video.** Its predicted
spread is **half** the in-corpus spread (0.500 [0.392, 0.835]), separated at **both**
sample sizes. The *means* are statistically indistinguishable between corpora
(−0.00042 vs −0.00296, CIs overlapping and both containing 0) — **so a mean-only or an
aggregate score would have shown nothing at all.** This is the regression-to-the-mean
signature of a head that has stopped trusting its input: it retreats toward the training
prior and predicts "going straight". **On the safety-relevant rotational axis, the IDM
does not hold up out-of-corpus.**

**② Steer's collapse did NOT replicate — retracted.** At n=100 I measured 0.378
[0.280, 0.713], separated. At n=200 it is 0.567 [0.414, 1.079], **not** separated. Had
I stopped at n=100 I would have reported "yaw AND steer collapse" as a finding. **It was
an n=100 artifact.** *(Root-cause class: a CI-separated result at small n treated as
established before replication. The replication is what caught it — nothing else would
have.)*

**③ Speed sits exactly on the boundary and must not be called either way.** Its point
ratio is remarkably stable (0.709 → 0.698) but its CI straddles 1 in both directions
(upper bound 1.034 then 0.998). **"Separated" at n=200 rests on 0.002.** The honest
statement is *suggestive of ~30 % compression, not established.* There is also a genuine
confound: PhysicalAI's speed p95 is 26.3 m/s vs YouTube's 18.8, i.e. the two corpora
differ in road mix, so part of any speed-spread gap is content, not model failure.

**④ `long_accel`'s "agreement" is meaningless.** Its ratio of 1.068 [0.937, 1.239] makes
it look like the best-transferring channel. It is not: IDM-v2 measures this channel at
R² **−0.240** on PhysicalAI, i.e. worse than predicting the mean. Two corpora agreeing on
the shape of a channel that carries no signal is two samples of the same noise. **This is
exactly why the brief demanded per-signal reporting: an aggregate would have let
`long_accel`'s spurious agreement offset yaw's real collapse.**

**⑤ A mean-level yaw claim I made at n=100 also evaporated — retracted.** At n=100 the
YouTube yaw mean was +0.00946 [0.0033, 0.0162], a CI excluding 0, which reads as a
systematic rightward bias. At n=200 it is **−0.00042 [−0.0060, 0.0048]**, centred on
zero. **There is no yaw bias.** Logged so it is not carried forward.

**Leading hypothesis for the yaw collapse (HYPOTHESIS, not measured): geometry.**
`fully_canonical` is false on **60 %** of the 200 clips (§4) — the crop never reaches the
`f_eff ≈ 266` the encoder was trained at. A wrong effective focal length distorts
apparent *rotation* far more than apparent forward motion, which fits the pattern (yaw
worst; `long_accel`, which is near-zero-mean noise on both corpora, unaffected). **The
discriminating experiment is cheap and needs no new harvest:** re-score these same
latents restricted to `fully_canonical == true`, paired.

**⑥ Speed's implausibility rate does NOT degrade out-of-corpus.** Predictions outside the
physical envelope: YouTube **2.01 %** [0.92, 3.49] vs PhysicalAI **1.88 %** [0.17, 4.01] —
overlapping, no separation. All violations are *negative* speed on a forward-facing
camera. yaw, steer and accel produced **0 %** out-of-limit predictions on both corpora.
**So the out-of-corpus failure is not wild values; it is collapsed range — which no
plausibility or sanity check in this pipeline would ever catch.** The shipped
`speed_sanity` block reports `frac_in_plausible_0_45_mps = 1.0` and calls it healthy.

**⑦ Removing the contaminated clips does not rescue it** (159 clean clips vs 200 all):
yaw spread **0.2426** [0.2129, 0.2723] clean vs **0.2539** [0.2262, 0.2878] all — the
intervals overlap almost completely. **The yaw collapse is a property of the domain, not
of the 41 bad clips.** Fixing the filters (§7) is still necessary for label quality, but
it will not recover the lateral channel.

### 6.1 Per-signal accuracy where GT exists — and the defects it inherits

INHERITED from `…/2026-07-26-idm-v2/IDM_V2_RESULTS.md` (MEASURED there; paired
episode-cluster bootstrap over 36 val episodes, n_boot 2000 — the decision-grade
estimator). **Reported per corpus, never pooled.** Not re-verified by me.

| signal | PhysicalAI | comma2k19 | status |
|---|---|---|---|
| **speed** | R² 0.907 / MAE 2.44 m/s | R² 0.759 / MAE 3.22 m/s | usable |
| **yaw-rate** | R² **0.9035** | R² **0.0719** | *pooled* R² 0.105 is an artifact — see below |
| **steer** | R² 0.742 | not comparable (`STEER_RATIO` 15.3 vs `atan(2.9·κ)`) | per-corpus only |
| **long_accel** | R² −0.240 | ≤ +0.12 | **unusable at any setting** |

**The three known defects are all still live, and they are load-bearing:**

1. **Impossible yaw labels remain in the label set.** Deleting **9 windows out of 4,195**
   whose ground-truth yaw is physically impossible moves pooled `yaw_rate` R² from
   **0.105 → 0.497**. Any bare "yaw R² ≈ 0.1" is measuring those 9 windows.
2. **The comma heading derivation is wrong** — `arctan2` of ENU velocity, undefined at
   standstill, producing |ω| up to 15.5 rad/s at v ≈ 0. comma's yaw R² 0.072 is
   substantially this defect, not the model.
   > 🔴 **CLOSED + RE-ISSUED 2026-07-27 (C29) — defect #2 was fixed, and this prediction was right.**
   > MEASURED: **26.27 %** of comma frames below 0.5 m/s carry a physically impossible `|yaw_rate|`,
   > **0.000 %** above (PhysicalAI **zero in every bin** ⇒ the `R² 0.9035` cell above is UNAFFECTED
   > and must not be re-issued). On the **identical windows**, deployed head, nothing retrained,
   > `heading_repair` ON with `v_min` 0.5: **comma2k19 `yaw_rate` R² 0.0719 → +0.3308** (and the
   > un-deleted legacy baseline was **+0.0114**); pooled **0.105 → 0.8108**. A head *retrained* on
   > repaired labels reaches comma **+0.679**. Superseded values kept above for audit.
   > ⭐ **Honesty condition:** comma-only MAE **−42.5 %**, but **medAE −1.1 % and nMedAE 8.0 % WORSE**
   > — tail and summary statistic, **not** typical accuracy. This section's own rule ("do not quote a
   > single aggregate IDM score") still stands. Inventory:
   > `…/Benchmarks & Eval/Implementation/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.
3. **`long_accel` is still in `SCALAR_NAMES`** — confirmed live on pod3 this run:
   `SCALAR_NAMES = ('speed', 'yaw_rate', 'steer', 'long_accel')`. IDM-v2 pre-committed to
   removing it; it has not been removed. It consumes 25 % of the scalar loss to emit a
   number no consumer may use.

**Therefore: I do not quote a single aggregate IDM score, and the published pooled yaw
number should not be quoted either.**

---

## 7. Recommendations (priority 4)

Ordered by value per unit cost. Items 1–4 need **no GPU and no new YouTube traffic.**

1. **Fix the time-manipulation filter — 0 GPU, 5 minutes.** Replace the hand-written
   `BAD_TITLE` substring tuple in `harvest_scaleup.py` with a regex
   (`\b\d+(\.\d+)?\s*x\b` plus the lapse/sped/fast-forward terms). The current list
   misses `3x` and would equally miss `1.5x`, `6x`, `8x`, `x3`. **This is corrupting the
   `speed` channel — the one the pipeline calls primary and scores its verdict on.**
2. **Add a game/simulator reject and a review/talking-head reject — 0 GPU.** `BeamNG`,
   `Assetto`, `Forza`, `Euro Truck`, `GTA`, `simulator`; and `best …`, `review`,
   `before you buy`, `top N`, `unboxing`. Regexes are in `collect_evidence.py` and are
   already validated against this harvest. Better still, gate on the *uploader* — the
   productive videos came from a handful of genuine drive channels.
3. **Re-run the geometry-restricted paired arm — 0 new harvest.** Re-score the *same*
   latents restricted to `fully_canonical == true` and compare paired against the full
   set. This is the cheapest discriminating test of whether the yaw/steer collapse is
   geometry or domain, and every input already exists on pod3.
4. **Apply the three IDM label fixes that are still pending** (drop impossible yaw
   labels, fix comma heading derivation, remove `long_accel` from `SCALAR_NAMES`). IDM-v2
   pre-committed to all three; none has shipped. `SCALAR_NAMES` still reads
   `('speed', 'yaw_rate', 'steer', 'long_accel')` on pod3 **today**.
5. **Persist all four channels in `pseudo_label.py`.** It currently discards `steer` and
   `long_accel` before writing, which is why answering Sayed's question required a
   separate re-run. Persisting four float arrays costs nothing.
6. **Change the discovery strategy, not the request rate.** The binding constraint on
   this run was **not** rate-limiting — it was that the 25 search queries exhaust their
   usable pool after ~100 clips and then return dashcam *merchandise* videos.
   `channels.txt` is **empty**; the NOTE itself records that channel enumeration of long
   continuous drives is the high-yield path. A curated channel list is worth more than
   any number of extra search queries.
7. **Do not re-run the harvest to chase 400 clips before 1, 2 and 6 land.** More clips
   from the current query set means more review footage, not more driving.

---

## 8. Deliverable manifest

**Nothing of value lives in only one place.** Every script and every JSON quoted above is
in the repo working tree and **staged** (`git add`). **I ran no `git commit` and no
`git push`.**

⚠️ **The documented index-sweep hazard fired again during this session — reporting it as
data, not as an accusation.** While I was mid-run, a concurrent agent's whole-index commit
`7e6b123` *("v4: λ_plan EXONERATED …")* swept in two of my in-progress files: the modified
`run_scaleup_parallel.sh` and an early draft of this `DB_RETRY.md`. Nothing was lost and
the content is correct, but **a YouTube-harvest driver change and a half-written IDM report
are now recorded under a v4 planner commit message** — exactly the failure CLAUDE.md
describes ("a quick commit of my thing silently sweeps in a sibling's half-finished code
under the wrong message", `60265d3`, `3d41bd0`). This is now at least the **third**
occurrence. The current, final versions of both files are staged and uncommitted.

| artifact | repo path (staged) | pod3 path |
|---|---|---|
| this report | `repo:…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md` | — |
| yield verifier (byte + shape + finite + duration) | `repo:…/verify_yield.py` | `/workspace/tmp/yt_scaleup/scripts/verify_yield.py` |
| per-signal instrument (4 channels + spread-ratio CI) | `repo:…/yt_persignal.py` | `…/scripts/yt_persignal.py` |
| evidence collector (block scan, evasion audit, contamination) | `repo:…/collect_evidence.py` | `…/scripts/collect_evidence.py` |
| **block-scan + evasion audit + contamination** raw JSON | `repo:…/pod_artifacts/db_retry_evidence.json` | `…/results/db_retry_evidence.json` |
| **yield verification** raw JSON | `repo:…/pod_artifacts/yield_verification.json` | `…/results/yield_verification.json` |
| **per-signal n=200** raw JSON *(primary)* | `repo:…/pod_artifacts/persignal_final_n200.json` | `…/results/persignal_final_n200.json` |
| **per-signal n=100** raw JSON *(the replication that failed for steer)* | `repo:…/pod_artifacts/persignal_interim.json` | `…/results/persignal_interim.json` |
| harvest manifest | `repo:…/pod_artifacts/harvest_manifest.json` | `…/results/harvest_manifest.json` |
| pseudo-labels (per worker) | `repo:…/pod_artifacts/pseudo_labels_w{0,1}.json` | `…/results/pseudo_labels_w*.json` |
| **archived harvest logs** (the driver truncates these per round) | `repo:…/pod_artifacts/logs_archive/` | `…/results/logs_archive/` |
| driver, now gentle-by-default (`SLEEP=${SLEEP:-4}`) | `repo:…/2026-07-25-youtube-idm-scaleup/run_scaleup_parallel.sh` | `…/scripts/run_scaleup_parallel.sh` |
| clip latents (2048-d, non-imagery) | — *(regenerable; transient by design)* | `/workspace/tmp/yt_scaleup/latents/` |

⚠️ **`patch_threads.py` exists ONLY on pod3** (`…/scripts/patch_threads.py`, 29 lines) —
it is the env-gated thread-cap patch for `yt_pilot_common.py` from the 2026-07-25 run and
was never staged. It is small but it is the reason the pod's `yt_pilot_common.py` md5
differs from the repo's. **Flagged as single-disk, per operating-standard rule 1.**

---

## 9. Escalations

1. 🔴 **Strike the "residential proxy / different egress" line from
   `…/2026-07-25-youtube-idm-scaleup/NOTE.md:150-152`.** It instructs the next agent to
   evade bot-detection. It needs an owner and a deletion, not a footnote here.
2. 🔴 **The three IDM label defects are still live and are now blocking a second
   deliverable.** IDM-v2 §5 items 1–3 were pre-committed as 0-GPU fixes and assigned to
   no one. They have now propagated into this run: `long_accel` is still computed and
   still meaningless, and I had to caveat every yaw number I touched. **Assign an owner.**
3. 🟠 **Corpus contamination is a pipeline defect, not a one-off.** The three filter holes
   (§5) will recur on every future harvest. Fixes are §7 items 1–2, both trivial.
4. 🟠 **`fully_canonical == false` on 64 % of clips is unreported by the pipeline.** The
   manifest records GeoCalib confidence but never surfaces that most clips never reach
   canonical focal. It should be a headline field, and it is the leading suspect for the
   yaw/steer collapse.
5. 🟢 **Harvest-log truncation destroys evidence.** `run_scaleup_parallel.sh` opens
   `w*/harvest.log` with `>` each round. The round-2 `HTTP 403` was already lost before I
   archived. Change to `>>`, or keep the archiver.

6. 🔴 **THE RUN IS STILL GOING AND ITS TAIL IS UNOWNED.** I finalised at round 4
   (200 clips) because the priority-1..4 deliverables were complete; the driver
   (`pid 2233954`, pod3) continues into round 5 and will keep going until it hits
   TARGET=400 or stalls twice, then runs the 4-seed downstream read and writes
   `results/DONE`. **Nobody is watching it.** To bank the tail — this is the whole
   handoff, no other context needed:

   ```bash
   ssh tanitad-pod3 'ls /workspace/tmp/yt_scaleup/results/DONE && \
     /workspace/venv/bin/python /workspace/tmp/yt_scaleup/scripts/verify_yield.py \
       --work /workspace/tmp/yt_scaleup --target 400 \
       --out /workspace/tmp/yt_scaleup/results/yield_verification.json'
   scp -r tanitad-pod3:/workspace/tmp/yt_scaleup/results \
     "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-idm-youtube-db-retry/pod_artifacts"
   ```

   The downstream verdict lands in `results/results_scaleup_downstream.json` and is scored
   against `…/2026-07-25-youtube-idm-scaleup/PRE_REGISTRATION.md` by that directory's
   `summarize_verdict.py`. **Caveat for whoever scores it: the pre-registered ①/②/③ bar
   assumed a clean 500–1000-clip corpus. This corpus is 20.5 % contaminated (§5) and will
   land near 200–250 clips, so a "HOLDS" verdict from it would not be decision-grade.**
   ⚠️ **Do NOT relaunch the harvest to reach 400** — that would be a second YouTube run,
   which is outside the single-run authorization.
