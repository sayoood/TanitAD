# D-B — the ONE authorized gentle YouTube retry

**Date:** 2026-07-26 · **Agent:** `db-youtube-retry` · **Pod:** `tanitad-pod3` (A40, idle; pod1/pod2/eval untouched)
**Status:** RUNNING — banked incrementally. Sections marked ⏳ are pending the run's completion.

> **HEADLINE (priority 1): the run was NOT blocked.** YouTube served pod3 normally on
> 2026-07-26 from 12:33 UTC. Discovery, metadata and downloads all succeeded. The
> 2026-07-25 hard block ("Sign in to confirm you're not a bot") **cleared during the
> cooldown**, and the gentle config did not re-trip it.
>
> **HEADLINE (priority 2): the yield is the problem, not the block.** ⏳ (final number
> below). And the harvested corpus is **contaminated at ≥20 %** by three classes the
> filters do not catch — a 3×-speed video, video-game footage, and dashcam *product
> reviews*. That is a label-quality finding that matters more than the clip count.

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

→ **zero hits** in `harvest_scaleup.py`, `run_scaleup_parallel.sh`, `yt_pilot_common.py`,
`pseudo_label.py`, `run_youtube_pilot_downstream.py`. The complete yt-dlp option set used
is `quiet, no_warnings, skip_download, extract_flat, noplaylist, noprogress, format,
max_filesize, outtmpl, sleep_interval, max_sleep_interval, sleep_interval_requests`.
Nothing in it disguises, authenticates, or reroutes the client; `sleep_interval*` only
*lowers* our request rate.

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
| bot-block signature scan | **0 hits** for `sign in to confirm` / `not a bot` / `HTTP 429` / `too many requests` | `db_retry_evidence.json` |

**The single non-200 seen was a per-video `HTTP Error 403: Forbidden` on `jum8PGur1PU`**
(12:55:42, w1). This is **not** a bot-block: videos immediately before and after it
downloaded normally from the same worker and IP, and the 403 recurred on no other video.
It is ordinary per-video access restriction (region/embed/age). The harvester recorded
`dl_fail` and moved on — **it was not retried**, which is the correct behaviour.

---

## 4. Yield ⏳

*Presence is not completeness.* `verify_yield.py` byte-sizes every latent, `torch.load`s
it, checks shape and finite-ness, and joins each to its pointer to confirm frame count →
duration. Final numbers pending run completion.

**Round 1 (complete, MEASURED — `yield_verification.json`):**

| check | result |
|---|---|
| latent files | 50 |
| **verified OK** (loads, finite, non-empty) | **50 / 50** |
| bad / zero-byte / non-finite | **0** |
| total latent bytes | 51,195,700 (~1.024 MB/clip) |
| pointers | 50, **0 duplicates** |
| full-length clips (248 frames = 24.8 s) | **50 / 50**, **0 short** |

`248` frames is the correct full length (`--clip-frames 250` stacked with `N_STACK=3`
→ 250−3+1). My first verifier pass used 250 and wrongly reported every clip as short —
corrected before any number here was quoted.

**Geometry (round 1):** GeoCalib per-video on **35 / 50** clips, fixed-HFOV fallback on
**15 / 50** (low confidence). Confidence: 10 high / 22 medium / 18 low.
⚠️ **`fully_canonical` is FALSE on 32 / 50 clips** — the crop did not reach the canonical
`f_eff ≈ 266`. Measured HFOVs run 40–89°, far from the fixed 100° assumption, which is
consistent with GeoCalib's published ~66.6° median, but it means **the majority of clips
are not on the canonical geometry the encoder was trained for.** This is a systematic
caveat on every YouTube number and is not currently reported by the pipeline.

---

## 5. 🔴 Corpus contamination — three classes the filters miss (MEASURED)

Found by auditing harvested titles against the `BAD_TITLE` reject list. **In the first 74
clips, ≥20 % came from videos that should never have been harvested.**

| # | video | clips | why it is contamination | why the filter missed it |
|---|---|---|---|---|
| 1 | `Kocl0ZOUejc` — *"4K Tokyo Met. EXPWY Night Drive **at 3x Speed**"* | **12** | **time-manipulated**: every pseudo-speed / trajectory label from it is inflated ~3× | `BAD_TITLE` lists `"5x"`, `"10x"`, `"4x speed"`, `"2x speed"` — **`"3x"` is simply absent** |
| 2 | `7hvoPyb-mlY` — *"Nissan Qashqai – Realistic POV Driving in 4K \| **BeamNG.drive**"* | **3** | **video-game footage**, not real dashcam video — it defeats the entire purpose of an *out-of-corpus real-video* validation | there is **no simulator/game filter at all** |
| 3 | `g8lL-_IwomM`, `gDLib8egfR0` — *"Best 4 Channel Dash Cam – 2025"*, *"Before You Buy A Dash Cam In 2026"* | **7+** | **product-review / talking-head** video, not continuous forward driving | the `dash cam` queries surface reviews; the shot-cut filter catches most (0 clips from 3 of them) but not all |

Class 1 is the damaging one: it corrupts **speed**, the channel the pipeline calls
`primary` and the one the downstream verdict is scored on. The `BAD_TITLE` tuple is a
hand-written substring list with an obvious hole; a regex over `\b\d+(\.\d+)?\s*x\b`
catches `3x` and every other multiplier.

*(Anonymiser counts corroborate class 3 independently: the review videos blurred 1643 /
431 / 384 faces-plates per clip versus 6–174 on genuine driving footage — talking heads.)*

---

## 6. Per-signal IDM results ⏳

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
resampling unit = clip; **never `overlapping_holdout_se`**). Results ⏳.

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
3. **`long_accel` is still in `SCALAR_NAMES`** — confirmed live on pod3 this run:
   `SCALAR_NAMES = ('speed', 'yaw_rate', 'steer', 'long_accel')`. IDM-v2 pre-committed to
   removing it; it has not been removed. It consumes 25 % of the scalar loss to emit a
   number no consumer may use.

**Therefore: I do not quote a single aggregate IDM score, and the published pooled yaw
number should not be quoted either.**

---

## 7. Deliverable manifest ⏳

*(final version at completion)*

---

## 8. Escalations ⏳
