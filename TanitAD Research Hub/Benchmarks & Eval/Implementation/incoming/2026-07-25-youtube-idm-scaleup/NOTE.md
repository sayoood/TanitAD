# YouTube-IDM SCALE-UP — running NOTE (bank-as-you-go)

**Agent:** youtube-idm-scaleup subagent · **Pod:** tanitad-pod3 (A40, idle, fully provisioned)
**Started:** 2026-07-25 · Evidence classes: MEASURED (ours+path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

## Mandate
Upgrade the directional 80-clip CC pilot to a DECISION-GRADE result at non-CC scale
(~500–1000 clips, ≥4 seeds). Privacy is MANDATORY and preserved verbatim (blur full-res
pre-downscale, delete raw video, persist only latents+pseudo-labels+pointers).

## Provenance / authorization (stated plainly)
- **Non-CC licensing authorization is INHERITED from the task brief** (Sayed committed 2026-07-25);
  I did not hear it from Sayed in-session. The pilot explicitly escalated "non-CC = Sayed's call"
  and the 2026-07-22 `LICENSING_TIER_ANALYSIS.md` is the groundwork. The defensibility rests on the
  **pointers-only + delete-raw + full-res-blur** safeguards, which I preserve exactly and did not weaken.

## Host decision (deviation from the literal brief, with reason)
Brief said "pod1 for harvest/label." **Ran on pod3 instead:** pod3 is idle (GPU 0 MiB, v2 build done)
AND has the ENTIRE provisioned environment — venv (yt-dlp 2026.7.4 + opencv 4.11 + torch 2.8 + PyAV),
the 3.3 GB v1 encoder ckpt (`/workspace/tmp/idm/ckpt.pt`), parity caches (incl. sacred
`e438721ae894`), comma2k19-val, and 380 cached parity latents. pod1 has NONE of this and the encoder
ckpt is **HF-403-blocked** (`Sayood/` storage full) so it cannot be cheaply moved. pod3 is the only
viable host without a multi-hour rebuild. pod2 (flagship) + eval (GeoCalib) untouched, as required.

## Environment (MEASURED, pod3, 2026-07-25)
- GPU idle 0 MiB / 46068 MiB, 0% util; no active python/training procs (v2 build complete).
- Disk: `dd` 3 GB write OK @ **583 MB/s** to `/workspace/tmp` (real MooseFS-quota check; df is invalid here).
  Footprint bounded by the driver (batched harvest → encode+delete → repeat); peak ≈ one batch (~20 GB).
- Deps present: yt-dlp, opencv 4.11 (Haar cascades), torch 2.8 cu128, PyAV. Encoder ckpt + caches + 380
  parity latents present. yt-dlp extracts from pod3's datacenter IP (pilot MEASURED, no bot-block).

## GEOMETRY: GeoCalib — LANDED mid-task, INTEGRATED (coordinator directive 2026-07-25)
The GeoCalib deliverable landed while harvesting (`…/incoming/2026-07-25-geocalib/`). Coordinator:
running the decision-grade lift on fixed-HFOV would bake a systematic error into the headline —
GeoCalib MEASURED YouTube dashcams at **median HFOV ~66.6° (range 32–77°), only 1/12 near the
assumed 100°**; fixed-100° over-crops ~1.4× and inflates pseudo-speed on most clips. So I switched
the harvest (which had 0 clips — nothing lost) to **GeoCalib per-video geometry**:
- `harvest_scaleup.py` now calls `geocalib_intrinsics.decode_canonical_geocalib(mp4, anon, estimator=est)`
  in place of the fixed-HFOV `decode_canonical`. Per-video focal from 16 frames (median vFoV + MAD
  outlier rejection + confidence gate); **falls back to fixed-HFOV internally when low-confidence**
  → never worse than the pilot. Decodes `thread_type="NONE"` (a threaded PyAV decoder torn down with
  a live CUDA context DEADLOCKS — MEASURED by the GeoCalib agent). Blur still full-res (privacy intact).
- Per-pointer records: `geocalib_vfov_deg`, `geocalib_confidence`, `geocalib_fallback_used`,
  `achieved_f_eff`, `fully_canonical`. Manifest records the confidence distribution.
- **Honest bound (carry into report):** GeoCalib is NOT a precise oracle — 6.8% median focal error
  (comma2k19 GT), weak absolute tracking (r=0.41, regresses toward a ~50–55° vFoV prior),
  resolution-robust ≤480p. Good enough to beat fixed-100°, not to quote per-clip intrinsics as truth.
- Required `pip install git+https://github.com/cvg/GeoCalib` into pod3's venv (was only on the eval
  pod). If the install/CUDA path fails, `harvest_scaleup` auto-falls-back to fixed-HFOV (`--no-geocalib`
  or import-fail) and reports it.
- ⚠️ **TRAP (MEASURED, fixed): GeoCalib's unpinned `opencv-python` dep pulled cv2 5.0.0, which
  clobbered the pilot's pinned `opencv-python-headless==4.11.0.86` — and cv2 5.0 DROPPED
  `CascadeClassifier`, so the privacy Haar blur silently broke (harvest would refuse-to-store).**
  This is exactly the pilot's pinning rationale. Fix: after installing geocalib,
  `pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless==4.11.0.86`.
  Verified after fix: cv2 4.11.0 + `CascadeClassifier` present + geocalib/kornia/torch-cuda all import.
  (Root-cause class: dependency clobber of a security-critical pin — belongs in RETRACTION_LOG if it
  ever ships a broken privacy pass.)
- Fixed-HFOV reference: the **pilot's 80-clip result IS the fixed-HFOV baseline**; this scale-up is the
  GeoCalib-geometry decision-grade read. (A paired fixed-vs-GeoCalib arm at the SAME scale is a
  follow-up — clips are deleted post-encode, so it needs a re-harvest, not a re-crop.)

## P1 — NON-CC PIPELINE EXTENSION  [status: DONE, staged]
`harvest_scaleup.py` extends the pilot harvest: CC gate → opt-out (`--allow-noncc`, default on; license
still RECORDED per pointer); discovery broadened to `ytsearch` over general forward-dashcam queries
(`queries_noncc.txt`) + optional channel enumeration (`channels.txt`); yield caps raised for long
continuous drives; GeoCalib hook (`--geocalib-json`). **Privacy code (`yt_pilot_common.Anonymizer`,
decode+delete-raw, pointers) reused BYTE-IDENTICAL.** `run_scaleup.sh` = footprint-bounded, resumable,
dd-checked driver → P4 at ≥4 seeds. Syntax-checked (py_compile + bash -n). Staged + scp'd to pod3.

## P4 PRE-REGISTRATION  [status: DONE — see PRE_REGISTRATION.md]
Committed before the read: ① HOLDS-decision-grade (all ≥4 seeds beat floor + per-seed CI excludes 0 +
fraction-of-ceiling ≥0.80 + CI tightens vs pilot's std 0.047) → GO; ② PARTIAL/BOUND (win holds but
frac<0.80 or CI doesn't tighten → name domain-heterogeneity/label-noise/geometry); ③ FAIL/REVERSAL
(a seed doesn't beat floor or a CI includes 0 → pilot didn't survive rigor, full harvest NOT justified).

## P2/P3/P4 — EXECUTION  [status: LAUNCHED + RUNNING on pod3, banks incrementally]

### Smoke validation (MEASURED, pod3 2026-07-24/25) — non-CC pipeline proven end-to-end
Ran `harvest_scaleup.py` on a few clips before the full run (pilot discipline). Result:
- Harvested video `GQVhmeYPoHM` — **"Highway 11 Muskoka Region Northbound … 4K Dashcam POV Drive"**
  (uploader "Another Sunday Drive") = exactly the target long continuous forward-dashcam content.
- Pointer recorded **`license: null, is_cc: false`** → a NON-CC video that the pilot's CC gate would
  have REJECTED is now **kept and correctly recorded as non-CC** → the P1 extension works.
- `geometry_source: "fixed"`, `hfov_used_deg: 100.0` (GeoCalib absent); `shotcut_score 2.45` (clean,
  < 9.0 cut thresh); `n_frames_10hz 248` (proper 25 s clip); pointer carries url + start/end timestamps.
- **Privacy MEASURED-honored:** raw mp4 deleted immediately after decode (dl/ held only the in-flight
  download); only clip latents-to-be + pointers persist. Anonymizer loaded (face+plate+body cascades).
- Smoke dir deleted after validation (transient blurred clips removed).

### Throughput pivot → PARALLEL (MEASURED: pod3 has 96 cores; single-process used ~5)
The single-process `run_scaleup.sh` decodes each long video fully before emitting clips and used only
~5.4 of pod3's **96 cores** (540% CPU) → 0 clips in 10 min (full-res Haar-blur decode is the inherent,
privacy-mandated bottleneck). Since it had produced nothing yet (nothing lost), switched to
**`run_scaleup_parallel.sh`** — a **bounded pool of W=8 harvest workers** over disjoint query slices,
each in its own `--work` dir (so the tested `harvest_scaleup.py`/`pseudo_label.py` run UNCHANGED, no
clip_id races), round-based (25 clips/worker/round → ~29 GB peak footprint, dd-checked) with a
latent-merge into one namespace, then P4 at 4 seeds. This is the brief's "bounded worker pool" (no
sub-agent spawn). ~8× throughput.

### Full run (LAUNCHED detached, pod3, self-completing, footprint-bounded)
`run_scaleup_parallel.sh` **W=8 TARGET=600 SEEDS=4**, `setsid nohup` → `/workspace/tmp/yt_scaleup/run.log`.
Confirmed running: driver alive, **8 harvest workers decoding in parallel**, ROUND 1 cap→25/worker,
dd-check passed, GeoCalib-absent → **fixed-HFOV fallback** (re-runnable later from the pointers). Banks
incrementally to `/workspace/tmp/yt_scaleup/results/` → repo `pod_artifacts/` (harvest_manifest,
per-worker pseudo_labels, then results_scaleup_downstream.json + DONE) via a bounded dev-box poller.

### GeoCalib smoke (MEASURED, pod3 2026-07-25) — full GeoCalib path validated end-to-end
2-clip GeoCalib harvest of a real non-CC video ("I-29 North Sioux Falls 4K Highway Drive", is_cc=false):
- **GeoCalib per-video: vfov 65.76°, hfov 98°, confidence "high", MAD 2.33°, fallback_used false**;
  crop landed at **achieved_f_eff 266.21, fully_canonical true** (canonical focal reached per-video).
- **No CUDA/decode deadlock** (clean completion, GPU freed); privacy blur intact (14 faces/58 plates/
  21 bodies blurred via restored cv2 4.11). Weights (111 MB) downloaded once → torch-hub cache (prewarm,
  so the 8 workers reuse it, no download race). → the GeoCalib path is MEASURED-working end-to-end.

### Full GeoCalib run (LAUNCHED detached 2026-07-25 ~23:48Z, pod3)
`run_scaleup_parallel.sh` **W=8 TARGET=500 SEEDS=4**, inline GeoCalib per-video geometry.
MEASURED healthy at round 1: 8 workers, **GPU 7.8 GB (8 GeoCalib models), loadavg ~16 on 96 cores
(no thread thrash — the thread-cap + single-thread NONE-decode fix)**. Banks incrementally to
`results/` → repo `pod_artifacts/` (harvest_manifest, per-worker pseudo_labels, then
results_scaleup_downstream.json + DONE) via a bounded dev-box poller. Est. ~1–1.5 h to verdict.

### 🔴 RESULTS — HARVEST BLOCKED BY YOUTUBE BOT-DETECTION (verdict NOT produced)
**Status (honest):** the full pipeline is BUILT + VALIDATED end-to-end, but the **decision-grade harvest
could not complete — YouTube hard-blocked pod3's datacenter IP** ("Sign in to confirm you're not a bot")
partway through. Confirmed a HARD block, not burst-throttle: a single isolated `extract_info` request
also fails (MEASURED 2026-07-25). **The decision-grade verdict is therefore UNANSWERED by this run.**

**What IS validated (MEASURED, real clips before the block):**
- Non-CC harvest works: real non-CC dashcam videos harvested (e.g. "I-29 North Sioux Falls 4K Highway
  Drive", "Tasman Highway Hobart") with `is_cc=false`, license recorded per pointer.
- Privacy intact: full-res face/plate/body Haar blur (14 faces/58 plates/21 bodies on one clip), raw
  mp4 deleted after decode, only latents+pointers persist.
- **GeoCalib per-video geometry works**: e.g. vfov 65.76°/hfov 98° high-confidence (I-29), hfov 59° low
  (Tasman) — the confidence gate + fixed-HFOV fallback behave as designed; crop lands at f_eff≈266,
  fully_canonical, NO CUDA/decode deadlock (thread_type=NONE).
- Env fixed: geocalib installed on pod3; opencv restored to 4.11 (CascadeClassifier) after the dep clobber.
- Reached ~65 clips in the last full run + ~80 in the pilot before the block; several rounds ran clean.

**ROOT CAUSE (own the mistake, RETRACTION-class = operational churn):** I churned too many high-volume
harvest runs during development — single→parallel(×3 restarts to fix thread-thrash/footprint/geometry)→
GeoCalib→3 smokes + a 65-clip run — from ONE datacenter IP. The cumulative burst volume tripped YouTube's
anti-bot. **Lesson: harvest GENTLY and get-it-right-first-time — few workers, rate-limited, validate on a
2-clip smoke BEFORE any wide run; do NOT iterate architecture against the live source.** The pilot's
"no bot-block" held only because it ran ONCE at low volume.

**I did NOT bypass the block** (no cookies/sign-in = prohibited credential action; no player-client
evasion = too close to prohibited bot-detection bypass). Respecting the block is correct.

### HANDOFF — the pipeline is one gentle command from the verdict, once the IP cools down (hours) OR on a different egress
1. Wait for the pod3 IP cooldown (YouTube blocks typically clear in hours), OR use a different egress
   (residential proxy / different pod IP / a machine YouTube hasn't flagged).
2. GENTLE re-run (rate-limited, low concurrency — harvest_scaleup now has `--sleep`):
   ```bash
   ssh tanitad-pod3 'PYTHONPATH=/workspace/TanitAD/stack W=2 TARGET=400 SEEDS=4 \
     setsid nohup bash /workspace/tmp/yt_scaleup/scripts/run_scaleup_parallel.sh \
     > /workspace/tmp/yt_scaleup/run.log 2>&1 &'
   # (for extra gentleness add `--sleep 4` to the harvest call in run_scaleup_parallel.sh)
   ```
3. It self-completes (GeoCalib geometry, ≥4 seeds) → `results/results_scaleup_downstream.json` + DONE.
4. Bank it: `bash collect_results.sh` (dev box) → `pod_artifacts/`, then `python summarize_verdict.py`
   prints the yield + fraction-of-ceiling + the pre-registered ①/②/③ verdict.

Everything needed to finish is staged + on pod3 (`/workspace/tmp/yt_scaleup/scripts/`, encoder ckpt,
parity latents, geocalib installed). Only YouTube egress is missing.

---

## DELIVERABLE MANIFEST
| artifact | repo path (staged) | pod3 path |
|---|---|---|
| non-CC + GeoCalib harvest (fast window-spread estimate, `--sleep` throttle) | `repo:.../2026-07-25-youtube-idm-scaleup/harvest_scaleup.py` | `/workspace/tmp/yt_scaleup/scripts/harvest_scaleup.py` |
| parallel driver (W workers, round-based, footprint-bounded, thread caps) | `repo:.../run_scaleup_parallel.sh` | `/workspace/tmp/yt_scaleup/scripts/run_scaleup_parallel.sh` |
| single-worker driver (reference) | `repo:.../run_scaleup.sh` | `/workspace/tmp/yt_scaleup/scripts/run_scaleup.sh` |
| discovery inputs | `repo:.../queries_noncc.txt`, `repo:.../channels.txt` | `…/scripts/` |
| P4 pre-registration (both outcomes ①/②/③) | `repo:.../PRE_REGISTRATION.md` | — |
| verdict summarizer (vs the bar) | `repo:.../summarize_verdict.py` | (run on dev box after collect) |
| collection one-liner | `repo:.../collect_results.sh` | — |
| GeoCalib contract/shim (legacy JSON path) | `repo:.../geocalib_shim.py` | — |
| README + this NOTE | `repo:.../README.md`, `repo:.../NOTE.md` | — |
| pilot scripts reused UNMODIFIED (pseudo_label, downstream) | (pilot dir) | `…/scripts/{pseudo_label,run_youtube_pilot_downstream}.py` |
| GeoCalib module (dependency, from the geocalib agent) | `repo:.../2026-07-25-geocalib/geocalib_intrinsics.py` | `…/scripts/geocalib_intrinsics.py` |
| **downstream verdict JSON** | — (NOT produced — harvest blocked) | — |

Nothing of value lives ONLY on the pod: all scripts + provenance are staged. `pod_artifacts/` is empty
(no round completed before the block). Latents/clips were transient and are cleaned.

## ESCALATIONS
1. 🔴 **YouTube bot-block on pod3's IP is the blocker to the decision-grade verdict.** Needs an IP
   cooldown (hours) OR a different egress, then the gentle re-run above. **This is the one thing gating
   the answer to "does the win hold at scale?"** — the pipeline is otherwise complete + validated.
2. **Own-mistake lesson (log to RETRACTION_LOG, class = operational-churn-against-a-rate-limited-source):**
   iterating pipeline architecture against the LIVE YouTube source with high-volume parallel bursts
   tripped the anti-bot. Future harvests: validate on a tiny smoke, then ONE gentle wide run.
3. **GeoCalib is a QUALIFIED instrument** (6.8% median focal err, r=0.41 absolute tracking, prior-regression)
   — beats fixed-100° but not a per-clip oracle; the confidence gate + fallback bound the risk. A paired
   fixed-vs-GeoCalib arm at scale (to isolate the geometry effect on the lift) is the follow-up once the
   corpus exists.
4. **Intake:** this `incoming/` folder should be intaken alongside the pilot + the geocalib deliverable.
