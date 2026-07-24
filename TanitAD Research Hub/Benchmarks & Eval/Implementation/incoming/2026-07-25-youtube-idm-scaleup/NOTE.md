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

## GEOMETRY: GeoCalib status
- `…/incoming/2026-07-25-geocalib/geocalib_intrinsics.py` **NOT landed** at launch (polled).
  → running the **fixed-HFOV fallback (100°)**, recorded `geometry_source:"fixed"` + `hfov_used_deg`
    per pointer. **Re-runnable with GeoCalib later** by re-decoding the SAME pointers (no re-harvest).

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

<!-- YIELD / GEOMETRY-USED / VERDICT banked here as they land -->
### RESULTS  [pending — the detached run is multi-hour; verdict lands in results_scaleup_downstream.json]
- Actual yield (clips / videos / license mix): _pending manifest_
- Geometry used: **fixed-HFOV 100°** (GeoCalib had not landed at launch; re-runnable)
- Decision-grade verdict vs PRE_REGISTRATION ①/②/③: _pending downstream_
