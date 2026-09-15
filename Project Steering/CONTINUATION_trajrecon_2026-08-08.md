# Continuation brief — render the 2026-08-08 trajectory reconstruction

*Written 2026-08-08. Complete handoff: a fresh session should need nothing but this file.*

## The one job

Process `2026-08-08_14-19-54-android.zip` (Sensor Logger export, phone dashcam, **drive is in
France**) through the trajectory-reconstruction pipeline and produce the **results video with
in-image trajectory overlays and BEV panels**.

The pipeline is already in the repo and tested. **The only thing that ever blocked this was getting
the 180 MB of data into the sandbox.**

## State as of handoff

- Branch `claude/trajectory-reconstruction-integration-kxy9mf`, 11 commits, pushed, working tree
  clean. Draft **PR sayoood/TanitAD#3**.
- `stack/tanitad/data/trajrecon/` — 24 upstream modules, **all byte-exact against Drive**, plus
  `contract.py` (ours: session → `ToyEpisode`), `__init__.py` (lazy), `TANITAD_INTEGRATION.md`.
- `stack/tests/test_trajrecon_{contract,integrity}.py` — 84 tests pass with both comma2k19 suites.
- **Nothing about the pipeline is known-broken.** Do not re-audit it before running.

## Step 1 — can you reach the data?

The previous session could not: `drive.google.com` was **403 at the agent proxy** (network access
level **Trusted**, which excludes Drive). The PI then set the environment's network access. **An
environment change only applies to sessions created after it**, so this is the first session that
can test it.

```bash
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 https://drive.google.com
curl -sS --max-time 20 "$HTTPS_PROXY/__agentproxy/status" | head -30
```

`000` + `CONNECT tunnel failed, response 403` ⇒ still blocked. **Do not burn the session working
around it** — there is no workaround. The Drive MCP connector returns file bytes **inline as
base64**, and 180 MB is ~240 MB of base64, which no context window holds. (Measured: even a single
sensor CSV is ~1 MB, also past the limit.) Report it and pick up `Project Steering/BACKLOG.md`
instead.

If it is reachable, continue.

## Step 2 — fetch the zip

| | |
|---|---|
| file | `2026-08-08_14-19-54-android.zip` |
| Drive id | `1DitzfASALJhJ5GqE-sBEqa3t_oyOipHN` |
| **exact size** | **179,607,367 bytes** — verify this |
| parent folder | `10lR3QsRtG_LLigYtubXvBEHcngFOwXhg` ("Trajectory reconstruction") |

The sandbox has **no Google credential**, so the fetch is anonymous and the file must be shared
"anyone with the link". ⚠️ **Google serves a virus-scan interstitial for files >100 MB**, so a plain
`curl` of the `uc?export=download` URL returns a few KB of HTML that looks like a successful
download. Use `gdown`, which handles the confirm token:

```bash
pip install gdown
mkdir -p ~/trajdata/in
gdown 1DitzfASALJhJ5GqE-sBEqa3t_oyOipHN -O ~/trajdata/in/2026-08-08_14-19-54-android.zip
ls -l ~/trajdata/in/          # MUST be 179607367 bytes
unzip -l ~/trajdata/in/*.zip | tail -5   # sanity: a real zip, not an HTML error page
```

If the size is a few KB, you fetched the interstitial — the file is not link-shared, or the token
flow failed. Say so; do not proceed with a corrupt input.

## Step 3 — environment

```bash
pip install numpy scipy pandas opencv-python-headless matplotlib
apt-get update -qq && apt-get install -y ffmpeg     # supplies ffmpeg AND ffprobe
which ffmpeg ffprobe
```

⚠️ **`imageio-ffmpeg` is NOT sufficient** — it ships `ffmpeg` only, and `trajrecon.timesync` shells
out to **`ffprobe`**. Measured the hard way.

## Step 4 — run it

```bash
cd /home/user/TanitAD/stack
python -m tanitad.data.trajrecon.pipeline \
    --input-dir  ~/trajdata/in \
    --output-dir ~/trajdata/out \
    --lane-width 3.50
```

`pipeline.py` takes the **zip directly** (no manual extraction), is resumable via a fingerprint
registry, and renders the video itself. Useful flags: `--video-stride`, `--panel-height` (720),
`--crf` (22), `--no-video`, `--limit`, `--only`, `--force`.

### Parameters — decided, with reasons

| flag | value | why |
|---|---|---|
| `--lane-width` | **3.50** | **The drive is in France.** The 3.65 default is the US 12-ft figure. `scale_calib` can only observe the product `f*h`; lane width is the external metric that separates focal length from camera height, so a wrong value biases the **entire ground-projection scale** by ~4 %. |
| `--cam-height` | leave default **1.17** | `--plane-calib` is ON by default and measures height from the road-plane homography. **Report what it measured**, not the default. (README says 1.25 in one table and 1.17 in another; the code says 1.17 — code wins.) |
| `--lateral-offset` | leave default −0.35 | Prior only; unobservable from motion. `lane_calib` recovers it per recording from the ego-lane centre when markings are usable. |
| vehicle | Audi A6 e-tron defaults | wheelbase 2.946 m, ratio 15.9. If fitted with progressive steering the wheel angle is right in *shape*, approximate in *scale*. |

## Step 5 — what to report

The pipeline writes `report.json` / `report.md` / `diagnosis.log` per recording, plus `frames/`,
`sensors/` and the video. **Quote the raw JSON, never a summary.** Report at minimum:

1. **The diagnosis verdict** — `OK` / `DEGRADED` / `REJECT` and the findings behind it. The pipeline
   is built to *reject* a recording whose GNSS is a frozen cached fix or whose video will not align,
   rather than emit confident fictional ground truth. **A REJECT is a valid, useful result — report
   it as the finding, do not tune flags until it passes.**
2. **Time sync** — the estimated shift and the cross-correlation quality.
3. **Calibration actually measured** — mount yaw/pitch/roll, and the camera height `plane_calib`
   found vs the 1.17 default; lateral offset from `lane_calib` vs the −0.35 prior.
4. **`validate.py` hold-out** — position / speed / heading RMS **on this recording**. This is the
   first chance to replace the INHERITED upstream figures with our own.
5. **The video** — path, duration, frame count.

⚠️ Upstream README figures (hold-out RMS **2.23 m** / **1.27 m/s** / **0.84°**, 2025-08-11 session)
are **INHERITED**. They are not ours and must not be quoted as a TanitAD result. Producing our own
is part of this job.

## Traps that already cost time

- ⛔ **Do NOT add a CRLF-normalising `.gitattributes`.** The 24 upstream files carry mixed line
  endings faithfully (`camera.py` is CRLF for 20,222 bytes then LF for the last 2,184 — one clean
  structural transition present in the source). Normalising changes the byte counts and invalidates
  every transfer verification in the branch.
- **The full `pytest -q` is RED in this environment for pre-existing reasons** — 29 failed / 13
  errors / 2056 passed (sitclf, rig_clean_fix, resim, scena, readout_onnx_pool). None mention
  trajrecon; at least one is flaky across runs. Collection fails entirely unless `pyarrow` is
  installed. **Do not attribute these to trajrecon and do not try to fix them as part of this job.**
- **`git commit` commits the WHOLE INDEX.** Check `git status --short` for other agents' staged work
  first; `git commit -- <pathspec>` segfaults on this repo. Use `git commit -F <msgfile>`.
- `git add`'s exit code is not evidence — verify with `git ls-files --cached <path>`.
- Do not commit `__pycache__/` from the package directory.

## If the recording is processed successfully

Commit the **code/doc** outputs, not the bulk data. The video and frames are large; put the video
somewhere the PI can see it (`SendUserFile`) and commit the `report.json`/`report.md` plus a short
results note into the branch. Then update **PR #3**, which currently states plainly that the
recording was NOT processed — that section must be corrected once it has been.

## Background the PI may ask about

The previous session's failure worth knowing: a **corrupt `vp_calib.py` was committed as
"verified"** — 2,883 bytes of binary garbage that passed *both* an exact byte-count check
(same-length substitution) and `ast.parse` (damage inside a docstring). It was caught only because a
second transfer disagreed. `stack/tests/test_trajrecon_integrity.py` now guards it (no control
bytes, valid UTF-8) and is proven against the real artifact. All 24 files were re-scanned; it was
the only one.
