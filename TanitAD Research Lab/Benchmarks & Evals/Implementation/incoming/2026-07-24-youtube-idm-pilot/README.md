# YouTube-IDM pilot — harvest -> pseudo-label -> pretrain -> measure

A bounded, privacy-safe pilot that tests whether pretraining the (small) world-model on
**YouTube pseudo-labeled** dashcam video lifts downstream driving vs no-YouTube pretraining.
Runs on **tanitad-pod3** only. All scripts take **simple-token CLI** (file inputs, no inline
quotes) so they drive cleanly over native-OpenSSH from the dev box.

## Layout
```
yt_pilot_common.py   shared: CC verify, face/plate/body blur, 10Hz resample,
                     canonical focal crop (reuses tanitad.data.calib + comma2k19),
                     shot-cut score, pointer records
harvest.py           P2 — CC-gated harvest -> anonymized clips + pointers
pseudo_label.py      P3 — frozen v1 encode -> latents + pseudo ego-motion + speed sanity
run_youtube_pilot_downstream.py   P4 — FLOOR vs PSEUDO_YT decision read (+ bootstrap CI)
env_probe.py / ytdlp_test.py / get_cascades.py   env + egress + privacy-tool checks
queries.txt / seed_urls.txt   CC search queries + hand-verified seed ids
```

## Run (on pod3, PYTHONPATH required)
```bash
export PYTHONPATH=/workspace/TanitAD/stack
V=/workspace/venv/bin/python
D=/workspace/tmp/yt_pilot/scripts

# P2 harvest (bounded)
$V $D/harvest.py --queries-file $D/queries.txt --seed-file $D/seed_urls.txt \
   --max-clips 120 --per-video-clips 8 --clip-frames 250 --max-videos 60

# P3 pseudo-label (encode -> latents, delete frames)
$V $D/pseudo_label.py

# P4 downstream decision read (add --with-parity-ref for same-domain fraction-of-ceiling)
$V $D/run_youtube_pilot_downstream.py --seeds 3 --with-parity-ref
```

## Privacy contract (non-negotiable)
- Harvest **only** Creative-Commons uploads: per-video `license` field is re-verified; non-CC is
  rejected. Time-manipulated (5x/timelapse/hyperlapse) and out-of-duration videos rejected.
- **Face + license-plate (+ body) Haar blur is applied to the FULL-RES frame BEFORE** the 256
  downscale, so identifiable detail is destroyed before any pixel is written.
- **No raw video and no full-res frames are persisted.** Source mp4 is deleted right after decode;
  clip frames are deleted right after they are encoded to latents. Persistent artifacts are
  **latents** (2048-d vectors, not human-viewable), **pseudo-labels** (numbers), and
  **URL+timestamp pointers** to already-public CC videos.
- If the privacy detector cannot load, harvest **refuses to store footage** (raises).

## Geometry caveat (named domain-shift source)
YouTube camera intrinsics are unknown. The pipeline assumes a nominal HFOV (`--hfov-deg`, default
100) and crops to the canonical F_REF=266 half-angle. A wrong HFOV biases apparent-motion scale
and therefore pseudo speed. This is a measured limitation of the pilot, not a bug.

## Parity firewall
The downstream read uses the physicalai-**val** episodes as its own IDM split (finetune 15 /
test 65, both rigs) — the SAME split `run_idm_parity_validation.py` uses. It creates **no** WM
parity arm and does **not** re-select the canonical WM episode selection. Licensing-free on the
measurement side (parity data, not YouTube).
