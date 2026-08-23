# JOB CARD — ZOD pilot ingest + real-bytes verification (M-1.3 / M-3)

**Owner:** Data Engineering. **Runs on:** a big-disk machine with net — pod3 (REF-A,
idle between runs) or Colab T4, NOT the 4060 dev box (ZOD mini ~ tens of GB; full is
TB-scale). **Blocked on (escalated):** ZOD dataset ACCESS. The HF repo `Zenseact/ZOD`
is a custom loader script (arbitrary-code viewer, no plain `hf download`); real bytes
come through the **`zod` SDK** + Zenseact's access agreement (`opendataset@zenseact.com`,
CC-BY-SA-4.0 + privacy/no-military notice). This card is fully runnable the moment
access lands — the loader (`zod.py`) + its 19 unit tests are already green.

## Falsifier — PRE-REGISTERED, and PRE-ANSWERED on grounded geometry (do not re-derive)
> "ZOD front-cam geometry can't reach f_eff=266 at >=50% observed_frac -> escalate
> to Sayed with the measured number before building further." (BACKLOG P0 #1)

**Verdict: PASS (does not trip).** Measured on the published spec (120-deg HFOV,
3848x2168, equidistant KB): **f_eff=266.0, observed_frac=1.00, drop_in=True** — and
robust to the real KB coeffs (f_px=1780 + realistic k1/k2 still gives 266.0 / 1.00;
the FOV alone decides it, the k terms only refine f_eff a few %). So the geometry
block is CLEARED; this job card's job is the REAL-BYTES CONFIRMATION + feature
precompute, not a go/no-go. (Contrast PandaSet: height-bound at f_eff=467 -> blocked.)

## Cell 1 — access + fetch 5 pilot drives (ZOD mini)
```bash
pip install zod huggingface_hub
# after the access agreement is signed:
zod download --url <signed-url-or-token> --output-dir /workspace/zod \
    --subset sequences --version mini --num-scans 5   # ~5 short drives, small disk
# SDK layout per drive: single_frames/<id>/{camera_front_original/*.jpg,
#   oxts.hdf5, vehicle_data.hdf5, calibration.json}  (pin exact names on first run)
```

## Cell 2 — build REAL FThetaIntrinsics from calibration.json + verify + precompute
```python
import json, glob, h5py, numpy as np, torch
from pathlib import Path
import zod as zd                                    # this intake module (on PYTHONPATH)
from tanitad.data.epcache import save_episode        # or mixing.save_episode

def real_intr(calib_json):
    c = json.loads(Path(calib_json).read_text())
    fc = c["FC"] if "FC" in c else c["front"]        # PIN the real key on first run
    K = np.array(fc["intrinsics"]).reshape(3, 3)     # [fx,0,cx; 0,fy,cy; 0,0,1]
    k = tuple(np.array(fc["distortion"]).ravel()[:4])   # KB k1..k4
    w, h = fc["dimensions"]                           # or ["width"],["height"]
    return zd.kb_to_ftheta(float(K[0, 0]), k, float(K[0, 2]), float(K[1, 2]),
                           int(w), int(h), per_clip=True)

def oxts_fn(seq):                                    # OxTS -> (ENU pos[N,3], head[N])
    with h5py.File(seq["oxts"], "r") as f:
        # PIN real field names; ZOD gives poses OR lat/lon + heading:
        lat, lon, alt = f["lat"][:], f["lon"][:], f["alt"][:]
        head = f["heading"][:]                        # rad (or deg -> radians)
    pos = zd.wgs84_to_enu(lat, lon, alt, lat[0], lon[0], alt[0])
    # resample OxTS 100 Hz -> camera 10 Hz timestamps here (nearest-index)
    return pos, head

for drive in sorted(glob.glob("/workspace/zod/sequences/*")):
    d = Path(drive)
    seq = {"seq_id": d.name,
           "frames": sorted((d / "camera_front_original").glob("*.jpg")),
           "oxts": d / "oxts.hdf5", "calibration": d / "calibration.json"}
    intr = real_intr(seq["calibration"])
    print(d.name, json.dumps(zd.verify_real_clip(
        {**seq, "positions_enu": oxts_fn(seq)[0], "headings": oxts_fn(seq)[1]},
        intr=intr)))                                  # geometry drop_in + speed/steer
    # bulk precompute -> epcache shard (drop-in for MixedWindowDataset)
    ep = zd.build_episode({**seq, "positions_enu": oxts_fn(seq)[0],
                           "headings": oxts_fn(seq)[1]}, intr=intr, oxts_fn=None)
    save_episode(ep, f"/workspace/data/zod/_epcache/data:zod-{d.name}")
```

## Cell 3 — checks to CONFIRM on real bytes (report these numbers back)
1. **Geometry drop-in** per drive: `verify_real_clip(...)["geometry"]["drop_in"]`
   must be `True` and `achieved_feff_px` in [260,272], `observed_frac >= 0.5`.
   (Expected PASS — if any drive FAILS, THAT is the escalation trigger.)
2. **OxTS<->camera timestamp alignment** (10 Hz cam vs 100 Hz OxTS): nearest-index
   resample residual < 50 ms; speed/steer finite and plausible (v in [0,40] m/s).
3. **Steer-ratio recovery** (pass `can_steer_wheel_rad` from vehicle_data.hdf5):
   `recovered_steer_ratio` stable across drives (~16 for the XC90 wheel:road) —
   a stable ratio would justify adding CAN steer as a 2nd action source.
4. **A8 consequence** `a8_frame_change_fraction` — expected HIGHER than comma's 0.06
   (ZOD urban/night has more scene change than CA-280 highway) = the diversity win.
5. **Semantic coverage** from the GeoJSON annotations: country / night-fraction /
   intersection count per drive -> the data card's curve-rebalance evidence.

## Push-back
Write `verify_real_clip` JSON + the epcache row-counts into
`Implementation/incoming/2026-07-18-zod-loader/RESULTS_realbytes.json` and update the
INTAKE evidence section. Colab storage is ephemeral (<=2 h) -> push before the session
ends; on the pod, the epcache persists for the lake ingest (BACKLOG P0 #2).
