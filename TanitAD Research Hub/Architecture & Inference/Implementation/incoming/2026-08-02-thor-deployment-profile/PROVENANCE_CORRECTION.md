# A15 — `thor_profile.py` did not produce `thor_profile.json`, and the gap is not cosmetic

**2026-08-18 · MEASURED (ours; every number below is reproduced on the dev box from
`stack/tanitad/config.py` + `stack/tanitad/data/calib.py`) · settles BACKLOG A15**

---

## Verdict

⭐ **`thor_profile.json` is CANONICAL. The banked `thor_profile.py` is NOT the script that produced
it, and running it would profile a DIFFERENT MODEL — quietly, without an error.**

⛔ **A banked script that cannot produce its banked result is worse than a missing one, because the
pair looks like provenance.**

---

## The evidence — three independent disagreements, not one

### 1. The script never assigns `out["frame"]`

`thor_profile.json` carries `"frame": "176x624 hfov 117.0"`. `thor_profile.py` assigns
`out["geometry"]` (line 76) and never `out["frame"]` anywhere in its 145 lines. Thor's copy adds it
via `resolve_v2_frames` (`stack/scripts/train_flagship_v4.py:764`), which the banked copy does not
call.

### 2. ⛔ The parameter count differs — the load-bearing one

| what | params |
|---|---:|
| the banked `thor_profile.py` as written (`flagship4b_config()` bare) | **263.44 M** (263 440 533) |
| `thor_profile.json` | **263.58 M** |

`flagship4b_config()` returns `image_size=256, image_width=None` — a **256×256 square** encoder. The
banked script sets `H, W = 176, 624` as a *tensor shape only* and never tells the config, so it
builds a 256×256 positional embedding (256 tokens) and feeds it 176×624 frames.

**That is exactly the failure `parity.assert_v2_geometry_matches` exists to catch** — *"omit the
flags and the trainer builds a 256×256 encoder and is fed 256×640 frames"*. It does not crash. It
produces plausible latencies for the wrong model.

### 3. And the model that DOES reproduce 263.58 M identifies a second missing flag

Sweeping the config against the JSON's own `"geometry": "176x624"`:

| `image_size` | `image_width` | `action_dim` | params | rounds to |
|---:|---:|---:|---:|---:|
| 256 | `None` | 2 | 263 440 533 | 263.44 ← **the banked script** |
| 176 | 624 | 2 | 263 573 397 | 263.57 |
| **176** | **624** | **3** | **263 575 190** | **263.58** ← **the JSON** |

Among width-624 configs (and the JSON's own `frame` string pins the width to 624), **only
`action_dim = 3` reproduces 263.58 M.**

⚠️ `action_dim = 3` is the **speed-as-third-action-channel** arm. The config default is **2**. So the
banked script, run as written, profiles the **no-speed** variant — the same
`phase0-30k`-vs-`speedjerk-30k` inversion `CLAUDE.md`'s first rule exists to prevent, arriving here
through an omitted flag rather than a copied name.

### 4. Bonus defect: the docstring's HFOV is the PARENT's, not the sub-frame's

`thor_profile.py:9` says *"176x624 sub-frame at **120 deg** HFOV cylindrical"*. Computed from
`tanitad/data/calib.py`:

```
parent  256x640 cylindrical @ 120.0000 deg  ->  f_ref 305.5775
sub     176x624 (centred slice rows[40,216) cols[8,632), same f_ref)
        ->  hfov 117.0000 deg   (624/640 x 120, exact for a cylindrical projection)
```

**120° is the parent cache's field. The deployed sub-frame is 117.0°.** The JSON is right; the
script attributes the parent's scope to the child — the same class as reading `df` on a pod.

---

## What is canonical, stated once so it can be quoted

| fact | value |
|---|---|
| profiled frame | **176×624, cylindrical, f_ref 305.5775, HFOV 117.0°** — a centred slice rows[40,216) cols[8,632) of the 256×640 @ 120° parent |
| profiled model | `flagship4b_config()` at that frame with **`action_dim = 3`** (speed channel) → **263.58 M** params |
| the latency numbers | **stand** — `encode_window_fp32` p50 187.81 ms, bf16 27.78 ms (6.76×), `predictor_roll20_fp32` 80.87 ms, all at the frame and model above |

---

## What was NOT done, and why

⚠️ **`thor_profile.py` is left byte-unchanged except for this correction being banked beside it.**
Retro-fitting the missing `resolve_v2_frames` call and `action_dim` would produce a script that
*looks* like the one that ran and still is not — the exact defect being reported. The recoverable
fix is to **re-ship the Thor-side copy** into this package; Thor is training (PID 25477, read-only
probes only), so that is a work item, not something to fake here.

⛔ **ESCALATION.** Until the Thor-side script is recovered, `thor_profile.py` must be read as a
*description* of the profile, not a reproduction of it. The three numbers above are what make the
JSON quotable on its own.
