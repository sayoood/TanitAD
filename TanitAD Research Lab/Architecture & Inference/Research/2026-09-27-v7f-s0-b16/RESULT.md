# S0 — the v7F DINOv3 seed at the LAUNCH geometry (ViT-B/16): does the cheap repair help, and what does the transplant lose?

**Pre-registered:** `Project Steering/PREREG_V7_SEED_POS.md` §6 rung **S0** (success criterion §2, controls §5,
deliberate-regression arm §4) — written 2026-09-04, before this run. **Run:** 2026-09-27 00:51–01:04 Berlin,
dev-box RTX 4060, by the TrainingFlyWheel. **Tier: none** — a step-0 representation / decodability diagnostic,
never a driving number. **Evidence class: MEASURED (ours)**, all numbers from `raw/`.

## What was run
E-SEED-2 (`…/2026-09-04-v7-seed-and-external-target/code/`, copied verbatim from tip `b3f7ea6f`) with ONLY these
changes (`code/s0_setup.py`, every patch asserted to land exactly once): DINOv3 **ViT-L/16 → ViT-B/16** (local HF
snapshot `5931719e…`, no download), `EncoderConfig` 1024×24×16 → **768×12×12**, feature buffers sized from the
config, and **`torch.manual_seed(0)` before every model build** (the L/16 run left `scratch` and the seeds' random
`pos` unseeded, so they were not reproducible; seeded here, verified bit-identical across two builds). The register's
instrument (`panel_kfold.py`, `rangeprobe_rff.py`) was imported from the tip, **code-identical** to the rescued copies
E-SEED-2c used (only a 13-line rescue header differs). `tanitad` imported from the tip snapshot (asserted).
The seeded B/16 trunk has **86,138,112** params, matching the seed stamp. Corpus: the same 24 clips / **2,372 rows**,
256×640 cylindrical, newest frame. Arms: `dino_hf` (published DINOv3 B/16, ImageNet-normalised), `seed_asis` (the
seed exactly as the v7F trainer feeds it: [0,1] input, random `pos`), `seed_imnet` (+ normalisation),
`seed_imnet_pos0` (+ `pos` zeroed — the pre-registered repair), `scratch` (random init — the deliberate-regression
arm), `pixel` (raw-input floor), plus the constant-only control.

## Controls — all hold
| control | must read | read |
|---|---|---|
| constant-only | exactly 0 | **0.0000** (exact) |
| instrument validity `dino_hf − scratch`, scene (`n_agents_psg`, register instrument) | SEPARATED | **+0.3205 [+0.1202, +0.5119]** ✅ |
| deliberate-regression arm `scratch` | must FAIL (contain the no-information value) | **−0.0606 [−0.214, +0.094]** ✅ |
| row-shuffled feature stream | ~ the no-information value | every arm within **[−0.021, +0.022]** ✅ |
| `n`, `d` | printed | n = 2,372 · clips = 24 · d_model 768 · probe PCA 96 → RFF 1,024 |

## The binding read — the register's own instrument (`raw/eseed2c_rff.json`, `raw/rff.log`)
Within-clip r, RFF probe, 6-fold clip-disjoint out-of-fold, paired clip-cluster bootstrap.

| contrast | scene (`n_agents_psg`) | ego (`speed`) |
|---|---|---|
| **the repair**: `seed_imnet_pos0 − seed_asis` | **−0.0053 [−0.1663, +0.1393]** — null | +0.0486 [−0.0797, +0.1672] |
| repaired seed vs raw pixels: `seed_imnet_pos0 − pixel` | +0.0698 [−0.1387, +0.2634] | −0.0154 [−0.2242, +0.1994] |
| **seed as wired vs random**: `seed_asis − scratch` | **+0.1941 [+0.0429, +0.3516] SEPARATED** | +0.0012 [−0.1762, +0.1862] |
| repaired seed vs random: `seed_imnet_pos0 − scratch` | **+0.1888 [+0.0113, +0.3537] SEPARATED** | +0.0498 [−0.1192, +0.2250] |
| **transplant loss**: `dino_hf − seed_asis` | +0.1264 [−0.0677, +0.3007] — not separated | **+0.2331 [+0.0884, +0.3661] SEPARATED** |
| `dino_hf − seed_imnet_pos0` | +0.1317 [−0.0040, +0.2793] — not separated | **+0.1845 [+0.0105, +0.3493] SEPARATED** |
| `dino_hf − pixel` | **+0.2015 [+0.0276, +0.3663] SEPARATED** | +0.1691 [−0.0494, +0.3811] |

(The linear-ridge pass 1, `raw/eseed2_panel.json`, is again VOID on the scene axis — every arm including
`dino_hf` reads a negative pooled R² on `n_agents` — exactly as at L/16; it is reported, not used.)

## Verdict against the pre-registration
- ⛔ **Binding success criterion: FAILED, as §0b predicted.** The repair does not raise scene content
  (−0.0053, null) and the repaired seed does not beat raw pixels (+0.0698, not separated). This **replicates the
  L/16 null at the launch geometry** (L/16: +0.0295 and +0.0061). The fold + zero-`pos` stays free hygiene; it is
  not a lever.
- ⭐ **New at B/16:** the transplanted seed **as wired already carries scene content beyond random init**
  (+0.1941, separated) — the v7F trunk is not starting from nothing.
- ⭐ **What the transplant loses at B/16:** a **separated** share of DINOv3's ego-speed content (+0.2331), while
  its scene-content loss (+0.1264) is **not established** on 24 clips — at L/16 it was separated (+0.2050). This is
  the evidence the PI needs for **`PREREG_V7F` §10 D1** (option A: wrap the real `DINOv3ViTModel` as the trainable
  initialisation, ~1 day + tests). It supports D1-A on the ego axis and leaves the scene axis open at this geometry.

## Caveats
Step 0 only — no training (the S1 rung, 2 matched tiny arms × 2,000 steps, asks whether any of this survives
training). 24 clips; one torch seed for the random inits. Representation diagnostic, tier none — no driving claim.

## Files
`code/` — the exact scripts run (patched copies + `s0_setup.py` + `run_s0.sh`); `raw/` — every result and log,
clip ids rewritten as sha12 (full ids and 12/8-char prefixes, 72 occurrences; `raw/MANIFEST.json` carries source
and banked md5s). Not banked (regenerable in minutes from `code/`): the 1.1 GB frame bank and ~700 MB of features.
