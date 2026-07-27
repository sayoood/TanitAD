---
license: other
license_name: tanitad-research
license_link: LICENSE
tags:
  - autonomous-driving
  - inverse-dynamics
  - video-pretraining
  - pseudo-labelling
  - ego-motion
extra_gated_prompt: >-
  These weights are released for research use. They were trained on frozen
  latents derived in part from NVIDIA PhysicalAI-Autonomous-Vehicles, which is a
  GATED dataset. No dataset content is redistributed here — weights only.
---

# TanitAD `idm_head_v3` — inverse-dynamics head on a frozen driving world-model encoder

**PLACEHOLDER — numbers are filled from `arms_v3.json` / `compare_v3.json` by the
final pass. Do not push this file while this banner is present.**

## What it is

A small non-causal transformer that reads a window of **frozen** latents from
the TanitAD flagship-v1 encoder and regresses the ego-vehicle's state at the
window centre: `speed` (m/s), `yaw_rate` (rad/s), `steer` (road-wheel rad), and a
2 s ego-frame trajectory. It is an **offline pseudo-labeller** — non-causality is
intended, it is never run as a policy.

## What is new in v3

1. **A repaired `yaw_rate` label.** comma2k19 heading is `arctan2` of the ENU
   velocity, undefined at standstill. Repaired by holding the last observable
   direction.
2. **Camera geometry was tested and REFUTED as a conditioning signal**, with
   three controls. Reported because a negative result with controls is worth more
   than an unrun idea.
3. `long_accel` is **not a supported output**.

## Provenance

| | |
|---|---|
| encoder | TanitAD flagship-v1 `flagship4b-speedjerk-30k`, FROZEN, md5 `b5f07d9e3dd2ca643949bc86832e6585`, step 29999, state_dim 2048 |
| training corpora | NVIDIA PhysicalAI-Autonomous-Vehicles (**gated**) + comma2k19 (`commaai/comma2k19`) |
| what is in this repo | **weights only** — no frames, no poses, no clip identifiers |

## Licence and redistribution

Weights only. PhysicalAI-AV is gated and none of its content is redistributed.
comma2k19 is MIT-licensed at source. Users must obtain both datasets themselves.
