# EXPLORATION — depth-probe grounding of DINOv3 features → metric BEV

**2026-08-31 · Master Mind · PI directive: "explore on 2 (Depth-probe grounding)"**
Status: **design + feasibility, nothing run.** Every number below carries its class.

## 1. What it attacks, and why it is worth a lane

Two of the programme's largest MEASURED deficits are the same deficit seen twice:

* **Longitudinal is 88.7 % of the oracle gap** (registry) — speed/headway/TTC need
  *metric range to the lead agent*, which a perspective feature grid never states.
* **The v6 readout geometry ceiling**: 4 azimuth bins over 120° (30°/bin), measured
  2.1–7.8× too coarse for BEV localisation — a GEOMETRY ceiling, not an encoder verdict.

A per-patch **depth read** converts the DINOv3 field from "what is where in the image"
to "what is where in the WORLD" — the frame the planner's costs (collision, headway,
free-corridor) actually live in.

## 2. Why DINOv3 specifically makes this cheap

**PUBLISHED:** DINOv3's dense features support strong *linear* depth probing (the
DINOv3 report's dense-task suite; the entire Depth-Anything line is built by
distilling DINO-family features into depth). We hold the features already — the
refav1 stage-1 cache (and today's 24-episode probe cache) — so a depth head is a
**ridge/linear probe over tensors on disk, no encoder forward, no new GPU class.**

## 3. The supervision question — three sources, ranked

| source | what it gives | cost / catch |
|---|---|---|
| ⭐ **(a) `obstacle.offline` cuboids** | METRIC range to 87,481 dynamic agents, 97.44 % corpus coverage, already program-read | sparse (agents only) — but agents are exactly what longitudinal needs. Free. |
| **(b) ground-plane geometry** | dense depth on the road surface from camera height + extrinsics (`sensor_extrinsics` + `vehicle_dimensions`, both in the episode build) | flat-world assumption; breaks on grades. Free. |
| **(c) a zero-shot monodepth pseudo-labeler** (Depth-Anything-V2 class, research-licensed — sanctioned) | dense relative depth everywhere | needs metric calibration — which (a)+(b) provide as anchors. One GPU pass over sampled frames. |

⭐ The natural recipe is **(c) for density, (a)+(b) for the metric scale**, with the
probe rules applied: fit on the fit split only, pixel floor + constant control, n and
d printed.

## 4. ⚠️ The projection trap, pre-empted (C-class: formula outside its scope)

Our corpus is **CYLINDRICAL** (`projection_mode: cylindrical`, measured on both
epcaches today). The column is **linear in azimuth**: `az = (u − W/2) / f_ref`,
`f_ref = 305.577`, giving the rig's own 120°. Any BEV lift that uses the pinhole
back-projection `x = z·(u−cx)/f` is **wrong on this data** — the correct lift per
patch column is:

```
az      = (u_center − 320) / 305.577          # rad, linear in column
BEV x,y = r · (cos az, sin az)                # r = probed depth along the RAY
```

One patch column = 16 px = **3.00° of azimuth** — 40 columns over 120°. A BEV splat
of the 640-token field at probed depth gives a **40-bin azimuth × continuous-range**
metric field: **10× finer than the v6 readout's 4 bins**, from the same encoder.

## 5. The cheapest discriminating experiment (pre-registerable next)

**E-DEPTH-LIN (T0, ~zero GPU):** on today's 24-episode DINOv3 probe cache +
`obstacle.offline` joins: ridge from the 1024-d patch feature (at the cuboid's
patch) → range r. Controls per the doctrine: raw-pixel floor, constant control,
row-count n printed, fit/val episode-disjoint.
* **PASS** (beats pixel floor + constant, R² materially > 0): a depth read exists →
  pre-register E-DEPTH-BEV: splat features to BEV, re-run the agent-slot probes
  there, and hand the planner a metric headway cost.
* **FAIL**: linear depth is not in the features at our resolution → escalate to a
  2-layer probe (function class stated) before any verdict; a linear negative is
  not a learnability negative.

## 6. Where it plugs into refav1 (no architecture change)

1. **Planner cost** (change #8's natural extension): headway / TTC / free-corridor
   terms computed from the BEV splat of the *imagined* future field — the cost gains
   metric teeth without touching training.
2. **An auxiliary state head** (PhyLatent PSG form, LIT-2 — already adopted as the
   validated supervised-aux design, binding-compliant: labels use privileged data,
   inference stays vision-only).
3. **The four-family eval**: LONGITUDINAL currently wants a lead-agent range; a
   depth read makes that computable from vision alone at eval time.

## 7. What is deliberately NOT proposed

Fine-tuning DINOv3 (FROST-Drive: frozen beats fine-tuned), a dense depth decoder
trained end-to-end (cost, and the probe answers the existence question first), and
any claim that depth fixes P1/P2 — action-conditioning is orthogonal and stays
gated on O11/MM-E19.
