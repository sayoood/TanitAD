# FORCE_LON -- does forcing the tactical lon head fix refcv8's S6 false brake?  (tag smoke)

Stamp for every number: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Model: refcv8-r101-s0 EMA step 30,000 on CPU (fp32 trunk), per-window sampler seed identical across arms. Estimator: episode-cluster bootstrap over the episodes present in the window set, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm), NOT another inference run (the sampler seed is fixed per window; arm N2 reports the draw-to-draw floor).

## Verdict (committed table: HEAD = closure >= 0.50 with CI lower > 0 | DECODER = CI upper < 0.25 | else INCONCLUSIVE)

**DECODER** -- closure of F1 on FB = **-0.012** [-0.012, -0.012], n 2 windows / 1 episodes (draws with a non-positive denominator: 0). Numerator ADE_N - ADE_F1 = -0.0246 m; denominator ADE_N - ADE_R7 = +2.0515 m.

n needed (ESTIMATED): `{"assumption": "CI half-width ~ 1/sqrt(n_windows); the point estimate stays; ESTIMATED, not measured", "to_reach_HEAD": "unreachable at this point estimate (point < 0.50)"}`

Sample: FB rolled 2 (of 460 in banked S6), with F1 2 (F1-excluded, unlabelled lon_v7: 0); CT rolled 2, with F1 2 (excluded 0).

## Paired cells (episode-cluster bootstrap, taniteval.ci; new - ref; lower ADE is better)

### FB
| cell | ADE 0-6 s (m) | \|along\| 6 s (m) | signed along 6 s (m) |
|---|---|---|---|
| N_minus_R7 | +2.0515 [+2.0515, +2.0515] SEP (n 2/1 ep) | +5.4974 [+5.4974, +5.4974] SEP (n 2/1 ep) | -5.4974 [-5.4974, -5.4974] SEP (n 2/1 ep) |
| F1_minus_N | +0.0246 [+0.0246, +0.0246] SEP (n 2/1 ep) | +0.0571 [+0.0571, +0.0571] SEP (n 2/1 ep) | -0.0571 [-0.0571, -0.0571] SEP (n 2/1 ep) |
| F2_minus_N | -1.6051 [-1.6051, -1.6051] SEP (n 2/1 ep) | -3.9871 [-3.9871, -3.9871] SEP (n 2/1 ep) | +3.9871 [+3.9871, +3.9871] SEP (n 2/1 ep) |
| F1_minus_R7 | +2.0761 [+2.0761, +2.0761] SEP (n 2/1 ep) | +5.5545 [+5.5545, +5.5545] SEP (n 2/1 ep) | -5.5545 [-5.5545, -5.5545] SEP (n 2/1 ep) |
| F2_minus_R7 | +0.4463 [+0.4463, +0.4463] SEP (n 2/1 ep) | +1.5103 [+1.5103, +1.5103] SEP (n 2/1 ep) | -1.5103 [-1.5103, -1.5103] SEP (n 2/1 ep) |
| N2_minus_N | +0.1707 [+0.1707, +0.1707] SEP (n 2/1 ep) | +0.4221 [+0.4221, +0.4221] SEP (n 2/1 ep) | -0.4221 [-0.4221, -0.4221] SEP (n 2/1 ep) |
| Fown_minus_N | +1.6156 [+1.6156, +1.6156] SEP (n 2/1 ep) | +3.9548 [+3.9548, +3.9548] SEP (n 2/1 ep) | -3.9548 [-3.9548, -3.9548] SEP (n 2/1 ep) |
| GPU_N_banked_minus_R7 | +1.9524 [+1.9524, +1.9524] SEP (n 2/1 ep) | +5.4097 [+5.4097, +5.4097] SEP (n 2/1 ep) | -5.4097 [-5.4097, -5.4097] SEP (n 2/1 ep) |
| CPU_N_minus_GPU_N_banked | +0.0991 [+0.0991, +0.0991] SEP (n 2/1 ep) | +0.0877 [+0.0877, +0.0877] SEP (n 2/1 ep) | -0.0877 [-0.0877, -0.0877] SEP (n 2/1 ep) |
| GPU_N_banked_s1_minus_s0 | -0.9337 [-0.9337, -0.9337] SEP (n 2/1 ep) | -4.0525 [-4.0525, -4.0525] SEP (n 2/1 ep) | +4.0525 [+4.0525, +4.0525] SEP (n 2/1 ep) |

### CT
| cell | ADE 0-6 s (m) | \|along\| 6 s (m) | signed along 6 s (m) |
|---|---|---|---|
| N_minus_R7 | +1.0454 [-0.2174, +2.3082] n.s. (n 2/2 ep) | +1.6132 [-1.8524, +5.0788] n.s. (n 2/2 ep) | +1.6132 [-1.8524, +5.0788] n.s. (n 2/2 ep) |
| F1_minus_N | -0.2531 [-0.5716, +0.0654] n.s. (n 2/2 ep) | -0.5184 [-2.4892, +1.4523] n.s. (n 2/2 ep) | -0.5184 [-2.4892, +1.4523] n.s. (n 2/2 ep) |
| N2_minus_N | -0.0442 [-0.2129, +0.1244] n.s. (n 2/2 ep) | -0.3440 [-1.0154, +0.3275] n.s. (n 2/2 ep) | -0.3440 [-1.0154, +0.3275] n.s. (n 2/2 ep) |
| Fown_minus_N | -0.2531 [-0.5716, +0.0654] n.s. (n 2/2 ep) | -0.5184 [-2.4892, +1.4523] n.s. (n 2/2 ep) | -0.5184 [-2.4892, +1.4523] n.s. (n 2/2 ep) |
| Fbrake_minus_N | +0.0265 [+0.0112, +0.0417] SEP (n 2/2 ep) | -0.0988 [-0.2106, +0.0131] n.s. (n 2/2 ep) | -0.0988 [-0.2106, +0.0131] n.s. (n 2/2 ep) |
| CPU_N_minus_GPU_N_banked | +1.0876 [+0.0205, +2.1547] SEP (n 2/2 ep) | +1.9571 [-0.4162, +4.3303] n.s. (n 2/2 ep) | +1.9571 [-0.4162, +4.3303] n.s. (n 2/2 ep) |
| GPU_N_banked_s1_minus_s0 | -0.0589 [-0.0635, -0.0543] SEP (n 2/2 ep) | -0.1143 [-0.2061, -0.0224] SEP (n 2/2 ep) | -0.1143 [-0.2061, -0.0224] SEP (n 2/2 ep) |

## Secondary closures
* F2_on_FB_all_windows: point +0.782 [+0.782, +0.782] n 2
* F1_on_FB_|along|6s: point -0.010 [-0.010, -0.010] n 2
* F2_on_FB_|along|6s: point +0.725 [+0.725, +0.725] n 2
* F1_on_FB_numerator_vs_BANKED_GPU_N (mixes devices: sensitivity only): point -0.063 [-0.063, -0.063] n 2
* F2_on_FB_numerator_vs_BANKED_GPU_N (mixes devices: sensitivity only): point +0.771 [+0.771, +0.771] n 2
* N2_as_the_numerator_arm_on_FB (a NULL intervention, another sampler draw: the noise floor of this statistic): point -0.083 [-0.083, -0.083] n 2

## Controls
* **KA** (CPU N vs banked GPU `os`): max abs waypoint diff 5.024 m (S6 slots) / 5.024 m (all 8 slots); mean ADE diff all +0.5934 m (iid SE +0.5237), FB +0.0991, CT +1.0876; 0.05 m bar exceeded: {'all': True, 'FB': True, 'CT': True}. Noise context on the same windows: GPU seed1-seed0 mean -0.4963 rms +1.0089; CPU N2-N mean +0.0632 rms +0.1736. Head argmax agreement CPU vs GPU 1.000; FB windows still BRAKE_TO on CPU +1.000.
* **KB** (the force reaches the planner): FB own-class minus F2 displacement -0.0556 [-0.0556, -0.0556] SEP (n 2/1 ep) -> PASS_FB True; CT own minus Fbrake +2.2119 [+0.4954, +3.9285] SEP (n 2/2 ep) -> PASS_CT False. F1 displacement over the sampler-noise displacement (FB): -0.1793 [-0.1793, -0.1793] SEP (n 2/1 ep).
* candidate allocation: `{"F1": {"n": 4, "mean_share_of_allocated_candidates_in_forced_class": 0.9375, "same_class_share_under_native_N": 0.453125}, "F2": {"n": 2, "mean_share_of_allocated_candidates_in_forced_class": 0.9375, "same_class_share_under_native_N": 0.0625}, "Fown": {"n": 4, "mean_share_of_allocated_candidates_in_forced_class": 0.9375, "same_class_share_under_native_N": 0.65625}, "Fbrake": {"n": 2, "mean_share_of_allocated_candidates_in_forced_class": 0.9375, "same_class_share_under_native_N": 0.21875}}`
* selected-candidate index changed vs N: `{"F1": 0.5, "F2": 0.5, "N2": 0.25, "Fown": 0.75, "Fbrake": 0.0}`
* **CT no-degradation** (F1 on cruising non-brake windows): -0.2531 [-0.5716, +0.0654] n.s. (n 2/2 ep); degrades beyond CI: False.
* **KD** (Fbrake on CT must CAUSE the defect): ADE +0.0265 [+0.0112, +0.0417] SEP (n 2/2 ep); signed along -0.0988 [-0.2106, +0.0131] n.s. (n 2/2 ep).

## What this does NOT answer
One training seed (H-ESTIM-SEED-1); one fixed sampler seed per window (N2 reports the draw-to-draw floor); CPU fp32 trunk vs the banked GPU bf16 trunk (KA); FB is defined by the head's argmax in the BANKED GPU roll (a selection on the association side); open loop on logged frames -- not driving performance.
