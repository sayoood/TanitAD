# RB — the candidate space: giant hierarchical vocabularies, tree/beam retrieval, prior-encoded pruning, and feeding the pruned set to the backbone

**HiCAP stream R-B · 2026-09-29 · author: Sonnet 5.5 research stream · status: RESEARCH (no GPU, no torch, no pod)**
**Files:** this note · `rb_hier_retrieval_toy.py` · `rb_hier_retrieval_toy_result.json` (same folder).
**Evidence classes:** PUBLISHED (id/venue/year) · MEASURED (ours, script path) · MEASURED-toy (synthetic world, mechanism only) · DERIVED (algebra done here) · ESTIMATED (arithmetic shown) · INHERITED (another stream or doc, not re-run) · HYPOTHESIS · UNVERIFIED.
**Budget used:** 30 of 30 WebSearch, 4 of 10 WebFetch (2 blocked: arXiv, alphaXiv); one public repo cloned (SparseDriveV2 @ `696ef77`, scratch only, not staged).

---

## 0. Headline

**Recommended candidate space (details §6):** a **factored product vocabulary** (geometric path factor × speed-profile factor, both partitioned by the v7 tactical tokens), **N = 1,024 × 256 = 262,144 entries** at first, with a **hierarchical index** (strategic gate → LAT/LON class → members), a **five-stage cascade** (ego masks → factored coarse scoring → compose + pair mask → late interaction → cross-attention re-rank of K = 16) and **tactical-cadence caching** of the survivor set. The whole cascade costs ≈ 0.34 GMAC per tick (ESTIMATED, K = 16), against a flat 262,144 × 512 table read of 268 MB (≈ 1 ms at Thor's 273 GB/s spec).

**Four findings that change how the PI's brief should be read**

1. **Speed is not the reason for a tree below about 4 M entries.** Brute-force scoring of 262,144 × 512 fp16 is 268 MB, i.e. 0.98 ms at the *spec* bandwidth (int8: 0.49 ms); 1.05 M entries: 3.9 ms (ESTIMATED, §4.4). A sibling cost model reaches the same order (143,360 leaves flat = 0.5 % of Thor bandwidth at 10 Hz; INHERITED, `hicap_cost_model_result.json`). The tree earns its place through **(i)** the *reduced set the backbone can afford to reason over*, **(ii)** level-wise priors and a native tactical readout, **(iii)** hierarchical estimation of priors for entries that have almost no data. A tree failure is therefore not fatal: flat scoring is the fallback.
2. **With marginal-consistent level critics the beam is not the bottleneck; the critic is** (MEASURED-toy, N = 65,536): a beam of (2 strategic, 8 cells) reproduces the flat pick in 99.0 % of states at **30.9× fewer multiply-adds**; a calibrated adaptive-margin beam uses **83–133× fewer** at 93–98 % recall of the expert's cell with no measurable loss in pick quality. What *does* hurt recall is the **node-critic type**: a mean-pooled centroid index loses 5 points of cell recall at beam (3,16) where a marginal critic loses 0.45 (§5.7).
3. **The log-prior correction is mandatory at every size tested, and the prior must be estimated hierarchically.** Uncorrected PMI scoring reaches only 0.47 of the exact-MAP hit rate (0.83 corrected) and picks the cruise cell in 46 % of states against 66 % for MAP; the bias did **not** grow between N = 4,096 and 131,072 in this toy. A prior estimated from ≤ 0.023 expert samples per leaf recovers only 50–75 % of the true-prior gain (§5.8).
4. **Hard comfort masks cost the rare, critical manoeuvres.** A nuPlan-comfort-limit mask removes 47.7 % of the vocabulary (MEASURED-toy) with 98.3 % marginal oracle survival, but survival is 93.3 % for stop and 94.3 % for turn states. Marginal conformal guarantees hide this; **class-conditional, episode-blocked calibration is required** (§3.4). A sibling stream's real-window check found 33.6 % of episode-disjoint splits below nominal coverage (INHERITED).

**Published precedent exists and is recent:** SparseDriveV2 (arXiv 2603.29163, ECCV 2026) scores a **1,024 × 256 = 262,144** path × velocity vocabulary ("32× denser than prior methods") with coarse factorised scoring over 1,280 tokens, then (128, 64), then (20, 10) = 200 composed trajectories (code read, §1). No vocabulary near 10⁶ was found in five searches plus that code (NOT found ≠ does not exist).

---

## 1. How the field builds and prunes huge action vocabularies (priority 1)

Status column: **P** = published id/venue seen in a search result this session; **C** = read in released code this session; **F** = inherited from `2026-09-29-reff-prior-art-and-theory.md` (F-R stream; V1-checked there where marked); numbers in quotes are as returned, not re-derived.

| system | vocabulary and mechanism | size / quality / compute evidence | implication for a hierarchical HiCAP | status |
|---|---|---|---|---|
| **SparseDriveV2** (2603.29163; ECCV 2026; github swc-17/SparseDriveV2 @696ef77) | **Factorised**: `K_PATH=1024` k-means geometric paths (`len_path=50`, 1 m arc-length interval) × `K_VELOCITY=256` k-means speed profiles (`len_vel_seq=8` at 0.5 s); a trajectory is composed by arc-length re-parameterisation (`cumsum(v·DT)` interpolated along the path; masked if the profile outruns the path). Decoder layer 1 scores 1,024 path tokens (deformable attention to image) + 256 velocity tokens and keeps (128, 64); layer 2 re-scores those, keeps (20, 10) (`path_filter_num=(128,20)`, `velocity_filter_num=(64,10)`), composes 20 × 10 = **200** trajectories and scores them with 8 PDM sub-score heads (BCE) + soft-target imitation CE. Losses use distance-softened targets (`path_sigmas`, `velocity_sigmas`, `trajectory_sigmas` = 4.0) | Abstract: a scaling study of Hydra-MDP shows performance "consistently improves as trajectory anchors become denser, without exhibiting saturation before computational constraints are reached"; 92.0 PDMS / 90.1 EPDMS (abstract), README 92.22 / 90.38 with ResNet-34, 50.4 M / 50.9 M params. **The per-size table could not be read (arXiv blocked): no numbers below 262,144 are quoted here** | The exact structure the PI asks for, at 2.6 × 10⁵ entries. Cost is set by **token counts (1,280 → 192 → 200)**, not by N. Path × speed decomposition makes speed kinematics depend on the profile alone, and lateral acceleration `v²κ` on the pair: the only coupling | P, C |
| **GTRS** (2506.06664; NAVSIM v2 challenge winner) | Scorer trained on a **super-dense static vocabulary with dropout**, inferred on smaller subsets and on unseen dynamic proposals | README: GTRS-Dense 16,384 vocab, GTRS-Aug 8,192, Hydra-MDP 16,384; EPDMS 41.7 / 42.1 / 37.5 (**benchmark split not stated in what I read: UNVERIFIED**). F: random selection 25.6 → 39.7 EPDMS; +11.1 zero-shot on unseen proposals | A shared action tower generalises to candidates it never saw: the reason to prefer a dual encoder over a fixed classifier | P, F |
| **DriveSuprim** (2506.06659; AAAI 2026) | Coarse-to-fine: vocabulary **8,192 → 256 → dedicated decoder**; rotation augmentation; soft self-distillation | "8192 and 256" (arXiv HTML); 93.5 PDMS v1, 87.1 EPDMS v2. F: oracle over a 256-fan top-1 91.9 · top-4 94.5 · top-16 96.1 · top-256 98.7 vs human 94.8; "easy-to-reject options dominate the gradient" | Retrieve-then-rerank is the published cure for easy negatives; top-K saturates by K ≈ 16 | P, F |
| **Hydra-MDP / ++** (2406.06978, 2503.12820) | 4,096 / 8,192 vocabulary; per-candidate sub-score heads distilled from rule teachers | F (V1 ✓): scalar PDM-score distillation 80.2 < imitation-only 80.9 < five separate heads 83.0; ++ 85.0 → 86.5 PDMS | Per-family sub-scores beat a scalar; matches the four-family eval rule | F |
| **VADv2** (2402.13243) | 4,096 FPS vocabulary; softmax `p(a|s)`; KL to a **soft** distance-weighted target | F: 1 s L2 0.082 with the distribution loss vs 1.415 without (17×); no size ablation | Supervision shape dominates inputs; soft targets are non-optional | F |
| **PDM-Closed / LLM-Assist** (2306.07962, 2401.00125) | 15 IDM proposals (3 lateral × 5 speed): the tiniest *factored* vocabulary | F: fixed scorer degrades 15 → 8,505 proposals (CLS-NR 92.51 → 77.78) | A **hand-set** scorer does not survive a large candidate set; a learned, calibrated one is required | F |
| **DiffusionDrive** (2411.15139) | 20 clustered anchors, truncated diffusion, 2 denoising steps | 88.1 PDMS (ResNet-34) | Few anchors + refinement is the opposite corner: small K, learned residual. HiCAP's leaf refinement head plays this role | P |
| **Trajectory tokenisers** | FAST (2501.09747: DCT + BPE, robot actions); DAP (2511.13306: **κ–a discretisation** = curvature × acceleration bins, 160 M params, 90.0 PDMS v1, 85.6 EPDMS v2); LaPla (2609.04070: residual VQ-VAE action tokeniser used as a physical prior); planning-aligned FSQ tokeniser (2606.07464); Unified Driving Tokens (2606.01935) | Snippet-level only; numbers **UNVERIFIED** beyond DAP's | A curvature × acceleration alphabet *is* a lateral × longitudinal factorisation, and it works as a token alphabet for a 160 M planner. Residual VQ gives a coarse-to-fine code for free (level = RVQ stage) | P |
| **MoE / conditional compute** — DriveMoE (2505.16278; CVPR 2026) | Skill-specialised action MoE: 1 shared + 6 routed experts, **top-3** selected, learned router | Bench2Drive SOTA claim (snippet) | Route by the **tactical token** deterministically instead of learning a router (§4.3) | P |
| **Retrieval planners** — VINN (2112.01511), RAG-Driver (2402.10828; RSS 2024), RealDrive (2505.24808, title only), VLADriver-RAG (2605.08133, title only) | Retrieve similar past experiences by state similarity and use them as in-context or kNN prior | Qualitative | The cached action arena doubles as a **state-conditional prior**: the cells of the k nearest train states' expert actions (§3.1, gate G7) | P, F |

**Our own numbers on size vs quality** (MEASURED, `Research/reff_v0_coverage_result.json`, val-40, 881 windows, oracle ADE@2s, matched-N ratios only, **no exponent is quoted** because the window is five points and CLAUDE.md forbids a bare exponent):

| N anchors | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| absolute | 2.307 | 1.661 | 1.148 | 0.803 | 0.599 |
| speed-normalised | 0.468 | 0.375 | 0.270 | 0.221 | 0.179 |

Four times more entries (64 → 256) cut the oracle ADE by ×0.52 (absolute) and ×0.66 (speed-normalised). The measured range ends at N = 256, so **nothing about 65,536 may be extrapolated** (more than 2× beyond the fitted range). REF-C's K4 result (INHERITED from the F-R file) shows the opposite side: oracle-in-fan 0.2213 / 0.1914 / 0.1640 for 64 / 128 / 256 anchors while `frac_sel_2x_worse` *rises* 0.3825 / 0.4109 / 0.4540. **A bigger vocabulary raises the oracle, not the pick**; hierarchy, priors and re-ranking are the counterweight.

**What the field implies for the design.** (1) Denser is better up to 262,144 on NAVSIM (PUBLISHED, abstract-level); (2) every system that goes large uses a factorised or coarse-to-fine structure to keep the *token count* near 10² – 10³; (3) a learned scorer over the reduced set (cross-attention decoder, sub-score heads) is what turns coverage into picks; (4) soft targets and per-family sub-scores are standard; (5) no published system prunes with calibrated guarantees or with cadence caching, which is where HiCAP can differ.

---

## 2. Tree-structured retrieval: theory and the formulas that matter (priority 2)

### 2.1 Costs (DERIVED)

- **Flat:** `C_flat = N·d` MAC and `T_flat = N·d·b` bytes read (b = 2 fp16, 1 int8). On Thor `ms = T_flat / 273 GB/s` (spec, peak).
- **L-level tree**, fan-out `B_l`, beam `w_l` (`w_0 = 1`): `C_tree = d · Σ_l w_{l-1}·B_l`. Balanced, constant w: `C = d·L·w·N^{1/L}`.
  Toy 8 / 32 / 256 with beams (3, 16): `8 + 3·32 + 16·256 = 4,200` node scores vs 65,536. **The last term dominates (4,096 of 4,200)**: cost ≈ kept cells × leaves per cell. Shrink it with an adaptive width or a further coarse-code level inside the cell (the sibling cost model uses beams (2, 4, 8) over 35 cells × 64 × 64 codes and touches 2,379 of 143,360 nodes; INHERITED).
- **Factorised product:** `C_fac = d·(K_P + K_V + k_P·k_V)`. For 1,024 × 256 with (20, 10): `1,280 + 200 = 1,480` scores vs 262,144 (177×).
- **Layout:** store leaves in **depth-first order** so a node's children are one contiguous slab; a beam step is then `w` contiguous matrix–vector products, with no pointer chasing (why a tree beats HNSW-style graphs on a static table on an SoC).

### 2.2 Beam recall as a function of the per-level margin (DERIVED, checked MEASURED-toy)

Level with `N` children; the true ancestor has score margin `m` over the competitors; per-node critic noise iid `N(0, σ²)`; beam width `w`. The ancestor is lost iff at least `w` competitors outscore it:

`P_miss = E_{ε*} [ P( Bin(N−1, q(ε*)) ≥ w ) ]`, `q(ε) = Φ̄((m+ε)/σ)`. (exact)
`P_miss ≤ (1/w) · Σ_j Φ̄( m_j / (σ√2) )`. (Markov; valid, loose at small m)

| children N, beam w | m = 0 | m = 2σ | m = 3σ | m = 4σ |
|---|---|---|---|---|
| 32, 4 | 0.875 (= 1 − w/N ✓) | 0.221 (bound 0.610) | 0.042 (0.131) | 0.0036 (0.018) |
| 32, 8 | 0.750 | 0.102 (0.305) | 0.0126 (0.066) | 0.00066 (0.009) |
| 256, 16 | 0.9375 | 0.326 (1.0) | 0.0745 (0.270) | 0.0074 (0.037) |

(`rb_hier_retrieval_toy_result.json` → `theory_beam_miss_iid_gaussian`.) Over levels, `P(survive) ≥ 1 − Σ_l P_miss,l`. **Design rule:** to keep level `l` at miss ≤ α_l, take `w_l ≥ (N_l − 1)·Φ̄(m_l/(σ_l√2)) / α_l`; the margin `m_l` is what the level critic must deliver, so the *width is a proxy for critic quality*.

### 2.3 What a node score must be (DERIVED; general-k framing from Zhuo et al. ICML 2020, 2006.15408, snippet-level)

For top-1 retrieval with exact node scores the Bayes-optimal node score is the **marginal** `log P_l(n | s) = log Σ_{a ∈ subtree(n)} p(a|s)`: the ancestor of the argmax is kept iff its marginal ranks in the beam. This is the *max-heap property* tree-based deep models need (TDM, 1801.02294; Deep Tree-based Retrieval, 2408.11345). Two consequences:
- **Training:** a level-wise softmax over that level's nodes, with the expert leaf's ancestor as label, estimates the marginal (hierarchical softmax, Morin & Bengio 2005; probabilistic label trees, Wydmuch et al. NeurIPS 2018; Bonsai 1904.08249; AttentionXML 1811.01727). Cost per sample `O(Σ_l B_l)` instead of `O(N)`.
- **Do not build the index by mean-pooling child embeddings.** With a dot-product critic a mean-pooled node scores the *average* child, not the best one, so a narrow peak inside a large cell is invisible. Measured in §5.7.

### 2.4 How level-wise log-priors compose (DERIVED)

- Chain rule: `log p(a|s) = Σ_l log p(n_l | n_{l−1}, s)`. A prior telescopes the same way: `Σ_l log p̂(n_l | n_{l−1}) = log p̂(a)`.
- InfoNCE with negatives drawn from the training marginal has the optimum `f_l*(s,n) = log p_l(n|s) − log p_l(n) + c(s)` (PMI; van den Oord 2018, Poole 2019, via the F-R file). Ranking by `f_l + β·log p̂_l(n)` recovers `log p_l(n|s)` when β = 1.
- **Two traps.** (a) **Beam ranking across parents needs the marginal prior `p̂_l(n)`** (sum over the subtree), *not* the conditional `p̂(n | parent)`. (b) **Sibling-only negatives leave a per-(state, parent) offset `c(s, parent)` unconstrained**, because a softmax over siblings is invariant to it; cross-parent comparison then mis-orders nodes. Train each level with **level-wide negatives**, or with a normalised softmax over all nodes of the level. Measured in §5.6 (it damages the flat scorer too if the leaf critic is trained sibling-only).
- **Prior estimation for sparse leaves.** With ~406 k train windows over 262,144 leaves the mean is ~1.5 samples per leaf. Use the hierarchical backoff `p̂(a) = p̂(S)·p̂(cell|S)·[(c_leaf + κ/L_f)/(c_cell + κ)]` (Dirichlet-tree form, κ ≈ 0.5 sample) rather than a flat smoothed histogram (§5.8).

### 2.5 Hierarchical contrastive objective (design; HYPOTHESIS until run)

`L = Σ_l λ_l · CE_l( s, q_l )`, with soft level-targets `q_l(n) = Σ_{a ∈ n} q(a)` from the plan's distance-kernel `q ∝ exp(−d²/σ²)`, negatives = **all** level-l nodes, and the ε-mask of the plan (§3.2) extended to nodes whose subtree contains a leaf within ε of the expert (false negatives are most frequent at coarse levels). HiMulCon / HiConE (Zhang et al., CVPR 2022, 2204.13207) supply the two ideas that transfer: **penalise by ancestry overlap** and **never let a farther pair have a smaller loss than a closer pair**. `λ_l` increasing with depth; a fixed choice is pre-registered before any run.

### 2.6 Product versus joint (DERIVED + PUBLISHED precedent)

| option | logits per sample | interaction | when it is exact |
|---|---|---|---|
| joint softmax over N | N (262,144 × batch 2,048 = 537 M) | full | always; memory-heavy |
| additive `s_P(p) + s_V(v)` | K_P + K_V | none | conditional independence given s |
| **coarse-then-compose (SparseDriveV2)** | 1,280 coarse, then 200 fine | fine stage models `v²κ` and braking-before-curve | independence up to the pair constraint |
| hierarchical softmax over cells → members | Σ_l B_l | inside each cell | tree = the label hierarchy |

Toy evidence (§5.2): a 20 × 10 composed set agrees with the flat pick in 95.3 % of states at 78× fewer scores; an 8 × 4 set agrees in only 56.5 % yet has an indistinguishable hit rate (near-equal leaves on the posterior plateau).

### 2.7 When a real MIPS index would be needed (ESTIMATED)

Brute force on Thor reads `N·d·b` bytes: 4.19 M entries fp16 = 15.7 ms (int8 7.9); 16.8 M = 62.9 ms (int8 31.5). Only beyond roughly 4 M entries does an approximate index matter: IVF (`nlist ≈ √N`, `nprobe` cells), product quantisation (Jégou et al., TPAMI 2011: a `Q`-way Cartesian product of small codebooks, `M^{1/Q}` storage per dimension-group), HNSW (1603.09320, logarithmic search, pointer-chasing), ScaNN (1908.10396, anisotropic quantisation that spends accuracy on the *highest-scoring* items). All are engineered for CPU or server GPUs; none is a Thor 10 Hz precedent (no source found). The structured tree above is the hardware-friendly choice at our sizes.

---

## 3. Prior-encoded pruning logic for driving (priority 3)

### 3.1 The rules: what each needs at inference, what it removes, how it fails

"Removes" is a fraction of the entries it acts on. **MEAS-toy** = §5; **MEAS-real** = INHERITED from the sibling artifact `hicap_prior_mask_result.json` (256 REF-C train anchors on val-40, not re-run here, no torch); **EST** = arithmetic shown. **Admissibility** applies the 2026-08-03 bindings: ego inputs are the PI's stated REF-F inputs (2026-09-29); nothing here may consume the *situation classifier's output*.

| # | rule | needs at inference | typically removes | failure mode (prunes the right action) | agents? |
|---|---|---|---|---|---|
| G1 | **Lateral-acceleration bound** `v_mean²·|κ| ≤ L_lat` on the (path, profile) pair | v0, path κ | MEAS-toy: with nuPlan-comfort 4.89, most of the 47.7 % removed | comfort used as *hard* limit removes emergency swerves | no |
| G2 | **Longitudinal bounds** `−L_dec ≤ ρ·v0/T ≤ L_acc(v0)`; no-reverse | v0 | MEAS-toy with G1 as above; MEAS-real (speed-normalised anchors): 14.6 % (physical 6/10/9), 21.9 % (4/8/6), 30.0 % (3/6/4); absolute anchors: 78.4 / 84.6 / 90.8 % | hard stops from high speed; engine-limit misestimate | no |
| G3 | **Initial-acceleration continuity** `|a_prof(0) − ax_fd| ≤ j_max·τ` | `ax_fd` (0.1 s speed difference; noisy) | EST: profile initial acceleration spans about [−8, +3] m/s²; a ±1.5 tube keeps about 27 %, so removes ~73 % of the profile axis if profiles are uniform in that range (FPS-built) | `ax_fd` noise; genuine jerk events | no |
| G4 | **Initial-curvature continuity** `|κ(0) − ψ̇/v0| ≤ δκ` | yaw rate, v0 ≥ ~2 m/s | EST: 10–50 % of paths kept, depending on how peaked the path density is at κ ≈ 0 | ill-conditioned at low speed; yaw-rate noise | no |
| G5 | **Route / goal gate**: `g_str` (v7: FOLLOW_ROUTE, TURN_L/R_FOLLOW_ROUTE, STOP_AT populated) → bitmask over (LAT, LON) cells; **soft log-prior**, hard only for logical impossibilities | a *predicted* strategic token or geometric goal point | EST: ROUTE = straight keeps 3 of 5 live LAT classes: 25–40 % of paths | a wrong token prunes the true class: never hard-mask LANE_KEEP; goal-dropout ≥ 0.5 in training (vocabulary rule R6) | no |
| G6 | **Speed-policy cap** (SPEEDPOLICY / sign-limit → `v_end ≤ cap`) | predicted cap (vision) | EST: 15–30 % of the profile axis when v0 ≈ cap, ~0 otherwise (ACCELERATE is 22.0 % of v7 windows) | misread sign | no |
| G7 | **Retrieval gate**: cells of the expert actions of the k nearest train states in embedding space | cached state embeddings + train arena | HYPOTHESIS: keeps the union of ≤ 5–10 cells; kNN over 406 k × 512 fp16 = 208 MB ≈ 0.8 ms at spec bandwidth (EST) | out-of-distribution states retrieve nothing relevant | no |
| G8 | **Corridor prior** `|y(2 s)| ≤ W/2 + margin` for non-turn classes (constant W = 3.5 m if no lane estimate) | none (constant) or lane estimate | EST: 10–20 % of paths (NUDGE tail) | avoidance, lane change | no |
| G9 | **Hysteresis tube** around the previous plan shifted one tick, Chebyshev radius ε | previous plan | EST: `(2ε/range)^{d_eff}`, with d_eff ≈ 3 and ε = 10 % of range → about 1 %; take 1–5 % | lock-in on a wrong plan → refresh at tactical cadence and on event triggers | no |
| G10 | **Learned gate head** over levels 1–2 with a *calibrated margin* (adaptive width) | state features | MEAS-toy: keeps 1.8–2.9 of 256 cells on average (p95 4–5) at 93–98 % cell recall; 0.75–1.2 % of flat MACs | distribution shift; miscalibration | no |
| G11 | **RSS longitudinal safe distance** `d_min = [v_r ρ + ½a_acc ρ² + (v_r + ρ a_acc)²/(2 a_brake,min) − v_f²/(2 a_brake,max)]₊` (Shalev-Shwartz et al., 1708.06374; formula RECALLED, not re-read) | **lead range and speed** | EST: in following scenes (FOLLOW 3.7 % + BRAKE_TO 19.6 % of v7 windows) 30–60 % of the profile axis | vision range error → phantom braking | **yes** |
| G12 | **TTC floor** against tracked agents | agent tracks | as G11 | as G11 | **yes** |
| G13 | **Scene-type prior** (junction / highway / urban → class priors) | a scene or situation classifier output | not measured | ⛔ **inadmissible as written**: it is the situation classifier's output entering the planner path (2026-08-03 binding). Substitute G7 (conditions on the state itself) or a head shown information-disjoint | no |
| G14 | **Map-free geometry** (lane width from a constant, road-edge from image) | none / image | as G8 | no map exists in PhysicalAI-AV (CLAUDE.md, settled at five probes), so only constants and image cues | no |

**What runs vision-only today:** G1–G10 need only ego state, cached tables, predicted tokens and the previous plan. **G11–G12 need agents:** per the plan (§2.10) they run only with simulator agent states, labelled an oracle-perception envelope test, until a range head exists. The toy's mask and the sibling's real-anchor measurement agree on *ordering* and on the survival level at the tight tier (97.6 % real anchors, 98.4 % toy); the removal *sizes* differ (a toy vocabulary spans the full ρ range; the real anchors are already speed-normalised), so **neither transfers as a number to HiCAP's vocabulary; E-R3 measures it**.

**Two-tier limits (DERIVED from §5.3).** Use **physical** limits (≈ 6 / 10 / 9 m/s²: acceleration / deceleration / lateral) as the *hard* mask (100 % oracle survival in toy and real anchors); use **comfort** limits (nuPlan defaults: max lon accel 2.40, min lon accel −4.05, max |lat accel| 4.89, max |lon jerk| 4.13; "determined empirically from expert trajectories", nuplan-devkit docs) as a **soft log-penalty** in the re-ranker. The nuPlan values are themselves exceeded by experts, so they cannot be a hard mask.

### 3.2 The "simple smart logic": bitset algebra plus a calibrated margin

```
live_P  = AND_r  mask_P[r]            # 1,024 bits: G3-G4 (path side), G5 lat gate, G8, G9 tube
live_V  = AND_r  mask_V[r]            #   256 bits: G2, G3 (profile side), G6, G9 tube
coarse  = score(live_P) , score(live_V)            # class level first: 5 LAT + 7 LON, then members
keep_P  = { p : f_P(p) >= max f_P - delta_P }, |keep_P| <= 64      # delta from split-conformal (§3.4)
keep_V  = { v : f_V(v) >= max f_V - delta_V }, |keep_V| <= 32
cand    = compose(keep_P x keep_V) AND pair_mask(v^2 kappa, [RSS]) ∪ guard_set ∪ prev_plan_shifted
```

The **guard set** (hold-current CTRA, gentle brake, firm brake, max brake in lane, previous plan shifted by one tick; ≈ 8 entries) is injected *after* every mask, so the candidate set is never empty and the recall floor is the Simplex fallback, not zero. Masks are bitsets built once per tactical update from v0 / `ax_fd` / yaw rate; the analytic box test below lets a mask act on a subtree without touching its leaves.

**Hierarchical box mask (MEASURED-toy).** For a node with curvature range `[κ_lo, κ_hi]` and ratio range `[ρ_lo, ρ_hi]`, a lower bound of the demand over the box is closed-form (`min|κ|` and the smallest feasible `v_mean`). Over 400 test states × 4 tiers it **never removed a node containing an alive leaf (0 violations)** and kept 0.5–1.1 % of cells that had no alive leaf (false keeps, harmless).

### 3.3 Feasibility masks must be identical in training and inference

Masking logits before the softmax renormalises over the valid set and gives valid policy gradients (Huang & Ontañón, 2006.14171, snippet-level). Train the level critics **with the same masks** (physical tier) so `p(a|s, feasible)` is what is learned; a train/inference mask mismatch is a silent distribution shift. FeaXDrive (2604.12656) folds curvature feasibility into training for the same reason (snippet-level).

### 3.4 The recall / safety trade-off, formalised

Let `a*` be the oracle action and `S = P(a* ∈ M(x) ∩ B(x))` the pipeline survival (M masks, B beams / gate).

- **Union bound:** `1 − S ≤ Σ_r ε_r + Σ_l ε_l`. Choose per-stage levels `α_r, α_l` with `Σ α ≤ α_total`.
- **Split-conformal per stage.** Nonconformity of a calibration example = the smallest inflation `μ` that keeps `a*` (mask: `r(a*) = max(a_lat/L_lat, a_lon/L_acc or −a_lon/L_dec)`, survive iff `r ≤ 1+μ`) or the score gap of `a*`'s ancestor to the level maximum (beam). With `n` calibration points, `k = ⌈(n+1)(1−α)⌉`, threshold = k-th smallest; marginal coverage ≥ 1 − α (Sadinle et al. 2019, 1609.00451; Angelopoulos & Bates, 2107.07511; SafePath, 2505.09427 applies it to trajectory choice, snippet-level).
- **The guarantee is marginal.** Coverage of a fixed-threshold set over an exchangeable draw has sd ≈ `sqrt(α(1−α)/n)`: n = 200 → 1.5 pts, n = 400 → 1.1, n = 1,500 → 0.56 (DERIVED). **Real windows are not exchangeable within an episode**: calibrate on **episode blocks** (the programme's cluster principle). Sibling real-window result (INHERITED, `hicap_conformal_cells_result.json`, val-40, 500 episode-disjoint splits of 14 / 12 / 14 episodes, kinematic-proxy cells): nominal 0.95 → mean coverage 0.957, **sd over splits 0.046, 5th-percentile split 0.870, 33.6 % of splits below nominal**, 33.9 % of cells pruned.
- **Class-conditional (Mondrian) calibration** is needed wherever the error cost is class-dependent. MEASURED-toy at μ = 0 (§5.3): coverage cruise 99.4 %, accelerate 96.0 %, turn 94.3 %, **stop 93.3 %**; speed tercile > 13.6 m/s 96.6 %. A per-class 5 % tail needs ≥ ~1,900 calibration windows *per class* for ±0.5 pt (DERIVED), more than the val-40 set supplies.
- **Price of the guarantee (MEASURED-toy, T0 tier):** raising the target from 98.3 % to 99.0 % needs μ̂ = 0.184 (limits ×1.18) and leaves 58.3 % of the vocabulary instead of 52.3 %.

### 3.5 Admissibility check for the two binding rules (ESCALATION, §6.5)

- **Goal must not carry the situation classifier's output.** In HiCAP the tactical posterior gating the candidates is the *planner's own level-2 marginal*, not the `sitclf` head. The `g_str` gate and the situation classifier read the **same frozen trunk** `z`; that is a shared-trunk question the binding text asks to be *stated and justified*. Position: the gate consumes only planner-level posteriors and predicted tokens, never a `sitclf` output; a **goal-swap causality test** and a `g_str`-null ablation are pre-registered (R7). **PI to confirm** before any G5 result is quoted.
- **Labels may use ego, inference is vision-only.** G1–G4 use **ego state at inference**. That is the PI's stated REF-F input (2026-09-29) but is *not* admissible in a vision-only classifier head. They are therefore mask rules on the candidate set, never features of a classifier.

---

## 4. Using the pruned candidates inside the backbone (priority 4)

### 4.1 Multi-cadence cascade (design; counts are DERIVED, FLOPs ESTIMATED)

| stage | cadence | set in → out | operation | MACs | notes |
|---|---|---|---|---|---|
| S0 ego masks | tick | 1,024 + 256 bits | bitset AND (G2–G4, G6, G9) | ~0 | v0, `ax_fd`, yaw rate |
| S1 factored coarse | **tactical (5 ticks)** | 1,280 → (k_P ≤ 64, k_V ≤ 32), typically ~20 × 10 | class level (5 LAT + 7 LON), then members; calibrated-margin width | 0.66 M per 5 ticks (1,280 × 512) | cached embeddings; strategic gate G5 refreshed every 20 ticks |
| S2 compose | tick | ≈ 200 → ≤ 256 | compose pairs, pair mask `v²κ` (+ RSS in sim), add guard set + previous plan | ~0 | survivors cached for the 5-tick window, re-masked each tick with the new v0 |
| S3 late interaction | tick | ≤ 256 → 32 | ColBERT-style MaxSim: n_w = 5 waypoint tokens × M = 72 scene tokens | 47.2 M | no new layers beyond projections |
| S4 cross-attention re-rank | tick | 32 (or 16, 8) → 1; top-8 to System 2 when the margin is small | 3 layers, d = 512, MoE FFN keyed on (LAT, LON) | K = 32: 476 M · K = 16: 294 M · K = 8: 203 M | 12.6 M parameters |

**Per tick, K = 16:** 0.294 + 0.047 + 0.0001 = **0.34 GMAC ≈ 0.68 GFLOP** (M = 72 scene tokens = 8 latent + 64 pooled patch tokens, as in the plan's state head). Bandwidth: re-ranker weights 25 MB = 0.09 ms, coarse tables 2.6 MB = 0.01 ms (spec bandwidth). **Latency will be launch- and memory-bound, not FLOP-bound; H-F5-style Thor measurement is required.**

### 4.2 Token counts and FLOPs for the re-ranker (script `design_arithmetic`, d = 512, M = 72, 3 layers)

Per layer `= K·14·d² + 2·M·d² + 2·K²·d + 2·K·M·d` (self-attention 4, cross-attention Q/O 2, FFN 8 units of d² per candidate; scene K/V projection shared).

| K | sequence seen by the decoder | GMAC (3 layers) | GFLOP | share of cost that is the K-independent scene K/V projection | Δ vs previous K |
|---|---|---|---|---|---|
| 8 | 8 candidate queries over 72 scene KV (80 tokens) | 0.203 | 0.407 | 56 % | — |
| 16 | 16 over 72 (88) | 0.294 | 0.587 | 39 % | +45 % |
| 32 | 32 over 72 (104) | 0.476 | 0.952 | 24 % | +62 % |

Read: at K ≤ 16 the scene projection dominates, so **going from 8 to 16 candidates costs about 0.09 GMAC**; SMs saturate near batch 8 on Thor (MEASURED on another model, CLAUDE.md), so K = 8 → 16 likely costs little wall-clock (**HYPOTHESIS**, needs H-F5). Quality side (MEASURED-toy, §5.9): the re-rank gain is almost all in by K = 8 (hit 0.827 → 0.940), K = 16 adds +0.9 pt, K = 32 +0.3 pt; oracle endpoint error of the candidate set keeps falling (0.70 → 0.36 → 0.25 → 0.17 m). DriveSuprim's oracle (F) shows the same shape (top-4 94.5, top-16 96.1, top-256 98.7 PDMS). **Recommendation: train at K = 32, deploy K = 16, escalate to K = 32 when the S1 margin is small.**

### 4.3 Injection designs

| design | how candidates enter | cost | when |
|---|---|---|---|
| **D1 (recommended): light decoder re-ranker** | candidate token = `[action embedding (cached) ⊕ cell embedding ⊕ kinematic descriptors]` as **queries**, cross-attending to the state head's 72 scene tokens (DriveSuprim, GTRS, WoTE pattern); per-candidate sub-score heads (Hydra-MDP: sub-scores beat a scalar) | table above | every tick |
| **D2: MoE keyed on the tactical token** | FFN expert chosen by the candidate's (LAT, LON) class: **deterministic routing**, 1 shared + 6 routed experts as in DriveMoE (which learns its top-3 router) | same FLOPs as dense (top-1), parameters × experts | every tick; needs a shared expert because rare cells train few samples |
| **D3: tactical-cell tokens into the frozen VLM** | K = 8 cell tokens (+ their top pair tokens) appended after the cached scene prefix (KV reuse), scores read from the last layer | one **weight pass** through the LLM, independent of K ≤ ~64: 8B bf16 **58.6 ms**, fp8 29.3, fp4 14.7 (weights ÷ 273 GB/s, ESTIMATED); for comparison the 87 M encoder: 0.64 ms | **tactical cadence or slower only** (÷ 5 ticks = 5.9 – 11.7 ms per tick) |
| **D4: late interaction (ColBERT)** | multi-vector candidate vs scene MaxSim (2004.12832); no interaction layers | 47 M MAC for 256 candidates | middle stage between coarse retrieval and D1 |
| **D5: registers** | DrivoR (2601.05083) compresses multi-camera features into camera-aware register tokens, then a trajectory decoder and a scoring decoder with sub-scores | M can be small | how to keep M = 72 honest |
| **D6: candidates as KV (scene queries attend to them)** | candidate-conditioned world-model rollout per candidate (WoTE, 2504.01941: 256 anchors, BEV rollout + reward head, 81.0 → 83.2 → 85.6 PDMS, F) | K × a scene-model pass; the review's estimate for world-model scoring of a full fan is ≈ 22× over the Thor budget (INHERITED, ESTIMATED) | **System 2 only**, K = 8 (fills Thor's batch saturation): the reduced set is what makes consequence-aware scoring affordable at all |

**Conditional computation keyed on the tactical cell:** (a) `k_P, k_V` and K come from the calibrated margin, so **compute follows uncertainty**; (b) RISK / ODD tokens (vocabulary v1) raise or cap the width; (c) experts by cell (D2). The dot-product bottleneck (sign-rank ≤ d, Weller et al. 2508.21038) is why the *interaction* terms (collision against the specific agents) must live in D1, not in the cached tower.

### 4.4 Memory-bound arithmetic (ESTIMATED, `design_arithmetic.flat_score_traffic`, 273 GB/s spec)

| N entries (d = 512) | fp16 MB | ms | int8 MB | ms |
|---|---|---|---|---|
| 65,536 | 67 | 0.25 | 34 | 0.12 |
| 262,144 | 268 | **0.98** | 134 | 0.49 |
| 1,048,576 | 1,074 | 3.93 | 537 | 1.97 |
| 4,194,304 | 4,295 | 15.7 | 2,147 | 7.9 |
| 16,777,216 | 17,180 | 62.9 | 8,590 | 31.5 |

Peak, single reader; the backbone competes for the same bandwidth. CPU illustration only (not Thor, non-deterministic, this sandbox): flat 65,536 × 512 matvec 5.9–15.1 ms vs tree (3,16) 0.58–2.3 ms across four runs (staged JSON: 6.1 vs 0.60 ms), i.e. 6–10× faster against a 15.6× MAC reduction.

### 4.5 Cadence and staleness

Strategic every 20 ticks (2 s), tactical every 5 (0.5 s), operative every tick. The S1 survivor set is **cached for 5 ticks and re-masked each tick with the new v0**; the guard set and the previous plan guarantee a valid candidate even if the cache is stale. Staleness is an empirical risk (R6). A margin-collapse trigger (top-2 gap below θ) forces an early S1 refresh.

---

## 5. Toy: hierarchical retrieval over a product vocabulary (priority 5) — MEASURED-toy

`rb_hier_retrieval_toy.py`, numpy only, **≈ 2 min (129–145 s per run)**, deterministic (a second full run reproduced every number except the `timing_*` block exactly). Every number below is a property of a **synthetic world**.

### 5.1 World and design

- **Vocabulary:** curvature axis κ (`nk` bins) × speed-ratio axis ρ = v_end/v0 − 1 (`nr` bins) over 2 s. **Config A: 8 strategic × 32 tactical × 256 leaves = 65,536** (`nk` = 512, `nr` = 128); **config B (v7-shaped: 8 LAT × 8 LON cells): 8 × 64 × 256 = 131,072**. Stored in depth-first order. Strategic = coarse κ bin; tactical = (LAT sub-bin × LON coarse bin); the cell holding κ = 0, ρ = 0 is the "cruise cell".
- **States:** v0 lognormal, median 10.3 m/s; 1–3 expert **modes** (cruise 62 %, slow 12 %, accelerate 8 %, turn 9 %, stop 5 %, nudge 4 %; drivers do not ask for the impossible); per-state true limits; `p(a|s)` **exact over all N leaves** ∝ mode bumps × feasibility. The prior `p(a)` is the train-sample marginal (1,000 states), so it is cruise-dominated (cruise-cell mass 0.586, top leaf 0.0092).
- **Critics:** `f_l = log P_l(n|s) − log P_l(n) + ε` (the InfoNCE optimum with level-wide negatives) + iid noise σ = 0.5; correction adds `β·log P_l(n)`. Bounded variant `c·tanh(PMI/c)`, c = 3.
- **Metrics:** hit rate `E[p(pick|s)]` (the convention of `reff_pmi_toy.py`), reported as a **fraction of the exact-MAP hit rate** (`hit_frac`); regret in nats; tactical-cell agreement with the marginal-MAP cell; cell recall = probability mass of the expert distribution inside the kept cells (= P(sampled expert's cell survives), exact). MACs = node scores evaluated × 512 (a count, not a latency).
- **Sizes and estimators:** n_test = 400 (A), 200 (B); calibration 1,200; conformal test 1,500. States are iid, so plain estimators apply (real windows are episode-clustered). "± SE" = sd/√n over states; "± CI95" on paired differences = 1.96·sd(Δ)/√n; E8 / E9 SEs = 300-resample bootstrap of the ratio `E[p(pick)]/E[p(MAP)]`. These are **not** the episode-cluster bootstrap the programme requires for real data.
- **Invariants asserted in the run:** the beam with all cells equals the flat pick; rows of `p(·|s)` sum to 1; the analytic box mask has 0 violations.
- **Reference points:** exact MAP hit 0.0207 (frac 1.000); always the prior's top leaf 0.488; **flat critic 0.0171 (frac 0.827)**, regret 0.200 nats, tactical agreement 0.91 (MAP 0.96).

### 5.2 (i) Flat vs tree vs factorised (mask off, β = 1, σ = 0.5)

Flat cost = 33,554,432 MAC (65,536 × 512).

| retrieval | MACs | vs flat | agrees with flat pick | cell recall (mass) | hit_frac | paired Δ E[p] vs flat (±CI95) |
|---|---|---|---|---|---|---|
| tree (1, 4) | 544,768 | **61.6×** | 0.970 | 0.958 ± 0.007 | 0.823 | −8.5e-5 ± 1.0e-4 (n.s.) |
| tree (2, 8) | 1,085,440 | **30.9×** | 0.990 | 0.990 ± 0.004 | 0.827 | 0 |
| tree (3, 16) | 2,150,400 | 15.6× | 0.9975 | 0.996 ± 0.002 | 0.827 | 0 |
| tree (4, 32) | 4,263,936 | 7.9× | 1.000 | 0.999 ± 0.001 | 0.827 | 0 |
| tree (8, 64) | 8,523,776 | 3.9× | 1.000 | 1.000 | 0.827 | 0 |
| factorised 8 × 4 (32 composed) | 344,064 | 97.5× | 0.565 | n/a | 0.822 | −1.0e-4 ± 3.2e-4 (n.s.) |
| factorised 20 × 10 (200) | 430,080 | 78.0× | 0.953 | n/a | 0.826 | −2.3e-5 ± 3.2e-5 (n.s.) |
| factorised 32 × 16 (512) | 589,824 | 56.9× | 0.983 | n/a | 0.827 | 0 |

**Critic-noise sweep, tree (3,16) vs flat:** hit_frac 0.914 / 0.693 / 0.558 at σ = 0.25 / 1 / 2 (flat identical to tree), agreement 1.000 / 0.993 / 0.995. **The ceiling is set by critic noise; the tree adds no loss at (3,16).** Config B (131,072 leaves) reproduces the pattern: (1,4) agrees 0.980 at 561,152 MACs, (2,8) agrees 1.000, cell recall 0.966 / 0.993 / 0.998 for (1,4) / (2,8) / (3,16).

### 5.3 (ii, iii) Prior mask, conservative inflation and conformal calibration (`mask_*`, `E2`, `E3`)

| tier (acc / dec / lat, m/s²) | leaves kept | removed | oracle survival (mass, ± SE) | cells with any alive leaf | hit_frac |
|---|---|---|---|---|---|
| T0 nuPlan comfort (2.40 / 4.05 / 4.89) | 0.523 | 47.7 % | 0.9829 ± 0.0049 | 58.1 % | 0.821 (Δ n.s.) |
| T1 tight (3 / 6 / 4) | 0.551 | 44.9 % | 0.9841 ± 0.0054 | 60.3 % | 0.760 (Δ E[p] −1.4e-3 ± 1.3e-3) |
| T2 comfort-plus (4 / 8 / 6) | 0.655 | 34.5 % | 1.0000 | 70.0 % | 0.827 |
| T3 physical (6 / 10 / 9) | 0.746 | 25.5 % | 1.0000 | 78.3 % | 0.827 |

Config B: removed 51.6 / 50.0 / 39.4 / 29.5 %, survival 0.975 / 0.993 / 1.000 / 1.000. The **same three tiers** were measured on real REF-C anchors by the sibling stream: 30.0 / 21.9 / 14.6 % removed (speed-normalised), 97.6 / 100 / 100 % of true paths pass (INHERITED). **Inside the tree**, the mask cut flat MACs by 47.7 % (17.6 M) but the (3,16) tree's by only 12.6 % (1.88 M), because the beam had already avoided most infeasible subtrees; survival after mask **and** beam was 0.9793 vs 0.9829 mask-only (beam loss 0.36 pt).

**Conservative inflation** `μ` (limits × (1+μ)), exact mass:

| μ | 0 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| T0 survival / kept | 0.9828 / 0.523 | 0.9910 / 0.557 | 0.9966 / 0.602 | 0.9997 / 0.662 | 1.0000 / 0.743 |
| T1 survival / kept | 0.9841 / 0.551 | 0.9979 / 0.579 | 0.9997 / 0.615 | 1.0000 / 0.663 | 1.0000 / 0.730 |

**Split-conformal calibration (T0, n_cal = 1,200, n_test = 1,500 sampled experts):**

| α (target) | q̂ | μ̂ | test coverage | vocabulary kept |
|---|---|---|---|---|
| 0.10, 0.05 | 1.000 | 0 (no inflation needed) | 0.9833 | 0.523 |
| 0.01 | 1.184 | 0.184 | 0.9940 | 0.583 |

Marginal coverage hides the classes that matter: **at μ = 0** coverage by dominant mode is cruise 0.994 (n = 946), slow 0.989 (184), accelerate 0.960 (101), **turn 0.943 (122), stop 0.933 (90)**, nudge 1.000 (57); by speed tercile 1.000 / 0.984 / **0.966** (v0 > 13.6 m/s). (An earlier run with n_cal = 600 gave 0.925 in the top tercile and per-tercile calibration with 200 points did not repair it: the calibration size, not the method, was the limit.)

### 5.4 Adaptive calibrated-margin beam vs fixed beams (`E5`)

Per-level margins from split-conformal on the ancestor's score gap (`α_1 = α_2 = α`); union-bound floor `1 − 2α`.

| rule | MACs (vs flat) | cell recall (mass ± SE) | union floor | hit_frac | cells kept p95 |
|---|---|---|---|---|---|
| adaptive α = 0.05 (δ = 0, 2.22) | 251,494 (**133×**) | 0.927 ± 0.006 | 0.90 | 0.827 | 4 |
| adaptive α = 0.02 (0.56, 3.29) | 335,667 (100×) | 0.966 ± 0.003 | 0.96 | 0.831 | 5 |
| adaptive α = 0.01 (1.25, 4.10) | 403,128 (**83×**) | 0.981 ± 0.002 | 0.98 | 0.831 | 5 |
| fixed (2, 8) | 1,085,440 (31×) | 0.990 ± 0.004 | — | 0.827 | 8 |

At α = 0.01 the adaptive rule is **2.7× cheaper than fixed (2, 8) for 0.9 pt less recall**; all three met their union floor. Level-2 calibration uses the global maximum over all cells, so it is conservative at test time (children of kept parents only).

### 5.5 (iv-a) Level-wise prior correction on/off, cruise-dominated prior (`E4`)

| critic | β | hit_frac | cruise-cell share (MAP 0.66; truth mass 0.586) | tactical agree | mean regret (nats) |
|---|---|---|---|---|---|
| exact PMI + noise, flat | 0 | **0.466** | **0.4625** | 0.665 | 1.31 |
| exact PMI + noise, flat | 1 | 0.829 | 0.6525 | 0.910 | 0.205 |
| exact PMI, **tree (3,16)** | 0 / 1 | 0.466 / 0.829 (identical) | same | same | same |
| bounded (tanh, c = 3) | 0 | 0.369 | 0.465 | 0.678 | 1.59 |
| bounded | 0.75 | 0.670 | 0.6625 | 0.853 | **0.713** |
| bounded | 1 | 0.672 | **0.765** (over-corrected) | 0.838 | **2.05** |

Bounded-critic fit `E[p(pick)]` by β (600 calibration states): 0.0067 (0), 0.0108 (0.25), 0.0124 (0.5), 0.0132 (0.75), **0.0134 (1.0)**, 0.0130 (1.25): a flat optimum over 0.75–1.25, so β = 0.75 and 1 are **not distinguishable by hit rate but differ 2.9× in mean regret** (over-selection of cruise produces rare, costly errors): **choose β by regret and tactical agreement, not by hit rate.** The tree gives the same picks as the flat scorer (agreement 0.9975–1.000) with cell recall 0.991–0.996 under every critic variant.

**Sibling-only critic training** (per-(state, parent) offsets σ_c added at levels 2 and 3), tree (3,16): tactical agreement 0.910 → **0.873 / 0.783 / 0.620** and cruise share 0.6525 → 0.6425 / 0.590 / 0.4875 at σ_c = 0.5 / 1 / 2; regret 0.205 → 0.239 / 0.316 / 0.745. The flat scorer is hurt identically (its leaf critic is sibling-trained in this variant), so this is a **training-recipe** effect, not a tree effect. (Hit fraction is non-monotone here, 0.795 / 0.807 / 0.664: it weights a few peaky states heavily and is the noisiest metric in this file; agreement and regret are the reliable ones.)

### 5.6 (iv-b) Does the PMI bias grow with vocabulary size? (`E9`)

| N | hit_frac β = 0 (± SE) | hit_frac β = 1 | cruise share β = 0 / 1 / MAP |
|---|---|---|---|
| 4,096 | 0.402 ± 0.023 | 0.869 ± 0.012 | 0.490 / 0.717 / 0.737 |
| 16,384 | 0.357 ± 0.021 | 0.812 ± 0.010 | 0.480 / 0.727 / 0.743 |
| 65,536 | 0.383 ± 0.021 | 0.830 ± 0.008 | 0.503 / 0.730 / 0.747 |
| 131,072 | 0.360 ± 0.018 | 0.845 ± 0.010 | 0.480 / 0.737 / 0.747 |

**No trend with N** (spread ≤ 2 SE, not monotone): in this toy the bias is a property of the prior's skew, not of vocabulary size. (300 states per row; the toy's bias is far larger than the 16-action `reff_pmi_toy.py` because this prior is far more skewed.)

### 5.7 Node-critic type: marginal vs max-pool vs mean-pool (`E7`; bounded critic, β = 0.75)

Cell recall (mass) at beams (1,4) / (2,8) / (3,16) / (4,32):

| node critic | (1,4) | (2,8) | (3,16) | (4,32) | agrees with flat at (3,16) |
|---|---|---|---|---|---|
| **marginal** (level-wise softmax target) | 0.952 | 0.988 | **0.996** | 0.999 | 0.9975 |
| max over children (ideal heap) | 0.927 | 0.979 | 0.995 | 0.998 | 0.9975 |
| **mean over children (centroid index)** | 0.853 | 0.912 | **0.944** | 0.972 | 0.9625 |

The centroid index loses **5 points of recall at (3,16) and 2.6 at (4,32)**, and its mean regret is 2.0 vs 0.71 at (3,16). Top-1 hit is nearly unaffected (0.662 vs 0.665): **the damage is in the multi-modal candidate set, which is what the backbone consumes.**

### 5.8 Estimating the prior from sparse counts (`E8`; hard counts from sampled experts; critic built with the true prior)

hit_frac (± bootstrap SE), n_test = 600, paired noise: true prior **0.832 ± 0.009**; no correction **0.433 ± 0.020**.

| expert samples (per leaf) | flat smoothed histogram (ε = 1e-3) | hierarchical backoff |
|---|---|---|
| 100 (0.0015) | 0.632 ± 0.016 | 0.640 ± 0.015 |
| 300 (0.0046) | 0.640 ± 0.015 | 0.655 ± 0.014 |
| 1,000 (0.0153) | 0.710 ± 0.015 | 0.721 ± 0.014 |
| 1,500 (0.0229) | 0.690 ± 0.023 | 0.731 ± 0.014 |

An estimated prior recovers **50 % (n = 100) to 75 % (n = 1,500)** of the true-prior gain. The hierarchical backoff is ≥ the flat histogram at all four sizes by +0.008 to +0.041, each within 1–2 SE: **direction consistent, not established.** The real corpus has ≈ 1.5 samples per leaf at 262,144 leaves; the toy's densest row is 65× sparser, so the real regime is *more* favourable than the toy but with real (non-smooth) counts.

### 5.9 Re-rank by K after the tree (`E6`; a low-noise critic that includes the prior stands in for the cross-attention re-ranker)

| K | hit_frac | mean regret | oracle endpoint error of the candidate set (m; proxy) |
|---|---|---|---|
| 1 (no re-rank) | 0.827 | 0.200 | 0.703 |
| 8 | **0.940** | 0.069 | 0.364 |
| 16 | 0.949 | 0.065 | 0.246 |
| 32 | 0.953 | 0.056 | 0.165 |

### 5.10 What the toy does and does not show

**It shows (mechanism, MEASURED-toy):** the cost formulas hold; a marginal-critic tree loses almost nothing at (2,8) and calibrated adaptive margins cut MACs by ~100× at 93–98 % recall; hierarchical **marginal** node training beats a centroid index for candidate-set recall; level-wide (not sibling-only) negatives matter; the prior correction is mandatory and tree-invariant; ego-kinematic masks and their hierarchical box test are exact-safe; marginal conformal coverage hides rare classes; the re-rank gain saturates near K = 8–16.

**It does not show:** (1) any real accuracy: critics are the exact PMI plus iid noise, so "tree ≈ flat" is partly **by construction**; real recall is decided by level-critic quality (R4, R5); (2) coverage / quantisation: experts are sampled on the grid, so nothing here measures oracle ADE against N (that is E-R2 on real data); (3) time, agents, perception or closed loop; (4) the transfer of any size: the toy's 2-D (κ, ρ) space has no label noise; (5) latency: MAC counts are counts, and the CPU timing block is illustrative and non-deterministic; (6) episode clustering: toy states are iid; (7) a robust hit metric: `hit_frac` is dominated by peaky states (turn, stop), which is why regret and agreement are reported next to it.

---

## 6. Recommendation: vocabulary, cascade, compute, risks (priority 6)

### 6.1 Vocabulary structure (sizes are proposals to be fixed in a pre-registration amendment)

| element | proposal | why |
|---|---|---|
| **path factor** 𝒫 | K_P = 1,024 geometric paths on an arc-length grid, **partitioned by `a_tac.lat`**: the v7 vocabulary declares 8 LAT tokens of which **5 are live** (LANE_KEEP 64.8 %, NUDGE_L, NUDGE_R, TURN_L, TURN_R; LANE_CHANGE_L/R and ABORT_LC unreachable), so allocate per-class path budgets by farthest-point sampling *within class* with a floor per class (k-means starves rare classes; INHERITED, the plan's §2.4) | product with 𝒱; class = tree level |
| **profile factor** 𝒱 | K_V = 256 speed-ratio profiles (residual over CTRA), partitioned by `a_tac.lon`: 8 declared, **7 live** (FOLLOW 3.7 %, CRUISE 27.3 %, BRAKE_TO 19.6 %, CREEP, HOLD, ADAPT_SPEED_FOR_CURVE 21.6 %, ACCELERATE 22.0 %) | speed kinematics depend on the profile alone |
| **entries** | **N = 1,024 × 256 = 262,144** (= SparseDriveV2's densest; 32 × the 8,192 of DriveSuprim / Hydra-MDP++); live tactical cells 5 × 7 = **35** (v7 census via P1, INHERITED) | published precedent; flat scoring still affordable (§4.4) |
| **growth path** | 2,048 × 512 ≈ 1.05 M only if E-R2 shows the oracle still falling; flat cost 3.9 ms fp16 / 2.0 ms int8 | growth is limited by prior estimation and training, not inference |
| **strategic gate** | `g_str` (v7: 8 tokens, 4 populated) as a **soft log-prior table** `log p(cell | g_str)` from label co-occurrence counts (hierarchical backoff) + hard zeros only for logical impossibilities; refreshed every 20 ticks | not a strict parent (a DAG): a cell is live if any kept strategic token admits it |
| **leaf refinement** | small residual head on the chosen entry (the plan's `L_ref`) | escapes quantisation; DiffusionDrive-style small-K corner |
| **label consistency** | every path and profile is labelled by the **same geometric classifier** that mints the training labels (`refb_labels` / v7 emitter thresholds) | otherwise the level-2 index and the labels disagree |
| **band mismatch (OPEN, UNVERIFIED)** | v7 tactical tokens describe the **[2, 6] s** band and the operative band [0, 2] s has no tokens (P1 §1.2), while REF-F candidates are 2 s trajectories. Either candidates become 6 s objects (tactical cell fixes [2,6], leaf refines [0,2]) or the LAT/LON class of a 2 s candidate is minted over 0–2 s | decision needed from the vocabulary stream / PI |

### 6.2 Cascade and compute per tick (ESTIMATED, §4.1)

S0 masks (~0) → S1 factored coarse every 5 ticks (0.66 MMAC) → S2 compose ≈ 200 + guard (~0) → S3 MaxSim 47 MMAC → S4 cross-attention re-rank K = 16: 294 MMAC (K = 8: 203, K = 32: 476) → top-1, top-8 to System 2 on a small margin. **≈ 0.34 GMAC (0.68 GFLOP) per tick**; tables ≤ 3 MB; re-ranker weights 25 MB. The flat fallback (262,144 × 512 fp16, 268 MB, 0.98 ms at spec) stays available and is the **reference arm** for R4.

### 6.3 Training recipe (HYPOTHESIS until run)

Level-wise softmax over each level's nodes with soft ancestor targets and level-wide negatives (§2.4–2.5); ε-mask extended to nodes; physical-tier masks applied to logits in training (§3.3); prior from hierarchical backoff on the train split only (parity key asserted); per-family sub-score heads on the re-ranker; the four metric families reported per family with paired **episode-cluster bootstrap** CIs (CLAUDE.md), the tactical family natively as cell confusion and cell-level top-b recall.

### 6.4 Ranked risks, each with the cheapest discriminating experiment (both outcomes committed in advance)

| # | risk | experiment (cost) | outcome A → | outcome B → |
|---|---|---|---|---|
| **R1** | Level critics cannot resolve the tactical cell from frozen features, so the tree adds nothing over priors | **E-R1**: after the embedding cache exists, level-wise linear / MLP probes and the VINN k-NN pre-gate at cell level on val-600, paired episode-cluster CI vs the marginal-prior baseline on **non-steady** windows (0 GPU beyond the cache) | cell recall@b₂ = 16 ≥ 0.99 and CI excludes 0: build the tree | recall plateau < 0.95: the frozen backbone carries no decision signal at level 2; switch backbone before any head work (the plan's E-F0 logic); flat scoring stays |
| **R2** | Coverage saturates early, so 262,144 entries buy nothing | **E-R2** (CPU): build the product vocabulary at N ∈ {2¹⁰, 2¹², 2¹⁴, 2¹⁶, 2¹⁸} from **parity-train only**, oracle ADE@2s on val-40 (full-set mean + episode-cluster CI), matched-N **ratios**, no exponent | 4× entries (2¹⁴ → 2¹⁶) still cut oracle ADE ≥ 15 %: keep 262,144, consider 1.05 M | < 5 % per 4× beyond 2¹²: stop at ≤ 16 k leaves; the giant vocabulary is a prior structure, not a size |
| **R3** | Pruning removes the right action in rare classes | **E-R3**: split-conformal on an **episode-blocked** train hold-out per tactical class; report survival per class and kept fraction on val-40 | survival ≥ 99 % in every class with kept ≤ 60 %: adopt the hard physical mask | any class < 97 %: demote that rule to a soft penalty; keep physical limits only |
| **R4** | Beam mis-prunes (heap violation, sibling offsets, mean-pooled index) | **E-R4**: same trunk, three level-critic recipes (marginal-softmax, centroid, sibling-only); cell recall@b₂ and agreement with the flat 262,144 arm | marginal recipe within 1 pt of flat at b₂ ≤ 16: ship the tree | gap > 3 pt: widen with the calibrated margin, or ship flat (≈ 1 ms) and keep the tree only for backbone reduction |
| **R5** | Prior estimated from ≈ 1.5 samples per leaf is too noisy | **E-R5**: 50 / 50 episode split; flat vs hierarchical-backoff prior vs vocabulary-softmax (which needs no prior); selected-class KL and steady-window speed MAE, paired | backoff beats flat by ≥ 1.5 SE: adopt | no difference: use the softmax default (plan §2.5b) and drop the prior table |
| **R6** | 5-tick survivor caching is stale at events | **E-R6** (0 GPU, logged windows): survival vs cache age 0 … 4 ticks | age-4 survival ≥ age-0 − 0.5 pt: keep 5-tick cadence | > 2 pt loss: cadence 2–3 ticks and/or a margin-collapse trigger |
| **R7** | Shared-trunk back door (binding rules): `g_str`, tactical gate and `sitclf` read the same `z` | **E-R7**: (a) goal-swap causality test, (b) `g_str`-null ablation, (c) pruning with `sitclf` output detached and asserted absent | pick ADE unchanged within CI when `g_str` is nulled and the swap changes the output: disjoint, admissible | large drop or swap has no effect: attribution lost (the `--v2` failure class): remove the gate or state the leak |
| **R8** | Factorisation loses the path × profile coupling | **E-R8** (real val): oracle-in-composed(20 × 10) vs oracle-in-top-N flat | ≤ 5 % relative oracle-ADE loss: keep 20 × 10 | larger: k ↑ or add a joint cell-level stage |
| **R9** | Vision-only inference has no agents, so G11–G12 are unavailable | none needed; report as an oracle-perception envelope test only | — | — |
| **R10** | Band mismatch (§6.1) makes the tactical level ill-defined for 2 s candidates | **E-R10** (0 GPU): mint LAT/LON for candidates over 0–2 s vs [2, 6] s and compare cell agreement with the v7 labels on val-40 | agreement ≥ 0.9: 2 s candidates suffice | < 0.7: 6 s candidates required |

### 6.5 Escalations (integration and decisions; not written into a README)

1. **PI decision:** may the *planner's own* tactical posterior and a *predicted* `g_str` gate the candidate set, given that the shared-trunk `sitclf` head exists (§3.5)? Until answered, G5 and G10 results are labelled "admissibility pending" and E-R7 runs first.
2. **Vocabulary stream / PI:** the band mismatch (§6.1, R10) must be settled before any candidate-side token is minted.
3. **Instrument work items (0 GPU):** E-R2, E-R3, E-R6, E-R8, E-R10 need only the committed val windows plus the parity-train egomotion; they can start today.

---

## 7. Implications for HiCAP (≤ 12)

1. Use a **factored product vocabulary (path × speed profile), partitioned by the v7 LAT / LON tokens**: 1,024 × 256 = 262,144 entries first; SparseDriveV2 is the published 2026 precedent for exactly this structure and size (1,280 tokens → 192 → 200).
2. **Tree for reduction and structure, not for speed:** flat 262,144-entry scoring is ≈ 1 ms at Thor spec bandwidth; the tree's job is the small set the backbone reasons over and level-wise priors.
3. **Train level critics as marginals with level-wide negatives** (level-wise softmax); do not build a mean-pooled centroid index; sibling-only InfoNCE leaves cross-parent offsets.
4. **The log-prior correction is mandatory** (hit 0.47 → 0.83 in the toy) and must be **estimated hierarchically**; choose β by regret and tactical agreement, not hit rate.
5. **Encode priors as bitset masks + one soft strategic table**: ego-only rules (G1–G4, G6, G8–G9) run vision-only; agent rules (G11–G12) only in sim oracle-perception mode.
6. **Hard mask = physical limits; comfort limits = soft penalty.** nuPlan comfort thresholds are exceeded by experts, so they cannot be hard: as a hard mask they cost 1.7 % of oracle mass and ~7 % for stop / turn.
7. **Calibrate every threshold by split-conformal on episode blocks, per tactical class** (n ≳ 1,900 per class for ±0.5 pt); marginal coverage is not a safety statement.
8. **Adaptive calibrated-margin width** replaces fixed beams (toy: 83–133× fewer MACs at 93–98 % recall) and makes compute follow uncertainty.
9. **Cascade with a guard set:** masks → factored coarse (tactical cadence) → compose → MaxSim → cross-attention re-rank; **train K = 32, deploy K = 16**; the guard set makes the candidate set never empty.
10. **Cadence:** strategic 20 ticks, tactical 5 (cache survivors, re-mask each tick), operative every tick; a frozen 8B VLM can only be touched at tactical cadence or slower (58.6 ms weight pass in bf16).
11. **Per-family evaluation is native:** cell confusion and cell recall give the tactical family; sub-score heads give longitudinal / lateral; every CI paired and episode-clustered.
12. **Three open decisions need the PI:** the admissibility of the tactical / strategic gate (shared trunk), the [2,6] s vs 2 s band mismatch, and whether to grow past 262,144 (gated on E-R2).

---

## 8. Deliverable manifest and citation status

| artifact | where it lives | notes |
|---|---|---|
| `RB_candidate_space.md` | repo: `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RB_candidate_space.md` | staged; only in this repo copy (no pod) |
| `rb_hier_retrieval_toy.py` | repo: same folder | staged; numpy only, ≈ 2 min |
| `rb_hier_retrieval_toy_result.json` | repo: same folder | staged; produced by the script; only `timing_cpu_numpy_nondeterministic` and `total_runtime_s_nondeterministic` vary between runs |
| SparseDriveV2 clone @696ef77 | scratch only (`/tmp/claude-0/.../scratchpad/SparseDriveV2`) | **not staged, single location, disposable**; the facts used are quoted above with file names |
| `__pycache__` | removed | the sibling stream's `.pyc` was left untouched |

**Integration needed:** none blocking; the three PI decisions in §6.5 gate the pre-registration text, not this file.

**Citation status (this session).** CONFIRMED = id and title seen in a search result; C = read in code; F = inherited from the F-R file; UNVERIFIED = snippet only or figures not read.

| item | id | status |
|---|---|---|
| SparseDriveV2 | 2603.29163 (ECCV 2026) | CONFIRMED + C; per-size scaling table **not read** (arXiv blocked) |
| GTRS | 2506.06664 | CONFIRMED; README sizes via fetch summariser; benchmark split UNVERIFIED |
| DriveSuprim | 2506.06659 | CONFIRMED (sizes 8192 / 256, 93.5 / 87.1 in snippet); oracle top-K numbers F |
| Hydra-MDP / ++, VADv2, PDM-Closed, LLM-Assist, WoTE, CLOVER | 2406.06978 / 2503.12820, 2402.13243, 2306.07962, 2401.00125, 2504.01941, 2605.15120 | F (Hydra-MDP V1 ✓); the Hydra-MDP GitHub README yielded no numbers |
| DiffusionDrive | 2411.15139 | CONFIRMED (20 anchors, 2 steps, 88.1) |
| FAST · DAP · LaPla · FSQ-token · Unified Driving Tokens · Planning in 8 Tokens | 2501.09747 · 2511.13306 · 2609.04070 · 2606.07464 · 2606.01935 · 2603.05438 | CONFIRMED titles; numbers UNVERIFIED except DAP's (snippet) |
| DriveMoE · DrivoR · FeaXDrive | 2505.16278 · 2601.05083 · 2604.12656 | CONFIRMED (snippets) |
| VINN · RAG-Driver · RealDrive · VLADriver-RAG | 2112.01511 · 2402.10828 · 2505.24808 · 2605.08133 | CONFIRMED (VINN, RAG-Driver, RealDrive titles); VLADriver-RAG title only, F |
| Zhuo et al. (beam-search-optimal trees) · TDM · Deep Tree-based Retrieval | 2006.15408 · 1801.02294 · 2408.11345 | CONFIRMED; theorem statements snippet-level, my top-1 marginal claim is DERIVED |
| Bonsai · AttentionXML · PLT (Wydmuch et al.) · hierarchical softmax (Morin & Bengio 2005, Mikolov 2013) | 1904.08249 · 1811.01727 · NeurIPS 2018 · AISTATS 2005 | CONFIRMED (snippets) |
| ScaNN · HNSW · Product quantization | 1908.10396 · 1603.09320 · Jégou et al. TPAMI 2011 | CONFIRMED (snippets); complexity statements not re-derived |
| ColBERT · Weller et al. | 2004.12832 · 2508.21038 | F |
| HiMulCon / HiConE | 2204.13207 | CONFIRMED (snippet) |
| Sadinle et al. · Angelopoulos & Bates · SafePath | 1609.00451 · 2107.07511 · 2505.09427 | CONFIRMED (snippets); SafePath is LLM-navigation, cited only as a trajectory-selection use |
| RSS | 1708.06374 | CONFIRMED id; **formula RECALLED, not re-read** |
| Invalid action masking | 2006.14171 | CONFIRMED (snippet) |
| nuPlan comfort thresholds | nuplan-devkit `metrics_description.md` | CONFIRMED (snippet: −4.05, 2.40, 4.89, 1.93, 0.95, 4.13, 8.37) |
| Jetson AGX Thor 273 GB/s, 1,035 FP8-dense TFLOPS | vendor and secondary spec pages | snippet-level, **peak spec, not measured**; every latency here is ESTIMATED |
| Sibling artifacts used | `hicap_prior_mask_result.json`, `hicap_conformal_cells_result.json`, `hicap_cost_model_result.json`, `P1_state_vocab_dataset.md` | INHERITED, not re-run (no torch here) |
