# v6 sizing — why the hierarchical model is SMALLER than v5, measured

**PI, 2026-08-12:** *"v5 was about 250 M parameters, why is v6 so small despite its
multihierarchical multi predictor architecture?"*

**Short answer: the hierarchy was never what cost the parameters.** In v6 the entire
hierarchy — tactical layer + strategic layer + planner — is **10.58 M, 12 % of the model**.
What cost 263 M in v5 was the **encoder width** and the **depth of two transformers**, and
v6 cut both. The two facts are independent, and conflating them is what makes 87.89 M look
like a downgrade.

Evidence class: **MEASURED (ours)** — both numbers are counted at instantiation in this
repo, not quoted from a doc. v5 is rebuilt from the banked
`experiments/pod-rescue-20260802/pod2/workspace/experiments/flagship-v5-w120-30k/config.json`
via `WorldModel(StackConfig(...))`; v6 from `V6Stack(V6Config()).param_report()`.

---

## 1. The two models, component by component

| component | **v5 flagship** | **v6 (a) shared** | delta |
|---|---:|---:|---:|
| encoder | **87.16 M** | **15.33 M** | **−71.83** |
| readout | 0.10 | 0.05 | −0.05 |
| operative predictor | **91.36 M** | **60.29 M** | **−31.07** |
| tactical predictor | 26.54 | 5.77 (`layer_tac`) | −20.77 |
| strategic | 8.39 | 4.15 (`layer_str`) | −4.24 |
| tactical policy | 22.74 | 0.66 (`planner`) | −22.08 |
| imagination field | **22.06** | — (absent) | **−22.06** |
| inverse dynamics | 5.25 | 1.65 (`aux`) | −3.60 |
| **TOTAL** | **263.58 M** | **87.89 M** | **−175.69** |

**Where the 176 M went, ranked:** encoder −71.8 · operative predictor −31.1 · tactical
policy −22.1 · imagination −22.1 · tactical predictor −20.8. Those five are 96 % of the cut.
**None of them is "the hierarchy".**

### Why each layer is cheap, by design not by accident
- v5's tactical predictor was a **transformer** (`d_model 512, depth 6, n_heads 8`) over a
  token window → 26.5 M. v6's `FTac` is a **residual MLP over a small, slow latent**
  (`d_tac 512 @ 2 Hz`, `d_str 256 @ 0.5 Hz`, 3 blocks) → 5.8 M / 4.2 M.
- A hierarchy adds **levels**, and a level is cheap when its state is small and its clock is
  slow. What is expensive is **attention over a long token window**, and in v6 that exists
  only in the operative layer, where the 10 Hz control problem actually lives.
- v5's `tactical_policy` (22.7 M) was a full anchored decoder stack. v6 replaces it with a
  0.66 M planner because **selection moved out of the parameters and into the roll-cost**
  (the W7 pattern per level), which is compute at eval time, not weights.

---

## 2. Where the encoder cut actually is

| | v5 | v6 default |
|---|---|---|
| `d_model` | **768** | **384** |
| `depth` | **12** | **8** |
| `n_heads` | 12 | 6 |
| input | 176×624, 9 ch, patch 16 | 256×640, 9 ch, patch 16 |

That single change is **−71.8 M, 41 % of the entire reduction**. ⚠️ It is also the one I
would flag as the **least defensible on current evidence**: vision is where the hierarchy has
to ground, and a 5.7× smaller visual trunk is a large bet taken silently as a default.

---

## 3. The headroom, and what it would buy — MEASURED

The invariant is **sub-300 M** and `build_stack_from_args` refuses to launch outside it
before any GPU time is spent. At 87.89 M there is **212 M of headroom**. Four arms, all
counted at instantiation today:

| arm | total | encoder | pred_op |
|---|---:|---:|---:|
| **(a) shipped default** — enc 384×8, pred 768×6 | **87.89 M** | 15.33 | 60.29 |
| **(b) per-layer encoders** (`--per-layer-encoders`) | **120.74 M** | 45.98 | 60.29 |
| **(c) v5-width encoder** — enc 768×12 | **159.93 M** | **87.32** | 60.29 |
| **(d) v5-width enc + pred depth 10** | **193.01 M** | 87.32 | 93.37 |

Arm (c) restores v5's exact visual trunk (87.32 M vs v5's 87.16 M) and still lands at
160 M — **47 % under budget**, with the hierarchy on top.

---

## 4. Recommendation

**Add arm (c) to the E-ENC pre-registration and decide it on the P-battery, not on ADE.**

Reasoning, with the counter-argument stated because it is strong:

- **Against growing:** v5.8f's measured defects were **not capacity**. T1 (the primary tier)
  found hold-action beating the closed loop **22×**, ~**99 %** of the gap longitudinal, and
  runaway acceleration — those are **conditioning and selection** failures. Spending 72 M on
  the encoder does not fix a selector. Staged training (S-W → S-T → S-S) also freezes lower
  layers, so a smaller trunk gets through three stages faster, and getting through the stages
  is the point. **Headroom is not an obligation.**
- **For growing:** the encoder cut was never *decided* — it is a default, and it is the
  single largest architectural difference from the model v6 must beat. If S-W fails P1/P3,
  the encoder is the first suspect and we will have burned a full stage to learn it.
- ⚠️ **(b) does not answer this question.** Per-layer encoders spend params on *three*
  encoders at 384 wide, not on *one* good one. (b) vs (c) are different questions and must
  not be run as one arm — that is the `--v2` conflation again (ten levers on two axes,
  result non-attributable).

**Cheapest discriminating experiment:** S-W to step 500 on (a) and (c), matched seed and
corpus, decided on P1 retention and P3 sign/gain — the same gate S-W already declares. Both
outcomes bound in advance: if (c) does not separate on the P-battery, the default stands and
the 72 M is banked as headroom for the tactical layer instead.

*This is a sizing question the programme can settle for roughly one GPU-hour, and it is
strictly cheaper to answer now than after a failed stage.*
