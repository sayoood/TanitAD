# `z_tac` HAS NO TEMPORAL EXTENT — what that invalidates, what it does NOT, and why it is a DEFECT

**2026-08-18** · branch `agent/arch-inf-20260803` · base HEAD `45b8e44` · dev-box CPU only
**Thor untouched — zero GPU.** No training run, no pod command, no eval launched.

---

## 0. Headline — five findings, in the order they change decisions

**1. ⭐ C115 REPRODUCES — independently, by mechanisms its author did not use, with every control
firing.** MEASURED (`code/probe_ztac_temporal.py`, `code/out.json`). The gradient from `z_tac` to
the history frames is an **exact structural zero**, while the positive control on the same graph is
nonzero at every frame.

**2. ⛔ IT IS A DEFECT, NOT A DESIGN — and the proof is that THE SPEC'D MODULE IS ALREADY BUILT AND
SITTING UNUSED.** `PhiTac` — the causal-TCN temporal pool the design, the backlog and **the paper**
all mandate — exists at `stack/tanitad/models/tactical.py:99`, is tested, and was **trained**
(registry §1.13b). `stack/tanitad/models/v6.py` references it **zero times**. No rationale exists
for the substitution at any of the locations checked.

**3. ⛔ THE STRATEGIC LAYER HAS THE SAME DEFECT, AND C115 DOES NOT MENTION IT.** `φ_str` is spec'd as
a pool over a **`z_tac` window**; `uplink_str` reads a single tick. **MEASURED here**, same probe:
`z_str`'s gradient to history frames is also exactly zero. This is a **second retraction line**, not
a footnote to the first.

**4. ⚠️ THE BLAST RADIUS REACHES THE PAPER'S FORMAL CONTRACT — the worst single site.**
`Paper/TANITAD_PAPER.md:686` states `z_T = φ_T(sg[z_O(t−3..t)])` under a section that at `:695`
calls these properties *"asserted by construction and then checked, not assumed"* — and the check
that exists (`assert_isolation`) covers **only the other property**. The temporal property was never
checked and is false of the shipped model.

**5. ⭐⭐ AND THE OVER-CORRECTION IS ALREADY AVAILABLE, SO NAME IT NOW: "THE TACTICAL LAYER IS FAKE"
IS WRONG.** The programme has **three** tactical implementations. **Two of them genuinely integrate
the window** — and I MEASURED one of them doing so with the *same probe that fails on v6*. The
finding is a fact about **`V6Stack`**, not about the tactical layer as a concept, and not about any
v1-flagship tactical result.

> ⛔ **ESCALATION — this is a PI / diagram-owner decision, not an implementation's to take.** Either
> the paper and design docs are corrected to describe an instantaneous tactical read, **or** `PhiTac`
> is wired in at S-T. §6 is the pre-registered experiment that decides which, at **0 Thor GPU**.

---

## 1. Re-verification — MEASURED, by mechanisms C115 did not use

C115 is one night old and rests on one stream, so re-running its assertions would re-measure **its
instrument**, not the architecture. `code/probe_ztac_temporal.py` therefore deliberately does not
import `stack/tests/test_v6_t2_contrastive.py`.

| | C115's probe | mine |
|---|---|---|
| P1 | — | **autograd reachability**: backprop to the input pixels, read `frames.grad` per time index |
| P2 | freeze non-last frames, compare outputs | **cross-sample history splice**: a *different* window's history under the *same* last frame |
| P3 | — | **train() mode**, to kill a norm layer that couples only in training |
| P4 | — | **the same probe on `TacticalStage0`** (the scope boundary) |

### 1.1 The numbers (MEASURED, `code/out.json`)

| quantity | frame 0 | 1 | 2 | 3 (last) |
|---|---|---|---|---|
| `∂z_tac/∂frames` abs-max | **0.0** | **0.0** | **0.0** | 0.0637 |
| `∂z_str/∂frames` abs-max | **0.0** | **0.0** | **0.0** | 0.1048 |
| **POSITIVE CONTROL** `∂ẑ_op/∂frames` | 0.0474 | 0.0220 | 0.0210 | 0.1315 |

- **P2:** splicing a different history under the same last frame leaves `z_tac` **bit-identical**
  (`torch.equal` → True, max-abs-diff **0.0**). Swapping the **last** frame moves it by **0.779 of
  its own scale**.
- **P3:** bit-identical in `train()` mode too.

### 1.2 The controls, all of which fired

| control | result | why it is required |
|---|---|---|
| **POSITIVE — `ẑ_op` on the same graph** | nonzero at **all 4** frames | If this were zero too, the probe would be inert and prove nothing (C109). |
| **POSITIVE — perturb the last frame** | `z_tac` moves, rel. **0.779** | Distinguishes "ignores history" from "ignores everything". |
| **TRIVIAL-PROXY — is the net dead?** | `z_tac` batch-std **0.366**; `z_op_win` across-time spread **0.214** | A constant network is invariant to everything; that would pass P1/P2 for the wrong reason. |

### 1.3 ⚠️ TWO CONFOUNDS I HIT IN MY OWN INSTRUMENT — both would have "confirmed" the finding falsely

**(a) The default config makes autograd UNINFORMATIVE.** `isolate_uplink=True` (`v6.py:3229`) means
`uplink_tac` does `src = self._cut(z_op, cfg.isolate_uplink)` → `z_op.detach()` (`v6.py:4354`). The
gradient is then `None` at **every** index *including the last* — which reads as "z_tac ignores all
frames" but is the **X3 stop-grad, a completely different mechanism**. Run only at the default, an
autograd probe "confirms" C115 for the wrong reason. Both settings are reported in `out.json`.

**(b) ⛔ `z_tac.sum()` IS AN IDENTICALLY-ZERO SCALAR, AND MY FIRST RUN REPORTED IT AS THE FINDING.**
`adapter_tac` ends in `nn.LayerNorm` (`v6.py:3979`). At init (`gamma=1, beta=0`) LayerNorm's output
is zero-mean across the normalised axis **by construction**, so `.sum()` is the constant 0 and its
gradient wrt the pixels is **exactly zero for every input, whatever the network does**. My first run
returned all-zero gradients *including the last frame* — a **stronger and false** version of C115.
Caught because P2 said the last frame *does* matter. Fixed with a fixed random projection, and the
defect is **pinned as a negative control** (`P1_layernorm_sum_null_is_REAL`: measured
`z_tac.sum() = 4.77e-07`, gradient exactly 0).

⇒ **ROOT-CAUSE CLASS: A REDUCTION THAT LIES IN THE NULL SPACE OF THE MODULE IT PROBES.** Same family
as C109 ("inert by construction at the deployed setting") — occurring **inside the instrument built
to audit somebody else's instrument**. ⭐ **Rule: never scalarise a LayerNorm/RMSNorm-terminated
module with `.sum()`.** Use a random projection; verify the scalar is not constant before trusting
any gradient taken through it.

### 1.4 ⛔ A THIRD HAZARD, AND IT WOULD HAVE CORRUPTED EVERY CITATION IN THIS REPORT

**MEASURED: `stack/tanitad/models/v6.py` grew from 4,914 to 5,154 lines WHILE I WAS READING IT** — a
sibling stream editing live. My first pass resolved `adapter_tac` to line **3737**; a re-resolve
twenty minutes later gave **3977**, a **+240-line shift**. Every `v6.py` line number taken before
that point was silently wrong.

⇒ All `v6.py` citations here are resolved against **`sha256 d1cd69d7…`, 5,154 lines**, recorded in
`code/out.json` `meta.v6_py_sha256`, and **the probe was re-run against that exact state** — same
numbers, all controls firing. ⇒ **Re-resolve by content (`grep -n`), never by trusting the integers.**

⇒ **ROOT-CAUSE CLASS: C114's torn snapshot, wearing a CITATION costume.** C114 caught it as a false
*test failure*; here it would have produced a report whose every `file:line` pointed at the wrong
code — **the more dangerous of the two, because a wrong citation looks authoritative rather than
broken.** ⭐ **A line number is a claim about a file STATE; under live concurrency it must carry that
state's hash, or it is not evidence.**

---

## 2. Is this a DEFECT or a DESIGN? — **DEFECT**, and it is not a close call

The decisive evidence is not an absence of rationale; it is the **presence of the correct module,
unused**.

| | |
|---|---|
| **The spec, 4 independent locations** | `HIERARCHICAL_WM_REDESIGN.md:115` *"φ_tac: TCN/attn pool over z_op(t−3..t)"* · `V18_BACKLOG.md:56` *"phi_tac: temporal pool over z_op(t−3..t)"* · `Paper/TANITAD_PAPER.md:546` *"z_tac … = φ_tac(z_op(t−3..t))"* · `stack/tanitad/models/tactical.py:184` *"`z_tac` is a pure temporal pool (PhiTac)"* |
| **The module** | `stack/tanitad/models/tactical.py:99` `class PhiTac` — causal dilated TCN, dilations (1,2,4), receptive field 15. Tested (`stack/tests/test_tactical.py:102`). **Trained** — `MODEL_REGISTRY.md` §1.13b, E4.4, 2026-08-10. |
| **What v6 built instead** | `v6.py:3977-3979` — `nn.Sequential(Linear, GELU, Linear, LayerNorm)` on a `[B, d_op]` vector. **No time axis in its input at all.** |
| **References to `PhiTac` in `v6.py`** | **0** (MEASURED, `grep -c`). `v6.py:92` imports only `FTac` from that module. |

**Rationale: ABSENT.** ⭐ **MEASURED by me:** the flattening arrived with the file — `git log -S
"reshape(b * w"` returns exactly one commit, **`0c30a0f`** (2026-08-11), whose message is entirely
about the W7-FULL selector verdict and the winner's curse, and which mentions `encode_window` /
"temporal extent" / "per-frame" **zero times** (`grep -c` on the full message = 0). The v6.py
addition rode along inside a results commit. **INHERITED (sweep, not re-opened by me):** the same
absence at five further locations — the redesign doc-set, `DIAGRAM_CONFORMANCE.md`, `BACKLOG.md`,
`RETRACTION_LOG.md`, and code docstrings.

⚠️ **The absence is the WEAKER half of the argument and I am not resting on it** (per "absence at one
location is not absence"). The decisive evidence is **positive and verified twice by me**: the spec'd
module exists, and `v6.py` references it zero times.

⚠️ **A separate, TRUE design rationale exists — for the ENCODER, and it does not cover this.**
DINO-WM/JEPA doctrine (frozen per-frame encoder, dynamics in the predictor) is documented in ≥4
design docs, and at the **operative** layer the codebase genuinely honours it: `OperativePredictor`
runs causal self-attention over the window (`predictor.py:165-168`) and there is a true
autoregressive rollout at `v6.py:4510-4516`. **That rationale justifies a per-frame *encoder*; it
does not justify a per-frame *tactical latent*, because the spec put the temporal pool in `φ_tac`,
above the encoder, precisely so the encoder could stay per-frame.**

⚠️ **It survived a dedicated audit, which is why it reads as unnoticed rather than chosen.**
`DIAGRAM_CONFORMANCE.md:113` passes the tactical layer — *"clock ≈ 2 Hz ✅ CONFORMS"* — by checking
`hz_tac`/`stride_tac`. It checked the **clock** and never whether anything is **pooled over** it;
`φ_tac`/`PhiTac` appears nowhere in that audit. The same audit caught the sibling F-1 defect, so the
auditing was live and competent — the row that would have caught this was never written.

### 2.1 ⛔ The strategic layer has it too — a finding C115 does not contain

`φ_str` is spec'd as a pool over a **window of `z_tac`** (`HIERARCHICAL_WM_REDESIGN.md:116`;
`Paper:547`, `:687`). `uplink_str` (`v6.py:4377`) takes a single `z_tac`. **MEASURED here:**
`∂z_str/∂frames` is exactly zero at every history frame. ⇒ **Both upward pools are missing, not
one.** Any correction that fixes only the tactical line leaves the same false statement standing one
level up.

---

## 3. What this INVALIDATES

Grouped by whether the claim *requires* `z_tac` to integrate a window. Full inventory with file:line
in `BLAST_RADIUS.md`.

### 3.1 ⛔ The paper — highest blast radius

| site | claim | status |
|---|---|---|
| `Paper/TANITAD_PAPER.md:686` | `z_T = φ_T(sg[z_O(t−3..t)])`, asserted at `:705` as **"The v6 contract is"** | ⛔ **FALSE of the shipped model.** The `sg[·]` is real; the `(t−3..t)` is not. |
| `Paper:546` | `z_tac … = φ_tac(z_op(t−3..t))` | ⛔ FALSE |
| `Paper:547`, `:687` | `z_S = φ_S(sg[z_T window])` | ⛔ FALSE (same mechanism, one level up) |
| `Paper:697` | *"Abstraction by temporal down-sampling under a shrinking state"*, one of two properties *"checked, not assumed"* (`:695`) | ⛔ **The temporal half is false AND was never checked.** The dimensional half (2048→512→256) survives. |

### 3.2 ⛔ Design docs of record
`HIERARCHICAL_WM_REDESIGN.md:115`, `:116`, `:118` · `V18_BACKLOG.md:56` ·
`V5F_DATA_WIRING_AUDIT.md:50`, `:51` — all describe a pooled tactical/strategic state. True of
`tactical.py`, **false of `V6Stack`**; each needs an explicit scope tag.

### 3.3 ⛔ Catalog T2 (already retracted as C115)
`V6_TRAINING_MEASURES.md:65`, `DIAGRAM_CONFORMANCE.md:56`, `:212` — the **time-reversal half** is not
expressible on v6. The **lane-mirror half survives, is built, and carries the catalog's entire stated
justification**. Half a row, not a row.

### 3.4 ⚠️ NEEDS A RULING (not invalidated) — the "temporal hierarchy" pillar
`V58F_FUSION.md:18` names **"temporal hierarchy"** as *"the pillar"*. In v6 the hierarchy **is**
temporal in the predictor **stride** (`stride_tac = 5`) and in the trajectory **band** (2–6 s), but
**not in the state**. If the pillar's claim was about the state, the pillar needs re-stating; if
about the clock, it stands. **PI call.**

⚠️ The six copies of `z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)` are **mechanically honoured** —
`predictor_tac` really does roll forward. What needs re-reading is whether *rolling an instantaneous
scene read* is "tactical **dynamics**". `V6_TRAINER_DESIGN.md:227` already flagged this trap from the
stride side: *"an identity map wearing a hierarchy's name"*. It now needs a second reading from the
**state** side.

---

## 4. ⚠️ WHAT IS **NOT** AFFECTED — read this before correcting anything

Over-correction has cost this programme before (the ladder's headline survived while its per-row
inventory died). The following are **untouched**, and three of them are MEASURED, not argued.

### 4.1 ⭐ The other two tactical implementations DO integrate the window — MEASURED

The programme has **three** tactical implementations. Only one is defective.

| implementation | temporal support | evidence |
|---|---|---|
| `V6Stack.uplink_tac` (`v6.py:3737`) | ⛔ **none** — last frame alone | MEASURED, §1 |
| `TacticalStage0` (`tactical.py:448`) | ✅ **real** — `z_tac = self.phi_tac(z_op)` over `[B,W,d_op]` | ⭐ **MEASURED HERE** |
| `TacticalPolicy` (`fourbrain.py:268`) — the **v1 flagship** | ✅ **real** — causal transformer over the state window (`fourbrain.py:332-339`) | read from source |

⭐ **The same probe, the same `scalarise()`, the opposite result** (MEASURED, `out.json`
`P4_TacticalStage0_DOES_integrate`): on `TacticalStage0`, history-splicing **moves** `z_tac`
(max-abs-diff **0.0630**) and the per-slot gradient is **nonzero at every slot**
`[0.0597, 0.0773, 0.0744, 0.7399]`. ⇒ **This is not a claim about "the tactical layer". It is a
claim about `V6Stack`.**

### 4.2 The operative path — explicitly out of scope, confirmed
`predictor_op` consumes the full `z_op_win` (`v6.py:4758`) under a causal mask
(`predictor.py:165-168`), plus a true autoregressive rollout (`v6.py:4510-4516`). **My positive
control is exactly this path and it fired at every frame.** The DINO-WM contract — per-frame
encoder, dynamics in the predictor — **is genuinely met at the operative layer.**

### 4.3 The four metric families — unaffected
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC results measure emitted trajectories and decisions.
Nothing in their computation assumes a temporal `z_tac`. **A capability claim about the tactical
layer may be re-scoped by §6's outcome; no already-measured family number changes.**

### 4.4 Every v1-flagship tactical result — unaffected
The 3.38 m tactical weakness, `wp_heads`, κ 0.6033, the anchored decoder, and **the whole
`ctx→tactical` seam family (incl. the +0.0148 corrected seam in `CLAUDE.md`)** all measure
`fourbrain.py`, whose tactical head **does** integrate the window. Untouched.

### 4.5 Band / horizon / vocabulary / imagine-and-select — unaffected
The 2–6 s tactical band is a property of the **emitted control sequence**, not of `z_tac`. The goal
vocabulary and the factored LAT×LON heads concern **the shape of the decision read off `z_tac` at
the last tick** — the very distinction that matters here, and on the safe side of it.

### 4.6 ⭐ One document was already right, two days early
`X4_P9.md:16` (2026-08-16, **before** C115): *"the uplink reads only the window's last frame"*. It
was correct and nobody propagated it — which is itself the finding-transmission failure worth naming.

---

## 5. What a fix costs, and where it may legally go

| | |
|---|---|
| **Legal insertion stage** | ⭐ **S-T — the only stage with a non-empty allowance.** `STAGE_MAY_INTRODUCE` (`stack/scripts/train_v6_staged.py:331`) is `()` for S-W, S-S and S-J. S-T is also the stage that trains `layer_tac`. |
| **How** | ⛔ **A NEW KEY PREFIX (`phi_tac.`), never a widened `adapter_tac.0.weight`.** The allowance *"adjudicates KEYS, and a shape change bypasses it entirely — `load_state_dict(strict=False)` still RAISES on shapes"* (`train_v6_staged.py:337-340`). |
| **Live-run risk** | ⛔ `load_resume` is hard `strict=True` (`train_v6_staged.py:3888`) — the allowance does **not** cover `--resume auto`. **Any change to the DEFAULT build kills the running v6F S-W resume** (87,893,449 params / 405 keys). ⇒ ship **default-OFF behind a flag**, exactly as F-7/F-1 did. |
| **Cost if done at the uplink** | **LOW, and no fresh S-W run** — `STAGE_INVALIDATES["S-T"] == ()` because S-T trains `layer_tac` on a frozen trunk. Work: reuse `PhiTac`, +1 group prefix, +1 allowlist entry, +1 test pin, re-measure params. |
| **Cost if done inside `encode_window`** | ⛔ **CATASTROPHIC — a full fresh S-W run** (multi-GPU-day) plus invalidation of every downstream certificate. `STAGE_MAY_INTRODUCE["S-W"] == ()`. **Do not fix it there** — and the spec never asked for it there. |

⚠️ `stride_tac` is already a derived property (`v6.py:3461`) giving the correct 1 Hz subsampling
stride, which is exactly the input `PhiTac` expects. The plumbing exists.

---

## 6. The cheapest discriminating experiment — PRE-REGISTERED, both outcomes fixed now

**Question: does the tactical layer NEED temporal integration to do its job?**

### 6.0 ⛔ Two prior results constrain this design, and one nearly makes it redundant

**`Project Steering/PREREG_TEMPORAL_LATENT.md` already exists and was already RUN**
(`…/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md`). I must not re-propose what it answered.

1. ⛔ **It fired OUTCOME V (VIDEO-LIMITED) for `long_accel`**: **0 of 35 arms** separated, including a
   `window` basis over all 9 frames (18,432 features), while the oracle reached **+0.9262**.
   ⇒ **"Just give the readout more frames" is ALREADY MEASURED INSUFFICIENT for `long_accel`.**
   ⭐ But that document **explicitly does not retire the thesis for "speed, TTC, headway or the
   manoeuvre decision, which must be argued on their own evidence"** — and the **manoeuvre /
   tactical-goal channel is precisely what this experiment targets.** It is the named gap.
2. ⚠️ **A single still frame reads `speed` at R² +0.6642** on that corpus, against a 800 ms window's
   +0.7145. ⇒ **static appearance is a powerful shortcut**, and any "temporal helps" claim here must
   survive a single-instant control. This is the C92 trivial-proxy trap with a measured magnitude.

### 6.1 Design — 0 Thor GPU, dev-box only, no new instrument

**The substrate already exists.** The sibling S-W latent dumper
(`stack/scripts/v6_dump_sw_latents.py`) already emits **`pooled_seq [n, W, d_op]`** — the full window
of operative latents — plus `z_tac`, `v0`, and `gt_endpoint`. One dump, then closed-form ridge
readouts on the dev box. **No trainer, no epochs, no learning rate to blame** (the `accel_probe.py`
protocol: exact kernel ridge over the full regularisation path, inner-split selection).

**Target = the TACTICAL family**, not ADE: the 2 s tactical goal endpoint (`gt_endpoint`) and the
E4.1 3-axis manoeuvre labels (`refb_labels.maneuver3_labels`).

| arm | input | what it isolates |
|---|---|---|
| **A — INSTANTANEOUS** | `pooled_seq[:, -1]` | what `V6Stack` actually deploys today |
| **B — TEMPORAL** | `PhiTac(pooled_seq)` | the spec'd module, the candidate fix |
| ⭐ **A-pad — PARAM-MATCHED NULL** | `PhiTac(last frame repeated W times)` | **identical params, identical architecture, ZERO temporal information** |

⭐ **A-pad is the arm that makes this decisive.** B-vs-A confounds *information* with *capacity*;
**B-vs-A-pad isolates the information alone**, and it is the direct constructive analogue of C115's
own second probe (a window made entirely of one frame).

**Controls, every one binding:**
1. **POSITIVE CONTROL — a target that provably needs history:** predict `z_op(t−3)` (a past latent)
   and the speed *change* across the window. **B must beat A-pad here, or the instrument is inert
   and no null is readable.** (C79 withdrew D1 for exactly this failure.)
2. **TRIVIAL-PROXY CONTROL — ego-speed echo (C92):** a `v0`-only baseline arm. If it matches the
   tactical arms, the comparison is meaningless and the run is VOID for that target.
3. **SHUFFLED control per arm**, paired ΔR² against the arm's own control.
4. **Single-instant shortcut audit**, mandated by the +0.6642 static-appearance result above.
5. **Empirical null** (constant = train mean) as a first-class arm — R² = 0 is not the null.

**Estimator:** paired **episode-cluster bootstrap** (`taniteval/ci.py`), B = 2000.
⛔ `overlapping_holdout_se` is not used. **Every bracket labelled** — dispersion is not a CI (C109).
**Tier: T0-diagnostic** — this is a representation probe, **never a driving-performance claim.**

### 6.2 The reading, fixed in advance

| outcome | fires iff | what the programme does |
|---|---|---|
| ⭐ **T — TEMPORAL NEEDED** | B's paired ΔR² over **A-pad** separates positive (CI lower bound > 0) on ≥1 tactical target, **and** the positive control fired, **and** the `v0`-only arm did not match | **The architecture is the defect.** Wire `PhiTac` at S-T, default-OFF, new key prefix. C115's T2 time-reversal row becomes *deferred*, not *dead*. |
| ⛔ **I — INSTANTANEOUS SUFFICES** | B ≈ A-pad (paired CI includes zero) while the **positive control still fires** | **The DOCS are the defect, not the model.** Correct `Paper:686/546/547/687/697`, the redesign docs and the catalog to describe an instantaneous tactical read. **`PhiTac` is not wired**, and the T2 time-reversal row is **permanently** retired, not deferred. |
| **VOID** | the positive control does **not** fire, or the `v0`-only arm matches | No verdict. Fix the instrument first; **quote nothing.** |

⚠️ **What would make me wrong, stated now:** if **I** fires, I will write it as the headline above
every proposed fix, and §5's cost analysis becomes moot rather than deferred — the correct action
would be a **documentation retraction, not an architecture change**, and I will say so in those
words. If **T** fires, the §3 invalidations become a *temporary* gap with a funded fix rather than a
permanent correction.

⚠️ **Limits, fixed in advance:** one corpus; a frozen-trunk *readout* probe is a **lower bound** on
what a trained `PhiTac` could carry (a positive is decisive, a null is bounded by the reported
sensitivity floor); and it does **not** speak to `φ_str`, which needs its own arm.

---

## 7. Recommended actions, in priority order

1. ⛔ **PI / diagram owner:** rule on §3.4 (is "temporal hierarchy" a claim about the **state** or the
   **clock**?). Everything else is downstream of that one word.
2. ⛔ **Correct `Paper/TANITAD_PAPER.md:686`** — it is a published false statement about the shipped
   model, presented as *"the v6 contract"*. Even under outcome **T** it is false *today*.
3. ⛔ **Open a SECOND retraction line for `φ_str`** (§2.1). Fixing only the tactical line leaves the
   identical false claim standing one level up.
4. **Run §6.** 0 Thor GPU, substrate already dumped.
5. **Add a `φ_tac` row to `DIAGRAM_CONFORMANCE.md`** — the audit passed the tactical layer by
   checking the clock and never checking the pool.
6. **Propagate `X4_P9.md:16`**, which was right two days early and was never read.

---

## 8. Deliverable manifest

| artifact | where it lives | class |
|---|---|---|
| `ZTAC_INVARIANCE.md` (this file) | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-ztac-invariance/` — **repo, staged** | — |
| `BLAST_RADIUS.md` — full file:line inventory | same dir — **repo, staged** | — |
| `code/probe_ztac_temporal.py` — the re-verification probe | same dir — **repo, staged** | MEASURED |
| `code/out.json` — probe output, every number in §1 and §4.1 | same dir — **repo, staged** | MEASURED |

**Nothing is stranded.** No pod path, no worktree, no uncommitted-only artifact. Nothing was
launched; Thor's GPU was never touched; no file outside this directory was modified.

⛔ **INTEGRATION ESCALATED, NOT WRITTEN INTO A DOC:** items 1–3 of §7 are **decisions for the PI /
diagram owner** and are raised here as escalations, not left as a "please fix" note in a README that
nobody re-reads.
