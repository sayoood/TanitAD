# PRE-REGISTRATION — how many trajectory hypotheses does REF-C actually need?

**Written 2026-08-04, BEFORE any sweep was run.** Stream: `arch-inf`. Agent branch:
`agent/arch-inf-20260803`.

**Verification instruction for the reader.** This file is content-pinned. Re-run

```
git ls-files -s -- "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-04-fan-width/PREREG_FAN_WIDTH.md"
git hash-object   "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-04-fan-width/PREREG_FAN_WIDTH.md"
```

If the two blob ids differ, the thresholds below moved after staging and *"fixed in advance"* is
**void**. The result document (`FAN_WIDTH.md`) prints both ids and the run JSONs record them.

⚠️ What was already done before this file was written, and therefore is **not** covered by it:
(a) the anchor vocabularies were extracted from the two checkpoints and the nested-FPS-prefix
relation was verified bit-exactly (§2.1); (b) the decoder source was read to establish
candidate-independence (§2.2). **No selection, ADE, oracle or latency number had been computed.**

---

## 1. The question

⭐ **MEASURED, INHERITED from `…/2026-08-03-esel-verdict/ESEL_VERDICT.md` (re-verified in this
pass, §6):** 73.76 % (base, 128 anchors) / 72.08 % (XL, 256 anchors) of REF-C's candidate fan is
outside a bounded-acceleration reachability band and is **never selected**; the clamp that removes
exactly those candidates is **paired-ADE-inert (Δ exactly 0.0)** on both arms.

Two readings have never been tested:

* **(i) COMPUTE.** If the fan is ~4× wider than the selector can exploit, the surplus is free
  latency and free energy — on Thor, the deployment target, that is the difference between a
  real-time planner and a demo.
* **(ii) ACCURACY.** If the surplus slots are simply *wasted*, spending the same budget inside the
  reachable set could be an accuracy lever.

This pre-registration covers both. ⛔ **The clamp's ADE-inertness is already MEASURED and is NOT
re-reported here as a finding.** The registered question is what its freed budget buys.

---

## 2. What is fixed by construction (established BEFORE this prereg, and stated so)

### 2.1 Prefixes are the principled subset — VERIFIED, not assumed

`furthest_point_sample` (`stack/tanitad/refs/refc.py`) is greedy FPS: it seeds one pool index from
`torch.Generator().manual_seed(seed)` and then appends `argmax` of the running min-distance. The
chosen list is therefore **nested by construction** — `chosen[:N]` *is* the FPS-N solution for the
same pool and seed. MEASURED here on the actual buffers pulled from the two checkpoints
(`decoder.anchors`, a persistent buffer, step 29,999 both):

| relation | result |
|---|---|
| `xl256[:128] == base128` | **bit-exact, maxabs 0.0** |
| `base128[:64] == xl256[:64] == small64` (`refc_anchors_small64.pt`) | **bit-exact, maxabs 0.0** |

The banked doc claimed only `base128[:64] == full256[:64]`; the 128-of-256 rung is verified here
for the first time. ⇒ **A prefix of length N is exactly the anchor set the model would have been
given at width N**, not an arbitrary subset.

### 2.2 The decoder is candidate-independent — so a prefix of the BANK is an exact decode at N

`CrossAttnLayer.forward` (`refc.py:1007`) is `q + cross(norm_q(q), kv, kv)` where `kv` is the image
conv map and `q` is the per-candidate query, followed by a per-token FiLM-MLP. **There is no
attention over the candidate axis.** `_decode` therefore maps candidate *i* to `(conf_i, offset_i)`
with zero dependence on the other candidates. Every graft on these two arms
(`decoder.maneuver_to_anchor`, an `nn.Linear(n_man, N, bias=False)`) is also per-anchor-row.

⇒ `fan[:, :N]` and `logits[:, :N]` are **bit-exactly** what the decoder emits when its anchor buffer
is truncated to its first N rows. This is an identity, not an approximation.

⚠️ **Two cross-candidate couplings exist in the source and would break the identity if they were
on**: `_goal_along_prior` z-scores terminal displacement *across the fan*, and `_apply_grafts`
group-norm-clamps when `sel.seam_clamp > 0`. Both are D-SEL/S6 additions that post-date these
checkpoints. **REGISTERED CHECK C-flags:** assert from the two checkpoints' `config.json` /
`state_dict` that no `goal_*`, `route_to_anchor`, `cons_gate` parameter and no non-zero
`seam_clamp` is present. If either is present, §2.2 is void and every prefix number in this pass
must be withdrawn.

⛔ **What the prefix identity does NOT establish, and will not be claimed:** it says what
*inference* at width N gives from *these* weights. It does **not** say what a model **retrained**
at width N would give. That is a separate, GPU-costed experiment and is explicitly out of scope.

---

## 3. Protocol — fixed

| | |
|---|---|
| corpus | canonical val, **881 windows / 40 episodes**, `wp_steps = [5,10,15,20]` (Δt = 0.5 s), window 8 stride 8 |
| arms | **`refc-xl-30k`** (256 anchors, primary — it carries the longest ladder) and **`refc-base-30k`** (128) |
| banks | `taniteval/results/fan_refc-{base,xl}-30k.pt` (shipped `logits`); `…/2026-08-03-esel-verdict/raw/fan_refined_refc-*-30k.pt` (adds `refined_logits`, `cons_score`); `…/2026-08-03-s3-deployable/raw/fan_deploy_refc-*-30k.pt` (adds `cons_deploy`) |
| ladder | **N ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}**, N ≤ the arm's width |
| subset rule | **FPS prefix `[:N]`** (primary) |
| selection rule | **UNCHANGED at every rung** — `argmax` of the shipped t=0 classifier `logits`, exactly `refc_rerank`'s ranker. A rung that changed the rule would not be a fan-width measurement. |
| candidate ADE | verbatim `taniteval.refc_rerank._score_row`: `norm(fan - gt[:,None], dim=-1).mean(-1)` |
| estimator | `taniteval.ci.episode_cluster_bootstrap` / **`paired_episode_cluster_bootstrap`**, unit = **episode**, `n_boot = 2000`, α = 0.05 |
| ⛔ | `overlapping_holdout_se` is **never** called |

---

## 4. THE REGISTERED QUANTITIES — the two curves, and how saturation is decided

For every rung N and every arm:

* `oracle_in_fan(N)` = mean over windows of `min_{i<N} ADE_i`
* `selected_ade(N)`  = mean over windows of `ADE_{argmax_{i<N} logits_i}`
* `sel_gap(N)`, `frac_sel_2x_worse(N)`, `rank_acc(N)`

**SATURATION, fixed now.** N\* is the **smallest N in the ladder** such that the paired
episode-cluster bootstrap of `metric(N) − metric(N_max)` satisfies **both**

1. `separated == False` (the 95 % CI contains 0), **and**
2. `|delta| ≤ 0.02 m`

0.02 m is not invented here: it is the `free_win_m` threshold transcribed in
`refc_sel_probe.PREREG_THRESHOLDS` from `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md`
§6.3. Both conditions are required, so a rung cannot saturate merely by being noisy.

**N\* is computed and reported SEPARATELY for `oracle_in_fan` and for `selected_ade`. Reporting one
without the other is out of contract — the gap between them IS the question.**

**Every rung is reported three-sided: `better` / `worse` / `not separated`,** where the sign is
read from the paired delta vs N_max and `not separated` wins whenever the CI contains 0.

### 4.1 Both outcomes, committed in advance

| branch | what it looks like | what it means | what we do |
|---|---|---|---|
| **O1 — SELECTOR-BOUND** | `N*(selected) < N*(oracle)` | the extra hypotheses reach a selector that cannot use them | the fan is over-wide **for today's ranker**; cut it, and re-widen only behind a better ranker |
| **O2 — JOINTLY BOUND** | `N*(selected) == N*(oracle)` | width is genuinely binding on both | do not cut; the compute lever is dead and the accuracy lever is fan-quality |
| **O3 — DISTRACTOR** | `selected_ade(N)` **worse** at large N, separated | extra candidates actively *harm* selection | cutting is an **accuracy** win, not only a compute win |
| **O4 — WIDTH-BOUND ORACLE** | `N*(oracle)` not reached at 256 | the oracle is still improving at the widest fan we have | the fan is too NARROW for a perfect ranker; report it and do not cut on the oracle's behalf |
| **O5 — NULL** | `N*(selected) == N*(oracle) == 1 or 2` | almost nothing is contributed by width at all | audit the bank before believing it |

**My registered prediction, stated before running: O1, with `N*(selected) ≤ 32` and `N*(oracle)`
either 128/256 or unreached (i.e. O1 ∧ possibly O4).** ⚠️ Disclosed influence on that prediction:
the PUBLISHED cross-arm triple (small64 0.5261 / base128 0.4728 / XL256 0.4714 selected, against
0.2213 / 0.1914 / 0.1640 oracle) already shows selected ADE nearly flat from 128→256 while the
oracle keeps falling. Those are **three different trained models**, so they confound width with
capacity and are not the measurement — but they are why I predict O1 and it would be dishonest to
pretend otherwise.

⛔ **If O2 or O5 fires I will say the compute lever is dead.** No reframing into "needs a better
selector to show the benefit" — that sentence is only admissible if O1 fires.

---

## 5. CONTROLS — with DIRECTION predicates, not only separation predicates

⛔ `C-shuffled` is **not** used as a control on subset choice: permute-then-argmax is a uniform
random pick for any score, so it is vacuous by construction here.

| control | what it does | **trigger — direction predicate included** |
|---|---|---|
| **C-flags** | assert no `goal_*` / `route_to_anchor` / `cons_gate` param and no non-zero `seam_clamp` in either checkpoint | **any present ⇒ §2.2 void ⇒ WITHDRAW every prefix number.** Not a warning. |
| **C-full-rung** | the N = N_max rung must reproduce the PUBLISHED selected / oracle ADE | `|Δ| ≤ 1.5e-4` (4-dp rounding half-width, the tolerance `refc_sel_probe` already uses). **Direction:** a rung that is *better* than published is as much a failure as one that is worse. |
| **C-monotone-oracle** | prefix sets are nested (§2.1) so `oracle_in_fan(N)` **must** be non-increasing in N | **any increase ⇒ the pipeline is broken, STOP.** This is a check on my own code that can genuinely fire. |
| **C-random-subset** | N candidates drawn uniformly without replacement, **24 seeds**, same selection rule | **Direction:** FPS-prefix must be **≤** random on `oracle_in_fan` at ≥ 7 of the 9 rungs. If random *beats* the prefix on oracle at most rungs, "the prefix is the principled subset" fails and the curves must be re-read as *"any subset of this size does as well"* — a different, weaker claim that I will then make instead. |
| **C-stride** | evenly spaced indices `arange(0, N_max, N_max//N)` — a second **non-random** subset | reported beside the prefix; no verdict rides on it, it exists so a prefix-specific artefact is visible |

⚠️ **Not an "oracle" control in the failed sense.** `oracle_in_fan(N)` is a genuine upper bound on
`selected_ade(N)` *pointwise per window* (it is the min over the same nested set the selector picks
from), so it cannot be beaten by any ranker on the same rung. This is checked per window, not
pooled — the AP-decomposition failure mode (an "oracle" that maximised per-fold AP while the
headline was pooled AP, and was beaten by arms it should have bounded) cannot occur for a per-window
min.

---

## 6. P2 — does reallocating capacity INTO the reachable set help?

The band is `flagship_v15.reachability_mask` re-exported through `tanitad.refs.refc_select` — the
**same function object** the 72.08 % was measured with, at its own `accel_max = 2.5`,
`horizon_s = 2.0`. Not tuned here.

Two operationalisations, both registered, because they answer different questions:

**P2a — INFORMATION (exact from the bank).** `reach-prefix-N` = the first N candidates in FPS order
that survive the band **on the decoded fan**, per window. Compare `selected_ade` against plain
`prefix-N`. ⚠️ **This is not a compute claim**: the survivors are found by looking at a decode of up
to 256 candidates, so the arm costs more than N decodes. It measures whether N *reachable* slots
carry more usable information than N mixed slots.

**P2b — COMPUTE (a realisable inference policy).** Apply the band to the **anchors** — which are
known before any decode, and `v0` is known before any decode — keep the first N surviving anchors,
and read exactly those candidates from the bank. This costs **N decodes and nothing else**, so it is
the arm that can actually ship.

**C-band-fidelity (a control that can fire).** P2b is only honest if the anchor-level band predicts
the decoded-level band. Report the full 2×2 over all window×candidate pairs (anchor-keep ×
decoded-keep), plus per-window survivor-count agreement. **Trigger:** if the anchor-level band
admits a candidate the decoded band rejects on **> 20 %** of pairs, or if `oracle_in_fan` under the
anchor band is **separated worse** than under the decoded band, P2b's saving is illusory and I will
report it as such rather than quoting P2b's ADE.

**Outcomes, committed:**

| branch | condition | meaning |
|---|---|---|
| **R1 — REALLOCATION WINS** | paired `reach-prefix-N` − `prefix-N` **separated and negative** at ≥ 2 consecutive rungs | the clamp is a **precondition that frees budget**; the freed slots buy accuracy |
| **R2 — NOT SEPARATED** | CI contains 0 | the surplus slots were neither helping nor hurting; the clamp's value stays **purely compute** |
| **R3 — REALLOCATION LOSES** | separated and **positive** | the unreachable candidates were load-bearing for the *selector* despite never being selected — report it, it would be a genuine surprise |

**My registered prediction: R2 at wide N and R1 at narrow N** (below ~16, where a mixed fan can run
out of reachable candidates entirely).

### 6.1 ρ hygiene — binding

Any Spearman ρ over the candidate axis is reported **restricted to reachable survivors** and
**always beside a selection ADE**. ⛔ ρ over the full candidate axis is not a proxy for a selector:
restricted to survivors one ρ in this programme went 0.6657 → 0.3008 and another crossed zero.

---

## 7. Latency — the compute half of the claim

A fan-width result is a compute claim as much as an accuracy one, so the ladder is timed.

* On **Thor** (the deployment target), the REF-C decoder forward at each N, `steps = 2`, batch 1.
* ⚠️ **Warm-up first; report p50 and p95 over ≥ 30 timed iterations; never a first call.** A
  224.98 ms render figure in this programme turned out to be a first call against a 0.10–0.17 s
  steady state.
* `torch.cuda.synchronize()` around every timed region. Memory reported **only** via in-process
  `torch.cuda.max_memory_allocated()` — on Thor `mem_get_info`, `free`, `tegrastats` and `VmRSS` all
  lie.
* Reported as **ms/frame and the implied Hz**, beside the accuracy at the same rung. Neither number
  is quotable without the other.

---

## 8. FOUR METRIC FAMILIES — per family, never pooled (binding, Sayed 2026-08-02)

At **every rung**, not only at the ends:

| family | what is reported | availability |
|---|---|---|
| **LONGITUDINAL** | target-speed abs/signed error, along-track abs/signed error | computable from the bank |
| — distance-keeping | headway / time-gap / TTC to the lead agent | via `taniteval/taniteval/lead_source.py` on the same 881 windows, ~270 carrying a lead; **stratified by speed**, and the 0–1 m/s band (20.7 % of lead windows) and the 15+ band (**n = 2, UNPOWERED**) are labelled as such |
| **LATERAL** | cross-track, heading, curvature, yaw-rate | computable from the bank |
| **TACTICAL** | goal/anchor-selection half: `rank_acc`, `sel_gap`, `frac_sel_2x_worse` — **this is the half fan width acts on** | computable. The manoeuvre-decision half needs decoded manoeuvre logits, which no fan bank stores ⇒ reported UNAVAILABLE **with its reason and n** |
| **STRATEGIC** | route/goal-setting quality | **UNAVAILABLE, n = 0** — `refc_rerank.dump` stores no route/goal label and decoded with `nav_mode='follow_constant'`, so the route input was never exercised. A WORK ITEM, not a pass. |

⛔ An ADE horizon sweep is **one row of four**. `Δt` comes from `four_families.infer_dt` on
`wp_steps`, never hard-coded (a hard-coded 0.1 s inflates speed 5× and accel 25× — R-2026-08-03-c).

---

## 9. What would make me withdraw this pass

* **C-flags** finds any cross-candidate coupling live in either checkpoint → §2.2 void, everything
  withdrawn.
* **C-full-rung** deviates by > 1.5e-4 → the bank is not the published decode; stop.
* **C-monotone-oracle** fires → my own code is wrong; stop.
* **C-random-subset** shows random ≥ prefix on oracle at ≥ 3 of 9 rungs → the "principled subset"
  premise is withdrawn and the claim weakens to *"any subset of this size"*.
* Selected ADE at any rung beats the arm's published full-fan value by **> 0.10 m** → treat as a
  leak, audit, do not publish (the D-SEL prereg's own `red_flag_m`).
