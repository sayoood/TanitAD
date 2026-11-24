# CARD_QUESTIONS: what `PREREG_SCORERFT_PICK.md` could not settle, with a recommended answer for each

Written 2026-11-24 for the MM, and revised the same day after the MM's decisions on Q-P1 to Q-P3 (decision 10 included), and again for Q-P17 (the
gradient clip), and for Q-P18 (MM decision 11, the clip logging). The card already carries
each recommended answer as drafted, so the MM can freeze it as it stands or change a line before freezing. The open questions are ordered by how much
they can cost if left wrong. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Settled by the MM (2026-11-24), recorded here and in the card

### Q-P1 · C's run log: SETTLED, banked, so the FULL step-0 control is the card's form
* **Banked files.** The MM banked `o3/card/evidence/metrics_scorerft_c.jsonl` (3,046 lines, sha256 `97a8631c…`, pulled framed and equal to the
  pod's), and T's `metrics_scorerft_t.jsonl` beside it (3,047 lines, `09cdaf83…`).
* **Card form.** CTRL-S0 runs in its full form only; the fallback is gone from the card, the spec and the pins.
* **Demonstration on Stage 1, MEASURED.** Card §3.1, from these two files:
  * at step 0, T vs C, **19 / 19 training keys and 10 / 10 `onpolicy` keys are exactly equal**, and every T-only key is a ranking key ⇒ PASS;
  * the same rule at step 1 FAILS on 10 + 4 keys, so the control is not vacuous;
  * CTRL-ORDER: **0 of 1,000 steps differ**.

### Q-P2 · K2 risk: KEEP PL1 as written (MM). Updated note: routing RAISES this risk
The original note stands. P's gradient enters through probabilities (factors such as p(1 − p)), while T's entered through logits. T's rule picked
w* = 1/32 (`--rank-w 0.03125`).

**What decision 10 adds.** Routing zeroes the term's gradient on every factor whose label points the other way.
* On a safe pick, that is NC and DAC, both labelled 1, and TTC and C wherever they are labelled 1.
* On a better j, it is every head labelled 0.

So ‖∇_S L_rank‖ at w = 1 falls, r(1) falls, w* moves up, and r(4) < 1 (K2 ⇒ PL1) becomes more likely. This is ESTIMATED, with no measurement.

**Recommended:** unchanged. Run the rule on P's routed argv first (~0.5 L40-h). It reads r(1) directly and decides between launching and PL1. PL1's
P-log keeps the routing.

**Update 2026-11-24 (Q-P17): the direction above is contradicted.**
* The o3b implementer reports the routed gradient **larger in net on the shared `score_head`**: the pick's downward push no longer cancels the
  better proposals' upward push (o3b README §3; INHERITED, dev-box synthetic smoke).
* The rule measures on parameters, so r(1) may rise instead of fall. Both K2 and K2b stay possible.
* Card §5.4 is corrected. The recommendation is unchanged: the rule on the pod decides.

### Q-P3 · The plain term pushed a safe pick's NC / DAC against their labels: SETTLED by MM decision 10, label-consistent routing
The term's gradient reaches p_h(x) only toward that proposal's own label. It is implemented as a per-factor stop-gradient inside s, and the forward
value, the pick and the loss value are unchanged (card §1.1). It is controlled by CTRL-ROUTE (card §4.4): on a pick whose NC label is 1, the NC-logit
gradient from the term is EXACTLY 0.0. The control has an analytic form and a real-smoke form, with the RED mutation `pick_noroute`. Two consequences
are recorded as open questions (Q-P15, Q-P16).

### Q-P18 · `train.py` logs the pre-clip norm and the clip coefficient: SETTLED by MM decision 11 (2026-11-24, before any P data), KEEP the logging
**The decision.** P's `train.py` keeps o3c's logging of `grad_norm_pre` and `clip_coef` on every logged training row. CTRL-CLIP needs it: V4
cross-checks the trainer's logged values against the wrapper's N_t / c_t, exactly. ⇒ **Q-P17 (b) is superseded.**

**Why the reuse of C is not weakened.**
* **Bit identity: MEASURED by the o3c implementer, INHERITED here** (the card author read the test source and did not re-run it).
  * Test: `o3c/files/test_clip_probe.py`, `chk_logging_changes_no_behaviour`; `tests.tsv` reads PASS, `ZZCLIPPROBE_PASS 52`.
  * On the CPU trainer, after 3 steps, every final checkpoint tensor is bit-identical with and without the logging.
  * The determinism control (current vs current) is identical, and a skip-`opt.step()` sensitivity arm DIFFERS.
* **Source: MEASURED by the card author.**
  * `gn_pre` occurs only at `train.py:2040` and `:2065`, and the `clip_grad_norm_` call is unchanged.
  * `train.py`'s blob is the same at `83411c28`, `5eb74f30` and `7ac49a18` (sha256 `59a737f4…`).
  * On the L40, bit identity is argued from source, not measured. CTRL-S0, CTRL-ORDER and CTRL-CODE still run on P's run.
* **The allow-list.** The two keys are P-only keys in CTRL-S0's allow-list (`S0_RANK_ONLY`, o3c `scorerft_read.py:108`), exactly as T's rank keys
  were. C's banked log predates them, and the 19 + 10 compared keys are unchanged.

**What changed in the card:**
* the header;
* §3.1 point 3 and CTRL-S0's allow-list;
* §3.2 (the clip fact, V4 and Scope);
* §4.2, which now lists the four `train.py` hunks by line;
* §4.10, the pins.

**Consequences.**
* **`PICK_TRAIN_SHA256` is no longer the `fd5e5dad…` that Q-P13 expected.** It is filled from the landed blob in the freeze-and-land commit, and
  its expected value is o3c's `b1f47518…` (card §4.10).
* **C's full-run clip path stays UNMEASURED.** P's full-run N_t / c_t are descriptive only.

**Residue, flagged and not acted on (outside the card hold):**
* the comment at o3c `scorerft_read.py:83` still names `fd5e5dad…` as the expected pin;
* an unset `CLIP_PROBE_SHA256` reads INVALID, not REFUSED. Both give no outcome. An up-front refusal would be one more entry in the read's `need`
  list.

## Open (top three first)

### Q-P17 · Grad clip: the trainer clips the global gradient norm at 1.0. Does that void the weight rule, and how is w* audited? Drafted into card §3.2, §4.9, §5.1 and §9 (PL3 / PL4)
**The trigger (INHERITED, o3b README §3).** In a dev-box tiny smoke, `--rank-w` 0.5 and 1.0 both sat on the clip at `train.py:2040`, so the reach
test had to use 0.001 vs 0.002.

**The two factual questions, answered (MEASURED).**
1. **`weight_rule_rank.py` is pre-clip and silent on the clip.** It stops at the optimiser build and takes each term's gradient alone with
   `torch.autograd.grad`. It covers only S and the trunk, computes no total and no global norm, and pools 8 micro-batches against the trainer's 64.
2. **T's output (`a9f13c78…`) cannot show the clip binding at w\* = 1/32.**
   * The four measured norms at w\* are 0.0992, 0.1736, 0.7132 and 0.2086. They sum to 1.195, and their root-sum-square is 0.770.
   * l_score on the trunk and l_traj on the decoder were never measured. ⇒ UNDETERMINED.
3. **`metrics_scorerft_c.jsonl` does NOT log a gradient norm,** and neither does T's: every key was enumerated, and `train.py:2040` discards the
   return value. So C's clip fraction cannot be read from it, and the probe measures it instead.

**What the card now says, and why.**
* **The rule is unchanged.** `clip_grad_norm_` scales every gradient by one c_t, so r and ρ (ratios inside one window) are exactly invariant.
* **AdamW is invariant to a constant c.** It changes only via ε = 1e-8.
* **Only a coefficient that varies across steps acts.** Toy (`evidence/adamw_clip_toy.*`), end-point difference as a share of the run's movement:
  constant ×0.3 gives 5.3e-5; alternating ×1 / ×0.3 gives 0.125; the real clip with a varying N_t gives 0.091.
* So the check asks **whether the term changes WHICH steps are clipped**: CTRL-CLIP, 8 real steps of C-form and P-form, b_t = [c_t < 1].
  * CLIP-SAME launches. **CLIP-T launches with a scope tag** (PL4). An invalid probe blocks the launch (PL3).
* **Why the gate smoke is not used:** at `--accum` 2 its norm is noise-inflated. T's per-micro-batch RMS norms are 1.6–3.0× the 8-pooled ones.

**Decisions for the MM before freezing.**
* **(a) PL4 launches rather than vetoes.** A veto has no lever-free next arm: a smaller w breaks r ≥ 1, and a new clip threshold needs a C′ / P pair
  (~28 L40-h, new card). **Recommended: launch.**
* **(b) The full-run N_t stays unlogged.** Logging it would add a non-string `train.py` hunk, which voids §3.1.3 and §4.2 and changes CTRL-S0's key
  set. **Recommended: probe only.** ⛔ **SUPERSEDED by MM decision 11 (Q-P18): the logging is kept.** It is measured behaviour-neutral, and its
  keys are P-only keys on the step-0 allow-list.
* **(c) The probe counts as an instrument, like the weight rule,** not as a training run. It runs gate-validated argvs and keeps no weights.
  **Please confirm.**
* **(d) `READ_TOOL_SPEC.md`** needs `ctrl_clip` and `--clip-probe`. Not written yet.

**Observation, out of scope and NOT acted on (ESTIMATED, low confidence).** This assumes i.i.d. micro-batches and n = 8.
* T's per-micro and pooled norms give E‖64-pool‖² ≈ (9·pooled² − mean per-micro²)/8.
* That estimate puts l_score on S at about **0.009**, against **0.099** 8-pooled, and L_rank on S at w\* at about 0.15. So the rule's r may be
  noise-dominated: r(1/32) would be about 17 on a 64-micro-batch window, against the 1.75 the rule read.
* It is the same tool for T and P, so it does not distinguish them, and it is not a clip effect. The rule is frozen in the parent; a fix would be a new
  rule card.

### Q-P14 · OP6's committed next arm: P-gatefree is subsumed, so what replaces it? Proposed: P-top8
**Why it changed.**
* **P-gatefree**, NC and DAC stop-gradiented inside s, existed to stop the term from lowering a safe pick's gates. Decision 10 already does that,
  label-consistently: on a safe pick, NC and DAC receive exactly 0.0.
* It is also too strong: it would stop the **correct** push on an unsafe pick's failing gate.
* So it is **subsumed**, and OP6 + Z↑ needs a new next arm.

**What OP6 + Z↑ would mean under routing.**
* The facts: P made the pick separated worse, the oracle was not damaged, the regret did not fall, and P zeroes **more** picks.
* The term can no longer lower a safe pick's gates. It pushes an **unsafe pick's** failing gate down, which is correct.
* So the extra zeroed picks must come from unsafe proposals that **were not the training-time pick** and became the pick on navtest. Two paths fit:
  * they ranked 2nd to 8th and were never anchored;
  * the live upward gate gradient on safe j's generalised through shared weights to nearby unsafe proposals (card §1.1, scope).

**Proposed: P-top8.**
* The same routed, gap-weighted, decision-level term, but **anchored on every proposal in the scorer's own top-8** by `navsim_v1`, not on the argmax
  alone.
* Pairs (j, i) for every i in the top-8 and every j with PDMS_j > PDMS_i.
* Routing per pair: the anchor's factors are live only where y < p, the better proposal's only where y > p. A proposal can be both an anchor and a
  better member of different pairs, so the implementation uses two routed copies: s̃↓ for anchors and s̃↑ for better members, with z_j − z_i =
  s̃↑(j) − s̃↓(i). Both equal s in value.
* Same normalisation (constant 1/`--batch`). New head key `pick8`. Own card; C reused; ~14.2 L40-h + weight rule; PI go.

**Why this one.**
1. It targets the measured failure directly. Unsafe proposals the scorer ranks near the top get pushed below the safer ones before they can become
   the pick.
2. It keeps decision 10. Every gradient is still label-consistent.
3. It keeps the decision-level weighting (the PDMS gap).
4. It adds **no new number**: 8 is the registered `SCORERFT_RANK_GATE_TOPK`.
5. It is the smallest extension of `pick`. k = 1 recovers P exactly, and a test can pin that identity.

**Rejected alternatives.**
* **`pick,gate`** (existing heads, no new code). The `gate` z = logsigmoid(NC) + logsigmoid(DAC) lowers **both** gate logits of the unsafe member.
  For a collision-unsafe proposal with DAC label 1, that is a push against the label, so it violates decision 10. T also ran `gate` pairs and still
  saw zeroed picks +1.26 pp [+0.97, +1.55].
* **P-detach** (`--rank-detach-ctx`). That is OP4 / PL2's arm, for generation damage or trunk pull. It does not touch the ranking of unsafe
  near-picks.
* **Heads-only.** That family was measured as swamped on P6 (parent card §2).

**OP6 + ¬Z↑** is unchanged: it goes to OP7's next.

**Recommended: adopt P-top8 as OP6 + Z↑'s next arm** (it is already in the card's §9).

### Q-P15 · Routing is a LOGIT-level guarantee, not a parameter-level one. Is that acceptable as the decision's scope?
**What is guaranteed.** The term never moves a proposal's own head output away from that output's label: an exact 0.0 by stop-gradient.

**What is not.** The head weights (`score_head`), `score_dec`, `score_q_mlp` and the trunk are shared across the 64 proposals and across tokens, so a
live gradient on one proposal still moves other proposals' outputs. Two consequences:
* a parameter-level zero (for example, the NC weight row of `score_head` receiving 0.0) is neither expected nor claimed;
* a residual against-the-label drift on *other* proposals stays possible.

That residual is exactly what M2 / Z↑, G1 and the within-token AUCs read, and what OP6's P-top8 answers.

**Recommended:** accept the logit-level scope, as written in the card. Do not add a parameter-level guard: it would mean per-proposal parameter
copies, an architecture change.

### Q-P16 · Partial labels: NC = 0.5 and continuous EP make the routing value-dependent. Pin the edge case?
* **Binary labels** (DAC, TTC, C, and NC ∈ {0, 1}): the live mask is a function of the label alone, because σ ∈ (0, 1).
* **NC = 0.5** (NAVSIM partial credit) and **continuous EP**: the mask depends on p.
  * On the pick, the factor is live iff p > y. On a better j, it is live iff p < y.
  * At an exact tie p == y, the factor is not live (strict inequality, as decision 10 states).

**Recommended:** pin both in CTRL-ROUTE (a).
* An NC = 0.5 pick with p_NC = 0.7 is live: its gradient is non-zero and lowers it.
* With p_NC = 0.3 it is not live: exactly 0.0.
* An EP tie is exactly 0.0.

These are cases inside the frozen rule, not a new lever.

### Q-P4 · P must run on C's hardware and software, or "exactly equal" at step 0 cannot hold.
C ran on an NVIDIA L40 with torch 2.7.1+cu128 (`pod_config_c.json`). On another GPU class, or another torch build, bf16 autocast, `--compile` and tf32
kernels can change the last bits of the step-0 losses without any code change.

The Stage 1 demonstration (Q-P1) held exactly because T and C ran on the same pod.

**Recommended:** the same L40 pod and the same image (card §4.5, checked by CTRL-CODE). If P cannot run there, re-run C as C′ first on P's pod
(+14.2 L40-h, a new PI go). ⛔ Never loosen CTRL-S0 to a tolerance after the fact.

### Q-P5 · OP5 ("regret falls, PDMS does not"): run seeds next, or detach the context?
By identity Δregret = Δoracle − Δpick. So with M4 not damaged, OP5 fires only when the oracle drifts down without separating, or when the two CIs
straddle their bars.
* **Seeds** answer *"is the decision gain real?"* and measure the rig's floor F, which is unmeasured.
* **Detaching the context** answers *"did the trunk path give the gain back through generation?"*.

**Recommended: seeds (Ps + Cs, ~28.4 L40-h)** under a successor card, frozen before Ps / Cs are read, with a pooled two-seed bar.
* Cs is needed by every SCORER-FT verdict in any case.
* If the successor card then shows a real gain eaten by an oracle drift, the P-detach arm follows.
* Splitting OP5 now on the sign of the oracle point estimate would decide on a non-separated number.

### Q-P6 · MM decision 1 (normalisation). Agree, with one consequence to record.
The constant `1/--batch` with no-pair sets counted keeps `--accum` exact (`train.py:1858-1871`), and it matches `ep` / `gate` / `ttc`.

**The consequence.** The term's effective strength **falls as P learns**: as navtrain top-1 rises, fewer sets carry a pair.
* C's navtest top-1 is 48.0 %, so roughly half the sets carry pairs at init.
* The weight rule measures r(w) at C1's init only. That is the same for T, and it is consistent: the rule is defined at init.
* Routing does not change the pair count. It changes only which factors carry gradient.

**Recommended:** keep decision 1. Report `rank_sets_pick / batch` over the run beside M5, descriptive only, so that a decay can be seen.

### Q-P7 · Controls added beyond MM decision 3. Accept them?
Decision 3 named CTRL-S0 and kept CTRL-W0, and decision 10 added CTRL-ROUTE. The card adds five more:
* **CTRL-ORDER:** the data stream is equal after step 0. Its Stage 1 demonstration is 0 of 1,000 steps differing.
* **CTRL-CODE:** the pod configs agree on device, torch, source shas, parsed args (defaults included) and data identities.
* **CTRL-HARNESS:** P's seam comes from the same harness export `751f3ae7` as C's.
* **CTRL-CREUSE:** the C in the read **is** Stage 1's C.
* **CTRL-WRULE:** P's `--rank-w` equals the banked rule output.

Each costs zero GPU and has a RED arm in the spec. Without CTRL-HARNESS, a P seam built on a newer harness tip would confound P − C, and no other
control would see it. **Recommended: accept all five.**

### Q-P8 · Reusing C's rows. Re-score C in P's pass; do not read Stage 1's rows beside it.
`field_read.load_slim` (`PK/eval/field/field_read.py:68-86`) replaces a token's whole `arms` dict whenever the token appears in a later file. Feeding
Stage 1's `pass1` rows (arms T, C, base) beside a P-only pass therefore drops C or P for every token. `per_token` would then refuse, so this fails loudly
rather than silently, but it fails.

**Recommended:** one transp_score pass with P, C and base. C is re-scored from its sha-pinned Stage 1 seam, and CTRL-CREUSE proves the result equals
Stage 1's C: levels and selection-quality levels to 1e-9, and the CIs too, since the draw matrix depends only on the logs, the draw count and the seed.
The cost is one extra arm in a CPU pass. A merging loader would be a new reader of shared row files, and that is a larger and riskier change.

### Q-P9 · The arm label in the outputs: `P_minus_C`, not `T_minus_C`.
Keeping the parent's JSON schema would make the tool change smaller. But every quoted number would then read *"T − C"* for an arm that is P: the
arm-label inversion class (CLAUDE.md, the `flagship4b` row).

**Recommended:** label-correct keys under `--card pick` (spec §1.7), and the parent schema byte-identical under the default card. The Stage-1
reproduction step (runbook B1-4) proves the second part.

### Q-P10 · OP1's next arm: the dose pair P₂ₖ vs C₂ₖ (~57 L40-h), or straight to the generator?
If P is adopted but BELOW GOAL (C is 92.67 and the goal is 94.55), the selection line's next lever is either more of the same term or carrying `pick`
into TEACHER-N7.

**Recommended:** the dose pair first. It extends a lever that has then been shown to work, on the parent's named alternative N = 2,000, while
`pick` also enters the TEACHER-N7 card whenever that card is written. Both need a PI go. If the PI prefers to save the ~57 L40-h, TEACHER-N7 with
`pick` on is the defensible alternative.

### Q-P11 · Should the launch gate refuse the retired T / Ts profiles outright?
MM decision 9 is not to re-pin T's stale `SCORERFT_RANK_CENSUS["rank_loss_sha256"]` (`c0e027d2…`). After the P land, `rank_loss.py` changes again; the
routed version supersedes the o3 copy's `3f782697…`. That is right for a retired arm.

But nothing stops `refe_scorerft_t` / `_t_s1` from being launched by mistake. Such a launch would fail only by accident, at G-LIVE L2, and only when
its smoke sees zero gate pairs. T's banked step-0 row shows `rank_pairs_gate` = 0, so that case is real.

**Recommended:** a one-line, zero-GPU follow-up. `scorerft_refusals` refuses arms `T` / `Ts` with the message *"RETIRED: H-SCORERFT-1 Stage 1 read O3
(2026-11-24)"*, and a test pins it. That is not a re-pin and not part of this card. Optional.

### Q-P12 · The power of P1 is set from T's CI width. Is that the right reference?
The card estimates the MDE at 80 % power as 0.44 points (§5.4), from the Stage 1 T − C width: same tokens, same C, same estimator. The paired width
scales with how many picks the treatment changes.

P anchors on the current pick and moves it only where a better proposal is labelled. With routing, its gradient is masked further. Both make it likely
to move fewer picks than T did, which would give a narrower CI and a smaller MDE.

**Recommended:** quote 0.44 as ESTIMATED and conservative. As the card's §5.4 says, routing does not change the numbers. P3's power is unknown,
because F has never been measured on this rig.

### Q-P13 · Who fills the pins, and when? (The three pins below are SETTLED in card §4.10 with MM decision 11, Q-P18.)
| pin | filled by | when |
|---|---|---|
| `PICK_PREREG_SHA256` | MM | in the freeze-and-land commit (card §4.10) |
| `PICK_TRAIN_SHA256` | MM | in the freeze-and-land commit, from the **landed** blob (card §4.10); expected `b1f47518…` (o3c, with the clip logging), **not** `fd5e5dad…` (Q-P18) |
| `PICK_CODE_TAG` | MM | at launch |
| `SCORERFT_P_RANK_W` (in `launch_gate.py`) | MM | from the rule JSON, run on the **routed** build |
| `CLIP_PROBE_SHA256` | MM | recorded in LOOP_STATE after CTRL-CLIP and before P's launch; the constant lands in a later commit, before the read (card §4.10) |

`C_METRICS_SHA256` / `T_METRICS_SHA256` are already known (`97a8631c…` / `09cdaf83…`), and the fallback pin `PICK_S0_FORM` is removed.

**Recommended:** record all five in LOOP_STATE beside the card's sha256. A real pick read refuses while any of them is `None`.
