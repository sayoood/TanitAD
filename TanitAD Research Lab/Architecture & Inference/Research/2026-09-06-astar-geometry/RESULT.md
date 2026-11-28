# RESULT — `D-REFCV4B-ASTAR-GEOMETRY` FIXED; the ceiling is a CEILING again

**2026-09-06 · a_star-geometry agent · branch `agent/arch-inf-20260803` · staged, not committed to `main`, not pushed.**
Pre-registration: `PREREG_D-REFCV4B-ASTAR-GEOMETRY.md` (same directory), written **before** any
measurement of the fixed binding.

---

## 1. The one-line answer

**Yes — `oracle_sel` is a ceiling again, and a selector CAN now be scored on the corrected
convention.** Under the corrected binding the oracle sits **BELOW** the arm it bounds
(**0.1993 m** best-in-fan vs `os` **0.2970 m**, 24,114 windows — `D-SELQ-ASTAR-10`, INHERITED),
where the defect had it **4× above**; my own independent measurement on the real checkpoint
reproduces the mechanism and its sign. ⛔ **One thing is still open:** the tool's own
`oracle_sel` *arm* number (the a_star anchor's **learned refinement**) has not been re-rolled,
because the re-roll is blocked at HEAD by an **unrelated** model-drift refusal (§6). The ceiling
property is settled; that one arm's headline figure is not.

---

## 2. The trainer function mirrored

**`compute_losses_v3`, `stack/scripts/refc_v3_train.py`.** Its `anchors` / `dist` / `a_star` triple
is the programme's single definition of the GT-nearest anchor, and it carries a comment naming this
exact failure — that on a v0-conditioned vocabulary `decoder.anchors` is *"the family rolled at the
REFERENCE speed, not this window's fan"*, so scoring against it supervises *"a geometry the model
never emitted — silently, and with `anchor_acc` still reading plausibly."*

The eval arm already matched the trainer on **everything except the bank**: same target function
(`refb_labels.waypoint_targets`), same valid-slots-only mask, same squared-error argmin, same
readout row. It hoisted `anchors_bank = model.core.decoder.anchors` out of the per-window loop —
and on a v0-conditioned vocabulary **that bank is not loop-invariant**.

The eval arm's own inline comment claimed it computed `a_star` *"exactly as the trainer computes
it."* That was true until the trainer was corrected on **2026-09-04** and was never updated. **Two
consumers, one convention, one of them stale** — which is the root-cause class, not the line.

### The fix

`taniteval/tools/refcv3_arm.py`, three changes:

1. The hoisted binding is **deleted**, with a comment saying why it must not be re-hoisted.
2. `a_star` is taken **per window** from `_decoded_bank(out, row=0)` — the forward's own
   `anchor_bank`, row 0, the row every consumer reads `anchor_traj` back from.
3. The two byte-identical branches (one under `--with-oracle-sel`, one not) are collapsed into
   **one unconditional computation**, because `anchor_acc` and `sel_agrees_oracle` consume `a_star`
   on **every** roll.

Two named helpers now carry the convention: `_decoded_bank` (which **refuses** rather than falling
back — a silent fallback to `decoder.anchors` *is* the defect) and `_oracle_anchor_index` (the
trainer's arithmetic, in one place). **AST-verified: zero executable `.decoder.anchors` sites
remain** in the tool; the two textual matches are this document's own prose in comments.

---

## 3. The pre-registered ceiling assertion — where it stands

### 3.1 What the defect actually was, confirmed EXACTLY on the real checkpoint

`refc.py`'s `roll_bank` branches on `anchor_v0_cond`. On the real `refcv4b@40284` checkpoint
(`ckpt_40284_FINAL.pt`, argv carries `--anchor-v0-conditioned` and
`--anchor-control-units alat`), loading the two persistent buffers and re-rolling gives

> `anchors == roll_bank(ref_speed)`, **max |Δ| = 0** — exact.

⇒ The checkpoint's `anchors` buffer **is** the family at the reference speed, bit-for-bit, exactly
as `refc.py` documents. The old binding therefore fed the argmin the reference-speed geometry while
the readout came from the window's own fan. Index *i* names a different trajectory in the two, so
the "oracle" returned the refinement of an anchor that is **not** the GT-nearest one — near-arbitrary
selection wearing an oracle's name. That is the mechanism of a ceiling reading **+0.9179 m
separated WORSE** than the arm it bounds, and of `anchor_acc` / `sel_agrees_oracle` sitting at
**0.0993**.

### 3.2 MEASURED — the real refcv4b vocabulary, the real eval windows

`raw/real_arm_astar.json` · `raw/real_arm_astar.py` · **n = 168 windows / 12 episodes** of the
141-clip B1 eval cache, stride 10 · **T0** (oracle selection) · evidence class **MEASURED (ours)**.
Both bindings fed to the **shipped** `_oracle_anchor_index`; the bank from the **real** `roll_bank`.

| | corrected binding | defect binding | Δ |
|---|---|---|---|
| `a_star` changed by the fix | — | — | **162 / 168 = 96.43 %** |
| anchor error, 6 s (mean L2 over 8 slots) | **1.2576 m** | 9.6741 m | **+8.4165 m** |
| anchor error, **2 s slot** (the reported grid) | **0.6732 m** | 3.7174 m | **+3.0442 m** |
| windows the defect made WORSE / BETTER | — | — | **162 / 0** |

Paired **episode-cluster bootstrap** on the per-window 2 s delta (2000 resamples, 12 episodes):
**[+2.3139, +3.7673]**, separated. ⛔ `overlapping_holdout_se` not used anywhere.

⇒ The defect degraded the oracle's starting geometry by **5.5×** at 2 s, one-signed, on 96 % of
windows. **The selection component of `oracle_sel` is a ceiling again.**

### 3.3 A1 — SUPPORTED, on an independent measurement at 143× my n

⭐ **The pre-registered assertion holds.** `D-SELQ-ASTAR-10` (register; **INHERITED** — a sibling's
measurement, not re-verified by me) rebuilt the per-window v0-conditioned fan the same way this fix
does — *"model-free from `anchor_controls` + `v0` through the programme's own integrator"* — and
over **24,114 windows** reports:

| binding | candidate ADE |
|---|---|
| defect (`decoder.anchors`) `a_star` | **1.4491 m** — worse than the arm it bounds |
| model's own SELECTED candidate | 0.4252 m |
| ⭐ **corrected binding, best-in-fan** | **0.1993 m** |
| refined output `os` | **0.2970 m** |
| uniform-random candidate | 2.3399 m |

⇒ **0.1993 ≤ 0.2970: the corrected oracle is at or below the arm it bounds — the definitional
test the pre-registration named**, and the two argmins agree on only **9.52 %** of windows, so the
defect and the fix are measuring genuinely different objects. **A1: SUPPORTED.**

⚠️ **Two scope limits I will not paper over.**
1. Both that number and mine are **CANDIDATE-level** (the raw anchor). The tool's `oracle_sel`
   *arm* emits the a_star anchor's **LEARNED REFINEMENT**, which needs a forward pass. The ceiling
   *property* is established; the refined `oracle_sel` *figure* is not re-measured.
2. My 2 s figure (0.6732 m) is **not comparable** to their 0.1993 m and must not be quoted against
   it: mine argmins over the full **6 s** horizon and then reads the 2 s slot, on **168 windows /
   12 episodes** at stride 10, including `v0 = 0.00` windows. Different selection objective,
   different grid, different n. It is a **mechanism and sign** measurement, and that is all I claim
   for it.

### 3.4 ⛔ What remains genuinely open

`oracle_sel` is *the a_star anchor's **LEARNED REFINEMENT***, `out["anchor_traj"][0, a_star]`. The
refinement needs a forward pass. What is measured above is the **raw anchor** the refinement starts
from. So:

* **The ceiling property — SETTLED** (§3.3), at candidate level, on 24,114 windows.
* **The `oracle_sel` arm's own headline number — NOT re-measured.** It is the refinement, it needs
  the forward pass, and the re-roll is blocked (§6).

⛔ **Anyone quoting a corrected `oracle_sel` *arm* figure before that re-roll is quoting a number
that does not exist.** The ceiling claim and the arm's figure are different objects and this
document keeps them apart.

---

## 4. refcv3 as the control — UNCHANGED, and structurally so

`roll_bank`'s fixed branch is:

```python
n = self.anchors.shape[0]
if not self.anchor_v0_cond:
    return self.anchors.to(dtype)[None].expand(batch, n, self.n_steps, 2)
```

⇒ On refcv3 (`--anchor-v0-conditioned` **absent**, verified in its `config.json` argv) the corrected
binding and the old one are **the same storage**, not merely close. The fix is a no-op there **by
construction**, and the unit gate asserts it with `torch.equal` rather than trusting it.

Two further checks that the no-op is real:

* **No autocast anywhere in the eval path** (`model.to(device).eval()`, no dtype cast), so
  `out["anchor_bank"]` is fp32 — the same dtype the old `decoder.anchors` buffer had. There is no
  precision round-trip hiding in the change.
* **`stack/tests/test_refcv3_arm.py`: 20 passed** on the corrected tool, plus
  `test_render_refcv3_video.py` — **34 passed** together with the new gate.

⚠️ **Scope, stated honestly:** refcv3's *banked* numbers (`oracle_sel` 0.3668 under `os` 0.4419) were
**not** re-measured by a roll — the same blocker in §6 stops refcv3 too. The claim that they are
unchanged rests on the structural identity above plus the `torch.equal` gate, which is a **stronger**
argument than a single re-roll would have been, but it is a different kind of evidence and is
labelled as such.

---

## 5. The two-sided mutation proof — COMPLETE

Gate: `taniteval/tests/test_refcv3_arm_astar_geometry.py` (5 tests).
Harness: `raw/run_mutation_proof.py`, which builds a mutant copy re-introducing the defect.
⛔ Deliberately **not** a source census — an AST census reads identically on a fixed and a broken
trainer, which is a lesson this programme has already paid for. The gate is behavioural, on the
**real** `AnchoredDiffusionDecoder` and the **real** `rollout_unicycle`.

| clause | outcome |
|---|---|
| corrected binding | **5 / 5 PASS** |
| defect re-introduced | **RED** — `test_v0_conditioned_mutation_is_detected` and `test_wrong_binding_inverts_the_ceiling` both FAIL |
| ⭐ fixed vocabulary under the same mutation | **GREEN** — the defect is invisible on refcv3 |

The third row is the load-bearing one: it is *why* the defect survived review, and it proves the gate
is specific to v0-conditioned builds rather than firing on everything. Fixture margin: **44/48
windows changed, +11.5861 m**, with thresholds set an order of magnitude below that.

⚠️ **One honest correction found while building the gate.** The first version asserted that the
oracle attains the fan's **ADE** minimum. It does not, in general: `a_star` is the argmin of
**summed squared error**, and ADE is a mean of L2 norms — different norms, whose argmins need not
coincide (**MEASURED 47/48** agreement on an out-of-span fixture, 48/48 on the corrected one). The
gate now asserts the **SSE** invariant, which is exact. ⇒ This is also a standing caveat on the
metric itself: **`oracle_sel` is an EMPIRICAL ceiling — oracle selection plus a *learned*
refinement, scored in a norm it was not selected in — never a mathematical bound.**

---

## 6. ⛔ THE BLOCKER — refcv4b (and refcv3) are UNROLLABLE at HEAD

The `--with-oracle-sel` re-roll was launched and **refused**, correctly, before the GPU:

```
[refcv3_arm] ⛔ config.json CONTRADICTS the rebuilt model — refusing rather than
guessing which describes the weights:
  param_breakdown: config.json {... 'total': 107058488}
             vs rebuilt {... 'tac_goal_tok_head': 11286, 'total': 107069774}
```

**Cause (MEASURED at source, not inferred):** `refc_v3.py` builds `tac_goal_tok_head`
**unconditionally whenever `tac_vocab_version != "kin3"`**. Every checkpoint on this box is
`v7.0` — refcv4b **and** all three refcv3 checkpoints — and **none** carries the head in its
recorded `param_breakdown`, because all of them predate it.

⇒ **This is not my defect and it is not refcv4b-specific: it currently makes every pre-drift
v7-vocabulary checkpoint unrollable by this instrument.**

⚠️ **The drift is RECENT, and that is evidence, not a guess:** `D-SELQ-XVAL-14` records a
successful local refcv4b re-roll (4,927 windows, stride 1) reproducing the published lateral
headline to 1.5 %. That roll therefore predates the head. ⇒ the refusal I hit is a **new**
regression against a previously working path, which raises its priority rather than lowering it.

**The head is provably inert for this eval** — verified by reading four files with a same-breath
non-zero control: `tac_goal_logits` is **written once** (`refc_v3.py`, into the cache) and **read
nowhere** in `refc_v3.py`, `refc.py`, `refc_v3_train.py` or `refcv3_arm.py`. It cannot touch `traj`,
`anchor_bank`, `anchor_traj`, `sel_idx`, `anchor_logits` or `a_star`.

⛔ **I did not route around it.** Loosening my own tool's `param_breakdown` cross-check to let my own
arm through is exactly the move this programme forbids, and the guard is shared. Escalating instead.

### ESCALATION — exact change requested, `stack/tanitad/refs/refc_v3.py` (NOT mine)

Gate the head on an explicit config flag so a pre-drift checkpoint rebuilds identically, e.g.

```python
self.tac_goal_tok_head = None
if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", True)):
```

with the flag recorded in `config.json`, **or** any equivalent that makes the rebuild reproduce the
recorded `param_breakdown` for a checkpoint that predates the head. The alternative — a
narrowly-scoped, **opt-in, recorded** tolerance in `refcv3_arm.py` for a head that is
(a) absent from the config breakdown, (b) absent from the state_dict and (c) provably unread — is
mine to build and I will, **on request**; I have not done it unasked because it weakens a guard that
is currently refusing correctly. There is precedent in the same function
(`tolerated_inert_buffers`), which is why the option is named rather than dismissed.

### `C3_os_reproduction` — ⛔ NOT settled

It could not be: settling it requires the very roll that is blocked. It therefore **remains FAIL**,
and **both refcv4b number families stay unquotable** until the blocker clears and the paired roll is
done on one surface in one process. ⚠️ Note the blocker also means the *contaminated* numbers were
produced against a **different model definition** than HEAD — a second, independent reason the
re-roll cannot be skipped.

---

## 7. What the fix unblocks (stated, not built)

Once the §6 blocker clears and the paired re-roll lands:

* **WP-7's selector scoring** — a selection metric that is no longer measured against a geometry the
  model never emitted.
* **A valid regret metric** — selected anchor vs best-in-fan by the **four families**
  (longitudinal / lateral / tactical / strategic, never pooled, never ADE alone), built on
  `_decoded_bank` and **without** the broken binding. Read LATERAL on **curvature MAE with the
  straight floor beside it**; ⛔ never the `|dyaw| > 0.15` turn gate — the human fails it 3/9.
* ⭐ **`D-SELQ-ASTAR-10` and `D-SELQ-REGRET-11` already rest on the corrected convention** (they rebuilt the per-window fan by hand). This fix is what moves that convention into the SHIPPED instrument, with a gate, so the next consumer inherits it instead of re-deriving it — which is the failure mode that produced the defect.
* **`anchor_acc` as a selector metric**, and `sel_agrees_oracle` with it — both are computed from
  `a_star` on **every** roll, so every refcv4b value of either, from any past roll, is contaminated
  regardless of whether that roll opted into `--with-oracle-sel`.

⛔ **No selector was built.** WP-7 is a sibling's scope and is separately escalated as DD's **deep
supervision**: `(reg, cls)` applied at **every** cascade layer, summed, with the cascade
**gradient-detached between layers**, under a **sigmoid focal loss (γ = 2.0, α = 0.25)**. REF-C has
**neither half**. Referenced, not implemented.

---

## 8. Evidence discipline

* Every number above is **MEASURED (ours)** with its artifact path, or quoted from the brief as the
  prior contaminated reading. **T0** on every oracle number; ⛔ never compared to a T1 arm.
* Estimator named on every interval: **paired episode-cluster bootstrap**. ⛔ `overlapping_holdout_se`
  is not used.
* ⚠️ **A single separated CI is necessary, not sufficient** (`H-ESTIM-SEED-1`: 6/42 = **14.3 %**
  false-`separated` on the tiny rig). The §3.2 result does not lean on that: it is a
  **determinism** comparison — one checkpoint, one window set, two bindings — so it is established
  by **exact index agreement/disagreement (162/168, 0 reversed)**, and the CI is reported as a
  magnitude, not as the existence proof.
* ⚠️ **Every number carries its roll and its n.** §3.2 is **n = 168 windows / 12 episodes**, stride
  10, of the 141-clip cache — **not** the 4,823-window grid, and not the 171-window/20-episode
  `sel_refined` panel. Do not quote it against either.
* Vocabulary: the nav command is an **INPUT simulating the vehicle's nav system** — never
  "oracle nav", never "deployment gap".

---

## 9. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| the fix | `taniteval/tools/refcv3_arm.py` | **staged**, blob-verified |
| the mutation gate (5 tests) | `taniteval/tests/test_refcv3_arm_astar_geometry.py` | **staged**, blob-verified |
| pre-registration | `…/Research/2026-09-06-astar-geometry/PREREG_D-REFCV4B-ASTAR-GEOMETRY.md` | **staged**, blob-verified |
| this result | `…/Research/2026-09-06-astar-geometry/RESULT.md` | **staged** |
| real-arm measurement | `…/2026-09-06-astar-geometry/raw/real_arm_astar.json` + `.py` | **staged** |
| mutation harness | `…/2026-09-06-astar-geometry/raw/run_mutation_proof.py` | **staged** |

⛔ Nothing is committed to `main`; nothing is pushed. Compute used: dev-box RTX 4060 and CPU only —
**the A40 was not touched** (reserved for `refcv5-cap-b1-v72-40k`). `OMP_NUM_THREADS=6` throughout;
no concurrent arms.
