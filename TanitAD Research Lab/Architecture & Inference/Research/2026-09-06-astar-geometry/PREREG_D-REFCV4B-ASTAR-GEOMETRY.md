# PREREG — `D-REFCV4B-ASTAR-GEOMETRY`

**Written 2026-09-06 BEFORE any measurement of the fixed binding.** Both outcomes are
committed below. Owner: a_star-geometry agent. Scope: `taniteval/tools/refcv3_arm.py` only.

---

## 1. The defect, localised at source

Two consumers compute `a_star` (the GT-nearest anchor). They disagree on **one term: the bank**.

**TRAINER — `compute_losses_v3`, `stack/scripts/refc_v3_train.py`** (the function this fix
mirrors; it is the definition of `a_star` the eval arm's own docstring already cites):

```python
anchors = out["anchor_bank"].to(traj_tgt.dtype)          # [B, N, S, 2]
dist = (((traj_tgt[:, None] - anchors) ** 2).sum(-1) * sv[:, None]).sum(-1)
a_star = dist.argmin(dim=1)
recon  = out["anchor_traj"][ar, a_star]
```

carrying this comment, verbatim, in the trainer:

> THE TARGET MUST BE MEASURED AGAINST THE BANK THAT WAS ACTUALLY DECODED. With a
> v0-conditioned vocabulary `decoder.anchors` is the family rolled at the REFERENCE speed,
> not this window's fan, so scoring `a_star` against it would supervise the anchor
> classifier on a geometry the model never emitted — silently, and with `anchor_acc` still
> reading plausibly.

**EVAL ARM — `taniteval/tools/refcv3_arm.py`** (pre-fix), binds the forbidden object once
outside the loop and uses it in **both** branches of the per-window `a_star`:

```python
anchors_bank = model.core.decoder.anchors.detach().cpu()    # [N, S, 2]
...
dist = (((tgt[:, None] - anchors_bank[None]) ** 2).sum(-1) * sv[None, None]).sum(-1)
a_star = int(dist.argmin(dim=1)[0])
```

Everything else already matches the trainer exactly: the same target function
(`refb_labels.waypoint_targets`), the same validity mask (`sv`, valid-slots-only), the same
squared-error argmin, the same readout (`out["anchor_traj"][..., a_star]`). **The bank is
the whole divergence.** The eval arm's own inline comment claims it computes a_star
*"exactly as the trainer computes it"* — that comment was true before 2026-09-04 and was not
updated when the trainer was fixed. Two consumers, one convention, one of them stale.

### Why it is arm-specific — the structural control

`RefCDecoder.roll_bank` (`stack/tanitad/refs/refc.py`) branches on `anchor_v0_cond`:

```python
n = self.anchors.shape[0]
if not self.anchor_v0_cond:
    return self.anchors.to(dtype)[None].expand(batch, n, self.n_steps, 2)
# ... else: roll anchor_controls through rollout_unicycle from THIS window's v0,
#     with alat -> kappa conversion, at the window's measured speed
```

* **refcv3** (`--anchor-v0-conditioned` **absent**): `out["anchor_bank"]` **is**
  `decoder.anchors[None].expand(...)` — the same storage, a broadcast view. The two
  bindings are **the same tensor**. Not "close": identical by construction.
* **refcv4b** (`--anchor-v0-conditioned` **present**): the bank is re-rolled per window
  through the unicycle at that window's measured `v0`. `decoder.anchors` is the
  **reference-speed** family and is a different geometry.

⇒ The defect is **v0-conditioning-specific**, exactly as the observed signature says, and the
fix is a no-op on refcv3 **structurally**, not merely empirically.

### The mechanism that makes the ceiling read below its own floor

`a_star` is argmin'd against the **reference-speed** family, and the resulting **index** is
then used to read `out["anchor_traj"][0, a_star]` — a refinement of the **window-specific**
fan. Index *i* denotes a different trajectory in the two geometries. The "oracle" therefore
returns the refinement of an anchor that is *not* the GT-nearest one in the geometry the
model actually emitted: near-arbitrary selection wearing an oracle's name. That is why
`oracle_sel` reads **+0.9179 m separated WORSE** than the arm it bounds, and why
`anchor_acc` / `sel_agrees_oracle` sit at **0.0993** — agreement with a near-arbitrary index.

---

## 2. The pre-registered assertions

⛔ Registered **before** the fixed binding was measured. Both outcomes committed.

### A1 — PRIMARY: the ceiling must become a CEILING (refcv4b)

> On refcv4b, with the corrected binding, `oracle_sel` ADE must be **at or below** the arm it
> bounds (`os`), i.e. the paired `oracle_sel − os` delta must be **≤ 0** at the reported
> horizons.

* **PASS** ⇒ the binding was the cause. `oracle_sel` is a ceiling again and the selector
  family unblocks.
* **FAIL** ⇒ ⛔ the binding was **not** the cause (or not the whole cause). The finding is
  reported as a FAILURE, the next candidate is named, and no selector metric is unblocked.

⚠️ **Stated honestly up front: `oracle_sel` is an EMPIRICAL ceiling, not a mathematical
one.** It is *oracle selection + **learned** refinement*. Oracle-selecting the GT-nearest
**x0** anchor does not guarantee the best **refined** output, so a small positive delta would
not by itself prove the binding is still wrong. What the defect predicts, and what A1 tests,
is the **large, separated, wrong-signed** gap (+0.9179 m). A residual delta that is small,
not separated, or negative is consistent with A1 passing.

### A2 — CONTROL: refcv3 must not move

> Every refcv3 number — `oracle_sel`, `anchor_acc`, `sel_agrees_oracle`, `a_star` itself —
> must be **unchanged** by this fix.

Basis: the `roll_bank` fixed branch above. If any refcv3 number moves, a working path was
changed and the fix is **withdrawn**. refcv3's banked reading is `oracle_sel` **0.3668**
under `os` **0.4419** — already a working ceiling, which is what makes it the right control.

### A3 — MUTATION, two-sided

> Re-introducing the wrong binding (`model.core.decoder.anchors`) must make the check
> **FAIL** on a v0-conditioned build; the corrected binding must make it **PASS**.
> On a **fixed** (non-v0-conditioned) build the mutation must be **INVISIBLE** — both
> bindings must agree exactly.

The third clause is the one that matters: a gate that fires on refcv3 would be measuring
something other than this defect, and the invisibility on refcv3 **is** the reason the defect
survived review. A gate that cannot fail measures nothing; a gate that fails everywhere
measures nothing either.

### A4 — BLAST RADIUS: the sidecar moves too

> `anchor_acc` and `sel_agrees_oracle` are computed from the **same** `a_star` and are
> emitted on **every** roll, not only under `--with-oracle-sel`. They must move on refcv4b
> and stay fixed on refcv3.

⇒ Every refcv4b `anchor_acc` / `sel_agrees_oracle` ever produced by this tool is contaminated,
independent of whether that roll opted into the ceiling arm.

---

## 3. What is NOT claimed here

* ⛔ No selector is built. WP-7's selector is a sibling's scope, and is separately escalated
  as DD's **deep supervision** — `(reg, cls)` applied at **every** cascade layer, summed,
  cascade **gradient-detached between layers**, under **sigmoid focal loss (γ=2.0, α=0.25)**.
  REF-C has **neither half**. Referenced, not implemented.
* ⛔ No claim that a corrected `oracle_sel` is a *deployment* number. It is **T0** and is
  never compared to a T1 arm.
* ⛔ No claim about run-to-run or inference variance. A single roll's separated CI is
  **necessary, not sufficient** (`H-ESTIM-SEED-1`: 6/42 = 14.3 % false-`separated` on the
  tiny rig). This fix is a **determinism** repair — the same checkpoint, the same windows,
  a corrected index — so it is verified by **exact agreement / disagreement**, which is the
  right instrument for it, and no CI is required to establish the mutation proof.
