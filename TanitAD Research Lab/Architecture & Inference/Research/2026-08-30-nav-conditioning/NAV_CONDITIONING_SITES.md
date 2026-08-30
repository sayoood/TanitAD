# NAV WIRING — the conditioning sites, established BEFORE injecting

**Date:** 2026-08-30 · **Owner:** TanitAD_TrainingFlyWheel · **Evidence class:**
MEASURED (source, file:line). Written before any injection, per the Master Mind's
instruction to name the real conditioning point at each layer first.

---

## 1. The operative site IS the action site — and the pattern is ADDITIVE

`stack/tanitad/models/predictor.py:199-207`:

```
cond = self.act_emb(actions)                  # [B, W, D]
if intent is not None:
    term = self.intent_proj(intent).unsqueeze(1)
    if self.intent_gate is not None:
        term = self.intent_gate * term
    cond = cond + term
for blk in self.blocks:
    x = blk(x, cond, mask)                    # FiLM consumes `cond`
```

⇒ **Nav enters here, as a third additive term**, which is exactly the PI's
*"conditioning the WM like the actions"* — same variable, same pathway, same FiLM
consumer. ✅ The answer to *"is the operative nav site the same place actions
enter?"* is **YES**, and that is what makes the directive implementable without a
design divergence.

⛔ **ADDITIVE, NEVER CONCATENATED.** The v6 docs state a **shape change bypasses
`STAGE_MAY_INTRODUCE`'s adjudication**, and `load_state_dict(strict=False)` still
RAISES on shapes (measured). Concatenating nav would widen the conditioning and
slip past the gate that decides what a stage may introduce.

## 2. ⛔⛔ THE FINDING THAT CHANGES THE DESIGN: H26

`predictor.py:126` records, MEASURED:

> the ungated `intent_proj` norm **~31.4 COMPETED WITH `act_emb` ~28.3, diluting
> the action conditioning; engaging intent measured net-harmful to the operative**

The fix there was a **ReZero-style learnable gate init 0.1**, *"so it starts
action-dominant and grows only if training earns it"*.

⇒ **Nav would be a THIRD competitor in the same sum.** An ungated nav term
reproduces H26 with one more voice, and H26's outcome was *net-harmful*.

⭐ **THE DESIGN CONSEQUENCE, and it is a real qualification of the directive:**
the PI's *"condition the WM like the actions"* is right in **kind** — nav is an
INPUT, not a head, entering the action pathway. But *"like the actions"* must
**not** mean *"at equal magnitude from step 0"*, because that exact configuration
is the one already measured harmful. ⇒ `NavConditioner` carries a **per-layer
ReZero gate (init 0.1)** and a **zero-init output projection**, so the channel is
present, mandatory, and starts inert — growing only if training earns it.

⚠️ **This is a design decision I am flagging, not hiding**, because it is the
nearest thing to a deviation from the directive's letter. It is architecturally
the directive as written (nav conditions all three layers, as an input, at the
action site); the gate governs the term's initial *magnitude*, not its presence.

## 3. ⛔ CORRECTED — the tactical and strategic sites are NOT additive

**This section originally said the tac/str layers use "the same additive
construction" as the operative site. THAT WAS WRONG**, and it would have died at
the first forward pass. I took it from the v6 **docstring** describing the
`tac_goal_cond` port instead of opening `FTac`.

⚠️ **The generalisable form, and it is not "read source not docs" — we both
already knew that.** It is that **a doc DESCRIBING a mechanism reads exactly like
a doc SPECIFYING it**, and only opening the implementation distinguishes the two.
The v6 docstring was accurate about `tac_goal_cond`; I read it as a statement
about `FTac`. *(Same week, the v7-labels spec was wrong in three places by this
mechanism and measuring against the data found all three — the trap caught both
sessions from opposite sides.)*

**What the source says** — `tactical.py:267-272`:

```python
h = self.in_proj(torch.cat([z_tac, g_flat], dim=-1))   # Linear(d_tac + d_goal, hidden)
```

⇒ **`FTac` CONCATENATES. There is no additive `cond` pathway at these layers at
all.** Widening the cat changes `in_proj`'s shape, which (a) bypasses
`STAGE_MAY_INTRODUCE`'s adjudication, and (b) ⛔ **makes every existing checkpoint
UNLOADABLE** — breaking exactly the comparability the PI ruled on.

⭐ **The precedent already in the tree solves it** — `cond_tac_dyn`
(`v6.py:4904-4907`, applied at `:5555`) faced this problem and answered it:

```python
self.cond_tac_dyn = nn.Linear(d_goal_embed, 2 * d_goal_embed)   # zero-init
g_cond_tac = e_a_tac + self.cond_tac_dyn(self._cut(e_g_str, cut))
```

**Project INTO the existing `g_flat` width and ADD.** No shape change, so the
strict-subset property the operative port has is preserved here too.

⇒ Nav therefore needs **PER-LAYER WIDTHS**, not one `d_model`:

| layer | width | added to | pathway |
|---|---|---|---|
| operative | predictor hidden `d` | `cond` after `act_emb` | additive FiLM |
| tactical | `2 * d_goal_embed` | `e_a_tac` | project-and-add |
| strategic | `d_goal_embed` | the strategic `g_flat` | project-and-add |

✅ **Landed**: `NavConditioner(widths=...)`, with `d_model` retained only for the
uniform-width test case and a `ValueError` if neither is given. 24 tests green.

## 4. ⛔ What must still be done before an arm can train

1. **Thread nav through the collate path** so a batch WITHOUT nav raises THERE
   too, not only in the module. ⚠️ A guard that fires only deep in the model is
   one the data path can route around — the batch dict is a **whitelist**
   (measured: a smoke test died `KeyError('ep_idx')` while the dataset, the mirror
   sync and the import were all correct), so nav must be added to that whitelist
   explicitly or it will silently never arrive.
2. Instantiate `NavConditioner` inside the v6 stack and add the term at the three
   sites, under `STAGE_MAY_INTRODUCE` adjudication.
3. Fit `NavArgStats` on the **fit split only** and record it in `config.json`.
4. Stamp provenance into `config.json` and every eval record.

⛔ **NONE of that is done.** The module and its gates exist; the injection does
not. `v6.py` still contains 2 occurrences of "nav", neither a wiring — unchanged
by me.
