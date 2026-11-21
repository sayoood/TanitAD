# SPEC — `--safety-head` in `refe/scorer_finetune.py` (for the implementing agent; default OFF)

Design source: `PREREG_SAFEHEAD.md` s.1-3 (sha256 in `PREREG_SAFEHEAD.sha256`). Line numbers are d06546a. This spec is the trainer-side form of the three designs simulated here.
Implement only the design(s) whose row in `RESULT.md` asks for it; the flag carries all three values so that the matched controls stay one switch apart.

## 1. The flag

`--safety-head {off, frozen, armA, stopgrad}`, default **`off`** (= R0 / R0b / R0c bit for bit). An EPO lever: add `"safety_head"` to `EPO_DESTS` (`:176-177`) so naming it without `--epo` is refused, and to the launch gate's G-DVB table (a new flag without a G-DVB entry is refused by the gate itself).

## 2. The one change in the loss (mirror `epo_reference`, `:194-215`)

```python
SAFE_ROWS = (0, 1, 3)                       # NC, DAC, TTC = rl_reward.safe_mask (rl_reward.py:87-89)
SAFETY_HEADS = ("off", "frozen", "armA", "stopgrad")

def safety_combine(mode, sx0, sxA, sxB):
    """The selection logits arm B's EPO term and arm B's deployed pick read.  off: sxB itself (the same object => bit-identical default path).
    Otherwise the SAFE columns come from S (frozen: the base sx0; armA: arm A's sxA; stopgrad: sxB itself), DETACHED, so EPO's gradient can never reach S;
    the other columns are sxB's.  torch.where also gives sxB's SAFE columns exactly 0 from EPO."""
    if mode == "off":
        return sxB
    S = {"frozen": sx0, "armA": sxA, "stopgrad": sxB}[mode]
    m = torch.zeros(sxB.shape[-1], dtype=torch.bool, device=sxB.device); m[list(SAFE_ROWS)] = True
    return torch.where(m, S.detach(), sxB)          # <- THE SEPARATION LINE (REFE_RL_MUTATE_SAFETY_NO_DETACH=1 drops .detach(): G11-S must turn RED)
```

`train_losses` (`:792-800`) changes ONE argument:
`ep = epo_term(epo_reference(a.epo_ref, sx0, sxA), safety_combine(a.safety_head, sx0, sxA, sxB), tg, rows, upd, lam_pg)`.
Arm B's BCE stays on all 7 of its own rows (`losses`, unchanged); arm A unchanged; base unchanged. **Zero new trainable parameters for all three modes.**

Plus `safety_combine_built(mode)` (sentinel reader like `epo_reference_built`, `:206-215`) recorded in the data event, so G-DVB compares the argv with the selector that runs.

## 3. Gradient diagram (armA; frozen = base in place of A and no BCE on it; stopgrad = S is sxB)

```
cxy, sctx ─► arm A scorer ─► sxA ─┬─(BCE, 7 rows)──────────────► lA ─┐
                                  └─ .detach() ──┐                   │
cxy, sctx ─► arm B scorer ─► sxB ─┬─(BCE, 7 rows)┼────────────► lB ──┼─► ((lA + lB + ep) / accum).backward()
                                  └──────────────┴─► comb = where(SAFE, sxA.detach(), sxB)
                                                     └─► epo_policy / J / KL(base || pi) ─► ep
arm A  ◄── d lA only            (EPO: EXACTLY 0.0)
arm B  ◄── d lB + d ep          (d ep enters through the EP / C / lane columns only; B's SAFE head rows get 0 from ep)
```

## 4. The deployed selection must read `comb` too (the "built, tested, unreachable from its caller" trap)

- `evaluate()` (`:828-` ; the arm-B pick at `:849` is `agg_v1(p).argmax`) and the score dump (`ScoreDump.add`, `:148`) must score arm B's pick on **`sel_score(sigmoid(comb))`**, or the A40 arm's RL-val PDMS / TTC read a selection that is never deployed. Dump `comb` as B's scores (and B's raw logits as a diagnostic).
- Checkpoint meta records `safety_head`; the deployed planner loads the safety scorer (arm A = the model's own `score_*` for `armA`; the base u400 scorer for `frozen`) next to arm B. Inference cost for `frozen` / `armA`: one more scorer forward (4,296,967 stored params, rows 0, 1, 3 of its head used); `stopgrad`: none.

## 5. Gates that must exist before the A40 arm (CLAUDE.md "built, tested, unreachable")

1. **G11-S** (in the G11 probe, `:478-`): one micro-batch, lam_pg 1: `autograd.grad(ep, S's parameters)` reads **exactly 0.0** (armA: `paramsA`; frozen: a grad-enabled copy of `base`; stopgrad: rows 0, 1, 3 of armB's `score_head`), with arm B's EP row and `score_dec` **non-zero** in the same breath; and `REFE_RL_MUTATE_SAFETY_NO_DETACH=1` turns it **RED**. (The CPU form of exactly this probe passed here: `controls.json` (b) / (c).)
2. **Identity**: `--safety-head off` vs no flag: arm B bit-identical over 2 steps (the CPU form: `controls.json` (a), `(a')`).
3. **Eval reach**: under `--safety-head armA`, the dumped B pick equals `argmax sel_score(sigmoid(comb))` on a fixture, and differs from `argmax sel_score(sigmoid(sxB))` on at least one set of a moved state.
4. **Matched control**: `armA` / `stopgrad`: arm A's own selection (g5 as registered). `frozen`: arm A's progress rows + base safety rows (an extra RL-val view of arm A); report g5 against both.
