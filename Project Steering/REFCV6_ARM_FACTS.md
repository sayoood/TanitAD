# refcv6 — the lever facts, read from source, with their controls

**Master Mind, 2026-09-10.** MEASURED from `stack/scripts/refc_v3_train.py` at HEAD and from
refcv5-v2's own `config.json` argv (`C:/Users/Admin/refcv5v2_final/config.json`, 35 top-level
keys, 65 argv tokens). Controls on every probe: **110** `add_argument` calls read from the same
file in the same breath (positive), **0** for a nonsense flag (negative). ⛔ A zero from this
mount is otherwise a claim about the mount, not the repo.

---

## 1. refcv5-v2's argv — the baseline, from the run's own record

```
--steps 40284  --batch 20  --lr 1e-4
--agents off
--w-u0 0.5
--anchor-control-units alat
--goal-str  --sel-refined  --sel-score-emitted
```

⛔ **ABSENT**: `--wp-index`, `--w-tac-goal`, `--max-speed-input`.

## 2. The four levers, with their declared defaults

| flag | file:line | default | refcv5-v2 passed | refcv6 arm |
|---|---|---|---|---|
| `--w-u0` | `refc_v3_train.py:5095` | `U0_WEIGHT_DEFAULT` = **0.0** (`:122`) | **0.5** | **D** = stop passing it |
| `--agents` | `:5151` | `"off"`, choices `off / head / oracle` | `off` | **A** = `head` |
| `--wp-index` | `:5340` | `"off"`, choices `off / on` | absent | **B** = `on` |
| `--w-tac-goal` | `:4922` | **0.0** | absent | **C** = a weight the PI must set |

### 2.1 ⭐ Arm D is a RETURN TO DEFAULT, not an addition

`U0_WEIGHT_DEFAULT = 0.0`, and the source comment at `:122` names it *"WP-4: the x0 loss, in
CONTROL space"*. refcv5-v2 **explicitly opted in** at 0.5.

⇒ Arm D does not add a flag; it **stops passing one**. That is the strongest single-lever form
available — the parsed-namespace diff is exactly one key by construction.

⇒ And it sharpens `D-DDV1-NO-DENOISING-LOSS`: DD-v1 has **no ε-prediction and no denoising MSE
at all** (its `diff_loss_weight = 20.0` is dead code). Our trainer's own default already agrees
with the paper. **refcv5-v2 is the arm that departed from both**, and it is the arm that lost
longitudinally. ⛔ Correlation, not cause — which is precisely why D runs first and costs nothing.

### 2.2 ⭐ Why `tac_goal_tok_head` received exactly zero gradient — the mechanical answer

`--w-tac-goal` is declared at `refc_v3_train.py:4922` with **`default=0.0`**, and refcv5-v2's
argv **does not contain it**.

⇒ The head's **11,286 parameters** took `grad_abs_sum` **exactly 0.00000 for all 40,284 steps**
because its loss weight defaulted to zero — **not** because of a wiring bug, and **not** because
the head was unbuilt. It was built, rollable, and unreached.

⭐ **"Rollable and trained are different claims."** ⛔ No refcv5-v2 result may be credited to the
22-token tactical vocabulary.

### 2.3 ⛔ Arm B is not one flag unless six sub-knobs are pinned

`--wp-index` ships with `--wp-index-mode` (`:5356`), `--wp-index-detach` (`:5367`),
`--wp-index-radius-m` (`:5373`), `--wp-index-hidden` (`:5380`), `--wp-index-scale-m` (`:5385`)
and `--wp-index-const-xy` (`:5388`). Every one must be **identical between arms A and B**, or B
moves more than one lever and the panel is non-attributable — the `--v2` conflation failure (ten
levers on two axes) is the precedent.

---

## 3. ⛔ `--max-speed-input` CANNOT run yet — and this blocks the PI's own named request

The PI asked, verbatim, that *"all necessary vocab are used weiter as inputs like nav commands
and max speed"*. The flag exists (`refc_v3_train.py:5502`, `action="store_true"`). Its own help
text states the blocker:

> reads the v8 label record's `speed_max_input` block (`v_max_ms`, units DECLARED on the wire as
> m/s); **the v7.2 release carries it on 0 of 4,572 records**, and **the flag REFUSES there
> rather than looking switched on**.

⭐ That refusal is **correct behaviour and the reason nothing silently broke** — it is the
opposite of the `tac_goal` failure above, where a zero default let a head look wired while
taking no gradient. Here the flag would rather stop than lie.

⇒ **`--max-speed-input` is gated on the v8 labels existing.** 0/4,572 is a hard zero, and it is
the PI's authorised item 3.

⚠️ **Where such a ceiling could come from is an open question, not an assumption.** PhysicalAI-AV
publishes **no map, no lane graph and no traffic-light feature** — the card says *"we do not
include open maps data"*, pinned by `stack/tests/test_physicalai_feature_readset.py`. So a
"posted limit" has no published supplier in the corpus. ⛔ Do not assume the v7 speed-bucket
ladder is a legal stand-in: a ceiling derived from the ego's own realised speed is the
**nav-echo defect** in a new costume, the same family as flagship v1's route head scoring
**1.0000** as an exact bijection of the nav it was fed.

---

## 4. ⛔ The consequence for refcv6's baseline — state it, do not absorb it

If `--max-speed-input` is ever folded into refcv6's inherited baseline, then **refcv6 versus
refcv5-v2 stops being a single-lever comparison.**

⇒ The clean structure is a **refcv6 baseline arm equal to refcv5-v2's argv exactly**, with every
lever arm differing from that baseline by **exactly one parsed-namespace key**. Max-speed is a
**separate, later, pre-registered lever**, gated on the v8 labels.

⛔ **Every arm carries a replicate from the start.** MEASURED on WP-D: an arm with **zero levers
moved** read "separably worse" on **5 of 9** family metrics and reproduced a headline ADE effect
at **+0.02460** against the lever's **+0.02610**. A one-seed panel cannot adjudicate this rig.
