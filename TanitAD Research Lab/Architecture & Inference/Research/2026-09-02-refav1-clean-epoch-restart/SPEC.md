# SPEC — the clean-epoch restart of refav1: EMA-anchored targets + bf16/TF32, validated against the fp32 incumbent's own first 500 steps

`E-ARCH-REFAV1-EPOCH-2` · 2026-09-02 · Master Mind · Thor · **pre-registered before any decision to restart**

```yaml
hypothesis: H-EPOCH-2   # "the final recipe (frozen op target + EMA tac/str targets + bf16 autocast + TF32) trains the
                        #  operative term at least as well as the fp32 incumbent, at <= 0.5x the wall-clock, with a
                        #  scale-stable tactical target"
gates_before_launch:
  G1: "E-ARCH-TSC-2 R2/R3 CONFIRMED (EMA pins the tactical scale on the slice) and R7 known"
  G2: "the precision flags exist, pass their OFF-path identity test, and the bf16 smoke is finite"
  G3: "the fp8 cache holds all 4,572 v7.2 train clips (16d325e9... rebuilt through the same builder, gate PASS)"
  G4: "PI go for stopping the incumbent (it is 4-10 h in; the restart discards those steps)"
held_constant: [seed 0, bs 8, lru 64, target_space frozen, detach_aux_targets on, bptt_truncate 15,
                min_participation 0, labels/nav = the v7.2 train blob md5 0ff90213, log_every 50, save_every 1000,
                steps 21109 (one epoch), the data order (seed-deterministic loader)]
one_variable_block: "EMA targets + bf16 + TF32 change TOGETHER vs the incumbent; attribution is by READ, not by arm:
                     the operative term never sees the EMA (its target is frozen), so loss_feat_op / grad_norm /
                     participation are the PRECISION reads, while tgt_std_tac/str and loss_feat_tac/str are the EMA reads"
controls: [the incumbent's rows 1-500 (fp32, no EMA) = the paired control, same seed and data order;
           known_value tgt_std_op ~ 1.0; the drift alarm's guards]
```

## 1. Why restart rather than continue

MEASURED on the incumbent (`/home/nvidia/experiments/refav1-b1-v72-1ep-21109`, 2026-09-02 20:32Z, steps 550–750):
the collapse is gone (`adapter_std` 0.478 → 0.68, `tgt_std_op` pinned) **but the unanchored tactical target
oscillates**: `tgt_std_tac` 3.59 → 3.65 → 4.35 → 2.56 → 4.59 between consecutive rows, the tactical loss
0.32 → 0.34 → 0.27 → 0.06 → 0.69, its share of the loss 17 → 20 → 19 → 5 → **36 %**, `grad_norm` 3.6 → 5.5 →
6.3 → 1.3 → 7.6, and `adapter_std` has stopped rising since step 500 (0.681 → 0.680). A run whose gradient is
owned in alternate rows by a term with a free scale is not the clean epoch the PI asked for. The EMA teacher
(D-REFAV1-EMA-SPEED) anchors that scale; E-ARCH-TSC-2 is measuring it on the slice tonight.

## 2. The restart recipe (if G1–G4 hold)

`--target-space frozen --detach-aux-targets --bptt-truncate 15 --ema-targets --precision bf16 --tf32
 --min-participation 0 --steps 21109 --bs 8 --lru 64 --seed 0 --log-every 50 --save-every 1000`
on the completed 4,572-clip cache, under `sup_refav1_v3.sh` (generated from v2: the new flags on the embedded
launch line, a fresh lock path, `200>&-` on every child).

## 3. Reads, COMMITTED IN ADVANCE — the restart's rows 1–500 vs the incumbent's rows 1–500 (same seed, same order)

| # | read | CONFIRMED | REFUTED |
|---|---|---|---|
| R1 | wall-clock: marginal s/step over rows 100–500 | ≤ 12 s (≥ 1.7× faster; ESTIMATED target ~8 s) | > 16 s ⇒ bf16 buys little on Thor; keep it only if R2–R4 hold |
| R2 | `loss_feat_op` at rows 100–500 (precision read; EMA-independent) | within 5 % of the incumbent's at the same steps, same sign of trend | > 15 % worse ⇒ bf16 hurts the operative term; fall back to `--precision fp32 --tf32` |
| R3 | `grad_norm` rows 100–500 | same band as the incumbent (0.5–8), no `inf`/`nan` | any non-finite, or median > 3× the incumbent's ⇒ precision fault |
| R4 | `participation` rows 100–500 | ≥ 8.56 throughout, median within 20 % of the incumbent's | below the floor ⇒ VOID, investigate |
| R5 | `tgt_std_tac` rows 100–500 (EMA read) | max/min ratio over the window < 2 (the incumbent: 0.057 → 5.95 → 2.56, ratio > 100) | ratio > 5 ⇒ EMA does not pin on the full corpus either |
| R6 | tactical share of the loss, rows 300–500 | stable, < 25 %, no row-to-row swing > 10 points | swings as the incumbent's ⇒ EMA insufficient |
| R7 | `tgt_std_op` | ≈ 1.0 pinned | drift ⇒ ⛔ instrument fault, **VOID** |
| R8 | the drift alarm | silent through 500 | any guard trips ⇒ stop and read |

A REFUTED R2/R3 falls back to fp32+TF32 (one more restart, ~1 h lost). A REFUTED R5/R6 with R2–R4 fine
means the EMA is not the anchor on the full corpus — the run continues (it is still collapse-proof) and a
fit-once frozen projection for the tactical target becomes the next pre-registered arm.

## 4. What this does NOT settle

Nothing here is a capability claim (tier T0, train-side instruments). The epoch's driving-relevant reads come
from the T1 adapter on checkpoints (step 1000 first, `taniteval/tools/refav1_arm.py`). The attribution between
bf16 and EMA rests on the argument in the YAML block, not on separate arms; if any read is ambiguous, the
cheapest disambiguation is a 250-step slice arm on the dev box, not a Thor restart.

## 5. AMENDMENT (2026-09-02 ~23:05 Berlin) — G1 is redefined: the slice rig is VOID, so the restart IS the EMA test

MEASURED (`…/2026-09-02-refav1-ema-inflation/raw/B_prime.log`): the deliberate-regression arm B′ on the 20-clip
slice did **not** reproduce the inflation — `tgt_std_tac` 0.0201 @1 → 0.0197 @250 (R1 ratio ≈ 1, the SPEC's
own VOID branch); the arms ran at the largest batch that fits the 4060 (bs 2; bs 4/8 OOM in the fit probe), on
20 EVAL clips, with a step-1 tactical scale 0.020 vs 0.057 on Thor — the rig differs in batch statistics and
data, and the inflation is evidently a full-corpus / bs-8 phenomenon (HYPOTHESIS). Arm A′ (collapse) is still
running as the sensitivity control. **R7 MEASURED** (`raw/R7_resume.log`): `--resume --ema-targets` from an
EMA-less checkpoint fails strict load on 20 missing `ema.*` keys ⇒ a MID-RUN switch is blocked as implemented
(a load-path change — initialise missing EMA copies from the student — is a separate small item); a FRESH
restart is unaffected.

⇒ **G1 becomes:** *the restart's own R5/R6 (tactical scale ratio < 2 over rows 100–500; share stable < 25 %)
against the incumbent's banked rows 1–500 (same seed, same data order) are the EMA test, on the real rig, paired.*
The fallback in §3 stands: if R5/R6 are REFUTED with R2–R4 fine, the run continues (collapse-proof) and a
fit-once frozen projection for the tactical target is the next pre-registered arm. G2–G4 unchanged.
