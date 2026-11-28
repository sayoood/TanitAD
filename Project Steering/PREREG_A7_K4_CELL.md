# PRE-REGISTRATION — A7: the fourth corner of the drift 2×2 (`k4_30k`)

**Written** 2026-08-27 ~06:50 (Europe/Berlin; Thor marker 04:51 UTC), **before launch** · **Author** Master
Mind · **Hypothesis** E-DEC-66 · **Tier** T0-DIAGNOSTIC.

```yaml
hypothesis: E-DEC-66
one_variable: --o5-k 4        # vs the incumbent's 8. NOTHING else changes.
matched_incumbent: postrain30k (trainable+k8, drift 0.669, nrmse 0.8115)
held_constant: [recipe, init distill_init.pt, corpus (parity train cache),
               steps 30000, batch 8, seed default, window 6, encoder TRAINABLE]
launch: chain-launched after omega30k finishes; verbatim postrain30k line
        with --o5-k 4 and --out k4_30k. Trainer md5 bc16941815ff… (same binary
        as D2; diff vs the trio's trainer = inert weight-0 args + cond block).
```

## Why this cell decides something

The recursive config diff (2026-08-27 02:00) proved `splitp30k` differs from
`postrain30k_freeze` by **exactly one substantive knob: `o5_k` 4 vs 8** (the rest:
weight-0.0 args and the derived `o4_span = window + o5_k`). The drift 2×2 is
three-quarters measured:

| | k8 | k4 |
|---|---|---|
| **trainable** | 0.669 (`postrain30k`) | **THIS ARM** |
| **frozen** | 0.3905 (`postrain30k_freeze`) | 0.199 (`splitp30k`) |

## Outcomes, committed in advance (drift via `latentmotion.py`, same rig as all four)

| outcome | criterion | conclusion |
|---|---|---|
| **ADDITIVE** | drift ≈ **0.48** (0.42–0.54) | k and freeze act independently; the splitp30k separation is fully explained as freeze + k main effects |
| **INTERACTION** | drift ≈ **0.67** (≥ 0.60) | k4 only helps when the encoder cannot move — freezing is a precondition, and the trainable line cannot buy drift with k |
| ⭐ **K-DOMINANT** | drift ≤ **0.42** | k, not freezing, was the main lever all along — reopens the encoder-policy reading of D1 |
| 🔶 between bands | anything else | state the numbers, no verdict (C160) |

Secondary reads: `meanpred` nrmse vs 0.8115 (does k4 cost prediction on a
trainable encoder?), participation, and the E-DEC-63 pixel-marginal probe
(unchanged rig) so the cell also lands in the representational-arm baseline table.

⚠️ **What this cell cannot say:** anything about driving (T0 only), and nothing
about `o5_k`'s effect at scale beyond 19 M params.


---

## OUTCOME — read 2026-08-27, instruments as pre-registered

**`k4_30k` drift = 0.6701 (t 148.81)** — inside the INTERACTION band (≈0.67,
≥0.60), nowhere near ADDITIVE (0.42–0.54) or K-DOMINANT (≤0.42).

| | k8 | k4 |
|---|---|---|
| **trainable** | 0.669 / nrmse 0.8115 | **0.6701 / nrmse 0.7682** |
| **frozen** | 0.3905 / nrmse 0.9301 | **0.199** (`splitp30k`) |

⇒ ⭐ **INTERACTION: `o5_k` does NOTHING on a trainable encoder** (0.669 → 0.6701,
within the 1.5 % seed band) **and helps only under freeze** (0.3905 → 0.199).
**The C164 leftover fully resolves**: `splitp30k`'s 0.199 = the freeze main effect
plus the freeze×k4 interaction — no unexplained residue remains, and the
trainable-encoder drift attractor (0.62–0.68) is untouched by k.

**Consequence for v7:** with the encoder TRAINABLE (D1) and k inert on drift,
⛔ **no known training-knob lowers drift on the trainable line — the
representational route (O14, E-DEC-67) is the only live anti-drift lever.**
Secondary: k4's nrmse 0.7682 (−5.3 % vs incumbent, ~1.5× the 3.5 % seed band —
noted, not separated-grade).

Raw: `k4drift.json` / `k4nrmse.json` (8fc25020 scratchpad; banking next package).
T0-DIAGNOSTIC. ⚠️ The read required a same-day loader fix (recorded-args without
`tac_vocab_version` must mean v6.0) — found because this ckpt was the FIRST loaded
after the v7-vocab mandate landed; the property is now test-pinned.
