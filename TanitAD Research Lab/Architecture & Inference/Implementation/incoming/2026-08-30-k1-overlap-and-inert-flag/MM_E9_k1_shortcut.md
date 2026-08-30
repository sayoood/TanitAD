# MM-E9 — the k=1 overlap shortcut is REAL at 2k and ABSENT at 30k

**Measured** 2026-08-30 · Master Mind · **zero GPU, zero new runs** — read from
`o5_step1` / `o5_stepK`, already logged in every banked `train_log.jsonl` ·
**Tier T0-DIAGNOSTIC**.

## The claim under test

The data-efficiency review found, in our own source, that **our k=1
latent-prediction target is 67 % already-observed**: with `n_stack=3`, stride-1
windows and `window=6`, the last context latent encodes frames {t+5,t+6,t+7} while
the k=1 target encodes {t+6,t+7,t+8} — **two of three shared, inside one latent, so
unmaskable**. It cited DINO-WM measuring this class of shortcut collapsing planning
success **0.76 → 0.08**.

## The discriminator, and why it cost nothing

If k=1 is a shortcut, its loss should be **anomalously low** against a horizon with
no overlap. Overlap by horizon at `n_stack=3`, stride 1: **k=1 → 67 %, k=2 → 33 %,
k≥3 → 0 %**. The trainer already logs `o5_step1` and `o5_stepK`, so the ratio is a
**read of banked runs**, not an experiment. *(The arm I had planned would have cost
~17 min and been confounded; the log had the answer.)*

| arm | steps | `o5_step1` | `o5_stepK` | **ratio** |
|---|---|---|---|---|
| `enc3f_2k` | 2,000 | 0.0654 | 0.2125 | **0.308** |
| `o14base2k` | 2,000 | 0.0804 | 0.2225 | **0.361** |
| `postrain30k` | 30,000 | 0.2877 | 0.2794 | **1.030** |
| `o14fut30k` | 30,000 | 0.2887 | 0.2892 | **0.998** |
| `emao14_30k` | 30,000 | 0.3005 | 0.2705 | **1.111** |

⭐ **Both 2k arms sit at 0.31–0.36. All three 30k arms sit at 1.00–1.11.** The
separation is by **SCALE, not recipe** — `o14base2k` (two-term) and `enc3f_2k` agree
at 2k, and the 30k trio spans three different recipes (plain two-term, +O14, +EMA)
and agrees at ~1.0.

## Verdict

**The shortcut is a 2k-scale phenomenon and is CLOSED by 30k.** At 2k the k=1 target
costs **a third** of the non-overlapped one; at 30k it costs the same or more.

**Mechanism (HYPOTHESIS, not measured):** these are JEPA-style losses whose *target
is itself learned*. At 2k the representation is still near-collapsed, so a
67 %-overlapped target is trivially predictable. By 30k the representation carries
enough content that even the overlapped target is as hard as the clean one — note
`o5_step1` rises **4.4×** (0.065 → 0.289) while `o5_stepK` rises only **1.36×**
(0.213 → 0.289). The overlapped horizon is where the loss grows.

⚠️ **What this does NOT establish.** Ratio ≈ 1.0 proves the k=1 loss is not
anomalously low; it does **not** prove no copying occurs. A model could copy the
shared 67 % and still find the remaining third as hard as a clean horizon, leaving
the ratio flat. ⇒ **The shortcut is not shown to be absent — it is shown not to be
CHEAP.** A direct test would mask the overlapped content, which is impossible here
(it lives inside one latent), or compare against an `n_stack=1` arm at 30k.

## ⛔ CONSEQUENCE FOR MM-E8, AND IT IS SERIOUS

The encoder ladder runs at **2k — precisely the scale where the shortcut is
ACTIVE (ratio 0.31)**. Its single-frame arm removes `n_stack=3`, which removes
**both** the encoder's local motion **and** the k=1 overlap (n_stack=1 ⇒ 0 % shared).

⇒ **At 2k those two changes cannot be separated, and one of them is measured to
vanish by 30k.** A 2k result would therefore answer a question that does not exist at
production scale. This is the `--v2` conflation failure — ten levers on two axes,
non-attributable — and I built it into a running experiment.

**Committed consequence:** the MM-E8 pair is **NOT readable as an encoder result**.
Either (a) add an `n_stack=1` arm at 30k where the shortcut is absent, so the only
live variable is encoder motion, or (b) read MM-E8 only for the shortcut question and
draw no encoder conclusion. **(a) is the honest one and costs ~8 h on an idle Thor.**

## Bearing on the review's alarm

The review's mechanism is **correct and correctly located in our source**. Its
severity transfer was scoped too widely: DINO-WM's 0.76 → 0.08 collapse is not
evidence for our production scale, where the ratio is measured at ~1.0. ⇒ **Real
mechanism, real at 2k, not demonstrated at 30k** — and worth re-checking on the first
B1-scale arm, since the corpus changes and the ratio is free to compute.
