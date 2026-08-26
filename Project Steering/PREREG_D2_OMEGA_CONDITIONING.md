# PRE-REGISTRATION — D2: the PI's conditioning channel, run for the first time

**Written** 2026-08-27 00:15 (Europe/Berlin), **BEFORE LAUNCH** · **Author** Master
Mind · **Hypothesis** E-DEC-65 (registered this turn) · **Tier of all reads**
T0-DIAGNOSTIC unless stamped otherwise.

```yaml
hypothesis: E-DEC-65
one_variable: --cond-param omega_accel_v      # [yaw_rate, a_long, v] as measured
                                              # state (Sayed, 2026-08-25/26)
matched_incumbent: postrain30k                # drift 0.669 · nrmse 0.8115 ·
                                              # T1 S-rate 0.2632 (exploratory)
held_constant: [recipe, init (distill_init.pt), corpus (parity train cache),
               steps 30000, batch 8, seed (default), window 6, horizons 1 2 4,
               o5-form l1, w-o5 1.0, w-o6 0.1, all other weights 0]
controls: [the E-DEC-59 panel's constant column (exactly 0.0000) and matched
           null; meanpred's nrmse_meanonly and nrmse_zero=1.0]
```

## Why this arm exists

E-DEC-57: the incumbent channel `[atan(L·κ), a, v]` is a **kinematic restatement of
realised motion** (closed form reproduces measured yaw-rate at r 0.9988) — so
action-conditioning **in the literature's sense has never been tested** and its
closure was withdrawn (PI, 2026-08-26). The PI's directive: **use the AV dataset's
ego data — `[ω, a_long, v] as measured state, not an action.** The conversion is
exact and wheelbase-free; the default path is **bit-identical to the incumbent**
(pinned by `stack/tests/test_cond_parameterisation.py`, 7 tests).

## The exact one-variable diff (C164 rule — commands, not intent)

The launch line is `chain_seed1.sh`'s **verbatim**, with exactly three edits:
`--out …/omega30k` (new dir) · **no `--seed`** (matching `postrain30k`, which ran
the default; `_seed1` is the arm that carries `--seed 1`) · **`+ --cond-param
omega_accel_v`** (the variable).

⚠️ **Trainer provenance, stated:** Thor's `train_v6_staged.py` predated the flag and
was replaced by the current one (md5 **bc16941815ff1cc693cade4c0935b1bf**, verified
both sides; Thor's old copy backed up at
`/home/nvidia/staging/train_v6_staged_BACKUP_pretrio.py`). The measured diff vs the
trio's trainer is confined to the **o13 annotation block** (weights 0 in this launch,
o13 verified a no-op at weight 0) and the **cond-param block** (default bit-identical
by test). No other subsystem differs; the context-free diff token summary is banked
in the drumbeat transcript.

## Reads and outcomes, committed in advance

Primary reads at 30k, all against `postrain30k` on the same instruments:

1. **Drift** (`latentmotion.py`, 80 clips, k=4, band 0:8).
2. **Held-out prediction** (`meanpred.py` nrmse; incumbent 0.8115, same-rig).
3. **The ego-marginal panel** (`latentmotion.py` columns): the incumbent's
   ego-state marginal over drift was **+0.0073 (t 2.78) — inside the |t|≈2.9 null**.

| outcome | criterion | conclusion |
|---|---|---|
| ⭐ **CHANNEL MATTERS** | ego marginal clears the matched null (t ≥ 2.9; SURVIVES ≥ 4.2) where the incumbent's did not, with drift and nrmse within 10 % of the incumbent | conditioning the trainer on measured state changes what the latent transition carries — the v7 conditioning question REOPENS with a working channel |
| ⛔ **CHANNEL IRRELEVANT** | ego marginal inside the null AND drift/nrmse within 10 % | the parameterisation was not the binding constraint; E-DEC-48b's scene→action reading extends to the measured-state form, and the v7 design's "planner owns the reaction" stands on stronger ground |
| ⚠️ **SIDE-EFFECT** | drift or nrmse degrade > 10 % | the channel change perturbed training itself; no conditioning conclusion is admissible — investigate before any further arm |
| 🔶 **MIXED** | any other cell | state the numbers; do not round to a verdict (C160/C166) |

⛔ **What this arm can NEVER show regardless of outcome:** driving capability (all
three reads are T0). A T1 pass on the parity corpus follows only if the arm earns it.

**Cost:** ~4.1 h Thor (30k steps at ~26.5 s/step... measured trio pace ~0.49 s/step
wall — 30k in ~4.1 h). Launched immediately after this file was written.
