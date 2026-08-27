# PRE-REGISTRATION — O14-fut: the future-observation auxiliary (R2, PI-approved 2026-08-27)

**Hypothesis** E-DEC-67 · **Tier of all reads** T0-DIAGNOSTIC · **Status** approved
by the PI (*"I approve R2"*); implementation next, launch after gates code review.

```yaml
hypothesis: E-DEC-67
design: R2 from DESIGN_NOTE_REPRESENTATIONAL_ARM.md — a small head from
        (z_t, conditioning) predicting 32x80 GREY PIXELS AT t+4; loss w_o14 * L1.
        Variant arm: the DIFFERENCE frame (pix_{t+4} - pix_t) as target.
rig: v7-tiny ladder, 2k-step matched pairs (E-DEC-9 pattern)
arms:
  - incumbent (w_o14 = 0)          # must be BIT-IDENTICAL — test-pinned
  - R2 w=0.1
  - R2 w=1.0
  - R1 reconstruction w=1.0        # the contrast: does the TEMPORAL form matter?
  - R2-DR: w=1.0, TIME-SHUFFLED targets   # deliberate regression — must NOT absorb
one_variable_per_pair: the auxiliary term and its weight; all else byte-identical
primary_read: the E-DEC-63 A3 pixel-marginal (banked rig, controls validated).
  SUCCESS = marginal falls from +0.0096 (t 5.11) to INSIDE the null (|t| < 2.9)
  on the R2 arms, with gates below holding.
gates_in_order:
  G-RANK:   val-side participation not below the incumbent's
  G-DECODE: n_agents / n_free_cols / occ_{l,c,r} vs pixel floor AND constant AND incumbent
  drift:    latentmotion drift not above the incumbent's band
  nrmse:    meanpred not >10% worse than incumbent (the D1 criterion, reused)
outcomes (committed):
  ABSORBED:  marginal -> null on R2, gates hold           => O14-fut earns a 30k arm
  FLOODED:   marginal falls, G-RANK or ego fails          => retry lower w once; else reject
  INERT:     marginal unchanged                           => check the gradient path
             (aux must reach the ENCODER, not only the head) BEFORE any conclusion
  DR-FIRES:  shuffled-target arm absorbs                  => instrument bug; read NOTHING
  R1-EQUAL:  R1 matches R2 on absorption + gates          => the temporal form is not
             the active ingredient — report as such, do not prefer R2 on aesthetics
leakage_check: the target is future OBSERVATION (data), never a model product and
  never an inference-time input; conditioning unchanged. Vision-only at inference
  holds. (2026-08-03 rules.)
cost: 5 tiny arms x ~20 min + probe re-runs; NO full-scale GPU before gates.
```

⚠️ **The loss curve is never the result** — future pixels are partly unpredictable
(other agents), the head will regress toward blur, and the raw L1 value is
uninformative by construction. Only the probes above are read.


---

## AMENDMENT B — recorded 2026-08-27 after arm 1's probe, BEFORE further reads

⚠️ **The primary read's premise fails at the ladder's scale.** The base 2k arm
(`o14base2k`, w=0) reads pixel-marginal **−0.0033 (t −0.59, INSIDE_NULL)** — there
is **no positive baseline marginal to absorb at 2k steps**. The E-DEC-63 baseline
(+0.0096, t 5.11) was measured at **30k**; the displacement evidently BUILDS with
training (consistent with E-DEC-61's within-recipe drift trajectory). The rig's
drift-validity band [0.60, 0.74] is likewise 30k-calibrated; 2k arms sit ~0.45
with clean controls — the band is scale-wrong, not the rig.

**Revised read, committed now:**
1. The tiny ladder's readable quantities are the GATES — drift, nrmse,
   participation vs the matched base — plus one exploratory cell: whether any
   O14 arm's pixel-marginal DIFFERS from base at 2k (either sign is
   informative; none is the absorption read).
2. ⭐ **The absorption PRIMARY moves to a 30k arm**: `o14fut30k` — the
   `postrain30k` recipe + `--w-o14 1.0 --o14-mode fut` (v6.0 vocab pinned for
   comparability with the incumbent), read against the E-DEC-63 baseline at
   matched scale: SUCCESS = pixel-marginal at 30k **below** the incumbent's
   +0.0096 toward the null, gates holding. The DR control at 30k is NOT
   retrained (8 h); the 2k DR arm plus the time-shuffle inside the probe carry
   the instrument-validity burden, stated as a limitation.
3. Launch order: tiny gates first (the prereg's own rule); the 30k arm launches
   tonight IF no tiny gate fails against base.


---

## TINY-LADDER GATES — read 2026-08-27 ~23:15, per AMENDMENT B

| arm | drift | nrmse | cos |
|---|---|---|---|
| `o14base2k` (w=0) | 0.4531 | 0.9876 | 0.183 |
| `o14fut01` | 0.4576 | 0.9886 | 0.184 |
| `o14fut10` | 0.4733 | **0.9746** | 0.246 |
| `o14rec10` | 0.4704 | 0.9769 | 0.233 |
| `o14dr10` (shuffled) | 0.4533 | 0.9895 | 0.179 |

**No gate fails vs base**: worst drift +4.5 % relative (~3× the 1.5 % seed band, no
explosion); the DR arm sits exactly at base on drift (correct inertness); fut10 /
rec10 marginally BETTER on nrmse. ⚠️ Scale fact, stated: at 2k EVERY arm including
base reads "IS the mean predictor" (nrmse ~0.98 vs mean-only ~0.998, cos ~0.2) —
the prediction gate is barely meaningful at 2k and the read is vs-base RELATIVE,
exactly as Amendment B prescribed. Exploratory cell: pixel-marginal null in base /
fut01 / fut10 alike — no dose effect at 2k.

⇒ **`o14fut30k` LAUNCHED** (2026-08-27 ~23:20, Thor PID 1629511, first log row
content-verified): the verbatim `postrain30k` line + `--w-o14 1.0 --o14-mode fut
--o14-k 4` + `--tac-vocab-version v6.0` (head-matched to the incumbent), NO other
change — the matched pair for the absorption PRIMARY at 30k. Raw gates:
`o14drift.json` / `o14nrmse.json` + absorb_*.json (8fc25020/b53d2f9f scratchpads;
banking with the ladder package). T0-DIAGNOSTIC throughout.
