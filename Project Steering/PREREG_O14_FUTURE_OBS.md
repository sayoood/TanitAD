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
