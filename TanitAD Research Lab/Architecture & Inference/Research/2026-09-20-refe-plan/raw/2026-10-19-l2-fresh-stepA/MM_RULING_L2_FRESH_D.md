# MM ruling on READ 2 / universe D (Master Mind, 2026-10-19 ~16:45; READ 1 has not been read; no D label or score exists)

**My ruling's premise did not survive the measurement, and READ 2 on D is WITHDRAWN as O3's next arm.**
- `MM_RULING_L2_FRESH.md` replaced O3's next arm with READ 2 on universe D, claiming 87 logs would narrow the log interval.
- MEASURED by the sealing agent (`PREREG_L2_FRESH_D_DRAFT.md`):
  - D is 99 tokens over 84 logs;
  - G0 is PASS at token level but FAIL at log level: 83 of 84 logs are inside M-PDM and M-SCENE through other tokens, so D is log-SEEN by P2 and log-fresh only for the L2 head;
  - the estimated A1 half-width is about 0.054, WIDER than B's 0.045, because the token count falls from 400 to 99.
- So READ 2 can neither resolve an O3 better than READ 1 nor give a clean fresh-log test. Its one unique property (log-fresh for the L2 head) is already covered by the banked 5-fold-by-log OOF and by READ 1's LOG-OOF companion.
- **The PI fetch request (412 / 396 jpgs) is WITHDRAWN.**
- D1-D4 of the D draft are moot. The draft is banked as a measured scoping record, not as an arm.
- Error class, for the log: I sized a power claim from the cluster count alone and ignored the token count. The half-width depends on both, and the cheap measurement corrected it before any build. No claim was made from it.

**O3's next arm, re-stated (Amendment 2 to `PREREG_L2_FRESH.md`, sealed before READ 1 is read; it changes the next-arm mapping only, never a gate):**
1. **First, a DESIGN calculation:**
   - CPU only, on data ALREADY read: Step A's banked labels and L2's banked OOF scores. No fresh label is touched.
   - It decomposes the log-cluster bootstrap variance of A1 into its between-log and within-log parts, and projects the A1 half-width for READ 1 extended to ALL 1,305 B tokens (24 logs).
2. **If the projected half-width ≤ 0.025 ⇒ O3's next arm is B-ALL:**
   - READ 1's rule, verbatim, on all 1,305 B tokens;
   - the extra 905 tokens' bundle costs one locked GPU forward (about 25 min) plus CPU labelling;
   - at 0.025, an A1 point of 0.827 (the banked OOF) would clear 0.80 with its whole interval.
3. **Otherwise ⇒ the Step A gate is structurally unresolvable at the data on this box,** and O3's next arm is the unseen-stratum DATA JOB (A1.5(ii): the 1,167 non-navtrain trainval logs, unseen by P2 and the trunk):
   - first a scoping probe of where their frames exist (pod, HF, nowhere), at TWO locations at least;
   - its priority is set by the Step B diagnostic: a NOT CLEARED Step B makes L2-BRANCH the lever and the data job moot for this head.

Either way, nothing is loosened. The gate stays L1 §4 + Amendment 1 `interval_aware`, verbatim.
