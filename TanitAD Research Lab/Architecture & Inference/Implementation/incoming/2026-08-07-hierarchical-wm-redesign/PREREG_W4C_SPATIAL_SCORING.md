# PRE-REGISTRATION — W4c: spatial cross-attention scoring port (REF-C conf-pass style)

**Registered 2026-08-10 ~19:25Z, BEFORE any run. CONTINGENT: runs only if W4b-kin's held-out
gate ALSO fails G1 (feat failed at 0.560; kin verdict pending). If kin passes, W4c is void and
this file is marked so.**

## Motivation (all MEASURED, banked)

- W4b-feat: held-out selected ADE 0.560 (gate ≤ 0.45) with a train-monitor at 0.21–0.33 —
  the pooled offset-query feature lets the rescorer MEMORISE train-window selection.
- Mode-structure: REF-C concentrates selection mass on ~4–5 clean candidates (entropy 0.97)
  via a conf pass whose queries cross-attend the SPATIAL 8×8 conv map (refc.py:1193);
  v5f's final sel_score (refined + factorised grafts off a flat rank≈16 state + vt gating)
  smears mass over ~12 (entropy 2.22).
- E-S1-0 (historical, REF-C line): scoring supervised at the ranked object 0.4728 vs 1.3100.

## Arm (ONE lever: the scoring head; fan and everything else frozen)

Queries = per-candidate embeddings of the EMITTED unicycle candidates (encode each (a,κ)
sequence + endpoint geometry with a small MLP); keys/values = the trunk's spatial feature map
(the same surface the base conf pass reads — cite exact tensor at implementation). One
cross-attention block + linear to a logit per candidate. Supervised at the ranked object:
margin rank loss against the unicycle fan's GT-nearest winner (identical to W4b's loss).
NO factorised grafts, NO vt gating on the output — the hypothesis under test is that the
SPATIAL grounding, not extra priors, restores generalising selection. ~2,000 steps, ≤2 h.

## Gates (bound now)

- **G1-c (port works):** held-out selected ADE ≤ 0.45 on the 881 grid (same as W4b's G1).
- **G-mode (mechanism check, secondary):** selection entropy on held-out windows ≤ 1.5
  (toward REF-C's 0.97 from v5f's 2.22) — passes only WITH G1-c; entropy alone proves nothing.
- **G-null:** selected ADE > 0.45 ⇒ per-candidate scoring on this trunk's features does not
  generalise regardless of input surface; selection moves ENTIRELY to W7 WM-roll re-rank
  (already primary per W4b's G2), and the fast selector is retired to a W7-distillation
  target (L4) — no third scoring attempt without new evidence.

## Measurement contract

881 grid; selected/oracle/top-k oracles/sel_gap via taniteval.selgap (cluster CI); train-vs-
held-out monitor gap reported explicitly (the memorisation diagnostic); entropy/mode stats;
four-families adjuncts on the selected trajectory; tier T0; artifacts banked.

## Status

- [ ] kin verdict read (voids or activates this prereg)
- [ ] launched · [ ] verdict appended + registry
