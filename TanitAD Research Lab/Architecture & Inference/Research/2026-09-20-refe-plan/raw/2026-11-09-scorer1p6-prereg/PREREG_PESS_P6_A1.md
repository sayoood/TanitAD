# PREREG_PESS_P6_A1 — PE3's named next arm: the 1-SD rule on NC and DAC only

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-10 at about 21:43Z, AFTER the PESS read
(`pess/read/real/pess_read.md`, outcome PE3) and BEFORE any number of the arm below exists. It does not change
`PREREG_PESS_P6.md` (sha256 `185de406...`); it carries out that card's PE3 clause: *"next: pessimism on NC and DAC only
(LCB1's result, read here, decides between min and 1 SD)"*.

## 0. What is known (MEASURED, E3, paired log-cluster bootstrap 10,000, eight members C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM)

- ENS9 91.414, +0.2174 [+0.0337, +0.4084] vs P6. PESS +0.2825 [+0.0756, +0.5042]; LCB1 +0.2968 [+0.1140, +0.4918].
- PESS - ENS9 +0.0651 [-0.0699, +0.2177]; zero-pick share -0.0021 [-0.0036, -0.0008]. EP x100 vs P6: PESS -0.5194
  [-0.7055, -0.3157], LCB1 -0.3773 [-0.5572, -0.1797], ENS9 -0.0821 [-0.2623, +0.1080].
- The card's choice between min and 1 SD: LCB1 has the higher point (+0.2968 > +0.2825), so the 1-SD rule is used.

## 1. Arm

**LCB1ND**: NC and DAC heads = mean minus 1 x SD (ddof 0, float64, cast to float32) of the member logits; TTC, EP, C and
DDC = mean. Same eight members as the PESS read (S1ZD's pass is still running, and keeping the members fixed means only
the rule changes). Pick with `read_scorer1p6.pick_logits`; score by the full truth table, as `read_pess.py` does.
ENS9 and LCB1 are recomputed from the same members as references; their logits are bit-identical to the PESS read's, so
their ledger keys already exist and add nothing to K.

## 2. Bars (E3, PDMS x100, paired log-cluster bootstrap 10,000, seed 20260927)

| id | metric | bar |
|---|---|---|
| P1 | LCB1ND - P6 | CI-lo > 0 and d > 0.4971 |
| P2 | LCB1ND - ENS9 | CI-lo > 0 |
| M | zero-pick share LCB1ND - ENS9 | CI-hi < 0 |
| E | EP x100 LCB1ND - LCB1 (mechanism: was TTC's pessimism the EP cost?) | CI-lo > 0 |
| G1-G4 | as `PREREG_PESS_P6.md` | as written there |

## 3. Outcomes (label order A1-1, A1-2, A1-3)

- **A1-1 ADOPT** (P1, P2, M, G1-G4): LCB1ND becomes P6's selector; then add S1ZD as a ninth member under its own read.
- **A1-2 EP RECOVERED, BELOW BAR** (E holds, P1 fails): TTC's pessimism was the EP cost, but a selector over these members
  tops out near ENS9 plus a few tenths. The selector family over P6's fine-tunes is then exhausted. Next: the PE4 path
  (a perception check of each proposal on richer features: an architecture change and the PI's decision), and member
  diversity (fine-tunes on disjoint navtrain halves: pod time, a PI check).
- **A1-3 TTC NOT THE EP COST** (E fails, P1 fails): the EP cost sits in the NC and DAC pessimism itself. Next: same as A1-2.
- **Unnamed** if none matches.

K: + 1 (`LCB1ND:<sha256 of its combined logits>:E3`).
