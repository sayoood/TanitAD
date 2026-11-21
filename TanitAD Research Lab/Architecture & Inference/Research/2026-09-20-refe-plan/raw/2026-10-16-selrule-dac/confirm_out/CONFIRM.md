# CONFIRM: V(0.8) vs B on P2's full navtest table (T1, MEASURED)

**Outcome: UNRESOLVED (family leg not computed)** -- lower bound > 0 and no adverse PDMS sub-score, but the longitudinal/lateral family leg was not computed; never ADOPT

- DeltaPDMS x100 +0.0235 [+0.0002, +0.0546] (paired log-cluster bootstrap, 136 logs, B 10,000, seed 20260927); picks changed 11/12146
- zero tokens B 509 -> chosen 505; DAC=0 380 -> 376; NC=0 133 -> 133
- EP +0.0103 [-0.0106, +0.0366]; TTC +0.0000 [+0.0000, +0.0000]
- adverse PDMS sub-scores: none; family leg: NOT COMPUTED
- oracle controls (DeltaPDMS x100, share of the +3.54 SEL ceiling): O-DAC +3.184 (90 %); O-NC +0.989 (28 %); O-both +4.043 (114 %)
- selection quality (random / actual / best-of-64), B: {'random_x100': 73.00314921968499, 'actual_x100': 88.8184088897293, 'best_of_64_x100': 98.23109369448531}; chosen: {'random_x100': 73.00314921968499, 'actual_x100': 88.84189293083969, 'best_of_64_x100': 98.23109369448531}
- STRATEGIC: PAUSED by the PI (2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored); reported as such, never as a pass

Estimator caveat: answers 'another draw of LOGS' only; P2 is one training run; the rule's parameter was fit on the u400 scorer (Amendment 1 A1).
