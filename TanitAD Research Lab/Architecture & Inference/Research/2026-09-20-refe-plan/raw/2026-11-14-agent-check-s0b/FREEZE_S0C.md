# FREEZE_S0C: the learned-fusion successor of S0B-READOUT, frozen by the MM before it runs (2026-11-14)

## Why S0-C
- **The committed row.** S0-B read S0B-READOUT (`READ_S0B.md`). The card's committed successor for that row is a learned fusion `q = f(R-true features, P6 logits)`, fit out-of-fold on fit-extra (`PREREG_AGENT_CHECK_DRAFT.md` section 3).
- **What this freeze adds:**
  - the same fusion for the deployable-shaped features, R-t0 and R-cv;
  - **F0**, which uses P6's logits only, with no perception.
- **Why F0.** S0-B shows that P6's own ranking, not perception, is where a third of the failures hide. F0 is the cheapest form of the card's "direct top-5 re-ranker" and costs nothing to read.
- **What has been seen:** the S0-B per-arm gains above. **No fusion number exists.**

## Pins
| file | sha256 |
|---|---|
| `s0c_fusion.py` | `8ea19d38e50e6551499a440ff38d0f498d51cffd08de5f96a4f6497bb6cf71e8` |
| `test_s0c.py` (synthetic, `ZZS0C_TEST_PASS 11`) | `a4e7af5c77c37e5c3af466dae9b80c041b4b18ef7971c3b1a163f5ac42491ec6` |
| `s0b_readout.py` (bootstrap, oracle; unchanged from `FREEZE_S0B.md`) | `f08e24846eab63556310605801b514b01e384b411f0d91021f34cd728b80ce80` |
| S0-B flags `run/s0b_flags.npz` | recorded in the result |
| fit pack | `e58653c5…` (asserted) |

## Design (everything is fixed in the script; nothing is tuned)
- **Folds:**
  - fit on the covered sets **outside** the SFT held-out logs (2,615 sets);
  - read on the **held-out** covered sets (1,343 sets, 13 logs): the S0-B deciding column.
- **Label:** TRUE agent-ok per proposal, `(nc > 0) & (ttc > 0)`.
- **Model:** logistic regression (sklearn, C = 1.0), with standardised features. The scaler is fit on the fit fold only.
- **Arms:**
  - **F0:** P6's 6 logits;
  - **F-R-true / F-R-t0 / F-R-cv:** F0 plus that S0-B arm's NC-proxy and TTC-proxy flags;
  - **F-C-SHUF (control):** F0 plus the shuffled-raster flags.
- **Pick:** `argmax(agg_torch(logits, lane=False) × p_ok)`, first index on ties.
- **Gain and capture:** gain vs P6's pick, and capture of the S0-B held-out oracle O_NC+TTC (+1.811), by the same code path.
- **Known caveat:** whether P6 trained on the fit-extra logs is UNVERIFIED (card Q-1). If it did, its logits are in-sample on the fit fold, and the fused weights can be miscalibrated for held-out. That most likely biases F0 and every F-X arm DOWN on held-out (HYPOTHESIS; not measured).

## Outcome rows (committed now; the point estimate decides; held-out)
| row | condition | next |
|---|---|---|
| **INVALID** | C-REG (p_ok ≡ 1) max \|gain\| ≠ 0.0, or capture(F-C-SHUF) − capture(F0) ≥ 0.10 | fix and re-run |
| **S0C-FUSE-GO** | capture(F-R-t0) ≥ 0.25 AND gain(F-R-t0) > gain(F0) | single-frame agent features add to P6's own scores. **S1** (a head trained to predict the R-t0 features, read through this fusion), which needs PI pod time |
| **S0C-RERANK** | capture(F0) ≥ 0.25, and not FUSE-GO | **the lever is P6's own NC/TTC heads, re-ranked, with no perception head.** Next: carry the recalibration to a navtest read under a keyed ledger entry (MM). In parallel, the card's direct top-5 re-ranker: hard-negative fine-tune of the NC/TTC heads |
| **S0C-NULL** | neither | the fusion route is closed at the raster level. Next: **R-sim-t0 / R-sim-cv** (agent list through the scorer's own at-fault logic), then the hard-negative re-ranker fine-tune |

Every row also reports F-R-true (the privileged fusion ceiling) and F-R-cv.

Run: `python test_s0c.py`, then `python s0c_fusion.py --flags run/s0b_flags.npz --out run_s0c`.
