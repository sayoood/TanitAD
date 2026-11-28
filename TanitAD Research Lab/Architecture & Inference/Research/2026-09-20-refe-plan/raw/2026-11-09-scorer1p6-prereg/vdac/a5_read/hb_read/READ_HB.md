# VDAC A5, HB pair read (frozen reader): HB-s0 is X-BOUND. Drivable IoU clears its bar, the DAC ranking does not.

Master Mind, 2026-11-13.

## Provenance
- **Cards:**
  - `PREREG_VDAC_P6_A5.md`, with amendment `PREREG_VDAC_P6_A5_A1.md` (sha256 `2b40a672…`, landed `2efb14dd`, GAC `D-VDAC-A5-A1-1`);
  - reader freeze `a5_read/READER_FREEZE_A5.md`.
- **Reader:**
  - `eval/s1_read_v2.py`, sha256 `796aea21…`;
  - frozen v1 `s1a_read/s1_read.py`, sha256 `58aa3a77…`;
  - both equal the freeze table, re-hashed today.
- **Template manifest:** `manifest_a5_hb.json`, sha256 `bd1c6f7e…`. It equals the freeze.
- **Read manifest:** `inputs/manifest_a5_hb.filled.json`, sha256 `8d29093c…`. It was filled by `fill_manifest_sha.py` (sha256 `0b6510a9…`) from independent hashes (`inputs/shas_hb.json`) before any array was opened.
- **Reader tests on this export:**
  - `test_s1_read_v2.py`: 36 OK;
  - `test_a5_reader_hbonly.py`: 7 OK (`logs/`).
- **Evidence class:** MEASURED.
- **Population:** 3,207 H48 frames over 45 log groups.
- **Estimator:** the reader's log-cluster bootstrap. It answers whether another draw of logs would give the same result. It is blind to training variance, and every number is "single head seed" (`H-ESTIM-SEED-1`).
- **Tier:** a perception read on navtrain H48 (S1). This is not a driving number.

## How HB-s0 was completed (amendment A1, section 5)
- **`--h48x-only`, on HB-s0's saved state** in pause stage `A5HBFIX`: rc 0, 14:08:06 to 14:16:31Z.
  - Measured cost: 8.4 min, about 0.14 L40-h. Card 5(f) estimated 0.1 to 0.5.
- **`h48x_provenance.json`:**
  - update read back 10000 / 10000;
  - `aux_head_equal` and `hb_lora_equal` both True;
  - frames 1,502 equal the audit's 1,502 excluded frames;
  - loaded sha256: `aux_drv.pt 7b881363…` and `scorer_states_last.pt 5bedea92…`, equal to section 5(a).
- **`vdac_postgate.py --updates 10000`:** `ZZVDAC_POSTGATE_OK`, 2,440,450 head parameters (`logs/hb-s0_postgate_and_egress.log`). No `ZZSFT_DONE` or chain marker was written by hand.
- **Egress:** `ZZEGRESS_OK arm_vdac_hb-s0_a1`, 10 files, 101,254,178 B, to HF `vdac/hb-s0`.
- **HB-SHUF** ran its own chain on the fixed tree:
  - `ZZVDAC_CHAIN_DONE HB-SHUF` at 17:06:06Z;
  - post-gate OK at 2,000 steps;
  - run check: 2,000 updates on 60,000 sets, 1,502 H48 frames forwarded;
  - LFS sha256 equals the pod's (`logs/`).
- **P7 resumed at 17:06:06Z.**
- **Pulls** (`HF_HUB_DISABLE_XET=1`; `inputs/pull_record_*.json`): every local sha256 equals the HF LFS sha256.

| key | file | sha256 |
|---|---|---|
| `hb_s0_a` | `head_pred_update10000.npz` | `7040a4ed…` (= amendment 5(a)) |
| `hb_s0_b` | `head_pred_h48x.npz` | `296f667e…` |
| `hb_shuf_a` | `head_pred_update2000.npz` | `ce8db817…` |
| `hb_shuf_b` | `head_pred_h48x.npz` | `0d5f7911…` |
| `hb_init_a` | `scores/head_pred_step0.npz` | `2ebe4b54…` |

- **`hb_init_a` is byte-identical to HA's `head_init0_a`.** The pod's `sha256sum` and the HF LFS hash both give `2ebe4b540b16…`. PREREG section 7.5's precondition (HB step 0 agrees in sign with HA's init on ≥ 99.9 % of cells) therefore holds at 100 %, by identity: the LoRA copy starts as the trunk.

## Result (`out/s1_read_v2.json`, `out/s1_read_v2_tables.md`)
| arm | S1-P1 IoU [0, 40) (bar ≥ 0.80, point decides) | S1-Q (bar ≥ 0.90) | S1-F lower bounds (P1, Q) | C-SHUF (< 0.50) | C-FLIP (< 0.80) |
|---|---|---|---|---|---|
| HA-s0 (reference, banked arm) | 0.7802 [0.7587, 0.7974] | 0.8332 [0.7951, 0.8691] | 0.512, 0.089 | 0.4468 | 0.7056 |
| **HB-s0** | **0.8113 [0.7924, 0.8260] holds** | **0.8376 [0.7963, 0.8744] fails** | 0.545, 0.090 (holds) | HB-SHUF 0.4464 ok | 0.7031 ok |

**Paired HB-s0 − HA-s0** (same frames, same resamples):
- S1-P1: **+0.0311 [+0.0224, +0.0359]**, separated.
- S1-Q: **+0.0043 [−0.0046, +0.0119]**, not separated.
- X-LIFT: False.

**Further readings for HB-s0:**
- Per city IoU: LV 0.908, BOS 0.773, PIT 0.774, SG 0.729.
- Bands: 0–20 m 0.834, 20–40 m 0.787, 40–72 m 0.703.
- Near-boundary SDF MAE: 1.436 m, against HA's 1.630.

**Verdict (mechanical, card section 8): HB-s0 = X-BOUND.** S1-P1 holds and S1-Q fails.
- The LoRA trunk copy moves the drivable-area IoU over its bar.
- It does not move the ranking of DAC-0 against DAC-1 proposals (S1-Q).

The separated S1-P1 lift is necessary, not sufficient, for a lever claim. A seed replicate is the sufficient form (card, `H-ESTIM-SEED-1`).

**Not read:**
- in-sample IoU of HB: the `aux_drv/2` loader is not built (card 7.6);
- `p6_dac_logit` and S1-QT: inputs missing, reported by the reader, not deciding.

## Next (committed rows, Rule Zero)
1. **HC is the pair's other arm.** It is running now in its own P7 pause (stage `A5HC`; PI 2026-11-13: "ok to exceed cap"). The pair is read together: HC's row is read after its chain, with `manifest_a5_hc.json`.
2. **The committed successor of X-BOUND is HBC:** 1 m cells plus a boundary weight on the LoRA copy, about HB's cost plus 1.1 L40-h per arm.
   - It is new pod time, so it is the PI's decision.
   - Its card is written after HC reads, so that HC's row can enter it.
3. **What X-BOUND says for the driving goal:** a head that segments drivable area better but does not rank DAC better gives the scorer nothing new.
   - S1-Q is the clause that gates S2 (the E3 read).
   - That gate is not passed.
