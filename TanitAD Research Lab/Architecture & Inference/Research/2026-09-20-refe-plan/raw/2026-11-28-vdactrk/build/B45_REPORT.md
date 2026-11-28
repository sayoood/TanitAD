# B4 + B5 build report (PREREG_VDAC_TRK.md, sha256 099a429f...)

Built by a build agent, CPU only, nothing run on real heads (none exist), no git writes. Deliverables under `E:/Projects/TanitAD-artifacts/vdactrk/build/b45/raw/2026-11-28-vdactrk/`.

## Deliverables

| file | intended repo path (PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`) |
|---|---|
| `vdt_read.py` (B5) | `PK/raw/2026-11-28-vdactrk/vdt_read.py` |
| `test_vdt_read.py` (B5 tests) | `PK/raw/2026-11-28-vdactrk/test_vdt_read.py` |
| `hc40_label_plan.md` (B4) | `PK/raw/2026-11-28-vdactrk/hc40_label_plan.md` |
| `hc40_label_run.py` (B4 skeleton, not run) | `PK/raw/2026-11-28-vdactrk/hc40_label_run.py` |
| `B45_REPORT.md` | `PK/raw/2026-11-28-vdactrk/B45_REPORT.md` |

## B5 `vdt_read.py`

Sub-commands: `calibrate` (HC40 only), `read` (navtest table, arms VT / VT-r / VT-SHUF / VT-0 / O-DAC-T, bars P1 P2 M G R X, all controls, outcome in the card's precedence, PROVISIONAL, K + 4), `confirm` (CTRL-T from the harness CSV at VT's picks to 1e-6 on every token, plus the four families; mirrors `l2next_read.py confirm`).
Reused, not re-invented: `refe/safehead_data.agg32` and `eval/p5c/p5c_read.py` Boot (imported from a landed PK export, refuses if seed/draws are not 20260927/10,000), the table sha pins of `l2next_read.py`, the table-lookup + controls pattern of `trkmin_read.py`.

Decisions the card left open (flagged for the MM; none moves a bar):
1. Ties on the gain go to the smaller m with a 1e-12 tolerance; "positive gain" means above 1e-9 (an exact 0 stays CAL-NULL).
2. If HT-s1 alone has no positive HC40 margin, VT-r is not defined: R is reported FAIL with the reason (outcome path SINGLE-SEED). The card is silent.
3. "none separated worse on a binding family component" is applied to VT; VT-r's families must be present and its separated-worse components are reported.
4. A control whose input was not supplied reads PENDING (None): never a pass, blocks ADOPT and finality. P1 P2 M G R all holding with CTRL-T or families pending gives `ADOPT-CANDIDATE`, not ADOPT.
5. P1 holds but M fails is "Unnamed" (no committed outcome matches); P1..R holding with a separated-worse family component is also "Unnamed".
6. If HT-s0 is CAL-NULL the navtest table is not read and the outcome is CAL-NULL.

## Input contract assumed (for B2 / B1 to match)

- navtest dump npz: `token [N] str`, `s_plan [N,64,6] float32` (C1 logits), `k_ship [N] int`, `dk_HT_s0`, `dk_HT_s1`, `dk_HT_SHUF` each `[N,64] float` (d_k in metres: min over the 41 tracked states x 4 corners of the head's signed distance; out-of-extent corner +6). Optional second-pass npz (same keys, subset) for the determinism control.
- HC40 dump npz: the same keys, plus `dk_HT_s0_flip [N,64]` (HT-s0's field y-flipped), plus for B4 `props8 [N,64,8,3] float32` and `log [N]`.
- HC40 labels: one `<log>.lab.npz` per log from `sh_label.py` (`token`, `tr_nc..tr_pdms [n,64]`).
- Side files: `iou.json {"HT_s0","HT_s1","HT_SHUF"}` (HC40 IoU, computed on the field side), `g0trk_c1.json {"n_match":777344,"n_total":777344,"second_dump_bit_identical":true}`, `--trk-npz` = banked `trk_c1.npz` (CTRL-PLAN, bit-for-bit on `s_plan`, `k_ship`), families json `{"VT"|"VT-r": {FAMILY: {component: {point, lo, hi, worse_if}}}}` (my adapter format; the real families output may need a thin adapter).
- Proposal axis j is the same axis in `s_plan`, `dk_*`, `props8`, `tr_*`.

## B4

`hc40_label_plan.md` quotes the producer of `tr` (`PK/raw/2026-11-20-p5c-next/p5c_prop_score.py`, one `TS.score_arms` call per token on the 8-pose proposals, navtest metric cache) and the HC40 equivalent `sh_label.py` (x150 copy, byte-identical to the 11-27 copy). `hc40_label_run.py` splits B2's HC40 dump per log, runs `sh_label.py` in the navsim venv per log, reports labelled/asked, optional byte-identity check. Dry-run (plan only) was exercised on a synthetic 3-log dump; the labeler itself was not run.

## Tests (see bottom for the logged counts)

`VDT_PK=<landed PK export> pytest -v test_vdt_read.py` -> `21 passed in 141.99s` (log line). Breakdown: 9 synthetic rule/outcome/boundary/family tests; 5 real-table tests (sha pins + CTRL-ID 12,146/12,146 and 92.5548; all-drivable field Delta exactly 0.0 at every grid m; always-veto = C1's pick; **true tracked-DAC field reproduces FILTER-ORACLE F_dac +1.5322694578 [1.1474, 1.9740] to 1e-9, 199 switched, 198 better / 0 worse**; calibrate -> read -> confirm end to end with oracle fields standing in for heads, reaching ADOPT-CANDIDATE then ADOPT, and a 1e-5 CSV corruption giving INVALID); 7 mutation tests.
Mutation record (each mutates one comparison in a copy and runs the synthetic suite in a subprocess; every one went RED): P1 `lo > 0` -> `>= 0`; P2 `>= 0.2316` -> `>`; M `hi < 0` -> `<= 0`; G `hi < 0` -> `<= 0`; X `lo > 0` -> `>= 0`; veto boundary `d_k >= m` -> `>`; dropping INVALID-on-X.
The real-table tests wait for the dev-box commit gate (free commit >= 6,000 MB; it swung between 0.7 and 8 GB during this session from other jobs) before loading the 80 MB table.
Not done / caveats: no head, dump, HC40 label or navtest-with-real-head number exists or was produced; the oracle-field end-to-end run is a plumbing check, not a result. `hc40_label_run.py` was only dry-run. The families json format is my assumption.
