# PREREG_P5C — Amendment A2: P5-R is the label unconditionally (PI 2026-11-12), the launch kit's deviations from the card, and the W-A7 launch-blocker read

**Status: FROZEN by the Master Mind 2026-11-14** (sha256 in `PREREG_P5C_A2.sha256`).
- **What existed at freeze, and what did not:**
  - No pod label of either kind (Tc20s or P5-R): the label runner has never run, because the pod is training P7.
  - No W-A7 number.
  - No P5-C training step and no P5-C read.
- **Neither earlier card is edited.** `PREREG_P5C.md` (sha256 `49338ab8…`, `PREREG_P5C.sha256`) and `PREREG_P5C_A1.md` (sha256 `3f81bfda…`, frozen 2026-11-12T09:45:48Z) stand. This amendment sits beside them.
- **Launch kit:** banked at `ea45c6bb`, `PK/raw/2026-11-14-p5c-launchkit/` (tar sha256 `71888abf…`; `HANDOFF.md`, `HANDOFF_2.md`, `RUNBOOK_P5C.md`, `TESTS.md`).

## 1 · The label: P5-R, unconditionally (supersedes A1 section 0)
- **The ruling.** On 2026-11-12 the PI answered the MM's recommendation, "run P5-C on P5-R labels regardless of the R4 trigger", with *"follow your recommendation"*.
  - A1 section 0 made P5-R conditional on the R4 trigger, and its last line called a switch without the trigger "a moved goalpost".
  - **This switch is not that.** It is made BEFORE any pod label exists, so no Tc20s or P5-R pod result can have informed it.
  - It rests on two things: dev-box evidence banked before A1's freeze, and a PI ruling.
  - **The dev-box evidence** (`RESULT_P5R.md`, rank 0, 1,200 rows over 294 logs; MEASURED):
    - TB-6 heading-rate ratio: P5-R 1.564 vs Tc20s 1.950;
    - acc ratio: 1.158 vs 1.736;
    - sim8 − Tc20s: −0.08 pp [−0.27, 0.00].
- **What carries over from A1.** A1 sections 1 (all six points), 2 (R-A to R-D), U and 3 apply UNCONDITIONALLY, with P5-R as the label.
  - Tc20s is still built: it is P5-R's warm start.
  - Its gates (with R-A's 1e-9) still run and are reported.
- **Label selection, committed now in this order:**
  1. **P5-R**, if its gate set is ZZTB_ALL_PASS, or fails only by the R-D pattern (rank-1-only TB-6 with rank 0 passing and policy rank0).
  2. Otherwise **Tc20s** (the frozen card's own label), if its gate set passes on the same rule.
  3. Otherwise **no launch**. The next lever is P5-T (PREREG_P5R section 10).
- **How the selection is recorded.**
  - The runner's printed `ZZP5CL_LABEL` line follows A1's logic and is **informational** under A2.
  - The MM writes the selection, with the two `gates.json` files it read, into `LABEL_SELECTION.md` BEFORE filling `P5C_LABEL_TABLE_SHA256`.
  - `P5C_LABEL` is set from that file and never defaulted. The chain refuses without it.

## 2 · Arms and the "C0r" naming
- **What the PI approved.** The PI's 2026-11-12 *"follow your recommendation"* approved the MM's recommendation, which read "P5-R labels **with the matched raw control C0r**, ~57–58 L40-h". The "C0r" wording was the MM's, not the PI's.
  - That cost is the C1 + C0 pair (card section 14).
  - So the recommendation's "C0r" is the **card's C0**: the matched raw-label fine-tune, with no table.
- **The card's C0r is not approved.** It is the `--seed 1` replicate of C0, about +28–29 L40-h.
  - It is NOT covered by that approval. It stays PI-gated, as in card section 17.3.
  - The chain runs it only with `--with-c0r`, which the MM passes only on a PI yes.
  - Without it, the verdict ceiling is SEPARATED-SINGLE.

## 3 · Pins that moved (pre-data, disclosed)
| file (`git hash-object --no-filters`; sha256) | A1 | A2 | why |
|---|---|---|---|
| `a1/p5c_pod_label.sh` (runner) | `f9220d1f…` | `d25e1a92b4e359e408f5602c467df97ab7ff85b7`; sha256 `f3b0046f2bc17c0eda0343fd6e689e0709821cfd60daccbe9d663e18a575369a` | passes the rank policy and `--meta-json`, asserts the policy in verify, and pins the table module (`PIN_TABLE_MOD`) |
| `refe/traj_label_table.py` | `9592b62c…` (A1 Point 2 / POD_SHIP_LIST) | `d65e01c43ab3145a093a85ac101f740c7d4ed839`; sha256 `ea763fb2bd857f172acba32d1a5a76ee17ae08077801bbc096970069451d77d8` | `--rank-policy all\|rank0` is REQUIRED (no default); `--meta-json` is sha-bound; the one reader ENFORCES the declared policy |
| `refe/p5c_gate.py` (new) | none | `b4d2e34abff1bacea27626c74cbfe03eb38b8647`; sha256 `bc187a84…` | G-LIVE (f) and (h) |
| `refe/p5c_postgate.py` (new) | none | `54275c42cf965799606143ef625b3804d4eef1d7`; sha256 `2695de23…` | post-training G-ADAM and the weight checks |
| `refe/p5c_pod.sh` (new) | none | `cf82384478d4ece850f7838dc1f2169585ff9b24`; sha256 `6b6e121f…` | the pod chain |

- **A1 Point 2's other blobs are UNCHANGED.** The MM re-read them at tip `4e21490f`: `p5r_labels.py 16d2aaf0`, `p5r_run.py d0a4084b`, `p5c_dual_labels.py 50b9f0db`, `p5c_table_gates.py 5c2f49cf`, `test_p5c_table_gates.py 82ce4a58`, `calibrate_A1.py 001e87e7`, `p5_target_labels.py 294644df`.
- **Any later edit to a file in this table moves its pin, and that needs Amendment A3.**

## 4 · The kit's deviations from the card's text (each one pre-data; none weakens a check)
| card | deviation | reason |
|---|---|---|
| §3 scorer row: "`--scorer-mode` as P7" | no `--scorer-mode` flag | P7's token-bound argv carries no `--scorer-mode`, and `refe_p6` FORBIDS it. "As P7" is what the card means, and that is met. |
| §10.1 "every other `refe_p7` entry unchanged" | **G-RESERVE dropped** from `refe_p5c*` | G-RESERVE requires `--aux-drv-labels`, and the aux head is OFF (§17.2). The reserve stays out through the bank pin instead (150,603 tuples, `targets_sha256 92af8d66…`). |
| §10.1 `resume`: "forbidden at start" | `--resume` is allowed in the argv | At MINT the gate refuses an `--out` that holds `ckpt_last.pt`, so the first start is fresh (fresh optimiser). A relaunch of the same arm after an interruption may resume its own state, as in P7. |
| §10.2 (f): "≥ 64 real rows equal `planner.py`'s strict load" | a bit-level digest of EVERY tensor in the trainer's model state at its first `zero_grad`, against the base file | RED arms: `warm_not_loaded` and `warm_perturbed`. Identical tensors prove identical weights. **Limit:** the digest does not run rows, so it does not test the forward path. That stays with G-EVAL (inherited unchanged: "the eval loader builds the identical model"). (f) + G-EVAL covers the card's (f). |
| §10.3 G-CKPT: "the meta carries `init_from_sha256`" | it is carried in `config.json` (`warm_start.sha256`) | That is where `train.py` writes it. The post-gate reads it there and must equal `P5C_BASE_SHA256`. `train.py` is not edited for a key it already records. |
| §10.4 G-ADAM: "non-zero `exp_avg_sq`" | non-zero `exp_avg` AND `exp_avg_sq` | Stricter. The control (one parameter's state deleted) must FAIL. |

## 5 · Gate evidence
- **G-LIVE-LAB** is reused from P7 (`P5C_REUSE_P7_LAB=1`) only if P7's `lab.json` was measured at `--label-workers 13`, `--label-chunks 2`, on the same D9 metric cache. Otherwise it is re-measured.
- **G-LIVE-MEM** (the P5-C model has no aux head) and **G-EGR** are always re-measured, with C1's argv.
- **G-LIVE (f) and (h), and G-CKPT,** are produced by the gate itself.
- **Label pod:** `P5C_LABEL_WORKERS = 13`, as P7. `P5CL_NSHARD` ≤ the pod's `nproc`; the chain refuses above it.

## 6 · W-A7, the launch blocker (§10.5, §17.6): the read is defined here, before any number exists
- **Question.** The labels are built without A7. Training's online labeller and the read both apply A7 (`refe/planner.py` `repair_last_heading`, :159-167, which sets the last pose's heading to the previous pose's).
  - Does applying A7 to the label change its tracked PDMS?
  - [H] about 0: only index 19's heading changes.
- **Population.** The card's 1,018 navtrain held-out rows: the rows of `p5c_labels.py control`.
- **Labels.** Built with the pinned dual labeller at A1's settings:
  - **P5-R** decides;
  - **Tc20s** (the card's named check) is reported and decides nothing.
- **Arms.** For each label, tracked PDMS **without A7** (the label as built) and **with A7**.
  - A7 is applied by the code that training and the read use (`repair_last_heading`, or `transp_score.py`'s A7 option if it calls it). It is never re-implemented.
- **Scorer.** The tracked navtrain scorer: NAVSIM's PDM scorer with its LQR tracker, as in TF1 / E2. `eval/instruments/transp_score.py`'s tracked arm with `--metric-cache-root`, pinned by blob at run time.
- **Estimator.** Δ = PDMS(with A7) − PDMS(without A7), per label. Paired log-cluster bootstrap over the rows' logs: 10,000 draws, seed 20260927. It answers "would another draw of logs say this?" only.
- **Controls** (each must read its known value, or the read is INVALID):
  - **(i) Reproduction.** The no-A7 Tc20s arm reproduces the banked TF1 / E2 tracked Tc20s per-row PDMS on these rows within 1e-6, where a banked per-row value exists. The n compared is reported.
  - **(ii) RED arm.** A +0.5 rad rotation of index 19's heading must change PDMS on at least 1 row.
  - **(iii) Identity.** A7 applied to a label whose index-19 heading already equals index 18's changes nothing (bit-identical).
  - **(iv) Determinism.** A rerun of one arm is bit-identical.
- **Outcome rows (committed now; P5-R's Δ decides; the point estimate is read with its 95 % CI):**

| row | condition | next |
|---|---|---|
| INVALID | any control fails | fix, re-run |
| **W-A7-PASS** | NOT (CI upper bound < 0 AND point ≤ −0.10 PDMS) | the launch blocker is cleared. A Δ that is separated but larger than −0.10 is disclosed with every P5-C number |
| **W-A7-REBUILD** | CI upper bound < 0 AND point ≤ −0.10 PDMS | the P5-R labels are rebuilt with A7 applied inside the solver's simulation BEFORE launch (Amendment A3, new pins), and W-A7 is re-read on the rebuilt labels |

- **Why the −0.10 bar.** It is one seventh of the read's bar B = 0.7031 (card §17.1). A smaller label-side shift cannot move a two-arm verdict.

## 7 · What A2 does not change
- Bar B = 0.7031.
- The arms' recipe (§3), except as section 4.
- The aux head is OFF.
- K accounting: one entry per arm, both scorers.
- The PI's two conditions: every number tracked AND tracker-transparent, and the wording "the student's plans survive the benchmark's tracker".
- W-TRANSP and W-FAM are READ blockers (§17.7).
- The closure-land rule (§10.6): the kit's code lands only after P7's last arm trains.
- The launch-gate PASS token.
