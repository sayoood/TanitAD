# PREREG_P5C — Amendment A1 (CONDITIONAL, with one UNCONDITIONAL section U): P5-R as the label, only if the R4 trigger fires

**Status: FROZEN by the Master Mind 2026-11-12T09:45:48Z** (sha256 in `PREREG_P5C_A1.sha256`). Drafted by the labels/gates agents (rev 1 and rev 2, `a1/PREREG_P5C_A1_DRAFT.md`), reviewed and frozen by the MM. No pod P5-R or pod Tc20s label existed at freeze time: the pod is training P7 and the label runner has never run.
- It must be frozen **before any pod P5-R label exists** (MM_RULINGS_P5R R3). The pod's dual run writes P5-R labels, so freeze first, ship second.
- `PREREG_P5C.md` (sha256 in `PREREG_P5C.sha256`) is not edited. This amendment sits beside it.

## 0 · When it applies (conditional, strict)
It applies **only if the R4 trigger fires** (`PREREG_P5R.md` §10; `MM_RULINGS_P5R.md` R4):
- the pod's full-table **TB-6 on Tc20s reads rank-0 FAIL**, or RK0 stops P5-C, **and**
- the pod's P5-R gate set (below) is ZZTB_ALL_PASS, or fails only by a rank-1 TB-6 with rank 0 passing and policy rank0 (MM ruling R-D: passing with policy rank0).

If the pod's TB-6 on Tc20s passes, P5-C stays on Tc20s, this amendment is void, and P5-R stays banked as a contingency. If both label sets fail, the next lever is P5-T (PREREG_P5R §10). Switching the label without the trigger would be a moved goalpost.
- The runner prints the outcome as `ZZP5CL_LABEL tc20s | p5r | p5t`; a Tc20s failure outside TB-6 rank-0 / RK0 is `ZZP5CL_FAIL gates` (a pipeline fault, not a verdict).

## 1 · The six points of PREREG_P5R §10

### Point 1 · Title, §1, §7 wording
- Label name: **"tracker-aware regularised labels, λ\* = 3.0 (P5-R)"**. The arm still pre-distorts the teacher's own trajectory for the benchmark's LQR tracker; the pre-distortion is chosen by a regularised inversion (Δ² of the correction, PREREG_P5R R1) instead of P5-S's spline knot match.
- The PI ruling of 2026-11-11 (tracker-aware labels admissible; every number tracked AND tracker-transparent; wording "the student's plans survive the benchmark's tracker", never "drives better") **is confirmed to cover P5-R by the MM** (MM_RULINGS_P5R R2). It is the same class of label. §7's wording is unchanged except the label name.
- R2's second step stands: the PI is told explicitly before any P5-C launch, with the C0r question and the 57–58 L40-h cost.

### Point 2 · §4.1 builder
- **Builder:** `iterate20` (unchanged; the Tc20s of §4.1, the warm start), then `p5r_labels.solve` at λ\* = **3.0** (`SG/p5r/raw/lambda_star.json`, status OK, selected on TUNING before any EVAL row; sha256 `83988dda…`).
- **Driver:** `SG/p5c/a1/p5c_dual_labels.py` — one pass writes Tc20s (the fields `p5c_labels.py` writes, same lines) and the P5-R label with its S1–S5 fields. P5-R is warm-started from Tc20s exactly as the P5-R EVAL run did (d0 = Tc20s − traj, λ list [λ\*]).
- **Constants (PREREG_P5R §2, unchanged):** K_MAX 10, forward-difference step 1e-3, LM damping 3 candidates per iteration, stop after 3 consecutive failures or ‖δ‖∞ < 1e-9, L = 2.588 m/rad, grid-selected λ\* only.
- **Code blob ids (`git hash-object`, no filters):**

| file | git blob | sha256 |
|---|---|---|
| `p5r/p5r_labels.py` | `16d2aaf05eea5c00bd081628e07b02b7dc7f480f` | `84e1cac32510fb277ad7d1a876059152dbded5076b4fd1928f5ade330b6c468d` |
| `p5r/p5r_run.py` (imported for `Sim`, `sanity`) | `d0a4084b7187005df3e966a1ba0132a42a61e7d2` | `58faf7a76cb1840f4a8ed76ee6eaa8bb1d8c835f20009cb7a2749990cc7feddb` |
| `p5c/a1/p5c_dual_labels.py` | `50b9f0dbaa22563239d6c711e9f7dab846e1cc69` | `07e84b1cfd26d72c4fef7ea5557b07460e39f808a6bbba6a4d0df582ea7bd197` |
| `p5c/a1/p5c_table_gates.py` | `5c2f49cf485241df4d7e7949ed629a04fd324d5c` | `d600a83042c3e66087417f69a6c1b5f13cdada29ab4d624ddcf7a47b12c5a633` |
| `p5c/a1/p5c_pod_label.sh` | `f9220d1fe21a07ca70e848635257d5b1d4f2d6c0` | `d1d9bbad6d228ec2ff915146d4b8091561853ae8f2fe04351eaaad2eb415a628` |
| `p5c/a1/test_p5c_table_gates.py` | `82ce4a583d7840c1199c25942030eefa38bca20f` | `e2e8295a1bc6ff86a466fc7f8d4f375a63cb4d9008a4fffbe30a1409bfe3d3bc` |
| `p5c/a1/calibrate_A1.py` | `001e87e79f9fd5d00a4a1905a95df32cd2b951f9` | `b555ba8b5897143a89fed1faa1e2437f1fc9d0be4e2be89ecb05629f10c606f2` |
| `p5_target_labels.py` (unchanged) | `294644df61ca479097971ebc92b033e24f3b7019` | `2b31af64d6f6bcff7ffc0a59d524528b48ad8fceb44033e9cc845b201326c8d0` |
| `p5r/raw/verdict.json` (R-TOL-3 context value only, new ship file) | (not a code file) | `7ae694a3bb754d8b7ddbd355f0b16928498fefd91cb4d0b71c996d9587a17e9e` |

  (`git hash-object --no-filters`, recomputed 2026-11-12 after the R-A..R-D edits, MEASURED. The MM recomputes these at freeze; any later edit moves a blob and the pin. The runner pins the dual labeller and the gates by these sha256 values.)
- **Threads (MEASURED; MM ruling R-C):** the dual labeller runs the Tc20s part in the shard's env and the P5-R part single-threaded, because the P5-R EVAL run ran with every BLAS at 1 thread and its labels depend on that setting. `threadpoolctl` is OPTIONAL (the pod's `venv-teacher` has none; nothing is installed there): with it, `threadpool_limits(1, user_api="blas")` wraps every P5-R row; without it the labeller REQUIRES `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` all equal to "1" and otherwise refuses (exit 2, `ZZDUAL_REFUSE thread_pin`). The runner exports all three and asserts it. The DONE record (and the repair record) carries `thread_pin` = `threadpoolctl` | `env` and the three env values.
- **Fallback:** a row whose P5-R solve raises gets `p5r_error` and the identity label (label := `traj`, `source: raw_fallback`); fallbacks are counted per label; a Tc20s error implies a P5-R fallback ("no tc20s warm start"). S4 bar: ≤ 1.0 % per rank.

### Point 3 · §4.2 table gates (`p5c_table_gates.py --label p5r`)
| id | Tc20s (frozen, unchanged) | P5-R (this amendment) |
|---|---|---|
| TB-0 | pod vs dev-box Tc20s on the 482 logs: max abs diff **0.0** (**changed by R-A, section U: <= 1e-9 and sim8 identical where available**) | **R-B**, per rank, on the EVAL control rows (pod relabelling of the 2,400 rows vs `raw/eval_r{0,1}.jsonl`, coverage n == 2,400), ALL of: **R-TOL-1** sim8 verdict identical on >= 99.5 % of common rows; **R-TOL-2** per-row max abs label difference (all components, m / rad) median <= 1e-3 and p99 <= 5e-3 (the max is recorded, not gated); **R-TOL-3** TB-6 acc and heading-rate ratios of the relabelled rows each within +-0.01 of the dev-box EVAL ratios computed on exactly the SAME common rows (row-matched; `p5r/raw/verdict.json` is a recorded context value only) and sim8-fail rate within +-0.25 pp of the dev-box EVAL value. **Also** the Tc20s warm start must equal the dev-box Tc20s build within R-A's 1e-9, because it feeds P5-R. |
| TB-1 | `p5c_labels.py control`: 1,018 / 1,018, diff 0.0 (**changed by R-A: n == 1,018, n_errors 0, max abs diff <= 1e-9; bit-identical count recorded**) | that result **plus** a reproduction of all 2,400 EVAL labels by the pod labeller: coverage n == 2,400 and R-TOL-1/2/3 as TB-0 |
| TB-2 | spline knot exactness ≤ 1e-9 | **replaced by S3** (re-simulation identity): `tracked8` of the stored label, simulated alone, equals the solver's final `s`, ≤ **1e-6** on every P5-R row (PREREG_P5R §5.4; it also proves batch independence) |
| TB-3 | identity control | unchanged |
| TB-4 | 150,603 / 150,603, fallback ≤ 1.0 % | unchanged, plus S4 per rank |
| TB-5 | finite, \|Δyaw\| < π, `sim_hz` 10 | unchanged, plus S1, S2, S5 recorded true on every P5-R row |
| TB-6 | <= 2x per rank, rank-0 FAIL => STOP | **unchanged and binding** on the full P5-R table. P5-R's B1p is a projection, never a substitute. Rank-0 FAIL => STOP, next lever **P5-T**. **R-D:** a rank-1-only FAIL is not a failure of the label (section 4.3: rank-0 only); see section 2 R-D |
| TB-7 | `n_differs` per rank | unchanged, on the P5-R label |

Every changed or new gate has a deliberate-regression arm that fails it by name (`test_p5c_table_gates.py`, `TEST_RUN_A1.log`, 163/163 with the R-A..R-D arms and the row-matched R-TOL-3).

### Point 4 · §4.3 rank-1 policy and RK0
- **r_k is recomputed with P5-R labels:** sim8-fail rate(T) − sim8-fail rate(P5-R label) on the same fixed-rule samples (lowest sha256("log|token"), 300 rows per rank, rows with a local DB), same estimator (P5 §3 paired log-cluster bootstrap, ratio, 10,000 draws, seed 20260927). The pod runs `p5c_dual_labels.py repair`, which writes `p5r_sim8` beside the raw and Tc20s sim8; in `--label p5r` mode the gate reads `p5r_sim8` and **never** the Tc20s column (a Tc20s-only repair file gives UNDECIDED, tested).
- **RK0 is unchanged** (r_0's upper bound < 0 ⇒ no launch; next lever P5-T under this amendment). RK1–RK5 are unchanged, with TB-2 read as S3.
- ⚠️ **Disclosure (carried from §4.3 / §17 item 9):** the rank-1 rule was written after the MM's one-line summary of the Tc20s repair. The P5-R repair has no such prior reading, but its rule is that same rule.
- **Reference (MEASURED, dev box, P5-R EVAL split, not the 4.3 estimator; `RESULT_P5R.md` bars B2/B3, rank 0):** P5-R − raw sim8 −4.92 pp [−6.50, −3.48]; P5-R − Tc20s −0.08 pp [−0.27, +0.00]. Context only; it decides nothing here.

### Point 5 · §10 launch-gate pins
- `P5C_LABEL_TABLE_SHA256` becomes the **P5-R table's** sha256 (the runner prints it with `label=p5r`); table meta `label_name` = `P5R`, plus `p5r.lambda_star`, the PREREG_P5R sha256 `cfd23e06…`, the code blob ids above, and the rank policy.
- The G-DVB `traj_label_table` row reads the P5-R table's sha; the mutation list is unchanged. Add the code blobs of Point 2 to the pinned set.
- ⚠️ A pin edit moves 6 profile fingerprints pinned in 3 tests: run every launch_gate-reading test (~69) before landing.

### Point 6 · §14 cost (pod CPU for the label table)
Measured on this dev box (CPU only, 1 process, N1 chain running beside it; mini-bank of 40 rank-0 rows, 10 logs; `seconds_*` in the labeller's DONE record):

| | s/row | CPU-h for 150,603 rows | wall on 64 / 128 pod cores (η 0.8, ρ 1.0, ESTIMATED) |
|---|---|---|---|
| Tc20s only (`iterate20`) | 0.11 | 4.6 | ≈ 5 min / 3 min |
| P5-R part (solve + score + S3) | 0.49 | 20.5 | ≈ 0.40 h / 0.20 h |
| **dual, whole loop** | **0.60** | **25.1** | **≈ 0.49 h / 0.25 h** |
| P5-R EVAL run, for comparison (`RESULT_P5R.md`) | 0.52 | 21.8 | ≈ 20 min on 64 |

- Plus the repair sample: 2 × 300 rows × ≈ 0.7 s ≈ 7 min on one process each, run in parallel.
- The §14 line "label table ≈ 1 h of pod CPU" becomes **25.1 CPU-h**, i.e. ≈ 0.5 h wall on 64 cores. The runner default of 16 shards makes it 25.1 / (16 × 0.8) ≈ **2 h** wall: set `P5CL_NSHARD` to the cores the pod really has. ρ and η stay ESTIMATED until measured on the pod; the dev-box per-row times were taken with the MM's N1 chain running beside the labeller.

## 2 · MM ruling R-A..R-D (2026-11-12)
The Master Mind ruled the three open decisions of the first draft (D-A1-1, D-A1-2, D-A1-3) and one runner bug. They are implemented exactly as ruled; no other number was chosen.

- **R-A (was D-A1-2; unconditional, applies to Tc20s whatever label is chosen).** Tc20s TB-0 and TB-1 become: max abs label difference **<= 1e-9** (m and rad) on every common row, AND the sim8 verdict identical on every common row where sim8 is available. `p5c_table_gates.py --tc20s-tol` (CLI default **1e-9**; `--tc20s-tol 0` reproduces the frozen bit-identity gate, and the library functions default to 0.0). The Tc20s warm start that feeds P5-R uses the same 1e-9. **R-A's sim8 half is vacuous for Tc20s:** the Tc20s records carry no sim8 field, so 0 rows are compared and the 1e-9 tolerance alone carries TB-0 / TB-1 (the comparison is implemented and tested, not exercised). Reason: the frozen 0.0 cannot survive a different CPU or BLAS; a thread-count change alone moved 28 of 40 rows by 9.4e-15 (MEASURED, HANDOFF), and the pod is an AMD EPYC 7452 against the dev box's i9-12900F. *(sim8 half: vacuous for Tc20s, see above; field `tc20s_sim8` on both sides if ever present.)*
- **R-B (was D-A1-1; conditional P5-R path).** R3's single `--p5r-tol 1e-6` is replaced by three criteria, each evaluated per rank on the EVAL control rows: **R-TOL-1** sim8 verdict identical on >= 99.5 % of common rows; **R-TOL-2** per-row max abs label difference over all components: median <= 1e-3 and p99 <= 5e-3 (max recorded, not gated; p99 = numpy linear percentile); **R-TOL-3** the relabelled rows' TB-6 heading-rate and acc ratios each within +-0.01 of the dev-box ratios **on the same common rows** (MM ruling on caveat 1: row-matched reference), and the sim8-fail rate within +-0.25 pp of the dev-box EVAL value (the mean of the control rows' stored `sim8`). The reference is the ratio of medians (`tb6_rank`, `p5_target_labels.py:388`) of the control records' per-row `learn` (lam* label) and `raw.learn`; if a row lacks them it is recomputed with the gate's `learn()` from the stored label and the bank trajectory (MEASURED identical, max diff 0, on all 33 fixture rows). `--p5r-verdict` stays as a recorded context value (`verdict_json_context`, with the difference to the reference); with full 2,400-row coverage the reference must agree with `verdict.json` to 1e-9; it is not gated. TB-0 and TB-1 for P5-R pass only if all three pass on every rank that has control rows and TB-1's coverage n equals 2,400. `--p5r-tol` and the 1e-6 default are removed; A missing `verdict.json` does not affect the gate. Each criterion has a deliberate-regression arm that fails it by name (`TEST_RUN_A1.log`).
- **R-C (was D-A1-3).** `threadpoolctl` is optional in the labeller and in the runner's import probe; mechanism and refusal are in section 1 Point 2, Threads. Nothing is installed on the pod.
- **R-D (runner bug against the frozen card).** PREREG_P5C 4.2 TB-6: "Rank-1 FAIL => rank-0 only (4.3)". The runner used to turn a line `ZZTB_FAIL TB-6` (rank 1 only, no STOP token) into `ZZP5CL_FAIL gates`. Now, when the ONLY failing token is `TB-6` and that label's `gates.json` shows TB-6 rank 0 PASS, rank 1 FAIL and `rank_policy.decision == "rank0"`, the label is classed `rank1only` = passing with policy rank0, for the Tc20s line and for the P5-R line, and `build_table` runs for it. If the rank-0 TB-6 also fails (token `TB-6-rank0-STOP`) the trigger/STOP path is unchanged; any other token, or a `gates.json` that does not show exactly that pattern, is still `ZZP5CL_FAIL gates`.

## U · R-A applies unconditionally, before any pod label (disclosed pre-data change to TB-0 / TB-1's tolerance)
R-A changes the tolerance of two gates of the **frozen** PREREG_P5C 4.2 (Tc20s TB-0: 0.0 to 1e-9; TB-1: bit identity to 1e-9). It is **not conditional on the R4 trigger**: it applies to the Tc20s table whichever label is finally chosen, and to the Tc20s warm start inside the P5-R gate. It is made **before any pod label exists**, on a measured mechanism (floating-point noise from the BLAS thread count / CPU, 9.4e-15 on the dev box), not after seeing a pod result, and it is disclosed here instead of being edited into `PREREG_P5C.md` (whose sha256 is unchanged). It loosens a bit-identity bar to 1e-9, about five orders above the measured 9.4e-15 noise. It adds a sim8-verdict identity requirement that the 0.0 bar did not have.

## 3 · Calibration of R-TOL-1/2/3 (MEASURED; `calibration_A1.json`, `CALIBRATION_A1.md`, `calibrate_A1.py`; rerun after the row-matched ruling)
33 real EVAL fixture rows (rank 0), dev box, CPU. Per-row max abs diff against the lam* = 3.0 EVAL labels (m / rad); R-TOL-3 against the row-matched reference:

| arm | bit-identical | median | p90 | p99 | max | R-TOL-1 (sim8 33/33) | R-TOL-2 | R-TOL-3 (row-matched) |
|---|---|---|---|---|---|---|---|---|
| control (1 thread, dev-box warm start) | 33/33 | 0 | 0 | 0 | 0 | pass | pass | pass (diff 0, 0) |
| **noise (a)** BLAS 4 threads | 0/33 | 6.0e-6 | 1.7e-4 | 3.4e-4 | 4.0e-4 | pass | **pass** | **pass** (acc -1.2e-7, hr -3.5e-7) |
| **noise (b)** warm start from a 1-thread Tc20s | 10/33 | 7.7e-7 | 4.2e-5 | 2.9e-4 | 3.4e-4 | pass | **pass** | **pass** (acc +1.6e-7, hr 0) |
| **regression** lam = 1.0 | 0/33 | 1.1e-1 | 3.4e-1 | 1.26 | 1.59 | pass | **FAIL** | **FAIL** (acc +0.060, hr +0.713) |
| **regression** lam = 10.0 | 0/33 | 1.0e-1 | 3.9e-1 | 0.85 | 0.95 | pass | **FAIL** | **FAIL** (acc -0.058, hr -0.441) |

Noise arms pass all three with margin (worst median 6.0e-6 vs 1e-3, about 170x; worst p99 3.4e-4 vs 5e-3, about 15x; ratio shifts 1e-7 vs 0.01); both regression arms fail R-TOL-2 and R-TOL-3. No threshold was changed. sim8 is 0 failures in all arms on these 33 rows, so R-TOL-1 and the sim8 half of R-TOL-3 are exercised only by the synthetic gate arms (429-row inflated fixtures). The earlier verdict.json-based comparison (which failed every arm on 33 rows through sampling of the subset) is superseded by the row-matched reference.

## 4 · MM freeze note
- Reviewed against `PREREG_P5C.md` (unchanged), `PREREG_P5R.md` and `MM_RULINGS_P5R.md` R2-R4. The code blob ids of Point 2 were recomputed at freeze and match the table; the runner's pins equal those sha256 values, and no pinned file carries a CR byte.
- The thresholds of R-A and R-B were set by the MM BEFORE the calibration was run; the calibration only checked that they separate the measured noise from the neighbouring-λ regressions. No threshold moved after it. The one change after the first calibration was the R-TOL-3 REFERENCE (row-matched instead of the 1,200-row `verdict.json` scalar), because the scalar compares different row sets; its threshold (±0.01) is unchanged.
- Limits that travel with every gate reading: R-TOL-1 and the sim8 half of R-TOL-3 were exercised only by synthetic arms (0 sim8 failures on the 33 real rows); R-A's sim8 half is vacuous for Tc20s; no rank-1 P5-R row could be run on the dev box (rank-1 fixtures are synthetic).
- R2: the PI is told before any P5-C launch (P5-R covered by the 2026-11-11 tracker-aware ruling; the MM recommends P5-R regardless of R4; the C0r question; 57-58 L40-h).
