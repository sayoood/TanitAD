# PREREG_V8F_PROOF_ADD139 — RHACC10K: does the adopted stage B (RH) pass the acceptance checks C1, C2, C3 and C5 of the candidate specification, read on box 2?

Frozen before any tool of this card is written and before any value of it exists. Open at this time: row 204 (card ADD137, landed). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance.

## 1. Why this card

The PI adopted RH as stage B of the training recipe (2026-11-24, "Adopt RH", relayed by the other session) and then asked, verbatim: "use the devbox 2 to continue the work, can we startr scaling on devbox2?". Section 5 of `CANDIDATE_SPEC_V8F_10K_2026-11-20.md` names the checks a model trained on its own rolled history must pass before readout K can be dropped from the candidate; rows 172 to 204 were read on the line models UE_s0..s3 only. This card is that read for RH. Its verdict is the fork's answer to "ready to scale": the recipe is the scaling candidate if C1, C2, C3 and C5 pass.

Stated guess before data: C1 passes (RH was trained on banks read out under `causal_g1x`), C2 passes on the label form and fails on the reader form (the reader's quality is unchanged), C3 not separated, C5 fails on position and heading (the cost of 6 s of own driving is far larger than what 3,000 steps of stage B were sized for). Name guessed: `RHACC10K-NOT-YET`.

## 2. Models

| model | source | sha256 (first 8) |
|---|---|---|
| RH_s0, RH_s1 | the other session's private HF repo `tanitad-v8f-rollhist-runs` (registry section 20) | 351bb5f2, c387f4ce |
| UE_s0, UE_s1 | the fork's copy (`integ10k/ckpt`, banked in `small_set.tar` of `tanitad-v8proof-fork-bank`) | as in the fork's records |

RH_s2 is only on the other session's pod and is out of the fork's reach; two model seeds is the floor of section 2.3 of the specification. RH_s<m> is paired with UE_s<m> (RH_s<m> is initialised from UE_s<m>).

## 3. Runs (box 2, one job at a time, safe mode of box 2; per model seed m in {0, 1})

All runs: `--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal`, heading bound `--kappa-max 0.2`, study wrapper `integ10k/v8f_t1_eval_reader_rule_rzgfh.py` (sha256 373ffe72f2655c35, unchanged), eval-141 grid of record (2,399 windows) unless the row says otherwise. Readout K = `--readout causal_g --acc-gain 0 --path-exec`; causal readout with own acceleration = `--readout causal_g --acc-gain 1 --path-exec`. Reader form `RR` with the reader and the map level 0.50 of the line candidate (`RQ50`).

| run | model | form | readout | rule every | grid | check |
|---|---|---|---|---|---|---|
| a `H_R0K` | RH_s<m> | R0 | K | - | eval-141 | C1 |
| b `H_R0A` | RH_s<m> | R0 | causal, own acceleration | - | eval-141 | C1 |
| c `H_RLcA` | RH_s<m> | RLc | causal, own acceleration | 5 | eval-141 | C2, C3 |
| d `H_RQ50A` | RH_s<m> | RR, 0.50 | causal, own acceleration | 5 | eval-141 | C2, C3 |
| e `U_RLcK` | UE_s<m> | RLc | K | 5 | eval-141 | C2, C3 partner; control k4 |
| f `U_RQ50K` | UE_s<m> | RR, 0.50 | K | 5 | eval-141 | C2, C3 partner |
| g `H_RLcA_P30` | RH_s<m> | RLc | causal, own acceleration | 5 | grid of record of card ADD70 (1,271 windows), pre-roll 30 steps (6 s) | C5 |
| h `H_RLcA_P0` | RH_s<m> | RLc | causal, own acceleration | 5 | the same 1,271 windows, no pre-roll | C5 |

Order of record: m = 0 runs e, a, b, c, d, f, g, h; then m = 1 in the same order. The partners e and f are run again on box 2 and never mixed with the pod's dumps (one environment per pair). This card has no map of arms to controls beyond the column "check" above.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, no value is read, and a successor card is frozen)

* k1 complete: every run ends with its done file, 141 tick files (eval-141) or the file count of the ADD70 grid, no non-finite value in any dumped plan.
* k2 models: the sha256 of every checkpoint equals its row of section 2 in full (the full hash is read from the registry or the fork's records before the first run).
* k3 argv: every run's logged argv carries exactly the flags of its row in section 3 and no other lever flag.
* k4 environment: run e for m = 0 on box 2 reproduces the pod's banked `RLcKF5Hb20` of UE_s0 within 0.010 on U and within 0.010 on C (levels of section 3 of the specification: 0.0391, 0). The pod dump is read for this level only.
* k5 reader input: from the wrapper's source (file:line), the reader form takes its input from the eval cache tokens and not from the model's latent; if it reads the model's latent, rows d and f are not read and C2 is read on the label form alone, stated as such.
* k6 identity: RH_s<m> and UE_s<m> differ in sha256 and their run-a and run-e dumps are not byte-identical.

Names opened after the controls pass, in this order.

## 5. Reads (landed readers, unchanged)

Headway U, C, F with the paired episode-cluster bootstrap pooled over the two model seeds (`headway_read_r2.py`); the four families with the two-level family reader of card ADD46 and the spread of the convention over the model seeds as the bar; STRATEGIC with the route-goal reader of card ADD66 at the third tick, 30, 50 and 80 m; C5 with `integ10k/read_longroll10k_n4.py`. Every family cell is reported per family; none is pooled.

* C1: pair b minus a reads no family WORSE and F not higher (separated). Name `RHACC10K-C1-PASS` or `-C1-FAIL`.
* C2: c minus e and d minus f read no family WORSE and U not higher (separated). Names `RHACC10K-C2-LABEL-PASS`/`-FAIL`, `RHACC10K-C2-READER-PASS`/`-FAIL`.
* C3: route goal at 30, 50 and 80 m of c minus e BETTER, or not separated with C1 and C2 passed. Name `RHACC10K-C3-PASS` or `-C3-FAIL`.
* C4: not read here. INHERITED from the other session's registry section 20: the T0 shift of RH against B0 is inside the four-seed floor of 0.1174 at every seed.
* C5: g minus h reads not separated or better on every family and U not separated or lower. Name `RHACC10K-C5-PASS` or `-C5-FAIL`, with the per-family cells beside the line-model reference of the specification.
* Overall: `RHACC10K-SCALE-CANDIDATE` iff C1, C2 (label form at least) and C3 and C5 pass; otherwise `RHACC10K-NOT-YET` with the failed checks named and, per Rule Zero, the next lever for each.

## 6. Cost and limits

Box 2 (the PI's second computer, RTX 4060 8 GB), safe mode (4 threads, below-normal priority), one job at a time, `BOX2_BUSY.txt` written; telemetry and the temperature guard of the box 2 chains on. 16 runs. A dry run of one run (m = 0, row e) is timed first; if the projected total exceeds 10 h the card goes back to the PI with its cost before the rest. Disk: D: has 4 GB free; the queue stops if D: falls below 1.5 GB. No training, no encode. The fork does not reach the other session's pod.
