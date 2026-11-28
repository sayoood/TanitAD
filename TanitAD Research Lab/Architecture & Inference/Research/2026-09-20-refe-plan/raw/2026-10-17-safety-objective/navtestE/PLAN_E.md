# PLAN_E: the navtest T1 read of the four SAFEOBJ u1000 states -> `navtest_E.json` (bar E), as a script

Written 2026-10-17 by a TanitAD build agent (Master Mind brief, IMPL_PLAN step 9). **Nothing here has run on a pod.** No R0d / R1d state exists yet; every cost below is cited from the free-point package (`PLAN.md`, which measured it) or marked.
Evidence classes: **M** measured (artifact cited) · **I** inherited from the free-point package, not re-measured here · **E** estimated · **U** unverified.
`<pod>` = the A40 pod (address deliberately not written here; it is read from `<work>/../pod_ssh.txt` by the shipper). Package: `navtestE/`. Free-point code used: byte-identical COPIES in `navtestE/fp/` (`fp_COPIES.md5`).

## 0. What is read, and what comes out

| | |
|---|---|
| states | `R0d_A`, `R0d_B`, `R1d_A`, `R1d_B` = key `A` / `B` of `scorer_states_last.pt` of arms R0d and R1d at **update 1000** (the worker refuses any other update) |
| model | the frozen u400 trunk `model_sft4_B_u400.pt` (md5 `8e382eeb…`); the trunk, generator and LoRA are frozen in every state, so ONE trunk pass serves all four scorers (free-point PLAN.md §1) |
| tier | **T1** = NAVSIM v1.1 PDMS on the full navtest, 12,146 tokens, 136 logs (pseudo-simulation of a 4 s open-loop plan; not closed loop) |
| estimator | paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927 (`stage3_full.read_pair` → `proxy_eval.boot`, imported). It answers *another draw of LOGS* only |
| output | `navtest_E.json` (exactly OUTCOME_BUILD §7 as revised 14:15: `tier`, `estimator`, `n_tokens` 12146, `n_logs` 136, `pairs`, `pooled` with `resample: "joint"`, `floors` with **only** `irep_abs_delta`) · `navtest_E.provenance.json` · `navtest_families.json` (four families per pair; STRATEGIC = PAUSED) |
| not in the file | **no seed floor.** The READER derives it (RL-val `\|R0d−R1d\|` and `\|R0−R1o\|` from bar L's inputs, plus the navtest `\|R0d−R1d\|`) and flags a `floors.seed_floor_abs` as IGNORED. This tool records the navtest part in the sidecar only, for cross-checking |

## 1. Gates before anything starts (a plan with an unmet gate is a refusal, not a delay)

| # | gate | why |
|---|---|---|
| G0 | R0d and R1d finished (or killed-and-decided); R1o finished or decided **for the READER's sake, not this tool's** | bar E is read after Stage 2 (PREREG §6). This producer reads no RL-val file. The reader takes clause 3's RL-val seed floor from bar L, which needs the RL-val reads of R0, R0d, R1d **and R1o**; without them the reader reports clause 3 unread (NOT_READ, A-E3b) whatever this file says |
| G1 | the A40 is free | CLAUDE.md: never add GPU/RAM load to a pod that is training |
| G2 | the worker's own gate `/workspace/ops/bank3c/full/started_epoch` (the P6 re-roll) is honoured, or the Master Mind passes `--no-gate` | inherited from the free-point worker (`fp_forward.py serve`, exit 3 when absent); it protects the P6 twin-bank build's CPUs, not this read — the Master Mind decides |
| G3 | `/workspace/ops/sobjE` is owned by this read alone | the free-point run uses `/workspace/ops/freept`; `sobj_ship.py` and `navtest_E.py plan` **refuse** that path |
| G4 | the free-point run has finished (`ZZCHAIN_DONE` in its `chain.log`) | steps 1–2 copy its finished artifacts; its worker deletes each frame chunk after use, so nothing of it is reusable on the pod (see open point 4) |

## 2. The sequence (every command is printed by `python navtest_E.py plan --work W`; `<dev>` = the dev-box download dir of the arm states, `$R` = `/workspace/data/rls1`)

Python everywhere: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -X utf8`; run from `navtestE/`. `W` = `C:/Users/Admin/sobjE/work` (any short path; MAX_PATH).

| step | where | command | done when |
|---|---|---|---|
| 1 | dev | `cp C:/Users/Admin/freept/work/{inputs_full.json,calib.npz,plan.json,ship_plan.json} W/` | the four files exist; their md5 are the free-point run's (IC1: 3,410 tokens bit-equal to the A9 recorded inputs, `work/IC1.json` `pass: true`, **I**) |
| 2 | dev | `navtest_E.py expected-states --state R0d_A=<dev>/rls1-r0d/scorer_states/scorer_states_u1000.pt:A --state R0d_B=…:B --state R1d_A=<dev>/rls1-r1d/…:A --state R1d_B=…:B --out W/EXPECTED_STATES_E.json` | `ZZEXPECTED_E_OK`: four tensor-set sha256, update 1000, each state's key equals its name's suffix, all four distinct. **This is the independent A/B binding** (computed from the HF copy, not from the pod) |
| 3 | dev→pod | `sobj_ship.py prep-pod --work W --remote /workspace/ops/sobjE` | `ZZSHIP_PREP_OK` (md5 of inputs/rig/`fp_forward.py` on the pod; u400 md5; model-code hashes) |
| 4 | pod | `ssh <pod> 'cd /workspace/ops/sobjE && OMP_NUM_THREADS=2 REFE_BACKBONE_ROOT=/workspace/data/backbones PYTHONPATH=/workspace/refe-plan-rls1/refe nice -n 10 setsid nohup /workspace/venv-teacher/bin/python fp_forward.py serve --work-dir /workspace/ops/sobjE --inputs inputs_full.json --calib calib.npz --ckpt /workspace/data/rls1/base/model_sft4_B_u400.pt --state R0d_A=$R/arms/R0d/scorer_states_last.pt:A --state R0d_B=$R/arms/R0d/scorer_states_last.pt:B --state R1d_A=$R/arms/R1d/scorer_states_last.pt:A --state R1d_B=$R/arms/R1d/scorer_states_last.pt:B --device cuda --repeat-first 128 > serve.log 2>&1 < /dev/null & echo $! > worker.pid'` | `serve.log`: `worker up … states [R0d_A, R0d_B, R1d_A, R1d_B]`. Stop only by pid: `kill $(cat worker.pid)` (never `pgrep -f` / `pkill -f`) |
| 5 | dev (detached: WMI `Win32_Process.Create`, CreateFlags `0x08000200`) | `sobj_ship.py run --work W --remote /workspace/ops/sobjE > W/ship.log` | `ZZSHIP_ALL_PULLED` (every chunk pulled and sha256-checked against the pod) |
| 6 | dev | `navtest_E.py seams --work W --chunks W/out` | `ZZSEAMS_E_OK`; `W/picks_report_E.json` (IC2 max, IC3, pick changes per pair, state_info, IC5) and `W/seam_{BASE,R0d_A,R0d_B,R1d_A,R1d_B}.npz` |
| 7 | dev | `fp/fp_score.py score --work W --points R0d_A,R0d_B,R1d_A,R1d_B --label-prefix refe_sobj_ --parallel 1` | `ZZSCORE_OK` ×4; the harness's OWN status `PASS` in `W/score_logs/refe_sobj_<STATE>.status.json` (a CSV existing is not evidence) |
| 8 | dev | `navtest_E.py families --work W` | `ZZFAMILIES_E_OK`: `W/families/sobj_<STATE>_families.json` ×4 (families6, the free-point call) |
| 9 | dev | `navtest_E.py build --work W --out-dir W/result --irep-json <irep.json>` | `ZZNAVTEST_E_OK <VALID|IDENTITY-FAILED>`; exit 2 and **no** `navtest_E.json` on any hard-gate failure. No `--rlval-arm` option exists any more |
| 10 | dev/pod | `python outcome/safeobj_outcome.py --arm R0d=$R/arms/R0d --arm R1d=$R/arms/R1d --arm R1o=$R/arms/R1o --arm R0=<R0 arm dir> --navtest-e W/result/navtest_E.json --dose raw/2026-10-14-rls1b-attribution/dose.json --prereg raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md --addendum raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.addendum1.md --out …` (OUTCOME_BUILD §3; the reader exits 4 without the registered addendum) | the reader's E record (`ADOPT` / `NOT_PROVEN` / `REFUTED` / `NOT_READ`); this tool decides nothing. The `--arm` set is what lets the reader derive the RL-val seed floor |

Before step 9 the I-rep file must exist (open point 1): `navtest_E.py irep --a <csv of run 1> --b <csv of run 2> --out irep.json` writes `{"irep_abs_delta", "unit": "PDMS x100"}`; the tool refuses a missing file, a negative or non-finite value and any unit but `PDMS x100`.

## 3. Cost (4 states instead of the free-point's 3; every rate cited)

| step | resource | estimate | basis |
|---|---|---|---|
| 1–3 | dev box, small uploads | < 5 min | **E** |
| 5 ship | uplink dev→pod, **9.7 GB re-upload** (the free-point worker deletes each chunk after use) | **2.1–2.3 h**, concurrent with step 4 | **I** PLAN.md §3 (1.25–1.29 MB/s, a cap on the link); the live free-point log shows 1.29 GB in 1,095 s = 1.18 MB/s (**M**, `freept/work/ship.log`) ⇒ 2.3 h |
| 4 forward | A40, 1 process, fp32, batch 1 | 1.7–2.5 h GPU if fed; **upload-bound at ≈ 2.3 h** | **I** PLAN.md §3. Two extra scorer forwards per token (q_mlp → blocks → head on the same two tensors): **E** < 5 % of the trunk pass (**U**) |
| pod disk | `/workspace` | ≤ 2 GB frames + 0.6 GB outputs; the shipper holds when free < 6 GB | **I** (it was 97 % full on 2026-10-17: re-check) |
| pull | pod→dev | 12,146 × (traj 15 KB + 5 logit sets × 7 KB) ≈ **0.6 GB** | **E** |
| 6 seams | dev CPU | minutes | **E** |
| 7 harness ×4 | dev CPU (RAM guard ≥ 4 GB free) | **4 × 0.371 h = 1.48 h** at `--parallel 1`; 0.8–1.5 h at `--parallel 2` | **I** 1,335.7 s per 12,146 tokens (`P2_READOUT_RUNBOOK` §3 via PLAN.md); `--parallel 2` **U** |
| 8 families6 ×4 | dev CPU | ≈ 8 min | **I** ≈ 2 min each |
| 9 build | dev CPU | **≈ 15 s** | **M** 12.5 s for the real-table build, 12,146 tokens (`logs/`, this package) |
| **total** | | **≈ 3.9–4.3 h wall; A40 ≈ 2.3 h (upload-bound), 0 A40-h of training** | sum of the above |

## 4. Gates inside the read, and what they do (`navtest_E.py build` → `read_run_gates`)

| class | gate | on failure |
|---|---|---|
| **hard** | harness status `PASS` for each of the 4 states · `picks_report_E.json` present · IC2 (the replicated scorer path equals the model's own score) **max |Δ| = 0.0 on every token** · 12,146 tokens · each state's sha256 equals `EXPECTED_STATES_E.json` and `update == 1000` · the four tensor sets are distinct · each state's key is its name's suffix (A is key `A`) | status `INVALID`: **no `navtest_E.json` is written** (exit 2) |
| hard (inside the build) | the token set is the pinned `navtest_full_tokens.json` (md5 `3a8b3653…`, 12,146 / 136) · every table has every token · PDMS is on the **x100** scale (a table whose PDMS never exceeds 1 is refused: UNIT SLIP) · no pair has B ≡ A · the two pairs are not identical · `stage3_full.py` / `proxy_eval.py` md5 are the pinned ones · an explicit I-rep is given (no default) · the emitted file passes the strict §7 validator (exact key sets, `n_tokens` 12146 / `n_logs` 136, `pooled.resample == "joint"`, pooled mean = mean of the pair means, CI order, no `seed_floor_abs`) | refusal, nothing written |
| **soft** | IC3: the BASE picks reproduce the banked u400 seam (≥ 97 % within 5 mm; every difference a near-tie ≤ 0.01 in `sel_score`; ≤ 12 unexplained) · IC5-forward: the repeated forward is bit-identical | status `IDENTITY-FAILED`: the file **is** written and the sidecar says so (the free-point precedent: rows computed, stamped, the Master Mind decides). The reader does not see the stamp: quote it with the verdict |

Not repeated from the free-point control set, and why: IC1 (inputs bit-equal to the A9 recorded inputs) is inherited with the copied input files; IC4 (BASE through the harness vs the banked CSV) would cost another 0.37 h and BASE is not an arm of this read (the BASE seam is written if you want it); IC6 (C3 table lookup) and the regression arms D1/D3 test the instrument, which this read does not change; **D2 (A/B swap) is replaced by the stronger hard gate above** (the binding is checked against hashes computed independently from the HF copy).

## 5. Reading rule

`navtest_E.json` goes to `safeobj_outcome.py --navtest-e` unchanged. The reader applies bar E (OUTCOME_BUILD §2): each pair ≥ +0.25 and mean ≥ +0.5; pooled (JOINT) lower bound > 0; mean contrast > 2 × max(RL-val seed floor from bar L, navtest `|R0d−R1d|`, I-rep); TTC and NC upper bounds ≥ 0 per pair (four bounds); all sub-scores reported. ADOPT / REFUTED (pooled upper < +0.25) / NOT_PROVEN; NOT_READ when clause 3 cannot be read.
The estimator carries the **LOG-draw** question only (H-ESTIM-SEED-1): neither the training-seed nor the inference-seed variance is in these intervals; the floors are how the reader accounts for them.
`navtest_families.json` is reported beside it: LONGITUDINAL (EP, TTC, C + families6 speed/along/accel components), LATERAL (DAC, DDC + heading/yaw-rate/curvature/cross-track), TACTICAL (decision accuracy and kappa per arm; the lateral-decision kappa rule of PREREG_RL_S1 §4), **STRATEGIC = PAUSED** (PI 2026-09-19, item 25) — never a pass, never omitted. The families6 components keep families6's own estimator (per arm, unpaired, 2,000 resamples) and are **not gating**.

## 6. Failure handling

* Pod disk < 6 GB free: the shipper holds (`holding` in `ship.log`); free nothing that is not under `/workspace/ops/sobjE`.
* `ZZFWD_IC2_FAILED`: INVALID, stop (the replicated scorer path differs from the model's own).
* A killed worker: rerun step 4 (finished chunks are skipped by their `.done`). `ssh` drops: rerun step 5 (it retries the chunk).
* `REFUSED: … hard gate failed: <names>`: fix the named input; do not edit a gate file by hand.
* Cleanup: delete `/workspace/ops/sobjE` on the pod only after the pulled chunks are verified on the dev box and banked.

## 7. Open points (each blocks or changes the read; none is decided here)

1. **I-rep has no producer for E.** PREREG §7 names "the I-rep of PREREG_RL_S1 §3 (eval batch 8 vs 32, shuffled order, non-deterministic kernels) on R0d-B u1000, dev box". Nothing in the repo computes it (**M**: a search of `*.py` under `C:/fp/PK` for `irep|I-rep|I_rep` returns no file, with a same-breath control on the safety package that returns four; only the reader and this tool mention it). The navtest forward is fp32, batch 1, deterministic on one card (free-point IC5), so a navtest-level I-rep may be exactly 0.0; the RL-val one is not. `--irep-json` is therefore **required** (no default: a defaulted 0 would silently weaken E clause 3). *Master Mind: pick the definition and who runs it.*
2. **The banked R0 arm directory must carry `rlval_vs_A0.json` in the `arms.CTRL` / `arms.R` layout** (the file bar L reads). The banked R0 read in the archive is `rlval_vs_C0.json` (`arms.C` / `arms.R`). This is now the READER's concern only (bar L, hence E clause 3): this tool no longer reads it.
3. **The seed floor is the reader's, not this file's.** Since the 14:15 revision of §7 the file carries only `floors.irep_abs_delta`; the reader computes `max(RL-val |R0d−R1d|, RL-val |R0−R1o|, navtest |R0d−R1d|, I-rep)` itself. The navtest term is written to the sidecar (`floors_components.navtest_seed_floor_abs_R0d_vs_R1d`) and a test pins it to the reader's own number. If R1o was killed or never ran, the reader reports clause 3 unread (A-E3b); nothing here can substitute for it.
4. **The frames are re-uploaded (2.1–2.3 h).** Keeping them on the pod needs the free-point worker changed (`shutil.rmtree` of a processed chunk) and ≈ 9.7 GB of pod disk (13 GB free on 2026-10-17, falling ~1 GB/h under the bank build). The free-point run is live: not touched.
5. **The `started_epoch` gate (G2)** belongs to the free-point run's P6 constraint. Whether it applies to this read is the Master Mind's call.
6. **families6 is unpaired.** A paired B − A interval for its components needs per-window values from families6 (not emitted today).
7. **`fp_score.py` hard-codes `SP` = another session's scratchpad** (`…/a085b71c-…/scratchpad`, where `score_e.py` lives; present on 2026-10-17). If that directory is cleaned the harness step fails; copy `score_e.py` and edit `SP` in the COPY.
8. **Dev-box state copies:** the layout `rls1-r0d/scorer_states/scorer_states_u1000.pt` follows the c0 / r0 precedent and `hf_egress.py` (`scorer_states_u{upd}.pt`); the HF repo names (`Sayood/tanitad-refe-rls1-r0d`, `-r1d`) follow `rls1_pod.sh`. **U**: not listed on HF from here.
