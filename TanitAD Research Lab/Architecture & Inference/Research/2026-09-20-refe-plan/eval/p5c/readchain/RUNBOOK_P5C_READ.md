# RUNBOOK: the P5-C read (C1 vs C0), one detached dev-box chain
Agent: REFe read-chain agent, 2026-11-17. Code pinned at mirror sha `751f3ae703097507b37548057393ae9a0d76fd78` (branch `agent/arch-inf-20260803`). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Evidence class of everything below: commands are quoted from the tools' own docstrings and argparse (INHERITED from the landed code, then exercised by the dry run in section 7); timings are ESTIMATED from the P7 read (`E:/Projects/TanitAD-artifacts/p7_full_read/CHAIN.log`) unless marked MEASURED.

## 0. What the chain is, and what it refuses
`p5c_read_chain.sh` runs serially, one marker per stage (`<work>/markers/<stage>.DONE`, failure `chain.FAILED`, retry = re-run, finished stages are skipped). Pre-registered by `PREREG_P5C.md` sections 5, 6, 8, 11 (+ A1-A3.3). It never trains, commits, lands, pushes; never touches the pod or box 2.
It refuses to start (exit 2/4) unless: both arms' `model_final.pt` and `model_final_sha256_<ARM>.txt` exist, the computed sha256 equals the pod's sha file for BOTH arms, and C1 and C0 differ. It seals both in ONE `SEALED.json` before any navtest pass. No per-arm number is printed by the chain itself before the read stage (stage-2 logs contain per-arm harness lines; do not open them before `chain.DONE`, card section 11 rule 3).

## 1. Prerequisites
| item | value |
|---|---|
| C0 | pod chain pushes `Sayood/tanitad-refe-p5c`, prefix `p5c/c0` (`model_final.pt`, `model_final_sha256_C0.txt`); ~2026-11-17 23:30Z |
| C1 | same, prefix `p5c/c1`, ~29 h after C0. HF file layout is UNVERIFIED until C0's push lands: `hf_pull.py` resolves `<prefix>/**/model_final.pt` and `model_final_sha256_<ARM>.txt`, refuses on 0 or more than 1 match and prints the repo listing |
| tip export | `git -C C:/Users/Admin/tanitad-push archive <PINNED SHA> -- "<PK>/eval" "<PK>/refe" "<PK>/raw/2026-11-12-instruments" "<PK>/raw/2026-11-16-p7-read/strategic" "<PK>/raw/2026-11-11-p5c-readiness/PREREG_P5C.md" stack taniteval ":(exclude)<PK>/eval/raw" | tar -x -C F:/TanitAD_p5cread_tip` (~4,000 files; takes ~1 h on F:, so do it BEFORE C1 lands; re-export if the landed wiring commit changes `eval/` or `stack/`). Set `P5C_TIP` to it |
| base | P7: `F:/TanitAD_p7read/run/model_final.pt` (sha256 `2613fc17b9e5...`), banked read `data/refe_navtest/seams/stage2_p7_full.{npz,proposals.npz}`, csv `score/refe_stage2_p7_full/`. Under B-P6 (PREREG 2.1) set `P5C_BASE_CKPT/SEAM/DUMP=NONE/CSV` to P6's |
| yaw table | `E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz` (sha prefix `f5ebe24f`) |
| dev box | free commit >= 6 GB (integer, `tools/commit_guard.py`; a stage waits up to 30 min then fails), F: >= 20 GB (real run), E: >= 3 GB; nobody else on the GPU lock |

## 2. The one command (real run)
```powershell
$env:P5C_TIP = 'F:/TanitAD_p5cread_tip'      # optional overrides: P5C_WORK, P5C_MODELS, P5C_BASE_*  (see the header of p5c_read_chain.sh)
& 'E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_readchain\launch_p5c_read.ps1'
```
It writes the `P5C_*` variables to `<work>\launch_env.cmd` (WMI children do not inherit your environment), then starts `run_p5c_read.cmd` through WMI `Win32_Process.Create` with `CreateFlags 0x08000200`. Heartbeat: `<work>/CHAIN.log` (default work `E:/Projects/TanitAD-artifacts/p5c_read`). Result: `<work>/out/VERDICT.txt`, `out/read.json`, `out/report.md`.
Resume after a failure: fix the cause, run the same command (`chain.FAILED` is kept as `chain.FAILED.attempt_N`).
**First real run, 2026-11-19 (MM), MEASURED:** (1) on the dev box `& '...launch_p5c_read.ps1'` is REFUSED by the execution policy (PSSecurityException); run the launcher's body inline in a PowerShell session instead (set the `P5C_*` variables, write `launch_env.cmd`, WMI create); never `-ExecutionPolicy Bypass`. (2) `hf_pull.py` saved the files under the repo sub-path (`arms/C1/p5c/c1/...`) and `arms_verified` refused "missing model"; fixed in `tools/hf_pull.py` (moves both files up). Resume = delete the two `hf_pull_*.DONE` markers of the failed attempt and relaunch. (3) The pod's post scripts never pushed `model_final_sha256_<ARM>.txt`; the MM pushed both with `hf_egress.py push` (tags `arm_p5c_c0_shafile`, `arm_p5c_c1_shafile`). A C0r read needs the same push. (4) `P5C_F_FLOOR_GB=8` was used (F: 12-19 GB free; the read writes ~nothing to F:).
Stop: kill the python/bash PIDs of the chain by explicit PID; the exit trap releases the GPU lock only if it names a `refe-stage2-p5c_*` job.

## 3. Stage table (order is the chain's order)
`G` = dev-box GPU (shared lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`, taken INSIDE `stage2_sft.py`, released in its `finally`, plus the chain's exit trap). All other stages: `CUDA_VISIBLE_DEVICES=-1`.
| stage | does | inputs -> outputs | proc | ESTIMATED wall |
|---|---|---|---|---|
| preflight (always) | free commit/disk guards; tip modules (`tanitad`, `taniteval`, `p5c_*`) imported from the tip (asserted) | -> `logs/preflight.log` | CPU | < 1 min |
| hf_pull_C1, hf_pull_C0 | `tools/hf_pull.py`: list repo, download model + sha file, sha256 vs pod file (HF token read in place from Keys.txt, xet off) | HF -> `arms/<ARM>/` | CPU/net | ~5 min each (1.30 GB; rate UNVERIFIED) |
| arms_verified | recompute sha256 of both models == pod sha files; C1 != C0; else exit 4 | -> marker | CPU | 1 min |
| seal | `p5c_seal.py seal --arm C1=... --arm C0=... --card PREREG_P5C.md --base-sha <P7 sha>` (hashes computed here, `hash_source` computed) | -> `out/SEALED.json` | CPU | 1 min |
| seal_verify | `p5c_seal.py verify --sealed --card --recheck-files` | -> `out/sealed_verify.json` (kind `B11_sealed`) | CPU | 1 min |
| s2_c1, s2_c0 | `eval/p7_read_chain.py --ckpt <arm> --name p5c_<arm> --expect-ckpt-sha256 <sha> --ego-yaw-table <yaw> --only seam1,score` = preflight (pins, yaw, tokens md5 `3a8b3653`, harness pins), then `stage2_sft.py --dump-proposals --ego-yaw-table` (the seam, rc 1 at the 36-token frame control is EXPECTED and validated by the chain), then the same command again (harness score) | -> `data/refe_navtest/seams/stage2_p5c_<arm>.{npz,proposals.npz,report.json}`, `score/refe_stage2_p5c_<arm>/*.csv`, `raw/2026-10-05-sft-stage2/p5c_<arm>/` | seam **G**, score CPU | seam 4.4 h (P7: 15,990 s) + score 0.6 h (2,041 s) per arm = ~10 h for both |
| convert | `p5c_read_prep.py convert --arm C1=seam,dump C0=seam,dump base=seam,dump` (registered pick20 form; pick20-equals-seam bit-exact control inside) | -> `plans/plans_{C1,C0,base}.npz`, `convert_report.json` | CPU | ~2 min |
| score_transp | `p5c_read_prep.py score` one `transp_score` pass over {C1, C0, base, human}, 2 shards, harness metric cache | -> `out/pass1/rows_s*of2.jsonl` | CPU (navsim venv) | ~35 min (P7 families: 23 min for 2 arms + human) |
| identity_C1/C0/base | `p5c_read_prep.py identity` tracked PDMS == pinned harness csv, bar 1e-6 (B6) | -> `out/identity_<arm>.json` | CPU | < 1 min each |
| identity_red | `p5c_read_prep.py red --n 60`: cache-route line deleted must FAIL identity (exit 0 iff detected) | -> `out/red/red_result.json` | CPU | ~5 min |
| det_select, det_plans, det_pass2, det_compare | B10: seeded 10 % subset (1,215 tokens, seed 20261112), second pass through the same scorer, bit-identical per-token under both scorers | -> `out/determinism.json` (kind `B10_determinism`) | CPU | ~5 min |
| families | `p5c_families.py --arms C1 C0 base --pairs C1:C0 C0:base C1:base` (families_run per arm + fam_ci, both frames) | -> `out/families/famci_C1_vs_C0.json`, `families_summary.json` | CPU | ~5 min (P7: 65 s + 53 s + 31 s) |
| tactical | `p5c_tactical.py --seam ... --dump ...` command adherence + random-proposal floor; selection quality is NOT in the chain (needs 64 single-proposal harness seams, ~4-6.4 h CPU per arm; the block says NOT COMPUTED with the reason) | -> `out/tactical.json` | CPU | ~1 min (P7: 13 s) |
| strategic | `strategic_family.py read --arm C1=<C1 seam> --base C0=<C0 seam>` REPORTED beside the read, no bar (its own commit guard) | -> `out/strategic_family_C1_C0.json` | CPU | MEASURED in section 7 |
| read | `p5c_read.py --rows ... --c1 C1 --c0 C0 --base base --human human --fam-ci ... --controls identity x3 + determinism --tactical --sealed --card --recheck-files --require-kinds B10_determinism B11_sealed` | -> `out/read.json` | CPU | ~2 min (10,000 draws) |
| render | `p5c_render.py` (printed twice, banned wording, one qualifier) | -> `out/report.md` | CPU | seconds |
| verdict | `tools/verdict_line.py`: the one-line verdict + C0r trigger | -> `out/VERDICT.txt`, `markers/chain.DONE` | CPU | seconds |

**Total ESTIMATED: ~11 h wall, of which ~8.8 h dev-box GPU (two 4.4 h seams); the rest is CPU.** Alternatively the seam of the first arm can start the moment its files are on HF only if BOTH arms are sealed first; the chain therefore waits for C1 (card section 11).

## 4. The verdict line and C0r
`out/VERDICT.txt`: `P5C-READ verdict=<V> qualifier=<TM|TM+C|TI|TU> n=12146 logs=136 d_tracked=<pt> [lo, hi] B=0.7031 d_transparent=... P1a P1b G1 G2 CTRL1 C0r_TRIGGERED=<YES|NO> | strategic: ...`.
Verdict names come from `p5c_read.verdict_table`: INVALID, REFUTED, NULL, SWAMPED, GUARD-FAIL, INCOMPLETE, SEPARATED-SINGLE, LEVER, SWAMPED-BY-REPLICATE. **C0r is triggered iff the verdict is SEPARATED-SINGLE** (PREREG 6.3: P1-a CI-lo(d) > 0, P1-b d > B = max(2.776 x 0.1791 x sqrt 2, 2 x 0.33) = 0.7031, G1, G2 and CTRL-1 all pass) **and the PI approves** (+28-29 L40-h ESTIMATED, PREREG 17.3); `verdict_line.py` exits 2 if the clauses and the verdict disagree. A REHEARSAL read never triggers. P1-c, LEVER and SWAMPED-BY-REPLICATE need C0r (`p5c_read.py --c0r C0r` with a C0r seam through the same chain, sealed with `--arm C0r=`).
K ledger (PREREG 11): `p5c_seal.py entries --sealed SEALED.json --ledger ledger_K.json --out entries.json` and `merge` are the Master Mind's append at read time, not in the chain (the tool never writes the real ledger).

## 5. Open questions (the tools' own, none decided here)
1. HF layout under `p5c/c0`, `p5c/c1` (stage hf_pull refuses and lists if it differs). 2. Base: P7 (default) or P6 (PREREG 2.1); under P6 the base has no dump, so its transparent arm is the 8-pose form: a MIXED form the converter warns about. 3. Selection quality (64 harness seams per arm) is not run. 4. Whether stage-2 logs may be opened before `chain.DONE` (card 11 rule 3): treat as no.

## 6. Failure map
| symptom | meaning |
|---|---|
| exit 4, `arms_verified` | a sha differs from the pod's sha file, a file is missing, or C1 == C0: nothing was sealed or run |
| `p7_read_chain` REFUSING (pins, yaw, tokens) | a stage-2 preflight check named in `logs/s2_*.log`; nothing was started |
| `identity_*` FAIL | the transp_score route does not reproduce the harness: the read must not run; `read` would be INVALID anyway |
| `det_compare` FAIL | inference is not deterministic (card: a defect, investigate before any read) |
| `INVALID` / `INCOMPLETE` in the verdict | a control failed or a required kind is missing: no verdict, C1 - C0 is descriptive |

## 7. Dry run
See `DRYRUN_REPORT.md` in this folder (written after the run). It is NOT a P5-C result.
