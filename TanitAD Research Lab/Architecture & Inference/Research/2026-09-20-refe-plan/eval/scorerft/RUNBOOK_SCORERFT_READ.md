# RUNBOOK: SCORER-FT reads (`scorerft_read.py`)

Part B below reads arm P against the reused C (card PREREG_SCORERFT_PICK.md). `--card` defaults to `scorer_ft`, so Part A's commands are unchanged.

# Part A: T-vs-C (card PREREG_SCORER_FT.md, Stage 1 DONE 2026-11-24, O3), from the two seams to the verdict

Card: `PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md` (frozen; sha256 `adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91`, verify before reading).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Everything here is CPU, dev box, no GPU. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (`PY_T`);
NAVSIM work (`transp_score`, `families_run`, `p5c_prop_score`) in `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe` (`PY_N`). PYTHONPATH separator is `;`. `CUDA_VISIBLE_DEVICES=-1`, at most 2 CPU processes,
`commit_guard.py --min 6` must print >= 6 before each heavy step. The PK path contains `&`: never pass it through `sed` replacement text or `${a//x/$y}`. A pytest "no tests ran" is a FAIL.

## 0. What the read needs, per arm (C and T = Stage 1; Cs and Ts = Stage 2, only if P1 and G1 hold)
| input | made by | where |
|---|---|---|
| seam + proposals dump | `sft_seam_chain.sh` (automatic) | `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_scorerft_<c\|t>.npz` and `.proposals.npz` |
| harness CSV | same | `.../data/refe_navtest/score/refe_stage2_scorerft_<c\|t>/refe_stage2_scorerft_<c\|t>.csv` |
| 64-proposal table | same | `E:/Projects/TanitAD-artifacts/scorerft_read/propscore/full_SFT<C\|T>_s{0,1}of2.jsonl` |
| transp_score rows (arms `T`, `C`) | step 2 below | `$O/pass1/rows_s*of2.jsonl` |
| pod argv files (JSON list or text) | the pod, per arm | `$O/argv_{c,t,cs,ts}.json` (copy from the arm's `--out` dir; they are NOT on the dev box today) |
| C's step-0 digest, C1's sha256 | the pod's C log / `SCORERFT_BASE_SHA256` | `$O/w0_c.json`, `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` |
| T's seam re-run, >= 200 tokens | step 6 | `$O/det_t.npz` |
Set `O=E:/Projects/TanitAD-artifacts/scorerft_read/read_out` (new dir; keep it off the live chains' dirs) and `TIP=<an export of the SCORER-FT land commit or later>`;
`export PYTHONPATH="$TIP/stack;$TIP/taniteval;$TIP/<PK>/eval;$TIP/<PK>/eval/p5c;$TIP/<PK>/eval/field;$TIP/<PK>/eval/scorerft"`.

## 1. Preflight (all must hold, else stop)
* `sha256sum PREREG_SCORER_FT.md` equals the card hash above.
* The seams exist for C and T (`stage2_scorerft_c.report.json` and `_t` read rows 12146, misses 0), and `chain.DONE` markers exist in `scorerft_read/markers_sft{c,t}/`.
* `pytest -q test_scorerft_read.py` (the WHOLE file, while holding `mkdir .../scorer_ft/pytest.lockdir`) is green on the tip you read from.

## 2. Plans and the ONE transp_score pass (the P5-C chain's steps 5a-5b, arm names `T` and `C`, plus `base` = C1's own seam for the tactical call)
```
PY_T $EV/p5c/p5c_read_prep.py convert --arm "T=<T seam>,<T dump>" "C=<C seam>,<C dump>" "base=<C1 seam>,<C1 dump>" --outdir $O/plans
PY_T $EV/p5c/p5c_read_prep.py score   --plans-dir $O/plans --out-dir $O/pass1 --procs 2 --python PY_N
```
(`$EV` = `$TIP/<PK>/eval`.) Stage 2: add `Cs=` and `Ts=` arms to the same two commands (one pass, all arms) and read them from the same rows glob.

## 3. Families (paired T - C on the executed-tracked frame; Ts - Cs the same way)
```
PY_T $EV/p5c/p5c_families.py --rows "$O/pass1/rows_s*of2.jsonl" --arms T C base --pairs T:C --out-dir $O/families --python PY_N --repo $TIP
```
gives `famci_T_vs_C.json` (`label_a` C, `label_b` T: the read's default labels). It carries headway / time-gap / TTC to the lead where a lead exists, with n, and the reason where none does (`families_run.py`, `fam_ci.py`).
Stage 2: `--arms Cs Ts --pairs Ts:Cs` into `$O/families_s` and pass `--fam-ci-s` (`--fam-labels` is the T/C pair only).
```
PY_T $EV/p5c/p5c_tactical.py --seam C1=<T seam> C0=<C seam> base=<C1 seam> --dump C1=<T dump> C0=<C dump> base=<C1 dump> --out $O/tactical.json
PY_T $EV/instruments/strategic_family.py read --arm T=<T seam> --base C=<C seam> --out $O/strategic_family_T_C.json
```
`tactical.json` keys its arms C1 / C0 / base (hard-coded); the read relabels C1 -> T, C0 -> C. Its STRATEGIC stub now reads REPORTED_SEPARATELY (it never prints PAUSED). STRATEGIC is
reported IN FULL from `strategic_family_T_C.json` (PI unpaused 2026-11-16); the read refuses a strategic JSON that contains the word PAUSED.

## 4. Controls that need artifacts the dev box does not hold yet
* **CTRL-ARGV:** copy each arm's launched argv to `$O/argv_<arm>.json` (a JSON list, `{"argv": [...]}`, or whitespace text). The rule is the card's section 3 exactly: argv(T) - argv(C) = {`--rank-w`, `--rank-heads ep,gate`,
  `--rank-ep-gap 0.02`, `--rank-gate-topk 8`, `--out`}; C names `--rank-w 0.0`; Xs - X = {`--seed 1`, `--out`}; every arm has one `--init-from` and `--steps 1000`. One extra flag anywhere FAILS it.
* **CTRL-W0:** `--w0-digest-c` (the file with C's step-0 digest; the first 64-hex string in it is read, card question Q-S2) and `--w0-ref` (C1's sha256). Not equal = FAIL.
* **CTRL-DET (wired 2026-11-21):** T's seam chain (`raw/2026-11-20-scorer-ft/devbox/sft_seam_chain.sh` v2, arms SFTT / SFTTs) runs `p7_read_chain.py --only rep` after the proptable:
  the FULL seam again (12,146 tokens, no dump, the same batch layout as `seam1`), and that step raises unless the tokens are equal and every pose is bit-identical. Pass its output
  `$DATA/seams/stage2_<T name>_rep.npz` as `--det-seam-t` (compared with `--seam-t`). A 200-token subset is NOT used: it changes the batch composition, so a bit-identity failure there
  could come from batching rather than from non-determinism. Cost: one more seam on the dev box (~5 h; C's `s2` took 21,175 s). The control function is tested (`dry_ctrl_det_p7_full.json`).
A real read REFUSES (exit 2, names each missing flag) without any of these; it never skips a control.

## 5. The read (Stage 1: C and T only)
```
PY_T $EV/scorerft/scorerft_read.py --rows "$O/pass1/rows_s*of2.jsonl" --t T --c C \
  --seam-t <T seam> --dump-t <T dump> --csv-t <T csv> --proptable-t "<propscore>/full_SFTT_s*of2.jsonl" --argv-t $O/argv_t.json \
  --seam-c <C seam> --dump-c <C dump> --csv-c <C csv> --proptable-c "<propscore>/full_SFTC_s*of2.jsonl" --argv-c $O/argv_c.json \
  --w0-digest-c $O/w0_c.json --w0-ref f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 --det-seam-t $O/det_t.npz \
  --seam-ledger $O/seam_ledger.json --convert-report $O/plans/convert_report.json \
  --fam-ci $O/families/famci_T_vs_C.json --tactical $O/tactical.json --strategic $O/strategic_family_T_C.json \
  --out $O/read_stage1.json --text $O/report_stage1.txt --verdict-line $O/VERDICT_stage1.txt
```
Defaults: 10,000 draws, seed 20260927, 95 % percentile (do not pass `--n-boot` in a real read). Exit 0 prints one `ZZSFT_READ` line. The read's own controls: CTRL-SET (12,146 tokens, 136 logs, identical across rows / seams / dumps),
CTRL-T (table pick == harness CSV to 1e-9, every token, every arm), CTRL-CSV, CTRL-SEAM (sha256 recorded on first sight and re-checked), CTRL-ID (the arm read against itself: every Delta exactly 0.0), plus the three above.
**Any FAIL = INVALID, no outcome; the Deltas are descriptive only.**

## 6. Reading the output (card sections 5-9)
* Stage 1 with **P1 and G1** prints `STAGE-2-TRIGGERED`, P2 and P3 `PENDING (Stage 2)`, never ADOPT. Run Cs then Ts (pod, ~28.3 L40-h; PI go 2026-11-20 is conditional on exactly this), repeat steps 2-3 with those arms, then the read with
  `--ts Ts --cs Cs` plus `--seam/--dump/--csv/--proptable/--argv-ts/-cs` and `--fam-ci-s`: it then prints O1 or O2.
* Otherwise the outcome is O5, O6, O3 or O4, evaluated in the card's order O1, O2, O5, O6, O3, O4. The text names the committed next arm. O3 / O4 / O5 / O6 on one seed say **training variance was not measured**.
* BELOW GOAL = tracked PDMS point < 94.55 (printed per arm with its CI). DDC is reported, not a guard. Transparent comfort is reported in both forms (20-pose plan and 8 submitted poses), descriptively, in no clause.
* Separated adverse cells in any family are listed as qualifiers in the verdict line; only G1 blocks ADOPT. Ledger: Stage 1 = look 1 of `H-SCORERFT-1`, Stage 2 = look 2; multiplicity across cards is not corrected.
* ADOPT does not clear C1's own GUARD-FAIL (TM+C). Update `Project Steering/GOALS_AND_CLAIMS.md` (`H-SCORERFT-1`) in the same turn as the verdict.

## 7. Dry runs already done (stand-ins; every output stamped `DRY RUN — NOT A SCORER-FT VERDICT`)
`.../scorerft_read/read/dry_runs/`: `dry_c1_vs_c0.*` (C1 as T, C0 as C; reproduces the P5-C read's d = +1.1702 [0.6730, 1.6450] and C1's regret 6.80, EP Spearman 0.413, AUCs 0.929 / 0.962 / 0.900),
`dry_ctrlid_c1_vs_c1.*` (CTRL-ID: every Delta exactly 0.0, no P1), `dry_stage2_exercise.*` (P2 / P3 plumbing only; Ts = C1, Cs = base), `dry_ctrl_det_p7_full.json` (CTRL-DET function on the P7 seam and its banked re-run).
Not exercisable on stand-ins and listed per run under `controls_not_available`: CTRL-ARGV and CTRL-W0 (no pod argv, no step-0 digest) and CTRL-DET for C1 (no re-run on 200 tokens): their code paths are covered by `test_scorerft_read.py` only.


---

# Part B: P-vs-C read (card `PREREG_SCORERFT_PICK.md`, sha256 `<PICK_PREREG_SHA256>`, amended 2026-11-24 for the gradient clip, `b47b2cd0…` at the time of writing)

`--card` defaults to `scorer_ft`, so Part A's commands are unchanged. Under `--card pick` the roles T / Ts are PRINTED P / Ps (never `T_minus_C`). Every pin below that reads `None` in `scorerft_read.py` makes a real read REFUSE
(`PICK_PREREG_SHA256`, `PICK_TRAIN_SHA256`, `CLIP_PROBE_SHA256`; `PICK_CODE_TAG` is optional); a dry run is allowed and stamped.

* **Directories.** `OP=E:/Projects/TanitAD-artifacts/scorerft_read/read_p` (new). `O1=E:/Projects/TanitAD-artifacts/scorerft_read/read_out` is Stage 1's directory and is READ-ONLY for Part B.

## B0. Inputs
| input | made by / from | pin |
|---|---|---|
| P seam, dump, harness CSV, table `propscore/full_SFTP_s*of2.jsonl`, CTRL-DET re-run seam `stage2_scorerft_p_rep.npz` | `sft_seam_chain.sh SFTP scorerft/p scorerft_p` (whitelist needs `SFTP|SFTPs`) | CTRL-SEAM ledger (new, `$OP/seam_ledger.json`) |
| P `config.json`, `summary.json`, `metrics.jsonl`, `weight_rule_rank_p.json`, `w0_p.json` | pulled from the pod, framed (GoTTY trap) into `$OP/pod/` | CTRL-CODE / CTRL-S0 / CTRL-WRULE / CTRL-W0 |
| C seam, dump, CSV, table, `argv_c.json`, `w0_c.json`, `pod_config_c.json` | REUSED from Stage 1 | `C_SEAM_SHA256` ... `C_W0_SHA256`, `C_POD_CONFIG_SHA256` |
| C `metrics.jsonl` | `o3/card/evidence/metrics_scorerft_c.jsonl` | `C_METRICS_SHA256` |
| Stage 1 `read_stage1.json`, `seam_ledger.json` | `$O1/` | `STAGE1_READ_SHA256`, `STAGE1_LEDGER_SHA256` |
| **CTRL-CLIP probe JSON** | `refe/clip_probe.py` (B0b), banked with its sha256 | `CLIP_PROBE_SHA256` |

## B0b. CTRL-CLIP probe (pre-launch, on the L40 pod with C's image, after P's launch-gate PASS token and before P's launch; card 3.2)
```
python PK/refe/clip_probe.py --argv-c <argv_c.json> --argv-p <P's PASS-token argv> --c-metrics <metrics_scorerft_c.jsonl> --scratch <scratch dir> --out $OP/pod/clip_probe.json
```
Runs the real `train.run` of each form (only `--out` replaced) under a pass-through wrapper on `clip_grad_norm_`, 8 steps (K = N_MICRO), stops before the optimiser step of step 8, writes no checkpoint, deletes its scratch.
Exit 0 valid; 5 = invalid (PL3: re-run; a V2 / V3 failure means C is not reusable, so C'); 4 = the probe did not run. State CLIP-0 / CLIP-SAME / CLIP-T (PL4: launch, tagged). The probe's scope is steps 0-7 only.
Bank the JSON (HF + sha256) and fill `CLIP_PROBE_SHA256`.

## B1. Preflight (all must hold)
1. The card sha equals `PICK_PREREG_SHA256`. 2. `sha256sum` of every C and Stage 1 input equals its pin. 3. The WHOLE test files pass holding the pytest lock dir: `test_scorerft_read.py` (67), `test_clip_probe.py`.
3b. Positive control on the real logs: `ctrl_s0(metrics_scorerft_c.jsonl, metrics_scorerft_t.jsonl, rank_w=0.03125)` and `ctrl_order(...)` both PASS (19 + 10 keys, 0 of 1,000 steps differ). Otherwise STOP.
4. Stage 1 reproduction: run Part A section 5's exact command with the modified tool into `$OP/repro/`; `verdict`, `T_minus_C`, `levels`, `goal`, `mechanism`, `families`, `controls[*].pass` equal Stage 1's. Otherwise STOP.
5. `CHAIN.log` holds `SFTP chain start tip=751f3ae7`. 6. Both seam reports read 12,146 rows and 0 misses.

## B2-B3. One transp_score pass with P, C and base; families, tactical, strategic
As Part A sections 2-3 with arms `P`, `C`, `base` and `--pairs P:C`; `famci_P_vs_C.json` must carry `label_a C`, `label_b P`, or the read prints NOT-EVALUATED. C is re-scored from its sha-pinned Stage 1 seam; CTRL-CREUSE proves it IS Stage 1's C.

## B4. The read
```
PY_T $EV/scorerft/scorerft_read.py --card pick --rows "$OP/pass1/rows_s*of2.jsonl" --p P --c C \
  --seam-p <P seam> --dump-p <P dump> --csv-p <P csv> --proptable-p "<propscore>/full_SFTP_s*of2.jsonl" --argv-p $OP/pod/argv_p.json \
  --seam-c <C seam> --dump-c <C dump> --csv-c <C csv> --proptable-c "<propscore>/full_SFTC_s*of2.jsonl" --argv-c $O1/argv_c.json \
  --w0-digest-c $O1/w0_c.json --w0-digest-p $OP/pod/w0_p.json --w0-ref f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 \
  --det-seam-p <P rep seam> --seam-ledger $OP/seam_ledger.json --convert-report $OP/plans/convert_report.json \
  --metrics-c <o3/card/evidence/metrics_scorerft_c.jsonl> --metrics-p $OP/pod/metrics_p.jsonl --pod-config-c <PK>/raw/2026-11-24-scorerft-read/pod_config_c.json --pod-config-p $OP/pod/config_p.json \
  --seam-report-c <C report.json> --seam-report-p <P report.json> --seam-chain-log E:/Projects/TanitAD-artifacts/scorerft_read/CHAIN.log \
  --weight-rule-p $OP/pod/weight_rule_rank_p.json --clip-probe $OP/pod/clip_probe.json --stage1-read $O1/read_stage1.json --stage1-ledger $O1/seam_ledger.json \
  --fam-ci $OP/families/famci_P_vs_C.json --tactical $OP/tactical.json --strategic $OP/strategic_family_P_C.json \
  --out $OP/read_p_stage1.json --text $OP/report_p_stage1.txt --verdict-line $OP/VERDICT_p_stage1.txt
```
`$OP/seam_ledger.json` is a NEW ledger (recorded at first sight, re-checked on a re-run); Stage 1's ledger enters only through `--stage1-ledger`.

## B5. Reading the output
Outcome order OP1 ... OP7 after the pre-launch PL1-PL4 and INVALID / INCOMPLETE; the text names the committed next arm. M1 is reported, not decisive; M5 is descriptive. The verdict line prints the CTRL-CLIP state
(`clip: CLIP-0` / `CLIP-SAME occ=a/8` / `CLIP-T on=.. off=.. ...`); under CLIP-T the OP4-OP7 notes name the clip as an unexcluded co-mediator (PL4 (c)). Every separated-adverse row is listed. STRATEGIC in full. Update
`GOALS_AND_CLAIMS.md` (`H-SCORERFT-PICK-1`) in the same turn as the verdict.

## B6. Stage 2 (only on STAGE-2-TRIGGERED)
Run Cs, then Ps (each with its own CTRL-CLIP), repeat B2-B3 with `Cs=`, `Ps=` and `--pairs Ps:Cs`, then the read with `--ps Ps --cs Cs`, `--pod-config-cs/-ps`, `--metrics-cs/-ps` and `--fam-ci-s`.

## B7. Dry runs
`dry_pick_*`: the pick card on stand-ins (the synthetic arms of `test_scorerft_read.py`), every output stamped `DRY RUN`.
