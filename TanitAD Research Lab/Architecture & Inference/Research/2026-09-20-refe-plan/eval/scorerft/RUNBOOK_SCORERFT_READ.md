# RUNBOOK: SCORER-FT T-vs-C read (`scorerft_read.py`), from the two seams to the verdict

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
* **CTRL-DET:** re-run T's seam on >= 200 tokens into `$O/det_t.npz` (the `seam1` command of `p7_read_chain.py` with a 200-token `--tokens` list, name `scorerft_t_det`; if the chain does not accept a subset,
  re-run the full seam as P5-C did and the read compares on the token intersection). Poses must be bit-identical. ⚠️ UNVERIFIED: no ready 200-token re-run tool exists in the repo; the control function itself is tested (`dry_ctrl_det_p7_full.json`).
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
