# Field read (F vs C): how to run `field_read.py` the moment both seams exist

Card: `eval/PREREG_AUX_FIELD_LOSS_P7.md` (FINAL, Amendments A, B). Instrument: `eval/field/field_read.py` (tests: `test_field_read.py`, run the whole file).
Prerequisite per arm (made by `field_seam_chain.sh`): seam `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7field_<arm>.npz` + `.proposals.npz` + `.report.json`,
harness CSV `.../score/refe_stage2_p7field_<arm>/refe_stage2_p7field_<arm>.csv`, 64-proposal table `E:/Projects/TanitAD-artifacts/field_read/propscore/full_<ARM>_s{0,1}of2.jsonl`.

1. **Rows (tracked + transparent), ONE pass over both arms** (the chain does not make these):
   `p5c_read_prep.py convert --arm F=<F seam>,<F dump> C=<C seam>,<C dump> --outdir PLANS` then `p5c_read_prep.py score --plans-dir PLANS --out-dir OUT/pass1 --procs 2 --python <navsim-crun python>`
   (~15 min for two arms; `convert_report.json` records each seam's sha256: pass it as `--convert-report`).
2. **Families:** `p5c_families.py --rows "OUT/pass1/rows_s*.jsonl" --arms F C --pairs F:C --out-dir OUT/families` -> `famci_F_vs_C.json` (fam_ci diff = F - C; labels C/F, the default of `--fam-labels`).
3. **Tactical command adherence:** `p5c_tactical.py --seam C1=<F seam> C0=<C seam> base=<P7 seam stage2_p7_full.npz> ...` (it hard-codes C1/C0; `field_read` relabels to F/C and drops its STRATEGIC stub).
4. **Strategic, in full:** `instruments/strategic_family.py read --arm F=<F seam> --base C=<C seam> --out strategic_family_F_C.json`.
4b. **M1 raw-plan table (field_rawdac.py), per arm, on the C3 tokens: after the seam exists, before step 5.** Needs `stage2_p7field_<arm>.proposals.npz`
   (the seam dump) and the arm's 64-proposal table `full_<ARM>_s{0,1}of2.jsonl` (its CTRL-SIM reference). navsim-crun python, PYTHONPATH = `<navsim devkit>;<PK>/refe` (`;`), CPU only, commit guard >= 6, at most 2 processes (the two shards):
   `for s in 0 1: field_rawdac.py score --arm <F|C> --proposals <seam>.proposals.npz --out <HOLD>/rawdac_<F|C>_s${s}of2.jsonl --shard $s/2 --tokens <PK>/raw/2026-10-11-c3-selection-headroom/tokens.json`
   (~0.5 s per token per process: 1,058 tokens = ~4.5 min per arm on two shards.) Resumable. Each shard writes `<out>.header.s<i>of2.json` with the dump's sha256; `field_read.py` refuses rows whose header does not carry the dump's sha256.
   Controls: `field_rawdac.py check --rows "<HOLD>/rawdac_<ARM>_s*of2.jsonl" --proposals <seam>.proposals.npz --table "...full_<ARM>_s*of2.jsonl" --out ctrl_<ARM>.json` must print `ZZRAWDAC_CHECK PASS` (CTRL-SIM >= 95 % per proposal, CTRL-PICK).
   RED (once per code version): `score ... --red --limit 150 --out rawdac_<ARM>_RED.jsonl`, then `check ... --red-rows rawdac_<ARM>_RED.jsonl` must print `RED ... PASS (the control drops)`.
5. **Read:** `field_read.py --rows "OUT/pass1/rows_s*.jsonl" --f F --c C --seam-f .. --dump-f .. --seam-c .. --dump-c .. --csv-f .. --csv-c .. --seam-ledger OUT/seam_ledger.json
   --convert-report PLANS/convert_report.json --proptable-f "...full_F_s*of2.jsonl" --proptable-c "...full_C_s*of2.jsonl" --rawdac-f "<HOLD>/rawdac_F_s*of2.jsonl" --rawdac-c "<HOLD>/rawdac_C_s*of2.jsonl" --c3-tokens <PK>/raw/2026-10-11-c3-selection-headroom/tokens.json
   --fam-ci .. --strategic .. --tactical .. --out OUT/field_read.json --text OUT/VERDICT.txt` (+ `--fs/--cs` with their seams/dumps/CSVs once the replicates exist).
   A real read REFUSES (exit 2) without any of the CSVs, ledger, family files or tables. Without `--fs/--cs` the best result is "separated, single seed, not yet a lever claim".
6. **CTRL-ID on the real arm:** same command with `--f C --c C` and `--dry-run "CTRL-ID: C against itself"` (stamped DRY RUN): must print NOT PROVEN, E1 0.0000 [0.0000, 0.0000].

Set `FIELD_READ_P5C_DIR` to the `eval/p5c` directory if `field_read.py` is not run from its repo location. CPU only, `CUDA_VISIBLE_DEVICES=-1`, commit guard >= 6 first.
M1 reports BOTH forms, labelled: `raw_navsim_dac` = `navsim_dac_violation(mode="raw")` (the card's form, from step 4b) and `raw_transparent` = the plan executed as given on the transparent NAVSIM scorer (table `tp`); plus `simulated_navsim_dac` and, as an M2 cross-check on the C3 subset, the PICK's mode="raw" dDAC. M2 itself is unchanged. A real read REFUSES without `--rawdac-f/--rawdac-c`. Q-F2 in the JSON.
