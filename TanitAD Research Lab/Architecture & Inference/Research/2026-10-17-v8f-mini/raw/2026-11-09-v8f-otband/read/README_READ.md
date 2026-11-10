# OTBAND read (card `PREREG_V8F_OTBAND.md`, frozen; prefix sha256 `fd81c66e`)

Written 2026-11-09 23:55 by the Master Mind, after OB-CTL_s0 and OB-BAND_s0 were banked and **before any OTBAND dump existed**. Tier T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), as the card's Read section commits.

## What runs, in order
1. `make_tree.py`: `git archive` (core.autocrlf=false) of commit `202849da` (`stack/tanitad`, `stack/scripts`, `taniteval`) to `F:/TanitAD-scratch/otband_read/tree`; `TREE_PINS.json`. MEASURED: `v8f_t1_eval.py`, `four_families.py`, `floors_kin.py`, `t1_eval.py` and `v8f_dagger.py` are byte-identical to the DAG-OACC baseline's pins (`v8f-dagoacc-t1/PINS.txt`); `v8f_mini.py` and `v8f_mini_data.py` carry the OTBAND diff; `ci.py` is the commit's own.
2. `run_dumps.py` (via `job_dumps.cmd`, launched detached with `launch_wmi.ps1`): the baseline's own argv (`v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv`) with only `--ckpt`, `--tag`, `--out-dir`, `--dump-ticks` replaced; CPU, 2 threads; checkpoint sha256 checked against its bank manifest; waits for 4 GB of free memory.
3. `parity_check.py`: **control P0.** The commit tree run on the DAG-OACC s0 checkpoint must reproduce the banked baseline dump. Rule, written before the run: PASS-EXACT if every common array is identical; PASS if the identifiers and model-free arrays (`eid, clip_index, ws, g, fl, v0`) are identical and every other float array differs by at most 1e-4; otherwise FAIL, and the read is INVALID until explained. `read_otband.py` refuses to run unless P0 passes.
4. `read_otband.py`: the card's four families per band against OB-CTL of the same seed (paired episode-cluster bootstrap, B 2,000, seed 0), DAG-OACC beside, the replicate floor from OB-BAND_rep, the bars B1 to B6 and the committed outcome. Geometry is `read_lib.py`, which is `band_eval/band_lib.py` (sha256 `7c098040`) with only its tree path changed and the stack path added.
5. `heads_read.py` (agent): T's own decision reads from the `--dump-heads` ticks, through the registered `t1hd_read.py` (sha256 `3c6d9f79`). TACTICAL family, no bar.
6. `topk_read.py` (agent): `forward_topk`, K = 3: best-of-3 against top-1 ADE per band and the share of windows whose GT manoeuvre is among the K. Beside, no bar.

## Reading choices, committed here before any OTBAND number was read
* **(r1)** "up" is the upper bound of the paired 95 % interval of OB-BAND minus OB-CTL. For an accuracy row (B5, higher is better) the bound on the worse side is the lower one: B5 reads lo(BAND minus CTL) >= -0.02.
* **(r2)** B3's distance-keeping rows are absolute errors against the human's own minimum headway and time gap over the same knots, in both bands.
* **(r3)** "Any tactical row worse than its bar" (BAND-COST) covers B3, B5, and B2 when its interval is separated on the worse side (lo > 0 on ADE_tac or FDE_tac). A B2 that only fails to improve is BAND-NEUTRAL. B4 is read on its own and never sets the outcome.
* **(r4)** A row's replicate floor is |delta(OB-BAND_rep minus OB-BAND_s0)|. The replicate exists at seed 0 only, so its floor applies to both seeds' rows.
* **(r5)** Without the replicate, an outcome that needs the floor carries the prefix PROVISIONAL and decides nothing.
* **(r6)** BAND-GAIN's "B1 and B2 hold and exceed the replicate floor" means: B1 holds, B2 holds, and B2's two rows exceed their floors.

## What a partial read is
With one seed, or without the replicate, the output is a per-seed reading that decides nothing; the card requires both seeds to agree. OB-BAND-TOK is BLOCKED (stream C's `stok` dump); STRATEGIC is PAUSED (PI item 25).

## Temporary data (delete after banking)
`F:/TanitAD-scratch/otband_read/` (the tree, about 350 MB, and the dumps, about 160 MB per arm). The dumps are results: they are banked privately on HF before that folder is deleted.

## Amendment 1 (2026-11-10 00:05, after P0, before any OTBAND row was computed)
* **P0 read FAIL under its rule as written** (`PARITY_P0.json`). Identifiers and model-free arrays are identical (`eid, clip_index, ws, g, fl, v0, ha0`, 141 of 141 files), the first-tick one-shot plan `ol` differs by at most 1.72e-5 m, the m = 10 rolled track `cl` by at most 3.91e-3 m, and later-tick tick-file rows by up to 0.237 (`P`, `tau`, `plan`) and 5e-3 (T logits). The tick file's `tag` field names the run and differs by construction; it is metadata and is listed outside the rule from here on.
* **Explanation run P0b, PASS-EXACT** (`PARITY_P0b.json`): the baseline's own tool tree (`dagt1_tree_b43c7a9a`) on the same DAG-OACC s0 checkpoint, at this read's settings (CPU, 2 threads), equals the commit tree's dump bit for bit, every array of all 141 episode files and of the tick file (its `tag` field aside). So the OTBAND code reproduces the baseline exactly; the P0 gap is the environment of the banked run (thread count not recorded there), amplified by replanning.
* **Consequence for the read:** the DAG-OACC rows use dumps made here (`PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1`), so every compared pair shares one environment; the banked dumps serve as the window-identity reference. `read_otband.py` runs if P0 passes or P0b is PASS-EXACT.
* **Beside, decides nothing:** the environment gap on the rolled track (at most 3.9 mm) is far below every bar of the card (the tightest, B1 cross-track, is +0.02 m).
* `run_dumps.py` fix: `v8f_t1_eval.py` writes `t1v8f_<tag>.json` and no done file; the driver now writes `done_<tag>.txt` itself after checking the header, 141 episode dumps, the ticks file and the checkpoint sha256 (`--finalize` for the three runs of the first job, all rc 0).

## Amendment 2 (2026-11-10, after the seed-0 bars were read; beside, post-hoc, decides nothing)
* The retrain's own init checkpoint (`INIT_M2b_s{s}`, M2b s0 sha256 `601cb435`, M2b s1 sha256 `6104fb15`, each the `levers.init_ckpt` of OB-CTL's summary) is dumped here and read beside the bars: CTL minus INIT and DAG-OACC minus INIT. It is not in the card and sets no outcome.

## Amendment 3 (2026-11-10 05:20, after the seed-0 beside read, before any ladder number was computed; beside, post-hoc, decides nothing)
* The seed-0 beside read put the gap between the retrain and DAG-OACC on DAG-OACC's fine-tune. DAG-OACC differs from M2b in three ways at once: the fine-tune schedule (3,000 steps, lr 1e-4, p_gt 0.2, fully trainable), the DAgger rows (`--dagger-p 0.1`) and the acceleration term (`--w-acc 49.6969`). So the gap cannot be put on the DAgger rows alone.
* The banked ladder splits the three: DAG0 (`v8f-dagger/M2b_s{s}_dag0`: schedule only, `--dagger-p 0`), DAG (`M2b_s{s}_dag`: schedule plus DAgger rows, `--w-acc 0`), DAG-OACC (schedule, DAgger rows and acceleration term). All share M2b's init. They are dumped here (`PARITY_DAG0_s{s}`, `PARITY_DAG_s{s}`), sha256-checked against their banked done files, in this read's environment.
* `read_otband.py` adds, per row, `ladder_post_hoc`: step 1 DAG0 minus INIT (schedule), step 2 DAG minus DAG0 (DAgger rows), step 3 DAG-OACC minus DAG (acceleration term), each paired with the same estimator; the steps sum to DAG-OACC minus INIT. The card's bars and outcomes are unchanged.

## Amendment 4 (2026-11-10 05:38, after the full read, before any number of the pair below was computed; beside, decides nothing in this card)
* DAG-OACC's own run-to-run spread, for the next card's floors: the DAG-ORC card's REP (`v8f-dag-orc-oacc/M2b_s{s}_dagoaccorcrep`, DAG-OACC's argv at training seed 10 / 11; argv differs in `--seed` and `--out` only) is dumped here (`PARITY_DAGOACCREP_s{s}`, sha256 from the run's `SHA256SUMS.txt`) and read against `PARITY_DAGOACC_s{s}` by `dag_floor.py` (`out/DAG_FLOOR.json`), with the same band rows and estimator, plus TAC-LAT and TAC-LON accuracy.
* Found when reading the pair's provenance: REP was trained with a later trainer file (`v8f_mini_train.py` sha256 `1c888280` against DAG-OACC's `30819a6f`) whose added levers read back off. The pair is therefore seed plus trainer version, stated wherever it is quoted.
* `floor_table.py` (every row of `READ_FULL.json` against the band recipe's floor) and `summarise_full.py` (print only) are readers of `READ_FULL.json`; they compute no new estimate.
