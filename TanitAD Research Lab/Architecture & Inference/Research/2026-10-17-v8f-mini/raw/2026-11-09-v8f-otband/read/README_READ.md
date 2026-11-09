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
