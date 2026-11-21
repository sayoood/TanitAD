# BRIEF: Amendment A3.2: TB-0's R-A judged on rank-0 rows (PI ruling 2026-11-16). Agent: Sonnet. MM, 2026-11-16.

Read `Project Steering/AGENT_OPERATING_STANDARD.md` (repo `E:/Projects/TanitAD`) first.
- **Binding:** stage, never commit or push; end with a deliverable manifest; escalate integration.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a32/`. Write everything there.
- Scratch goes under `F:/tanitad_scratch/p5c_a32_tmp/`. TMP/TEMP/TMPDIR live there for every test run, with `CUDA_VISIBLE_DEVICES=-1`.

## The ruling (the MM's options; the PI chose option 1)
- **Measured facts:** `PK/raw/2026-11-16-p5c-window/tb0/TB0_RA_FINDINGS.md` (repo, landed `147ba68a`; read it with `git -C C:/Users/Admin/tanitad-push show FETCH_HEAD:<path>` after `git fetch -q origin agent/arch-inf-20260803`). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- **TB-0** = R-B (P5-R EVAL labels, per-rank criteria) AND R-A (the Tc20s warm start equals the dev-box Tc20s build within 1e-9 on EVERY common row).
- **On the pod:** R-B passes. R-A misses at 1.49e-9 on 2 of 2,400 rows, both rank 1; the rank-0 max is 9.36e-10.
- **PI 2026-11-16, verbatim:** *"Follow your recommendatiobs"*. This selects option 1: **R-A is judged on rank-0 rows only.** Rank-1 rows' R-A is computed and DISCLOSED (max, count over tol), never gated.
  - Reason: under A3 §3.3's forced `rank0` policy, C1 uses only rank-0 Tc20s/P5-R labels; rank-1 rows stay raw.
- **Unchanged:** R-B, TB-1 and every other gate.

## Deliverable, in priority order
1. **Gate code.** `p5c_table_gates.py` (source: repo `PK/raw/2026-11-11-p5c-readiness/a1/p5c_table_gates.py`, blob `5c2f49cf`, sha256 `d600a830…`; export with `git cat-file blob`; the mirror has `core.autocrlf=true`, so never go through a working tree).
   - Add a CLI flag `--tc20s-ranks {all,0}`. **Its default `all` must reproduce today's behaviour bit-for-bit.** With `0`, `gate_tb0` (the R-A part of TB-0 only) judges max abs diff, sim8 identity and missing rows on rank-0 common rows only.
   - It also writes into the detail: `ranks_judged`, the rank-0 max and n; and for rank 1, `disclosed_rank1_max_abs_diff`, `disclosed_rank1_n_over_tol` and n.
   - ⛔ TB-1's Tc20s check stays on all ranks. The `ok` logic for R-B is untouched.
   - Put the flag value in the gates.json and meta, so a reader can see which scope was judged.
2. **Runner.** `p5c_pod_label.sh` (source: hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a31/a1/p5c_pod_label.sh`, blob `727e72d4`):
   - pass `--tc20s-ranks 0` on the CONTROL-pass gates call only (lines 220–223);
   - update `PIN_GATES` to the new sha256;
   - nothing else.
3. **Tests**, extending `a1/test_p5c_table_gates.py` (blob `82ce4a58`) and `p5c_a31/a1/test_p5c_pod_label_a31.py`. Each item needs a test, and each mutant must turn the suite RED:
   - (a) the default `all` gives the identical result to the old code on the same fixture: FAIL when only a rank-1 row exceeds tol;
   - (b) with `0`, a rank-1-only excess gives PASS and is disclosed with its exact max;
   - (c) with `0`, a rank-0 excess still gives FAIL;
   - (d) with `0`, a missing rank-0 row gives FAIL;
   - (e) mutants: flag ignored; ranks inverted; disclosure dropped; TB-1 also rescoped; runner without the flag; runner flag on the MAIN gates call.
   - Run the full old suites: they must stay green apart from deliberately updated pin assertions, which you list.
4. **Real-data proof on the pod, read-only.** Run the NEW gates module against run 1's control outputs with the same arguments the runner used (read lines 220–223), plus `--tc20s-ranks 0`, with `--out-dir` set to a NEW dir `/workspace/data/p5c_a32_check/`.
   - Expected: TB-0 PASS, rank-0 max 9.36e-10, disclosed rank-1 max 1.4901395672950457e-09.
   - Also run with `--tc20s-ranks all`: it must reproduce run 1's FAIL value bit-for-bit.
   - Run-1 inputs: `/workspace/data/p5c_label/control/{bank,out}`. The other inputs are under `/workspace/data/p5c_ship/`.
   - ⛔ Write nothing into `/workspace/data/p5c_label*` or `/workspace/data/p5c_ship/`.
5. **`PREREG_P5C_A3_2_DRAFT.md`** (pins only, plus the ruling): the PI quote verbatim, the change, the new blobs and sha256s (`git hash-object --no-filters`; 0 CR bytes), the test results, the real-data proof, and a disclosure line saying the ruling is post-data.
   - Write `HANDOFF.md` (at most 2k tokens).

## Pod access (binding)
- **Commands:** `bash C:/Users/Admin/par18/pd.sh <t> '<cmd>' 2>/dev/null | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<IP>/g; /hf_/d'`.
- **Ship a file:** `pd.sh 60 'cat > /path' < file`, then verify the sha256 on the pod.
- **Pod python:** `/workspace/venv-teacher/bin/python`, run with `nice -n 19`.
- ⛔ Never print `C:/Users/Admin/freept/pod_direct.txt`, IPs, ports or tokens.
- Use no GPU. Start no long jobs. Kill only your own processes, by PID.

## ⛔ Do not
- Commit or land anything.
- Edit A3/A3.1 or any other gate or rule.
- Touch other run dirs.
- Delete anything outside your own scratch (literal paths; log each deletion).
- Exceed about 150k context.

**Report:** at most 300 words, plus a manifest.
