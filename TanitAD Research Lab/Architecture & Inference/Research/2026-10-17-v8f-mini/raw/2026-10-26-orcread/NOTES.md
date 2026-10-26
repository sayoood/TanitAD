# orc_read.py -- reader for PREREG_V8F_ORECOVER (blob 28e28c88, sha256 c1925795..., + Amendment 1). Built and tested on SYNTHETIC inputs; no O-RECOVER data was read.
**Reuse:** `t1hd_read` imported as a module (cells, F1 reducer, `pbrow`/`srow`, K1-K7, `run_one`); `t1k_read.comps` for ADE6 (loaded through its `--selftest` door, which runs its R-3 test); `taniteval.ci` for every interval.
**CLI:** `python orc_read.py --manifest runs.json --out ORC_READ.json --tool-root <stack+taniteval export> --eval-cache --tracks --eval-grid --eval-join --m2b-ckpt 0=P --m2b-ckpt 1=P [--prereg FILE] [--key-perturb-n K --key-perturb-eligible K --key-built-p K --key-init-sha K]`.
rc 0 outcome written (prints sha256) | rc 2 REFUSED, **nothing written** | rc 3 UNPLACED. `--n-boot != 2000` labels the JSON "TEST: NOT A READ".
**Manifest:** `arms.{ORC,ORC0,M2b?}.{"0","1"}` each `{train_dir (argv/summary/selfcheck.json), token, ckpt_tail?, t1:{tag,ticks,dump_dir,json,k7?}, t1_logged?:{...}, t0:{json,npz}}`; M2b needs no train_dir/token.

## Choices where the PREREG is silent (all in code, all tested)
1. **Matched cells.** R-TLAT/R-TLON score only cells (window, tick 1-2) scored in BOTH runs; the own-cells version is a `_own_cells_beside` row. F1 deltas are the estimator's own 4-dp value (as T1-HEADS); R-T0/R-ADE use the unrounded point.
2. **"separated-better and >= 2F"** = CI excludes 0, delta > 0, delta >= 2F (inclusive). "Within +F/2" = point <= 0.0604 (inclusive). HARMS fires at ANY seed.
3. **PARTIAL** = exactly one head moved at BOTH seeds, guards held. Predicates are evaluated literally, every match listed in table order (they are mutually exclusive by construction).
4. **THE TABLE HAS A HOLE -> UNPLACED (rc 3).** A head separated-better at both seeds but < 2F at one is neither NO EFFECT nor PARTIAL nor ROBUST. The reader reports UNPLACED with all rows and no committed-next, rather than guess. Cheapest repair, to be decided BEFORE data: NO EFFECT := "no F1 row separated-better AND >= 2F at both seeds".
5. **K4 cannot pass on the `--limit causal` runs the PREREG lists** (MEASURED, `out/smoke_real.out`, pre-existing T1-HEADS data: tick 0 of the causal run differs from its T0 eval by lon F1 0.045, lat 0.0075; of the LOGGED run by 0.0). T1-HEADS ran K4 on `--limit logged` runs. So each arm needs a `t1_logged` run (6 extra CPU rollouts); without it K4 is evaluated on the causal run and the read is REFUSED with a hint.
6. **K7 ("== banked causal dumps")** has no reference for plain arms. Modes per run (`t1.k7`): `banked` (HACC), `ref` (flags-off dump of the same checkpoint, bit-for-bit), default `x1` = rebuilt executed track == dumped `cl` and tick poses, both <= 1e-9 (read 0.0 on real data).
7. **Trainer key names are unknown**: `ego_perturb_n/eligible/p` and `init_ckpt.sha256` are looked up bare and under `totals.`/`levers.`/`built_levers.` in summary/selfcheck/metrics; CLI-overridable; conflicting records FAIL. Init sha is also taken from the file at `--init-ckpt` if it exists here; M2b sha must equal the PREREG's banked 601cb435.../6104fb15... (`--banked-sha` is TEST ONLY).
8. **Token:** kind `launch-gate-token`, verdict PASS, `binding.argv_sha256` = this run's argv (launch_gate's JSON-array hash), 40-hex commit, data_sha256 present; ORC/ORC-0 data_sha256 equal. **HMAC is NOT verified** (key lives on the launch host). Commit equality across runs is not required.
9. ORC vs ORC-0 argv identical except `--ego-perturb-p`/`--out`: ANY other `--ego-perturb-*` flag must match on both (or be absent on both). `parse_flags` keeps all values of a flag: `v8f_pgt_read._flags` keeps only the first, so `--ego-perturb-vscale 0.5 1.0` vs `0.5 9.9` would pass it.
10. Extra controls: run identity (T1 json + ticks meta + T0 json name this arm's `<dir>/ckpt_last.pt`; guards ORC/ORC-0 swaps), T1 config (causal, m=10, periodic, logged scene, deadreckon, heads), window identity ticks=dumps=T0 across arms, 2399/141 counts, R-2 (reader's ADE6 delta = the run's own `paired_model_minus_floor_t1`), the reader's own K1 on its rows.

## Evidence
`out/pytest_full.out` (49 passed, 96 s), `out/mutation.out` (23 mutants, 23 KILLED; per-mutant pytest tails in `out/mutation_tails.out`), `out/smoke_real.out` (RealBackend wiring on T1-HEADS data: K1-K7 PASS on real causal s0/s1, K4 logged PASS, K4 causal FAIL, ORC-vs-itself exactly 0).
## Not done
Checkpoint `cfg`/`built_levers` not opened (summary/selfcheck levers are); beside rows: S-POS vs CTRV and crawl fraction are carried per run (not paired), four families / HACC re-fit not computed; strategic family reported PAUSED. Real runtime is ~1 min per run (6-12 rollouts' K1-K7 at B 2000) -- run detached, not as a waiting agent.
`pytest` cap: a venv launcher + child count as ONE run (the literal process count reads 2 for one run); the harness counts runs. Test fixtures are small-file heavy: E: mkdir is ~50 ms.

## Amendment 3 (A3-2026-10-26, before any run)
* `apply_a3.py`: `g_argv` requires ORC to name the law once at its registered values plus `--ego-perturb-seed <seed>`, and ORC-0 to name none
  of it. `g_pair` drops the law flags from the compared set (`drop = {--out, --ego-perturb-p, --ego-perturb-seed} | LAW`).
* `apply_a3b.py`: `PREREG_ID` names A3; `PREREG_V8F_ORECOVER.A3.md` is the landed PREREG with Amendment 3 appended; one test checks it extends A2
  and carries the reader's law value for value.
* `out/pytest_full_a3b.out`: 53 passed. `out/mutation_a3.out`: 25/25 mutants KILLED (M24, M25 are A3's).

## Amendment 4 (A4-2026-10-26, before any run)
* `apply_a4.py`: K4's comparator is the arm's own T0 eval at `--ages fresh`, read from its OWN manifest key `t0_fresh`. R-T0, T0-CKPT and
  the families stay on `t0`, box 2's standard eval. A manifest without `t0_fresh` is REFUSED. Two new controls:
  * `T0-AGES`: `t0` must say `ages: cycle` and `t0_fresh` must say `ages: fresh`, so neither eval can stand in for the other;
  * `T0F-CKPT`: the fresh eval names this run's checkpoint.
* Tests (+4): the A4 text extends A3 and carries the K4 rule. K4 reads `t0_fresh` and R-T0 the standard eval. In the fixture the fresh eval is
  7 m worse for ORC, so a reader that took R-T0 from it would read +7 m, and the synthetic K4 fails unless it is handed the `-fresh` file.
  A missing `t0_fresh` is refused, and the ages and fresh-checkpoint controls fire on exactly the edited runs.
* `out/pytest_full_a4.out`: **57 passed**. `out/mutation_a4.out`: **30/30 mutants KILLED** (M26-M30 are A4's). `out/snap_a4_check.out`:
  the sources after the proof equal `snap_a4_SHA256.txt`, taken before it.
* Correction to "Not done" above: the strategic family is NOT reported paused. The reader labels it "NOT PAUSED for v8F (PI ruling
  2026-10-17, R73)" (`orc_read.py`, `strategic_family`), and a test pins that label. The note was stale.
