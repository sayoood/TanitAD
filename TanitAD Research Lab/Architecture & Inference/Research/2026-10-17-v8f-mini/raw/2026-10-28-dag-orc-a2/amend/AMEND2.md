
## Amendment 2 (A2-2026-10-28): the composed perturbation drew from the swap's own stream; fixed, and all four runs re-run, before any DAG-ORC row is read

Written 2026-10-28 ~02:30 by the Master Mind.

**State when written.**
* No DAG-ORC T1-kin rollout, read or row exists.
* r1 (`chain_v8f_dagoaccorc.sh` at `e7ea20d1`, started 01:16 under the PI's go of 2026-10-27) passed all four gates, A and REP at both
  seeds (`box2_ship/logs/chain_v8f_dagoaccorc.out`).
* r1 then trained A s0 (3,000 steps), pulled it and passed its run record. **K-REACH failed:** `ego_perturb_n / ego_perturb_eligible` =
  19,161 / 76,898 = **0.2492**, outside 0.3 ± 0.02. The swap share was 0.1013 and the overlap was exactly 0.
* The chain stopped, as designed (`ZZDAGOACCORC-CONTROL-FAIL`). REP s0, A s1 and REP s1 never ran. The refused run is never rolled out
  or read.

**The defect.**
* `DaggerSwap._seed_of` (`stack/tanitad/data/v8f_dagger.py`) and `EgoPerturb._seed_of` (`stack/tanitad/data/v8f_mini_data.py`) are the
  same formula, `(seed × 1,000,003 + step) mod 2^62`. The registered argv sets both seeds to s.
* Both levers read their selection uniform from row 0 of `torch.rand(k, B)`. The two rows are therefore identical (verified:
  `torch.equal` is True).
* So a sample eligible for both levers is swapped iff u < 0.1 and perturbed iff 0.1 ≤ u < 0.3. That gives P(perturbed | not swapped) =
  0.2 / 0.9 = 0.222, against 0.3 for a sample only the perturbation may touch. The measured 0.2492 lies between the two.
* Amendment 1 §1 fixed the semantics as *"drawn, at its own p and from its own seeded generator"*, which means independent of the swap.
  The build met that in form but not in fact: two generators with the same formula and the same seed are one generator.
* The composition tests missed it, because their fixture did not combine equal seeds with overlapping eligibility. K-REACH caught it on
  the first real run, which is the job the PREREG gave it (*"the gate does not cover the loop's draw"*).

### 1. What changes

* **The trainer.** On the composed path only, the perturbation's generator takes a domain-separated seed, so its stream is independent of
  the swap's at equal seeds. Every other path stays bit-identical: flag off, either lever alone, plain M2b. The commit, the recorded field
  and the proofs are listed in §3.
* **The runs.** All four runs, A and REP at both seeds, are re-run from their gates at the fixed commit by the r2 chain.
  * Every r1 artifact is renamed `<name>_r1_kreach_stop`, on box 2 and on the dev box. Nothing is deleted.
  * Gates are re-run because the commit changed, and the launch-gate tokens bind to the new commit.

### 2. What does not change

* The arms, the registered argv (including `--ego-perturb-seed <s>` and `--dagger-seed <s>`) and the K-REACH bands.
* The one-lever rule, the reader (`2e5f9f17…`), the crawl set, the domain, the floors and §7's outcome table.

### 3. Proofs, before r2 launches

| check | result | file (under `raw/2026-10-28-dag-orc-a2/`) |
|---|---|---|
| the fix: `EgoPerturb` takes a `stream`; 0 = the original formula, so every non-composed path is unchanged; the composed stage sets `EGO_PERTURB_COMPOSED_STREAM = 1`, a sha256-mixed seed; recorded as `levers.ego_perturb_stream`; a G-DVB read-back and a resume guard | trainer `1c888280…`, data lib `c117e1cf…`, tests `faad3be2…` | `fix/NOTES.md`, `fix/diff.patch` |
| the new tests on `e7ea20d1` (the shared stream) | **RED**: P(perturbed \| not swapped) 0.0000 at 0.5 / 0.5 and 0.221 / 0.224 at 0.1 / 0.3; the two selection rows equal; a 200-step run at box 2's shape reads 1,561 / 6,345 = 0.2460 | `fix/logs/new_red_on_old.log` |
| the same tests on the fix | **GREEN**: the same 200-step run reads 1,860 / 6,345 = 0.2931, swap 720 / 7,380, overlap 0, `levers.ego_perturb_stream` 1 | `fix/logs/new_green2.log`; the run record `r2/fixtures/compose_fix_real/A_summary.json` |
| the flag-off golden (`ego_dagger_compose_golden.json`, unchanged) | passes | `fix/RESULTS.md` |
| the seven v8F suites, one file at a time | 36, 80 (+2 skipped), 95 (+2 skipped), 57, 7, 56 and 13 (+32 skipped) passed, 0 failed | `fix/RESULTS.md` |
| the mutation sweep | **27 / 27 killed**, including `M17b`: the stream set, gated and recorded as 1 but not mixed into the seed | `fix/RESULTS.md`, `fix/mutation.txt` |
| the r2 chain (aside, the predecessor check, `STREAM_PATTERN`, `levers.ego_perturb_stream` = 1 on A and 0 on REP) | 66 / 66 tests, 223 / 223 mutants killed | `r2/logs/selftest_chain_r2.out` |

* The two skips in `test_v8f_dagger_train` need `raw/2026-10-26-dagger-train/`, which the fix's tree did not export. They are re-run on an
  archive of the landed tip before r2 launches.

### 4. Cost

* r1 used box 2 from 01:16 to 02:10 (0.9 h). r2 is about 2.8 h (four runs, the r1 basis). That is about 3.7 h in all, against the ~3 h the
  PI's go was asked as. The 0.7 h overrun is disclosed here.

### 5. The class

*Two seeded generators with the same formula and the same seed are one generator.* Two levers can be independent in a specification and
still be coupled in a build. The test that exposes this has to reproduce the registered argv's seeds together with overlapping
eligibility; a fixture with either one alone passes.
