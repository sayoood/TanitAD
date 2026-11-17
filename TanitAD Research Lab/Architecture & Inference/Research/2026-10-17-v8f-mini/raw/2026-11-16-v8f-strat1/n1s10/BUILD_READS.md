# BUILD_READS: the STRAT1 N1-at-seeds-10/11 read (run) and the S1d read (built, not started)

Evidence class MEASURED (ours, zero training). Tiers: the 24 driving cells are T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; 141 episodes / 2,399 windows per seed); STRATEGIC is T0 at the window's first tick. Estimator: paired episode-cluster bootstrap B 2,000 seed 0, "another draw of EPISODES" only. All paths are under `F:/TanitAD-scratch/strat1/read/`.

## 1. N1 at s10 / s11 (addendum PREREG_ADDENDUM_N1S10.md, frozen, no rule changed)

### Result (`n1s10/out/RESULT_STRAT1_N1S10.json` + `.md`, wall 857 s, problems none)
* COST4 (S1 - C0): TAC-LAT at seeds 0, 10, 11 (same as the SEEDS read).
* NAV4: none. N1 - C0 on TAC-LAT costs at seeds 1 and 11 only (both init 1): s0 -0.0158 [-0.0329, +0.0012], s1 -0.0233 [-0.0450, -0.0037], s10 -0.0063 [-0.0233, +0.0096], s11 -0.0396 [-0.0597, -0.0188].
* HEADS4: none. S1 - N1 on TAC-LAT costs at 0 of 4 seeds: s0 -0.0067, s1 +0.0083, s10 -0.0088, s11 +0.0146 [+0.0042, +0.0263] (a gain flag at s11).
* **Attribution of TAC-LAT: NEITHER** (not attributable at 4 seeds). Reported as written; S1d is read as the card says (the S1d chain is training on the pod).
* Beside, no bar: C0_s10 - C0_s0 on TAC-LAT -0.0475 [-0.0692, -0.0275] (about 7x F 0.0067); C0_s11 - C0_s1 +0.0075 [-0.0088, +0.0255]. The two replicate pairs differ in sign and size; the next STRAT1-family card should set F from at least two replicate pairs (H-ESTIM-SEED-1).
* Four families per arm pair (N1-C0, S1-N1, S1-C0 at 4 seeds; C0 replicate pairs): LONGITUDINAL, LATERAL, TACTICAL cells in the JSON/MD (24 cells, n 2,399 windows, 141 episodes each); STRATEGIC route following (real / donor / zero at three arcs, with n per arc, N1 and S1 at four seeds) and route-setting RC-A in the JSON. STRATEGIC detected info for N1: not computable, N1 has no context heads (n = 0); S1 detected info at seeds 0 / 1 / 10 / 11 is in RESULT_STRAT1.json and RESULT_STRAT1_SEEDS.json. The card's S-label is defined on S1 at two seeds; the four-seed label the frozen strategic reader prints (S-PARTIAL) is not quoted.
* Controls: K1 self-pair exactly 0.0; K-RE passes (144 rows equal RESULT_STRAT1.json, 192 rows equal RESULT_STRAT1_SEEDS.json, 0 mismatches).

### What ran, in order (timings from `n1s10/out/n1_read.log`)
1. Bank (`after_n1s10.sh`, run with SKIP_LAUNCH=1 since the pod chain had ended): pod uploader `strat1_bank_pod_n1s10.py` (line 7 TAGS) through `bank_n1s10.sh`; PRIVATE HF dataset `Sayood/tanitad-v8f-strat1-runs` (the repo of the other STRAT1 arms), rev 7c8a4c86, hub LFS sha256 = manifest for every file. ckpt sha256: N1_s10 fa20a653a7bd, N1_s11 26d95e4f87d1; commit a4357e79. Pulled to `ckpts/N1_s10`, `ckpts/N1_s11` (ckpt_last.pt, argv.json, BANK_MANIFEST.json), local sha256 equals the manifest.
2. `launch_n1_read.sh` (WMI detached, launched 08:16:42Z): two T1 dumps (437 s, 268 s), six T0 passes (33-40 s each), unit tests (15 passed), strategic read, then `n1_read.py`. Done line `ZZ STRAT1 N1S10 READ DONE ZZ`.

### Pieces (all in `n1s10/`)
* `n1_run.py`: wrapper over the frozen `seeds/seeds_run.py`. Lines 22-28 add N1_s10 / N1_s11 to ARMS and TEXT_ARMS and not to CTX_ARMS (text nav, no context heads, as N1_s0 / N1_s1); lines 30-47 let those two tags through the bank-manifest commit check at a4357e79 (the seven first arms still need e81bde01).
* `n1_read.py`: `attribution` (line 40), `n1_outcome` (line 49, NAV4 / HEADS4 / COST4 through the frozen `strat1_seeds_read.count_rule`), `k_re` (line 72, refuses a vacuous compare), `main` (line 104). Recomputes all 13 arms from the dumps with the frozen `strat1_read` cells.
* `after_n1s10.sh`, `bank_n1s10.sh`, `strat1_bank_pod_n1s10.py`, `launch_wmi.ps1`, `job_n1_read.cmd`, `launch_n1_read.sh`: bank, pull, verify, launch.
* `test_n1_read.py`: 15 tests green. Per rule: the four attribution labels, only-COST4-cells-attributed, cell independence, NOT-YET on a missing seed, pair-swap; mutations: threshold 3 to 2 flips, threshold 3 to 4 flips, attribution branches distinct, K-RE changed row and vacuous compare, arm classification, commit widening for the two N1 tags only.
* **Finding on the "at least 1 seed in each init" condition:** at 3 of 4 seeds it is implied (any 3 of {0, 10} + {1, 11} hold a seed of each init), so dropping it alone cannot flip a case at the addendum's k = 3. The test proves this exhaustively over all 16 seed subsets, and shows the condition does its work at k = 2 (dropping it flips {0, 10}); a mutated rule (k = 2, no init) flips the full outcome. The rule is kept as frozen.
* Frozen readers used unedited (sha256 prefixes): strat1_seeds_read.py 2d30dd83, seeds_run.py 237e8655, strat1_read.py fec13926, strat1_lib.py 538edf86, strat1_strategic.py f27c005d, run_dumps_strat1.py 1ebc3692, run_t0_strat1.py f0072c63 (full sha256 in the result JSON `provenance`).

## 2. S1d at s0 / s1 / s10 / s11 (built; `after_s1d.sh` NOT started; the MM starts it)

Pod state at build time: tree new9 = 4dedac45, chain `strat1_s1d.sh`, `S1d_s0` training (mint 08:08Z, run from 08:12Z); sentinels `ZZ STRAT1 S1D DONE ZZ` / `ZZ STRAT1 S1D FAIL ZZ` in `/root/strat1/logs/s1d.log`.

* **The card's rule.** Section 6, COSTS-HEADS row: "arm S1d = S1 with `--ctx-detach on` (built, default off), same seeds and reads". The card gives S1d no outcome rule. **Decision for the MM:** `s1d/PREREG_ADDENDUM_S1D_READ.md` holds PROPOSED rules (per S1 COST4 cell: CURED = not COST4 under S1d and S1d - S1 GAIN4; PARTIAL = not COST4 under S1d without GAIN4; REMAINS = COST4 under S1d; NEW-COST and GAIN4 vs C0 listed; DETECT-LOST for detected info). They must be frozen, or replaced, before the first S1d number exists. The reader prints them into its result.
* 4dedac45 differs from e81bde01 in `stack/scripts/strat1_check.py`, `stack/tests/test_strat1_s1d.py`, `stack/tests/test_v8f_strat1.py`, and from a4357e79 in the first two (MEASURED, git diff --name-only in the repo mirror), so S1d arms are accepted at 4dedac45 for the four S1d tags only (`s1d_run.py`).
* `s1d_run.py`: adds the four S1d tags to ARMS, TEXT_ARMS and CTX_ARMS (text nav with ctx heads, as S1) and widens the commit check; wraps `n1_run.py`.
* `s1d_view.py`: the frozen STRATEGIC and ctx readers read arm names S1_s{s}; S1d T0 passes are copied under the S1 name into a temporary view folder together with the C0 passes (nothing in the real T0 folder changes); `launch_s1d_read.sh` deletes the view after use.
* `s1d_read.py`: `s1d_outcome` (line 46), `detect_lost`, K-RE against RESULT_STRAT1.json, RESULT_STRAT1_SEEDS.json and RESULT_STRAT1_N1S10.json; pairs S1d-C0, S1d-S1, S1d-N1, S1-C0, N1-C0 at four seeds; four families in full as the N1 read; STRATEGIC route following and detected info (with K4) at the pairs {0, 1} and {10, 11}.
* `after_s1d.sh`: polls the pod log (sleep 300, no model turns, up to 48 h), banks the four arms through `strat1_bank_pod_s1d.py` / `bank_s1d.sh` (same private HF repo), pulls and sha256-checks against the manifest at commit 4dedac45, then launches `launch_s1d_read.sh` through `launch_wmi.ps1` / `job_s1d_read.cmd`. SKIP_LAUNCH=1 stops before the launch. Bash syntax checked (`bash -n`); the polling and pull path is the one that ran for N1.
* `launch_s1d_read.sh`: tests, 4 dumps (about 5-7 min each), 16 T0 passes (about 35 s each), view, strategic and ctx reads, reader. ESTIMATED 45-60 min in total (N1 measured rates). Done line `ZZ STRAT1 S1D READ DONE ZZ`.
* `test_s1d_read.py`: 16 tests green (CURED / PARTIAL / REMAINS, NEW-COST, GAIN4, 2-of-4 not COST4, NOT-YET; mutations: threshold 3 to 2, REMAINS branch dropped, CURED without the detach gain, swapped pairs; DETECT-LOST; view builder renames and refuses a missing pass; arm classification; commit widening for the four S1d tags only).
* Not exercised end to end: the S1d dumps, T0 passes, view and strategic / ctx steps have no checkpoints yet. Only the unit tests and `bash -n` ran. The ctx reader at seeds 0 1 and 10 11 on the view is the SEEDS-read pattern.

## Deviations and notes
* The preliminary driving-only run of `n1_read.py` was stopped (its process killed) once the launcher reached the same step; no result file came from it.
* The reader `strat1_strategic.py` loops `for fam in (S1, N1, C0)` and prints a label over whichever seeds it is given; for four seeds that label is not the card's and is not quoted.
* The N1 pod scripts remain on the pod in `/root/strat1/tmp/` (bank scripts, manifests, bank log); not deleted (the pod's own tmp; the manifests are evidence).

## Deletions (own temp, literal paths)
* `F:/TanitAD-scratch/strat1/read/n1s10/bt2` 11 KB; `F:/TanitAD-scratch/strat1/read/s1d/bt2` 102 KB; `F:/TanitAD-scratch/strat1/read/card_tmp.md` 20 KB (a pulled copy of the card, source in git); `F:/TanitAD-scratch/strat1/read/n1s10/out/n1_read_driving.log` 0.3 KB. The launcher deleted `n1s10/bt` itself. All succeeded.

## Manifest (all on F:, none in git, nothing staged: scratch study files, the MM lands)
| artifact | where |
|---|---|
| N1 result, JSON + MD | `n1s10/out/RESULT_STRAT1_N1S10.json`, `.md` (F: only) |
| N1 strategic | `n1s10/out/STRAT1_STRATEGIC_N1S10.json` (F: only) |
| N1 checkpoints | HF private `Sayood/tanitad-v8f-strat1-runs` `N1_s10/`, `N1_s11/`; local `ckpts/N1_s10`, `ckpts/N1_s11`; pod `/root/strat1/runs/N1_s10`, `N1_s11` |
| N1 dumps / T0 | `t1/dump_STRAT1-N1_s1{0,1}_m10_causal_heads`, `out/t0/eval_STRAT1-N1_s1{0,1}_*` (F: only; re-creatable from the banked checkpoints) |
| N1 code and tests | `n1s10/*.py|sh|ps1|cmd` |
| S1d code and tests, proposed rules | `s1d/*` |
