# Brief: build the ROLLHIST arm "RHK": a DAgger bank generated under readout K, RH-like arms at two seeds (pod chain + gate + read extension). Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing; the MM lands and launches.
* No GPU. Do not edit protected files: launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py.
* No secrets, and no IPs, ports or pod ids in any file.
* PI go, first-hand: "Build both". It covers the build and the pre-registration; running needs a second go with the cost.

## Why
* RHp3 (p = 0.3 on the RH bank) read **ROLLHIST-HARMS**: P1 better on LAT, TAC and STRAT, LON MIXED, but G-T0 fails at both seeds. See `F:/TanitAD-scratch/rollhist/read/out/read_p03/MM_VERDICT_RHP3.md` and `RESULT_ROLLHIST_P03.json`. RH at p = 0.1 read ROLLHIST-PARTIAL (`read/out/read/RESULT_ROLLHIST.json`).
* The card's HARMS row (`F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST.md`, sha256 781eea87, line 123) names the next lever: "a bank under readout K (train on what K hands)". Its localisation step (UE.B) is NOT-COMPUTABLE, because STAGEB never ran.
* Read the card and Amendments 1 to 7 (same folder); they bind. Amendment 7 (81f8c5ac) and its build `build/b8p03/` are the closest pattern: checker kind variant, chain, arm, argv renderer, marker watcher, launcher, tests, DRY_MINT_PLAN.

## The arm (MM design; flag in the draft if the source contradicts it)
* **RHK_s0 / RHK_s1** = the RH_sK argv (`/root/rollhist/runs/RH_s*/argv.json`) with ONLY the bank flags changed, to a NEW bank `/root/rollhist/bank_k/sK` generated under readout K (the card's `R0KHb20` form; find its readout token and how `rh_bank.sh`, `rh_env.sh` `rh_readout_flags` and `RH_READOUT` select the readout).
* Everything else is identical to RH, including the dose `--dagger-p 0.1` (RH's dose; p = 0.3 harmed G-T0 more), init, steps, seed, dagger-seed and code 97fe0a56.
* The comparator is the EXISTING B0_s0 / B0_s1 (no new B0). RHK - RH goes beside, with no bar.
* The bank step: the RH chain's own bank step (`/root/rollhist/rh_chain.sh`, `rh_bank.sh`) with readout K, and the same coverage gate (COVERAGE >= 0.5, else BLOCKED). It writes `bank_k/`, never `bank/`.

## Priority order (bank each part as it is done)
1. **DRAFT_AMENDMENT_8.md** (the MM freezes it):
   * the arm spec above and the bank spec;
   * the read: card section 3 on RHK - B0, same cells, floors, estimator, tier and precedence;
   * outcome names, with a one-line answer to "does K's bank repair P2/P3 and hold G-T0?";
   * the next lever per outcome;
   * cost, ESTIMATED from the measured times in `/root/rollhist/rh_chain.log` (the bank step and the arms) and `rh_p03_chain.log`.
2. **Pod chain**, staged in `F:/TanitAD-scratch/rollhist/build/b9rhk/` (the MM copies it to the pod):
   * `rh_k_chain.sh`, modelled on `rh_p03_chain.sh` plus the bank step. It holds `/root/gpu_queue.lock` to the end and follows the marker protocol.
   * Plus its arm, check wrapper, argv renderer (only the bank flags change against RH's argv), marker watcher and launcher.
   * The study checker `dscale10k_check_rhp3.py` (sha256 dcee7cc1, in `build/b8p03/`) gains the kind RHK: dose pinned to exactly 0.1, bank path `bank_k/`, the readout token K in C-RH-BANKGEN and C-RH-PAIR. The old checkers stay untouched.
   * Tests: copy `build/b8p03/test/` and add RHK tests. Mutation arms must be refused: the RH bank under the RHK name; a C* bank under the RHK name; a second changed flag; a dose other than 0.1.
   * Note: the tests load their own checker copy in `test/`, so keep it synced.
3. **Read extension** in `F:/TanitAD-scratch/rollhist/read/`:
   * RH_AKIND=RHK, following how p03 did RH_AKIND=RHp3;
   * `rh_read_k_chain.sh` / `.cmd`;
   * `dry_check.py` must PASS under RH_AKIND=RHK and at the default.
   * Keep `.bk` copies of any file you edit. The PARTIAL and P03 reads must still re-run unchanged.
4. **DRY_MINT_PLAN.md** for the MM, as in b8p03. It runs before the launch, with the real bank_k after its bank step, or a dry form if the bank does not exist yet; say which checks that form cannot prove.

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp "$POD_TARGET:/path" local`)
Read and pull only. Never write there, run no python or pytest there, and never touch a process, lock or marker. Another session's jobs run on the pod.

## Dev box
* CPU only, one python process at a time, each gated on commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.
* TMP/TEMP go under your folder. Delete your own temp at the end, by literal path, and log it.
* F: must stay >= 6 GB. Never wait more than about 4 minutes in one call.
* Hand off (write `HANDOFF_RHK.md` and end) if your context passes about 150k.

## Deliverables
* `build/b9rhk/`: DRAFT_AMENDMENT_8.md, the chain, the checker and tests with logs, DRY_MINT_PLAN.md, HANDOFF_RHK.md (at most 800 tokens), MANIFEST_SHA256.txt.
* The read extension with its dry check.

Hand back in at most 250 words, with a deliverable manifest, and say plainly which checks PASSED and which did not run.
