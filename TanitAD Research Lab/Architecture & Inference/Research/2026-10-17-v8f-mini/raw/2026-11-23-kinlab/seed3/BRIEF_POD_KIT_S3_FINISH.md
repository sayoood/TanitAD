# Brief: FINISH the SEED3 pod kit (a previous build agent stopped at its context limit). Tests, mutants, pins, ship, CPU prep. No GPU, no lock, no marker; do not launch the chain

You are a TanitAD build agent working for the Master Mind (MM). First read, in this order:
1. `F:/TanitAD-scratch/rollhist/seed3/BRIEF_POD_KIT_S3.md` — the original brief. Its **Rules** and **Dev box** sections bind you unchanged: pod sharing, nice 19, rss < 40 GB, write only under `/root/rollhist/kit_s3/`, no lock or marker, no launch, permission refusals.
2. `F:/TanitAD-scratch/rollhist/seed3/pod_s3/HANDOFF_POD_S3.md` — what is built and what is left, in order.
3. `F:/TanitAD-scratch/rollhist/seed3/pod_s3/FACTS_POD_S3.md`.
4. The card `PREREG_V8F_SEED3.md` (75f3b191) and **`PREREG_V8F_SEED3_AMENDMENT_1.md` (0d1d9226)**, both in `F:/TanitAD-scratch/rollhist/seed3/`. B0_s2 and RH_s2 train at **97fe0a56** on `tree_cdb`; KI_s2 at 4097a89e on `tree_ki`.

## What to do (in this order; bank each step in `pod_s3/`)
1. **Tests not yet run:**
   * Run `test_s3_marker_watch.sh` ALONE, with nothing else running. It hung before only because the chain test ran at the same time. If it hangs alone, find out why; do not mark it passed.
   * Run `make_mutants_s3.py` over the argv tool, the gate-record tools, the bank check, the chain and the marker watcher. Every mutant must be RED; fix the source for any survivor and re-run.
   * Re-run every test file whole on the final files.
   * Record the results in `TESTS_S3.md`, with the commands, pass counts and mutant table.
2. **Check `U_s2.B.tmpl.json`** (the kit generated it; none existed on the pod). Diff it against `/root/dscale10k/argv/U_s1.B.tmpl.json` and `U_s0.B.tmpl.json`. Every difference must be a seed substitution the card admits (seed, dagger-seed, init); record the diff. Also show the argv tool's rendered B0_s2 / RH_s2 skeleton against the s0 argv, with only the admitted substitutions.
3. **Pins, SHA256_S3.txt, ship, prep:** the handoff's items 1 to 4 exactly.
   * The SHARED pin files come from the pod.
   * `SHA256_S3.txt` is computed over the kit files.
   * Ship to `/root/rollhist/kit_s3/` and run `sha256sum -c`.
   * Blob-check both trees.
   * B0_s2 dry mint: `bash kit_s3/s3_dry_mint.sh` at nice 19. Record its selfcheck values: `declared_vs_built`, `live.pass`, the gate-record mint and verify rc.
   * `DRY=1 RH_READOUT=causal_g1x bash kit_s3/rh_s3_chain.sh` must print `ZZ ROLLHIST-S3 DRY PASS ZZ`.
   * Fix whatever the dry mint finds in `dscale10k_check_s3.py`, then re-run its tests and mutants.
4. **Clean up:** delete your own scratch, including the earlier agent's `F:/TanitAD-scratch/rollhist/seed3/tmp`, by literal path, and log each deletion in `F:/TanitAD-scratch/rollhist/seed3/DELETIONS.log`.
5. **Rewrite `HANDOFF_POD_S3.md`** (at most 600 tokens) with:
   * the exact launch command;
   * the facts;
   * the test and mutant results;
   * the pin and ship results;
   * the dry-mint values and the dry walk line;
   * the projected lock: bank 6,843 s plus three arms at about 0.26 pod-h each, about 2.7 pod-h, with a return line of 4.0 pod-h.

Hand back in at most 250 words, saying plainly what passed and what did not run.
