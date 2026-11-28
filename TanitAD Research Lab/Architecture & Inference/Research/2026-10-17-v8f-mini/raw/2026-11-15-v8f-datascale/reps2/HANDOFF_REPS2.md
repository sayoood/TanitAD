# HANDOFF_REPS2 (builder, 2026-11-22): UE_s2 + UE_s3 stage A on the LoRA pod. BUILT AND DRY-CHECKED, NOT LAUNCHED. Zero spend so far.

Kit: `F:/TanitAD-scratch/dscale10k/reps2/`. Facts: `FACTS_REPS2.md`. What ran / did not: `DRY_RUN_REPS2.md`. The pod holds NOTHING of REPS2 (dry dir deleted); the pod's existing files are untouched.

## What it is
UE_s0.A's 35-token argv with exactly `--seed 2|3` and `--out /root/dscale10k/runs/UE_s{2,3}/A`; same tree and code of record cdbff971, same data, same gate. The checker change is two ARMS rows plus one rule: the argv must equal `UE_s0.A.json` except `--seed` and `--out` (any other flag, another data manifest, the wrong seed under a name, UM or other names: refused; the reference file is bound in the token). Chain order: lock (fd 9, held to the end) -> prep (dry builds, init records, reach probes, evidence, mint + verify) -> per arm: pair check, token verify, token-refusing runner, selfcheck must read PASS, summary check, zero-grad post gate, bank to the private HF repo (sha read back), ARM_DONE.
Estimated (MEASURED UE_s1 walls: 255 s start + 11,561 s train + 25 s gates): 3.3-3.8 h per arm, **about 7-8 pod-h** with ~0.35 h of prep under the lock.

## Before launching (the MM)
1. **PI go on spend.** The PI's words for the fork were "Yes, two more seeds" (INHERITED, COORD_V8F line 372); the cost question (3-5 pod-h per seed guessed) goes back to him only if above that: the estimate is inside it.
2. **Disk.** The pod had 51-55 GB free; the chain refuses below 60 GB (`DS_MIN_FREE_GB`). A stage-A run dir is 84 MB, so free space or launch with `DS_MIN_FREE_GB=30` (MM decision; the builder did not lower the guard).
3. **Install** (new files only; refuses if `blockreps2/` exists): `bash F:/TanitAD-scratch/dscale10k/reps2/install_reps2.sh`. Then check `/root/dscale10k/blockreps2/PAIRING_PROOF.json` reads `all_pairs_pass` and `all_regressions_refused` true.
4. **Land the kit** into the repo under the reps1 pattern (a `reps2/` folder beside `reps1/`; the pod_copy checker is `pod_copy/dscale10k_check_reps2.py`, sha256 f8b3e509...). Nothing is staged or committed by the builder.
5. Marker: the watcher sets `/root/N1DW_READY` the moment it starts, which stops TanitADRuled's next slice until the chain holds the lock. If the RA chain still holds the lock, start the watcher only when you want Ruled to yield (cap 30 h).

## Launch (each as its OWN ssh command; `source C:/Users/Admin/lorapod/podenv.sh`)
1. `pssh 'nohup setsid bash /root/dscale10k/blockreps2/reps2_marker_watch.sh < /dev/null > /dev/null 2>&1 &'`
2. `pssh 'cd /root/dscale10k && nohup setsid bash /root/dscale10k/blockreps2/blockreps2.sh < /dev/null > /root/dscale10k/blockreps2.launch.out 2>&1 &'` (add `DS_MIN_FREE_GB=30` before `nohup` if you choose that)
Log `/root/dscale10k/blockreps2.log`; last line `ZZ DSCALE10K REPS2 DONE ZZ` or `... FAIL ZZ <why>`; per arm `ZZ ARM UE_s2 DONE ZZ`. Stop between steps: `touch /root/dscale10k/chains/STOP`. **Resume after any kill or reboot: run the same two commands again** (finished arms skipped, a killed run resumes from `ckpt_last.pt`, never retrained over). A refusal exits at once and frees the lock; the watcher then clears the marker.

## Delivery (dev box, after `ZZ ARM <arm> DONE ZZ`; not run by the builder)
`bash F:/TanitAD-scratch/dscale10k/reps2/deliver_reps2.sh UE_s2 UE_s3` -> `/workspace/v8proof/integ10k/ckpt/UE_s{2,3}/A/{ckpt_last.pt,arm_record.json,gate_token_copy.json}`, sha256 pod = dev box = fork pod, never overwrites a differing file. The 2026-11-19 route was the banker's private HF copy (kept); the relay is added because the fork pod's HF access was not verified.

## Do not forget
* The dry mint used UE_s1's evidence as a stand-in and still read PASS: the checker does not bind evidence to its arm. The real prep builds seed-2/3 evidence and the assembler refuses another arm's evidence; still, read `gate/init_ue_s{2,3}.json` and the two minted `checks.json` before the chain reaches training (they are written in the first ~25 min of the lock).
* Re-run `test_reps2.py` once (whole file, command in its docstring): the last run (39 passed, 16/16 mutants RED) pre-dates the assembler guard and its test; the dev box's commit gate never opened for the final run.
* The reads (UE_s2 / UE_s3 against UE_s0 / UE_s1) are for a fresh read agent; this build reads no evaluation number.
