# Brief: the SEED3 pod kit (B0_s2, RH_s2, KI_s2 on UE_s2.A, with an s2 bank), then the CPU prep on the pod BEFORE any lock. No GPU, no lock, no marker; do not launch the chain

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (`git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"Project Steering/AGENT_OPERATING_STANDARD.md"`);
* the card `F:/TanitAD-scratch/rollhist/seed3/PREREG_V8F_SEED3.md` (FROZEN, sha256 75f3b191). Sections 2, 3 (lever-acts) and 5 bind the kit;
* your models:
  * the KI kit in `F:/TanitAD-scratch/kinlab/pod_ki/`: `HANDOFF_POD_KI.md`, `launch_ki.sh`, `rh_ki_chain.sh`, `rh_ki_marker_watch.sh`, `ki_argv.py`, `ki_gate_record.py`, `ki_arm.sh`, `chk_arm_ki.sh`. On the pod it is at `/root/rollhist/kit_ki/`;
  * the RH chain that generated the s0 / s1 banks under readout causal_g1x and trained RH / B0 at cdbff971 (pod `/root/rollhist/`: `rh_chain.sh` and its kit; find its local copy with `grep -rl "rh_chain.sh" F:/TanitAD-scratch/rollhist --include=*.md | head`);
  * the RHK chain (`rh_k_chain.sh`), which generated banks under the lock and then minted and trained, for the bank-then-mint order.

## Rules
* Stage, commit and push nothing.
* No GPU. Never touch `/root/gpu_queue.lock` or `/root/N1DW_READY`. Do not launch the chain or its marker watcher.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, IPs or ports in any file. Never name a file `*token.json`.
* Pod access: `source C:/Users/Admin/lorapod/podenv.sh`, then `pssh` / `pscp` (scp syntax: `pscp local "$POD_TARGET:/remote"`). Wrap each call as `timeout N bash -c '...' < /dev/null`.
  * Long pod jobs go under `nohup setsid ... &`, and you poll their log. An ssh launch may hang; check the job is alive rather than relaunching it.
* **Permission refusals:** if a tool call is refused, do not retry it in another form and do not ask anyone else to run it. Record it and hand back.
* **The pod is shared with TanitADRuled** (its CPU jobs run at nice 19 until about 17:30Z).
  * Write ONLY under `/root/rollhist/kit_s3/`.
  * Touch no other folder's files and no other process.
  * Every pod CPU job runs at `nice -n 19` with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""`.
  * The container's total_rss (`/sys/fs/cgroup/memory/memory.stat`, total_rss) must be below 40 GB before each start.
  * One pod job at a time.

## What to build (in `F:/TanitAD-scratch/rollhist/seed3/pod_s3/`; bank each step)
1. **Facts first** (`FACTS_POD_S3.md`, each with its file:line or command):
   * the pod path of UE_s2.A, with its sha256 equal to `6d434525…` (86,970,901 B);
   * the commit and tree RH_s0 / B0_s0 trained at, read from their gate records under `/root/rollhist/runs/`. The card says cdbff971 / `tree_cdb`; confirm or report the difference;
   * the exact bank-generator invocation that made `/root/rollhist/bank/s0`, and the header fields K-B2 and coverage check;
   * the G-SUITE results that cover cdbff971 (RH's) and 4097a89e (`/root/rollhist/gsuite_result_ki.json`).
2. **The kit:**
   * **Argv tool:** renders B0_s2 from `runs/B0_s0/argv.json`, RH_s2 from `runs/RH_s0/argv.json`, and KI_s2 = RH_s2 + `--dagger-base logged`, each with `--out runs/<arm>/train`.
     * It admits exactly the card's substitutions (seed, dagger-seed, init, bank path and sha, out) and refuses any other change.
     * There is one regression case per refusal.
     * RH_s2 and KI_s2 render AFTER the bank exists, with its sha256.
   * **Gate-record tool** per arm, in the form of `ki_gate_record.py` / RH's. For KI_s2, include the card's KINPUT items.
   * **Bank step:** the s2 bank by UE_s2.A with RH's generator invocation. Its check: the generator record names UE_s2.A's sha256, and coverage ≥ 0.5.
   * **Chain `rh_s3_chain.sh`:**
     * Log `/root/rollhist/rh_s3_chain.log`. Last line `ZZ ROLLHIST-S3 DONE ZZ` or `ZZ ROLLHIST-S3 FAIL ZZ <why>`.
     * It validates in seconds before the lock: kit sha, trees, G-SUITE results, B0_s2 argv, UE_s2.A sha.
     * It takes `flock /root/gpu_queue.lock` on fd 9, holds it to the end, and hands it back on every exit.
     * Order: B0_s2 (selfcheck, mint, train, zerograd, post, summary), then the bank and its check, then RH_s2, then KI_s2, each with its own selfcheck and mint after the bank.
     * Log the lock start, so the MM's waiter can apply the 4.0 pod-h return line.
     * Keep a `DRY=1` mode that writes nothing.
   * **Marker watcher `rh_s3_marker_watch.sh`** in KI's form: noclobber own token at its start, cleared at the chain's "GPU queue lock held to the end of the chain" line, only if the content equals its token.
   * **`launch_s3.sh`:** refuses if any `rh_*_chain.sh` or `rh_*_marker_watch.sh` runs; prints `ZZ S3 LAUNCHED ZZ`.
   * **Tests on the dev box**, whole files; every mutant RED.
3. **Ship** to `/root/rollhist/kit_s3/` (new folder). Then on the pod CPU:
   * (a) blob-check `tree_cdb` at its commit and `tree_ki` at 4097a89e;
   * (b) the B0_s2 dry mint: the trainer's `--selfcheck-only` on CPU with B0_s2's argv, `--device cpu` the only difference, and the gate-record mint and verify on a scratch root;
   * (c) `DRY=1` walk of the chain.
4. Write `HANDOFF_POD_S3.md` (at most 600 tokens). It holds the exact launch command, the facts, the prep results, and the lock time projected from MEASURED walls (RHK bank 1 h 52 min; RK and KI about 0.26 pod-h per arm).

## Dev box
* CPU only, one python process at a time behind `F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* Nothing on C:. TMP and TEMP go under `F:/TanitAD-scratch/rollhist/seed3/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never touch `F:/TanitAD-scratch/rollhist/read/` (another agent works there).
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_POD_S3.md` with what is left.
* Delete your own temporary files by literal path when done, and log each one in `F:/TanitAD-scratch/rollhist/seed3/DELETIONS.log`.

## Priority order if you run short
1. facts;
2. the kit with its tests;
3. ship and the tree checks;
4. the B0_s2 dry mint;
5. the dry walk.

Hand back in at most 250 words, saying plainly what passed and what did not run.
