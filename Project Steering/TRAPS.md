# TanitAD traps catalogue (moved verbatim from CLAUDE.md on 2026-10-11, token-budget fix P6)

Each entry cost hours more than once. CLAUDE.md keeps a one-line index; this file is the full text. Read the relevant entry BEFORE touching git, Thor/pods, memory probes, landings or data artifacts.

## Traps preflight (each of these has cost hours more than once)

- ⛔ **A POLLING MONITOR WHOSE FILTER CONTAINS THE PATTERN IT SEARCHES FOR WILL MATCH ITS OWN
  ECHOED COMMAND — and report a failure that never happened.** MEASURED THREE TIMES this session,
  most recently 2026-08-12: a monitor grepping a pod log for `"Traceback|CUDA out of memory"` fired
  `W7PROG: Traceback CUDA out of memory` while the run was **healthy and 3 minutes in** (8 930 MiB
  GPU, progressing) — the interactive PTY **echoes the command line back**, so the literal pattern
  text appears in the output stream and the client-side filter matches it. This is the same family
  as the `pgrep -f` trap below, in a monitoring costume, and it is worse because it invents a
  **false failure** rather than a false absence. ⇒ **Make the emitted token disjoint from the
  searched token**: compute counts pod-side and emit an opaque marker
  (`echo "ZZ${done}-${errs}-${arms}ZZ"`), then parse `ZZ…ZZ` client-side. Never grep the raw
  stream for the same words your command contains.
- **`pgrep -f <trainer>` / `pkill -f <trainer>` self-matches your own ssh command** and kills your
  session — returns empty output and looks like nothing happened. Kill by **explicit PID**.
- **`PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED** on pods or trainers die with
  `ModuleNotFound: tanitad`. `cd` alone is not enough.
  ⚠️ **The same trap wears a dev-box costume: the venv's EDITABLE `tanitad` install points at the
  G: mount.** MEASURED 2026-08-18 with G: down: any import of `tanitad` from the off-Drive clone
  dies with `Errno 22` at pytest collection — the suite cannot run in the clone at all without
  `PYTHONPATH=<clone>/stack`, and the death looks like a broken suite, not a broken mount. Set it
  for every clone-side suite run and re-check which tree actually got imported
  (`python -c "import tanitad; print(tanitad.__file__)"`).
- **After deleting a large file on the VAST `/workspace` volume, writes can fail with `Disk quota exceeded` for a short window** — the quota is reclaimed asynchronously. MEASURED 2026-10-17: a 19 GiB `dd` write test (the probe the next entry prescribes) succeeded and was deleted, and the very next small write failed with EDQUOT, then succeeded within seconds. ⇒ size the write test well below the headroom you need, and retry a write before launching a job straight after a big delete.
- **On the dev box, a job can die at an arbitrary TINY allocation, and the traceback names the victim allocation, not the cause.** MEASURED 2026-10-20 (SH5 grid 5): numpy failed to allocate 2.25 MiB, and two minutes later CUDA failed on 20 MiB with 3.58 GiB reported free. C: was at about 0.1 GB at the time. The likely cause is the host commit limit (RAM + pagefile, and C: full means the pagefile cannot grow). It is INFERRED, not verified: at 04:10 commit read 23.1 GB free of 67.7. ⇒ Before blaming the code, the GPU or a leak, record `Get-PSDrive C` and `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory` IN THE SAME BREATH as the failure; on a WDDM GPU a CUDA OOM with GiBs free is a host-side event. (`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-sh5-grid5/RESULT_GRID5.md` §6)
- **`ssh pod 'cd X && setsid nohup cmd > log 2>&1 < /dev/null & echo $!'` NEVER RETURNS** — the `&` backgrounds the whole `&&` list, so bash forks a SUBSHELL to run it, and that subshell keeps the ssh channel's stdout open for as long as `cmd` lives (the redirect applies to `cmd`, not to the subshell). MEASURED 2026-10-17: the free-point starter hung at its worker-launch step for 29 min; the worker was healthy and idle, and the uploader that should have followed never started. ⇒ launch with `setsid -f nohup cmd > log 2>&1 < /dev/null` as its OWN statement (`cd X; setsid -f …`), or wrap the list: `( cd X && cmd ) > log 2>&1 < /dev/null &`. Test: the ssh call must return in seconds. And killing the stuck local starter does NOT kill a `setsid nohup` worker on the pod (verified the same day).
- **The A40 pod's container is capped at 11.05 CPUs and 71.0 GB of memory, and `nproc` / `free` report neither** — they read the HOST (96 CPUs, 503 GB). MEASURED 2026-10-17: `cpu.cfs_quota_us` 1,105,000 / `cfs_period_us` 100,000; the same 74-twin rollout control took 13.4 s per rollout at 10 processes and 49 s at 32, so throughput stops at ~11 processes. `memory.limit_in_bytes` = 70,999,998,464; a 12-worker `build_metric_cache.py --lane-rank 2` build drove `memory.max_usage_in_bytes` to exactly that limit and the kernel OOM-killed it (`memory.oom_control` oom_kill 2), with no Python traceback — the log simply stops. ⇒ size worker pools from `/sys/fs/cgroup/cpu/cpu.cfs_quota_us` and `/sys/fs/cgroup/memory/memory.limit_in_bytes`, never from `nproc` or `free`; and when a pod job's log stops without an error, read `memory.oom_control` before anything else. Same family as the Thor memory-probe entry above: a probe answers for the layer it reads, not the layer you run in.
- **Never judge pod disk with `df`.** It reports the 965 TB cluster and hides the per-pod MooseFS
  quota. Use a real `dd` write test. A full quota killed the flagship mid-checkpoint.
- **`step_s` in trainer logs is ACCUMULATED over `--log-every`** (÷50), not per-step. This has
  caused false "training is 430 s/step" alarms.
  ⛔ **BUT THIS RULE IS INVERTED FOR `train_v6_staged.py` — THE LIVE v6 TRAINER — AND APPLYING IT
  THERE PRODUCES A 50× UNDER-ESTIMATE.** MEASURED 2026-08-18: `train_v6_staged.py:3034` computes
  `step_s` as `(time.time() - t0) / max(step - start_step, 1)` — **already divided** — and its own
  comment at `:3032` says it is written that way *precisely to prevent* the 430 s/step alarm. It also
  emits `step_s_note` naming the divisor. ⇒ Dividing again turns the live run's true **26.47 s/step**
  into **0.53**. ⇒ **CHECK WHICH TRAINER WROTE THE LOG BEFORE APPLYING EITHER READING** — this is the
  `df`/`free`/cgroup family in doc form: a rule that is true somewhere, quoted where it does not
  apply. A per-run fact belongs beside the per-run divisor, and this log carries one.
- **Moving multi-GB files between pods: POD→POD DIRECT SSH WORKS — the long-standing "pods cannot
  SSH each other" is RETRACTED (C56).** MEASURED 2026-07-28: **42 MB/s cross-datacenter**
  (US-TX-1 → ca-mtl-1), 3,415,808,330 B in 77 s — **42× the ~1 MB/s dev-box relay**, and it does
  not depend on HF (which has been 403-storage-full for days). Recipe:
  1. `ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519` **on the DESTINATION**;
  2. append that pod's **PUBLIC** key to the SOURCE's `~/.ssh/authorized_keys`
     (⛔ never copy a private key — that is correctly classifier-blocked, and you never need to);
  3. connect to the source's **direct** mapping — `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`, read
     from the source's own env.
  ⚠️ **The env-advertised direct mapping can be DEAD while the pod is healthy.** MEASURED
  2026-08-11: BOTH pod4 and pod5 refuse connections on their own `$RUNPOD_PUBLIC_IP:
  $RUNPOD_TCP_PORT_22` (Connection refused; sshd up and listening on :22; both directions
  tried) — stale mappings after migrations/restarts, and no `.runpod.internal` DNS on this
  account. ⇒ **Probe with `ssh -n ... 'echo OK'` BEFORE building a transfer on the direct
  path, and fall back to the HF relay (shard tar + MANIFEST-last protocol, ~118 MB/s) when
  it refuses.** The C56 42 MB/s measurement was on a different pod pair.
  ⚠️ **Use the DIRECT port, not the `ssh.runpod.io` proxy.** The proxy genuinely cannot move files
  (sftp → `subsystem request failed on channel 0`; `scp -O` → exit 2); it serves an interactive
  shell only. *That* is the true limitation the old rule had generalised into "pods cannot SSH".
  ⚠️ **A nested `ssh` inside a piped script EATS THE REST OF THE SCRIPT'S STDIN** — the tail silently
  never runs and looks like a hang or a truncated log. Always `ssh -n` (or `< /dev/null`) inside
  a heredoc/pipe. This cost two debugging rounds here.
  HF push/pull (~118 MB/s) remains the fastest path **when HF has quota**; verify md5 either way.
- **A RunPod volume resize stops the pod and reassigns its SSH port** (`Connection refused`, not
  `timed out`). The working key is `~/.ssh/tanitad_pod`, not the console's `id_ed25519`.
- **`torch` spawns ~113 threads PER PROCESS, and concurrent eval arms then make NO PROGRESS —
  it looks exactly like a hang.** MEASURED 2026-07-27: 7 concurrent arms sat at GPU `sm` **0–6 %**
  for **50 minutes** with zero progress; `OMP_NUM_THREADS=6` and the same arm finished in **232 s**.
  Set it before any multi-arm panel; do not diagnose the symptom as a deadlock or a dead job.
- **`git add <exact path>` can SILENTLY NO-OP on a file in a newly created directory** — exit 0,
  the usual CRLF warning printed, and the file **not staged**. MEASURED 2026-07-27: renaming the
  parent dir (the previously documented un-poison) **did NOT help**; rewriting the file through a
  **fresh inode from Bash** did. ⛔ **`git add` exit codes are not evidence. Verify with
  `git ls-files --cached <path>`** — and note the sibling rule that `git status --short` scoped to a
  path can also mislead, so `git ls-files --stage` is the check for what is really in the index.
  ⚠️ **SHARPENED 2026-08-16 — `--cached` is NOT sufficient for a file that was ALREADY TRACKED and
  then MODIFIED.** It answers *"is this path in the index?"*, which is **yes** for any tracked file,
  **including one whose index blob is still the PRE-EDIT version**. MEASURED: an agent's edit read as
  "staged" while the index held the old content. ⇒ For a modified tracked file the check is a **BLOB
  COMPARISON** — `git ls-files --stage <path>` (index blob) against `git hash-object <path>`
  (worktree blob); equal means the edit is really staged. `--cached` remains the right check only
  for a NEW file, where presence and content coincide.
  ⚠️ **And with several agents live, RE-VERIFY AT THE END OF THE TURN.** MEASURED 2026-08-16: five
  files were staged and verified, three orchestrator commits then landed, and the same check
  afterwards reported them `??`. Staging is not a latch — the index moves under you.
  ⛔⛔ **AND A COMMIT IS NOT A LATCH EITHER — MEASURED THREE TIMES INDEPENDENTLY ON 2026-09-03.**
  A commit that has *landed* can be silently removed from the branch by a sibling whose scratch
  index was seeded from a tree that **predates** it. The replacement still points its `-p` parent
  at your commit, so `git log` shows your subject sitting in the history **while its file content
  is gone from HEAD**. Three cases in one night: one agent's five paths reverted under a parent
  that named them; a second agent's whole commit vanished when a third replaced the tip; and my
  own doctrine commit lost `PROGRAM_OVERVIEW.md` and a retraction-log entry, with the *other three*
  files from the same commit surviving — so even a partial survival proves nothing about the rest.
  ⛔⛔ **AND THE BLOB COMPARISON ITSELF HAS A HOLE: BOTH OPERANDS CAN FAIL IDENTICALLY, AND THE SHELL THEN REPORTS A MATCH.** MEASURED 2026-09-04, and it made me report a commit as landed that had not landed. `a=$(git rev-parse HEAD:<path>)` and `b=$(git hash-object <path>)` BOTH return the **empty string** when the mount is in an outage window; `[ "$a" = "$b" ]` is then **true**, and the check prints its success message. HEAD held `887e9aab` while the worktree held `507ba1ad`. ⚠️ This is the *positive-assertion* rule defeating itself: the assertion was positive in FORM but its two halves came through the SAME broken channel, which is the same error as taking "multiple samples" through one failing probe. ⇒ **ASSERT THE SHAPE BEFORE COMPARING: require BOTH sides to be 40 characters, and report INCONCLUSIVE — never MATCH — when either is not.**
  ```
  a=$(git rev-parse "HEAD:$p" 2>/dev/null); b=$(git hash-object "$p" 2>/dev/null)
  if [ ${#a} -ne 40 ] || [ ${#b} -ne 40 ]; then echo "INCONCLUSIVE"
  elif [ "$a" = "$b" ]; then echo "VERIFIED $a"; else echo "MISMATCH"; fi
  ```
  The same hole exists in every `grep -c` verification: a count of `0` from a file that could not be READ is indistinguishable from a genuine absence, so pair it with a same-breath control that must read non-zero.

  ⛔⛔ **AND `$?` AFTER A PIPELINE IS THE *LAST* ELEMENT'S STATUS — SO `cmd | tail` REPORTS
  `tail`'S SUCCESS AND HIDES `cmd`'S FAILURE. MEASURED TWICE ON 2026-09-07, IN TWO INDEPENDENT
  STREAMS, ON TWO DIFFERENT TOOLS.**
  * `mm_commit.py` exhausted its retries and exited **1** (`git add kept failing`); the wrapper
    was `python mm_commit.py … | tail -25`, which reported **exit 0**. I read that as the TOOL
    manufacturing a false success and was about to "fix" a tool that had been honest — the
    shell had lied, not the program.
  * A 25-minute `timeout` killed `pod_currency_audit.py` with **zero lines of output and no
    `--json` file written**, and its wrapper printed **`GATE_EXIT=0`**. A timed-out, output-less
    **pre-launch gate** reported clean.
  ⇒ **Never read `$?` through a pipe.** Use `${PIPESTATUS[0]}`, `set -o pipefail`, or redirect to
  a file and read it afterwards.
  ⭐ **THE STRONGER FORM, AND THE ONE THAT GENERALISES: ASSERT ON THE ARTIFACT, NOT THE STATUS.**
  *"The admissible evidence that this gate did not run is the MISSING JSON, never the exit code."*
  A status code is a claim by the process about itself, routed through a shell that can overwrite
  it; the artifact is the thing you actually wanted. This is the `df` / cgroup / `step_s` family
  once more — **a success criterion disconnected from the thing being checked** — and both
  instances were caught only by reading CONTENT (an empty output, an absent file) rather than a
  number that said everything was fine.
  ⇒ **`git log` is NOT evidence that your change is in HEAD.** The check is a positive content
  assertion per file — `git show HEAD:<path> | grep -c <marker>`, or a blob comparison
  (`git rev-parse HEAD:<path>` vs `git hash-object <path>`) — run **at the end of the turn**, on
  **every** path you committed. ⭐ **The mechanism is a slow `read-tree`, and the fix is cheap:**
  a scratch index **on the G: mount** takes **13 minutes** to seed under ~97 concurrent git
  processes, which is a 13-minute window for another agent to land a commit inside; the *same*
  `read-tree` with the index **on local disk** is **3m39s**. Put `GIT_INDEX_FILE` on local disk,
  re-read `HEAD` **immediately before** `commit-tree`, and make the final write a
  **compare-and-swap** (`git update-ref HEAD <new> <old>`) so a lost race FAILS instead of
  clobbering. ⛔ A plain `update-ref HEAD <new>` is how a sibling's work disappears — it cannot
  detect that HEAD moved. *(`stack/scripts/scoped_commit.py` reads HEAD **after** `write-tree`,
  so its window is the entire tree build; its stray-path guard then refuses forever under
  contention. Correct, but livelocked — it needs the CAS and the local-disk index.)*
  ⛔⛔ **AND THE CAS GUARDS THE REF, NOT THE TREE — BOTH RULES ABOVE PASS WHILE YOU REVERT A
  SIBLING.** MEASURED 2026-09-06, caught **before** landing. `read-tree` into a local-disk scratch
  index took **65 s**; inside that window a sibling landed `4be8bef`, rewriting the exact path I was
  about to write. Re-reading `HEAD` immediately before `commit-tree` then does the *opposite* of
  what it looks like: it staples a **current parent pointer** onto a **stale tree**, and the
  compare-and-swap **SUCCEEDS** — because HEAD is precisely what you just read. The commit lands,
  `git log` looks clean, and the sibling's content is gone. This is the 2026-09-03 vanishing-commit
  mechanism *from the other side*: those three losses were not caused by a missing CAS, so adding
  one does not prevent them.
  ⇒ **After `write-tree`, diff YOUR TREE against CURRENT HEAD by positive per-path blob assertion,
  and re-seed if anything moved:**
  * for every path you did **not** name: `git rev-parse <tree>:<p>` **must equal** `HEAD:<p>`;
  * for every path you **did** name: your blob must be a **descendant** of what HEAD holds — it
    contains HEAD's content **plus** your change. ⛔ "differs from HEAD" is not the test; a revert
    also differs. The cheap descendant check is a line count plus a content marker from the version
    already in HEAD, both of which must survive in yours.

  ⭐ **The discriminating control is what caught it, and it is the cheap half.** I asserted my new
  blob ≠ `HEAD:<path>` and read back the blob I had recorded as the **worktree** version minutes
  earlier — i.e. HEAD already held the newer content and my "improvement" was a revert of 54 lines.
  A positive assertion alone (*"my marker is present in my tree"*) passes happily on a revert,
  because the marker was in **both** versions. ⚠️ Same family as the empty-string blob-comparison
  hole above: an assertion that is positive in FORM can still be blind to the failure that matters.

  ⛔⛔ **A CHECK THAT SHARES THE DEFECT IT CHECKS FOR IS GREEN FOREVER — FOUR INSTANCES IN ONE
  NIGHT, AND EACH ONE GUARDED SOMETHING THAT WAS ALREADY BROKEN.** MEASURED 2026-09-07:
  * A label builder rounded its ladder to 4 dp (`round(k / 3.6, 4)` → `13.8889` for a step that
    is `13.888888…`), then verified the shipped buckets **against that same rounded ladder**.
    Re-snapping read **0 %** moved and the artifact passed its own build gate. Against the ladder a
    consumer would naturally derive — from the km/h integers — **57.5 % of the corpus (2,631/4,572)
    moves one step up.** ⭐ *Internally consistent and externally wrong.*
  * A census emitted `status = "UNDERPOWERED" if t in TACTICAL_GOAL_UNDERPOWERED or n < 30`, and a
    test pinned the set **against that status** — a union with the set being checked. It could
    detect a missing token and was **permanently blind to an extra one**, which is how a member
    that was not a v7 token at all survived.
  * `test_rl_channel_guard.py` asserted *"if missing: expect refusal; else: expect pass"* — an
    expected value of **whatever the code does**. It was green while the preflight it guarded was
    telling operators to plumb the label.
  * A conditioning test used a **flat two-attribute config stub** to check a map whose real fields
    are **nested**, and `hasattr` on the class. Doubly inert: it iterated entries that skipped the
    loop body, and could not have seen the fields even if it had.
  ⇒ **A CROSS-CHECK MUST BE DERIVED INDEPENDENTLY OF THE VALUE IT CHECKS.** Re-running the
  producer's own derivation and finding agreement measures **determinism, not correctness** — the
  same family as repeated samples through one channel being one sample. ⭐ The cheap discriminators,
  in order of strength: an **analytic target** (a circle's curvature is exactly `1/R`; a
  straight-line plan's is exactly `0`); an **independently authored** reference (road-law step
  values, not a histogram fit); a **mutation** that reintroduces the real historical defect and
  must go RED. ⛔ And write the expectation as a **literal**, never as an expression over the code
  under test.
- ⛔ **PODS HAVE NO GIT CREDENTIALS — `git fetch` on a pod HANGS (not fails), and the checkout's
  HEAD is ancient.** MEASURED 2026-08-11: pod5 HEAD sat at `6d714ad` (weeks old) while its working
  tree was fully current — every pod-side script this campaign arrived by md5-verified FILE-SHIP,
  never by git. A chain step doing `git fetch && git checkout -B <branch> origin/<branch>` on a pod
  therefore (a) hangs on the fetch, and (b) if the fetch fails but the checkout runs, RESETS the
  tree to the ancient commit, destroying the shipped files. ⇒ **Never put git sync in a pod chain.
  Ship files (xz+b64 PTY push, per-file md5) or the HF stack-tar relay, and grep-verify the
  specific fix is present before any launch.** (The verify-gates caught this: three chains
  refused to run stale code rather than running it.)
- **A pod's `stack/` checkout drifts silently and a launch from it resurrects fixed bugs.** MEASURED
  2026-07-27: pod2 sat at `0f93b98` while the v5 gate fix was at HEAD — **a v5 launch from that pod
  would have restored the crash the fix removed.** ⇒ **syncing the pod and verifying with a real
  `import` is a RUNBOOK STEP before any launch**, not a nicety; `git log` on the pod is not proof.
- ⛔ **On the Jetson Thor, THREE standard memory probes all lie — and they lie in BOTH directions.**
  MEASURED 2026-08-03: `torch.cuda.mem_get_info()` reported **3.4 GB free while 60 GB was allocated
  AND WRITTEN**; `free`/`tegrastats` showed **106 GB "used" on a completely idle box** and moved only
  **+596 MB** for those same 60 GB; `/proc/self/status` `VmRSS` read **0.62 GB against 24 GB
  allocated**. On unified memory these numbers are not merely imprecise, they are unrelated to the
  question. ⇒ **Only in-process `torch.cuda.max_memory_allocated()` is admissible on Thor.**
  *(This cost a wrong answer that was caught: a batch-64 OOM was read as "the ceiling is 32" while a
  phantom 100 GB sat on the box; the true ceiling is 64.)* Same class as the `df` trap above —
  **a probe that reports the wrong scope is worse than no probe, because it looks like an answer.**
- ⚠️ **Thor inverts both A40 batching instincts.** Throughput is **flat at 12.3–14.1 windows/s across
  a 6× batch range** — the 20 SMs saturate at **batch 8**, so a bigger batch buys nothing and only
  costs memory. ⇒ **small batches** — and the live v6F run uses `--batch 8` for exactly this reason.
  ⚠️ **CORRECTED 2026-08-16 — the "~8.6 GB per dataloader worker" half DOES NOT BIND on this
  trainer, and I cited it tonight as though it did.** `train_v6_staged.py` spawns **ZERO** DataLoader
  workers; the episode cache is read in-process. **The real host-RAM knob is `--v2-lru`** (the live
  run carries `--v2-lru 64`; the chain uses `6`). The worker figure was MEASURED on a different
  trainer and is true *there* — it is a **scope** error to quote it here, the same family as reading
  `df` on a pod. ⇒ **Before citing a per-worker cost, check the trainer actually HAS workers**;
  `cuda_max_mem_gb` is now logged per step, and on Thor only `torch.cuda.max_memory_allocated()` is
  admissible for device memory anyway.
- ⛔ **`memory.usage_in_bytes` (cgroup v1) COUNTS RECLAIMABLE PAGE CACHE — it is NOT memory
  pressure.** MEASURED 2026-08-03 on `tanitad-new` **with nothing running at all**: `usage_in_bytes`
  **37.2 GB of a 50 GB cap (74 %)**, of which `cache` **37.0 GB** and `rss` **0.1 GB**. Under load it
  read **98–100 %** while `rss` was **4.9 GB**. ⇒ **Read `memory.stat`'s `rss`, and read
  `memory.failcnt` / `memory.events` — a cgroup that has never hit its limit reports `failcnt 0`,
  and that is the fact that settles it.** *(Cost: I read 98 % as an imminent OOM and restarted the
  v5f trainer THREE times on it, losing ~40 min of training and inventing a container-OOM diagnosis
  that `failcnt 0` refutes.)* **Exactly the `df` trap and the Thor `free`/`tegrastats` trap in a
  third costume: a counter that aggregates something reclaimable, read as pressure.**
- ⚠️ **`supervise_run.sh` SOURCES ITS MANIFEST ONCE, at supervisor startup — not per relaunch.**
  Editing `runs.d/<run>.env` under a live supervisor changes nothing: it replays the `TRAIN_CMD` it
  captured when it booted, and the relaunch looks successful while running the OLD config. ⇒ **To
  change a supervised run: edit the manifest → kill the SUPERVISOR first → kill the trainer →
  start a fresh supervisor.** Killing the trainer first just makes the supervisor restore the stale
  command. **Verify by grepping the flags out of the RUNNING process**, never by reading the manifest.
- ⚠️ **A supervisor whose run never wrote its DONE-MARKER will RESURRECT the finished run the
  moment whatever made its relaunches crash gets fixed.** MEASURED 2026-08-11: the v5f run finished
  2026-08-09 but `summary.json` was never written; its supervisor kept relaunching for 2 days, each
  attempt insta-crashing on the `kin_weights` NameError — until the fix was synced to the pod, at
  which point a relaunch SUCCEEDED, resumed from a stale `ckpt.pt`, and started overwriting
  `config.json`/`metrics.json`/`ckpt.pt` in the canonical run directory while burning GPU next to a
  live eval. ⇒ **When a supervised run completes, write the done-marker (`summary.json` with
  `"done": true`) in the SAME turn** — it is also the correct remote off-switch: writing it made the
  supervisor exit cleanly ~20 min later with no kill needed. Backups live in
  `/workspace/experiments/v5f-30k-SAFE/` (md5-verified `ckpt_30k_final.pt`). A silent-crashing fix
  landing IS the trigger, so audit `ps` for supervisors after shipping any trainer fix.
- ⚠️ **Restarting a supervisor immediately after killing the old one RACES ITS `flock`** — the new
  one prints *"another supervisor holds …lock — exiting"* and dies, leaving **nothing running** while
  the log looks like a normal startup. Wait until the old supervisor **and** trainer are actually
  gone (poll `ps`), then start. If a lock is left behind with no holder (scan `/proc/*/fd`), it is
  debris — `rm` it. Same shape as the stale `.git/index.lock` rule below.
  ⛔⛔ **AND THERE IS A SECOND, WORSE MECHANISM — THE SPAWNED TRAINER INHERITS THE LOCK FD AND HOLDS
  IT FOR ITS ENTIRE LIFE.** MEASURED 2026-09-02, and it made the run **unsupervised for ~3 min while
  I believed I had restarted supervision.** `exec 200>"$LOCK"; flock -n 200` puts the lock on fd
  200; a child launched as `nohup python3 … &` **inherits every open fd**, so the *trainer* ends up
  holding the supervisor's lock. Scanning `/proc/*/fd` named the holders as the trainer PID **and
  its five dataloader workers** — not a supervisor at all. ⇒ once the old supervisor is killed the
  lock is **never released while its trainer lives**, so *no* replacement can ever start, and the
  documented advice above ("wait until the old is gone, then start") **cannot succeed**. It is not a
  race in that state; it is a permanent block that merely *looks* like a race, and the `.out` file
  says the same reassuring sentence either way.
  ⇒ **Launch children with the lock fd CLOSED: append `200>&-` to the trainer's redirection list**
  (`… >> train.log 2>> train.stderr.log 200>&- &`). To recover a live run that is already in this
  state, do **not** kill the trainer: point the supervisor at a **fresh lock path** so it can
  supervise the process holding the old one.
  ⛔⛔ **AND `200>&-` ON THE TRAINER ALONE IS NOT ENOUGH — I SHIPPED THAT PARTIAL FIX AND IT FAILED
  THE SAME DAY.** MEASURED 2026-09-02 on `sup_refav1.sh`: after killing the supervisor and its
  trainer, the lock was STILL held — by **`sleep 180`**, the supervisor's own poll child, which
  had outlived its parent and inherited fd 200. ⇒ **EVERY child inherits the lock fd, not just
  the interesting one.** Put `200>&-` on the `sleep`s too — on every command the supervisor
  spawns. ⚠️ And when you patch them with `sed`, an anchored `^    sleep 120$` will MISS a
  `sleep` that sits inline after a `;` — I left exactly that one unfixed on the first pass.
  ⭐ The diagnostic that settles it in one line is `/proc/*/fd` with the holder's cmdline:
  `for p in /proc/[0-9]*/fd/*; do [ "$(readlink $p)" = "<lock>" ] && tr '
' ' ' < /proc/$(echo $p|cut -d/ -f3)/cmdline; done`
  — it names the holder, and the answer has twice been a process nobody suspected.
  ⚠️ **And never `sed -i` a supervisor script while it is running** — bash reads a script lazily by
  byte offset, so an in-place edit can make a live shell execute garbage from the middle of a line.
  Edit only while it is stopped; otherwise write a new file and switch to it.
  ⭐ **The check that would have caught it in one line:** after starting a supervisor, assert it is
  actually there — `ps -eo args | grep -c 'sup[_]refcv3'` — instead of trusting the launch. Every
  failure in this family reports success and leaves nothing running.
- ⛔ **`uv pip install <anything>` CAN SILENTLY REPLACE TORCH WITH A WHEEL THE DRIVER CANNOT RUN.**
  MEASURED TWICE on pod4 2026-08-11/12: `uv pip install -U accelerate` and then, 20 minutes later,
  `uv pip install "compressed-tensors>=0.15.0"` each resolved **torch from the default PyPI index**
  and landed **torch 2.13.0+cu130** on a **CUDA-12.8 driver (570.195.03)**. Result:
  `torch.cuda.is_available()` **False**, and every GPU job on the pod dies. Neither command names
  torch — it arrives through the dependency closure (`accelerate`, `compressed-tensors` both
  require it). ⇒ **Install torch-dependent packages with `--no-deps`, and (re)install torch from
  the pinned index LAST so nothing can drag it forward again**:
  `uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
  "torchvision==0.23.0"`. **Verify with a real `conv2d` on CUDA, not with `import torch`** —
  cuBLAS/matmul can succeed while cuDNN/conv is broken.
- ⚠️ **`CUDNN_STATUS_NOT_INITIALIZED` on a healthy pod usually means CUDA NEVER INITIALISED —
  it is NOT evidence of a cuDNN version conflict.** MEASURED 2026-08-12: I read that error as
  `nvidia-cudnn-cu13` shadowing `-cu12`, purged the cu13 wheels, and **removed the cuDNN torch
  actually needed** (`ImportError: libcudnn.so.9`) — turning a one-command fix into three rounds.
  The real cause was the cu130 torch above. ⇒ **Fix the torch/driver pair first and purge nothing;**
  if a reinstall is needed use `--reinstall` from the pinned index, which restores the whole
  `nvidia-*-cu12` set. *(Same class as the `df` / Thor `free` / cgroup `usage_in_bytes` traps: a
  symptom read as its own root cause.)*
- ⛔ **AN ANALYSIS-TIME IMPORT THAT FAILS AFTER THE ROLLOUT DESTROYS THE RUN'S OUTPUT WHILE THE
  COMPUTE IS ALREADY PAID FOR.** MEASURED 2026-08-11: `t1_eval.py` rolled **both arms, all 40
  episodes, 6 844 windows each (~11 min/arm)** and then died in `analyze()` on
  `from taniteval import selgap` — pod5's package predates the module. `T1_EXIT=NO_ARMS_PRODUCED`
  reads like a total failure; it was a **100 %-complete run with a missing last step**. Same class
  as the `UnicycleStepReadout` failure earlier the same night. ⇒ **Check for `--analyze-only` (or
  the dump dir) BEFORE re-running anything** — re-analysing the banked dumps recovered every number
  with **zero GPU**. The durable fix is a **preflight import probe at startup**, so a missing
  optional module fails in 2 seconds instead of after the expensive part.
- ⚠️ **THE GOTTY PTY DROPS OUTPUT IN A SYSTEMATIC PATTERN, AND A PLAIN `base64` PULL GOES SILENTLY
  CORRUPT.** MEASURED 2026-08-12 pulling a 24 KB JSON: **5 of every 14 lines dropped** (lines
  11–15, 25–29, 39–43, 53–57 of 62) — the blob still looks like base64 and fails only at
  `binascii` with a padding error, i.e. it can also decode to *garbage* rather than erroring.
  Widening to `-w 200` made it worse, not better. ⇒ **Frame every line with its number
  (`awk '{printf "@%04d@%s#\n", NR, $0}'`), emit the expected line count, then refetch exactly the
  missing/short lines with `awk 'NR==n'` and reassemble.** Verify the reassembled bytes by md5 or
  by parsing the JSON — never by eye.
- ⛔ **A DENSE WINDOWED TENSOR DUPLICATES EVERY FRAME `W` TIMES AND SILENTLY SWAPS — AND THE
  GPU-UTIL PROBE THEN MISATTRIBUTES IT.** MEASURED 2026-08-19/20: `e_trunk_pooling.py` built
  `X = [N, W, n_tok, d]` float32 from a 5.6 GB token cache. Consecutive windows overlap by `W-1`
  frames, so the stack held **~25.6 GB** for 5.5 GB of unique data, and the very next line
  (`X - mu[:, None]`) allocated a **second** 25.6 GB. On a 34.2 GB box the job paged to disk for
  **2.5 h at 41 % GPU without finishing**; killing it dropped host RAM 67 % → 25 %.
  ⚠️ **I read "GPU 41 %" as "PCIe-bound batching" and was wrong** — the GPU was idle because the
  HOST was thrashing, and `nvidia-smi` cannot see that. ⇒ **Before blaming the accelerator, price
  the host tensor: `N × W × tokens × d × dtype`, and compare it to FREE RAM, not total.** The fix is
  never a smaller batch — it is to **bank unique frames once (fp16) and window by INDEX**, centring
  per batch so no second full copy exists (`e_trunk_pooling2.py`: 25.6 GB → 2.76 GB, 3 seeds in
  42 s). *Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps: a probe that
  reports the wrong scope, read as an answer.*
- ⛔ **AN ARTIFACT'S COST IS THE COST OF THE FILE ITS *CONSUMER* OPENS — NOT THE ONE YOU
  FOUND ON DISK.** MEASURED 2026-08-29: the B1 epcache was sized by loading
  `_epcache/.../ep_00000.pt` (raw `frames_u8 [199,9,256,256]` uint8 = 117.4 MB, **exactly**
  the file size), scaled to 256×640, and published as a **1.38 TB capacity wall** — a format
  change, a geometry downgrade, a sharding design and "a PI decision" all followed from it.
  Every number was right; the **artifact** was wrong. The v7 trainer's `--v2-cache` reads
  `*.v2ep.pt` (`tanitad/data/v2_dataset.py`, written by
  `scripts/v2_compressed.py::build_compressed`), which stores **ENCODED** frames —
  **34.0 MB/ep, 161 GB**, confirmed independently at ~36 MB/ep over 2,403 live episodes.
  **It fits with ~4× headroom; there was never a wall.**
  ⇒ **Before pricing ANY derived artifact, open the CONSUMER'S LOADER and price what IT
  reads; name the consumer and the loader file in the estimate, or the estimate is
  inadmissible.** Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s`
  traps with the object swapped: **a true measurement quoted outside its scope reads exactly
  like an answer**, and it is worse than no estimate because it manufactures decisions.
  ⚠️ **Corollary — when you correct the artifact, RE-RUN the cost; never port the old timing
  onto the new format.** Re-timing the real path changed the plan more than the retraction
  did: **PNG encode is 70 % of the build** (6.19 s decode+remap + 14.79 s encode), so the true
  wall-clock is **3.4–4.6 h**, not the ~1.2 h a decode-only benchmark reported — and a 4 h job
  was about to be started in a 1.5 h gap.
  ⚠️ The codec is **load-bearing, not a speed knob**: `v2_dataset.py:325` and
  `slice_v2_cache.py` **refuse to sub-frame a LOSSY cache**, so only a PNG cache can be sliced
  to another geometry without a rebuild. *(DE-C152 / class C82; drafted by the DataFlyWheel,
  applied by the Master Mind on the PI's direct authorisation 2026-08-29.)*
- ⚠️ **A DERIVED CONSTANT SILENTLY CHANGES THE EXPERIMENT WHEN ITS INPUT CHANGES.** Same file:
  `HORIZON` moved from a hardcoded `7` to `round(6.0 * 10.0 / STRIDE)`. At stride 8 that is **8, not
  7** — 6.4 s instead of the banked 5.6 s — which cost exactly **one window per clip (1,509 →
  1,379 over 130 clips)** and makes a "reproduction" run *not* a replay. ⇒ **When a constant becomes
  derived, re-derive it for EVERY cache it will meet and state the value in the result**, or a
  refactor validation silently compares two different experiments. *(At stride 4 it lands on 15 × 4
  = exactly 6.0 s — the true design horizon, and a reason to prefer that cache.)*
- ⛔⛔ **A DECODE THAT RAISES INTO A PRE-ALLOCATED MEMMAP LEAVES A FULL-SIZE FILE OF ZEROS —
  AND THE JOB CAN STILL EXIT 0.** MEASURED 2026-08-21 building E-DETECT-1's raw-patch floor:
  the episode cache's buffer is named **`jpeg_buf`** but its `codec` field says **`png`**
  (magic `0x89 0x50`), so `decode_jpeg` raised — after `open_memmap` had already created
  **2.76 GB**. `ls` showed the expected size and the wrapper reported success.
  ⚠️ **WHY THIS ONE IS WORSE THAN A CRASH: the poisoned array was the FLOOR ARM.** An
  all-zero floor scores at chance, so **every trunk would have appeared to beat it** — a
  silent FALSE-POSITIVE generator, not a missing number. ⇒ **Assert on CONTENT before
  using any generated bank** (sample rows, require non-zero, print the mean), and **read
  the codec/dtype field rather than trusting the buffer's NAME**. Same family as the
  library tool's "reported success while filing 11 papers under the query string".
- ⚠️ **THE PINHOLE FOV FORMULA IS WRONG ON A CYLINDRICAL PROJECTION.** MEASURED
  2026-08-21: our corpus is `256x640` **cylindrical**, `f_ref` 305.577, where the column is
  **LINEAR IN AZIMUTH** (`az_max = (W/2)/f_ref`) giving **120°** — matching the rig's own
  name `camera_front_wide_120fov`. Applying `2·atan((W/2)/f)` yields **92.6°** and looks
  entirely plausible. ⇒ **State the projection before using any camera formula**; a correct
  formula quoted outside its projection is the `df`/`free`/`step_s` scope error in optics
  costume. *(Caught because the corrected number reproduced the rig's published FOV — an
  independent cross-check is what makes such a value trustworthy.)*
- ⛔⛔ **A PROBE THAT TUNES ON THE DATA IT SCORES WILL MANUFACTURE A RESULT — AND THE ONLY
  THING THAT CATCHES IT IS A CONTROL THAT MUST READ A KNOWN VALUE.** MEASURED 2026-08-22, **four
  distinct failures in ONE ridge probe in one afternoon**, each of which produced a confident,
  publishable-looking number: **(1)** EM normalised by the target's **raw energy** when the target
  had a large constant (ego moves ~0.47 m/tick) → the latent, `[z, dz]` and **raw pixels** all
  scored "+0.54", identical to three decimals, because every one of them merely reproduced the
  MEAN; **(2)** λ selected on the **point estimate** → reported +0.130 with a CI straddling zero
  and HID the real +0.020 whose CI excluded zero; **(3)** λ selected on the **test set** → picked
  λ=1e6, which shrinks the ridge to the constant predictor and scores EXACTLY +0.0000 with a
  zero-width CI, beating every noisy positive estimate, so the latent AND the constant control both
  read +0.0000 and the panel concluded *"the latent carries no dynamics"*; **(4)** **n ≪ d**
  (2,050 features on ~700 rows) → the validation CORRECTLY chose maximal regularisation and
  everything again read +0.0000, which is **underpowered by construction, not absence**.
  ⇒ **THE RULE: every probe panel carries (a) a CONSTANT-ONLY control that must read the
  no-information value exactly, (b) a RAW-INPUT floor (pixels) — a learned representation that does
  not beat raw input has added nothing, and (c) its `n` and `d` printed in the table.** Fit ALL
  hyper-parameters (λ, PCA basis) on the FIT split only; the scored split is scored, never tuned on.
  *(Failures 1, 3 and 4 were each caught ONLY because a control read the same value as the thing
  being measured. Without the floor, #1 would have shipped as an encoder indictment.)*
- ⚠️ **A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY.** The implication runs
  one way only: a linear oracle **beating** the baseline proves the target IS learnable; a linear
  oracle **failing** proves only *not learnable by a linear map*. MEASURED 2026-08-22: I wrote
  "~98 % unpredictable ⇒ unreachable by ANY predictor" from a ridge fit — in 2048 dims with
  nonlinear scene motion that is a WEAK lower bound and the claim was unsupported. State the
  function class in the verdict, or use a nonlinear probe with a **time-shuffled control** (structure
  surviving a shuffle is leakage, not dynamics).
- ⛔⛔ **`git ls-tree -r` SILENTLY TRUNCATES ON THE G: MOUNT — IT EXITS 0, AND IT TRUNCATES
  *CONSISTENTLY*, SO REPEATING IT LOOKS LIKE CONFIRMATION.** MEASURED 2026-09-02: the same tree
  read **7,535** files five times in a row while `git cat-file -e <tree>:<path>` proved six
  "missing" load-bearing files (`tools/secret_scan.py`, `taniteval/tools/t1_eval.py`,
  `stack/tanitad/refs/refa_v1.py` …) were all **PRESENT**. Other reads of the same tree gave
  7,536 and 7,559 — so it is neither stable nor honest. ⇒ I concluded a commit had **deleted 989
  tracked files**, reverted a perfectly good commit, and spent an hour "recovering" content that
  was never lost; the replacement commit's tree hashed **identical** to the one I had reverted.
  ⚠️ **The sibling failure in the same hour:** `git diff --name-only A B`, `git diff-tree -r`
  and `git show --stat` ALL returned **empty** for a commit that really did change 146 files.
  **Three independent-looking probes agreed on a wrong answer because they share one failure
  mode** — which is exactly what "take multiple samples" is supposed to protect against, and
  does not, when the samples are drawn through the same broken channel.
  ⇒ **ON THIS MOUNT, AN EMPTY OR SHORT GIT RESULT IS NOT EVIDENCE OF ABSENCE — it is
  indistinguishable from a failed query. Only POSITIVE assertions are admissible:
  `git cat-file -e <ref>:<path>`, a blob-hash comparison (`git rev-parse A:path` vs `B:path`),
  or `git ls-files` on the index.** Never guard a commit on a file COUNT; guard it on the
  presence of named load-bearing paths. *(Same family as the `df`/Thor `free`/cgroup
  `usage_in_bytes` traps — a probe reporting the wrong scope, read as an answer — with the
  aggravation that this one is reproducible enough to pass a "measure it twice" check.)*
  ⚠️ Two more from the same session, both real: **`.git/COMMIT_EDITMSG` can become POISONED**
  (writes fail `File too large`, `rm` fails `Device or resource busy`) which breaks every
  porcelain `git commit` while `commit-tree` plumbing keeps working — the fresh-inode class
  again; and a bulk `git add -- <directory>` across large trees can die with
  **`3221225478` = `0xC0000006` STATUS_IN_PAGE_ERROR**, so add explicit FILE paths in batches,
  never whole directories.
- ⛔⛔ **A SEARCH TOOL WILL REPORT "NO MATCHES" FOR FILES IT COULD NOT OPEN — AND
  DIRECTORY LISTINGS KEEP WORKING WHILE CONTENT READS FAIL.** MEASURED 2026-09-04 with an
  INTERLEAVED SAME-BREATH CONTROL (100+ samples): the mount served content in only 5, and
  in **all 5** the control file read fine while the target failed — the target never read
  once. Metadata resolved throughout (listings showed the file and its exact byte count),
  so every tool reported the file as present and matchless. ⇒ **"0 hits" is a claim about
  the SEARCH, not about the CONTENT, unless you also assert the file was READ.** Count
  unreadable files explicitly and retry them to zero, or report the gap. ⚠️ The
  discriminator between "the mount is flapping" and "this file is unreadable" is an
  interleaved control **in the same breath** — a control that succeeded in a *different
  second* proves nothing, and reading one that way is how two wrong diagnoses got made
  before the third stuck. *(Local cause here: `.claude/worktrees/` content is cloud-only
  and NEVER hydrates on this machine — 27 identical copies, 0 readable, while their
  metadata resolves perfectly.)*
- ⛔ **SEARCH THE COMMA-FORMATTED FORM OF EVERY NUMBER.** MEASURED 2026-09-04: a retraction
  took THREE revisions — first "the number does not exist", then "it was back-constructed"
  — because `45,456` and `168,873` live in prose as **comma-formatted** and every
  `\b45456\b` / `\b168910\b` probe missed them. Mount health was irrelevant; the regex
  was. The true account (a transposed digit plus a denominator from a different run) only
  appeared once someone searched `45,4` and `168,`. ⇒ **When hunting a figure in prose,
  search BOTH forms, or your absence claim is about your regex.**
- ⚠️ **A number carries its ARM and its ARTIFACT PATH, or it is not quotable — and
  PERCENTAGES are the worst offenders, because two different fractions round to the same
  one.** `45,456/168,873` and the fabricated `45,466/168,910` both read 26.9 %, which is
  exactly why nothing looked wrong for hours. *(Same night, same file: `168,910` is REF-A
  v1's window count; refcv3's is **805,687** — a 4.77x error — because the two arms window
  the SAME episodes at 170.9 vs 36.9 windows/episode. The vector was one word: "corpus"
  means the CLIP SET, which they share; the WINDOW GRID is not shared.)*
- ⛔ **A CORRECT FORMULA APPLIED UNDER THE WRONG UNITS READS EXACTLY LIKE AN ANSWER — AND AN
  ARTIFACT THAT DOES NOT NAME ITS UNITS INVITES IT.** MEASURED 2026-09-04: refcv4b's live
  `anchors.pt` is a dict of exactly `anchors [117,8,2]` + `controls [117,2]`; column 1 of
  `controls` is LATERAL ACCELERATION (m/s²), and nothing in the file says so. Read as
  CURVATURE (1/m), `a_lat = v²κ = 36² × 3.0 = 3,888 m/s²` = **396 g at 36 m/s, with 104/117
  anchors over a μ = 0.7 friction circle**; read correctly, `3.0 m/s²` = **0.31 g, 0/117**. Both
  tables look plausible, the Kamm arithmetic is right in both, and the 396 g one was produced
  first — from the shipped file. The RUN record was complete (`config.json['argv']` carries
  `--anchor-control-units alat`); the standalone artifact was not, and an artifact is opened in
  isolation far more often than its run record. ⇒ **Before applying any formula to a tensor,
  find the line that states its units — in the FILE, not in your memory of the builder. If the
  file does not carry them, the number you are about to compute is inadmissible until it does.**
  Durable fix (2026-09-05): builders write `control_units` / `horizon_s` / `dt` / `ref_speed_ms` /
  `kappa_cap` / `alat_v_floor` + provenance INTO the `.pt` (`stack/tanitad/refs/anchor_meta.py`),
  and `refc_v3_train.py` REFUSES a `controls`-carrying file that declares nothing unless
  `--anchor-control-units` is passed explicitly — recorded in `config.json` as
  `control_units_source: cli-override-legacy-file`, so the record says the units came from the
  operator, not the artifact. *(Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` /
  `step_s` / cylindrical-FOV traps — a true quantity quoted outside its scope — with the scope
  being the UNIT.)*
- **Verify before alarming.** Check the metric's definition and take multiple samples first;
  several "outages" were measurement artifacts. ⚠️ **But see the `ls-tree` trap above: repeated
  samples through ONE broken channel are one sample.** A second *probe* means a different
  mechanism, not the same command run again.
- ⛔ **A POD CHAIN WHOSE OUTPUTS HAVE NO AUTOMATIC EGRESS HAS PRODUCED NOTHING UNTIL SOMETHING
  COPIES THEM OFF.** MEASURED 2026-10-11 (`…/2026-09-20-refe-plan/STOP_CONSEQUENCES_2026-10-11.md`):
  the REFe A40 pod (no persistent volume) became unreachable, ssh rc 255, while SFT-4 and P2 ran on
  it unattended after the weekly API limit stopped the orchestrating session. The final run's model
  had been pushed to HF the moment it finished, but the fine-tunes chained behind it had no push
  step. **Lost unless the pod disk survives:** SFT-4's final checkpoints, its `sft.log` and
  `chain.log`, P2's gate, run and checkpoints, and the 177,836 PDM training labels (~11.3 h of CPU to
  redo). The only SFT-4 checkpoint off the pod is an update-400 peek someone happened to copy
  for an exploratory read, and the update-800 held-out numbers survive only as prose in a handoff.
  ⇒ **Every pod chain ends with an HF push of its checkpoints AND logs, and its success marker is
  the REMOTE listing, never the local file.** Label sets and other derived data a later arm needs
  get pushed when they are written, not when the arm that uses them finishes. Same family as
  operating standard 3 (*an artifact on one disk is not done*), with the disk being one you rent.
- ⛔ **A QUOTA GUARD THAT CANNOT READ THE QUOTA MUST REFUSE, AND THE HF LIST ENDPOINTS DO NOT SERVE
  `usedStorage`.** MEASURED 2026-10-11 against the live API while building
  `…/2026-09-20-refe-plan/refe/hf_egress.py`: `list_models` / `list_datasets` / `list_spaces` with
  `expand=["usedStorage"]` return **HTTP 400 "Invalid option"**. Only the per-repo
  `repo_info(rid, repo_type=…, expand=["usedStorage"])` serves it. So the private total is a list with
  `expand=["private"]`, then one `repo_info` per private repo. Summed that way it was
  **199,397,374,781 B against the 1e12 B Pro allowance**.
  ⚠️ The tempting shortcut is to catch the 400, or read a missing `used_storage` attribute as 0. Either way
  the guard reports a total of 0 and approves every push, and private storage over 1 TB is **billed, not
  refused**. ⇒ An unknown `private` flag or `used_storage` **raises**, and the push is refused
  (`private_used_bytes`, `hf_egress.py:104`). Same family as *a zero is a claim about the probe*.
- ⛔ **ON THE POD'S `/workspace` (VAST NFS), DELETED SPACE IS RECLAIMED LAZILY, SO A FULL QUOTA
  OUTLIVES THE `rm` THAT CLEARED IT. A CHAIN THAT LOGS ITS REFUSAL ONTO THAT VOLUME LOSES THE
  REFUSAL.** MEASURED 2026-10-16 (P6 restart):
  - A relaunch that omitted the starter's env file ran P6's default **96 GB `dd` write test**. That
    test filled the quota at 01:00Z. The chain's own refusal line could not be written, so the log
    went silent instead of saying why.
  - After the test file was deleted, an `echo` into a log still hit **`EDQUOT` at 03:10Z while `df`
    showed ~36 GB free**. A relaunch one minute later wrote normally.
  - ⇒ **(a)** Size a pre-flight `dd` test to the MEASURED need of the stages that remain (P6:
    `P6_DD_GB=16`), never to a default, whenever a training run shares the volume. The test
    consumes the same quota the trainer's next checkpoint needs.
  - **(b)** After deleting multi-GB files, re-test with a small `dd` before relaunching anything.
    `df` was already wrong in the full direction (see the `df` entry above); here it is wrong in
    the free direction too.
  - **(c)** A chain that goes silent on a full volume has not "hung": check its PID and its exit
    code, not its log.
  Same family as *a zero is a claim about the probe*: an empty log is a claim about a write that
  may have failed.
- ⛔ **RELAUNCH A CHAIN EXACTLY THE WAY ITS STARTER DID: SOURCE THE SAME ENV FILE. A bare
  `bash chain.sh` silently runs on defaults the starter had overridden.** MEASURED 2026-10-16:
  - P6's starter sources `/workspace/ops/p6_start.env` (`P6_FREE_NAVLOGS=1`, `P6_YAW_WRAP=off`,
    `P6_DD_GB`; each override is justified in that file).
  - After an instrument fix, the Master Mind restarted with the bare script. That reverted all
    three overrides with no warning, and the default `dd` size filled the volume (entry above).
  - ⇒ Before any relaunch, read how the dead chain was started: its starter script, or
    `/proc/<pid>/environ` while it is alive. Relaunch with the same `set -a; . <env>; set +a`
    prefix.
  - ⇒ Every chain starter keeps its overrides in a file next to the chain, with a backup before
    any edit. An override that exists only in a shell history is lost on the first restart.
  Same family as *`supervise_run.sh` sources its manifest once*: a run's configuration lives in
  how it was LAUNCHED, not in the script you are about to re-run.

- ⛔ **A UNIT TEST RUN BY A GATE INHERITS THE CHAIN'S ENVIRONMENT, AND A HAND-KEPT SCRUB LIST
  GOES STALE THE DAY A NEW KNOB LANDS. The leaked knob here was a path the chain `rm -rf`s.**
  MEASURED 2026-10-18 (P6 launch-gate dry run, G-SUITE FAIL; diagnosis
  `…/2026-09-20-refe-plan/raw/2026-10-18-p6-gate-diagnosis/RESULT.md` §2):
  - `gate_checks.check_suite` gives each suite script `env = dict(os.environ, …)`, so the suite
    runs with the CHAIN's env. The dry-run exports `P6_CBANK=<the real 3-C bank>`.
  - `test_p6_chain.Sandbox.run` copied `os.environ` and popped a hand-kept list of 20 names. The
    list predates the `P6_CBANK` knob, so the knob reached the sandboxed chain.
  - Only the chain's own guard stopped it (`REFUSED: P6_CBANK is set … would rebuild (rm -rf)
    it`). ⚠️ **Without that guard the unit test would have deleted the real bank: ~13 h of pod
    CPU.** Discriminating runs: with `P6_CBANK` set rc 1; without it **ZZP6CHAIN_PASS 341**.
  - ⇒ **Scrub by NAMESPACE PREFIX, never by list**: the sandbox drops every variable of the
    chain's prefix, so the next knob is covered without an edit. The fix adds a hermeticity case
    with a canary bank and a CONTRAST arm that hands the same value to the chain and must be
    REFUSED (proves the probe can see a leak): **349** under the full dry-run env and clean env.
  - ⇒ **Any chain knob that names a path the chain deletes or rebuilds needs a refusal guard in
    the chain itself**, not only in its tests. The guard is what turned a data loss into a FAIL.
  - Second finding, same battery: on the dev box Git's `bin/bash.exe` launcher prepends
    `/usr/bin` and hides a test's `nice` stub. Fix: `bash_exe()` picks the MSYS bash. ⚠️ Eleven
    other `test_*.py` still use `shutil.which("bash")` and are exposed to the same runner choice.
  Same family as *a derived constant silently changes the experiment when its input changes*:
  a list derived from the knobs that existed when it was written.
- ⚠️ **A TEST THAT FAILS ON A STARVED DEV BOX IS NOT A REGRESSION VERDICT — DISCRIMINATE WITH BASE UNDER THE SAME
  STRESS.** MEASURED 2026-10-18 (P6 gate-fix battery): `test_rls1_queue.py:184` failed on the FIX tree and passed on
  BASE, which reads as a regression. It was a race in the test's own setup (it started the queue without waiting for its
  fake S1 to hold the lock), exposed while another stream's sharded `pytest` (9+ processes, 1.5-3.8 GB each) exhausted
  the memory COMMIT limit (`0xC000012D`) and MSYS `fork` failed. Discriminating runs: the scenario's import closure
  contains none of the changed files; FIX passes 8/8 idle; BASE fails 3/3 with an injected 2.5 s slow start.
  - ⇒ Before calling a FIX-only failure a regression: (1) does the failing path reach a changed file? (2) re-run idle;
    (3) inject the suspected stress into BASE. A timing assertion in a test is a probe of the box as much as the code.
  - ⇒ Cap concurrent test processes at 2 on the dev box, and keep ≥ 5 GB free on C: (it hit 0.17 GB the same morning).

- ⚠️ **TWO LOCK SCHEMAS SHARE `devbox_gpu.lock`, AND A DEAD-PID "STALE" RULE CANNOT SEE AN MSYS PID — IT STEALS
  LIVE LOCKS.** MEASURED 2026-10-18 (found while building the P6-HN screen chain, confirmed in source):
  `eval/aux_navtest_forward.acquire_gpu_lock` removed any lock whose `pid` `tasklist` could not find once it was
  > 600 s old. The refcv8 battery's schema is `{job, pid = an MSYS/Git-bash pid, win_pid_of_locker, host, acquired}`,
  and its `win_pid_of_locker` is a transient process that exits at once. So a LIVE battery in a CPU-only phase reads
  as dead, and the helper would delete its lock and start a second GPU job on top of it. A foreign-schema lock (MSYS
  pid 861623, 3600 s old) was deleted by the old code and is refused with `GpuBusy` by the fixed one.
  - ⚠️ **`win_pid_of_locker` is not a liveness signal either.** It names the transient process that WROTE the file.
    MEASURED 17:36 Berlin the same day: `tasklist` finds no such process while the battery's lock is live. A stale
    rule that tests it instead of `pid` steals the same live lock.
  - ⇒ **A lock helper may clear ONLY a lock carrying its own owner marker.** Every other schema is a live job's lock
    and is waited on. "pid not alive" is evidence only inside the pid namespace that wrote it.
  - ⇒ **Test it with a foreign-schema lock** (MSYS-style pid, old mtime) in a temp dir: require `GpuBusy` and an
    untouched body. Restoring the removal branch must turn that test RED. A test that PINS the old behaviour
    (`raw/2026-10-16-scene-tap-a2/test_scene_tap_probe.py:753`) is inverted in the same landing.
  - The same rule was also in `c3_run.py:114`, `c3_run_p2.py:126`, `p2s1_autoread.py:449`, and worse in refav1's
    `gpu_lock.py:58-83` (a lock with no `t` field always read "before the last boot" and was renamed aside). None
    was running when found. All four FIXED 2026-10-18 (BACKLOG `LOCK-STEAL-2`).
  - ⚠️ **`subprocess.run(["tasklist", ...], text=True)` CRASHES on a dead pid on this box.** The German-locale "no
    tasks" line carries byte 0x81, cp1252 cannot decode it, `stdout` comes back `None`, and `'"pid"' in None` raises
    `TypeError` (MEASURED 2026-10-18 in `c3_run.py` / `c3_run_p2.py`). The old helpers therefore crashed here instead
    of stealing, and would steal on an English-locale box. ⇒ Read `capture_output=True` as BYTES and
    `.decode("utf-8", "replace")`; a probe that only ever ran on live pids never meets the line that breaks it.
  Same family as the `pgrep -f` self-match: a liveness probe answers for one namespace only.

- ⛔ **A W3 NAVTEST RUN KILLED BY ITS RAM GUARD LEAVES A RECONSTRUCTED PARTIAL CSV THAT LOOKS FINISHED — AND THE
  HARNESS'S GLOB FALLBACK RE-COPIES IT ON LATER ATTEMPTS.** MEASURED 2026-10-18 (RL-1 V1 navtest).
  - The W3 RAM guard aborts the devkit when available memory falls under 3,000 MB. Two other sessions' full stack
    test suites held 7.3 GB. The harness then **reconstructs a CSV from the streamed rows** (4,433 of 12,146).
  - With `--out` on C: and the package on E:, it crashed at `run_v1.py:459` (`os.path.relpath` across drives)
    **before its C2/C3 row-count checks ran**. Attempts 2-3 aborted at once on RAM, and the glob fallback re-copied
    the stale partial, so the run dir held a file that read like a result.
  - A relaunch lost one more attempt the same way (8,756 rows; a sibling agent's 3 GB stream scan). This time the
    checks RAN and rejected it. The run-dir guard then moved the partial aside, and attempt 2 scored 12,146/12,146.
  - ⇒ **Fix pattern** (`raw/2026-10-18-rl-headwise/raw/code/nt_run2.py` + `score_e2.py`):
    (1) a cross-drive relpath fallback;
    (2) a memory gate before each attempt: `psutil.virtual_memory().available` >= 6,500 MB for 3 minutes. ⚠️ The
        `\Memory\Available MBytes` perf counter does NOT EXIST on this German Windows, and a gate built on it reads
        0 forever;
    (3) move every `*.csv` in the devkit run dir aside before each attempt;
    (4) OK requires the full token-row count, not only harness PASS.
  - ⚠️ **A failed or partial run must not print its average row.** Two partial averages (88.44, 88.47) were read
    before the full read and had to be disclosed. Keep RAM-heavy sibling jobs off the box while navtest scores.
  - ⭐ **FIXED UPSTREAM 2026-10-19 (NT-FIX1)** in `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/run_v1_fix1.py`, a pinned wrapper (`run_v1.py`'s sha256 is recorded in the Thor admission record, the Colab code manifest and 24 `*.thor.json` stamps, so it is NOT edited). It fixes (1) the cross-drive relpath (absolute path, `csv_path_kind`), (3) stale `*.csv` moved to `quarantine_*` at run start and recorded, and (4) a count- or set-mismatched CSV emitting NO summary/average, printed or stored. Fix (2), the memory gate, stays in the launcher (`nt_run2.py`). 20 tests, one RED arm per fix. ⚠️ **It is inert unless the caller loads `run_v1_fix1.load()`.** Adopted (each with a reachability test + RED arm) in REFe's `eval/score_navtest_refe.py`, `score_navtest6.py`, `score_navtest7.py`, `colab/navtest/linux_run_v1.py` (and shipped by `pack_code.py`); `score_navtest_thor.py` stays on the admitted copy. NOT adopted: the taniteval `navsim_v1` plugin, `after_a8_chain.py`, `after_bridge_chain.py` (BACKLOG `NT-FIX1`). `incoming/2026-10-19-navtest-harness-fix/`

- ⛔ **A BASH VARIABLE NAMED `GROUPS` (or `SECONDS`, `RANDOM`, `FUNCNAME`, `LINENO` …) IS A SHELL BUILTIN. ASSIGNING IT IS SILENTLY
  IGNORED, AND THE SCRIPT RUNS ON WITH THE WRONG VALUE.** MEASURED 2026-10-21 (PREREG_V8F_TAC Amendment 1).
  - `chain_v8f_tacgrp.sh` held its eight group names in `GROUPS="goal latdec …"`.
  - Bash ignores the assignment (no error, rc 0), and `$GROUPS` expands to the user's gid: **197121** on the dev box.
  - So after 32 evals had finished on box 2, the pull asked for `eval_M1_s0-tacorc-197121.npz` and the chain stopped.
    The evals were intact; a finisher did the pull and the read without re-running one.
  - `bash -n` passes, a dry run passes, and `set -u` does not fire: the variable exists.
  - ⇒ **Never use a bash special name for your own data. Prefix it** (`TAC_GROUPS`).
  - `stack/tests/test_bash_specials.py` scans every box-2 / pod / stack shell script for an assignment, a `for` or a `read` over 18
    such names, and checks the mechanism itself. At the fix, it was 1 offender in 1,009 tracked `.sh` files.
  - ⚠️ **From Python on Windows, `bash` can resolve to `System32\bash.exe`, the WSL launcher, which fails when no distro is
    installed.** The test uses the first bash that actually runs `echo ok`.

- ⛔ **A MUTATION PROOF ON A TREE THAT A LIVE CHAIN READS SILENTLY CHANGES THE CHAIN'S CODE, AND NOTHING IN THE RESULT WOULD SHOW IT.** MEASURED 2026-10-26 (`PREREG_V8F_T1KIN_CAUSALBUCKET`):
  * **What happened:** the C-L2 mutation proof `sed`-edited `stack/scripts/v8f_t1_eval.py` in the patched export for about 14 s. A detached chain was running from that export, and both decisive runs STARTED at 01:19:00, inside that window.
  * **Why it is silent:** Python compiles its main script once, at start. Each run would have carried whichever mutant sat on disk at that second for its whole 6 min, and its JSON would look normal.
  * **What was done:** the runs were killed by explicit PID and re-run, with their partial outputs banked under `killed_0119/` of that package's run directory.
  * ⇒ **Mutate only a COPY that no chain reads.** Make every chain assert the sha256 of the files its runs import, BEFORE and AFTER each run, so that the file a result came from is pinned and not assumed.
  * ⚠ **When pinning a checkout against a git blob, strip CR first.** `git archive` under autocrlf writes CRLF, so the raw hash never equals the blob's. The same chain's first guard refused a correct tree on exactly that.
  * ⚠ **The same family:** a `git checkout` or pull in a tree a job runs from, and a `uv pip install` into a venv a job is using.

## Git hygiene — the mistake that has now happened twice

**`git commit` and `git commit --amend` both commit the ENTIRE INDEX, not the files you
just `git add`ed.** When several agents stage work concurrently — the normal state here —
a "quick commit of my thing" silently sweeps in a sibling's half-finished code under the
wrong message. This has happened twice in one session (`60265d3` swallowed the eval
tooling; `3d41bd0` swallowed REF-C v1.2's in-progress rescorer).

**Rule: when the index contains other agents' work, commit with an explicit pathspec and
do NOT follow it with `--amend`:**
```
git commit -F <msgfile> -- <path1> <path2>        # pathspec form, no amend afterwards
```
Check `git status --short` for foreign staged entries FIRST. If a long message is needed,
write it to a file and pass `-F`, because the `--only ... && --amend` pattern re-opens the
whole index and defeats the pathspec.

⚠️ **`git commit -- <pathspec>` (the partial-commit path) SEGFAULTS on this repo and is NOT
usable as the default** (measured 2026-07-25: exit 139 under MSYS git *and* `0xC0000005` under
native Windows git — so not the shell; **not** fsmonitor, already `false`).

**No mechanism is stated here on purpose.** Three of my root-cause readings were falsified in
one session — "it's 178+ files" (a 2-file commit crashed), "it's the pathspec shape"
(the *identical* single-file command crashed then succeeded), and "it's flaky, retry fixes it"
(**18/18** consecutive attempts then failed against a different index state). Each fit every
observation available at the time. **Do not re-derive a theory from a handful of runs — use the
procedure below, which is what actually holds.**

1. **Prefer a pathspec-free `git commit -F <msgfile>`.** It uses the normal (non-partial) path
   and has never crashed. It commits the WHOLE INDEX, so it is admissible **only** after
   listing the index (`git diff --cached --name-only`) and confirming every entry is intended
   program work — which is the check the pathspec rule above exists to force anyway. When a
   sibling agent's deliverables are in there, say so in the commit message rather than
   splitting them out.
⛔⛔ **AND THE SHARED INDEX CAN CARRY PHANTOM DELETIONS THAT TURN THAT EXACT ADVICE
   INTO A MASS DELETE.** MEASURED 2026-09-04: `git diff --cached --diff-filter=D` listed
   **269 staged deletions**, and on classification **269 of 269 were PHANTOM** -- every
   single path was present in `HEAD` **and** non-empty on disk. Zero were genuine. The
   mechanism is the `read-tree`-from-a-stale-tree one already documented above for scratch
   indexes, except it landed in the **SHARED** index: seeded from a tree that predates the
   recent commits, so everything newer reads as deleted. It had swallowed whole banked
   packages (`2026-09-03-refc-v4-design/`, `2026-09-03-ego-zero-collision/`), two
   pre-registrations, `GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`, and a decisions file
   committed and blob-verified **ten minutes earlier**.
   ⇒ **`git status` showing `D ` is NOT evidence a file is gone. Classify before acting:**
   a path with `git cat-file -e HEAD:<path>` **and** a non-empty worktree file is a phantom,
   and the repair is `git reset -- <paths>` in batches of ~30 (it updates the index only and
   never touches the worktree). ⚠️ **Run that classification BEFORE any pathspec-free
   commit**, every time -- item 1 above tells you to list the index, and this is the reason
   the listing must be read for `D` lines specifically, not just skimmed for foreign paths.
   ⭐ `stack/scripts/mm_commit.py` is IMMUNE and is why several commits survived this: it
   seeds a **private scratch index from HEAD** and never reads the shared one. That immunity
   is the argument for making it the default, not merely the safest option.

   ⛔⛔ **AND THE SAME STALE-TREE MECHANISM ALSO PLANTS STALE *CONTENT*, WHICH HIDES BEHIND
   A PLAIN `M` AND IS THEREFORE WORSE THAN THE PHANTOM DELETIONS.** MEASURED 2026-09-07,
   caught **before** anyone committed: **five steering files carried an index blob that was
   NEITHER `HEAD` NOR THE WORKTREE** — worktree **==** HEAD in every case, index older than
   both. A single pathspec-free `git commit` would have **reverted 397 lines**: `CLAUDE.md`
   1,033 → 965 (losing the two rules landed that night), `PI_DECISION_QUEUE.md` 187 → 105
   (**both** open PI items), `PI_VIDEO_REVIEW_2026-09-06.md` 208 → 56, `MODEL_REGISTRY.md`
   4,742 → 4,692, `REFCV5_MISSING_PIECES_PLAN.md` 545 → 500. Every one was a file committed
   hours earlier — i.e. **the index goes stale precisely for what you just landed.**
   ⭐ **WHY IT IS HARDER TO SEE THAN THE DELETIONS ABOVE:** a phantom deletion prints `D `
   and the rule above tells you to look for it. Stale content prints **`M `**, which is
   **indistinguishable from a sibling's genuine staged edit** — so the documented advice
   *"check the listing for foreign paths"* passes it straight through.
   ⇒ **CLASSIFY, DO NOT SKIM. For every `M ` path, read THREE blobs and assert all three are
   40 chars first:**
   ```
   h=$(git rev-parse "HEAD:$p"); i=$(git ls-files --stage -- "$p" | awk '{print $2}')
   w=$(git hash-object "$p")
   # i==h -> clean · w==h and i!=h -> STALE INDEX (a revert waiting to happen)
   # w!=h -> a genuine edit; LEAVE IT ALONE
   ```
   ⛔ **Before clearing a stale entry, prove HEAD is a SUPERSET — "differs from HEAD" is not
   the test, because a revert also differs.** Diff the two line sets and require **zero**
   lines present only in the staged blob (all five read exactly 0). Then `git add <path>`
   restores index == worktree == HEAD. ⚠️ Expect one false "staged-only" line per file from
   whitespace/newline rendering — read it before dismissing it.

   ⛔⛔ **AND `git add` ITSELF CAN PAGE-FAULT FOR MINUTES WHILE `git hash-object` ON THE SAME
   FILES WORKS.** MEASURED 2026-09-07: `git add` exited **0xC0000006 (STATUS_IN_PAGE_ERROR)
   WITH EMPTY STDERR** on all 8 retries, for minutes, on 13 paths — while `git hash-object`
   on **the same paths in the same seconds** succeeded every time. **The mount was not down;
   one code path was** (`add` reads the working file through a memory map). It then recovered
   on its own, which is what makes it dangerous: it looks like a broken FILE, not a broken
   MINUTE. ⇒ **an `add` failure is not evidence about the file.**
   `stack/scripts/mm_commit.py` now falls back to `hash-object -w` + `update-index
   --cacheinfo` after a SHORT ladder, and latches the state so the remaining paths skip the
   probe. ⭐ `--cacheinfo` is also **strictly safer than `add` for a scoped commit**: it
   writes exactly the entry named and cannot pick up a sibling file or a directory's contents.
   `stack/scripts/commit_cacheinfo.py` is the standalone route for when `add` **and** file
   reads are both failing — it stages from blobs already in the object store and reads the
   worktree not at all. ⛔ Its hazard is therefore **staleness**, and the precondition is
   ENFORCED rather than documented: it refuses when the staged blob differs from the worktree,
   and reports **INCONCLUSIVE** — never agreement — when the worktree cannot be read.
   ⚠️ Pinned by `stack/tests/test_mm_commit_page_fault_fallback.py`, which reproduces both
   failures at the **real subprocess boundary** via a git shim and ships a
   deliberate-regression arm for each. MEASURED while writing it: **a PATH shim is NOT
   reachable** — Windows `CreateProcess` appends only `.exe`, so a `git.bat` on `PATH` is
   silently ignored and the real git answers, and the test would have been green against a git
   that was never shimmed.

2. **Every crash leaves a stale `.git/index.lock`**, so the next attempt dies with *"Another git
   process seems to be running"* — that reads like contention but is debris. Confirm no git
   process is alive, then `rm -f .git/index.lock` (the index survives intact). Clear it between
   attempts or you will chase a phantom error.

⭐ **THERE IS NOW A THIRD ROUTE, AND IT IS THE DEFAULT WHEN AGENTS ARE LIVE:
`stack/scripts/scoped_commit.py -F <msgfile> -- <path> [<path> …]`.**

```
python stack/scripts/scoped_commit.py --dry-run -F msg.txt -- <path>   # inspect first
python stack/scripts/scoped_commit.py         -F msg.txt -- <path>     # then commit
```

It points `GIT_INDEX_FILE` at a scratch index seeded from HEAD, adds **only** the named paths, and
creates the commit through **plumbing** (`write-tree` / `commit-tree` / `update-ref`) — so it
**never enters the porcelain partial-commit code that segfaults**. The real `.git/index` is never
opened for writing, so **other agents' staged work stays staged**.

Two guards make it worth using rather than merely slower: it **refuses** if the resulting tree
touches any path you did not name, and it **checks** — rather than assumes — that the shared index
kept its other entries. *(Verified 2026-08-18 by committing the tool with itself while 55 foreign
files from four streams sat staged; the shared index went 63 → 64 across the commit, i.e. agents
kept staging undisturbed.)*

⚠️ **WHY "BE MORE CAREFUL" WAS NEVER THE ANSWER.** The sweep is **two correct rules colliding**: the
operating standard tells agents to *stage as they go*, and this file tells committers the index is
*shared and whole*. Obey both and an agent's incremental `git add` is **guaranteed** to be swept by
the next commit. MEASURED: one agent's 44-file deliverable landed across **three** commits, 41 of
them under subjects about unrelated work.

⚠️ **This does not replace reading `git diff --cached --name-only` first.** When foreign work
genuinely belongs in your commit, take it and **name it in the message** — a recorded sweep is
recoverable, a silent one is not.

- ⛔ **A FLOAT32 ROUNDING BAR TAKEN AT THE RESULT'S MAGNITUDE FAILS UNDER CANCELLATION, AND A SMOKE WILL NOT SHOW IT.** MEASURED 2026-10-26
  (`PREREG_V8F_HACC_MC`, Amendments 1 and 2).
  * Amendment 1 set C-1's bar to "|d| ≤ 1 float32 ulp at |plan|", calibrated on a 24-window smoke (0.50 ulp).
  * The full run then REFUSED: 1.16–3.72 ulp, on 1–3 of 86,364 tick-knots. All of them sat on a near-stationary tick where the blend's
    result (−0.0017 m) is smaller than its operands (P +0.0046 m, τ −0.0267 m), which have opposite signs.
  * At the largest operand, every run read 0.49997–0.50000 ulp, and the regression arm still failed by 5.2e6 ulp.
  * ⇒ **Set a rounding bar at the largest operand of the operation that rounds**, `spacing(float32(max(|a|, |b|, |result|)))`, never at
    its output. This is the `R74` class, a third time in one day, and this time inside the amendment written to fix it.

- ⛔ **cmd.exe REFUSES A LINE OVER 8,191 CHARACTERS ("Die eingegebene Zeile ist zu lang" / "The input line is too long"), AND THE QUEUE IS NEAR
  IT.** MEASURED 2026-10-26.
  * `v8f_mini_box2_queue.cmd evalmixo`'s dry-run echo line is about 8,092 characters with a 57-character checkout and a long pytest root.
  * A longer checkout or `--basetemp` / `--tmp` pushes it over. Then `test_queue_oacc_leaves_every_other_stage_untouched`, the MIXO
    real-line tests and the KINDX trainmixo-line test fail. Every mutation baseline built on them fails the same way.
  * This happened three times in one day: the HMC chain's C-7 suite, the OACC `--only` mutation baselines, and an agent's "pre-existing
    failures" on its long-path export.
  * ⇒ **Keep checkouts and pytest roots short** (`E:/Projects/TanitAD-artifacts/<short>/tree`, `--basetemp <short>/p/x`). A queue-test
    failure from a long path is the environment, not the code: re-run it on a short path before calling it a regression.

- ⚠️ **`pytest --basetemp X` ERRORS EVERY `tmp_path` TEST WHEN X's PARENT DOES NOT EXIST.** It reports "2 errors in 4.68 s" and no failure.
  A wrapper that only greps for "failed" reads it as clean. ⇒ `mkdir -p` the parent first, and treat `error` like `failed`.

- ⛔ **A HOLD BUILT FROM A TREE THAT A MUTATION RUN IS EDITING LANDS A MUTANT, AND EVIDENCE MADE BEFORE THE RUN CANNOT SEE IT.** MEASURED 2026-10-26
  (`R77`).
  * `04d56671` landed `v8f_mini_train.py` with `if False:` in place of `if acc_w != acc_0:`. The hold builder had read the file from the tree
    where `mutate_oacc.py` was running its trainer mutants. The banked "10 changed lines" diff was made before that run, so the evidence and
    the landed bytes disagreed silently.
  * A second copy from the same tree put another mutant into a re-check tree.
  * Caught by a later builder's tip-equality assertion, before any run trained on it.
  * ⇒ **Never build a hold, or copy files, from a tree with a running mutation proof.** Each lander-side check compares the bytes it lands
    against a blob or the banked diff. After every code landing, run the touched test files on an archive of the LANDED tip, never only on the
    tree you assembled.
