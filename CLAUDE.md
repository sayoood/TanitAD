# TanitAD — working agreements

Sub-300M hierarchical 4-brain latent world model for autonomous driving. PI: Sayed.

## Source of truth (this rule exists because prose lied to us)

**`Project Steering/MODEL_REGISTRY.md` is the ONLY quotable source for model facts** —
architecture, params, training args, parity key, results, status. Any number in any report
cites the registry or the **raw eval JSON**, never a summary, changelog, or weekly report.

Three errors propagated for days because they were copied from prose:
- `flagship4b-phase0-30k` is the **no-speed ablation control** (2.918 m), NOT the deployed v1
  (that is **`flagship4b-speedjerk-30k`**, 0.452 m). The HF repo name invites this inversion.
- "REF-B v2 died at 22,600" — it did not; `metrics.json` says step **29999**. The *log* went stale.
- REF-A **I-JEPA's val number is unusable**: ~80% of val leaked into its train set.

If a doc and the registry disagree, the registry wins and the doc gets fixed.

**Never quote a learning-curve exponent bare.** Any exponent carries its **fit window, R² and n**,
or it is not admissible — and it may never decide a restart. Below **R² 0.80** there is no quotable
exponent at all; use the matched-step ratio. Never extrapolate more than **2×** beyond the fitted
range, and never compare exponents fit over different windows. *(Measured 2026-07-20: the same
`g_op_fwd_ade_m` log gives −0.387/−0.505/−0.564/−0.621/−0.738 depending on the window, all at
R² 0.09–0.58; v1's reference "−0.84" is the 1500–7500 window at R² 0.541, and on matched windows v1
and v3enc are statistically indistinguishable.)* Restart/continue decisions follow
`Project Steering/GATE_PROTOCOL.md` via `stack/scripts/run_gate.py`.

**Never quote an interval without its estimator.** The block historically labelled *"8-split
episode-disjoint jackknife"* is neither a jackknife nor a valid SE — it is
`overlapping_holdout_se`. The decision-grade interval is the **episode-cluster bootstrap** over the
40 val episodes (`taniteval/ci.py`); for two arms on the same windows use the **paired** version,
never a combination in quadrature.

⚠️ **It is not only an interval problem — `overlapping_holdout_se` also BIASES THE POINT ESTIMATE**,
because its central value is a **mean-of-split-means (`heldout`)**, not the **`full_set`** mean.
*(Measured 2026-07-25 over 27 arms with raw per-window data: headline `ade_0_2s` shifts **−6.67 % to
+11.69 %**, **bidirectional — 11 arms inflated, 16 deflated, none flat**; on hierarchy seams up to
**×3.3**, and on paired deltas up to **×−4.15 including a SIGN FLIP**. It manufactured the program's
one "load-bearing" hierarchy seam: `ctx→tactical` +0.0439 → true **+0.0148**.)* Interval narrowing
is **1.107–3.100×, median 1.499×** (27 arms; the older "1.28–2.06×" came from only 10 and was
under-sampled). **Before trusting ANY pre-2026-07-25 number, check whether it is the `heldout`
split-mean or the `full_set` mean — `MODEL_REGISTRY.md` publishes both and they differ.** Blast
radius + per-arm corrections: `…/incoming/2026-07-25-jack-blast-radius/JACK_BLAST_RADIUS.md`.

## Briefing a subagent — the contract

Every subagent brief MUST carry the preamble in
`Project Steering/AGENT_OPERATING_STANDARD.md`. Its three binding rules:

1. **Stage, never push.** Agents `git add` their deliverables into the working tree and
   **never commit and never push**. They must NOT leave work only on a pod or only in a
   worktree. *(The old "commit nothing" default stranded REF-B v2's architecture, the entire
   TanitEval harness, the pod ops bundle, and 486 lines of TanitResim — each on a single disk.)*
2. **End with a deliverable manifest** — every artifact and **where it lives**
   (repo path / pod:path / worktree). Stranding must be visible in the report, not discovered
   in an audit months later.
3. **Escalate integration, don't write "please merge" into a doc.** An orthogonality instrument
   sat unmerged for **10 days** because the request lived in a README nobody re-read.

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
- **Never judge pod disk with `df`.** It reports the 965 TB cluster and hides the per-pod MooseFS
  quota. Use a real `dd` write test. A full quota killed the flagship mid-checkpoint.
- **`step_s` in trainer logs is ACCUMULATED over `--log-every`** (÷50), not per-step. This has
  caused false "training is 430 s/step" alarms.
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
  costs memory. Each dataloader worker costs **~8.6 GB host RAM**. ⇒ **small batches, few workers.**
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
- **Verify before alarming.** Check the metric's definition and take multiple samples first;
  several "outages" were measurement artifacts.

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
2. **Every crash leaves a stale `.git/index.lock`**, so the next attempt dies with *"Another git
   process seems to be running"* — that reads like contention but is debris. Confirm no git
   process is alive, then `rm -f .git/index.lock` (the index survives intact). Clear it between
   attempts or you will chase a phantom error.

## Operating standard — raised by Sayed 2026-07-21

The program's pace goes up, and so does the bar. Five rules, each with the failure that earned it.

**1. State the evidence class or don't state the claim.** Every number carries
`MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (another agent/doc, NOT
re-verified)` · `ESTIMATED` · `HYPOTHESIS`. **A claim that decides a GPU-day must be MEASURED or
PUBLISHED — never INHERITED.** *(2026-07-21 alone: five retractions, every one from quoting a
faster-moving source than the harness. "v1.6 is best-in-program" was a **trainer log**, ~10 % optimistic
vs `eval_*.py`. Trainer val watches a curve; only eval output is quotable.)*

**2. Absence found at ONE location is not absence.** Before writing "X does not exist", probe a second
path, a second name, and the tool that owns the fact. *(Cost this session: the Vulkan ICD is in
`/etc/vulkan/icd.d/`, not `/usr/share/` → "our pods cannot render" stood for **12 days** and blocked
AlpaSim + CARLA. `ps -C python3` returns EMPTY for a healthy job because pods run
`/workspace/venv/bin/python` → a near-miss "the VLM job is dead". `obstacle.offline` — 3D agent tracks
on **97.44 %** of the corpus — was declared non-existent for days; our ingest reads **4** of 36 features.)*
⚠️ **The "2 of 36" in this very sentence was ITSELF stale — corrected 2026-07-26 to 4.** True of
`physicalai_r0.py` alone, but the episode build also reads `camera_intrinsics` and `sensor_extrinsics`
(`physicalai.py:153-154`) since D-016 R1. It had propagated into **≥7 documents including this one** —
a stale absence-claim living inside the rule that warns about stale absence-claims.
**And the answer to what is in the other 32 is now settled, at five independent probes:** there is
**no map, lane graph, junction annotation, roundabout label, traffic-light feature or route/goal signal**
in PhysicalAI-AV — the card says verbatim *"we do not include open maps data"*, and `obstacle.offline`'s
enum over **87,481 cuboids is 10 classes, all dynamic agents**. Stop re-asking; the strategic-brain
topology must come from AlpaSim or an external corpus. *(Also settled: `egomotion` carries **no
lat/lon/GNSS** — coordinates are clip-local metres, so **OSM map-matching on our traces is impossible**.)*

**3. Finish before you start. An artifact on one disk or in one agent's context is NOT done.**
Definition of done = **in the repo, staged, with its provenance**. *(LAL-v2 anticipation: implemented,
tested, **unmerged 12 days**. An orthogonality instrument: **10 days**. TanitEval, REF-B v2's
architecture, the pod ops bundle — each stranded on a single disk.)*

**4. Retractions are the learning mechanism — log the ROOT-CAUSE CLASS, not just the correction.**
`Project Steering/RETRACTION_LOG.md` is append-only and **must be read before asserting in a known
class**. A retraction with no class taught nobody anything.

**5. Aim above the published state of the art, and settle conflicts with experiments, not deference.**
When ambition meets inconvenient evidence, the answer is the **cheapest discriminating experiment**,
pre-registered with **both outcomes committed in advance** — not a scoped-down goal. *(The "strategic
choice is a ~2 % lever" refusal was **confounded**: REF-C evaluates with `nav_cmd=None`, so a decoder
that never had a working route input learned the marginal. I nearly designed the hierarchy away on it.)*

**Orchestration.** Parallel streams are the default, but: every brief carries a **priority order** so a
killed agent still yields value; agents **bank incrementally** rather than holding a final synthesis;
and **fan-out is capped** — uncontrolled sub-spawning exhausted the weekly API budget on 2026-07-21 and
cost three agents' work.

## Invariants

- **`Keys.txt` is git-ignored — NEVER commit it.** Read tokens in place
  (`grep -oE 'hf_[A-Za-z0-9]+'`); never copy, print, or write them to args.
- **Agents never commit to `main`** and never edit `Project Steering/Mission Plan.md`.
- **Parity is sacred:** the canonical train corpus is `physicalai-train-e438721ae894`
  (2376 episodes) with skip-hash `f09e44db`. Anything that re-selects episodes breaks
  cross-arm comparability and must be refused.
- **Never add GPU/RAM load to a pod that is training**, and never eval on a training pod.
- Full suite lives at `stack/` — `pytest -q` must stay green before any commit.

## ⛔ NEVER IDLE — a report is not work (Sayed, 2026-07-29, flagged THREE times)

**The failure pattern:** check the fleet → find the top item blocked (pod down, checkpoint not
ready, PI decision pending) → write a well-organised report about being blocked → re-arm the loop.
**The report feels like output. It is not.** Twice in one session a turn ended with *"next step is
implementing X"* and no implementation followed.

**The root cause:** treating the priority list as a **queue** — top item blocked ⇒ wait. It is a
**PRIORITY ORDER, NOT A DEPENDENCY CHAIN.** Item 1 being blocked says nothing about items 2–12.

**The rule:**
1. ⛔ **Never end a turn having only reported.** If the top item is blocked, **drop to the next
   UNBLOCKED item and execute it in the same turn.**
2. **`Project Steering/BACKLOG.md` is the live pull-list.** When the headline work is gated, pull
   from it. **Gated ≠ idle.**
3. **"Blocked on the PI" blocks ONE item, not the programme.** There are always 0-GPU items:
   implementation, banking stranded artifacts, pre-registrations, instrument fixes, doc corrections,
   re-verification of refuted-but-load-bearing claims.
4. A progress report is the **last 10 % of a turn**, never the whole turn.
5. ⚠️ **Long heartbeats are for GENUINELY gated states only.** Do not use a long heartbeat to make
   idling look deliberate.

*(Same class as "A report is not a launch": when an agent files, refill the pod in the SAME turn
before writing the summary.)*

### ⛔ THE RE-ARM IS NOT AN ACTION (Sayed, 2026-08-02, FOURTH flag)

**The specific failure:** I ended a turn with a status report followed by `ScheduleWakeup`, and
called that "continuing the loop". **Scheduling the next turn is not work in this turn.** It is the
same idling the rule above forbids, wearing a timer.

**Hard rules, mechanically checkable before ending any turn:**

1. ⛔ **Never call `ScheduleWakeup` in a turn whose only outputs were a report, a probe, or a
   commit of documentation.** The wakeup is admissible ONLY after something was *executed* —
   code changed, a job launched, an artifact produced, a measurement taken.
2. ⛔ **"I'll do X next iteration" is banned.** If X is unblocked, X happens in THIS turn. If X is
   blocked, name the blocker and **execute the next unblocked item**, then report both.
3. **A blocked headline item does not license a short turn.** `BACKLOG.md` exists precisely so a
   gated turn still ships something. Gated ≠ idle.
4. **Before ending a turn, ask: "what did I CHANGE?"** If the honest answer is "nothing — I looked
   and I described", the turn is not finished.
5. ⚠️ A **PI decision** blocks that ONE item. Provision, spend, and publish are the PI's; every
   implementation, measurement, instrument fix and banked artifact around them is mine, and none of
   them wait.


## ⛔ BINDING — A GOAL INPUT IS ADMISSIBLE, BUT MUST NOT CARRY THE SITUATION CLASSIFIER (Sayed, 2026-08-03)

**Sayed:** *"yes a goal input is admissible, at the same time, we need to be careful not to include
the result of the situation classification in the goal input."*

| | |
|---|---|
| ✅ **Admissible at inference** | a goal / route signal — including a **predicted geometric goal point**, which the literature shows is the lever that actually works (categorical command +0.2 PDMS; goal point **+4.7**). |
| ⛔ **NOT admissible inside it** | the **output of the situation classifier**, in any form — class posterior, argmax, embedding, or any feature derived from them. |

**Why the second half matters more than the first.** If the goal input carries the situation
classifier's output, then:
1. **Attribution dies.** A planner improvement can no longer be assigned to the goal or to the
   classifier — they are one path. That is the `--v2` conflation failure again (ten levers on two
   axes, result non-attributable), and the C6 confound again.
2. **The classifier stops being independently evaluable.** Its errors enter the planner and come back
   as planner metrics; we would be scoring a loop.
3. **It is the nav-echo defect in a new costume.** MEASURED today: flagship v1's route head is an
   exact bijection of the nav we feed it (369/369 and 81/81) and scored **1.0000** — an echo of its
   own input read as skill. A goal built from the classifier would let the *planner* echo the
   *classifier* the same way.

**The design rule:** the goal path and the situation path stay **information-disjoint at inference**.
State, for any arm, what the goal is computed from — and if a shared trunk feeds both, say so and
justify why that is not a back door.

⚠️ **The admissibility check to run:** for any goal signal, ask *"could this have been computed from
the situation classifier's output?"* If yes, it is inadmissible until shown otherwise. Same family as
the leak test in [[the vision-only rule]]: **ask whether an input at inference contains something the
thing being measured also produces.**

⚠️ **A supplied route is optimistic by construction on PhysicalAI** — our only route supplier there is
the ego's own future path. Prefer a **predicted** goal, which sidesteps both this and the
vision-only rule.

## ⛔ BINDING — LABELS MAY USE EGO; INFERENCE IS VISION-ONLY (Sayed, 2026-08-03)

**Sayed, verbatim:** *"for ground truth data of scenario classification you can use both ego and
other label, for inference only vision."*

| stage | what may be used |
|---|---|
| **Ground truth / label derivation** | ego state, other agents, maps, future poses — anything. Labels are built offline; privileged signals are FINE here. |
| **Inference** | ⛔ **VISION ONLY.** No ego state, no privileged channel. |

⇒ For the scenario classifier the deployable arm is **`head_img` (image-only)** — NOT `head_img_ego`
and NOT `head_ego`. This supersedes both earlier candidates: the ego-only swap (already ruled out,
*"no ego heads"*) **and** score-level fusion of image+ego, because that is still ego-at-inference.

⭐ **WHY THIS PROBABLY DISSOLVES THE SITCLF ANOMALY — verify, do not assume.** The situation labels
are derived from **ego dynamics** (`stack/tanitad/data/situations.py`). If a label is a function of
ego state, a classifier *given* ego state at inference is partly reading **the label's own source** —
that is a **LEAK, not a capability**. It would make the banked ranking (`head_ego` 0.0697 >
`head_img_ego` 0.0525 > `head_img` 0.0376) a measure of leak magnitude rather than evidence about
vision, and it makes `situations.py:19`'s *"VISION ADDS NOTHING OVER EGO STATE"* **unfalsifiable as
stated**. Establish the label provenance from source (two probes, quote file:line) before any further
sitclf claim.

⚠️ **"Vision scores worse" is never a reason to reopen ego at inference.** If the vision-only arm is
weak, the finding is *how much*, *why*, and *what would fix it*.

⚠️ **Generalise the test, not just the case:** for ANY head, ask whether its inputs at inference
include something the label was derived from. That is the same family as the C6 confound (a decoder
compared on its marginal) and the REF-A I-JEPA leak (~80 % of val inside train).

## ⛔ BINDING — AT LEAST FIVE PARALLEL STREAMS, ALWAYS (Sayed, 2026-08-03)

**Sayed:** *"dont idle, always have at least 5 streams in parallel, use the loop, goal and workflow
concepts of claude."*

**The rule:** at every point in the programme there are **≥5 work streams in flight simultaneously**.
One agent running while the orchestrator waits is IDLING WITH EXTRA STEPS. A turn that ends with
fewer than five live streams and unexhausted backlog is not finished.

**The three mechanisms, and when each is right:**

| concept | use it for |
|---|---|
| **Loop** (`/loop` + a CRON DRUMBEAT) | the heartbeat — re-enters, re-checks the fleet, and picks the next unblocked item. Must be a COMPLETE handoff: a fresh session reads only the cron prompt. Rewrite it every run. |
| **Goal / priority order** | the PI's numbered list. It is a **PRIORITY ORDER, NOT A DEPENDENCY CHAIN** — item 1 blocked says nothing about 2–12. Drop to the next unblocked item IN THE SAME TURN. |
| **Workflow** (`Workflow` tool) | fan-out across independent streams, each with its own adversarial verify. Prefer `pipeline()` so a stream verifies as soon as it finishes; use a barrier only when a stage genuinely needs ALL prior results. |

**How to keep five alive without thrash:**
1. **Never let a stream finish without launching its successor.** When an agent reports, commit its
   work AND start the next item in the same turn.
2. **Streams must be INDEPENDENT.** Five agents editing the same file is one stream with a merge
   conflict. Partition by directory//concern, and say which files each owns.
3. **Mix horizons:** long compute (training, engine builds), medium (experiments), short (docs,
   instrument fixes, banking stranded artifacts). A GPU-gated turn still ships the short ones.
4. **A blocked stream is replaced, not waited on.** `Project Steering/BACKLOG.md` is the pull-list.
5. ⚠️ **Fan-out is still capped** — uncontrolled sub-spawning exhausted the weekly API budget on
   2026-07-21 and cost three agents' work. Five to eight concurrent is the band; scale depth, not
   breadth, beyond that.

*(Root cause: the orchestrator repeatedly ran ONE agent and then reported. Sequential delegation is
still sequential. The programme's throughput is set by how many independent questions are being
answered at once, not by how carefully one is.)*

## ⛔ BINDING — EVERY NUMBER CARRIES ITS EVAL TIER (EVAL_DOCTRINE.md, 2026-08-06)

**T0** (teacher-forced / true-future-conditioned) is a **WM diagnostic — NEVER "driving
performance"**. **T1** (action-closed loop: the model conditioned on its OWN actions) is the
**PRIMARY tier for any capability claim**. **T2** (re-perception sim) is not provisioned.
*Why this exists (MEASURED, §1.12): open-loop lateral skill was an ACTION ECHO — S-curve
reproduction 97.9 % open-loop, 0.0 % hold-action, ~5 % closed-loop.* A registry row or report
quoting a number without its tier stamp is incomplete; comparisons across tiers are invalid.
Instrument: `taniteval/tools/t1_eval.py` (E1.2).

## ⛔ BINDING — EVERY EVAL REPORTS FOUR METRIC FAMILIES, NOT ADE (Sayed, 2026-08-02)

**Sayed, verbatim, after asking repeatedly:** *"Despite I told you many times, don't consider only
ADE at the different horizons, you are still doing this and ignoring the other metrics I requested.
So we will make it now very formal: I want you to ADD (not replace) ... Any future eval must include
these metrics and this is binding."*

**ADE stays. These are ADDED to it. An eval that reports ADE alone is INCOMPLETE and must not be
presented as a result.**

| family | must report | why it is not optional |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and distance-keeping** (headway / time-gap / TTC to the lead agent) | 88.7 % of our oracle gap is longitudinal. ADE hides it — an arm can win ADE while setting the wrong speed. |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross-track | "lateral is fine" has been asserted from cross-track alone; curvature and yaw are where a smooth-but-wrong path shows up. |
| **TACTICAL** | manoeuvre-decision quality **and tactical goal-setting** (selected vs executed manoeuvre, confusion over the classes, goal/anchor selection) | the 5-way softmax that MIXES lat+lon is our single largest known defect; a scalar ADE cannot see a decision error. |
| **STRATEGIC** | strategic decision + goal/route setting quality | the hierarchy is the programme's thesis. If we never measure the strategic level we cannot claim it works. |

**Rules that travel with them:**

1. **Per-family, never pooled into one score.** A single composite hides exactly the trade-off we
   are trying to see.
2. Each family carries its **estimator** (paired episode-cluster bootstrap) and its **CI**, on the
   same windows as the ADE it accompanies.
3. **A missing metric is a work item, not an excuse.** If a family has no instrument, implement it —
   do not report the eval as complete with the family absent.
4. ⛔ **Never present a horizon sweep of ADE as "the result".** It is one row of four families.
5. Where a family genuinely cannot be computed (no lead agent in frame, no route label), say so
   **per family with the reason and the n**, rather than silently dropping it.

*(Root cause this rule exists: ADE is the cheapest number to produce and the easiest to compare, so
it crowded out the metrics that actually decide whether the car drives well. Three separate reports
went out with ADE-only tables after this was requested.)*
