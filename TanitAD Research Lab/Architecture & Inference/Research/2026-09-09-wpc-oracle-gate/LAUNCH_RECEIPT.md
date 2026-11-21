# LAUNCH RECEIPT — the WP-C ORACLE GATE was LAUNCHED on `tanitad-refcv3` (A40), and the POD THEN WENT UNREACHABLE

> ## ⛔⛔ READ THIS FIRST — STATUS AS OF 2026-09-09 22:55Z: **THE POD IS UNREACHABLE**
>
> Everything below §0 was true and MEASURED when written. It is kept unedited as the launch
> record. **It is no longer a description of a running job**, and this box exists so no reader
> takes it for one.
>
> | | |
> |---|---|
> | **last CONFIRMED-healthy probe** | **2026-09-09 22:36:27Z** — B0 at **step 500 / 6,000**, first in-run eval fired, supervisor PID 2585994 and trainer PID 2586005 both alive, `oom_kill` **24**, GPU 44,529 MiB / 100 % |
> | **first failure** | the very next probe: `Connection reset by peer`, then `Connection refused` |
> | **now** | `69.30.85.211:22001` **refused on 6 consecutive tries over ~2 min** |
> | host reachable? | ⭐ **YES** — TCP to `69.30.85.211:22000` is accepted, so the machine answers; ICMP is blocked, as normal on RunPod |
> | is 22000 our pod? | **NO** — no SSH banner, connection closed immediately. Not an sshd we can use. |
>
> ⚠️ **`Connection refused`, not `timed out`, is the documented RunPod signature** for a pod that
> has been **stopped / resized, with its SSH port reassigned** (`CLAUDE.md`, traps preflight). The
> host answering on another port is consistent with exactly that.
>
> ⛔ **THIS IS A NAMED BLOCKER, NOT A DIAGNOSIS I CAN CLEAR.** `runpodctl` is **not installed**
> (checked locally, so that reading is reliable), and whether a RunPod API credential exists is
> **INCONCLUSIVE** — the `Keys.txt` probe returned 0 while its same-breath control ALSO returned 0,
> i.e. the G: mount was down and the count is a claim about the read, not about the file.
> **Restarting or re-mapping a pod is a PI provisioning action in any case.**
>
> **WHAT UNBLOCKS IT:** the PI (or anyone with the RunPod console) restarts the pod and reads its
> **new** SSH port, then `~/.ssh/config`'s `tanitad-a40 / tanitad-refcv3` block is updated. ⚠️ The
> working key is `~/.ssh/tanitad_pod`, not the console's `id_ed25519`.
>
> **WHAT SURVIVES A RESTART, and what does not:**
> * ✅ `/workspace` is the persistent volume: the join (327 MB, md5-verified), the shipped stack at
>   HEAD, the patched trainer, `wpc_gate/`, `anchors.pt` and B0's partial output are all on it.
> * ⛔ **B0's ~33 minutes of training is LOST** — `refc_v3_train.py` has **no `--resume`**, and B0
>   had not reached a done-marker. The runner is resumable at ARM granularity and will simply
>   restart B0 from step 0.
> * ⚠️ **The supervisor does NOT come back by itself.** After the pod is reachable again, someone
>   must re-run `setsid nohup bash /workspace/wpc_gate/sup_wpcgate.sh &` — and then **assert it is
>   running by `/proc/<pid>/cmdline`**, never by `grep -c`.
> * ⚠️ Clear `/workspace/wpc_gate/.sup_wpcgate.lock` **only** after confirming no live holder via
>   the `/proc/*/fd` scan; a lock with no holder is debris, a lock with one is not.
>
> ⚠️ **I cannot say whether the pod stopped on its own, was stopped by the provider, or was
> stopped by the same actor that ran `kill -9` on my supervisor 40 minutes earlier (§3.1).** I have
> no evidence that distinguishes them and I am not going to guess between them.
>
> ⇒ **§11's "in the morning" sentence is SUPERSEDED by this box.** In the morning, unless the pod
> is restarted, **nothing will be running and no arm will have finished.**

---

## (the launch record as written at 22:30Z follows, unedited)

# LAUNCH RECEIPT — the WP-C ORACLE GATE is TRAINING on `tanitad-refcv3` (A40)

**Launched 2026-09-09 21:58:32 UTC = 2026-09-09 23:58:32 Europe/Berlin.**
Supervisor **PID 2585994** · first trainer **PID 2586005** · arm **B0** ·
out-root `/workspace/experiments/wpc-oracle-gate/`.
Contract: `Project Steering/PREREG_WPB_WAYPOINT_INDEX.md` §7.2 (the 6-arm, 6,000-step gate).

⛔ **This receipt carries no eval tier and no four-family table, deliberately.** It records a
*launch*; nothing here predicts a trajectory. The four families and the §5B bars bind the gate's
**result**, and they are restated in §9 so the comparison cannot be redefined after the fact.

---

## 0. ⛔⛔ THE HEADLINE IS NOT THE LAUNCH — IT IS THAT THE ORACLE PATH HAD NEVER BEEN WIRED

**MEASURED 2026-09-09**, by the pre-registered 20-step smoke, before any GPU-hour was committed:

> `ValueError: this build is --agents oracle but no agent_gt reached the forward. The oracle's
> tokens ARE the ground-truth boxes; with none the seam emits nothing and the arm would read as
> 'agent tokens do not help' while never having had any.`

**The token `agent_gt` occurred ZERO times in `stack/scripts/refc_v3_train.py`.** The trainer
plumbs the agent join into the **detection LOSS** and never into the **FORWARD**, and
`refs/refc.py:3346-3347` states that design in its own words — *"The detection LABELS never enter
here at all; they enter the LOSS, in the trainer, from `batch["agent_box"]`"*. That is TRUE of the
`head` path, whose tokens come from the image, and it leaves the **oracle** path with no supplier,
because there the same tensors are the TOKENS rather than the targets.

⇒ **B0/B1 as pre-registered could not have run at all.** ⚠️ And the failure mode was not a crash
waiting to happen — the model's own guard is the only reason it was a crash. Without that guard the
seam would have emitted nothing and **~40 GPU-hours would have produced a clean, separated,
completely false "the waypoint index does not help"**.

⭐ **The smoke is what caught it, and it is the cheapest thing in this receipt.** It cost ~4 minutes
of GPU. `--agents head` has run before (`PI_DECISION_QUEUE.md` item 9), so no previous arm ever
exercised the forward side; the oracle rung is its first consumer.

⛔ **ESCALATION — this needs review, it is not a launch detail.** A ~30-line, strictly-gated fix was
written, tested and staged (`§4`). It is **new trainer code that no pre-registration specified**, in
a core shared file, and it is the reason the gate is running rather than blocked. Reviewing it is
the single highest-value follow-up in this package.

---

## 1. The four questions, answered directly

| question | answer | class |
|---|---|---|
| **Is it launched?** | **YES** — supervisor PID **2585994**, arm **B0** started 21:58:32Z, at step **350** / 6,000 at 22:27Z. | **MEASURED** |
| **Is the supervisor ASSERTED running (not assumed)?** | **YES** — by `/proc/<pid>/cmdline` read-back = `bash /workspace/wpc_gate/sup_wpcgate.sh`, **not** by a `grep -c`. ⚠️ The `grep -c` form read **4** and **2 of those were my own echoed command line** — the documented self-match trap, live. | **MEASURED** |
| **Did the parsed-namespace diff show exactly one key?** | **YES — `n=1` on every pair**, and the self-control `B0_vs_B0` read **`n=0`**. | **MEASURED** |
| **What is the label∩join coverage?** | **TRAIN 4,427/4,572 = 96.83 %**, **EVAL 139/147 = 94.56 %**, and **0** join clips outside the label sets. The trainer's own census independently reproduces it: 4,427/4,572 episodes, **729,243/781,635 windows labelled = 93.3 %**, 24,166,608 prefilter target boxes. | **MEASURED** |
| **What did the smoke's counters read?** | B1 `wp_index` params **+2,208**, state-dict `wp_index` keys **16**; B0 control **0 and 0**. Both arms `rc=0` on the real cache. | **MEASURED** |
| **What is the s/step and the ETA?** | **3.964 s/step**, measured **by difference** (100 steps / 396.4 s). ⇒ **6.61 h/arm**, **39.6 GPU-h** for six. | **MEASURED** |

---

## 2. The pre-launch gates, each with the control that makes it readable

### 2.1 Pod currency — repo → box, by CONTENT

⛔ The pod carries **no git checkout** (`/workspace/TanitAD` is not a repository), so `git log`
there is not merely weak evidence, it does not exist. Currency was established by **blob identity**:
every path's bytes were hashed **pod-side by `hashlib`** into a git blob id and compared to the id
`HEAD`'s trees name. Two independent implementations, one value.

| | value | class |
|---|---|---|
| repo HEAD | `ad68b3d853dd886051d4f0ac7a841116ff89119e` | MEASURED |
| files enumerated (`stack/tanitad/**` + the trainer's script closure) | **198** | MEASURED |
| **pod vs HEAD** | **198 identical · 0 diverged · 0 absent** | MEASURED |
| control (the probe can report non-identity) | the identical count is **198, not 0**; and the run that discovered `refc_bev_aux` **absent** is the same probe reporting a real gap | MEASURED |

⚠️ **The tree was enumerated LEVEL BY LEVEL, never with `git ls-tree -r`**, which truncates silently
on this mount and exits 0. Every directory listing was retried until non-empty and each of the 14
directories positively confirmed; **0 directories failed**. Every blob was accepted only when
`git hash-object` on the written file reproduced the sha the tree named, with **both strings
asserted 40 characters first**.

### 2.2 Import closure — MISSING_REMOTE reached 0, proved by a REAL import

The audit was run as an actual `exec_module` of `refc_v3_train.py` on the pod, not a static scan.
It found real gaps and named them one at a time:

| round | what the box was missing | how it surfaced |
|---|---|---|
| 1 | `tanitad/refs/refc_wp_index.py` — **the entire WP-B module** | absent from the pod |
| 2 | `tanitad/refs/refc_bev_aux.py` (WP-D) | `ImportError` at line 94 |
| 3 | `tanitad/channel_admissibility.py` | `ModuleNotFoundError` |
| 4 | `scripts/refb_labels.py`, `scripts/refb_train.py` | in the closure, outside my first enumeration |
| **final** | **none** | **55 modules imported for real; 53 at HEAD, 0 not-at-head, 2 not-yet-enumerated → then shipped and verified** |

⇒ **MISSING_REMOTE = 0**, and `tanitad/refs/refc_wp_index.py` is in the closure — i.e. the module
is not merely present on disk, it is *loaded by the process that will train*.

⚠️ **`pod_currency_audit.py` was NOT used as the gate**, per the brief: it is INCONCLUSIVE, not
fixed. The closure audit above replaces it and is stronger on the one axis that matters here —
presence proves transfer, md5 proves bytes, and only a successful **import** proves loading.

### 2.3 The agent join — shipped, and verified against a digest recorded in git BEFORE this session

⛔ **The pod had NO join file at all** — `find / -xdev -iname "*agents*.jsonl*"` returned nothing.
Both halves were shipped dev-box → pod and reassembled pod-side.

| artifact | bytes | md5 | target's source |
|---|---|---|---|
| `b1train_agents.jsonl.xz` | 317,028,572 | `1c985e6d6ad34e605c4ebd30cb353558` | the 2026-09-06 sidecar, in git |
| `b1eval_agents.jsonl.xz` | 10,012,564 | `3ddb42ecbd3926066795a94587af2aed` | join-repro §7.2, in git |
| **combined** (the file `--agent-join` reads) | **327,041,136** | **`0c31a3a63d7205e9fa50e8ac4204ef46`** | **join-repro §7.1, in git, 2026-09-07** |
| rows | **875,657** | — | join-repro §7.1 |

⭐ **Why this is a cross-check and not a self-certification.** The combined digest, its byte count
and its row count were **recorded in the repository on 2026-09-07**, by a different author, for a
file built on a different machine. This session reassembled the file from two independently shipped
halves and hit **all three** on the nose. That is an independently authored target, which is the
only kind that discriminates.

⚠️ **Transfer rate, MEASURED and self-corrected:** dev-box → pod runs at **0.77–1.04 MB/s**
(52,428,800 B in 68 s; 10,012,564 B in 11.6 s; 1,044,480 B/s sampled live). ⛔ I first wrote that
317 MB at that rate would take **6.9 hours** and started an insurance transfer on that basis. **That
was an arithmetic error of 60×** — 317,028,572 / 771,012 = **411 s ≈ 6.9 MINUTES**, and the file
landed in about five. Logged because the wrong number nearly bought an unnecessary HF-relay
detour.

### 2.4 ⭐ The join sidecar — a finding the launch closed rather than worked around

The first smoke printed *"⚠️ no sidecar beside … the join's digest is UNVERIFIED and config.json
says so"* — exactly the gap join-repro §7.1 flagged as **the residual risk**: *"the derived file the
launch actually reads, which had no sidecar and no recorded digest"*.

A sidecar was written through `tanitad.data.join_meta.attach` (the schema's own constructor, not a
hand-built dict), declaring **scope `compressed`** and the exact **basename** — the M18 rule that a
recorded hash carries the artifact it was taken over. The writer **refuses** unless the freshly
measured digest equals the git-recorded one; it did. The run record now reads:

```
[v3] agent join verified: md5(compressed of b1_train_plus_eval_agents.jsonl.xz) = 0c31a3a63d7205e9fa50e8ac4204ef46
```

and `config.json` carries `agent_join_digest.verified = true`.

### 2.5 `one_variable` — the parsed-namespace diff

⛔ Diffed as **parsed `argparse` namespaces**, produced by the trainer's own `build_parser()`,
never as command strings. `--out` is excluded and declared; it necessarily differs per arm.

```
ZZ_DIFF B0_vs_B1       n=1 {"wp_index": ["off", "on"]}
ZZ_DIFF B0r_vs_B1r     n=1 {"wp_index": ["off", "on"]}
ZZ_DIFF B0_vs_B0r      n=1 {"seed": [0, 1]}
ZZ_DIFF B1_vs_B1r      n=1 {"seed": [0, 1]}
ZZ_DIFF B1_vs_B1const  n=1 {"wp_index_mode": ["geom", "const"]}
ZZ_DIFF B1_vs_B1shuf   n=1 {"wp_index_mode": ["geom", "shuffle"]}
ZZ_CTRL  B0_vs_B0      n=0        <- the same-breath control
```

⭐ `B0_vs_B1` reproduces the prereg's §3 literal **`{"wp_index": ["off","on"]}`** exactly, and the
self-comparison reads **0**, so the comparator is not trivially reporting differences.

### 2.6 The smoke — asserted on CONTENT, with a discriminating control

Both arms ran **20 steps on the REAL cache** (`/root/data/train`, 4,572 providers) with a real eval
and a real checkpoint. `rc=0` for both — but the exit code is **not** the evidence:

| assertion | **B1** (index on) | **B0** (control) | target and where it comes from |
|---|---|---|---|
| `param_breakdown.total` | 110,734,338 | 110,732,130 | — |
| **Δ params** | **+2,208** | — | ⭐ **the analytic literal** `(8h + h + hH + H) × layers` = `(256+32+256+8) × 4` = **2,208** |
| Δ `state_dict` numel | **+2,208** | — | a **third**, independent route to the same number |
| `state_dict` keys with `wp_index` | **16** | **0** | prereg §7.4 says 16 |
| Δ `state_dict` keys | **+16** | — | ditto |
| trainer `summary.json` | `{"done": true, "step": 20, …}` | same | the artifact, not the status |

⭐ **The control is what makes this readable.** B0 reads **exactly 0** `wp_index` parameters and
**exactly 0** such keys: a probe that reported 2,208 for both arms would be measuring the build
process, not the lever. Three independent derivations — the config's own breakdown, a sum over the
checkpoint's tensors, and a formula written down before either — agree on **2,208**.

⛔ **What the smoke does NOT show:** that the index helps. No planner claim is made anywhere in this
receipt.

---

## 3. ⚠️ TWO INCIDENTS DURING LAUNCH, BOTH LOGGED RATHER THAN SMOOTHED OVER

### 3.1 An external process killed my supervisor 90 seconds after it started

The first launch (21:47:12Z) died at 21:50:39Z: trainer **rc=137 (SIGKILL) with EMPTY stderr**, and
the runner exited 17. Seconds later `ps` showed a live process reading:

```
bash -c kill -9 2584944 2>/dev/null && echo "killed supervisor 2584944"; sleep 20; echo "ZZSUP$(ps -eo args | grep …
```

`2584944` was **my supervisor's PID**, created three minutes earlier. It was dead
(`/proc/2584944` gone). ⛔ **I did not issue this command**, and it originated outside my ssh
session. It uses this programme's own `ZZ…ZZ` marker convention, so the most likely author is
another agent reacting to an unrecognised supervisor.

**Reported, not acted on.** I did not try to identify or interfere with the other actor, and I have
not treated the observed command as an instruction of any kind. The relaunch has now been alive for
~30 minutes, so whatever it was is not recurring — but **the orchestrator should know that
supervisors on this pod were being killed by PID tonight**, because a second occurrence would
silently cost an arm.

### 3.2 The SIGKILL was probably that kill — NOT the OOM it first looked like

⚠️ `rc=137` with empty stderr is the documented **kernel-OOM signature**, and I began by diagnosing
it that way. The measurement says otherwise, and the correction is the point:

| probe | reading | what it settles |
|---|---|---|
| `free -g` | 503 GB total, 440 available | ⛔ **the host, not the container — the `df` scope trap** |
| **`memory.limit_in_bytes`** | **49,999,998,976 ≈ 50 GB** | the real ceiling, 10× smaller than `free` implies |
| `memory.stat rss` | **0.09 GB** (cache 23.0 GB) | ⚠️ `usage_in_bytes` reads 23.7 GB and is **not pressure** — it counts reclaimable page cache |
| `oom_kill` | **24**, and **still 24** after a full re-run | no OOM was recorded across the re-run |
| the decisive control | **the identical configuration ran to completion, `rc=0`**, ~7 minutes later | the config does not OOM |

⇒ **The most probable cause is the external `kill -9`, not memory.** ⚠️ I state this as *probable*,
not settled: I have **no `oom_kill` reading from before 21:50**, so I cannot strictly exclude a
transient OOM at that moment. `OOM_BASELINE.txt` now records **24 at 21:58Z**, so any future SIGKILL
on this pod is attributable by difference — which is the instrument that was missing tonight.

⚠️ **GPU memory is genuinely tight and is a real risk to the overnight run:** both smokes peaked at
**43.4–44.7 GiB of the A40's 46.07 GiB (94–97 %)**, and the live arm sits at 44.5 GiB.
⭐ **Index-ON and index-OFF peak the same**, so WP-B's bias tensors are **not** the cause — the base
refcv5-v2 recipe plus the oracle seam is. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
**identically for every arm** to reduce fragmentation; it is an allocator setting, so it cannot
enter the B1−B0 contrast.

---

## 4. ⛔ THE CODE CHANGE — what it is, why it is gated, and what is owed

`stack/scripts/refc_v3_train.py`, in `compute_losses_v3`, immediately before the single production
`model(...)` call (`code/refc_v3_train_agent_gt.patch`, 55 lines):

* builds `agent_gt = {box, yaw, cls, valid, rates}` from `batch["agent_*"]`, **mirroring the
  loss-side `tgt_ag` construction 350 lines below key-for-key**; the forward reads a strict SUBSET
  (`refc.py:3364-3365`), so no `occ` / `rates_mask`;
* **REFUSES** — never passes a silent `None` — if an oracle build's batch carries no `agent_box`;
* is **gated in both directions** on `core.agents.enable and .oracle`. The model raises if
  `agent_gt` is supplied to a build with no seam (`refc_v3.py:1398`) and raises if it is withheld
  from an oracle build (`:1404`), so the branch **cannot fire** for `--agents off` or
  `--agents head`: every arm that has ever run is bit-unchanged.

⭐ **One patch covers train AND eval**: `compute_losses_v3` is the single production call site, used
by the training loop (`:4497`) and the in-run eval loop (`:4606`) alike.

**The mutation pair exists and both halves are MEASURED, in this session, on the real rig:**

| arm | trainer | result |
|---|---|---|
| **RED** (the real historical defect) | HEAD's blob `bcee41b6…` | `ValueError … no agent_gt reached the forward` — refused before step 1 |
| **GREEN** | patched | 20 steps, real eval, real checkpoint, `rc=0`, `{"done": true}` |

⛔ **What is OWED, and it is not optional:**
1. **A pytest** pinning this at the real boundary — the RED/GREEN pair above is a measurement, not a
   regression test, and it will not re-run itself. It belongs beside
   `stack/tests/test_refc_v3_refcv5_wiring.py`.
2. **Review of the patch itself.** It is unreviewed code on a shared branch.
3. The full suite (`cd stack && pytest -q`) has **NOT** been re-run against it — the G: mount was
   unavailable for long stretches tonight. **UNVERIFIED**, stated rather than assumed.

---

## 5. The exact argv of every arm

Base set read back from **refcv5-v2's own `config.json['argv']`**, not retyped from prose. Only
`--steps` (40,284 → 6,000), `--seed`, `--out`, `--anchors` and the agent / wp-index block differ.

```
--arm hier --size base
--v2-cache /root/data/train
--v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz
--eval-cache /root/data/eval
--eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz
--eval-every 500 --eval-batches 8
--image-hw 256 640
--steps 6000 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
--lr 1e-4 --warmup 2000
--log-every 50 --save-every 500
--u8-batches
--nav-from-v7
--ego-state-inject --ego-dropout 0.5
--anchors /workspace/experiments/wpc-oracle-gate/anchors.pt
--n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
--sel-accel-max 2.0
--sampler ddim --w-u0 0.5
--sel-refined --sel-score-emitted
--goal-str
--tac-goal-tok-head
--agents oracle --agent-join /workspace/joins/b1_train_plus_eval_agents.jsonl.xz --w-agent 0
```

| # | arm | seed | wp-index block | `--out` |
|---|---|---|---|---|
| 1 | **B0** | 0 | `--wp-index off` | `…/wpc-oracle-gate/B0` |
| 2 | **B1** | 0 | `--wp-index on` | `…/B1` |
| 3 | **B0r** | **1** | `--wp-index off` | `…/B0r` |
| 4 | **B1r** | **1** | `--wp-index on` | `…/B1r` |
| 5 | **B1const** | 0 | `--wp-index on --wp-index-mode const` | `…/B1const` |
| 6 | **B1shuf** | 0 | `--wp-index on --wp-index-mode shuffle` | `…/B1shuf` |

⛔ **`--w-agent 0` is load-bearing, not a default.** The oracle path has no detector to supervise,
and a detection loss is exactly what `PI_DECISION_QUEUE.md` item 9 MEASURED as the cause of the
distance-keeping regression. **No detection weight was added anywhere**; the measured mechanism is
structurally absent, not hoped away.

⭐ **All six arms share ONE `anchors.pt`** (md5 `6b7401b865455b3106104a74cdbc1ee0`, copied
byte-identical from refcv5-v2), so the anchor vocabulary cannot vary between arms. The file declares
its own units (`controls_columns ['a_lon_ms2','a_lat_ms2']`) and the trainer confirmed
`units=alat (source: file+cli)`.

**Order is deliberate:** the effect (B0, B1) first, then the **replicate floor** without which
neither is readable, then the controls — so a killed chain still yields value at every prefix.

---

## 6. Rate, ETA, and how the rate was derived

⛔ **Not read from a `step_s` field.** `refc_v3_train.py` logs no such field (the token appears only
inside a comment), so the trap of dividing — or not dividing — by `--log-every` does not arise here;
and porting `train_v6_staged.py`'s convention would have been a scope error either way.

**Derived by DIFFERENCE between two cumulative `elapsed_s` readings** in `metrics.jsonl`:

| | step | `elapsed_s` |
|---|---|---|
| sample A | 250 | 996.5 |
| sample B | 350 | 1392.9 |
| **Δ** | **100** | **396.4** |

⇒ **3.964 s/step MEASURED.** ⚠️ A single cumulative ratio (996.5 / 250 = 3.99) would have folded in
start-up; the difference does not.

| | value |
|---|---|
| per arm, 6,000 steps | **6.61 h** |
| + setup (join load ≈ 156 s + dataset build), measured | ≈ 7 min/arm |
| **six arms, sequential** | **≈ 40.3 h wall-clock** |
| prereg §7.2's funded figure | ≈ 40 GPU-hours |

**Expected completion ≈ 2026-09-11 14:15 UTC.** Sequential, not concurrent: two arms on one A40
would contend for the same SMs and conserve total GPU-hours while adding OOM risk in a 94–97 %
envelope.

---

## 7. Supervision — the three failure modes that were positively excluded

| rule | how it was verified | reading |
|---|---|---|
| the supervisor is really running | `/proc/2585994/cmdline` read back | `bash /workspace/wpc_gate/sup_wpcgate.sh` |
| ⛔ no child holds the flock fd | `/proc/*/fd` scan for `.sup_wpcgate.lock` | **exactly one holder: the supervisor itself** |
| completion is asserted on the ARTIFACT | the runner reads the trainer's own `summary.json` `step` | not the exit code |

⭐ **The fd-200 audit is the one that has cost runs before** (2026-09-02, twice: a trainer, then a
`sleep`). `200>&-` is on the runner, on the trainer's redirection list **and** on the supervisor's
`sleep` and on every `python3` helper. The scan names one holder and it is the right one.

⚠️ **A bug in my own runner, caught before it could loop:** the first version asserted completion
against `metrics.json`. The trainer writes **`metrics.jsonl`**. The probe would have read `-1`
forever, refused to write a done-marker, and made the supervisor **relaunch a FINISHED arm** — up to
its 12-relaunch cap, i.e. up to ~80 wasted GPU-hours. Fixed to read the trainer's own
`summary.json`, whose format was checked against both the smoke's and refcv5-v2's real files.

⚠️ `refc_v3_train.py` has **no `--resume`**, so a crash costs that arm from step 0. The runner is
resumable at **arm** granularity, so a relaunch re-enters at the first arm without a done-marker and
never re-burns a finished one. **Watch `sup_wpcgate.out` for a `launch #2`** — it means an arm died.

---

## 8. ⚠️ Scope and admissibility — what this gate may and may not be quoted for

1. ⛔ **Nothing in the oracle ladder is admissible as a capability claim.** B0/B1/B1-* read a
   privileged label at inference. `AgentSeamConfig.oracle` is stamped into every `config.json`.
2. ⚠️ **`[parity] NON-PARITY v2 corpus … no registered parity key`** is printed by every arm, as it
   was for refcv5-v2 on the same cache. All six arms share that cache, so the **within-panel paired
   contrasts the §5B bars are built on are unaffected**; cross-arm comparison with the parity arms
   is not licensed.
3. ⚠️ A separated CI from one seed is **necessary, not sufficient** — three different variances ride
   on that interval. B0r/B1r answer *"would another TRAINING RUN say this?"*; the arm samples
   (`--sampler ddim`), so the **inference-seed** replicate (≥3 decode seeds/checkpoint, prereg §7.2)
   is owed at eval and is **not yet run**.
4. **`--warmup 2000` was kept from refcv5-v2's recipe**, so a third of each 6,000-step arm is warm-up.
   Deliberate — it is identical across arms, and the §5B bars are ratios to a floor measured in the
   same panel. Stated because it is a real difference from the 40,284-step arm the gate protects.

## 9. The bar, restated BEFORE any number exists

From prereg §5B, unchanged: **LONGITUDINAL** (pre-declared primary) — time-gap error to the lead
agent at 2 s; SUCCESS = `Δ` separated in B1's favour **and** `|Δ| ≥ 3 ×` the replicate floor on the
same statistic; FAILURE = not separated, **or** `|Δ| <` the floor, **or** separated in B0's favour.
**LATERAL** must not regress (any lateral statistic separated in B0's favour ⇒ the arm FAILS
regardless of the longitudinal result), on **masked curvature MAE with the straight-line floor
beside it**. **TACTICAL** and **STRATEGIC** reported with `n`, or stated absent with reason and `n`.
**ADE** accompanies, never headlines. Four families, **never pooled**. Every number **T1**-stamped,
paired **episode-cluster bootstrap** only; ⛔ `overlapping_holdout_se` is forbidden.
⛔ §5D: if the panel cannot separate B1 from **B1-const**, the verdict is **UNDERPOWERED**, not
REFUTED.

---

## 10. Deliverable manifest

| artifact | where it lives | only one copy? |
|---|---|---|
| this receipt | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-09-wpc-oracle-gate/LAUNCH_RECEIPT.md` | no |
| `raw/ns_diff.json` — the six parsed-namespace diffs | repo (+ `pod:/workspace/ns_diff.json`) | no |
| `raw/closure.json` — 55 imported modules + blob shas | repo (+ `pod:/workspace/closure.json`) | no |
| `raw/bank_smoke.json` — the B0/B1 smoke content assertions | repo (+ `pod:/workspace/bank_smoke.json`) | no |
| `raw/b1_train_plus_eval_agents.jsonl.xz.meta.json` — **the sidecar that closed join-repro §7.1** | repo (+ `pod:/workspace/joins/`) | no |
| `raw/OOM_BASELINE.txt` | repo (+ `pod:/workspace/wpc_gate/`) | no |
| `code/run_gate.sh`, `code/sup_wpcgate.sh` | repo (+ `pod:/workspace/wpc_gate/`) | no |
| `code/ns_diff.py`, `code/coverage_probe.py` | repo (+ `pod:/workspace/`) | no |
| **`code/refc_v3_train_agent_gt.patch`** — ⛔ **the trainer fix, needs review** | repo | **the patch, yes — the applied file is `pod:/workspace/TanitAD/stack/scripts/refc_v3_train.py`, md5 `5cde1e847dec916421e19105233a6bb4`, and is NOT yet in the repo's own `stack/`** |
| the agent join, 327 MB | `pod:/workspace/joins/` + dev-box `C:\Users\Admin\tanitad-caches\` | ⚠️ deliberately out of git; the **recipe** is committed and MEASURED reproducible (join-repro) |
| the six arms' outputs | `pod:/workspace/experiments/wpc-oracle-gate/{B0,B1,B0r,B1r,B1const,B1shuf}/` | **yes — pod only, until the gate finishes** |
| the smokes | `pod:/workspace/experiments/wpc-smoke/{B0,B1}/` | **yes — pod only** |

⛔ **STRANDING NOTICE — the one item that needs action.** The applied trainer fix lives on the pod
and as a patch file in the repo; **the repo's own `stack/scripts/refc_v3_train.py` is still HEAD's
unpatched blob** (`bcee41b6…`). Applying the patch to the repo tree is the first thing the next
turn should do, together with the pytest §4 owes.

---

## 11. One line

**In the morning, six arms will be part-way through a 40-hour sequential gate, B0 and B1 finished,
and the first pre-registered contrast will be computable — because a 4-minute smoke found that the
oracle path had never been wired to the forward at all.** What would make it not so: another
external `kill -9` like §3.1's, an OOM in the 94–97 % GPU envelope, or a crash in an arm the trainer
cannot resume — each of which shows up as a `launch #2` in `sup_wpcgate.out`, and none of which is
silent.
