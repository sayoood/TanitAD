# flagship-v4 30k FORMAL GATE — end-to-end dry run (2026-07-25)

**Agent:** v4-gate-dryrun subagent · **Pod:** `tanitad-eval` ONLY (A40; pod1/pod2 untouched)
**Checkpoint under test:** `Sayood/flagship-v4.2b` (HF, gated-manual) — a **known FAILED arm**, used
purely as a pipeline carrier. *Nothing below is a claim about v4.2b's model quality.*

Evidence classes: **MEASURED** (ours + artifact path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## HEADLINE — is the 30k gate ready to run? **NO.** Three blockers, one of them fatal today.

| # | blocker | severity | status |
|---|---|---|---|
| **B1** | The v4 trainer **never writes `g_op_fwd_ade_m`**. This does not merely make `speed_benefit_recovered_frac` read NOT SUPPLIED — it makes `run_gate.py check` **`SystemExit` with `REFUSING: no matched steps for g_op_fwd_ade_m` before it evaluates anything at all.** The 30k gate would produce **no verdict**, not an INCOMPLETE one. | **FATAL** | root-caused; **one-line bug FIXED + regression test** (below). Fix cannot retro-fill the in-flight arm → **needs a card decision from Sayed**. |
| **B2** | `deploy_tick_p99_ms` has **no v4 emitter input**. `taniteval.efficiency` is registry-driven, has **zero v4 awareness**, and no `eff_levers_*` panel has ever been produced for any v4 arm. Worse, a trunk-only panel would be **wrong**: v4's deployed tick includes the FlagshipV4Head's truncated-denoise passes, which `efficiency.py` does not model. | HIGH | **not fixed** — needs a v4-aware lever panel. I deliberately did **not** emit a trunk-only number that would have looked like a measurement. |
| **B3** | `nonav_route_beats_majority` is now **emittable for v4** (I built the path) and **it returns 0 → FAIL**. With the nav command withheld, v4.2b's route head predicts **straight on 240/240 valid windows**; accuracy `0.6708` equals the majority-straight rate `0.6708` exactly. A pure command echo, inherited from the v1 trunk. | HIGH | **path built + MEASURED.** As a KILL secondary this **fails the 30k gate on its own**, independent of ADE, unless the P6 strategic planner lands first. |

**Also load-bearing, and cheap to fix (both done here):**

- **B4 — the gate PRIMARY was being silently dropped.** `eval_flagship_v4.py` imports `taniteval`
  *non-fatally*. With the PYTHONPATH its own docstring documents, the import fails and the harness
  writes `gate_primary_ade_0_2s: {"value": null, "note": "NOT COMPUTED"}` **and exits 0**. My first
  run lost the primary *and* `miss_at_2m` this way and looked successful. `taniteval`'s **parent**
  (`/root/taniteval`) must be on PYTHONPATH. Runbook below has the preflight that makes this loud.
- **B5 — the pod's `run_gate.py` is stale.** `/root/run_gate.py` is **621 lines** and predates the
  2026-07-21 hardening; the repo's is **847 lines** and adds the `miss_at_2m`/`miss_2m`/`miss_rate@2m`
  alias resolution, the k-consecutive `reference_reached_at` fix, and the refusal to decide on the
  deprecated `overlapping_holdout_se` interval. **The 2026-07-23 10k gate JSON was rendered by the
  stale tool.** I installed the current version at `/root/v4eval/stack/scripts/run_gate.py`
  (md5 `de09a9e8…`, matches repo); `/root/run_gate.py` is left in place and **must not be used**.

**The machinery itself is sound.** With all eight secondaries supplied the gate renders a
**COMPLETE** verdict (`VERDICT: RESTART — pre-registered gate failed`), not INCOMPLETE — artifact
`raw/gate_RUN_B_machinery_probe_COMPLETE.json`. The gap is **upstream inputs**, not the gate tool.

---

## P0 — housekeeping (done)

Three GPU processes (1202 / 942 / 848 MiB) were **orphaned deadlocked GeoCalib jobs**, not live work:

- all three parents were `bash -c …geocalib_work…` **reparented to PID 1** (launching ssh long gone);
- **zero CPU progress** across a 25 s sample (`00:06:46 → 00:06:47`, `04:25 → 04:25`, `01:11 → 01:11`)
  against 5 h+ elapsed, all blocked in `futex_wait_queue` with 224–239 threads;
- two were writing to **deleted** log files (`decode_check.log (deleted)`, `threadtest.log (deleted)`);
- they are the exact PyAV-teardown-with-live-CUDA deadlock that
  `incoming/2026-07-25-geocalib/NOTE.md` documents as *found and fixed*; that agent's work is
  complete and staged.

Reaped **by explicit PID** (`1597494 1598837 1605444` + parents `1597493 1598836 1605443`) — never
`pkill -f`. **GPU now 0 MiB / 0 %, no survivors.** MEASURED.

---

## P1 — checkpoint pull + integrity (PASS)

| item | value | evidence |
|---|---|---|
| source | `Sayood/flagship-v4.2b` (files: `ckpt.pt`, `config.json` — **no anchors file**) | MEASURED |
| size | 3 243 109 310 B (3.24 GB) | MEASURED |
| sha256 | `c3e7ddb4ce870b1b132dfdfaa4b77fb9de86739dd9f3338ac4b5dae98a703735` — **matches the HF LFS sha256 exactly** | MEASURED |
| pull time | **15.7 s** (~207 MB/s) to `/workspace/v4gate/ckpt/` | MEASURED |
| load | `{controller, goal_head, grounding, head, lam_mult, model, opt, phases, step}` — `model` 631 entries, `head` 98, `grounding` 42, `opt` present | MEASURED |
| **step** | **4000** (`args.out = /workspace/experiments/flagship-v4.2b-30k`) | MEASURED |

Two facts worth carrying forward:

1. **v4.2b ≠ the pod-local `flagship-v4.2-step4000`.** Identical byte size, **different sha256**
   (`02d44018…`). Same step number, different run. Size alone is not an identity check.
2. **The missing anchors file on HF is a false alarm — for this ckpt format.** The harness warns
   `no --anchors-dense found … will NOT reproduce its numbers`, but `decoder.anchors (256,20,2)` is
   **inside `ck["head"]`**, and `head.load_state_dict(ck["head"])` is STRICT and runs *after*
   `load_anchors` — so the trained anchors are restored regardless. Verified numerically: the
   ckpt's anchors are `allclose` to the trained anchors file (`method: fps, seed: 0`, built from the
   parity train cache). **Do not let this warning stall the 30k gate.**

---

## P2 — held-out eval on the clean split (`physicalai-val-0c5f7dac3b11`)

Leaky split `physicalai-val-f1b378f295ae` was **never touched**.

**MODE A (harness validation, GATE_PROTOCOL O-03)** — required before MODE B is trusted:
canary ADE@2s = **0.4214799702167511** on 881 windows / 40 episodes, **75 s**. Registry v1 full-set
reference 0.4271 → delta **−0.0056** (tol 0.05) → `HARNESS_VALIDATED: true`. Bit-identical to the
2026-07-22 run ⇒ the harness is deterministic. MEASURED (`raw/v1-validation-dryrun.json`).

**MODE B (v4.2b, planner path)** — **224 s**, 881 windows / 40 episodes:

| metric | value | card threshold | pass |
|---|---|---|---|
| **`ade_0_2s`** (PRIMARY, episode-cluster bootstrap) | **0.8604** CI **[0.7480, 0.9885]** | ≤ 0.6 | FAIL |
| `wm_canary_ade_2s` | 0.6968 | ≤ 0.55 | FAIL |
| `oracle_in_fan` (4-wp, steps 5/10/15/20) | 0.4016 | ≤ 0.30 | FAIL |
| `miss_at_2m` | 0.3473 CI [0.2611, 0.4325] | ≤ 0.10 | FAIL |
| `seam_norm_ratio_max` | 0.1136 | ≤ 1.0 | PASS |
| `encoder_touching_levers` | 2 | ≤ 2 | PASS *(PUBLISHED design fact, not a GPU measurement)* |

The primary's two independent code paths **agree**: self-computed 0.86045 vs
`taniteval.driving.from_windows()` 0.8604, `agree_within_1pct: true`. Estimator is
`episode_cluster_bootstrap` over the 40 val episodes (n_boot 2000) — decision-grade per CLAUDE.md.

*(Again: v4.2b is a known failed arm at step 4000. These numbers exercise the pipeline; they are not
a finding about the v4 line.)*

---

## P3 — the three emitters, one at a time

**All three emitters are CORRECT code. `gate_emitters.py` reads panels; it has no checkpoint path
at all.** Every gap below is a missing *upstream input*, not a broken emitter.

### `speed_benefit_recovered_frac` — **the real point of the exercise. It cannot fire for any v4 arm.**

| run | value | rows |
|---|---|---|
| flagship **v1** (`v1-speedjerk_train_log.jsonl`) | **0.8184 → PASS** | n_arm 40 / n_nospeed 40 |
| flagship **v4.1-10k** (a real v4 arm) | **None → NOT SUPPLIED** | **n_arm 0** / n_nospeed 40 |

The emitter reproduces the design's 81.8 % headline exactly on v1 and correctly refuses to fabricate
on v4. `n_arm_rows = 0` is the whole story: **the metric is absent from every v4 train log**
(MEASURED: `flagship-v4.1-10k` 0 occurrences, `flagship-v4.2-step4000` 0 occurrences).

**ROOT CAUSE (found, fixed, tested) — `stack/scripts/train_flagship_v4.py:143`.**

`grounding_losses` emits the key **already `g_`-prefixed** (`metric_dynamics.py:389` →
`log[f"g_{lvl}_fwd_ade_m"]`), and `flagship_loss` merges that dict verbatim (`**g_log`). So the key
present in `wm_log` is **`g_op_fwd_ade_m`**. The joint-step log filtered for the **unprefixed**
`"op_fwd_ade_m"` and then re-prefixed:

```python
**{f"g_{k}": v for k, v in wm_log.items() if k in ("op_fwd_ade_m",)},   # matches NOTHING
```

It never matched — and would have written `g_g_op_fwd_ade_m` if it had. Fixed to:

```python
**{k: v for k, v in wm_log.items() if k in ("g_op_fwd_ade_m",)},
```

**Why this stayed invisible:** the row-writer at line ~680 had *already* been patched to forward
`g_op_fwd_ade_m`, with a comment asserting *"It is already computed in `log`"*. It was not — line 143
starved it. **The earlier fix shipped and was silently inert**, and `flagship-v4.2-step4000`'s log
(written *after* that patch, and carrying its other new fields `eff_batch`/`gnorm_encoder`) proves
it: still 0 occurrences. A partial fix that reads as complete.

Regression test `test_joint_step_log_carries_g_op_fwd_ade_m` — **verified to fail on the old line and
pass on the fixed one** (not merely green). Full suite: **837 passed, 3 skipped**.

> **This fix does NOT rescue the in-flight arm.** `flagship-v4-fromscratch-30k` is already past
> step 8–10 k, a running trainer cannot be hot-patched, and the card's bucket is `(8000, 10000]`.
> The metric will exist only for runs launched **after** this fix. **Sayed's decision required** —
> see escalation 1.

### `deploy_tick_p99_ms` — emitter works; **no v4 input exists**

Verified on the v1 panel: **18.7641 ms → PASS**, lever `all_levers`, accuracy-equivalent
(`ade_0_2s_delta_m −6.6e−05`, cosine 0.99999994), `MEASURED (NVIDIA A40)`. Reproduces the documented
18.76 exactly.

For v4: `eff_levers_*` panels exist **only** for `flagship-30k`. `efficiency.py` has **0** matches for
`flagship_v4|FlagshipV4` and loads via `taniteval.registry.MODELS`, which has **no v4 entry**. A
trunk-only panel would omit the head's denoise passes (`gate_emitters.py` says so itself). **Not
emitted — deliberately.**

### `nonav_route_beats_majority` — **path built here; emits 0 and FAILS**

No hierarchy JSON existed for any arm. I ran `taniteval.hierarchy` on the clean val for **both** v1
and v4.2b via a **runtime** registry entry (no pod file edited): v4's trunk *is* the v1 fourbrain
WorldModel, so `arch="flagship-worldmodel"` loads `ck["model"]` + `ck["grounding"]` STRICT.

| arm | route_acc (cmd given) | route_acc_follow (cmd withheld) | majority straight | emitted |
|---|---|---|---|---|
| flagship-30k (v1) | — | 0.6708 | 0.6708 | **0 → FAIL** |
| **flagship-v4.2b** | **1.0** | **0.6708** | **0.6708** | **0 → FAIL** |

`follow_pred_distribution = {left: 0, straight: 240, right: 0}` — n_valid 240. Accuracy 1.0 with the
command and exact collapse to the majority class without it is the signature of a **pure command
echo**. Identical v1/v4.2b values are expected, not a bug: both collapse to constant-straight, and
constant-straight scores exactly the majority rate by construction.

*(Note: `gate_emitters.py`'s docstring quotes v1 at 0.7083/0.7083; I MEASURE 0.6708/0.6708. The
emitted gate value is 0 either way, but the docstring figure should not be re-quoted as-is.)*

---

## P4 — does the gate render COMPLETE? **Yes — once all eight are supplied.**

**RUN A (honest, v4.2b's own log, 7 of 8 obtainable)** →
`[gate] REFUSING: no matched steps for g_op_fwd_ade_m`, **no verdict, no JSON written.**
This is B1 in action: the card carries `reference_log` + `compare_metric: g_op_fwd_ade_m`, so
`matched_step_ratio()` raises `SystemExit` (`run_gate.py:298`) **before** primary/secondary
evaluation. Omitting `--reference-log` does not help — the *card* supplies it.

**RUN B (machinery probe, all eight supplied)** → renders fully:

```
[primary]   ade_0_2s = 0.8604 (cluster_bootstrap, episode_cluster_bootstrap; headline)
            CI [0.748, 0.9885]  <= 0.6  -> FAIL
[secondary] wm_canary_ade_2s 0.6968 FAIL · speed_benefit_recovered_frac 0.8184 PASS
            oracle_in_fan 0.4016 FAIL · miss_at_2m 0.3473 FAIL · seam_norm_ratio_max 0.1136 PASS
            encoder_touching_levers 2.0 PASS · deploy_tick_p99_ms 18.7641 PASS
            nonav_route_beats_majority 0.0 FAIL
VERDICT: RESTART - pre-registered gate failed
```

**`COMPLETE`, not `INCOMPLETE`** — every one of the eight was adjudicated. That is the deliverable.

**Honesty markers on RUN B:** `--log` was substituted with **v1's** train log (the only log carrying
the card's compare-metric), and `deploy_tick_p99_ms` / `speed_benefit_recovered_frac` are **v1's real
emitter outputs, not v4 measurements**. Consequently the matched-step ratio block is degenerate
(1.0, v1-vs-v1) and must be ignored. The probe proves **machinery**, not v4's verdict.

---

## P5 — RUNBOOK for the real 30k gate

### Preconditions that must be true BEFORE the ckpt lands

1. **Ship the arm's `train_log.jsonl` off pod2 with the checkpoint.** `run_gate.py check --log` is
   mandatory and the log lives only on the training pod. Push it to the HF repo next to `ckpt.pt`
   (~118 MB/s) — the dev-box relay is ~1 MB/s.
2. **Decide B1** (escalation 1). Without a decision the gate emits no verdict at all.
3. **Use `/root/v4eval/stack/scripts/run_gate.py`** (installed today, md5 `de09a9e8…`).
   **Never `/root/run_gate.py`** — stale, 621 lines.
4. **Register a 30k card.** `flagship-v4.card.json` has `gate_step: 10000`. GATE_PROTOCOL §2
   (Multiplicity) allows **one** pre-registered gate step; re-checking the 10k card at 30k is
   exactly the "look at every milestone and decide" the protocol forbids. Register **before** the
   run reaches 30k or the gate is inadmissible (`run_gate.py register` **refuses to overwrite**).

### The sequence (MEASURED wall-clocks, A40, 40 episodes / 881 windows)

```bash
# 0) PREFLIGHT — makes B4 loud instead of silent.  (~5 s)
export PYTHONPATH=/root/v4eval/stack:/root/v4eval/stack/scripts:/root/taniteval
python3 -c "import taniteval, taniteval.driving; print('taniteval OK', taniteval.__file__)" || exit 3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # must be EMPTY (eval needs the GPU alone)

# 1) PULL + INTEGRITY                                             (~20 s, 3.24 GB @ ~207 MB/s)
#    -> /workspace, NOT /root: the container overlay is 99% full (3.0 G avail; a 4 GB dd short-writes).
#    Verify sha256 against the HF LFS sha. Size alone is NOT identity (v4.2b vs v4.2-step4000).

# 2) MODE A — harness validation, GATE_PROTOCOL O-03. MUST pass first.        (~75 s)
python3 scripts/eval_flagship_v4.py --ckpt /root/models/flagship-30k/ckpt.pt --canary-only \
  --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --key v1-validation --out <out>/v1-validation.json --results-dir <out>
#    expect canary 0.42148 (registry full-set 0.4271, tol 0.05) and HARNESS_VALIDATED: true

# 3) MODE B — the arm. Emits primary + 5 secondaries + windows_<key>.pt.     (~225 s)
python3 scripts/eval_flagship_v4.py --ckpt <30k ckpt> \
  --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --key flagship-v4-fromscratch-30k --out <out>/<key>.json --results-dir <out>
#    ASSERT afterwards: gate_primary_ade_0_2s.value is NOT null  (else PYTHONPATH -> B4)

# 4) HIERARCHY panel -> nonav_route_beats_majority                            (~5 min)
#    Runtime registry entry, arch="flagship-worldmodel", speed_input=True, action_dim=3.
#    Driver: raw/hier_driver.py in this folder.

# 5) EFFICIENCY lever panel -> deploy_tick_p99_ms                             (NOT AVAILABLE — B2)

# 6) EMIT + CHECK                                                            (~10 s, no GPU)
python3 scripts/gate_emitters.py gate-values --eff-json <eff_levers_*.json> \
  --hierarchy-json <hierarchy_*.json> --arm-log <arm train_log.jsonl>
python3 scripts/run_gate.py check --card "Project Steering/Gates/flagship-v4-30k.card.json" \
  --log <arm train_log.jsonl> --eval-json <out>/driving_<key>.json \
  --secondary-value <all eight> --json <out>/gate_30k.json
```

**Total GPU time ≈ 11 min** (75 s + 225 s + ~5 min hierarchy) + ~20 s pull. The gate is **cheap**;
the risk was never runtime, it was missing inputs.

### Gotchas hit, in the order they bite

| gotcha | symptom | fix |
|---|---|---|
| `taniteval` not on PYTHONPATH | **exit 0**, primary + `miss_at_2m` silently `NOT COMPUTED` | add `/root/taniteval` (the **parent**); run the preflight |
| `/root` overlay 99 % full | 4 GB `dd` short-writes at 3.2 GB | download to `/workspace` |
| `/usr/bin/time` absent | `exit 127`, nothing runs | use shell `date +%s` |
| card's `reference_log` | `REFUSING: no matched steps` — **no verdict** | B1 |
| stale `/root/run_gate.py` | renders where the current tool refuses; no alias resolution | use `/root/v4eval/stack/scripts/run_gate.py` |
| `--anchors-dense` warning | scary, **harmless here** | anchors live in `ck["head"]`, restored by STRICT load |
| `nohup … &` over ssh | ssh holds the connection open | `setsid nohup … < /dev/null &`; log to `/tmp` (logs on `/workspace` get swallowed) |
| PowerShell + `2>/dev/null` | PS eats the redirect (`G:\dev\null`) | base64 the script, `base64 -d | bash` |

---

## DELIVERABLE MANIFEST

| artifact | repo path (staged) | pod path |
|---|---|---|
| **This runbook** | `repo:…/incoming/2026-07-25-v4-gate-dryrun/V4_GATE_DRYRUN.md` | — |
| **Trainer fix** (`g_op_fwd_ade_m`) | `repo:stack/scripts/train_flagship_v4.py:143` | ⚠️ **NOT on pod2** (training; off-limits) |
| **Regression test** | `repo:stack/tests/test_train_flagship_v4.py::test_joint_step_log_carries_g_op_fwd_ade_m` | — |
| MODE A validation | `repo:…/raw/v1-validation-dryrun.json` | `/workspace/v4gate/results/` |
| MODE B eval + diagnostics | `repo:…/raw/flagship-v4.2b-step4000-dryrun{,_v4_diagnostics}.json` | `/workspace/v4gate/results/` |
| Driving panel (primary + CI) | `repo:…/raw/driving_flagship-v4.2b-step4000-dryrun.json` | `/workspace/v4gate/results/` |
| Hierarchy panels (v1 + v4.2b) | `repo:…/raw/hierarchy_flagship-{30k,v4.2b-dryrun}.json` | `/workspace/v4gate/results/` |
| nonav-route emission | `repo:…/raw/emit_nonav_route_v4.2b.json` | — |
| Gate RUN B (**COMPLETE**) | `repo:…/raw/gate_RUN_B_machinery_probe_COMPLETE.json` | — |
| Hierarchy driver (reusable) | `repo:…/raw/hier_driver.py` | `/tmp/hier_driver.py` |
| Current `run_gate.py` installed | `repo:stack/scripts/run_gate.py` | `/root/v4eval/stack/scripts/run_gate.py` |
| `windows_<key>.pt` (paired CI) | — | **`/workspace/v4gate/results/windows_flagship-v4.2b-step4000-dryrun.pt` — ONE PLACE ONLY** (99 KB, regenerable in 225 s; not staged as it is a binary intermediate) |

Nothing else lives in only one place.

---

## ESCALATIONS — decisions I cannot make

1. **`speed_benefit_recovered_frac` for `flagship-v4-fromscratch-30k` is unrecoverable.** The metric
   was never logged, the trainer cannot be hot-patched, and the card's `(8000, 10000]` bucket is
   already behind the run. Options, cheapest first: **(a)** amend the 30k card to drop or replace
   this secondary, **pre-registered before 30k lands**; **(b)** accept no verdict; **(c)** recompute
   it off archived 8 k/10 k milestone ckpts — changes the pinned estimator, so it needs explicit
   sign-off. **My fix only helps runs launched after it.** Sayed's call.
2. **`nonav_route_beats_majority` will FAIL the 30k gate on its own** (measured 0, command echo).
   Either P6's strategic planner lands first, or the card must be amended, or the gate fails by
   design. This is a KILL secondary — worth knowing **now**, not at the gate.
3. **`deploy_tick_p99_ms` needs a v4-aware efficiency lever panel** (must include the head's denoise
   passes). Nobody owns this. It is a KILL secondary with no path to a number today.
4. **Register a 30k card** (see runbook precondition 4) — the current card is a 10k card.
5. **The restart budget may already be exhausted.** The card says `restarts_used: 0`,
   `restart_cap: 2`, family `joint-planner-wm`, and its note assigns v4.1 to this family. v4.1,
   v4.2, v4.2b and now from-scratch look like ≥ 3 attempts. If from-scratch belongs to the same
   family, `REFUTE_LEVER_FAMILY` may already apply. I did not adjudicate this — **please confirm the
   family assignment and update `restarts_used`.**
6. **pod2 runs the pre-fix trainer.** The fix is in the repo only; it takes effect on the **next**
   launch. Do not restart the in-flight arm for it — the fix is log-only and cannot recover the
   bucket anyway.
