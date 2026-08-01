# TanitAD — return-from-limit assessment

**Written 2026-08-02 00:30 Berlin / 2026-08-01 22:30 UTC.** The weekly API limit stopped the
programme on **2026-07-29 ~09:55 UTC** and reset **2026-08-02 00:00 Berlin**. This report covers
the ~4-day gap: what finished, what died, what was lost, and what is running again now.

⚠️ **Times: pods and logs are UTC; Sayed reads Europe/Berlin (UTC+2).** Both are labelled below.

---

## 0. ⛔ First — an error of mine, stated plainly

**I killed v5 and never relaunched it. It sat dead for ~4 days.**

MEASURED, `pod2:/workspace/v5d_run.status`:

```
=== v5 EXIT 2026-07-29T09:55:23Z rc=143 ===
    rc>128 => killed by signal 15
```

`rc=143` is SIGTERM — **my kill**, executed to remove the `--heldout-gate` flag as the PI
instructed. The sequence was: kill the run → edit the launcher → relaunch. The edit step
**failed** (`sed: -e expression #1, char 11: unterminated 's' command`), leaving `run_v5e.sh`
still carrying the gate flag, and the weekly limit hit before I noticed. The relaunch never
happened.

| | |
|---|---|
| **Cost** | pod2 idle **2026-07-29 09:55 → 2026-08-01 22:14 UTC ≈ 3 d 12 h** |
| **Model state lost** | none — `ckpt.pt` survives at step 1000 (Jul 29 01:25) |
| **Progress lost** | the 1000→2000 stretch, for the **third** time |

⭐ **ROOT-CAUSE CLASS — `destructive action taken before its replacement was verified ready`.**
The kill and the fix were two separate operations. A kill is irreversible; an edit can fail
silently. Ordering them kill-then-edit means any failure in the edit leaves the pod dead with
nobody watching.

⇒ **RULE ADOPTED: never stop a running job until the replacement launcher is written AND
verified.** Prepare, verify, then kill-and-launch as one action. Logged as **C67**.

A second, compounding failure: I read `grep -c heldout` → `1` and had to check whether that was a
real flag or my own comment. It was the comment. **`grep -c` on a keyword is not a check for a
flag** — the check is "does any non-comment line carry it", which is what I now run.

---

## 1. Fleet — MEASURED 2026-08-01 22:10–22:35 UTC

| pod | GPU | state now |
|---|---|---|
| **pod2** `69.30.85.123:22091` | A40 | ✅ **v5e TRAINING** — relaunched 22:14:13Z, **gateless**, step 1050, 5 procs, **stderr 0 B** |
| **pod3** `69.30.85.16:22079` | A40 | ✅ **RR-CTL TRAINING** — launched 22:22:06Z, `[resume] resuming at step 30001`, GPU **100 %**, 276.88 M params |
| **newpod** `69.30.85.48:22192` | A40 | ⚪ free — v2corpus **DONE**; currently serving the checkpoint relay |
| **eval** `69.30.85.106:22073` | A40 | 🔵 receiving the v2corpus checkpoint; harness + val + video deps **ready** |
| **pod1** `tanitad-pod` `38.147.83.15:39198` | **8× A6000** | ⛔ **`Connection refused`** — still needs the PI's console stop/start |

⚠️ **pod1 correction:** I first probed `tanitad-pod1`, which does not exist — the alias is
**`tanitad-pod`**. That was a probe error, not a new outage. Re-probed correctly: `Connection
refused`, which per the runbook is a **stopped pod with a reassigned SSH port**, not a hang.

---

## 2. What the 4-day gap actually produced

### ✅ v2corpus 30k — FINISHED CLEANLY

MEASURED, `newpod:/workspace/v2corpus_run.status`:
`=== v2corpus EXIT 2026-07-29T16:59:18Z rc=0 ===`

| | |
|---|---|
| final step | **29999** |
| checkpoint | 3,415,816,138 B, md5 `a9372010613c2931295d66ef8b7539d3` |
| trainer-internal `g_op_fwd_ade_m` at 29,900–29,999 | 0.1366 / 0.4686 / 0.3837 |
| `erank` | 17.4–18.9 — **no representation collapse** |

⛔ **Those are TRAINER-INTERNAL readings, ~10 % optimistic vs `eval_*.py`, and NOT quotable.**
The real number requires the harness (§4).

### ✅ RR-20 — FINISHED CLEANLY

MEASURED, `pod3:/workspace/rr20.status`: `=== RRCTL/RR20 EXIT Wed Jul 29 17:35:10 UTC 2026 rc=0 ===`
Final step **31999** — a 2,000-step fine-tune of v1 at `--rollout-k 20`. Checkpoint
`/workspace/rrft/ckpt.pt`, 3.30 GB.

⚠️ It took **three** launches: the first was **correctly refused** by the parity guard
(`PARITY VIOLATION [cache-dirs/val] — 78.5 % of episodes are IN the parity train set`), the
second died on an optimizer-group mismatch, the third ran. **The guard did its job** — that
refusal prevented a void result.

### ❌ v5 — dead the whole time (§0)

### ⏸️ The real cost: **~4 days × 4 A40s idle**

---

## 3. Everything now running again

| pod | job | verification performed |
|---|---|---|
| pod2 | **v5e**, gateless | non-comment lines carrying `heldout` = **0**; `bash -n` OK; 5 procs; stderr 0 |
| pod3 | **RR-CTL**, `--rollout-k 4` | seed md5 `132d7fd87f1d01636fd5f7298d859fda` **identical** to RR-20's `ckpt_step30000.pt`; diff vs RR-20 launcher = **only** `rollout-k` and paths |

⭐ **Why RR-CTL matters:** RR-20 alone is uninterpretable. The comparison is **RR-20 vs RR-CTL**
(same seed, same steps, same pod, one flag different) — ⛔ **never RR-20 vs v1**, which would
confound rollout-k with the fine-tune itself.

⭐ **Removing the gate also removes the C65/C66 failure mode by construction:**
`heldout_gate._taniteval()` is never imported, so its hard-derived `sys.path.insert(0, …)` — which
**ignores `PYTHONPATH`** and force-loaded a stale tree — can no longer run. The gate moves offline,
against saved checkpoints, where it cannot take the trainer down.

---

## 4. The v2corpus evaluation — IN PROGRESS, and the constraint on it

⚠️ **C64 BINDS AND IS NOT NEGOTIABLE: 21 of the 40 canonical val episodes are INSIDE
v2corpus's training corpus.** A full-40 number for v2corpus is **void**.

Per `PREREG_v2corpus_vs_v1.md`, and the PI's "do both A and B":

- **Option A** — score both arms on the **19 leak-free episodes**, with **v1 RE-SCORED there**.
  v1's published **0.4271** is a full-40 number and is **NOT** comparable to a 19-episode number.
  Headline must carry `leak_free_n = 19`.
- **Option B** — a clean v2-line val from the 9,987 unselected clips. Feasibility already
  MEASURED (`fe400f0`): junction is binding, needs 368, remainder holds 2,491 = **6.77× headroom**.
- Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`), B=2000. ⛔ never
  `overlapping_holdout_se`.

### The transport problem, and what I did about it

| finding | evidence |
|---|---|
| same-DC pod→pod `scp` is **REFUSED** | `scp rc=255 elapsed=0s`, eval → newpod, both `69.30.85.x` |
| C56's 42 MB/s was **cross-DC** (US-TX-1 → ca-mtl-1) | not a contradiction — a different case |
| dev-box relay measures **~0.24–0.4 MB/s** | 3.42 GB ⇒ ~2.4 h |

⭐ **Fix: the eval never reads optimizer state.** Stripped it on newpod:

| | |
|---|---|
| source | 3,415.8 MB (keys `model`, `grounding`, `opt`, `step`) |
| stripped | **1,145.6 MB** (keys `model`, `grounding`, `step`), step **29999** |
| reduction | **3.0×** |

Relay of the stripped file is running to `tanitad-eval`, which already holds `val40cache`
(4.4 GB, 40 episodes), `taniteval`, and torch 2.8.0+cu128.

---

## 5. Long overlay videos — staged, PI request

Standard (Sayed's standing preference): **camera projection + metric BEV inset together**, plus a
text overlay of the decoded **tactical manoeuvre** and **strategic route/goal**, plus ADE.
Implemented in `taniteval/taniteval/corpus_overlay.py`.

Prepared on `tanitad-eval` this session:

- deps installed — cv2 5.0.0, imageio, imageio-ffmpeg, matplotlib 3.11.1
  (needed `--break-system-packages`; the pod is PEP 668-managed)
- `/root/taniteval/taniteval` → `/workspace/TanitAD/taniteval/taniteval`
- `/root/valdata/physicalai-val-0c5f7dac3b11` → `/workspace/val40cache` (**40 episodes visible**)
- registry import verified: 17 models, 3 corpora, `physicalai` root resolves

⚠️ **`--max-frames` defaults to 200** — must be raised for "long".
⚠️ **Every clip will be labelled leak-free vs in-train.** A clip from the 21 training episodes is
**not** evidence of generalisation, and an unlabelled video invites exactly that misreading.

---

## 6. Deep-research learnings — implementation ledger

| finding | status |
|---|---|
| **E-CR** (compounding vs task difficulty) | ✅ **IMPLEMENTED + RUN** — `taniteval/compounding.py`. Verdict **H-COMPOUND**: CR 3.50 → 80.77, teacher-forced arm **flat**. Replicated. Resolved **C61**. |
| **Rollout-recovery** (the H-COMPOUND-indicated fix) | ✅ **RR-20 DONE** (31999); 🔵 **RR-CTL RUNNING**. This is the deep research's main actionable lever and it is executing. |
| **E-DPSI** (speed head keyed to heading?) | ✅ **RUN → NULL** below 12°. Shortcut hypothesis closed cheaply. |
| **E-ROLL** (recursive k=1 past 2 s) | ⬜ **NOT RUN** — unlocked by H-COMPOUND. Backlog **B2**, ~2–4 GPU-h. |
| **EPDMS `filter_m` contract** | ✅ **RECORDED** (`8d4a138`). `filter_m` appeared **nowhere** in pseudosim, so the gate would have shipped PDMS-v1 and over-penalised every arm. ⛔ still not applied — needs a human-reference rollout. |
| **sitclf** (camera-only situational classification) | ✅ complete, two model classes |
| **A4 — factorised path × velocity vocabulary** | ⚠️ **RE-VERIFICATION FAILED, NOT ADMISSIBLE** (below) |

### ⚠️ A4 must be re-run — the limit destroyed it

The targeted re-verification launched 2026-08-01 and **lost 71 of 102 agents** to the weekly limit
mid-flight. Outcome: *"1 claim refuted, 24 unverified (verifier agents failed). No claims
survived. Research inconclusive."*

⛔ **Nothing from that run is quotable**, including the primary source it surfaced. The one claim
that did get 3 valid votes was **refuted 0-2**. A4 remains the **highest-value open
re-verification** — it maps 1:1 onto our 5-way mixed lateral+longitudinal softmax and the
**88.7 % longitudinal** oracle gap.

---

## 7. Repository state

**Clean.** `git status --short` shows only `.claude/settings.local.json` (local editor config).
Branch, `origin/main` and `origin/agent/benchmarks-eval-20260721` are **all at `5fbc4ec`** —
nothing unpushed, nothing stranded.

---

## 8. Decisions owed by the PI

| # | decision | blocks |
|---|---|---|
| 1 | ⛔ **pod1 console stop/start** | all 8-GPU work; `/dev/nvidia*` empty + now `Connection refused`; **not fixable over SSH** |
| 2 | **X2 verdict run** (30 pod-days) | not authorised without you |
| 3 | **Wheelbase fix** | you chose C = measure first; decision still pending |
| 4 | **Old CPU pod release** | deletion needs you |

---

## 9. Next, in priority order

1. **v2corpus option A** the moment the relay lands — both arms on the 19, v1 re-scored, paired
   bootstrap, `leak_free_n=19` in the headline.
2. **Long overlay videos** on the canonical val, leak-status labelled per clip.
3. **RR-20 vs RR-CTL** when RR-CTL completes — the compounding verdict.
4. **E-ROLL** (B2) — now legitimately unlocked by H-COMPOUND.
5. **Re-run A4** — the limit killed it and it is the best open lever on the longitudinal gap.
6. **v2corpus option B** — the clean v2-line val; needed before any further v2-line training.

## Evidence class

| claim | class |
|---|---|
| every pod state, step, rc, md5, byte count in this report | **MEASURED (ours)** — read from the pods 2026-08-01 22:10–22:35 UTC |
| v2corpus `g_op_fwd_ade_m` 0.1366–0.4686 | **MEASURED but TRAINER-INTERNAL** — ⛔ not quotable as a result |
| E-CR CR 3.50 → 80.77 | **MEASURED (ours)**, replicated |
| the A4 re-verification | ⛔ **INADMISSIBLE** — 71/102 agents died; no claim survived |
| "same-DC pod→pod is refused" | **MEASURED** — `rc=255`; scope: `69.30.85.x` only, does **not** contradict C56's cross-DC result |
