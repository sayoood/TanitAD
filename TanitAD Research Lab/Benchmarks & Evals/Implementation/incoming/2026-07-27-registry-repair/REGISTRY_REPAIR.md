# The registry said the arm was never launched. It ran for 59 hours.

**2026-07-27 · dev box only (no pod loaded, no training touched) · HEAD `f45b100`**
**Evidence class + tier on every number. `overlapping_holdout_se` is called NOWHERE in this work.**
🔒 Gated-confidential discipline observed: counts only, never clip UUIDs.

---

## 0. HEADLINE

1. ⭐⭐ **`MODEL_REGISTRY.md` §1.5.5 was false, and I corrected it in place.** The row said
   `flagship-v4-fromscratch` was *"READY, **not launched** … **Zero GPU-day committed**"*. The run's
   own artifacts say **launched 2026-07-23T21:54:44Z, `final_step 29999`, `rc=0`, 59.04 h**. I
   re-read every field from the artifacts myself (pod2, 2026-07-27) — **not** from the proposed text.
   ⚠️ **This is the arm every selection experiment of the week measured on.** The superseded claim is
   kept visible with its date.
2. ⭐ **Root-cause class logged as `RETRACTION_LOG.md` **C41** — a registry row that recorded an
   *intent* and was never advanced to an *outcome*.** The standing consequence names a **mechanical
   check** that would have caught it (§3.3).
3. **Re-render 1 of 2 done and PROVEN identical.** `blockA_full_panel_20arm.json`'s `recovery` node:
   every statistic byte-identical, the display corrected from `0.0 [0.0, 0.0]` to
   **`8e-07 [1e-07, 1.8e-06]`** at 7 dp. **No conclusion moves.**
4. 🔴 **Re-render 2 of 2 is BLOCKED, and the block is itself a finding.**
   `e1a_horizon_heldout44.json`'s `p = 0.975` node **cannot be re-rendered from anything that
   exists** — its raw draws are persisted in no artifact, in the repo, or on the producing pod
   (3 probes). **No conclusion moves either**, but the contradictory line still stands in a
   committed file. **Escalated to its owner in §5.3 — not written into a README.**
5. **`0.4907` now carries its deployment tag** in the registry: **881 windows / 40 episodes**,
   in-sample, `a0` **0.4714** — against the 600-episode panels where the same `a0` reads **0.5015**.

⚠️ **REVIEWABLE DIFF — flagged as the brief requires:** I edited `Project Steering/MODEL_REGISTRY.md`
(**+41 / −4**) and appended to `Project Steering/RETRACTION_LOG.md` (**+38 / −0**). Full diffs are
committed beside this file as `raw/MODEL_REGISTRY.diff` and `raw/RETRACTION_LOG.diff`. §3.2 lists
every hunk. `Project Steering/Mission Plan.md` was **not** touched.

---

## 1. THE ARTIFACT EVIDENCE FOR §1.5.5 — quoted directly, re-read by me

⛔ **The class this job exists to avoid is C21 / C4: treating prose as verification.** The stream that
found the conflict proposed a replacement text; I used it as a *starting point* and then re-opened
every artifact myself. Verbatim capture: **`raw/pod2_v4fromscratch_artifacts_2026-07-27.txt`**; the
run's own config: **`raw/pod2_v4fromscratch_config.json`**; the HF read: **`raw/hf_flagship-v4-fromscratch_2026-07-27.json`**.

### 1.1 The run completed — `metrics.json` and `supervisor.log`

```
$ cat /workspace/experiments/flagship-v4-fromscratch/metrics.json
{
  "final_step": 29999,
  "canary_ade@2s": 1.1409059762954712,
  "canary_baseline": 15.674169540405273,
  "val": { "n": 881, "ade@2s": 0.5063393451569435, "oracle_ade@2s": 0.1891792784111995,
           "sel_gap@2s": 0.3171600678451486, "miss@2m": 0.21452894438138478 },
  "lam_mult_final": 1.0,
  "milestone_archives": ["ckpt_step10000.pt","ckpt_step15000.pt","ckpt_step20000.pt","ckpt_step5000.pt"],
  "wallclock_s": 212544.6
}

$ cat .../supervisor.log
[2026-07-23T21:54:44Z] supervisor UP on 08f6ce7d8e55; OUT=/workspace/experiments/flagship-v4-fromscratch; ...
[2026-07-23T21:54:44Z] launch attempt (restarts=0) in /workspace/TanitAD/stack
[2026-07-23T21:54:44Z] trainer pid=108011
[2026-07-26T09:01:37Z] trainer exited rc=0
[2026-07-26T09:01:37Z] clean finish (summary.json done)
[2026-07-26T09:01:37Z] supervisor exiting (run complete)
```

`MEASURED` **tier 1** (mine, direct read 2026-07-27). Against the row's *"NOT LAUNCHED … Zero GPU-day
committed"*: **`wallclock_s` 212 544.6 s = 59.04 h**, versus the row's **~53 h ESTIMATED** — the
estimate was **~11 % low** and the real spend was **≈2.5 GPU-days**.

### 1.2 ⚠️ One correction to the proposed text — `summary.json` does not exist

The proposed replacement quotes the supervisor's *"clean finish (**summary.json** done)"* as if a
`summary.json` artifact existed. **It does not:**

```
$ ls -la .../flagship-v4-fromscratch/summary.json
ls: cannot access 'summary.json': No such file or directory
```

That string is a **fixed literal** in `stack/scripts/supervise_run.sh:138`. Reading `is_done()`
(`:83-87`) shows two branches; the one that actually fired is the second — the **DONE token in the
trainer's own stdout**, which I read:

```
$ tail -1 /tmp/flagship-v4-fromscratch-train.out
{"done": true, "final_step": 29999, "canary_ade@2s": 1.1409059762954712, "lam_mult_final": 1.0,
 "milestone_archives": ["ckpt_step10000.pt","ckpt_step15000.pt","ckpt_step20000.pt","ckpt_step5000.pt"]}
```

`MEASURED` **tier 1**. The registry now states it this way so nobody hunts a `summary.json` that was
never written. **This is exactly what re-verifying instead of copying buys** — the proposed text was
right about the outcome and wrong about the artifact.

### 1.3 The `--from-scratch` lever was in force, and parity held — the run's own `config.json`

| field | value | why it matters |
|---|---|---|
| `from_scratch` | **`true`** | the lever, not an intention |
| `trunk.init` / `trunk.ckpt` / `trunk.step` | `"from-scratch (random)"` / **`null`** / **`-1`** | no v1 warm-start, proven at the run's own record |
| `parity.train_corpus_key` | **`physicalai-train-e438721ae894`** | ⭐ the canonical corpus — cross-arm comparability intact |
| `parity.skip_hash` | **`f09e44db`** | same |
| `args` | `steps 30000 · batch 16 · accum 4 (eff 64) · lr_head 1e-4 · lr_trunk 1e-4 · warmup 2000 · lam_mult_floor 0.25 · phase_a_steps 2000 · phase_b_steps 8000 · gate_step 10000 · labels v3 · strategic full · dense_plan true · rollout_k 4 · eval_episodes 40 · seed 0` | as launched, not as planned |

`MEASURED` **tier 1** (`raw/pod2_v4fromscratch_config.json`, pulled verbatim).

### 1.4 Host, checkpoints, train log

- **Host** — supervisor says *"UP on `08f6ce7d8e55`"*; pod2's live `hostname` is **`08f6ce7d8e55`**
  and its GPU is **NVIDIA A40, 46 068 MiB**. `MEASURED` **tier 1**. That closes the run onto pod2 by
  identity rather than by assumption.
- **Checkpoints** — `ckpt.pt` **3 243 109 310 B** + `ckpt_step{5,10,15,20}000.pt`, same size each.
- **Train log** — `train_log.jsonl` **661 rows**; first row is the step-0 canary baseline
  `15.674169540405273`; last row is **step 29999** (`plan_ade 0.3659`, `oracle_ade 0.1647`).

### 1.5 ⭐ An independent, repo-side corroboration that needs no pod at all

The **committed** eval JSONs `…/2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-{produced,oracle}.json`
both carry **`ckpt_step: 29999`** and `ckpt: /workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`;
`…/2026-07-26-bar-a-selector/raw/bar_a_produced.json` carries the same checkpoint with
**`md5 8771c1d9d3da696dcde2a745d628f6a8`**. `MEASURED` **tier 1**.
⇒ **A committed eval naming step 29 999 of an arm the registry called unlaunched is a contradiction
findable in the repo alone** — which is what makes §3.3's check cheap.

### 1.6 The decision-grade results the corrected row now records

| read | value | estimator | artifact |
|---|---|---|---|
| @15 000 `ade_0_2s` | **0.5839 [0.4962, 0.6821]** | `episode_cluster_bootstrap`, 881 w / 40 ep, B=2000 | `…/2026-07-25-flagship-v4-midtrain-eval/flagship-v4-fromscratch-15k.json` |
| @15 000 **paired vs v1** | **+0.1568 [+0.0630, +0.2504]**, `p_delta_gt0` 1.0, ✅ sep | **`paired_episode_cluster_bootstrap`** | `…_lateral_and_paired.json` (its own `_orientation`: positive ⇒ **behind** v1) |
| @29 999 **produced** (deployable) | **0.8563 [0.7282, 1.0035]**, `beats_cv` ❌ | `episode_cluster_bootstrap`, 881 w / 40 ep | `…/2026-07-26-v4-30k-gate/raw/…-30k-produced.json` |
| @29 999 **oracle-goal** | 0.6423 [0.5348, 0.7586], `beats_cv` ✅ but `deployable: false` | same | `…-30k-oracle.json` |

`MEASURED` **tier 1**, all four re-read by me from the raw JSON. The trainer's own `ade@2s 0.5063`
is recorded separately and marked **not quotable against v1** (class C1).

### 1.7 HF backing — verified, not inherited

`Sayood/flagship-v4-fromscratch`, `private: false`, **`gated: manual`**, `lastModified`
**2026-07-26T09:12:06Z** (11 minutes after the run exited), files: `ckpt.pt`, **`ckpt_step20000.pt`**,
`config.json`, `metrics.json`, `train_log.jsonl`, `README.md`. `MEASURED` **tier 1** (HF API, this
agent, `raw/hf_flagship-v4-fromscratch_2026-07-27.json`).
🔴 **`ckpt_step{5,10,15}000.pt` are NOT on HF.** The **15 k** milestone — the one carrying this arm's
decision-grade read (§1.6) — is **single-disk**. Recorded in the registry's `Location` row.

---

## 2. THE DIFF I MADE — every hunk

### 2.1 `Project Steering/MODEL_REGISTRY.md` (+41 / −4)

| # | where | what changed |
|---|---|---|
| **1** | **§1.5.5 heading** | `flagship-v4 from-scratch **fallback** — ✅ READY, not launched` → `flagship-v4 from-scratch — ✅ **COMPLETE (30 k, rc=0)**` |
| **2** | **§1.5.5, new correction banner** | 🔴 **ROW CORRECTED 2026-07-27**, quoting the **superseded Status and Cost text verbatim** with its date, naming the blast radius, and pointing at `RETRACTION_LOG` **C41** |
| **3** | **§1.5.5 `Status`** | replaced with the launch/exit evidence of §1.1 + the `summary.json` precision of §1.2 + the parity key |
| **4** | **§1.5.5 `Cost`** | `~53 h ESTIMATED` → **`MEASURED 59.04 h`** on pod2/A40, with the estimate's error stated |
| **5** | **§1.5.5 `Result`** | **NEW row** — trainer-val and decision-grade kept apart, trainer-val explicitly non-quotable |
| **6** | **§1.5.5 `Code state`** | now cites the run's own `config.json` as proof the lever was in force, plus args as launched |
| **7** | **§1.5.5 `Location`** | **NEW row** — pod path, checkpoint sizes, HF state, and the single-disk 15 k flag |
| **8** | **§1.5 v4-family GATE-COMPLETENESS block** | one parenthetical: its *"not HF-backed"* sentence now carries the measured exception (§1.7) |
| **9** | **§1.5 through-line HYPOTHESIS block** | one ⚠️ line: the block was written while this arm was a *fallback*; it has since run, canary **1.1409** from baseline **15.6742**. **Explicitly records evidence and does NOT re-adjudicate the hypothesis** |
| **10** | **§1.2, after the `0.4271` consequence note** | the **`0.4907` deployment tag** (§4) |

**Kept verbatim, as instructed:** §1.5.5's `Distinguishing lever` and `Validation` rows are unchanged.

### 2.2 `Project Steering/RETRACTION_LOG.md` (+38 / −0)

Append-only, new class **C41** (previous maximum was C40). §3 below.

---

## 3. THE ROOT-CAUSE CLASS — `C41`

### 3.1 What the class is

**A registry row that recorded an INTENT and was never advanced to an OUTCOME.** The same row served
as the pre-registration *and* as the status, so the launch had **nowhere to be written**.

### 3.2 ⚠️ Why nobody caught it — the part that makes it a class and not a slip

**Every fast surface was right and the authoritative one was wrong.** `LOOP_STATE.md`, four program
reports and two `RETRACTION_LOG` entries all record the run correctly (07-24 *"step 4350"*, 07-24
*"step 9550"*, 07-26 *"step 29,300 / 30,000 … PID 108011 alive, restarts 0"*). That is the **inverse
of C1** — and it is precisely why the error survived four days: the standing rule tells a reader to
**trust the registry over the prose**, so a reader who followed the rules correctly inherited a false
fact. *(Those corroborations are listed as corroboration only; §1's finding rests on the artifacts.)*

### 3.3 ⭐ The standing consequence — the check that would have caught it

1. **A pre-registration row and a status row must not be the same row.** A planned arm carries
   `Status: PLANNED` **plus a separate empty `Outcome:` field**. An empty field is visibly
   unanswered; a stale sentence is not.
2. **The mechanical check, and it needs no judgement:** every `Location:` in the registry is a real
   directory. **Any row whose status says NOT LAUNCHED while its run directory holds a `metrics.json`
   with a `final_step` is a contradiction a script can find** — the same nightly shape as
   `stack/scripts/pod_git_drift.py`, pointed at experiment dirs instead of code. **And the repo-only
   form is cheaper still:** a committed eval JSON naming `ckpt_step 29999` for an arm the registry
   calls unlaunched (§1.5) is the same contradiction, findable without touching a pod.
3. **`AGENT_OPERATING_STANDARD.md` already says "refresh whenever a model version is created,
   retired, or re-measured" — the gap is that LAUNCH and COMPLETION are none of those three.** Add
   both: refresh on **launch**, on **completion**, and on **first decision-grade eval**.

---

## 4. THE `0.4907` DEPLOYMENT TAG

⛔ **No result is restated here; only the scope is made explicit**, so the comparison cannot be made
silently.

| fact | value | source (`MEASURED` **tier 1**, re-read by me) |
|---|---|---|
| the number | `in_sample_ceiling.ce.ade_0_2s_in_sample` = **0.4907** | `…/2026-07-26-bar-a-selector/raw/bar_a_produced.json` |
| windows / episodes | **881 / 40** | same file, `_cache.n_eval_windows`, `_cache.n_episodes` |
| what it is | **IN-SAMPLE** — the artifact's own `_read`: *"fit and scored on the same windows … NOT deployable, NOT a generalization number"* | same file |
| what it is measured on | the **frozen produced-surface fan** of `flagship-v4-fromscratch` @ `_ckpt_step` **29 999**, `_goal_mode "produced"` | same file |
| the deployment's `a0` | **0.4714** = REF-C-XL's full-set `ade_0_2s` on that same 881/40 surface | `MODEL_REGISTRY.md` §1.5 v1.6 table, row *ADE@2s full-set* |
| ⛔ the deployment it is being compared against | **13 198 windows / 600 episode clusters**, where the same `a0` reads **0.5015** | `…/2026-07-28-egoal-4-joint/raw/e4_summary.json`, `deployment.a0_as_trained` |

⇒ The registry now states that a 0.4907-vs-600-episode comparison is **cross-deployment and not a
result**, by the same rule that forbids substituting v1's 600-ep 0.4108 for its 0.4271 (§1.2a).

⚠️ **Note the coupling to §1:** `0.4907` is measured on **`flagship-v4-fromscratch` @ 29 999** — the
arm §1.5.5 said had never run. The bar and the substrate were in the same registry, disagreeing.

### 4.1 v1's `0.4271` — checked and left alone, as instructed

`MODEL_REGISTRY.md` §1.2's annotation is **present and legible**: the metric row is labelled
*"= `wm_fidelity_ade_2s` — a WORLD-MODEL FIDELITY number, NOT a planning number"*, and the paragraph
below it cites `taniteval/rollout.py:170` (`actions_source="expert_future"`) and `:174`, then states
the consequence in ⛔ form. **Nothing changed there.** My only edit near it is the additive `0.4907`
tag placed *after* that paragraph.

---

## 5. THE TWO RE-RENDERS

⛔ **Neither source artifact was modified.** Both corrected renderings live as raw JSON in this
directory; folding them back is the owning streams' call (§5.3).

### 5.1 ✅ `blockA_full_panel_20arm.json` — RE-RENDERED, and PROVEN identical

Node `paired / refc_base_produced__minus__refc_base_v0on / recovery`.
Replayed by `code/rerender_blockA_recovery.py` → `raw/rerender_blockA_recovery.json`.
**No model, no GPU, no pod** — the committed per-window `pw_*.npz` dumps, the panel's own
`score_windows`, `paired_episode_cluster_bootstrap` at **B=2000, seed 0**, and the panel's own
row-identity gate re-asserted (`identical_to_reference: true` for all three arms; `eid` arrays equal).

| field | committed (old renderer) | re-rendered (fixed renderer) | |
|---|---|---|---|
| `delta` | `0.0` | **`8e-07`** | identical at 4 dp ✅ |
| `lo` | `0.0` | **`1e-07`** | identical at 4 dp ✅ |
| `hi` | `0.0` | **`1.8e-06`** | identical at 4 dp ✅ |
| `ci95` | `0.0` | `9e-07` | identical at 4 dp ✅ |
| `p_delta_gt0` | `0.9885` | `0.9885` | **exact** ✅ |
| `separated` | `true` | `true` | **exact** ✅ |
| `n_windows` / `n_episodes` / `n_rows_paired` | 13184 / 40 / 13184 | 13184 / 40 / 13184 | **exact** ✅ |
| `reducer` / `n_boot` / `estimator` | mean / 2000 / `paired_episode_cluster_bootstrap` | same | **exact** ✅ |
| — | — | `display_dp: 7` + `display_note` | the fix's own audit trail |

⭐ **`RE_RENDER_VERIFIED_IDENTICAL: true`, `_mismatches: {}`.** That is the check that this was a
**re-render and not a recompute**: every field the rendering fix does not touch is equal, and the
four it does touch are equal at 4 dp. The script **exits non-zero** if that ever fails.

**Does any conclusion move? NO.**
- The verdict `separated: true` is **unchanged** — the separation test runs on unrounded bounds and
  was never touched.
- What changes is that the magnitude is now **visible**: the delta is **≈8 × 10⁻⁷**, and the interval
  no longer prints as a null. It is **not** `degenerate` (bounds exceed 1e-12), so no marker is
  emitted — the numbers alone now carry the reading.
- `TACTICAL_ACTION_INPUT.md` **does not quote this contrast**; the two arms are the same arm on this
  surface (identical panel PSS `0.5439 [0.5345, 0.5519]`, `MEASURED`, `blockA_full_combine.log`). The
  panel's gate and every headline run on `PSS_recovery_progress` / `ego_progress`, not on this node.

### 5.2 🔴 `e1a_horizon_heldout44.json` — RE-RENDER BLOCKED. This is a finding.

Node `paired_common_start / deltas_vs_K20 / longitudinal / 160 / d_closed_ade2s_m`
(`delta 0.0 [0.0, 0.0] separated=true`, `p_delta_gt0` **0.975**, 80 w / 21 ep).
Attempt + evidence: `code/rerender_e1a_horizon.py` → `raw/rerender_e1a_horizon_BLOCKED.json`.

**The raw draws do not exist.** `e1a_horizon.py` builds the two per-window `ade2s` vectors inside
`main()` from closed-loop GPU rollouts and writes **only** the summary JSON — there is no
`torch.save`/`np.savez` in the file. **Three probes** (CLAUDE.md rule 2 — absence at one location is
not absence):

| probe | result |
|---|---|
| the committed artifact | summary nodes only; `all_windows[K]` holds bootstrapped blocks, not per-window arrays |
| repo-wide search for this run's dumps | none. `…/2026-07-26-horizon-envelope-closeout/artifacts/perwindow_K{20,60,70,185}.pt` is a **different design** (K set 20/60/70/185, 41 common windows / 40 eps vs this run's 20/40/80/120/160, 175 common windows) and **cannot substitute** |
| `tanitad-pod3:/workspace/e1a_e2a/` (the producing host) | JSONs + scripts survive; **no per-window tensor** |

⇒ Recovery means re-running the closed-loop rollouts — a **RECOMPUTE**, explicitly out of scope, and
a GPU job. **NOT DONE, and not faked.**

**What IS derivable from the committed artifact alone** — `MEASURED` **tier 1** (arithmetic on the
estimator's own definition; both branches run in the script):
`p_delta_gt0 = (d > 0).mean()` over B=2000 ⇒ **exactly 1950 draws > 0 and 50 ≤ 0**.
`lo = np.percentile(d, 2.5)`, whose 0-based linear-interpolation index is `0.025 × 1999 =` **49.975**
⇒ `lo` is fixed by **d[49] (≤0) and d[50] (>0)**, weighted **0.975** toward d[50].
⭐ **So `separated` here rests on a single order statistic sitting exactly on the 2.5 % boundary, and
holds only while `d[49] > −39 · d[50]`.** The script demonstrates both outcomes at the *same*
`p_delta_gt0` the artifact printed: `lo = +9.75e-07` (separated) vs `lo = −1.52e-06` (not separated).
From the printed rounding alone, `|delta| < 5e-5` and `|lo| < 5e-5`.

**Does any conclusion move? NO** — and this one is worth stating precisely, because it is the node
the brief flagged as *"the marginal one that matters"*. `E1a_E2a_RESULTS.md`'s verdict rests on
**corridor departure at an 18.5 s horizon** and says in its own words that ADE@2s *"CANNOT see 18 s
drift"*; it quotes `closed_ade2s` **only** as the all-window `0.485 → 0.496` move, never as this
paired node. ⚠️ **But the contradictory line is still standing in a committed file**, and I could not
fix it — see §5.3.

### 5.3 🔴 INTEGRATION NEEDED — stated here, in the headline, not in a README

**(a) OWNER of `…/2026-07-25-closedloop-horizon-and-shift`** — your `d_closed_ade2s_m` @K=160 node
**cannot be re-rendered by anyone** from what exists. Two admissible fixes, both cheap:
**(i)** re-run `e1a_horizon.py` with a one-line per-window dump added (`np.savez` of the `ade2s`
vectors and `eid` at each K) — which also makes every future interval in that artifact
re-renderable; or **(ii)** stamp the node unquotable exactly as `legA_v5config_structural.json` was
stamped. ⛔ **Do not leave `0.0 [0.0, 0.0] separated` standing.**

**(b) OWNER of `…/2026-07-28-tactical-action-input`** — your `recovery` node's corrected rendering is
in `raw/rerender_blockA_recovery.json`, verified identical. **Folding it into
`blockA_full_panel_20arm.json` is a display swap with zero statistical content.** I did not edit
your artifact.

**(c) `taniteval` maintainer — the generalisable lesson.** Both defects share one cause: **an
estimator that persists only its summary cannot have its own rendering repaired.** Consider making
the per-window vectors a required side-artifact of any committed interval, the way
`…/2026-07-27-pseudosim-arm-panel` already does with `pw_*.npz` — which is exactly why §5.1 was
possible and §5.2 was not.

---

## 6. SUITES

`MEASURED` **tier 1**, venv `C:/Users/Admin/venvs/tanitad`, transcripts committed beside this file.

| suite | result | expected by the brief | new skips |
|---|---|---|---|
| `stack/` `pytest -q` | **1523 passed, 12 skipped**, 2 warnings, 127.11 s | 1523 / 12 | **0** ✅ |
| `taniteval/` `pytest -q` | **644 passed**, 1 warning, 104.50 s | 644 | **0** ✅ |

**No code was changed by this work** — the deliverables are two Markdown edits, three raw evidence
files, two standalone analysis scripts in this directory, and their JSON outputs.

---

## 7. WHAT I DID NOT TOUCH

- ⛔ **pod1** — not contacted at all; it is training.
- ⛔ **pod2** — **read-only text reads only** (`ls`, `cat` of ≤8 KB files, `hostname`,
  `nvidia-smi --query-gpu`). **No GPU or RAM load added**, no process started, no file written. A
  sibling holds it for the PI's validation.
- **pod3** — one read-only `ls` of `/workspace/e1a_e2a/` (§5.2 probe 3).
- ⛔ **No training launched.** ⛔ **`Project Steering/Mission Plan.md` untouched.**
- **The two source artifacts of §5 are unmodified**, and `legA_v5config_structural.json` was left
  alone as instructed (already stamped unquotable).
- `Keys.txt` read in place for the HF token; never printed, copied, or passed as an argument.

---

## 8. DELIVERABLE MANIFEST

**STAGED, NOT COMMITTED, NOT PUSHED.** Nothing here lives in only one place.

| # | artifact | where | note |
|---|---|---|---|
| 1 | `REGISTRY_REPAIR.md` (this file) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-registry-repair/` | |
| 2 | `raw/pod2_v4fromscratch_artifacts_2026-07-27.txt` | same | ⭐ verbatim pod2 capture — the §1 evidence |
| 3 | `raw/pod2_v4fromscratch_config.json` | same | the run's own `config.json`, byte-for-byte (8 233 B) |
| 4 | `raw/hf_flagship-v4-fromscratch_2026-07-27.json` | same | HF API read (§1.7) |
| 5 | `raw/MODEL_REGISTRY.diff` | same | the reviewable registry diff (+41 / −4) |
| 6 | `raw/RETRACTION_LOG.diff` | same | the C41 append (+38) |
| 7 | `code/rerender_blockA_recovery.py` | same | re-render + its own pass/fail check |
| 8 | `raw/rerender_blockA_recovery.json` | same | ⭐ `RE_RENDER_VERIFIED_IDENTICAL: true` |
| 9 | `code/rerender_e1a_horizon.py` | same | blocked-re-render evidence + the order-statistic arithmetic |
| 10 | `raw/rerender_e1a_horizon_BLOCKED.json` | same | 🔴 the escalation payload for §5.3(a) |
| 11 | `raw/pytest_stack.txt`, `raw/pytest_taniteval.txt` | same | suite transcripts |
| 12 | **`Project Steering/MODEL_REGISTRY.md`** | `repo:` (modified) | ⚠️ **the reviewable edit** — §2.1 |
| 13 | **`Project Steering/RETRACTION_LOG.md`** | `repo:` (modified) | class **C41** — §3 |

**Nothing exists on a pod only.** The pod2 artifacts that carry the §1 evidence are now transcribed
into items 2–3; the checkpoints themselves remain pod-side (item: `ckpt_step{5,10,15}000.pt`
single-disk, recorded in the registry `Location` row and flagged in §1.7 — a **rescue-or-drop
decision for the PI**, out of scope here).
