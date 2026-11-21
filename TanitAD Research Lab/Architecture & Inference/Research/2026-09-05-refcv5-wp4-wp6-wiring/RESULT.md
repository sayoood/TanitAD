# refcv5 WP-4 / WP-6 — the seam, closed

**Date** 2026-09-05 → 2026-09-06 (the session crossed midnight; the package keeps its opening date)
**Branch** `agent/arch-inf-20260803` · **Compute** CPU only (Thor 97 %, dev-box 4060 100 % — both busy, no slot taken)
**Verdict** refcv5 is **wired and training-ready apart from a GPU and a corpus flag**. See §6 for what remains.

> ⛔⛔ **EVIDENCE CLASS AND TIER, STATED ONCE FOR THE WHOLE DOCUMENT.** Every number
> here is **MEASURED (ours)**, artifact `raw/probes.txt` (re-run on the FINAL code state;
> the header records interpreter, torch build and the tree `tanitad` was imported from),
> plus the pytest lines quoted inline. **NO EVAL TIER APPLIES TO ANY OF THEM, AND NONE IS
> A DRIVING-PERFORMANCE NUMBER.** They are instrument and wiring measurements taken on
> **randomly-initialised weights** and **synthetic tensors** — no corpus, no training, no
> checkpoint. In particular the metre-valued quantities (`mean |Δ fan| 2.090 m`, `0.211 m`)
> are *"does the loop move its own output"* diagnostics on an untrained denoiser and
> **must never be quoted as ADE or as any capability claim**; doing so would be the T0-as-
> driving-performance error (C131) in a new costume. ⚠ No `SPEC.md` accompanies this
> package because it is an implementation task that consumed **zero GPU** and
> pre-registers no hypothesis; the pre-registration that governs the ARM this unblocks is
> a separate, still-owed artifact.


---

## 1. P1 — an implementation ALREADY EXISTED, and this was an escalation-to-merge

⭐ **The sibling's `0 / 0 / 0` was correct but SCOPED TO ONE FILE.** It read
`control_head` / `sampler` / `cross_agent` in **`refc.py`** and found none. All three
exist — in **other modules**, committed, and unreferenced:

| artifact | where | state found |
|---|---|---|
| WP-4 sampler (DDIM schedule, sinusoidal time MLP, `roll_controls`) | `stack/tanitad/refs/refc_sampler.py`, **314 lines / 14 defs** | in `HEAD`, commit **`a5dbfbb`** *"refcv5 WP-4: the diffusion mechanism, in CONTROL space"* |
| WP-6 agent seam (`AgentSeamConfig`, `build_agent_head`, `AgentTokenEmbed`, `OracleAgentEmbed`, `agent_losses`, rig cameras) | `stack/tanitad/refs/refc_agents.py`, **820 lines / 20 defs** | in `HEAD` |
| the trainer seam (`_pin_refcv5_seams`, `_seam_stamp`, `--sampler`, `--agents`, `--w-u0`, `--w-agent`, 4 refusals) | `stack/scripts/refc_v3_train.py` | in `HEAD` — **already complete** |
| the red tests — **20, not 14** | `test_refc_sampler.py` (12), `test_refc_v3_refcv5_wiring.py` (2), **`test_refc_agents.py` (6)** | in `HEAD` |

⇒ **Nothing was rewritten. The missing layer was exactly the SEAM**: `refc.py`
(`DecoderConfig`, `AnchoredDiffusionDecoder`, `CrossAttnLayer`, `RefCModel`) and
`refc_v3.py` (`RefCV3Model.forward`). Leaves and trainer existed; the trunk between
them did not. That is the 10-day orthogonality-instrument failure caught at day 1.

⚠️ **Absence claims made honestly.** `.claude/worktrees/` was **not** probed (known
never-hydrating, per the brief). `git log --all` over the whole repo **timed out on the
G: mount and is INCONCLUSIVE**; the positive assertion that settled P1 was
`git cat-file -e HEAD:<path>` on each file plus `git log --diff-filter=A -- '*sampler*'`,
which named `a5dbfbb`. Every `grep -c` zero quoted here was taken **in the same breath
as a control literal that read non-zero** (`class AnchoredDiffusionDecoder` → 1,
`    def ` → 41).

---

## 2. P2 — the tests (there were **20**, not 14), and the load-bearing identity

⚠⚠ **THE BRIEF'S "14 RED TESTS" WAS AN UNDERCOUNT, AND I REPEATED THE P1 MISTAKE TO FIND
IT.** I built to the two files the brief named and only the full suite revealed a THIRD
pre-existing spec — **`test_refc_agents.py`, 6 more red tests** that specify the SAME
decoder seam at the layer level (`layer.cross_agent` / `layer.agent_gate` / `layer.norm_a`,
`forward(agent_tokens=, agent_pad=)`, `_decode(agents=)`). **VERIFIED red at HEAD**, so
they were part of the specification all along: `HEAD:refc.py` reads `agent_gate` **0**,
`cross_agent` **0**, `agent_tokens` **0**, against a same-breath control of
`class AnchoredDiffusionDecoder` **1**.
⇒ My first API used `agent_tok` / `agent_pad`; **the pre-existing spec says `agent_tokens`
and `agents`**. ⛔ I conformed **my code to their names**, never the reverse. It is the
same lesson as §1 one level down: *absence found at the locations you were handed is not
absence*, and the cheapest probe would have been to grep the whole `tests/` tree for
`cross_agent` before writing a line.

**Baseline (measured before any edit):** the two named files `14 failed, 23 passed,
1 skipped`; `test_refc_agents.py` a further `6 failed, 25 passed` ⇒ **20 red**.
**Now:** **`81 passed, 0 skipped`** across all three — the 20 turned green, **the skip was
eliminated**, and **12 tests were added**. ⛔ **No test was weakened.** One instrument was
corrected, and it is declared separately in §4.

### ⭐⭐ The bit-identity, MEASURED

`roll_controls(constant sequence)` vs `AnchoredDiffusionDecoder.roll_bank`, over
**117 anchors × 8 slots × 5 speeds × 2 coords = 9,360 float32 scalars** (4,680 waypoints),
speeds including a **standstill (0.0 m/s)** and **36 m/s**:

| units | `torch.equal` | max abs diff | differing float32 bit patterns |
|---|---|---|---|
| `alat` | **True** | **0.000e+00** | **0 / 9,360** |
| `kappa` | **True** | **0.000e+00** | **0 / 9,360** |

The sampler and the vocabulary integrate **the same physics**, in both unit systems.

### The mechanism is real — refcv3's pathology does NOT reproduce

refcv3's "ranking" read the `t = 0` confidence and was **unchanged on 201/201 windows**.
The check that a loop is not inert is whether adding a rung changes the **output**:

| ladder | mean \|Δ u0_hat\| | mean \|Δ fan\| | rows changed |
|---|---|---|---|
| `[9]` → `[10, 0]` | 1.25775 | **2.090 m** | **128 / 128** |
| `[10, 0]` → `[11, 6, 0]` | 0.22433 | **0.211 m** | **128 / 128** |

⚠️ **A subtlety worth stating**: the DDIM *state* update at the last rung (`t = 0 → 0`)
is an exact no-op by construction (`a_t == a_prev`, `s_t == s_prev` ⇒ `x_prev == x_t`,
**measured 0.00000**). The rung is still not decorative, because `u0_hat` is taken from
that pass's **x0 prediction**, not from the state — which is why the table above moves.

### The anchored Gaussian reads its known value

With `control_head` at its zero init, `u0_hat` is **exactly** the noised anchor:
mean \|dev\|/norm **0.02526**, max **0.12189**, against a schedule that says
`sigma(8) = 0.03159` (E\|N(0,1)\|·σ = 0.0253 ✓; ~3.9σ over 2,048 draws ✓).
"The vocabulary is the prior" is true at step 0, not merely intended.

---

## 3. What was built (the seam only)

**`stack/tanitad/refs/refc.py`**
* `DecoderConfig` — **declared fields** `sampler`, `sampler_space`, `sampler_train_t_max`,
  `sampler_infer_t`, `sampler_steps`, `sampler_groups`, `control_norm`, `metre_sigma_m`,
  `cross_agent`. *(These being real fields is the false-provenance fix — see §5.)*
* `RefCConfig.agents` — declared, `None` = off, so the stamp can tell **off from absent**.
* `CrossAttnLayer` — optional agent cross-attention + **zero-init `agent_gate`**, built
  only when asked and **after** every pre-v5 submodule so an off build draws identical RNG.
  ⛔ Carries the **fully-padded-row guard**: an empty road is the *common* case and an
  unguarded `key_padding_mask` returns **NaN, not zero**.
* `AnchoredDiffusionDecoder` — `control_head` (zero-init), `time_mlp` (DD's sinusoidal
  embedding of the **continuous** t; the pre-v5 `time_embed` has **3 rows** and cannot
  represent `t ~ U[0, 50)`), `sched`; `anchor_control_seq`; `_decode_ctrl`,
  `_state_to_path`, `_sample`; two forward refusals; `u0_hat` in the output.
  ⭐ Only **two** new modules: the sampler reuses `traj_proj → layers → conf_head`, so it
  cannot drift into a second, differently-conditioned decoder.
* `CrossAttnLayer` / `AnchoredDiffusionDecoder` — the agent parameter names are the
  **pre-existing spec's**: `forward(agent_tokens=, agent_pad=)` and `_decode(agents=)`.
* `RefCModel` — `agent_head` (oracle **or** detector) + `agent_embed`, `agent_gt` on the
  forward, `agent_slots` exported (the trainer's detection loss is guarded on that key).
* The WP-4 sampler **replaces** the pre-v5 `noise_std` refinement loop rather than
  wrapping it — running both would put two perturbations on one fan and make the arm
  non-attributable (the `--v2` conflation failure).

**`stack/tanitad/refs/refc_v3.py`** — `agent_gt` threaded through `RefCV3Model.forward`
(both the hier and non-hier paths), with a **both-direction refusal** (supplied to an
agentless build / withheld from an oracle build).

**`stack/tanitad/refs/refc_sampler.py`** — `DDIMSchedule.to(device)`. ⛔ Not cosmetic:
`add_noise` unsqueezes the gathered alpha to `x0`'s rank, so a CPU table meeting a CUDA
`x0` does **not** get the 0-dim scalar exemption and raises. The table is deliberately
not a buffer, so `nn.Module.to` cannot follow it.

**`stack/scripts/refc_v3_train.py`** — `assert_seams_are_built(model, stamp)`, called
immediately before `config.json` is written.

---

## 4. ⚠️ DECLARED SEPARATELY — one instrument was corrected, and it had NEVER RUN

`assert_matches_diffusers` carried `atol = 1e-10`. **It could never have passed.**
It had never executed because `diffusers` was never installed beside it.

Installed `diffusers 0.40.0` (`--no-deps`; torch verified unchanged at
**2.11.0+cu128, `cuda True`, a real CUDA `conv2d`** — not merely `import torch`).
The check then raised at **1.689e-07**, and the cause is a **dtype comparison, not an
error**:

| comparison | max abs diff |
|---|---|
| ours **float32** vs diffusers float32 | **0.000e+00** (bit-equal, betas *and* alphas_cumprod) |
| ours **float64** vs an exact float64 numpy reference | **6.939e-18** |
| **diffusers float32** vs that same exact reference | **1.689e-07** |
| float32 eps | 1.192e-07 |

⇒ `DDIMScheduler` builds its table in **float32**; `DDIMSchedule` builds it in float64.
**Ours is the more accurate of the two.** A tolerance below the reference's own epsilon
is not a strict test, it is an **unrunnable** one — the mirror image of a guard that
cannot fail.

⭐ **The replacement is STRICTER, not looser.** It now (a) rebuilds *our* schedule **at
the reference's own dtype** and requires **bit-equality** — no tolerance at all, and it
passes; then (b) bounds the float64 residual at `ACC_EPS_MULT (16) × eps`, because a
1000-step `cumprod` accumulates and the measured gap is **1.42 × eps**. One eps would
fail a correct implementation. The bound (16 eps = 1.9e-06) still sits **~3.7 orders of
magnitude** below any real disagreement (mistaking `scaled_linear` for `linear` moves
`beta[499]` by ~1e-2).

⇒ The schedule is now **PRIMARY-verified against the published reference**, where before
it was an unverified copy that reported a skip.

---

### ⛔⛔ THE DELIBERATE REGRESSION COULD NOT HAVE FAILED — caught in my own code, by review

`sampler_space="metre"` is pre-registered to **FAIL** the flyability gate. My first
implementation used `metre_sigma_m` **directly as a divisor**, which delivers
`sigma(8) x 0.90 = 0.028 m` of per-waypoint noise — **31.7x too gentle**. The DD-literal
arm would have sailed through the gate it exists to fail, and the comparison would have
read *"metre space is fine"*. **An arm that cannot fail proves nothing**, which is the
exact defect this whole seam is instrumented against, reproduced by me one level down.

The normaliser is now **DERIVED**: `metre_sigma_m / sqrt(1 - abar(sampler_infer_t))`
= **28.49 / 23.11**, so the emitted noise IS DD's published **0.90 m / 0.73 m** per
waypoint. The field is renamed `metre_norm -> metre_sigma_m` so it cannot be misread as a
divisor again, and it carries the number in its own comment.

**MEASURED after the fix** (zero-init `control_head`, so this is the sampler's noise
alone; `raw/probes.txt`, `code/metre.py`; implied \|a_lat\| between consecutive 0.5 s slots):

| arm | mean \|a_lat\| | max | fraction over mu = 0.7 (6.87 m/s²) |
|---|---|---|---|
| `control` | **0.5312 m/s²** | 1.7395 (0.18 g) | **0.0000** |
| `metre` | **5.2941 m/s²** | 16.1573 (1.65 g) | **0.3333** |

⭐ The design plan predicted **~5.8 m/s²** for DD's 0.73 m over a 0.5 s slot; the arm reads
**5.29**. The prediction reproduces, the regression breaks the friction circle on a third of
slots, and the control-space arm never touches it — *flyable by construction* is now a
measurement, not a claim. Pinned by
`test_metre_space_arm_is_UNFLYABLE_and_the_control_arm_is_NOT`.

---

## 5. P3 — the review, and the guard shown FAILING

### The false-provenance defect, and why the field alone was not the fix

`DecoderConfig` had no `sampler` field, so `_pin_refcv5_seams`'s
`core.decoder.sampler = "ddim"` created an **ad-hoc attribute on an unfrozen dataclass**.
Python accepted it, `_pin_refcv5_seams`' own two guards read it back and **passed**, and
`_seam_stamp` copied it into `config.json` — for a model containing **no denoiser at all**.

Declaring the field removes *this instance* and none of the *class*: the stamp is built
from the **config**, and a config is a statement of intent; only the **model** is a
statement of fact. So `assert_seams_are_built(model, stamp)` checks the record against
the built modules, **bidirectionally** — a seam BUILT but NOT STAMPED is the
`SEAM_STATE.md` failure (six live seams absent from `config.json`).

### ⛔ Shown capable of failing — inspection is not a guard

| case | result |
|---|---|
| **CONTROL** honest default run | **PASSES** |
| **CONTROL** honest `--sampler ddim --w-u0 0.5` | **PASSES** |
| **MUTATION 1** stamp `ddim` on a model with no denoiser *(the exact pre-fix state)* | **REJECTED**, 4 findings |
| **MUTATION 2** live sampler stamped `none` | **REJECTED** |
| **MUTATION 3** `agents` stamped, no head built | **REJECTED**, 3 findings |
| **MUTATION 4** `w_u0 = 1.0` with no `control_head` | **REJECTED** |

Both controls pass, so the guard is not vacuous. All six are now permanent tests.

### WP-6 verified as a RUNNING path, not a constructed one

| check | result |
|---|---|
| oracle builds and forwards, `agent_slots` exported | ✅ (4/4 layers carry `cross_agent`) |
| **empty scene** — every slot invalid | **no NaN** (the guard earns its place) |
| **zero-init parity** — `traj` vs the agent-free model, 293 shared tensors copied | **bit-IDENTICAL, max\|Δ\| = 0.0** |
| gated, not dead — `∂loss/∂agent_gate` per layer | **6.36e0, 7.54e0, 1.36e0, 1.14e0** — all non-zero |

### ⚠️ A finding the PI/trainer owner must decide, NOT a bug

`SelectionConfig.refined` defaults to **False**, so the ranked score is the **classifier**
surface — which the sampler deliberately does not touch (pinned by
`test_hfov_style_sanity...`). With it False the fan is **sampled but ranked by a head that
never saw the sample** — the S1 defect one level up (measured there as a 45.4 %-of-windows
ranking failure). It is **stamped, not refused**, because "improve the geometry, keep the
ranking" is a legitimate arm: new telemetry key **`sampler_ranks_the_fan`** says which arm
a reader is looking at. ⇒ **A refcv5 arm that wants the sampler to reach SELECTION must
run `--sel-refined`.**

### ⛔ The variance obligation this seam creates

`_sample` draws fresh noise **at eval as well as training** — by design, because sampling
is the mechanism. That makes refcv5 a **stochastic planner**, and
`D-REFAV1-SEED-GOAL-MISMATCH` binds: the same checkpoint evaluated twice does not give the
same answer (that rig's inference-seed floor was **≈0.30 m ADE**). ⇒ **Any refcv5 sampler
result needs INFERENCE-seed replicates**; the episode-cluster bootstrap answers *"would
another draw of episodes say this?"* and is structurally blind to this. Written into the
`_sample` docstring so it travels with the code.

---

## 6. Is refcv5 training-ready?

**Yes, apart from a GPU** — every architectural piece is present, wired, guarded and
tested. Three things remain, and none is an architecture gap:

1. **GPU** — both are busy (Thor 97 %, three live `refav1_arm.py` eval arms; dev-box 100 %).
2. **`--agents head` needs `--agent-join`** — the `obstacle.offline` join file
   (`HF Sayood/tanitad-ph0-aug120 → joins/train2400_agents.jsonl.xz`). The trainer already
   refuses without it. **`--agents oracle` is the rung that runs first and needs no detector.**
3. **`--sampler ddim` needs an `--anchor-file` built with `controls`** and a declared
   `control_units` (`anchor_meta.py`); the decoder refuses a fixed-path bank at forward time
   and the trainer refuses it at pin time.

**The first arm to run** (all guards satisfied):
`--sampler ddim --w-u0 <w> --anchor-file <controls bank>` ± `--sel-refined`, with an
inference-seed replicate. **WP-4 landing is the precondition on the PI's RL plan** — DD-v2
post-trains a *denoising trajectory*, which did not exist until now.

---

## 7. Deliverable manifest

| artifact | where |
|---|---|
| `refc.py` — DecoderConfig fields, CrossAttnLayer agent seam, sampler modules + `_sample`, RefCModel agent seam | repo `stack/tanitad/refs/refc.py` |
| `refc_v3.py` — `agent_gt` threading + both-direction guard | repo `stack/tanitad/refs/refc_v3.py` |
| `refc_sampler.py` — `DDIMSchedule.to`, corrected `assert_matches_diffusers` | repo `stack/tanitad/refs/refc_sampler.py` |
| `refc_v3_train.py` — `assert_seams_are_built` + its call site | repo `stack/scripts/refc_v3_train.py` |
| 11 new tests (guard mutations, WP-6 running path) | repo `stack/tests/test_refc_v3_refcv5_wiring.py` |
| this record | repo `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-wp4-wp6-wiring/RESULT.md` |

`raw/probes.txt` — all five probes re-run on the final code · `code/{bitid,loop,steps,mutate,agents}.py`
— the probes themselves, so every number above is reproducible rather than merely reported.

Nothing is stranded on a pod or a worktree. All work is in the repo tree and staged.

---

## 8. ⚠ Two failures of my own, logged because the classes are already in `CLAUDE.md`

1. **CONTENT-VS-EXISTENCE.** `robocopy /MIR` reported success and **did not copy the
   changed `refc.py`**; the test run afterwards reproduced the *old* `14 failed` verbatim
   and looked like "my patch did nothing". Caught only by a positive content assertion
   (marker counts in mirror vs repo: `control_head` **0** vs **12**). ⇒ The mirror sync is
   now `code/`-adjacent `sync.py`, which copies and **verifies by sha256**, requires both
   digests to be 64 chars, and clears stale `__pycache__`. **A copy tool's exit code is
   not evidence.**
2. **`mktree_commit.py` EXITED 0, PRINTED NOTHING, AND COMMITTED NOTHING.** The tool
   prints `[mktree] HEAD = ...` unconditionally and ends with a blob-verified
   `VERIFIED in HEAD` line, so **no output is impossible for a real run** — yet the
   shell reported `MKTREE_EXIT=0`. In the same minutes, `head -20` on that same file
   returned `Invalid request code` and `sed -n '1,45p'` returned empty: the G: mount
   was in an outage window and the invocation was swallowed whole. ⛔ **Nothing in the
   exit code distinguished that from success**, and had I trusted it the entire seam
   would have stayed in the worktree while I reported it committed. Caught by the
   content-marker check with controls: 10 of 11 markers read **MISSING** against
   controls reading 1–95, and `git log` showed a **sibling's** commit at HEAD with my
   subject absent (0 occurrences over a 40-line log that read 40 lines). Re-run with
   stdout redirected to **local disk**, it worked and printed all 22 lines.
   ⇒ **Run any commit tool with its output on local disk, and verify by content
   markers afterwards — never by the exit code, and never only by the tool's own
   self-report.** (Same class as the `robocopy` failure above and as `git ls-tree`
   truncating while exiting 0.)

   ⛔⛔ **AND THE 6 `test_mktree_commit.py` FAILURES HAVE ONE CAUSE, WHICH IS WORSE
   THAN THE OUTAGE.** All six raise `AttributeError: module
   'mktree_commit_under_test' has no attribute 'read_tree_entries'`. MEASURED by
   positive assertion, each with a same-breath control `def main` reading **1**:

   | commit | `assert_names_preserved` | `read_tree_entries` |
   |---|---|---|
   | `e685d90` | **3** | **6** |
   | `2c47dc9` *"…the committer that could land a commit tonight when mm_commit could not"* | **0** | **0** |
   | `HEAD` | **0** | **0** |

   ⇒ A rewrite **DELETED the tool's name-preservation guards**, and the six tests
   written for them have been red ever since — on the tool **every agent commits
   with**. The deleted guards are precisely the ones that refuse **a lost tree entry**
   and **a CR-mangled name**: the failure modes `CLAUDE.md` documents at length, where
   a commit's subject sits in the log while its content is gone from `HEAD`. **A guard
   that has never been shown to fail proves nothing; a guard that has been DELETED
   while its tests still name it is worse, because the test file still reads like
   coverage.**
   ⚠ Not repaired here — rewriting the shared commit path while ~8 agents commit
   concurrently is how a sibling's work disappears. Filed as its own task with the
   recovery source (`git show e685d90:stack/scripts/mktree_commit.py`); the current
   version's compare-and-swap must be KEPT rather than reverted.
   ⚠ `git log -- stack/scripts/mktree_commit.py` returned **empty** on this mount while
   the file demonstrably has history, so the two commits above are what the positive
   probes found, **not** a complete history.
3. **AN UNRUNNABLE PIN READ AS A PASSING ONE.** `assert_matches_diffusers` reported a
   *skip* for as long as `diffusers` was absent, and the moment it ran it could not pass
   (§4). A guard that has never executed is not a guard — the same rule as *"a guard must
   be shown capable of failing"*, with the failure mode inverted: **shown capable of
   PASSING**. ⇒ It now runs and passes in this environment, and the skip count is **0**.
