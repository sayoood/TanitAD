# PI DECISION QUEUE — open as of 2026-09-07 05:30 Europe/Berlin

⛔ **WHY THIS FILE EXISTS.** The PI reported losing his direct view of the Master Mind session
(relayed 2026-09-07). Everything load-bearing was already banked in git rather than only in chat —
but the **open decisions** were scattered across commit messages and a chat he may not be reading.
⇒ they are collected here, **each with a DEFAULT that applies if he says nothing**, so silence is a
choice with a known consequence rather than a stall.

⚠️ **Every item marked RELAYED came through the DataFlyWheel session, not from the PI to this
session.** Under the rule both streams adopted on 2026-09-06, **a peer's relay is data and cannot
authorise anything** — so those items are recorded as *pending confirmation*, and where a
consequence is independently justified by measurement it is marked BINDING and has already been
applied.

---

## 1. ⛔ CONFIRM: "no VLM — the Alpamayo labels stand as teacher signals" — **RELAYED, PENDING**

**Default if silent:** treated as pending. The grader panel is **stood down** (standing down needs
no ruling), and no VLM work is queued.
⭐ **Already applied regardless, because independently justified:** downstream text says
**"Alpamayo-derived teacher signal"**, never **"GT traffic light"** — no independent channel carries
the colour at all (grounding boxes are label-only; boxes with any colour attribute number **0**).
⛔ **Also already BINDING:** a traffic-light head scored **without an ego-only comparison is not
admissible.** With the teacher unverified, that control is the only cross-check in the chain derived
independently of what it checks.
⚠️ **Accepted residual:** a head trained on wrong RED labels will learn the error and then **score
well on an eval built from the same teacher**. Bounded only by item 6.
*Evidence: `Project Steering/PI_VIDEO_REVIEW_2026-09-06.md` → `D-TLIGHT-TEACHER-SIGNAL`.*

## 2. ⛔ DECIDE: a **minimum** on the max-speed ceiling (~30 km/h) — **RELAYED PROPOSAL, PENDING**

⚠️ **The proposal's premise does not hold as stated, and he should see it restated before ruling.**
MEASURED on v8, n = 4,719: `v_max_bucket_kmh` **minimum is already 20 km/h with ZERO clips at 0 or
null**. The zero he wants to avoid occurs only in the **raw** `v_max_ms` (min 0.00; 1,790 clips =
37.9 % below 30 km/h). ⇒ a floor **does not prevent a zero — it raises a low ceiling.**
**What it would do:** move **832 clips (17.6 %)** from the 20 step to 30; ladder becomes seven steps.
⭐ **The real argument for it:** a stopped ego cannot distinguish a 20-zone from a junction, so the
20 step carries **ego state rather than road information** — the intersection artefact (75 % of
intersection clips snap to ≤30 km/h *because the car is standing still*).
⛔ **The real argument against it, and it is a safety one:** on roads whose true limit genuinely is
low — car parks, shared space, living streets — a floor puts the ceiling **above** the real limit, so
the scored `frac_over` **cannot fire on a trajectory that really is speeding**, in exactly the places
where speeding is most dangerous.
**Default if silent:** ladder unchanged at eight steps `(20, 30, 50, 70, 80, 100, 120, 130)` km/h.
⚠️ If it changes, the label side and the model side must re-pin **in the same turn** — the model
asserts the shipped bucket against its pinned ladder (4,572/4,572 + 147/147), and a seven-step blob
against an eight-step pin will refuse loudly.

## 3. DECIDE: the **15-token strategic vocabulary** — a CORPUS gap, not a code gap

`refc_strategic.py` is imported by exactly one file (a test); no CLI flag exists; and **P4 already
failed its support bar — 6 of 15 tokens populated, 11.43 % of the horizon supervisable.** Wiring it
would train a head on almost nothing.
**Default if silent:** stays out. refcv5-v2 trains the **tactical** vocabulary only.
⚠️ `--goal-str` **is** in the running arm's argv but is the 3-unit geometric **bearing** head — it
must not be read as satisfying this.

## 4. DECIDE: `Sayood/tanitad-refc-v3` on HuggingFace is **PUBLIC**

Public since 2026-09-03 — not changed by this session's push, which went to the **private**
`tanitad-refc-v4b`. Links may already be shared.
**Default if silent:** left public. One call flips it.
⚠️ Account storage measured at **991.415 GiB** across 46 repos; **no endpoint exposes a numeric
ceiling**, so no headroom can be stated.

## 5. DECIDE: the `g_str` ~40k-step retrain — a spend decision

The repaired steering signal turns left and **the plan does not follow it** (MEASURED, with a
null-patch control reading exactly zero). Converting it needs a full retrain.
**Default if silent:** not launched. The A40 is occupied by refcv5-v2 until ~Monday evening.

## 6. DECIDE: a **human spot-check of ~50 frames** for traffic-light colour

⭐ The only thing that would bound item 1's residual risk, and it **needs no VLM**, so it is not
excluded by that ruling. It is **PI time, not compute.**
**Default if silent:** not done; the residual stands as an accepted risk.
⛔ Recorded as *available*, deliberately **not proposed**.

## 7. PENDING AN AGENT VERDICT, then possibly a decision: the **RL pilot's adapter**

MEASURED: `assert_conditioning` reads **0** in `refcv3_adapter.py` (control: 4 `def`s), which is what
`stack/scripts/rl_pilot_refc21.py` uses; `refc_adapter.py` — hardened by four commits tonight — has
**no production callers**. ⇒ the guarded path is not the executed path.
**Default if silent:** the executed path is guarded **in place** if that is the right fix; ⛔
*repointing the pilot* would change which code a live RL arm runs and would come back here as a
decision. No RL arm is currently running, so nothing is at risk today.

### ⭐ UPDATE 2026-09-10 — THE MEASUREMENT THIS ITEM WAS WAITING FOR (found while closing item 8)

**MEASURED (Arch+Inference FlyWheel, `…/2026-09-10-rl-pilot-cold-start-allowance/raw/PROBE_V0.json`),
`eval()`, identical frames, both controls valid** — same-`v0`-twice is **bitwise identical**, and a
v0-conditioned build with a real vocabulary **does** move (0.1328 m), so the probe can detect an effect:

| arm | max abs Δ on `anchor_traj` |
|---|---|
| `v0 = 5` vs `v0 = 25` | **0.1356 m** |
| `v0 = 5` vs **`v0` DROPPED** | **0.0130 m** |

…and on `refc.refc_config()` the contract resolves **all 10 channels `False`**, so
`contract.check({"frames": …})` with **`v0` removed PASSES with no refusal.**

⛔ **The path the predicates miss.** The `v0` predicate is `anchors.v0_conditioned OR
sel_reach_clamp` (`refcv3_adapter.py:240`), both False by default — but `refc.py:3199-3200` builds
the measurement encoder's input **unconditionally**, and `refc.py:3204-3206` derives the X15 `keep`
flag from `v0 is not None`. So a batch that merely OMITS `v0` does not just lose the speed: it
asserts **`keep = 0`**, which that same file calls the **X15 zero-collision** — a withheld speed made
indistinguishable from a genuinely stationary car.

⚠️ **Scoped honestly: the contract is NOT wrong about the ACTION SPACE.** With
`v0_conditioned = False`, `roll_bank` really does return the stored `anchors` unchanged. The finding
is about **SCOPE** — the refusal is keyed on the action-space consumer only, while the forward has a
second, always-live one, so a dropped `v0` changes the **policy** with nothing raising.

⛔ **NO BEHAVIOUR WAS CHANGED.** Making `v0` REQUIRED alters which channels a live RL arm must carry,
which is this item's call and its owner's. This supplies only the measurement.


## 8. DECIDE: the RL pilot **cannot load its own cold start** — and the obvious fix is dangerous

⛔ **MEASURED:** `rl_pilot_refc21.py` dies at load with
`RuntimeError: Missing key(s) in state_dict: "decoder.anchor_controls"` — **487 keys in the July
checkpoint against 488 built.** `p_rc21_chain.sh` as written cannot run.
**Cause:** `refc.py:1400` registers `anchor_controls` **UNCONDITIONALLY**, added by the same
2026-09-04 change (`187c513`) that introduced `v0_conditioned`. The cold start
(`refc-diffusion-base-v21-30k`) is from **2026-07-20** and predates it.
⭐ **Nobody noticed for three days, because nothing runs the pilot.** That is as much the finding
as the error is.

**Three fixes, and they are not equally safe:**

| | option | verdict |
|---|---|---|
| **(a)** | `strict=False` | ⛔⛔ **NO.** It deletes the only guard that notices anything — and the buffer initialises to `torch.zeros`, so a **`v0_conditioned=True`** build would then run on an **all-zero control vocabulary**. `refc.py:2091` already calls that *"a plausible-looking WRONG experiment"*. |
| **(b)** | register the buffer **conditionally** on `v0_conditioned` | ⚠️ principled, but it changes the **key set** of a live model class while refcv5-v2 trains against it. This is the exact D-ROLL-1 class — a registration change that made **four checkpoints unrollable** in one commit. High risk for a dormant benefit. |
| **(c)** ⭐ | a **declared allowance**: permit `anchor_controls` to be absent **only when `v0_conditioned` is False**, refuse otherwise, and **stamp what was defaulted** | **RECOMMENDED.** |

⭐ **Why (c) is exactly right rather than merely cautious, and it is derivable from the code:**
`anchor_controls` is **only read when `v0_conditioned` is True** — with it False, `roll_bank`
(`refc.py:1715`) returns the stored `anchors` unchanged and never touches it. ⇒ for a July
checkpoint an all-zero buffer is **harmless because it is never read**; the danger is *only* a
**conditioned** build receiving zeros. So the allowance keys on **the exact flag that determines
whether the buffer matters**, which makes it a narrow checkable rule rather than a blanket
relaxation. It also matches the shape the pilot's new config contract already uses: an absent value
becomes a **stated assumption**, not an invisible one.

**Default if silent:** nothing changes. The pilot stays unrunnable — ⚠️ which is **safe** (it cannot
produce a wrong result) but means **no RL arm can start** until this is settled.
⛔ **What must NOT happen by default:** someone hitting the error and reaching for `strict=False`.
That is why it is here rather than left as a traceback for the next person.

*Owner: `refc.py`'s stream — the registration decision is theirs.
Evidence: `…/Research/2026-09-07-rl-pilot-config-contract/`.*


### ⭐⭐ UPDATE 2026-09-10 — OPTION (c) IS IMPLEMENTED, MUTATION-PROVEN, AND THE PILOT HAS RUN. This item needs NO decision to proceed.

**MEASURED (Arch+Inference FlyWheel).** `stack/tanitad/refs/cold_start.py` (NEW) implements the
declared allowance; `rl_pilot_refc21.load_model` routes through it; `run_posttrain` gained
`extra_record=` so the stamp lands in the run's `config.json`. Evidence:
`…/Research/2026-09-10-rl-pilot-cold-start-allowance/`.

* ⭐ **THE PILOT LOADS AND STEPS.** Its own `load_model` took a **487-key** checkpoint to **488 keys
  on the model** (104,191,577 params) and ran **3 real GRPO steps** — finite losses, weights moved
  (`raw/SMOKE.json`). ⛔ Synthetic weights and frames: this proves the **PATH** and makes **no
  R1/R2/R3 claim**.
* ⛔ **The load is still `strict=True`.** The allowance **completes** the state dict with one
  LITERALLY declared key from the model's own zero buffer; a second missing key, or any unexpected
  key, still raises. MEASURED by AST: exactly one `strict=` in executable code, value `True`.
* ⛔ **The regression arm goes RED when the guard is removed** — mutant M1 (delete the gate): exactly
  the 2 expected arms RED; mutant M2 (`strict=False`, option (a)): exactly the 12 expected RED
  (`raw/MUTATION_PROOF.json`). Baseline GREEN 23/23.
* ⛔ **An operator cannot fake the precondition** — the pilot exposes **no** flag that can set
  `anchors.v0_conditioned` and no `--allow-…`; the flag is read off the constructed `nn.Module`. The
  test intercepts the **real** parser and pins its **15-option surface as a literal**.
* ⚠️ **`anchor_controls` initialises to ZEROS, not random** (`refc.py:1496`, MEASURED `[128, 2]`,
  0 non-zero) — which is why option (a) is worse than it looks: it produces no absurd number at all.

⚠️ **ONE DESIGN POINT THE PI MAY WISH TO OVERRULE, stated rather than buried.** The brief asked that
`v0_conditioned` be read from **the checkpoint's own config, refusing if absent**. Applied literally
that rule **can never be satisfied by this checkpoint** — `AnchorConfig.v0_conditioned` did not exist
until 2026-09-04, so a 2026-07-20 config cannot carry it, and the rule would be a permanent refusal
wearing a guard's costume. As built, the checkpoint's config is a **VETO** (it refuses on `True`, it
cannot authorise), the safety condition is the **constructed decoder's** flag, and an absent leaf is
stamped `ckpt_confirmation: "UNVERIFIED_BY_CKPT_CONFIG"` — never as agreement. A caller wanting the
strict rule anyway can pass `require_ckpt_confirmation=True`.

⛔ **WHAT REMAINS IS COMPUTE, NOT CODE.** MEASURED 2026-09-10: `tanitad-pod3` / `pod4` / `pod5` /
`a40` all refuse SSH (*Connection refused*); Thor is alive. `refc-diffusion-base-v21-30k` is on no
reachable box and not in the repo. ⇒ **a real RL number needs a box + those weights, and that is the
only thing now standing between here and one.**


## 9. DECIDE: refcv5-v2 IS the compose arm, and P1 was excluded ON EVIDENCE — but you asked for environment grounding by name

⭐ **The launch was legitimate and I want that stated first.** §4 of the plan gates the composed
arm on P1 being *"IN or **explicitly declared OUT**"*. P1 was **declared OUT by a two-seed gate**, so
the condition was met. Leaving it out was **not** a shortcut and **not** a validate-arm decision —
it was the gate returning a verdict.

⛔ **But the verdict was a FAILURE, and you asked for this piece by name.** MEASURED, `head − off`,
T1, paired episode-cluster bootstrap, n = 776–779 windows / 28 episodes, **both seeds**:

| distance-keeping | seed 0 | seed 1 |
|---|---|---|
| min headway (m) | **−0.5015** [−0.7712, −0.2366] | **−0.4535** [−0.7915, −0.1639] |
| min time-gap (s) | **−0.0702** [−0.1074, −0.0385] | **−0.0351** [−0.0694, −0.0037] |
| min TTC (s) | **−2.1959** [−4.1408, −0.6110] | **−1.2529** [−2.2522, −0.3694] |

Same sign, overlapping intervals, separated **worse** at both seeds.

⭐ **AND THE DELIBERATE-REGRESSION ARM RELOCATED THE CAUSE, which is why this is a decision and not
a closed case.** A third arm with the join's `clip_id` **deranged** (0 of 26,394 rows on their own
clip; coverage 94.9 % vs 94.7 %) **reproduces the entire degradation**, while `head − shuf` is **not
separated on any distance-keeping metric at either seed**. ⇒ the cost is the **auxiliary detection
task competing for a 17 M trunk at 500 steps** — **not the agent information.**

⚠️ **WHETHER THAT COST VANISHES AT 108 M OVER 40 k STEPS IS UNTESTED.** It is a hypothesis, and the
gate cannot support it. ⛔ The honest framing for any retry is *"an arm whose exclusion may not
generalise"*, never *"recovering a lever we know is there"* — the seed-1 retraction is why: a
one-seed lateral cell was reported as "a direction" and the **deranged** join then beat the real one
on four lateral metrics.

⚠️ **AND WP-A ADDS A REASON NOT TO RUSH IT.** MEASURED (`E-READOUT-CEILING-1`): under a *perfect*
front-end the readout prices out at 16×40 **AP 0.4713** vs 4×4 **0.1583**, but **on the real trunk
none of it transfers** — test AP **0.027–0.034** against a **0.0325** marginal control. ⇒ ⛔⛔ **RETRACTED 2026-09-08 — `R-2026-09-08-wpa-mirror`. THIS SENTENCE WAS THE SECOND ARGUMENT FOR THE DEFAULT BELOW AND IT IS GONE.** WP-A's azimuth address is **MIRRORED** (`s6_oracle.py:82` and `s5_indexed.py:78` are character-identical and are an exact mirror of `bev_aux.azimuth_column` / `psg_targets.azimuth_column`, `max|colf_prog + colf_wpa − 39| = 0.000e+00`), and the mirrored branch **reproduces the banked real-trunk panel**. ⇒ *"the map does not yet contain agents"* is **FALSE** — `D0 − pos_only` is **separated on both geometries** — and the *"cheap order is to give the map spatial structure first"* conclusion is **VOID**.
✅ **What still stands, re-measured under the corrected address:** the perfect-front-end cap **16×40 AP 0.4762 (seed 0) vs 4×4 0.1584**, the axis attribution (**1.64×**, azimuth the dearer axis) and the lateral ratio (1.346 m vs 2.665 m = **1.98×**). ⭐ The oracle ladder is **INVARIANT** to the sense — max effect **0.0038 AP** against a replicate floor of **0.0122** — because its map is *built with the same address the head reads with*; **cross-wiring the two ends collapses it 0.4762 → 0.0750 (105× the sense effect)**, which is why the real trunk, where only ONE end moved, was destroyed and this ladder was not.
⚠️ **CONSEQUENCE FOR THIS DECISION, STATED PLAINLY: the WP-A objection to re-enabling agents NO LONGER EXISTS.** The default below is unchanged — but it now rests on **one** argument (the two-seed gate FAILED, with the deranged-join arm reproducing the whole degradation), not two. See item 11 for the option that argument does not cover. An agent seam re-enabled today would be feeding a representation that
cannot localise them, so the cheap order is to give the map spatial structure first.

**Default if silent:** P1 stays out. refcv5-v2 completes as launched (~step 16,400 of 40,284 at
18:04 Z, landing late Monday) and is scored against refcv4b on the four families. ⛔ The result will
carry *"agent conditioning excluded on a two-seed tiny-rig gate"* as a stated limitation, not as an
omission.

*Evidence: `…/Research/2026-09-07-p1-agent-gate/` (both seeds + the derangement),
`…/2026-09-07-wpa-readout-localisation/`.*


## 10. DECIDE: refcv5-v2 is training 11,286 parameters that receive NO GRADIENT — and the fix needs a LOSS-WEIGHT BUDGET, which is yours

⛔ **MEASURED 2026-09-07** (gradient probe, real model, live argv, trainer's own loss, one
backward — `…/Research/2026-09-07-v7-vocab-reach-census/raw/gradreach_live.json`):

| module | grads that are `None` | `grad_abs_sum` | verdict |
|---|---|---|---|
| `tac_goal_tok_head` | **2 of 2** | **0** | ⛔ **NOT WIRED** |
| `tac_goal_head` | 0 of 2 | 14.368 | ✅ GRADIENT REACHES |

The 22-token tactical head is **built, rollable, and never learns.** ⭐ **The arm is NOT
tactically blind** — the factored `tac_goal_head` trains normally; what is inert is the TOKEN
head. ⚠️ This CONFIRMS the registered `D-TACGOAL-TRAINER-SEAM-OPEN`; what is new is gradient
evidence rather than a code reading, and that **the live run is carrying the dead parameters
now**.

⭐ **Why nobody noticed:** `assert_seams_are_built` asks *"is it built?"* — and it is.
`effective_weights_stamp_v3` enumerates **declared loss weights**, and this head **has no weight
flag**, so it produces **no row at all**. An instrument that enumerates weights cannot see a head
with no weight. Neither guard was wrong; both were asked a narrower question than the claim hung
on them. Now pinned by `stack/tests/test_built_heads_receive_gradient.py`.

**What the fix costs:** the head, the targets, the emitter, the negative policy, the `pos_weight`
and the loss **all exist**. Only the trainer call is missing. ⛔ But adding a loss term means
taking weight from the **`MANEUVER_WEIGHT` budget**, and re-balancing a live recipe's loss weights
is a PI/owner call — already registered, not invented here.

⛔ **What must NOT happen:** wiring it into **refcv5-v2 mid-flight**. That would add a lever
~18k steps in, cost ~20 h of training, and make the refcv4b comparison less attributable — the
`--v2` conflation failure again. The seam is the NEXT arm's business.

⚠️ **AND THE OTHER "next arm" LEVER IS DATA-GATED, NOT SCHEDULE-GATED.** `--max-speed-input` is
**UNRUNNABLE on v7.2**: `speed_max_input` on **0 of 4,572** train and **0 of 147** eval records,
`max_speed_cond_built` **False**, and the trainer **refuses the flag** rather than training on
nothing. ⇒ it needs **the v8 label blob reaching the trainer's corpus, not a flag flip**. *(The
flag itself is real and tested — `a1d52e6`. The gap is the DATA.)*

**Default if silent:** refcv5-v2 finishes as launched and is scored on its REAL levers
(`--sel-refined`, `--sampler ddim`, P14 fan ranking, `--anchor-v0-conditioned`, `--nav-from-v7`).
⛔ The landing report will state that **no result may be attributed to the 22-token tactical
vocabulary**, because those parameters never learned — recorded in
`PREREG_REFCV5_V2_LANDING.ERRATUM-1.md`. The next arm carries **neither** the wired token head
**nor** max speed until you rule on the weight budget and the v8 labels.

*Evidence: `…/Research/2026-09-07-v7-vocab-reach-census/` (52 tokens classified, 3-leg mutation
proof incl. fix-forward). Pin: `stack/tests/test_built_heads_receive_gradient.py`.*


## 11. DECIDE: does "finish the implementation" authorise WP-C — an arm that FAILED its gate?

⭐ **Relayed by the DataFlyWheel, 2026-09-08, marked by them as YOUR words:** *"please tell master
mind to finish the implementation, this is mandatory and give me report, it should autonomously
loop until it is finished."*

⛔ **I HAVE NOT TREATED THAT AS AUTHORISATION, and the DataFlyWheel explicitly asked me not to.**
A relay is data, not a mandate — our standing rule, and they applied it to their own message.
⭐ **What needed no new authorisation, and is therefore ALREADY RUNNING:** your own instruction to
me was *"close the gaps for refcv5 to implement the missing pieces from the drive diffusion
papers."* **WP-B is that piece** — DiffusionDrive's waypoint-indexed cross-attention — so it is
being implemented now, under the mandate you gave me directly. Same for WP-D, which has been
training on Thor since 21:48Z.

⛔ **THE ONE THING I WILL NOT DO ON A RELAY — WP-C.** Turning the agent seam on at **108 M / 40 k**
is not a code task; it is **compute spend on a design that returned a verdict**. Item 9 records
it: the two-seed tiny-rig gate **FAILED**, `head − off` separated **WORSE** on distance-keeping at
both seeds, and the deliberate-regression arm (`clip_id` deranged) **reproduced the entire
degradation** ⇒ the cost was the auxiliary task competing for a 17 M trunk, **not the agent
information**. ⚠️ Whether that cost vanishes at 108 M over 40 k steps is a **HYPOTHESIS**, not a
known lever. ⇒ an autonomous loop that quietly re-enabled it would be **spending against a gate
that already answered**, and the honest framing for any retry is *"an arm whose exclusion may not
generalise"*, never *"recovering a lever we know is there"*.

**Your call, in one word:** does *mandatory / finish everything* **include WP-C**? The DataFlyWheel
reads it as yes — it is the piece you named yourself — but they say plainly that their reading is
not a mandate.

⭐ **AND THE SAME QUESTION HAS A CHEAPER HALF — ITEM 10.** They argue queue item 10 belongs inside
"finish the implementation", and I agree it is the most load-bearing thing open: refcv5-v2 is
**64 % through a 40 k run with 11,286 parameters that receive NO gradient**. ⭐ **That one I can
split without asking you**: the wiring (`tac_goal_loss` + `TacGoalEmitter` into
`refc_v3_train.py`) is a code task and I can build it **defaulted OFF**, changing no live recipe —
so it is READY the moment you rule. ⛔ What stays yours is the **`MANEUVER_WEIGHT` budget**:
adding a loss term takes weight from somewhere, and re-balancing a recipe's losses is a decision,
not an implementation.

**Default if silent:** WP-B and the item-10 wiring are implemented and **left OFF**; **WP-C is NOT
launched**; refcv5-v2 finishes as launched and is scored on its real levers. ⛔ Nothing spends
GPU on a gated arm without your word, and ⛔ no result is attributed to the 22-token tactical
vocabulary, which never learned.

### ⭐⭐ UPDATE 2026-09-08 — WP-B FOUND A THIRD OPTION, AND IT IS NOT THE ARM THAT FAILED

WP-B is now implemented, tested and banked (`835286c5`) — 0 GPU, `--agents` untouched, WP-C left
exactly as open as it was. Building it surfaced an argument that **changes this decision rather
than answering it**, so it is recorded here rather than acted on.

⛔ **Item 9's failure had a MEASURED cause, and that cause is structurally absent from the oracle
rung.** Item 9 recorded `--agents head` degrading **distance-keeping at both seeds** (min headway
−0.50 / −0.45 m, time-gap −0.070 / −0.035 s, TTC −2.20 / −1.25 s). But the deranged-join arm
— `clip_id` shuffled, 0 of 26,394 rows on their own clip — **reproduced the entire degradation**,
while `head − shuf` was not separated on any distance-keeping metric at either seed. ⇒ the cost
was the **auxiliary DETECTION TASK competing for a 17 M trunk**, not the agent information.
⭐ **`--agents oracle --w-agent 0` carries no detection loss at all**, so the mechanism that
produced item 9's failure **cannot occur in it**. That makes the honest ladder:
*does the ADDRESS help, given agents?* — **before** — *can a DETECTOR supply them?*

**So the choice is three-way, not two-way:**

| | option | what it costs, and what it would settle |
|---|---|---|
| **(a)** | launch nothing | WP-B stays built and unrun. Nothing spends. |
| **(b)** | `--agents head` at 108 M / 40 k | ⛔ **~14 GPU-DAYS**, and it re-runs the arm whose gate returned a verdict. The hypothesis is that scale dissolves the trunk competition — untested. |
| **(c)** ⭐ | `--agents oracle --w-agent 0` + `--wp-index on`, a **6,000-step gate** | **~40 GPU-hours.** A DIFFERENT question, whose known failure cause is structurally absent, and every bar is a **ratio to a floor measured in the same panel** so a short arm is still valid. |

⚠️ **(c) is still compute and still your call** — it is recorded as an option, not taken. And it
is not a way to smuggle WP-C in: it does not answer whether a *detector* can supply the agents,
which is what item 9 was about. It answers the prior question, more cheaply, and would tell us
whether the address is worth a detector at all.
⚠️ **The cost was nearly misquoted by 6×.** The receipt's MEASURED rate is **4.0 s/step**, and it
explicitly retracts its own earlier ~1.2 s/step as the `step_s` trap; a 0.67 s/step figure was
nearly quoted from it. The 14-GPU-day number above is the corrected one.

**Default if silent, unchanged:** nothing launches. WP-B is built, tested and OFF; `--agents`
stays `off`; the arm that failed its gate is not re-run.

*Evidence: item 9 (P1 gate, both seeds + the derangement), item 10 (the gradient probe),
`…/Research/2026-09-07-v7-vocab-reach-census/raw/gradreach_live.json`.*


---

## Not a decision — the state, for orientation

**refcv5-v2 is training** on the A40 (PID 2560646, watchdog 2561632), step ~5,400 of 40,284 at
05:00 Z, **4.11 s/step ⇒ ~37 h**, landing ≈ Monday evening Berlin. It carries P14's validated
emitted-fan ranking (`sampler_ranks_the_fan: True`, the exact key v1 shipped `False`) and the
22-token tactical goal vocabulary on a **provably rollable** head.
⭐ **The comparison is already built and validated**: run on refcv4b it reproduces the published
landing (66/66 model-free rows bit-exact, `refcv4b − refcv3` **−0.1455 [−0.1655, −0.1240]** against a
published −0.1444) and **writes FAIL on both bars by itself** — the bar bites its own baseline.
⛔ **The bar it must clear:** refcv4b only **ties** the do-nothing baselines (`os − ha` −0.0021,
`os − ha0_ext` +0.0101, neither separated) ⇒ **refcv5-v2 must BEAT `ha0_ext` separated, or it has not
learned to drive either.**
⚠️ **What a clean margin will still NOT establish:** `--sampler ddim` is **stochastic at inference**
where refcv4b was deterministic, so it will not show the result survives a second **inference** draw.
And both arms are **oracle-nav** — fair to each other, but neither carries a production nav command.
