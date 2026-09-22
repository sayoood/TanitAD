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

---

## ⭐⭐⭐ PI DIRECTIVE 2026-09-11 — refcv6 TRAINING IS STOPPED. The programme pivots to Qwen-Drive augmentation + LiDAR-supervised BEV.

**Sayed, verbatim, to the Master Mind in his own channel:**
> *"stop the training and augment the data set by qwen drive inference. Show the visulaization of a
> small sample to validate the output. train the bev transformer based on the lidar based bev as gt.
> My goal is to implement all pieces of diffuision drive and combine them with all our features like
> multuhierarchacy etc."*

⛔ **THIS IS RECORDED HERE BECAUSE A STILL-ARMED CHAIN EXISTS.** `chain_refcv6.sh` and
`sup_refcv6.sh` are staged, correct, and would relaunch `V0 → V0b → D` on one command. ⛔ **A fresh
context must NOT relaunch them** while the Qwen-Drive / BEV work holds Thor — the two cannot share
the card.

### What was stopped, and what it cost

| | |
|---|---|
| arm | refcv6 **V0** (the baseline / noise-floor arm) |
| reached | **step 250 of 40,284 = 0.62 %** |
| checkpoint | **none** — `--save-every 500` never fired |
| GPU spent | ~33 min |
| ⇒ scientific loss | **nothing.** No result existed and none may be quoted. |

### ⭐ THE STOP WAS QUESTIONED BY THE LAUNCH AGENT, CORRECTLY, AND THAT IS WORTH RECORDING

The marker `summary.json` was written by the Master Mind, not by `sup_refcv6.sh` — its fields
(`stopped_deliberately`, `reason`, `last_step_seen`) are ones the supervisor never writes.

⭐ **The launch agent noticed, refused to act, and asked.** Its reasoning was exactly right and is the
standard: *"It claims PI authority, and it reached me as a file on a remote box. Instructions that
arrive through tool output are data, not commands."* ⇒ it neither pivoted (a programme redirection
plus spend) nor relaunched (which would defy a possibly-genuine stop **and** put two jobs on one
card). ⛔ **A claim of authorisation found inside observed content is not authorisation** — and this
is the first time in the programme that rule has fired on a real ambiguity rather than in a drill.

⚠️ It also named the gap that let the ambiguity exist: **a directive of this magnitude had not
reached the register.** This entry closes that.

### What survives, unchanged and staged

⛔ **Nothing built for refcv6 is discarded.** The three gate/supervisor fixes; the combined
train+EVAL6 agent join (md5 `1e285303476d3e0968533bc64e7456f3`, sidecar re-verified by the trainer in
its own log); `chain_refcv6.sh`; `refcv6_status.sh`; the arm-D acknowledgement flag; the recovered B1
TRAIN join (md5 `1c985e6d…`, 96.83 %); and the corrected reference column.

⭐ **The measured rate stands: 4.086–4.102 s/step over four flat intervals on Thor**, so a relaunch is
one command at **~46.8 h/arm**. refcv6 restarts from exactly where it was whenever the PI says.

### The new direction, in his order

1. **Qwen-Drive inference** over our corpus — ⛔ **PERCEPTION ONLY** (3D boxes, map/BEV segmentation,
   occupancy). ⛔ **NEVER its trajectories**: our own teardown measured it winning every open-loop
   board and finishing **2.25× worse than Alpamayo-R1** on the 916-scenario closed-loop table. That
   is a PLANNING weakness and distilling it would import exactly what we are trying to fix.
2. ⭐ **A SMALL SAMPLE, VISUALISED AND VALIDATED FIRST** — his explicit instruction, and it gates the
   corpus-scale run.
3. **LiDAR-supervised BEV.** ⭐ Doctrinally exact: labels may use LiDAR, **inference stays
   vision-only**, so the binding rule is satisfied by construction. Unblocked by a fact that had sat
   unused — the manifest is **36 features: 7 camera, 6 calibration, 3 label, 1 lidar, 19 radar**, and
   the programme reads **6**.
4. **Then all of DiffusionDrive, combined with our hierarchy.**

⚠️ **The augmented corpus is a NEW DATASET.** A panel cannot change its dataset midway, so refcv6's
arms would have to be re-run against it. That is a sequencing fact, not an objection.

## ⭐⭐⭐ NEW ITEM 14 (2026-09-13) — Qwen-Drive: the rejected video is diagnosed and fixed. Your validation gates the corpus run.

**What you rejected was not the model.** Two of your three complaints (*"occupancy has no details"*,
*"map small and noisy"*) were **my renderer** — I decoded the outputs from a guessed schema while
Qwen-Drive ships its own visualizer. The third (*"boxes not correct"*) was **both** my renderer and our
input packing, which fed focal lengths the weights never saw.

⭐ **MEASURED on Qwen-Drive's own demo, where GT exists:** on Thor it finds **54/57** boxes, map mIoU
**0.799**. Changing **only** the focal to our old packing's value gives **0/57**; dropping two cameras
changes nothing (**54/57**).

⭐ **The fix (v2): a virtual nuPlan 8-camera rig** reprojected exactly from the PhysicalAI cameras.
Same 14 instants, same GT: recall **0.182 → 0.404**, precision **0.250 → 0.429**, near-range (0–20 m)
recall **0.292 → 0.646**, occupancy agreement with LiDAR **0.370 → 0.477**. Full-clip videos rendered by
Qwen-Drive's own visualizer are in `…/2026-09-13-qwen-drive-usage-review/media/`.

⚠️ **v2 is a large fix, not a finished teacher**: recall 0.40 on our GT vs 0.95 on their demo; side
views are weakest (cross cameras sit at 0.81 m); **map and occupancy cannot be scored** on PhysicalAI
(no semantic GT exists).

**DECIDE (a):** do the v2 videos pass your visual validation?
**DECIDE (b):** scope of the augmentation run. v2 needs **5 more cameras per clip** than we hold (the
front tele is unused): **7.13 GB per 100 clips** (MEASURED, chunk 768's five zips, 100 clips each), and
Thor runs **~4.1 s per frame** (MEASURED, 391 s for 96 frames incl. model load). At 2 Hz over 20 s clips:
B1-EVAL 139 clips ≈ **9.9 GB + ~6.3 h Thor**; parity 2,376 clips ≈ **170 GB + ~108 h Thor** (ESTIMATED by
scaling those two measurements; no transfer wall-clock is claimed — single-stream HF rates have spread 7.3×).
**DECIDE (c):** which layers to keep. Detection is already covered better by `obstacle.offline`; the
teacher's unique value is the **map** (PhysicalAI ships none) and **semantic occupancy** — both unscored.

⭐ **MAP QUALITY MEASURED (same day, `…/RESULT.md` §12):** the map layout is real but coarse (road edges on a real curb 50.8 % vs 74.7 % for a true map); the occupancy covers only ~56 % of agents vs ~99 % in-distribution. ⇒ for **(c)** the recommendation is **map yes (after fusion + LiDAR cleaning), occupancy no (use LiDAR)**.

**Default if silent:** nothing corpus-scale launches. The 139-clip B1-EVAL slice is prepared as the
first rung (it is the slice the LiDAR-BEV stream builds GT for, so map/occupancy can be checked against
LiDAR at scale) and waits for (a).

---

## ⭐⭐ STATUS AS OF 2026-09-11 — read this first; four items moved and TWO ARE NEW

### ✅ CLOSED — no decision needed

| item | what changed |
|---|---|
| **8** — the RL pilot's 487-vs-488 cold start | ⭐ **CLOSED.** Option (c) implemented, mutation-proven, and **the pilot loaded the real July checkpoint and ran 2,000 steps.** ⚠️ Correction: the missing tensor is **ZEROS, not random** — so every one of 128 anchors reads *"hold speed, go straight"*, and the arm trains, converges and produces a full plausible result with **no absurd number anywhere** to catch it by. ⛔ `strict=False` appears nowhere. |
| **`--w-agent`'s value** — I had listed this as needing you | ⭐ **Resolved from the record, not by asking.** `1.0` is **pre-registered** at `…/2026-09-07-p1-agent-gate/PREREG.md:46`, written before any outcome under `D-P1-AGENTCOND-1`, and executed with a banked config. ⛔ Retracts my own claim that it was measured-but-unregistered. |
| **2** — a ~30 km/h minimum on the max-speed ceiling | ⭐ **The premise fails and the channel is BUILT.** The bottom step of the ladder is **already 20 km/h with 0 nulls**. Per your ruling *"stick to the labels we created in the data set with the logic of minimal speed"*, the v8 block now exists at **4,572/4,572 train and 147/147 eval** — a pure transform of `SPEED_BAND`, no new labels. |

### ⏳ MEASUREMENT IN FLIGHT, so you need not answer

| item | what is happening |
|---|---|
| **10** — the `MANEUVER_WEIGHT` budget for `--w-tac-goal` | ⭐ **Being MEASURED on Thor rather than guessed**, which is what I offered. A tiny-rig sweep across four weights plus a **zero-weight control that must read exactly 0.0** returns a grad-versus-weight curve and the point at which the primary objective starts to degrade. ⛔ **The value remains yours to ratify** — you will get a measurement and a recommendation, not a decision. |

---

### ⛔ NEW ITEM 12 — refcv6's arm D **CANNOT START**, and the refusal's own reason is false

`refc_v3_train.py:468` raises a hard **`SystemExit`** on any arm combining **`--sampler ddim`** with
**`--w-u0 0`**. refcv6's BASE carries `--sampler ddim` (`arms.py:110-111`) and **arm D IS `--w-u0 0`**
(`arms.py:286`). ⇒ **that is exactly the refused pair.**

⛔ **Arm D is the arm I told you to run FIRST and called "one flag and free."** It would have died in
the first second of a run — discovered only **after** a pod was provisioned and paid for.

⭐ **The refusal's stated reason is measurably FALSE.** It claims `control_head` *"stays at its zero
init forever"* without the `u0` term. In a single forward, `control_head` reaches `out["anchor_traj"]`
— the tensor the **matched-anchor L1** gathers — for `grad_abs_sum` **9.39e4**, against **1.07e5**
through `u0_hat` and **exactly 0.0** for three no-information controls in the same forward. The
justification rotted while the code kept enforcing it. Class **`JUSTIFICATION-ROT`**.

⛔ **The REASON was refuted, not the DECISION.** The refusal stands and only its message was
corrected, because whether that arm runs is **your ruling and not an agent's**. A false justification
licenses re-opening a question; it never licenses overriding the answer.

⭐ **Why the arm still matters:** `D-DDV1-NO-DENOISING-LOSS` — DiffusionDrive v1 has **no
ε-prediction and no denoising MSE at all**, and its `diff_loss_weight = 20.0` is dead code. Our
`--w-u0 0.5` is an **invention, not a port**; our trainer's own default is already **0.0**; and
refcv5-v2, the arm that opted into 0.5, is the one that lost longitudinally.

| | |
|---|---|
| **(a) ⭐ DEFAULT if you say nothing** | authorise `ddim` + `--w-u0 0` behind an **explicit acknowledgement flag**, so the record shows a deliberate operator choice rather than a bypass — the same shape as `control_units_source: cli-override-legacy-file` |
| **(b)** | drop arm D, and rewrite the review's §3.3 ordering, which currently rests on it |

---

### ⛔ NEW ITEM 13 — compute: refcv6 needs a pod, and it is the only hardware blocker left

The A40 is **stopped and gone**; every irreplaceable artifact was verified byte-exact off it first.
Thor and the dev-box RTX 4060 are **probe-and-eval class**, not 40 k-step class.

**Priced from refcv5-v2's own record, cross-checked two ways** (40,284 steps × 4.2 s/step measured
live = 47.0 h; the banked checkpoint's own wall-clock ≈ 47 h): **47 GPU-hours per arm.**

⭐ **The panel is 235 GPU-h, not 423, because the noise floor only has to be measured ONCE** — one
baseline replicate, then four single-lever arms read against it, with a confirmation replicate bought
only by an arm that looks like it cleared.

**DEFAULT if you say nothing:** provision **one A40-class pod** and run **arm 0's pair only —
94 GPU-hours.** ⛔ That pair is not a consolation item: it **is** the rig's noise floor, and without
it no other arm's bar is readable. ⚠️ This replaces my earlier default of *"run arms 0 and 1"*, since
arm 1 is arm D and arm D cannot start.

⚠️ **A measured reason this is not optional, from tonight:** on the dev-box RL rig, the **control**
arm's own seed-to-seed spread was **145 percentage points** on the headline metric — and a lever that
looked like a clean 5.2× improvement turned out to sit **inside** it.

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

> ### ✅ RULED BY THE PI, 2026-09-20: **SKIP.**
> Verbatim: *"skip checking traffic light check"*. ⇒ the spot-check is **NOT done** and item 1's
> residual traffic-light risk is an **ACCEPTED RISK**, which was the stated default-if-silent.
> ⛔ **Item 6 is CLOSED.** It leaves the PI queue and must not be re-proposed.
> ⚠️ What the accepted residual IS, so it is not lost: the **779** GT traffic-light tactical
> goals in the v7.2 label release carry `provenance: "vlm-cot"` and were never human-verified.
> Any claim resting on traffic-light colour states that.


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


### ⭐⭐ CORRECTION 2026-09-10, SAME DAY — THE REAL PILOT HAS RUN. Item 8's defect is closed against the artifact that produced it.

⛔⛔ **RETRACTING MY OWN LINE ABOVE** (*"what remains is compute … the weights are on no reachable box"*).
**FALSE.** It rested on two true probes — the RunPod fleet refuses SSH, and the repo does not hold the
file — and a false conclusion. The checkpoint is on **the dev box**, at exactly the path
`p_rc21_chain.sh` has always waited on:
`C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt`, **1,250,838,325 B**, md5
**`8f10d6f934f4199e11ddc7352e074939`** — the chain's own `WANT_MD5`, to the character.
⭐ Operating-standard **rule 2** (*absence at ONE location is not absence*), committed by the agent
quoting it. The vector: I searched **where a big checkpoint usually lives** instead of **what the
consumer says it opens** — and the consumer named the path all along.

**MEASURED (dev-box RTX 4060, `PILOT_EXIT=0`, `raw/realpilot/`):**
* `CONFIG CONTRACT: case PARTIAL via sidecar` — **38/98 fields compared, 0 disagreements**;
* `DECLARED ALLOWANCE: 487/488 keys from the checkpoint; DEFAULTED ['decoder.anchor_controls']`;
* **`cold start loaded: 104,191,577 params @ step 29999 (488 state-dict keys)`** — the real
  `refc-diffusion-base-v21-30k`, not a stand-in;
* `train eps 54 · val eps 15 · device cuda`, both readouts produced, `ckpt_after.pt` saved.
* ⭐ **The ABSENT case is CONFIRMED on the real artifact:** the sidecar's `cfg.anchors` carries
  exactly `{n_anchors: 128, pool_size: 4096, seed: 0}` and **no `v0_conditioned`** — so the literal
  "refuse if the checkpoint's config does not carry it" rule **would have refused this exact
  checkpoint**, and did not need to. The pilot's independent weight-side corroborator agrees:
  `state: BUFFER_ABSENT`, `verdict: CORROBORATED`.

⛔ **THE R1/R2/R3 THIS RUN PRINTED ARE NOT A RESULT AND MUST NOT BE QUOTED AS ONE.** It ran **8
steps**, not the pre-registered 2,000; **one seed**, so the run-to-run floor (`H-ESTIM-SEED-1`) is
unmeasured; and no paired CI was computed. They evidence only that the reward/advantage/readout
paths execute on real windows.

⇒ **THE ONLY OPEN ITEM IS A SPEND DECISION, AND IT IS YOURS:** `p_rc21_chain.sh` is runnable **as
written** — checkpoint, both `_epcache` dirs and both agent `jsonl` files all verified present — and
wants **~2.5 h of dev-box GPU** for the 2,000-step P-RC21 plus the 300-step `hackable` regression arm.


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

## 15. DECIDE: the VERTICAL field at 256x1024 — your instruction costs a 1.88 m blind strip in front of the ego

**You said, 2026-09-16:** *"so let increase the azimut resolution and use 256x1024, we will do this
for all our future trainings."* That is being followed. This item exists because following it has a
cost you were not told about when you said it, and because **the alternative is already built**, so
changing your mind is cheap right now and expensive after the corpus rebuild.

**The mechanism.** `calib.cylindrical_rays` uses **one `f_ref` for BOTH axes**
(`phi = (u-(W-1)/2)/f_ref`, `y_n = (v-(H-1)/2)/f_ref`). Holding `H = 256` while `f_ref` rises
305.577 -> 488.924 — which is what keeps azimuth at 120 deg across 1024 columns — narrows the
**vertical** field by the same 1.6x. Azimuth is bought with elevation. This is a property of the
shared resampler, not of the cache build.

| | **256x640** (today) | **256x1024** (your instruction) | **408x1024** (BUILT) |
|---|---|---|---|
| HFOV / deg-per-column | 120.000 / 0.1875 | 120.000 / **0.1172** | 120.000 / **0.1172** |
| **VFOV** | **45.456 deg** | **29.341 deg** (64.5 %) | **45.296 deg** (99.6 %) |
| rows of today's frame kept | all | **47.5 … 207.5** (drops 48 top + 48 bottom) | **0.0 … 255.0** |
| **nearest visible road ahead** | **3.14 m** | **5.02 m** | **3.15 m** |
| stride-16 tokens | 640 | 1024 | 1664 |
| corpus cache (4,713 clips) | 178.0 GB | **273.6 GB** | **386.5 GB** |
| resnet101 K=3 activations | 645.9 MB | 1033.5 MB | **1662.9 MB (1.609x)** |
| corpus build wall | — | **8.7 h** | **10.6 h** |

⭐ **The number that matters is the last-but-four row.** MEASURED camera height **1.3158 m**
(median over the 141 eval extrinsics; range 1.213–1.662) and mount pitch **+0.016 deg** (median).
The steepest downward ray is `atan((H/2)/f_ref) + pitch`, so the nearest patch of road the camera
can see moves from **3.14 m to 5.02 m** ahead: **a new blind strip of 1.88 m directly in front of
the vehicle** (4.68–6.01 m across the 141 clips' own heights and pitches).
⚠️ Flat ground, no suspension pitch — first-order. ⚠️ **This corrects my own earlier figure of
"3.6–5.7 m", which assumed a 1.5 m camera height instead of measuring it.**
⚠️ **And one thing I did NOT verify:** I read pitch from the extrinsics quaternion assuming an
intrinsic ZYX convention with +x forward, and did not independently confirm the rig's rotation
convention. It barely matters here — the median comes out **+0.016 deg**, so the pitch term is
negligible and **the 3.14 -> 5.02 m result is driven by the field geometry alone**
(`atan((H/2)/f_ref)`), which needs no convention at all. If the mount were genuinely pitched a few
degrees down, both numbers shrink together and the 1.88 m gap changes little.

**Why it may or may not matter.** Our planner places waypoints in exactly that near field, and the
BEV lift integrates the ground plane it can see. Against that: DiffusionDrive/NAVSIM's own input is
**1024x256**, a 4:1 frame — so **256x1024 is the paper-matching shape** and the papers evidently
drive with it. We have no measurement either way on OUR corpus; the ablation has not been run.

**⭐ THIS DOES NOT BLOCK ANYTHING TONIGHT.** Both caches are built and validated on the 139 eval
clips, so the pipeline validation you asked for runs either way. The decision binds only the
**corpus rebuild**, which cannot start until SAM3 production finishes (~22 Sep).

**Default if silent: 256x1024 — your instruction, unchanged.** Two reasons, and neither is that I
think it is the better geometry: (a) you named the shape explicitly and overriding an explicit
instruction on my own judgement is not my call; (b) **the HF quota is a hard ceiling** and 408
costs **+41 % cache** (386.5 vs 273.6 GB). If your intent was *"more azimuth"* rather than *"that
exact shape"*, **408x1024 delivers the azimuth and keeps the field**, and it is built and gated
today — say the word and it is a flag.

⚠️ **A fourth option exists and was NOT built.** An exact 1.6x supersample needs H = **409.6**, so
no integer height is both field-exact and aligned to the 640-row grid; 408 lands half a pixel off.
**H = 416** is integer-aligned and slightly *exceeds* today's field (VFOV 46.092 deg, nearest road
**3.09 m**) at ~2 % more disk than 408. Evidence class **DERIVED, not measured** — it has never
been built or gated.

*Evidence: `TanitAD Research Lab/Data Engineering/Research/2026-09-16-256x1024-cache/` —
`RESULT.md` Escalation 1 and `raw/geometry_comparison.json`; camera height and pitch recomputed
here from `extrinsics141.json`.*

---

## 16. DECIDE: the gradient-conflict statistic we pre-registered CANNOT SEE the defect it was written to catch

This is the *"a check that shares the defect it checks for"* class (`e4af94f`), and it was caught
by the agent that built the instrument, before any arm ran.

**The algebra.** `cos(g_traj, g_aux)` is invariant under positive scaling of either argument — that
is precisely what a cosine quotients out. `E-DEC-18`'s measured failure mechanism was **magnitude**
("the aux gradient is 10–30x the planner's"). An **angle** statistic can never see a magnitude.

**Verified independently, not taken on report.** On a shared trunk with two heads: a 30x aux moved
the cosine by **1.49e-08** (rounding), and a **32x** rescale — a power of two, so binary-exact —
left the cosine **bitwise identical**. The agent measured **3.4e-8** on the real model with the same
32x bitwise result. Two scale-sensitive channels computed from the **same two gradients at no extra
backward** do see it: `ratio = |g_aux|/|g_traj|` and `proj = <g_traj,g_aux>/|g_traj|^2`, both
reading **exactly 30x**.

⛔ **Consequence for the panel: pre-registered criterion `B2` would have called the KNOWN defect
REFUTED.** On the 40-step worked example the deliberate 30x defect reaches median cos **-0.0515**
against B2's `< -0.05` line with **75 %** of steps negative against B2's `>= 80 %` — it scrapes one
threshold and misses the other. `ratio` separated by **21x** and was never ambiguous.

**Default if silent:** the detector ships with **all three channels** logged (`cos`, `ratio`,
`proj`), and **`B2` is restated as DIRECTION-ONLY** with a new magnitude criterion on `ratio`
beside it. Nothing is gated on `cos` alone. This is the conservative reading and it costs nothing.

⚠️ **A second correction rides with it: the cost estimate in the prereg is out by ~2x.** The
prereg said "one extra backward ... no extra GPU-day". MEASURED over 3 interleaved replicates on
the real model at 3 trunk sizes: **+71.7 % / +83.8 % / +104.2 %** for the pre-registered `probe`
mode, rising with trunk size. A `subtract` mode (one backward, bit-identical) costs
**+26.3 % / +53.1 % / +62.2 %** but changes what the planning side *means*
(`L_total - L_aux`, not `L_traj`) — stamped `cd_plan_side` on every row so the two can never be
silently compared. ⛔ Every number here is a **smoke model on a synthetic corpus**: the instrument
is proven, the readings are not a result, and nothing was measured on GPU.

*Evidence: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-conflict-detector/`
— `RESULT.md`, `raw/overhead.json`, `raw/worked/{1x,30x}/metrics.jsonl`.*

---

## 17. DECIDE: the repo has been leaking CLIP IDS into banked records for months — 64 full UUIDs and 530 prefixes

**The binding rule:** clip ids appear in repo artifacts **only as sha12**, where
`sha12(clip_id) = sha256(clip_id)[:12]` (`semantic_map_gt.py:117`) — a HASH, not a
truncation. ⛔ **MEASURED 2026-09-17** across `Project Steering/**/*.md` and
`TanitAD Research Lab/**/*.md` on the current tip:

| | count | files |
|---|---|---|
| **full clip UUIDs** | **64** | across **53 files** |
| **8-char UUID prefixes** (e.g. *"clip `<8 hex>` is a road bend"* — not quoted here, for the obvious reason) | **530** | |

Worst offenders: a label REVIEW_SHEET (25 full UUIDs), the refcv4b RUNBOOK (15),
a SHIP_VERIFY (8), and `GOALS_AND_CLAIMS.md` itself (2 full + 23 prefixes). ⚠️ An
8-char prefix is not a partial identifier in practice — across 4,719 clips it is
**uniquely identifying**, so it is a clip id wearing a short name.

⛔ **This is NOT something to fix quietly, which is why it is a decision and not a
task I did.** These are **banked evidence records**. Rewriting an identifier
inside a sentence like *"`<8 hex>` genuinely CONTAINS cyclists"* edits the
evidence a past conclusion rests on, and a reader who later pulls the raw artifact
by sha12 must still land on the same clip. There are three honest options and they
are not equivalent:

1. **Redact in place, mapping each id to its sha12.** Clean going forward; it
   rewrites history-bearing documents, and any external copy of them stops
   matching.
2. **Leave the record, fix the RULE going forward** — a pre-commit guard that
   refuses new leaks, with the existing 594 grandfathered and listed. Honest about
   what happened; the leak stays in git history either way (it always does — a
   redaction does not remove it from earlier commits).
3. **Decide the rule was never meant to cover prose in the Research Lab**, only
   code, filenames and machine-readable artifacts — in which case 530 of these are
   not violations at all and the rule should say so.

**Default if silent: option 2.** It is the only one that is both reversible and
honest — the history already contains them, so a redaction buys appearance rather
than secrecy, while a guard stops the count from growing. ⚠️ I have added no guard
tonight: a guard that fires on 594 pre-existing hits would be red from the first
run and would be turned off within a day.

*Evidence: MEASURED by a scan of the two trees at tip `0243ce4`; the scan is the
same `UUID`/8-hex matcher every landing this session has run against the corpus's
own clip list.*

---

## 18. DECIDE: the tactical layer can see the AGENTS but not the MAP — and unblocking it is a design call, not plumbing

**You asked, 2026-09-16:** *"The tactical layer must learn to emitt the valid tactical behaviors …
It shoudl learn them from the scene embeddings, **for the agent and the map**."*

⭐ **The agent half now works.** As of `ec08291` the decoder trains: all 8 probed heads reach,
`grad_abs_sum` **0.162 → 991.2**, `n_grad_none = 0` on every module on every step — where at the
previous tip **2.26 M parameters read exactly 0**.

⛔ **The map half cannot be reached without a structural change.** MEASURED:

| | |
|---|---|
| `refc_v3.RefCV3Model.forward` | **never passes `bev_tokens=`** to `self.core(...)` — the call site passes `**_core_kw`, carrying `scene_hook` alone |
| the BEV encoder | lives on the **trainer's wrapper**, and runs **AFTER** the core forward, on `out["fmap_s16"]` |
| ⇒ at the hook | **no BEV token exists yet** |

So `--tac-decoder-d-bev > 0` **refuses**, and a run stamps `sources: ["agent"]` /
`bev_tokens_reach_decoder: false`. ⭐ **The refusal is the right behaviour** — a decoder quietly
attending to agents only while the record says "agent and map" is exactly the class of defect that
has cost this programme the most.

### What it costs while it stays blocked

⚠️ `E-REFCV6V2-TACTICAL` can only be tested in its **agent-only** form. Behaviours that are a
property of the **map** — lane keeping, corridor offset — have **no evidence to learn from**, so a
per-class result on those tokens is **uninterpretable**, not merely weak. They must carry that
scope or be excluded from any headline.

### The two options, and the second is the real question

1. **Move the BEV encoder into the model forward**, so a BEV token exists before the hook fires.
   Mechanical, and it changes the model's forward contract — every bit-identity proof landed
   tonight would need re-running against the new baseline.
2. ⭐ **Then decide whether the behaviour decoder may BACKPROP INTO THE SHARED TRUNK.** This is the
   design question, not the plumbing one:
   * **if yes** — the tactical loss shapes the trunk, and the trunk is then jointly optimised by the
     planner, both perception heads **and** the tactical decoder. ⚠️ Attribution gets harder: a
     trunk improvement can no longer be assigned to one head, which is the `--v2` conflation failure
     the programme has already paid for once.
   * **if no** — the decoder reads a **detached** BEV, learns from the map, and cannot corrupt the
     trunk. ⚠️ But then the map representation it reads is shaped only by the perception losses, and
     "the tactical layer learned from the map" means "learned from a map it had no say in".

**Default if silent: neither.** ⛔ The decoder ships **agent-only**, the refusal stays, and every
tactical number is reported with `sources: ["agent"]` attached. That is honest and it is testable —
it simply does not answer the map half of your question. ⚠️ I am **not** defaulting to option 1,
because doing the plumbing without the ruling would land a model whose attribution properties you
had not chosen.

*Evidence: `…/2026-09-17-refcv6-tactical-training/RESULT.md`;
`PREREG_REFCV6_V2.ERRATUM-1.md` §E8.*

---


## ⭐⭐⭐ PI RULING 2026-09-17 — ITEMS 15 AND 18 ARE CLOSED

**Sayed, verbatim:** *"You can take 408x1024, yes use also the map for tactical behavior decoding and you can backpropagate to the trunk."*

| item | ruling |
|---|---|
| **15 — the vertical field** | ⭐ **408 × 1024.** The field is kept (VFOV 45.296°, nearest road **3.15 m**) and the **1.88 m blind strip is not taken**. Costs **+112.9 GB** of corpus cache (386.5 vs 273.6) and **1.63×** the tokens at both strides. ⛔ The **HF quota must be checked BEFORE the rebuild is pushed.** |
| **18 — the tactical layer's map half** | ⭐ **Unblocked by instruction, both halves.** The BEV encoder moves INTO the model forward, and the behaviour decoder **may backprop into the shared trunk**. ⚠️ Attribution is now four-way (planner + map + box + tactical): the per-head gradient-reach report and the **conflict detector must be ON for every arm**. ⚠️ Every bit-identity proof taken before this is about a **different forward** and must be re-run. |

⇒ the two remaining open items are **16** (the conflict statistic, default stands), **17** (clip-id leakage) and **§12 refusal 10** — see the full text above.

<!-- PI-RULING-2026-09-17 -->

## ⭐ PI RULING 2026-09-17 (second) — ITEM 17 IS CLOSED

**Sayed, asked and answered:** *"Its hygiene and reproducibility."*

⇒ **option 2.** The banked record is **left alone**; a guard stops the count growing. Redacting 53 files of evidence would have bought **appearance only** — git history keeps every one of them regardless — and rewriting an identifier inside a sentence a past conclusion rests on is not housekeeping.

`tools/clipid_scan.py` + `tools/clipid_baseline.json` + its tests. ⛔ **The baseline stores COUNTS, never identifiers** — an allowlist of the leaked ids would itself be a list of clip ids, i.e. the guard's own artifact would be the thing the guard exists to stop growing. Four refusals, **4 of 4 killed** by the guard-removal audit, 10 tests.

⭐ **The scope is broader than clip ids, deliberately.** Under a hygiene argument the defect is that **a bare UUID is an unstable handle**; `sha12` is a hash and is the identity every GT file already stores. So the always-runnable half needs **no corpus list** and cannot rot when one moves. ⚠️ Both figures are true and answer different questions: **64** corpus-matched clip UUIDs across 53 files, and **120** UUIDs of any kind across 27 files under the same globs. The 8-char prefix half (530) needs a clip list and is **opt-in**.

⇒ the only item still open is **16** (the conflict statistic), whose default stands.

<!-- PI-RULING-2-2026-09-17 -->
## Not a decision — the state, for orientation

⚠️ **CORRECTED 2026-09-17 — this paragraph described a run that has since FINISHED, and said so
in the present tense long after it landed.** `refcv5-v2` is **COMPLETE at step 40,284**, evaluated
2026-09-09 at two inference seeds, and it **FAILED its pre-registered bar `BAR-REFCV5V2-1`**: it
does not beat the echo control `ha0_ext` — it is **separated WORSE** — and it loses to `refcv4b` on
ADE by **+0.0114 m, separated**. `MODEL_REGISTRY.md` §4.7 **on this same branch** carried the
settled outcome while this file still said "is training", so a reader orienting from here was
being told the opposite of the record. ⛔ Read §4.7, not this paragraph, and note its own warning
that the +0.0114 m is **NOT attributable to WP-4** — two mechanism groups moved together.

*What the paragraph said when it was written, kept so the correction is legible:* **refcv5-v2 is
training** on the A40 (PID 2560646, watchdog 2561632), step ~5,400 of 40,284 at
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

---

## ⭐⭐ NEW ITEM 19 (2026-09-17) — the AGENT SEAM now has a PRICE, on a ledger the decision that closed it never touched

**This needs no GPU to state and it is not a re-litigation.** The agent channel was gated off
because the **auxiliary agent task** cost accuracy at two seeds on the tiny rig. That is a finding
about a **training task**. What follows is a finding about a **selection constraint**, measured
today at zero GPU on 493 held-out windows, and the two do not meet.

### What was measured

| | |
|---|---|
| windows where the **selected** plan collides (untouched `refcv5-v2`) | **28 / 493** |
| …of which the fan **still held a collision-free candidate** | **28 / 28 = 100 %** |
| mean share of the fan that was collision-free there | **55.3 %** |
| **oracle-repair ceiling from fixing collision-selection ALONE** | **+0.0485 `sel_pdms` [+0.0228, +0.0801] = 62.4 % of the entire oracle gap** |
| windows it covers | **25 / 493 = 5.1 %** |

⭐ **Five per cent of windows carry sixty-two per cent of the recoverable PDMS**, and the
intervention is a **hard constraint**, not a learned weight: *never select a colliding candidate
while a collision-free one is in the fan*.

⭐ It **reproduces on four checkpoints in three training states** (`base` 28/28, `base_repeat`
28/28, `L1-NORL-s0` 25/25, `L1-RL-s0` 37/37), so it is **not** an artifact of the RL experiment
that surfaced it — it is a **refcv5-v2 property**.

### Why it is an AGENT-SEAM item and not a re-ranking item

⛔ **These arms have no agent input at all** — verified three ways: `argv` carries `--agents off`,
the run directory is `refcv5-v2-noagents-b1-v72-40k`, and `agent_join` / `agent_join_digest` /
`agent_join_stats` are all `None`. ⛔ **And the REWARD does see the agents**: the proxy scores
collision and TTC against tracks from `b1eval_agents.jsonl.xz`, and **172 windows were dropped for
missing agent data**.

⇒ **the model is graded on a constraint it has no structured channel for.** It is not blind — the
trunk sees the image, so the vehicles are in the **pixels** — but nothing converts them into a
collision-relevant representation. That is the same defect **D3** measured from the other side the
same night: the planner **attends** to the lead, and greying the lead out moves the time gap by
**0.09× the noise floor**.

### ⛔ What this is NOT

* **NOT** "recover a lever we know is there." Re-opening `--agents on` is an **ARM whose tiny-rig
  exclusion may or may not generalise**, exactly as the WP-C framing was corrected to say.
* **NOT** a claim the auxiliary task became free. Nothing here touches that measurement.
* **NOT** a free re-ranking. `sel_nc` is scored against the **RECORDED future** (T0), so the 62.4 %
  is what a **perfect collision checker** would have bought. At inference the vision-only rule
  forbids the recorded future ⇒ a gate needs a **predicted** occupancy.

### The decision, with its default

**Your call is whether the refcv6 arm matrix carries an agent-channel arm.** The seam already
exists — `refc_agents.slot_features` passes continuous metric range and bearing, gated behind
`--agents off`.

| option | |
|---|---|
| **(a) DEFAULT — carry it as a named arm** | add `--agents on` as one arm in the refcv6 matrix, pre-registered against the collision statistic above rather than against ADE. Costs one arm. ⭐ It is also the organ the **BEV map head** you directed is being wired for, so the two are the same bet from two directions. |
| (b) hold it until the perception heads report | spend nothing now; re-read this item when the BEV/box3d wiring lands and a predicted occupancy exists to gate on. |
| (c) decline | the 62.4 % stays on the table as measured headroom nobody is reaching for. |

⚠️ **If you say nothing, (a) is what happens** — as a *named arm carrying its own pre-registration*,
never as a claim that the channel works.

<!-- PIQ-ITEM-19-AGENT-SEAM-PRICED-2026-09-17 -->

---

## ⛔⛔ NEW ITEM 20 (2026-09-17) — **408 × 1024 CANNOT BE BUILT.** The trunk refuses it, and the geometry you authorised needs replacing

**You authorised this geometry in session on 2026-09-17:** *"You can take 408x1024, yes use also the
map for tactical behavior decoding and you can backpropagate to the trunk."* ⛔ **The first half is
not buildable.** This item is only about the geometry; the map/trunk half is built and measured.

### The refusal, verified by construction rather than by reading

`stack/tanitad/models/timm_trunk.py:216` — an explicit guard, not an accident:

```
if h % 32 or w % 32:
    raise ValueError("refcv6 trunk: image {h}x{w} — each axis must divide by 32 or "
                     "the stride-32 map is silently mis-sized ...")
```

Constructed each candidate directly:

| geometry | `h % 32` | result |
|---|---|---|
| 256 × 640 (today) | 0 | ✅ accepted |
| **256 × 1024** | **0** | ✅ **accepted** |
| **408 × 1024** | **24** | ⛔ **REFUSED** |
| **416 × 1024** | **0** | ✅ **accepted** |

⭐ **And the guard is right.** At `h = 408` the trunk would declare `h // 32 = 12` rows while a
stride-32 CNN emits `⌈408/32⌉ = 13` — the guard's own comment names it *"the 2026-07-27 single-axis
check defect, per axis this time"*. ⚠️ The refusal is at **stride 32**, not stride 16 as the
implementing agent reported (it wrote *"408 % 32 == 8"*; the true values are **408 % 32 = 24** and
408 % 16 = 8, so its modulus was wrong and its conclusion right).

### ⚠️ This was already half-known, and saying so matters

`…/Data Engineering/Research/2026-09-16-256x1024-cache/RESULT.md` §*"One honest wrinkle"* already
recorded that an exact 1.6× supersample needs height **409.6**, that **H = 408 lands half a pixel
off**, and that **H = 416** (VFOV **46.0921°**, offset −2 rows) is *"integer-aligned AND slightly
exceeds today's field, at ~2 % more disk than 408 — arguably the cleanest choice, but it was **not
built**"*. ⇒ **the wrinkle that package flagged as cosmetic is a hard blocker one layer down.**
⭐ Two independent sources agree on 416's VFOV (46.0921° there, 46.09° from the implementing agent).

### What already exists

**Both** 256 × 1024 and 408 × 1024 caches are **built and validated on the 139 B1 EVAL clips**
(`MANIFEST_eval139_408x1024.json`). ⇒ the 408 cache is real work that **cannot feed the refcv6
trunk**. 416 has never been built.

### The decision, with its default

| option | |
|---|---|
| **(a) DEFAULT — `416 × 1024`** | the **lose-nothing** shape: keeps ~100 % of today's VFOV *and* 1.6× finer azimuth, integer-aligned to the 640 row grid, legal at both strides. ⛔ **Cost: the eval-139 cache must be rebuilt** (~2 % more disk than the 408 one, which is then dead), and the corpus rebuild after SAM3 finishes uses 416. |
| **(b) `256 × 1024`** | **already built and legal today** — zero rebuild. Paper-matching (DiffusionDrive/NAVSIM is 1024 × 256). ⚠️ Buys azimuth by **spending elevation**: 64.5 % of today's VFOV. |
| (c) `400 × 1024` | also integer-aligned (VFOV 44.4952°, offset +3 rows) but **below** today's field — strictly worse than (a) on your own criterion. |

⚠️ **If you say nothing, (a) is what happens** — 416 × 1024, because it is the only legal geometry
that satisfies the intent of "take 408×1024" (lose nothing, gain azimuth). ⭐ **(b) is the cheap
answer if disk or schedule is the binding constraint**, and it is already on disk today.

⛔ **Do not read this as a reason to relax the guard.** It exists because a silently mis-sized
stride-32 map is the failure it names, and the 2026-07-27 defect it cites is in the retraction log.

<!-- PIQ-ITEM-20-408-UNBUILDABLE-2026-09-17 -->

---

## ⭐⭐ NEW ITEM 21 (2026-09-17) — **D9 (RL post-training) as configured is REFUTED at T1.** The evidence now orders the refcv6 work differently

You asked for *"prepare the rl post training, implement D9"*. It is implemented, it ran, and it has a
pre-registered verdict. **This item is the consequence, not a request for permission.**

### What D9's own lever measured

| | |
|---|---|
| `H-DDV2RL-2` (the pre-registered T1 harm guard) | ⛔ **FAIL-HARM**, both seeds |
| T1 `ade_m` vs the **untouched cold start** | **0.2994 → 0.5502** and **0.5561** |
| against its own two-seed floor (0.0059 m) | **43×** |
| against the measured inference floor (≈0.0001 m) | **2,538×** |
| versus the harm it was built to repair (2026-09-15, +0.083 m) | **≈3× worse** |

⛔ **The lever makes the car drive worse than not training it at all**, on the primary tier, robustly
on both variances.

### ⚠️ What this does NOT say — please read this before it is quoted

* ⛔ **It does not refute RL post-training.** It refutes **one configuration**: GRPO with the
  matched-anchor IL term, 600 steps, on the 9.47 M-parameter decoder subset, with a **proxy reward
  that has no drivable-area term**. A different reward, budget or parameter set is a different arm.
* ⚠️ **The T0 endpoint PASSED** (§13.2 SUCCESS, `Δfan` +0.0363 and +0.0616). T0 is a diagnostic tier
  over a **recorded future**; the two results are not in conflict, and this package is the clearest
  demonstration the programme has of why a T0 number may never be quoted as driving performance.
* ⭐ **Amendment A-1 and the matched-anchor form are VINDICATED** on what they were for: F1 passes on
  all three arms (73.7–94.2 % of the fan retained against a 60 % bar, where the release form kept
  **6.52 %**), and they **stabilised the rig ~10×** (a zero-lever replicate separates on 4/10 metrics
  against the release rig's 10/10).

### What the same night's evidence says to do instead

Three instruments at three tiers converged on one diagnosis — **`refcv5-v2` has a USE problem, not
an information problem**:

| | the information IS there | and is NOT used |
|---|---|---|
| **D3** (attention, T1) | attends to the lead, **1.92×**, all four decoder layers | masking it moves **0.09×** the floor (bar ≥ 3×) |
| **collision** (T0) | a collision-free plan is in the fan in **119 of 119** windows across five checkpoints | it selects a colliding one every time |
| **nav** (T1) | nav rules a turn **out with certainty** — `P(turn \| NAV_FOLLOW_ROAD)` = **0.0000** | deleting nav moves **0 of 10** metrics |

⇒ **adding inputs is measurably not the lever; forcing the existing reads into the decision is.**
And the one such lever that is priced: **a collision gate on selection is worth 62.4 % of the oracle
gap from 5.1 % of windows** (item 19), blocked only on a predicted occupancy — which is exactly what
the BEV map head you directed is for, and which is now wired (R2/R3, opening **3,146,752** trunk
parameters to the tactical layer).

### The decision, with its default

**Ordering, not cancellation.**

| option | |
|---|---|
| **(a) DEFAULT — perception and selection FIRST, D9 parked** | spend the next refcv6 arms on the BEV/occupancy path and a collision gate; re-open D9 **after** selection has something to select with. D9's instrument, pre-registration and harness all survive and are banked. |
| (b) re-run D9 with a changed reward | the proxy has **no drivable-area term** (our maps did not cover the training clips when it was built; SAM3 finishes ≈2026-09-22). A DAC-carrying reward is a genuinely different arm — but it is a **new** pre-registration, not a re-run. |
| (c) proceed with D9 as specified | ⛔ not recommended: the measured effect is a large, robust regression on the tier that decides capability claims. |

⚠️ **If you say nothing, (a) happens.** ⛔ Nothing is deleted — D9 stays implemented, banked and
re-runnable the moment a reward or a scale argument justifies it.

<!-- PIQ-ITEM-21-D9-REFUTED-REORDER-2026-09-17 -->

## ⭐⭐ ANSWERED 2026-09-17 — the PI ruled on all four open items in session

**Verbatim, in two messages:** *"for 1, chose (b), for 2 move to 416X1024, for 3 chose
defult, what is D9?"* and, after D9 was explained, *"dont park D9, solveit and prove it."*

⛔ **Recorded here because a decision that lives only in a chat transcript is not a
decision the programme can act on later.** Each row states what was asked, what was
answered, and what has ALREADY been done about it.

| # | the question | the PI's answer | what was done, same day |
|---|---|---|---|
| **NEW** | `resnet101` — §10.2's PRIMARY trunk — OOMs on the 8 GB dev box. How does it get its pre-pod gate? | **(b) local batch-1 proof** — shapes and wiring, not memory | ⭐ **DONE, with a harder answer than the option assumed.** It OOMs at 416×1024 **batch 1** too (*"tried to allocate 156.00 MiB … 22.34 GiB is allocated by PyTorch"*), so the GPU route to (b) is closed at **either** geometry. Delivered on **CPU** instead: `{"done": true, "step": 5, "wallclock_s": 382.9}` with every head live and rig coverage **139/139**. ⇒ **shapes correct; the card is the only obstacle**, and the >20 GB figure is pod-sizing information. |
| **20** | input geometry — `408×1024` is unbuildable (`408 % 32 = 24`) | **move to 416 × 1024** | ⭐ **DONE.** Cache rebuilt 139/139, 0 failures, 10.75 GB, 18.8 min; HFOV exactly 120.0000°, VFOV 46.0921°, `416 % 32 = 0` ⇒ 13 stride-32 rows. ⛔ **Building was not enough** — the first run was refused for an undeclared `CanonicalFrame`, the same omission that blocked 256×1024 the day before. Declared derived-not-retyped + 7 mutation-audited tests (`d8306f1`). |
| **19** | the agent/selection seam — priced at **62.4 %** of the oracle gap from **5.1 %** of windows | **default** — carry it as a named arm with its own pre-registration | recorded; its pre-registration follows the D9 one. Nothing about it changes the D9 ordering below. |
| **21** | D9 (RL post-training) is refuted at T1 — park it, re-reward it, or proceed? | ⭐⭐ **"dont park D9, solve it and prove it"** — i.e. **NOT option (a)**; a repaired reward, with proof | ⭐ **DONE (diagnosis) + PRE-REGISTERED (repair).** Two named defects, one per seed, each separated: **DAC ≡ 1** (seed 0, off-road +0.0151 SEPARATED) and **EP with no speed-appropriateness term** (seed 1, +0.372 m/s SEPARATED). `H-DDV2RL-3` written with both outcomes committed (`ce8e892`). |

### ⛔ The one thing item 21's answer does NOT unblock

`H-DDV2RL-3` **must not start** until SAM3 maps cover the **RL-train** split (corpus
finishes ≈2026-09-22). A partially covered split makes `dac_from_drivable` return its
no-map value of **1** on the uncovered part — which is *precisely the defect being
repaired*, reintroduced silently. The pre-registration states this as a gate, not a
preference.

<!-- PIQ-ANSWERED-19-20-21-RESNET101-2026-09-17 -->

## ⭐⭐ NEW ITEMS 22–24 (2026-09-18) — the backlog's "blocked on the PI" rows, finally ASKED

⛔ **Why these appear only now.** `BACKLOG.md` carried rows reading *"needs the PI"*,
*"NOT AUTHORISED without the PI"*, *"decision pending"* — and **not one of them was in this
file**. Six probed by name returned **0 hits** each, against a positive control (`ITEM 21`)
reading 1. ⇒ they were **not gated, they were UNASKED**: a row saying *"blocked on the PI"*
is skipped by every sweep while the question never reaches the PI. That is worse than an
open decision, because it looks like one.

⚠️ **The count was wrong too, and smaller than reported.** A FlyWheel sweep said *"11 of
11"*. Re-derived here: **10 lines** mention a PI gate, of which **3 are rule statements**
(not items) and **2 (`R4`, `R5`) are already marked DONE/DIAGNOSED** — their gate is
historical. Of the **8** real rows, **4 are MOOT** and **1 appears already answered**
(§B below). **3 are genuinely live**, and only those are asked.

---

### ⭐ ITEM 22 — `A3` / C64 option B: freeze the clean v2-line val at **n = 400**, or reject B?

**MEASURED, and the interesting part is what it cost to find out.** The clean val was
BUILT (2026-08-02) and its column semantics are a machine-checked contract (34/34 on
18,988 rows). Then three findings landed against it:

* the **6.77× advantage was ONE AXIS** — on junction+turn+speed it is **1.07×**, and
  +brake it is **0.77 = infeasible**;
* **cell-quota matching does not balance** (max |d| **0.3997**, 10/13 axes over bar);
  greedy covariate balancing reaches **0.0094**;
* ⛔ **a clean v2 val is NOT clean for v1**: **62 of a 600-draw are inside v1's TRAIN**
  ⇒ excluding the parity corpus makes **600 unavailable** (headroom 0.95).

**Shipped: n = 400**, max |d| **0.0409**, sha256 `abe041db72a045b3…` (an n = 300 variant
exists). ⚠️ No draw from this remainder is exchangeable with train (within-cell median
|d| **0.359**).

| option | |
|---|---|
| **(a) DEFAULT — freeze n = 400** | take the balanced 400 as the clean v2-line val and state its non-exchangeability wherever it is quoted. It is built, balanced and hashed; the alternative is no clean val at all. |
| (b) take n = 300 | tighter balance, less power. Only if 400's |d| 0.0409 is judged too loose. |
| (c) reject option B | no clean v2-line val; cross-line comparisons keep carrying the leak caveat instead. |

⚠️ **If you say nothing, (a) happens** — and the non-exchangeability caveat travels with
every number drawn from it.

---

### ⭐ ITEM 23 — `R54` / D3b: which margin **leads** the H-vs-F table?

A reporting convention, **zero compute**, and it changes what the programme appears to
claim.

The choice is between the **fed-nav** margin and the **deployment (nav-zero)** margin.
⛔ **Nav is an oracle input that will not exist at deployment**, so leading with the
fed-nav margin **overstates the system**.

| option | |
|---|---|
| **(a) DEFAULT — lead with the DEPLOYMENT margin** | report the nav-zero margin as the headline and the fed-nav one beside it. The headline then describes what ships. |
| (b) lead with fed-nav | only defensible if nav is guaranteed at deployment — which contradicts the vision-only rule. |

⚠️ **If you say nothing, (a) happens.**

---

### ⭐ ITEM 24 — `R24`: authorise (or refuse) the **DINOv3 ViT-B/16** pull?

Only **ViT-L/16** and **dinov2-base** are on the box; the converter never downloads. This
blocks the **v7f seed at the pre-registration's chosen geometry**.

⚠️ **Scope honestly:** v7f is **not** the refcv6 line, and refcv6 is where the programme's
attention currently is. This is a small download, not compute — but the standing rule is
that **downloads need their own permission**.

| option | |
|---|---|
| **(a) DEFAULT — authorise the pull** | it is a model download inside the research-licence class, it unblocks a pre-registered seed, and it costs no GPU. |
| (b) refuse | the v7f seed runs at a geometry its own pre-registration did not choose, and that must then be stated in its results. |
| (c) defer until refcv6's panel is decided | avoids splitting attention; costs nothing but time. |

⚠️ **If you say nothing, (c) happens** — deferral, not authorisation, because nothing in
refcv6's critical path needs it this week.

---

## §B — the five that should be RETIRED rather than asked

⛔ **Drafting these as live decisions would be asking the PI to rule on infrastructure that
no longer exists.**

| row | why it is not a live decision |
|---|---|
| **`D1`** — how many **A40-hours** for S-W | ⛔ **MOOT.** *"The A40 is stopped and gone"* (`PI_DECISION_QUEUE.md:172`), *"There is no A40 — the pod was stopped and is gone"* (`PREREG_REFCV6.md:6`, `:456`). |
| **`D2`** — **which pod** runs S-W | ⛔ **MOOT**, same evidence. |
| **`C4`** — release the **old CPU pod** | ⛔ **MOOT** — the fleet it belongs to is gone (MEASURED 2026-09-10 on `pod3`/`pod4`/`pod5`). |
| **`C5`** — **X2 verdict run, 30 pod-days** | ⛔ **MOOT** — 30 pod-days of a fleet that does not exist. If X2 is still wanted it is a **new** item priced against today's compute. |
| **`C6`** — wheelbase fix | ⚠️ **APPEARS ALREADY ANSWERED.** The PI chose *"C = measure first"*; the measurement landed. `H-ECHO-6` is **SUPPORTED** — `curvature = tan(steer)/2.9` inverts the corpus encoding exactly, cross-checked against `d/dt(unwrap(yaw))` at **r = +0.9430**, where *"a wheelbase error would read about 0"* — and `GOALS_AND_CLAIMS` records *"exact and the wheelbase cancels, so no re-cache and no parity break"*. ⇒ **verify and strike**, do not ask. |

⚠️ **The lesson under all five is the same one `BACKLOG.md` diagnosed about itself:** a
blocker note is not revisited when the thing it blocks on changes, so it keeps reading as a
live gap. Four of these became moot when the pods went; one when a measurement landed.
**None of them was ever asked, so none of them was ever closed.**

<!-- PIQ-ITEMS-22-24-BACKLOG-PI-ROWS-ASKED-2026-09-18 -->

## ⛔ CORRECTION TO ITEM 23, 2026-09-18 — the PI reversed its premise, and the default flips

**PI, verbatim:** *"nav is an input which will exist at deployment."*

⛔ **Item 23 above is WRONG and its default is withdrawn.** I wrote that nav is *"an
oracle input that will not exist at deployment"* and defaulted to leading with the
**nav-zero** margin on that basis.

⚠️ **The premise came from `BACKLOG.md` row R54 and I propagated it without checking it
against the spec.** It contradicts the PI's own directive of **2026-09-16**, recorded in
`SPEC_REFCV6_V2.md` §0: *"Confirm using nav command as mandatory input for tactical and
operative planning. The selection of the tactical plan and the planing and selection of the
trajectory must use the nav command and follow it. We dont need any head to estimate the
route."* ⇒ nav is a **mandatory deployed input**, supplied by the vehicle's navigation, not
an oracle borrowed from the future.

### The corrected decision

| option | |
|---|---|
| **(a) NEW DEFAULT — lead with the FED-NAV margin** | it is the **deployed configuration**. Report the nav-zero margin beside it as an **ablation** — *"what the system is worth without its nav input"* — which is what that number actually measures. |
| (b) lead with nav-zero | ⛔ withdrawn. It would describe a configuration the programme does not ship and the PI has ruled mandatory. |

⚠️ **And the ablation keeps its teeth.** Leading with fed-nav does **not** retire the
nav-ablation reads — `D-REFCV5V2-NAV` measured that deleting nav moves **0 of 10** metrics,
and `P(turn | NAV_FOLLOW_ROAD) = 0.0000` says nav rules a turn out with certainty. A
mandatory input the model does not USE is a finding, and it is a different finding from
*"the input will not be there"*. The first is ours to fix; the second was never true.

### ⛔ The class, because it is the third time tonight

**A steering file's PREMISE quoted forward without re-deriving it against the spec.** The
row was written before the 09-16 directive and never revisited; I read it, found it
plausible, and drafted a decision on it. ⇒ **When a backlog row states a fact about the
system — not a task — check it against the binding spec before building on it.** The same
class as `R4`/`R5` reading as PI-gated when their gates had closed, and as the four pod
rows surviving the fleet they depended on.

<!-- PIQ-ITEM-23-CORRECTED-NAV-IS-DEPLOYED-2026-09-18 -->

## ⭐ CLARIFICATION TO ITEM 24, 2026-09-18 — it is an **initialisation**, not an encoder

**PI, verbatim:** *"does v7f need DINOv3 VIT encoder? Do you mean init by DINO? v7f is our
world model based predictor multihierarchacy flagship."*

✅ **The PI's reading is correct and my Item 24 wording was loose.** It said the pull *"blocks
the v7f seed at the pre-registration's chosen geometry"* — true, but *"the DINOv3 ViT-B/16
pull"* reads like v7f depends on a DINOv3 encoder. It does not.

### What v7f actually does with DINOv3 (MEASURED, from source)

| | |
|---|---|
| **the trunk** | our own `ViTEncoder` / `ViT5Encoder`, **TRAINABLE** — `PREREG_V7F.md:108`: *"DINOv3 ViT-B/16, **trainable**, with a discriminative learning rate and a distillation anchor"* |
| **DINOv3's role** | **the starting weights, and nothing else.** `stack/scripts/dinov3_seed_checkpoint.py:4`: *"The PI directed that v7 trains its trunk **from a DINOv3 initialisation**."* |
| **at inference** | ⛔ **no frozen DINOv3 in v7f's forward pass.** |
| **the three OTHER roles** | frozen DINOv3 IS used elsewhere in the programme — O7's distillation teacher, REF-A's precomputed feature bank, the fp8 shipper — which is exactly why the bare phrase *"v7f needs DINOv3"* is ambiguous. |

### ⚠️ The seed is NOT bit-identical DINOv3 — two declared losses, printed by every run

1. **POSITIONAL INFORMATION DOES NOT TRANSFER.** DINOv3 is RoPE-only and carries **no**
   learned absolute-position table (verified on the ViT-L/16 snapshot: **415 tensors, none
   positional**); `ViTEncoder` is the mirror image — a learned `pos` table, no RoPE. `pos`
   is LEFT AT ITS OWN INIT. ⇒ **a seeded trunk is DINOv3's *content* at a *fresh* positional
   code**, and that is the converter's own phrasing, not a gloss.
2. **CLS AND REGISTER TOKENS ARE DROPPED** — `ViTEncoder` has neither.

⭐ The converter REFUSES rather than silently seeding a partial trunk: every source tensor
mapped or explicitly allow-listed, every target tensor written or explicitly left-at-init,
shape and geometry disagreements named. *(The failure it is built against — a 60 %-seeded
trunk that looks exactly like a success in every log — is the `df`/`step_s` family in a
checkpoint costume.)*

### Why **B/16** when **L/16 is already on the box**

⛔ **Parameter budget, and it is not close.** B/16 is **~86 M** (`PREREG_V7F.md:450`);
DINOv3 **ViT-L/16 is 303,129,600 parameters of ENCODER ALONE** (`TANITAD_PAPER.md:3728`),
which busts the programme's sub-300M whole-model budget on the trunk by itself. `PREREG_V7F`
§10 D1 chose B/16, and §R3 specifies the rung run **the same B/16 trunk v7f will deploy**
rather than a cheaper stand-in.

### What is actually blocked, precisely

⚠️ **Only the FILE.** The converter is wired and tested (`R15 — STRUCK 2026-09-03`) and
**never downloads**. Missing: `facebook/dinov3-vitb16-pretrain-lvd1689m`. Only ViT-L/16 and
dinov2-base are on the box.
⚠️ **And R23 needs the same file** — `--w-trunk-anchor` runs a SECOND frozen forward of the
seed's own ViT-B/16 at the loss site, so refusing the pull blocks the anchored arm too, not
just the seed.

⚠️ **The default stated in Item 24 is UNCHANGED — (c) defer** — and the reason is unchanged
too: nothing in **refcv6's** critical path needs it this week, and refcv6 is where the
programme's attention is. This clarification changes what the decision is ABOUT, not what
happens if you say nothing.

<!-- PIQ-ITEM-24-CLARIFIED-DINO-IS-AN-INIT-2026-09-18 -->

## ✅ ITEM 24 — RESOLVED BY THE PI, 2026-09-18, AND EXECUTED THE SAME TURN

**PI, verbatim:** *"you can downloa the dino v3 vitb16"* ⇒ option **(a) authorise the pull**.
The standing default was (c) defer; it is superseded.

### What was done, end to end

| step | result |
|---|---|
| download `facebook/dinov3-vitb16-pretrain-lvd1689m` | ✅ **342.66 MB**, 3 files. The licence was already accepted on the account — no human gate was hit. |
| token handling | read IN PLACE from the git-ignored `Keys.txt`, never printed, never on a command line, and redacted out of any error string. |
| TLS | `truststore.inject_into_ssl()` — `certifi` fails behind this box's proxy. |
| convert to an encoder seed | ✅ `dinov3_seed_checkpoint.py --target vit` → **148 tensors, 342.7 MB**. |
| provenance | `source_sha256=9a21ac3df0c63839…`, `seed_sha256=c00cdd8ff3d83272…`, full mapping table stored IN the seed's stamp. |

⭐ **The converter did NOT refuse, and that is the point.** It refuses unless EVERY source
tensor is mapped or explicitly allow-listed AND every target tensor is written or on the
left-at-init list — built against the failure where a load puts 60 % of a trunk in place,
leaves the rest at random init, and looks exactly like success in every log.

### ⚠️ The two declared losses travel with every quotation of this seed

1. **POSITIONAL INFORMATION DOES NOT TRANSFER.** DINOv3 is RoPE-only and carries **no**
   learned absolute-position table; `ViTEncoder`'s `pos` is **left at its own init**.
   ⇒ the seeded trunk is **DINOv3's *content* at a *fresh* positional code**.
2. **CLS / register / mask tokens are dropped** — `ViTEncoder` has none, and DINOv3 was
   trained with those tokens attending, so the seeded forward is **not bit-identical** to
   the published trunk.
⇒ the Observer-Effect step-0 control (`PREREG_V7F` §6.2b) measures **THIS** trunk, not the
published DINOv3. Any report quoting the published rho 0.91 beside this seed must say so.

### What it unblocks

* the **v7f seed at the pre-registration's chosen geometry** (B/16 ≈ 86 M; L/16 is
  **303,129,600 params of encoder alone** and busts the sub-300M budget on the trunk).
* **`R23` / rung R3's `anchored` arm** — `--w-trunk-anchor` runs a SECOND frozen forward of
  the seed's own ViT-B/16, so it needed the same file and was blocked behind this.

⚠️ **Still true and unchanged:** v7f is **not** the refcv6 line, and the PI's 2026-09-18
ruling is that **the pod is used only after the preparation is proven on the dev box**. This
authorisation buys a file, not a launch — nothing trains on it without a separate decision.

⭐ Artifact: `D:/Projects/TanitAD-artifacts/dinov3-seeds/dinov3_vitb16_seed.pt` (342.7 MB,
out of repo by size); its **stamp** is banked at
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-18-dinov3-b16-seed/raw/`.

<!-- PIQ-ITEM-24-RESOLVED-AUTHORISED-AND-EXECUTED-2026-09-18 -->

## ⛔⛔ ITEM 25 (2026-09-18) — AS WRITTEN, THE refcv6 PANEL CAN NEVER EMIT `SUCCESS`

**MEASURED, and verified by me directly rather than taken from the agent that found it.**

`verdict_refcv6.py:121` carries
`Clause("S1", "STRATEGIC", "route_acc not separably worse than control, and n > 0")`,
and `:39` states the rule plainly: *"`n` must be > 0. ⛔ `n = 0` is `MISSING_DATA`, never
a pass"*. Absence deliberately cannot pass — that design is CORRECT and is not the problem.

⛔ **The problem is that refcv6 turns the strategic layer OFF by the PI's own directive.**
`SPEC_REFCV6_V2.md` §0/§1: *"Deactivated for this experiment: the whole strategic layer…
No head estimates the route (PI)"*, and the 2026-09-16 ruling is that nav is a **supplied
input**, not something a head estimates: *"We dont need any head to estimate the route."*
⇒ `route_acc` has no data by construction ⇒ `n = 0` ⇒ `MISSING_DATA` ⇒ **S1 can never
be satisfied, so the panel can never reach `SUCCESS` no matter how well refcv6 drives.**

⚠️ **This is not a bug in either document.** The verdict script predates the amendment
that switched the strategic layer off. Two correct decisions, taken weeks apart, that
contradict each other — the same shape as the `Research Hub` rename silently blinding
three guards tonight.

| option | |
|---|---|
| **(a) DEFAULT — score `strat.nav_compliance` in S1's place** | it is the only strategic metric that is COMPUTABLE with the layer off and nav supplied: does the trajectory FOLLOW the nav command it was given? MEASURED: **0 present / 3 refused / 38 missing across all 41 banked artifacts**, so it is owed regardless. This keeps S1 a real clause rather than deleting a family. |
| (b) mark S1 `n/a-by-design` and require the other clauses | honest, but it removes the STRATEGIC family from the verdict entirely, which the four-family directive forbids ("a missing metric is a work item, not an excuse"). |
| (c) re-enable a route head for refcv6 | ⛔ contradicts the 2026-09-16 directive. Listed only so the option set is complete. |

⚠️ **If you say nothing, (a) happens** — and `strat.nav_compliance` gets built, because it
is owed under the four-family rule whichever way S1 is resolved.

<!-- PIQ-ITEM-25-S1-ROUTE-ACC-BLOCKS-SUCCESS-2026-09-18 -->

<!-- ITEM-26-2026-09-19 -->
## ⛔ ITEM 26 (2026-09-19) — a 455.4 GB PUBLIC HF dataset whose file names carry raw clip ids, and a storage ceiling that bills instead of refusing

**MEASURED by the DataFlyWheel** (raw JSON in the package; the Master Mind scanned and landed it but did NOT re-measure the HF numbers) (`TanitAD Research Lab/Data
Engineering/Research/2026-09-19-c4-hf-quota-check/`):
- `Sayood/tanitad-physicalai-w120-256x640cyl` holds the parity caches, is **455.4 GB**, is
  **PUBLIC**, and **3,000 of its 6,061 file names carry raw clip ids** — against two standing
  rules: augmented datasets stay PRIVATE, and clip ids appear only as sha12.
- Private storage today **176.554 GB** of the PRO **1 TB**. The account is `canPay: true`,
  prepaid: **over 1 TB is BILLED ($18/TB/mo), never refused.**
- If that repo goes private AND the 386.5 GB 408×1024 cache is pushed privately: **1,018.4 GB,
  18.4 GB over the ceiling, billed.** Pushing the 256×1024 cache instead: 905.5 GB, fits.
  `periodEnd` now reads 2026-10-01.

**Options:** (a) make the repo PRIVATE (+455.4 GB private) and push only what fits under 1 TB;
(b) re-publish it with sha12 file names, then decide visibility; (c) leave it as is.
**⭐ DEFAULT if unanswered: (c) status quo for the repo, AND no private push of the 386.5 GB cache
until this item is ruled** — the push is the only action that could silently start billing.

<!-- PI-RULINGS-2026-09-19 -->
## ⭐⭐ PI RULINGS 2026-09-19 — items 25, 22, 23, 26 (+4) and the pod, answered in session

**Sayed, verbatim:** *"1. again the strategic layer will be only switched off temporarily until we
proved that both tactical and operative layers are driving with high quality. So the 4 layer rule
is just temporarily paused and concerns only the strategic part. The nav command is an input at
both training and inference as I stated many times · 2. a · 3. default, again nav is an input both
in training and inference · 4. leave it as it is and make it with gated manual approval for access ·
5. we will do it later"*

| item | ruling | what was done, same turn |
|---|---|---|
| **25** — the refcv6 panel could never emit SUCCESS | ⏸ the **STRATEGIC family is PAUSED, temporarily** — not removed — until the tactical AND operative layers are proven to drive with high quality. The four-family rule is paused **for the strategic part only**. Nav is an **input at training AND inference**. | `verdict_refcv6.py`: S1 is still evaluated, reported as **PAUSED** beside the ruling and its would-be verdict, **never counted as a pass**, and does not block SUCCESS; every other clause keeps full force; `--unpause-strategic` restores S1 byte for byte. `verdict_dropproof.py`: 13/13 drop-mutants still blocked + 5/5 pause controls; two deliberate mutations of the pause (leaking it to every clause; counting it as a pass) both turn the proof REFUSE. |
| **22** — the clean v2-line val | **(a) freeze n = 400** (sha256 `abe041db72a045b3…`, max \|d\| 0.0409) | adopted as the clean v2-line val; its non-exchangeability caveat travels with every number drawn from it |
| **23** — which margin leads H-vs-F | **default (a): lead with the FED-NAV margin** — nav is an input at training and inference | the nav-zero margin is reported beside it as an ablation |
| **26 (+ item 4)** — public HF repos | **leave them as they are, GATED with manual approval** | read back via the Hub API: `Sayood/tanitad-physicalai-w120-256x640cyl` (dataset) and `Sayood/tanitad-refc-v3` (model) are both `private: false, gated: "manual"` — **already in that state before the ruling; no change was needed.** ⚠️ A gated repo's FILE LIST stays visible, so the raw clip ids in 3,000 of the dataset's 6,061 file names remain readable; the ruling accepts that. The ceiling still BILLS rather than refuses, so any private push keeps its pre-push arithmetic check (E18). |
| **D10** — the pod | **later** | parked until the dev-box checklist is complete |

---

## §C — audit 2026-09-19: which of the older items are still live

**By the DataFlyWheel, on the Master Mind's request. Append-only — no old item above is edited.**
Evidence is quoted from tip `9e5f430`; `:N` means line N of THIS file at that tip. Verdicts:
**CLOSED-BY-RULING** (a later PI ruling answered it, quoted) · **OVERTAKEN** (what it asks about is
gone or replaced) · **STILL-LIVE** (a genuine open question, with its default). Package:
`TanitAD Research Lab/Data Engineering/Research/2026-09-19-pi-queue-audit/`.

⇒ **Of the 14 rows audited, only three still want the PI: 4 (fold into 26), 6 (optional, and
newly relevant), and the v6F runbook — and that one only if v6F is to be revived.**

| item | verdict | evidence |
|---|---|---|
| **1** — no VLM; the Alpamayo labels stand as teacher signals | **CLOSED-BY-RULING** | The PI's own words, typed into the DataFlyWheel's session on 2026-09-07 — *"no vlm, we stick to the alpamayo labels as teacher signals"* — first-hand there, relayed here; banked verbatim at `…/2026-09-01-v8-tacsit-release/V8_MANIFEST.json:166`. Consistent first-hand ruling to the Master Mind 2026-09-15 (c): the tactical layer learns *"using the gt labels in our data set"* (`Decisions/2026-09-15-pi-directives.md`). The default is fully in effect either way. |
| **3** — the 15-token strategic vocabulary | **OVERTAKEN** | refcv6 deactivates the whole strategic layer, *"route head, `g_str`, strategic GRU … (PI)"* (`SPEC_REFCV6_V2.md:51`; :1321). Nothing in the current plan trains a strategic vocabulary; the corpus gap (6 of 15 tokens) returns only if a later arm re-enables the layer. |
| **4** — `tanitad-refc-v3` is PUBLIC | **STILL-LIVE → fold into ITEM 26** | Still public: 2.142 GB, 12 files, **0** raw clip ids in its file names (read-only HF audit, 48 repos). ⛔ **The audit found TWO MORE public repos whose file names carry raw clip ids:** `tanitad-ph0-aug120` (39.9 GB, **6,833** names) and `tanitad-flagship-v5f-w120` (31.7 GB, **16** names), beside item 26's 455.4 GB repo (3,000). Default: status quo — item 4's own, and item 26's (c). |
| **5** — the `g_str` ~40k-step retrain | **OVERTAKEN** | *"The A40 is **stopped and gone**"* (:172), and refcv6 deactivates `g_str` (`SPEC_REFCV6_V2.md:51`). Heavy compute is now the pod the PI will provide after dev-box preparation (`Decisions/2026-09-15-pi-directives.md:28`). |
| **6** — a human spot-check of ~50 traffic-light frames | **STILL-LIVE — optional, and newly relevant** | The colour labels now have a consumer. `TRAFFIC_LIGHT_REACT_*` sit in the tactical vocabulary (`stack/tanitad/models/vocab_v7.py:108-109`); the tactical-goal gate was widened on 2026-09-18 so a refcv6 tactical arm gets those targets (`stack/tests/test_tacgoal_eval_target_wiring.py:50-55`); and 2026-09-15 (c) has the tactical layer learn every behaviour from the labels. **Default:** not done; accepted risk; the ego-only control on any traffic-light head stays the mandated guard. **Unblocks:** a per-instance error bound on the colour labels before refcv6 trains on them. |
| **7** — the RL pilot's adapter | **OVERTAKEN** | The executed RL path is now D9, `stack/scripts/ddv2_rl_refcv5.py` (PI 2026-09-17: *"dont park D9, solve it and prove it"*, :1052), which imports **none** of `refcv3_adapter`, `refc_adapter`, `rl_pilot_refc21`. ⚠️ Engineering note, not a decision: `refc.py` still consumes `v0` unconditionally, so a dropped `v0` reads as a stationary car (X15). |
| **9** — refcv5-v2 as the compose arm; environment grounding | **OVERTAKEN** | refcv5-v2 is **COMPLETE at 40,284 and evaluated** (`MODEL_REGISTRY.md:3372`). Environment grounding moved to refcv6 and was ruled 2026-09-17: the BEV encoder moves into the model forward (:796). |
| **10** — 11,286 parameters with no gradient; the `--w-tac-goal` budget | **OVERTAKEN** | refcv5-v2 finished (above). The budget was measured (`9d13fd2`, 2026-09-11: 0.05). E19 closed from the record in `9e5f430` (`PREREG_REFCV6_DEVBOX_PREPARATION.md:924-952`): `--w-tac-goal` gates only V1 arm C, and V2 supplies tactical targets through `--w-tac-v6`. |
| **11** — does "finish the implementation" authorise WP-C? | **CLOSED-BY-RULING** | PI 2026-09-17, verbatim *"for 3 chose defult"* (:1040-1041), mapped at :1051 to the agent/selection seam: *"carry it as a named arm with its own pre-registration."* WP-C's lever runs only that way. |
| **12** — refcv6 arm D: `ddim` + `--w-u0 0` refused | **CLOSED-BY-RULING** | `stack/scripts/refc_v3_train.py:839-841`: *"THE PI RULED 2026-09-11: 'follow your recommendation' — authorise this configuration behind an EXPLICIT ACKNOWLEDGEMENT"*. The flag is `ack_ddim_no_u0` (:849), and V2's F6 sets it by construction (:386-390). |
| **13** — refcv6 needs a pod | **CLOSED-BY-RULING** | *"Use my computer for prepration and final design of refcv6 then I will provide a pod for heavy work"* (`Decisions/2026-09-15-pi-directives.md:28`, consequence (e)). The request itself is earned by `PREREG_REFCV6_DEVBOX_PREPARATION.md` §8.4. |
| **14** — Qwen-Drive videos, scope, layers | **OVERTAKEN** | The corpus augmentation the PI ordered on 2026-09-15 runs as **SAM3** production on Thor (`Decisions/2026-09-15-pi-directives.md` §1). On SAM3-only maps, PI 2026-09-13: *"it can significantly augment our data set"* (`GOALS_AND_CLAIMS.md:11573`). Qwen-Drive's occupancy was measured unusable (:11550 there); its map regions survive only as an open fusion hypothesis (:11558 there). |
| **16** — the conflict statistic cannot see its defect | **NO PI RULING NEEDED — default half-built** | Built: `stack/tanitad/train/grad_conflict.py:271-273` logs `cos`, `ratio` and `proj`, and A9 priced it. ⛔ **Not built: B2 "restated as DIRECTION-ONLY with a magnitude criterion on `ratio`" exists only here (:669), in no pre-registration.** Owed by the detector's owner before any arm is scored on B2. |
| **v6F runbook** — `ChainConfig` has no nav label | **NOT A PI DECISION — a RED TEST with TWO defects** | Nav is already ruled mandatory (PI directive 2026-08-30, quoted by the trainer's own preflight; 2026-09-15 (b); :1169). `stack/tests/test_runbook_commands.py::test_every_runbook_launch_line_passes_the_trainers_own_preflight` **FAILS** (`:259`): the lines `ChainConfig` builds (`stack/scripts/v6_chain.py:604`, no `nav_cond` / `nav_labels`) are refused for **(1)** a missing `--nav-cond` and **(2)** `--horizons (1, 2, 4)`, whose heads 2 and 4 **no loss consumes**. `test_nav_v6stack.py` passes 17/17 — it is not the red one. v6F is absent from the PI's 2026-09-15 order (refcv6 → refav1 → v7). **Owner action:** fix `ChainConfig`, or retire the v6F runbook and its test with that reason. A PI question only if v6F is to be revived. |

<!-- PIQ-AUDIT-OLDER-ITEMS-2026-09-19 -->

> ⭐ **Master Mind, at landing (2026-09-19):** §C above was written at `9e5f430`, BEFORE the PI ruled
> in session on items 25, 22, 23 and 26 (+4) — see *PI RULINGS 2026-09-19* earlier in this file
> (`537b028`). Item 4 is therefore ANSWERED with 26 ("leave it as it is, gated with manual
> approval"). The audit found two MORE public repos with raw clip ids in their file names
> (`Sayood/tanitad-ph0-aug120`, 39.9 GB, 6,833 names; `Sayood/tanitad-flagship-v5f-w120`,
> 31.7 GB, 16 names), so ITEM 26 spans ~527 GB across three repos. Read back via the Hub API,
> **all three are already `private: false, gated: "manual"`** — the ruling holds for every one
> of them with no change. The ordering trap stands, larger: flipping all three PRIVATE would reach
> ~704 GB, and adding the 386.5 GB cache push ~1,090 GB — over the 1 TB ceiling, and BILLED.
> Still wanting the PI: **item 6** (optional traffic-light spot-check — MORE live now that
> `TRAFFIC_LIGHT_REACT_*` are refcv6 tactical targets; default: not done, accepted risk) and the
> **v6F runbook** only if v6F is to be revived.

---

## §C addendum — item 17, the guard the PI chose, and three rows the 2026-09-19 rulings moved

**DataFlyWheel, 2026-09-19, after `537b028` and the landing note above. Append-only.** Package:
`TanitAD Research Lab/Data Engineering/Research/2026-09-19-pi-queue-audit/` (revision section in
`RESULT.md`; `raw/evidence_rev2.json`).

| item | verdict | evidence |
|---|---|---|
| **17** — clip ids in banked records | **CLOSED-BY-RULING** | PI 2026-09-17: *"Its hygiene and reproducibility."* ⇒ option 2: the banked record is **left alone**, and `tools/clipid_scan.py` stops the count growing (:802-814). No history rewrite is proposed. |
| **3**, **5** — the strategic vocabulary; the `g_str` retrain | **CLOSED-BY-RULING — deferred**, not overtaken | PI 2026-09-19: *"the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality."* Both return when the strategic family resumes; the corpus gap (6 of 15 tokens) is unchanged. |
| **13** — refcv6 needs a pod | **CLOSED-BY-RULING** — reinforced | 2026-09-19: *"5. we will do it later"* — the pod is parked until the dev-box checklist is complete. |

### The guard, run on tip `07541b7`

| half | how read | at the ruling `9d19020` | at `07541b7` | grew? |
|---|---|---|---|---|
| UUID (default) | the CLI against `tools/clipid_baseline.json` | 120 in 27 files | **120 in 27 files** | **`NO-GROWTH`**, exit 0 |
| prefix, corpus list (4,719 ids) | the tool's own per-file counts, both trees | 530 | **530** | **0 files** |
| prefix, full index (306,152 ids) | the same | 870 | **870** | **0 files** |

⚠️ **The prefix half has no stored floor.** `clipid_baseline.json` records **0 prefixes for every
file** — it was written without `--clips` — so running `--clips` against it reads red on every
existing prefix. That is an unrecorded floor, **not growth**: the ruling-to-tip comparison above
shows none, and the corpus-list 530 reproduces the ruling day's own count. Recording it
(`--write-baseline --clips <list>`, naming the list) is the guard owner's task.

⇒ **Unchanged: only item 6 (optional) still wants the PI, and the v6F runbook only if v6F is to
be revived.**

> ### ✅ RULED BY THE PI, 2026-09-20: **v6F is RETIRED — the line goes to v7F.**
> Verbatim: *"we will go directly to v7f, we dont need revival of v6f"*. ⇒ the v6F runbook is
> **not** to be revived and v6F's fate stops appearing as an open item. ⛔ Both queue items are
> now closed: **the PI queue holds no open item.**


<!-- PIQ-AUDIT-ADDENDUM-17-GUARD-2026-09-19 -->

<!-- PREFIX-FLOOR-RECORDED-2026-09-19 -->
> ✅ **Item 17 guard, UPDATE 2026-09-19 — the prefix floor is RECORDED** (supersedes the §C addendum's
> *"⚠️ The prefix half has no stored floor…"*). `tools/clipid_baseline.json` now holds **530 prefixes in
> 53 files** against the **NAMED** 4,719-clip v7 corpus list (sha256 `a48251e89c7a8603…`), beside the
> unchanged UUID floor (120 in 27 files). `--clips` with that list reads NO-GROWTH; **any other list is
> REFUSED rather than compared** (the same tree reads 530 against the corpus and 870 against the 306k
> index, so an unnamed floor would manufacture or hide growth). The baseline stores counts and the
> digest only — no ids. Package: `TanitAD Research Lab/Data Engineering/Research/2026-09-19-clipid-prefix-floor/`.


---

## 19. DECIDE (or let the safe default stand): how the §4 re-tier lands in the HF repo — the only option that leaves it self-consistent DELETES published files

**Asked because the ruling is yours and already made; only the MECHANISM is open.** You ruled §4 on
2026-09-20 — *"a) for the term, (c) for the reporting"* — and the prereg applies it as ONE pass
after production completes (**ETA today 17:44**, 4,545/4,719 at the time of writing). The premise is
now verified on live data (`b513a26`): all **77** clips carrying the flag *"near path unlabelled
(seen, no class), no path cell on a non-drivable class"* have **zero** NON_DRIVABLE path cells, so
their compliant share under (a) is **exactly 1.0** and they clear the 0.9 threshold outright. The
(c) quantity is measured: withheld share min 0.1001 / median 0.1372 / mean 0.1706 / **max 0.4603**,
none above 50 %.

**What is NOT decided is where the bytes end up.** `corpus_publisher.gt_path()` routes validated
clips to `semantic_maps/gt/` and flagged clips to `semantic_maps/gt_flagged/`, so a re-tier changes
a clip's DIRECTORY. MEASURED: the 77 local sources all still exist in `corpus/out/` (4,546 npz,
14 GB), so any option is executable; they total **148.6 MB** (mean 1.93, max 3.10).
⚠️ And the publisher imports **`CommitOperationAdd` only** — it has no delete path today.

| option | storage | leaves the repo SELF-CONSISTENT? |
|---|---|---|
| **A** — manifest-only; bytes stay in `gt_flagged/` | **0 MB** | ⛔ no: 78 clips the manifest calls *validated* sit under `gt_flagged/`, and a consumer globbing `gt/` gets 78 fewer maps than the manifest promises |
| **B** — add to `gt/`, keep the `gt_flagged/` copy | **~150 MB** | ⛔ no, mirrored: `gt_flagged/` then holds 78 clips the manifest calls validated |
| **C** — add to `gt/`, **delete** from `gt_flagged/` | 0 net | ✅ yes — and it is the only one |

⛔ **I am not taking C on my own authority.** It permanently removes files from a published dataset
repo, and deleting published data is outside what I do without your word — your §4 ruling authorises
the **re-tier**, which is a judgement, not a deletion. ⚠️ Note the deletion in C is of a file the
**same pipeline produced and is simultaneously re-uploading**, not of unique data; that is an
argument for C, not a reason to skip asking.

⭐ **SAFE DEFAULT IF YOU DO NOT ANSWER BEFORE PRODUCTION ENDS: B**, because nothing about it is
irreversible and C stays available afterwards at any time. ⚠️ Its cost is honest and stated: ~150 MB
and a `gt_flagged/` directory that contradicts the manifest for 78 clips — so the pass will record
the duplicate **explicitly** in the manifest, and the re-tiered entries will name **both** paths,
so no reader of the record is misled even while the layout is untidy.
⛔ A is NOT the default despite being free: a file under `gt_flagged/` that the record calls
validated is precisely the *artifact that cannot be read in isolation* failure the anchor-units
disaster taught, and it would sit in the corpus permanently.

**What unblocks it:** one word — A, B or C. Nothing else waits on this; the pass itself is written
against whichever you pick and runs after DONE.

> ### ⛔ AMENDED 2026-09-22, BEFORE ANY APPLY — **the "default B if silent" I offered is WITHDRAWN**
>
> I wrote that if you did not answer before production ended I would apply **B**. That was **my own
> statement, and my own statement is not your approval.** A/B/C was put to you as a DECISION; acting
> on my own default would be **taking** it, not covering for its absence. ⚠️ And B was never free:
> it commits the repo to an untidy layout that only **C — your call — resolves**.
>
> ⭐ **The default is narrowed to what your §4 ruling ITSELF authorises: the JUDGEMENT, recorded in
> the manifest, with NO bytes moved.** `s4_retier.py --layout` now defaults to `pending`; every
> re-tiered entry carries `layout_open: true`, names this queue item, and names the path its bytes
> are **actually** at. ⛔ **`--layout A`, `B` and `C` ALL now refuse** without
> `--i-have-the-pi-ruling <X>` — the guard previously covered only C, which is how B became a silent
> default in the first place.
>
> ⚠️ **`pending` is NOT option A.** A's hazard was **silence** — a file under `gt_flagged/` that the
> record calls validated, with nothing saying so. A **documented** interim state is not a silent
> inconsistency: a reader in isolation is told exactly what is true, which is the entire point of
> the rule A would have violated.
>
> ⇒ nothing is foreclosed, nothing irreversible happens, and **A, B and C all remain fully available
> whenever you answer.** The question is unchanged: **A, B or C.**

> ### ✅ RULED BY THE PI, 2026-09-22: **C** — verbatim *"follow C"*. ITEM 19 IS CLOSED AND APPLIED.
>
> Applied the same day, **one atomic commit**, 166 operations (83 adds + 83 deletes), manifest
> updated and uploaded. VERIFIED FROM THE HUB, not from the tool's own success message:
>
> | | before | after | expected |
> |---|---|---|---|
> | `semantic_maps/gt/` | 4,569 | **4,652** | 4,569 + 83 ✅ |
> | `semantic_maps/gt_flagged/` | 108 | **25** | 108 − 83 ✅ |
> | `semantic_maps/worldmap/` | 4,677 | 4,677 | unchanged ✅ |
> | **total repo files** | 14,589 | **14,589** | **unchanged — it was a MOVE** ✅ |
>
> ⭐ The unchanged total is the load-bearing check: a copy would read 14,672, a botched delete
> fewer. The manifest agrees independently (`clips` 4,652 / `clips_flagged` 25), and
> `flagged_by_reason` now contains **only** *"path untestable (parked / stopped ego)": 25* — the
> *"near path unlabelled"* reason is gone entirely, which is the ruling taking effect.
>
> **(c) IS DISCHARGED.** `counts.s4_retier` publishes the withheld / unlabelled share beside the
> corpus numbers — min **0.100077** · median **0.137173** · mean **0.175081** · max **0.485261** ·
> **0** above 0.5 — and **all 83** entries carry a per-clip `retier` block with their own share.
>
> ⭐ **PROVENANCE SURVIVED, WHICH WAS THE POINT.** Each re-tiered clip KEEPS its `flag`,
> `failed_checks` and `path_classes` and gains `retier` naming the rule, your ruling, the layout,
> its compliant share (1.0) and its previous path. **A clip judged by rule (a) is still
> distinguishable from one that passed outright** — the corpus did not silently lose which rule
> judged it.
>
> ⚠️ Nothing was destroyed: the same bytes are in `gt/` from the same commit, the local sources
> remain in `corpus/out/`, each clip's `worldmap/` copy is untouched, and the pre-pass manifest is
> snapshotted at `publish/SEMANTIC_MAPS_MANIFEST.pre-s4.json` (md5 `705afde1fcb57f608be71ffe22440c86`).

<!-- RECONCILE-2026-09-22-REFCV6-READINESS-IS-NOT-A-LAUNCH -->

---

## ⛔⛔ RECONCILIATION 2026-09-22 — refcv6 READINESS work resumed on the PI's word; the STOP on LAUNCHING is **NOT** superseded

⚠️ **READ THIS BEFORE ACTING ON EITHER THE DIRECTIVE ABOVE OR THE NEW REGISTER ROWS.** The record
now contains two things a fresh session could combine into the wrong action, and this note exists
to make that combination impossible.

**What the record says in two places:**

1. **This file, 2026-09-11:** *"refcv6 TRAINING IS STOPPED"*, with the warning that
   `chain_refcv6.sh` and `sup_refcv6.sh` are staged and correct and *"a fresh context must NOT
   relaunch them"*. ⭐ **STILL TRUE AND VERIFIED TODAY:** both are present in the banked tree
   (**186** and **244** lines) and on disk at
   `TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-refcv6-build/code/`.
2. **`GOALS_AND_CLAIMS.md`, 2026-09-22** (`d014414`, `ff9a073`): rows stating that `H-BOXCLS-1` is
   implemented, stamped, mutation-proven, re-based onto the B1 corpus, and that *"still blocked,
   and still only this: the GPU call is the PI's"*.

**What the PI actually said, 2026-09-22, verbatim, in his own channel:**

> *"are we ready now to train refcv6?"*

> *"build the train agent-join and do also the rest to achieve the traibning readiness of refcv6"*

⛔ **THOSE ARE READINESS INSTRUCTIONS. NEITHER IS AN INSTRUCTION TO RELAUNCH.** He asked whether we
are ready and told me to *achieve* readiness; readiness work proceeded on that word and is what
`d014414` and `ff9a073` contain. **He has not said to restart the training**, and the 2026-09-11
stop therefore stands on the launch itself.

⇒ **"The GPU call is the PI's" means the call is HIS TO MAKE — it does not mean "launch when a card
frees."** A free-card reading is not authority. This is the same shape as the standing A7 rule
(*"Stay paused until I say"* — a clear card does not override what he asked), and the drumbeat
already records a session acting on *"A7 IS RUNNING"* after he had paused it.

⚠️ **What I have NOT verified, and therefore do not assert:** whether Thor is still held by the
Qwen-Drive / LiDAR-BEV work that motivated the 09-11 stop. The directive's reason was that *"the
two cannot share the card"*; I have not probed Thor this firing, so treat the card's occupancy as
**UNKNOWN**, not free. ⛔ Do not infer it from the dev box's `boxstat` — that reads the USER'S
DESKTOP GPU, a different machine entirely.

**The honest one-line state:** refcv6 is *materially* readier than it was — the class-weight lever
is built, guarded and pointed at the right corpus, and the eval-139 line at 416×1024 is complete —
**and it is not cleared to launch.** Two things gate an actual run, and both are the PI's:
the go-ahead itself, and the 4,713-clip corpus cache (MEASURED absent; ~375.4 GB against 402.3 GB
free on `D:`; ~10.6 h build) — see the register row *"THE NEXT GATING ITEM FOR refcv6"*.
