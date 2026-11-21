# refcv6 — the lever facts, read from source, with their controls

**Master Mind, 2026-09-10.** MEASURED from `stack/scripts/refc_v3_train.py` at HEAD and from
refcv5-v2's own `config.json` argv (`C:/Users/Admin/refcv5v2_final/config.json`, 35 top-level
keys, 65 argv tokens). Controls on every probe: **110** `add_argument` calls read from the same
file in the same breath (positive), **0** for a nonsense flag (negative). ⛔ A zero from this
mount is otherwise a claim about the mount, not the repo.

---

## 1. refcv5-v2's argv — the baseline, from the run's own record

```
--steps 40284  --batch 20  --lr 1e-4
--agents off
--w-u0 0.5
--anchor-control-units alat
--goal-str  --sel-refined  --sel-score-emitted
```

⛔ **ABSENT**: `--wp-index`, `--w-tac-goal`, `--max-speed-input`.

## 2. The four levers, with their declared defaults

| flag | file:line | default | refcv5-v2 passed | refcv6 arm |
|---|---|---|---|---|
| `--w-u0` | `refc_v3_train.py:5095` | `U0_WEIGHT_DEFAULT` = **0.0** (`:122`) | **0.5** | **D** = stop passing it |
| `--agents` | `:5151` | `"off"`, choices `off / head / oracle` | `off` | **A** = `head` |
| `--wp-index` | `:5340` | `"off"`, choices `off / on` | absent | **B** = `on` |
| `--w-tac-goal` | `:4922` | **0.0** | absent | **C** = a weight the PI must set |

### 2.1 ⭐ Arm D is a RETURN TO DEFAULT, not an addition

`U0_WEIGHT_DEFAULT = 0.0`, and the source comment at `:122` names it *"WP-4: the x0 loss, in
CONTROL space"*. refcv5-v2 **explicitly opted in** at 0.5.

⇒ Arm D does not add a flag; it **stops passing one**. That is the strongest single-lever form
available — the parsed-namespace diff is exactly one key by construction.

⇒ And it sharpens `D-DDV1-NO-DENOISING-LOSS`: DD-v1 has **no ε-prediction and no denoising MSE
at all** (its `diff_loss_weight = 20.0` is dead code). Our trainer's own default already agrees
with the paper. **refcv5-v2 is the arm that departed from both**, and it is the arm that lost
longitudinally. ⛔ Correlation, not cause — which is precisely why D runs first and costs nothing.

### 2.2 ⭐ Why `tac_goal_tok_head` received exactly zero gradient — the mechanical answer

`--w-tac-goal` is declared at `refc_v3_train.py:4922` with **`default=0.0`**, and refcv5-v2's
argv **does not contain it**.

⇒ The head's **11,286 parameters** took `grad_abs_sum` **exactly 0.00000 for all 40,284 steps**
because its loss weight defaulted to zero — **not** because of a wiring bug, and **not** because
the head was unbuilt. It was built, rollable, and unreached.

⭐ **"Rollable and trained are different claims."** ⛔ No refcv5-v2 result may be credited to the
22-token tactical vocabulary.

### 2.3 ⛔ Arm B is not one flag unless six sub-knobs are pinned

`--wp-index` ships with `--wp-index-mode` (`:5356`), `--wp-index-detach` (`:5367`),
`--wp-index-radius-m` (`:5373`), `--wp-index-hidden` (`:5380`), `--wp-index-scale-m` (`:5385`)
and `--wp-index-const-xy` (`:5388`). Every one must be **identical between arms A and B**, or B
moves more than one lever and the panel is non-attributable — the `--v2` conflation failure (ten
levers on two axes) is the precedent.

---

## 3. ⛔ `--max-speed-input` CANNOT run yet — and this blocks the PI's own named request

The PI asked, verbatim, that *"all necessary vocab are used weiter as inputs like nav commands
and max speed"*. The flag exists (`refc_v3_train.py:5502`, `action="store_true"`). Its own help
text states the blocker:

> reads the v8 label record's `speed_max_input` block (`v_max_ms`, units DECLARED on the wire as
> m/s); **the v7.2 release carries it on 0 of 4,572 records**, and **the flag REFUSES there
> rather than looking switched on**.

⭐ That refusal is **correct behaviour and the reason nothing silently broke** — it is the
opposite of the `tac_goal` failure above, where a zero default let a head look wired while
taking no gradient. Here the flag would rather stop than lie.

⇒ **`--max-speed-input` is gated on the v8 labels existing.** 0/4,572 is a hard zero, and it is
the PI's authorised item 3.

⚠️ **Where such a ceiling could come from is an open question, not an assumption.** PhysicalAI-AV
publishes **no map, no lane graph and no traffic-light feature** — the card says *"we do not
include open maps data"*, pinned by `stack/tests/test_physicalai_feature_readset.py`. So a
"posted limit" has no published supplier in the corpus. ⛔ Do not assume the v7 speed-bucket
ladder is a legal stand-in: a ceiling derived from the ego's own realised speed is the
**nav-echo defect** in a new costume, the same family as flagship v1's route head scoring
**1.0000** as an exact bijection of the nav it was fed.

---

## 4. ⛔ The consequence for refcv6's baseline — state it, do not absorb it

If `--max-speed-input` is ever folded into refcv6's inherited baseline, then **refcv6 versus
refcv5-v2 stops being a single-lever comparison.**

⇒ The clean structure is a **refcv6 baseline arm equal to refcv5-v2's argv exactly**, with every
lever arm differing from that baseline by **exactly one parsed-namespace key**. Max-speed is a
**separate, later, pre-registered lever**, gated on the v8 labels.

⛔ **Every arm carries a replicate from the start.** MEASURED on WP-D: an arm with **zero levers
moved** read "separably worse" on **5 of 9** family metrics and reproduced a headline ADE effect
at **+0.02460** against the lever's **+0.02610**. A one-seed panel cannot adjudicate this rig.

---

## 5. ⛔ CORRECTION — it was THREE heads and 18,472 parameters, not one head and 11,286

**MEASURED 2026-09-10** from refcv5-v2's own checkpoint, after this document was written. ⭐ The
probe was not asked for: it was run while verifying the Hugging Face upload, and it refused to
inherit the count in §2.2 above.

**The method is an analytic one, which is why it is trustworthy:** Adam allocates optimizer state
**lazily**, so a registered parameter with **no entry in `opt["state"]`** never received a
gradient. The checkpoint registers **357** parameters against **351** state entries.

| head | shape | params | state |
|---|---|---|---|
| `core.decoder.offset_head.weight` / `.bias` | (16, 384) / (16,) | **6,160** | default init, **untrained** |
| `tac_goal_tok_head.net.weight` / `.bias` | (22, 512) / (22,) | **11,286** | default init, **untrained** (as documented) |
| `scorer.goal_point.weight` / `.bias` | (2, 512) / (2,) | **1,026** | ⛔ **EXACTLY ZERO** — 0 of 1,024 weights nonzero |
| | | **18,472** | **never trained** |

### 5.1 ⭐ `scorer.goal_point` is structurally inert — and this one is an ANALYTIC prediction

`v6.py:2760` **zero-initialises** it. Zero init **plus** zero gradient predicts that it is still
**exactly zero**, and it is. ⛔ That is an identity, not an estimate, and no seed changes it.

⇒ **That head emits the constant origin.** The "free regression" half of the goal scorer
contributed nothing to refcv5-v2.

⚠️ ⇒ **The panel's `goal_setting FDE 0.6607 m / bearing 1.3730°` row is now UNVERIFIED**, because
it is not established whether that number reads this constant head or the parameter-free anchor
prior (`refc.py:1638`). ⛔ Do not quote it until the path is settled. **One read settles it.**

### 5.2 `core.decoder.offset_head` — MEASURED untrained, cause is a HYPOTHESIS

⚠️ It sits at random init **inside the arm whose `--sel-refined` lever failed its bar**. The
proposed cause — that `--sampler ddim` legitimately orphans it because the sampler pass refines
via `control_head` — is a **HYPOTHESIS, not confirmed**. The discriminating check is **one read
of any non-ddim refc checkpoint**.

### 5.3 Why the mapping is trustworthy — three independent cross-checks

⛔ A census that re-runs the producer's own derivation measures **determinism, not correctness**.
These do not:

1. **351 of 351** present ids match their aligned parameter's shape **exactly**.
2. The mapped parameter total is **108,257,502**, equal to `config.json`'s
   `param_breakdown.total` — an **independently authored** reference the probe never touched.
3. Same-breath controls `str_goal_head`, `scorer.cand_bias` and `conf_head` all read **TRAINED**,
   which excludes a global-zero artifact — the control that discriminates a real finding from a
   broken probe.

### 5.4 ⇒ What this changes for refcv6

⭐ **This defect class has now surfaced THREE times, and every time only AFTER an arm was trained
and published.** ⇒ `probe_zerograd3.py` belongs in `stack/scripts/` as a **post-training gate**,
run before any checkpoint is published or quoted. ⛔ A head that is built, rollable and unreached
must fail the gate, not survive into a model card.

⚠️ And the §2.2 framing above needs its scope widened: `--w-tac-goal`'s `default=0.0` explains
**one** of the three. The other two have different causes, and **a single explanation covering
one third of the evidence is how the count stayed wrong.**

---

## 6. ⭐ TWO PI ITEMS RESOLVED FROM THE RECORD — and one of them was about to manufacture a negative

**MEASURED 2026-09-10**, per-file greps with a positive control on every probe. ⚠️ Instrument note
first, because it changes how much of this is quotable: the **ripgrep-backed search tool returned 0
on a control that bash grep read as 4**, and a glob-wide bash grep under-reported **35 files across
~200**. Every absence claim below rests on a **per-file loop with a per-file content control**.

### 6.1 `--w-agent 1.0` IS pre-registered — item resolved, no PI decision needed

⛔ Correcting a claim made earlier today that it was *"MEASURED (provenance) / NOT PRE-REGISTERED"*.

| where | class |
|---|---|
| `…/2026-09-07-p1-agent-gate/PREREG.md:46` and `:90` | ⭐ **REGISTERED**, written before any outcome, register row **`D-P1-AGENTCOND-1`** |
| `…/2026-09-07-p1-agent-gate/code/launch_arms.py:50` (`head`), `:56` (`shuf`) | MEASURED, executed |
| `…/2026-09-07-p1-agent-gate/raw/config_head_s0.json` → `"w_agent": 1.0` | MEASURED, banked run record |
| `…/2026-09-05-agent-join-into-batch/raw/smoke_agt_head_600.config.json` → `seams.w_agent = 1.0` | MEASURED, 600-step smoke |

The registered line reads: `--agents head --w-agent 1.0 --agent-join <B1 EVAL join> --agent-join-allow-legacy-ids`.

⇒ **1.0 is not an invention; it is the value the P1 gate registered and ran.** ⭐ refcv6 still
refuses to hardcode it (`arms.py:289-293` raises `PIDecisionRequired`, *"pass `--w-agent` explicitly
so the record shows who chose it"*), and `arms.py:59` classifies it **CONSTITUTIVE, not a second
lever**, so supplying it does not break one-variable discipline.

⚠️ The one registered **0** is `PREREG_WPB_WAYPOINT_INDEX.md:204-205`, and it is **oracle-path
only** — `:489` says verbatim *"`--w-agent 0` is load-bearing, not a default."* ⛔ Do not carry it
to a `head` arm; `refc_v3_train.py:583` refuses `--agents head` at `w_agent <= 0`.

### 6.2 ⛔ `train2400_agents.jsonl.xz` NAMES TWO DIFFERENT FILES — pull by md5, never by name

⭐ **This is the 182-clip manufactured negative, and the architecture review names the wrong one.**

| artifact | clips | md5 | coverage vs v7.2 train (4,572) |
|---|---|---|---|
| **parity** join (HF `Sayood/tanitad-ph0-aug120 → joins/`) | 2,308 | `24cbdca8c3b23aafc2fb17e6bf99cf76` | ⛔ **182/4,572 = 3.98 %** |
| **B1 TRAIN** join (pod `/root/data/joins/`) | **4,427** | **`1c985e6d6ad34e605c4ebd30cb353558`** | ✅ **4,427/4,572 = 96.83 %** |

⇒ My standing premise *"the train2400 join shares only 182 clips with v7.2"* is **true of the wrong
file**. The right file covers **96.83 %**. `D-B1-OVERLAP-IS-4-22-PCT` says it plainly: the parity
file *"is **the wrong corpus**, not a missing file, and substituting it would train the agent seam
on ~4 % of B1 and manufacture 'agent tokens do not help' from a starved seam."*

⛔ **A 4 %-coverage arm does not crash. It trains, converges, and reads as a clean negative** — which
is worse than a crash, because it looks like a result. ⇒ **Identify the join by md5 `1c985e6d…`.**

### 6.3 The trainer has NO coverage floor — the gate is the only thing standing there

`refc_v3_train.py::enable_agent_join` (`:1708-1820`) refuses on **three absolutes only**: zero
joined episodes, zero stable-id matches without `--agent-join-allow-legacy-ids`, and zero labelled
NOW-frames. ⛔ **There is no minimum-coverage threshold.** A 4 %-coverage join passes every trainer
guard and is merely *stamped* into `config.json` as `agent_join_stats.frac_windows_labelled`.

⇒ That gap is exactly why the pre-launch gate's **C4** exists (`prelaunch_gate.py`, floor `0.90`),
and why C4 was proven **in both directions**: **0.968285 PASS** against **0.039808 FAIL**. ⭐ Its
read-failure guard states the discipline outright: *"⛔ one side produced NO clip ids. A zero here
is a claim about the READ, not about the corpus — refusing to report it as coverage."* It returns
**`INCONCLUSIVE`, never `0.0`.**

### 6.4 Supervisor audit — copy `sup_refcv6.sh`, not `supervise_run.sh`

| script | trainer launch | its own `sleep`s | verdict |
|---|---|---|---|
| `sup_refcv6.sh` | ✅ `:179` | ✅ `:197`, `:217` | ⭐ **complete**, and closes fd 200 on its heredocs too |
| `sup_refcv5_v2b.sh` (the refcv5-v2 watchdog) | ✅ `:78` | ✅ `:140`, `:155` | complete |
| `sup_refcv5.sh` | ✅ `:145` | ✅ `:221`, `:225` | complete |
| ⛔ `stack/scripts/supervise_run.sh` | fd **9**, closed only in a subshell `:130` | ❌ `:119, :123, :134, :142` | **PARTIAL — its heartbeat `sleep` still inherits the lock** |

⚠️ `supervise_run.sh` is the **exact partial fix that shipped and failed the same day**: the lock was
held by `sleep 180`, the supervisor's own poll child, which had outlived its parent. ⛔ **Every child
gets `200>&-`, the sleeps included.**

⚠️ **UNVERIFIED, not clean:** `sup_refcv5_v2.sh` and `stack/scripts/sup_refcv3.sh` never became
readable, and `pod_currency_audit.py`'s contents resisted every read (`EISDIR` on a regular `.py`).
Their existence and sizes are measured; their contents are not.

---

## 7. ⭐ WHAT A HEALTHY `--agents head` RUN STAMPS — the number to compare a launch against

**MEASURED** from `…/2026-09-07-p1-agent-gate/raw/config_head_s0.json`, the `agent_join_stats` block
that `enable_agent_join` wrote during the real `--agents head --w-agent 1.0` run (control: `w_agent`
read 2 both before and after).

| | train | eval |
|---|---|---|
| episodes joined by **stable id** | **104 / 104** | **35 / 35** |
| joined by **legacy id** | **0** | **0** |
| ambiguous legacy ids in join | **0** | **0** |
| `id_space` | `stable-63bit` | `stable-63bit` |
| windows labelled | 16,845 / 17,787 | 5,818 / 5,985 |
| ⭐ **`frac_windows_labelled`** | **0.947** | **0.9721** |
| target boxes pre-filter | 563,180 | 213,621 |

⭐ **This independently reproduces `PREREG.md:72` from the BANKED ARTIFACT rather than from prose** —
104/104 episodes, 16,845/17,787 = 94.7 %, 563,180 boxes. The rig line is corroborated, not asserted.

⛔ **Use `frac_windows_labelled ≈ 0.95` as the launch's sanity number.** A starved join stamps
something near **0.04** here **and the run still starts**, because the trainer has **no coverage
floor** — only the pre-launch gate's C4 does. ⇒ *A 4 %-coverage arm trains, converges, and reads as
"agent tokens do not help": a manufactured negative, worse than a crash because it looks like a
result.*

### ⚠️ `--agent-join-allow-legacy-ids` — passed by P1, never needed, and NOT to be carried forward

The P1 gate passed it while `id_space` was `stable-63bit` and `n_episodes_joined_legacy_id` was
**0**. ⇒ **a no-op on that join.** ⛔ **But not a no-op in general:** on a different join it silently
disables the collision guard at `refc_v3_train.py:1749`, which exists because **34 of 2,308 clips
share a prefix**. ⚠️ And needing it at all is itself a signal of being on the wrong id space.

⭐ **VERIFIED for refcv6:** `arms.py` carries it **zero** times (control: `DEFAULT_AGENT_JOIN` reads
4 in the same file). The P1 argv was **not** carried forward wholesale, and the collision guard stays
armed.

### ⚠️ Two configs remain UNVERIFIED, not clean

`…/2026-09-07-p1-agent-gate/raw/config_shuf_s0.json` and `config_off_s0.json` **never became
readable** across 40 then 50 attempts. ⛔ A background job exited **0** while printing nothing for
them — no `READ_OK` line ever passed — and the greps then ran against unreadable files and returned
empty. **A clean exit code on a silent gate** is exactly the documented failure mode. Nothing in this
document rests on those two; the `shuf` arm's weight is independently established from source at
`…/code/launch_arms.py:56`.
