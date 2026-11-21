# SPEC — `--no-strategic`: bypass the strategic layer, feed nav straight to tactical and operative

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-strategic-bypass/`
**Written** 2026-09-06, **BEFORE any arm of this package has produced a training number.**
**Skill** `TanitAD_ValidateAIDesign`. **Tier** T1 (self-action open loop) for every driving row.
**Compute** 0 GPU spent by this package. The arm below is **pre-registered, not launched** —
the A40 is running refcv5 to ≈2026-09-08 07:33 UTC and **was not touched**.

---

## 0. The directive this implements (PI, 2026-09-06) — implemented, not re-litigated

> *"Our strategic layer is just setting the goal constraints, mainly coming from the route. In
> further steps we will add strategic thinking like setting the driving mode, steering long-term
> efficiency. So: remove the strategic layer in the next experiments, feed the nav command to
> tactical and operative planning, solve the driving task, then add the strategic layer. Include a
> flag to ignore the strategic layer in the next iterations."*

⛔ **BINDING PI RULING carried into every row here:** *"The nav command is NOT a training signal. It
is an INPUT simulating the nav system of the vehicle."* ⇒ nav is a **first-class ROUTE INPUT
available at inference**. The words *"oracle nav"* and *"deployment gap"* do not appear in this
package, and `os_navzero` is a **ROBUSTNESS ABLATION**, never a deployment estimate.

---

## 1. The pre-registration block

```yaml
hypothesis: D-STRAT-BYPASS-1        # registered in GOALS_AND_CLAIMS.md this turn
one_variable: no_strategic          # the ONLY difference from refcv4b
held_constant: [corpus, labels, anchors_file, seed, steps, batch, lr, warmup,
                workers, v2_lru, image_hw, ego_dropout, sel_accel_max,
                anchor_v0_conditioned, anchor_control_units, nav_from_v7,
                u8_batches, every other refcv4b flag, byte for byte]
success: >
  the bypassed arm's ade_0_2s is separated BETTER than refcv4b's `os` on the SAME
  windows under the paired episode-cluster bootstrap (n_boot 2000, seed 0,
  cluster = episode), AND the margin survives the training-variance replicate
  of clause 4.
failure: >
  no separated improvement (CI straddles 0, or separated WORSE). The strategic
  indirection is then NOT the binding constraint and the next lever is named.
controls: [ha, ha0, ha0_ext, os_navzero, os_navshuf, frames_blind, ego_zero,
           replicate_arm]
splits: {train: B1 4,572 v2ep (refcv4b's own, NON-PARITY by design),
         eval: the banked 4,823-window / 141-episode open-loop surface}
```

⚠️ **`held_constant` is checked by DIFFING THE LAUNCH COMMANDS, not by intent** (§8). The
2026-08-22 precedent: a row-bank arm changed `n` *and* silently multiplied effective λ, and two
sweeps were invalidated.

---

## 2. What "the strategic layer" IS here — named by module, because a bypass of a vague thing
proves nothing

On `--arm hier` (refcv4b's arm) the strategic layer is **eight parameter groups** and it reaches
the emitted plan through **exactly four edges**:

| # | edge | file | on refcv4b |
|---|---|---|---|
| S-BYPASS-1 | `ctx` → `ctx_to_cond` → the decoder CONDITION (**operative**) | `refc.py` | **LIVE** |
| S-BYPASS-2 | `g_str` → `gstr_embed`/`gstr_film` FiLM on `z_tac` (**tactical**) | `refc_v3.py` | **LIVE** |
| S-BYPASS-3 | `route_logits` → `route_to_anchor` re-ranks the fan (S5) | `refc.py` | off (`graft_route` False) |
| S-BYPASS-4 | `gp_head(ctx)` → tactical + selection (E15) | `refc_v3.py` | off (no goal-point flag) |

The parameter groups: `core.strategic` (the `StrategicCtx` GRU + proj), `core.decoder.ctx_to_cond`,
`core.route_head`, `str_goal_head`, `gstr_embed`, `gstr_film`, `nav_to_str`, `ego_to_str`
(+ `gp_head`/`gp_cond` when built).

⛔ **`hierarchy=False` IS NOT THE BYPASS.** It *deletes* the modules, which removes `state_dict`
keys and breaks strict loads on banked checkpoints — the documented 11/11 failure with exactly 6
unexpected keys that made a dead 1.57 M tensor be *kept* rather than removed. §7 measures that this
is still true today.

---

## 3. What tactical and operative RECEIVE under the bypass — stated plainly, because a sibling
measured what happens when a route signal arrives stripped

**OPERATIVE (the anchored-diffusion decoder).** Receives the measurement vector
`m = measurement([v0/10, nav_onehot(4), ego_valid, (nav_known)])` through `cond_proj`, plus the
conv-map tokens, plus the tactical latent (E7) and the tactical goal (E9). **The nav command is
INSIDE `m` and is unaffected by the bypass** — it was always a direct edge and never went through
the strategic layer. What it loses is `ctx_to_cond(ctx)`, one additive term on the condition.

**TACTICAL (`PhiTac` → `z_tac` → lat/lon heads → the tactical goal).** Receives the pooled window
features, **plus E13's `nav_to_tac(nav_embedding(nav_cmd))` added directly into `z_tac_raw`**, plus
E11′'s ego block. What it loses is the E4 FiLM by the strategic goal `g_str`.

⛔⛔ **WHAT THE NAV TOKEN CARRIES, AND WHAT IT DOES NOT — the honest answer.** It is a **4-way
CATEGORICAL** (`NAV_COMMANDS`), one-hot at the core and a 64-d `nn.Embedding` row at E13. It
carries **NO RANGE, NO DISTANCE AND NO TIME**: nothing in either path multiplies it by a metre or a
second, and there is no arc-length, no "in 40 m", no time-to-manoeuvre. It is *which way*, never
*how far* or *when*. ⚠️ This matters and is not a quibble — MEASURED on the goal-point surface, a
**bearing with its range stripped is separated WORSE by +2.3632 m**, and an **oracle 3-way
command's best decoding is "go straight" for all three classes**. ⇒ **the bypass hands the lower
layers a DIRECTION, not a goal constraint**, and if the arm fails, that is the first thing to look
at (§6, lever 1). *(Coordinated with the sibling establishing this from source — not duplicated.)*

⚠️ **E13's `nav_to_tac` is ZERO-INIT.** nav→tactical is therefore **bit-inert at step 0** and only
becomes live through training. MEASURED in §7 P2b: make that projection non-zero and `z_tac`,
`lat_logits_tac`, `lon_logits_tac` and `g_tac` all move. So the pathway is real, but "nav reaches
tactical" is a statement about the **trained** arm, and an untrained one is not evidence either way.

---

## 4. ⛔ ONE SEPARATED CI IS NOT ENOUGH ON THIS RIG — what the criterion actually requires

refcv4b is **DETERMINISTIC at inference** (`refc.py` sets the decoder's noise to `zeros_like` when
not training), so the inference-seed floor for this rig is **0** and the refav1 sampling-planner
floor (≈0.30 m) does **not** apply. The binding uncertainty is therefore **TRAINING variance**, and
`H-ESTIM-SEED-1` is explicit: the paired episode-cluster bootstrap resamples **episodes with the
models held fixed**, so a separated interval answers *"would another draw of EPISODES say this?"*
and never *"would another TRAINING RUN say this?"* — two arms differing in **nothing** cleared it on
**6 of 42** family cells (**14.3 %**).

⇒ **The criterion has two clauses and both are pre-committed:**

1. **NECESSARY** — `bypass − os` separated better on the paired episode-cluster bootstrap.
2. **SUFFICIENT** — the margin **exceeds the training-variance floor measured by a REPLICATE ARM**
   (refcv4b's argv + a different `--seed`, same 40,284 steps).

⛔ **The replicate is a COMPUTE DECISION FOR THE PI, and it is named as a blocker, not assumed:**
each arm is ~44.5 h of A40 at refcv4b's measured 3.844 s/step, so the full package is **~89 h**.
**If only one arm is funded, clause 1 is reported ALONE and the row is stamped
`NECESSARY-NOT-SUFFICIENT — no lever claim`.** That is a reporting rule committed in advance, not a
fallback invented after seeing a number.

---

## 5. The four families — reported per family, never pooled

⛔ **ADE alone is an INCOMPLETE eval.** All four families are reported with their estimator and CI
on the same windows.

| family | reported | reference arms (`MODEL_REGISTRY.md` §4.6, refcv4b @40,284, T1, 4,823 win / 141 eps) |
|---|---|---|
| **ADE / overall** | `ade_0_2s` + horizons | `ha0_ext` **0.2874** · `os` **0.2975** · `ha` **0.2996** · `os_navshuf` **0.3013** · `os_navzero` **0.3928** · `ha0` **0.6723** · `ego_zero` 1.1310 · `frames_blind` 1.0491 |
| **LONGITUDINAL** | speed MAE, target-speed acc@0.5, along MAE, headway / time-gap / TTC **with the censoring count** | refcv4b speed MAE 0.2909 m/s; `os − ha` **+0.0368 separated WORSE**; 753/1,225 TTC-censored at 30 s |
| **LATERAL** | heading, yaw-rate, cross-track, **curvature MAE with the straight floor beside it** | cross-track **0.0979** (best of any arm) but curvature MAE **0.008097** vs the `ha0` straight floor **0.006802** and the echo control 0.003712 |
| **TACTICAL** | LAT/LON acc + κ, per-class recall, goal FDE | LAT κ 0.8289 · LON κ 0.5178 (`ha` κ 0.6071 still ahead) |
| **STRATEGIC** | route acc + κ, nav echo index, `nav_compliance` | route κ **0.4852** [0.4057, 0.5671]; echo index 0.6405 |

⚠️ **The brief's reference list gives `os` 0.2965 and `os_navzero` 0.3926; the registry gives
0.2975 and 0.3928.** The registry is quoted, per CLAUDE.md's source-of-truth rule. The
third-decimal difference does not move any criterion here, and it is flagged rather than silently
resolved.

⛔ **INVALID ON THIS ROW, do not compute or quote:** `oracle_sel`, `anchor_acc`,
`sel_agrees_oracle` — `refcv3_arm.py` computes `a_star` against `model.core.decoder.anchors`, which
the trainer forbids for a **v0-conditioned** vocabulary, and this arm carries
`--anchor-v0-conditioned`. ⛔ **Never the `|dyaw| > 0.15` gate.**

⛔ **The curvature row carries the straight floor beside it, always.** A plan that never steers
scores `ha0` **0.006802** here (and **0.040083** on the refav1 rig — a different rig, never mixed).
⚠️ **A sibling has traced refcv4b's curvature defect to the decoder's FREE-WAYPOINT REFINEMENT** —
it halves ADE and *doubles* curvature error. ⇒ **a curvature change in this arm must NOT be
attributed to the bypass** without separating it from that mechanism, and this package will not
re-attribute it.

---

## 6. If it FAILS — the next levers, named in advance (RULE ZERO: a refutation is a waypoint)

1. **The nav token carries no range.** Give the lower layers a **metric goal point** instead of a
   direction (E15 `gp_cond` already exists and is off here). MEASURED elsewhere: a goal POINT is
   worth **+4.7 PDMS** where a categorical command is worth **+0.2**.
2. **E13's `nav_to_tac` is zero-init and may simply never open.** Log its weight norm per 500 steps
   — the Caveat-B discipline: a 0.0000 gate must be a LOGGED fact, never inferred later.
3. **The strategic layer was not the binding constraint; the tactical decision is.** LON κ 0.5178
   still loses to `ha` 0.6071 — a longitudinal-decision lever, not a hierarchy lever.
4. **`os − ha` was already only −0.0021 [−0.0178, +0.0154], NOT separated.** If the bypass moves
   nothing, the honest reading is that this arm's headroom is not in the hierarchy at all.

---

## 7. The gates this package DOES and DOES NOT claim

| gate | status |
|---|---|
| **G-RANK** (participation ratio) | ⛔ **NOT APPLICABLE and not claimed.** This is a WIRING change: it adds no parameter and changes no encoder. There is no representation to score, and the bare 8.56 floor is retired anyway — a participation number without a matched-corpus, matched-`n`, matched-`d` reference is **UNDECIDABLE**. |
| **G-DECODE** | ⛔ **NOT APPLICABLE** for the same reason. |
| **G-DRIVE** | ⭐ **THE GATE FOR THIS ARM** — T1, four families, paired episode-cluster bootstrap, plus the clause-4 replicate. |
| **wiring proofs** | ⭐ **ALL PASS, 0 GPU** — §7 of `RESULT.md`: mutation proof both directions, byte-identity, strict load with a negative control. |

---

## 8. The exact launch command (⛔ NOT to be run before the A40 is free ≈2026-09-08 07:33 UTC)

**refcv4b's argv byte for byte, plus `--no-strategic`, plus a new `--out`. Nothing else moves.**

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --anchors /workspace/experiments/refcv4b-b1-v72-40k/anchors.pt \
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
  --sel-accel-max 2.0 --goal-str --ego-state-inject --ego-dropout 0.5 \
  --no-strategic \
  --out /workspace/experiments/refcv4b-nostrat-b1-v72-40k
```

**Preflight first, and it must print the bypass line:**

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --preflight --arm hier --size base --image-hw 256 640 --nav-from-v7 \
  --anchors /workspace/experiments/refcv4b-b1-v72-40k/anchors.pt \
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
  --sel-accel-max 2.0 --goal-str --ego-state-inject --ego-dropout 0.5 \
  --no-strategic --out /workspace/experiments/refcv4b-nostrat-b1-v72-40k
```

Expected, verbatim (MEASURED on the dev box, `raw/PREFLIGHT_no_strategic_ON.log`):

```
[v3-preflight] strategic_layer=BYPASSED (--no-strategic) | ctx->decoder=OFF |
g_str->tactical_FiLM=OFF | route_readout->selection=OFF | goal_str_loss=NOT APPLIED |
nav->operative=ON(measurement) | nav->tactical=ON(E13)
```

⛔ **If that line reads `ACTIVE`, the flag did not reach the config — STOP.** A GATE ROW CARRIES ITS
ARM; three arm-substitutions have already been found in this programme.

**Notes on the command, each one a decision:**

* `--goal-str` is **KEPT** although the bypass makes its loss inert. It keeps the **LAN label
  pathway and therefore the dataloader byte-identical to refcv4b's**, which is what makes the
  comparison matched. The trainer stamps `goal_str_loss_applied: false` so the record cannot be
  misread, and the preflight says so out loud instead of skipping silently.
* `--anchors` points at **refcv4b's own file**, read-only (that run is COMPLETE): file sha256
  `e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb`. The byte-identical
  **declared** bank `pod:/workspace/anchors_117_alat_declared.pt` may be substituted; it changes
  `control_units_source` from `cli-override-legacy-file` to `file` in the record and **nothing in
  the model**.
* ⛔ **`--out` is a NEW directory.** Never write into `refcv4b-b1-v72-40k`, and ⛔ **never into
  `refcv5-*`** — the A40 is running refcv5 to ≈2026-09-08 07:33 UTC.
* The **replicate** of clause 4 is this identical command with `--seed 1` and its own `--out`.

---

## 9. Closing the loop

`GOALS_AND_CLAIMS.md` carries `D-STRAT-BYPASS-1` (this pre-registration) and
`D-STRAT-BYPASS-WIRED-1` (the wiring proofs, MEASURED). `RESULT.md` carries the proofs with their
artifact paths. Raw JSON + logs under `raw/`. **No registry row** — no model was trained.
