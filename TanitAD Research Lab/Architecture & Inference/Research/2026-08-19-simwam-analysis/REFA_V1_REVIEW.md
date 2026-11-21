# REF-A v1 and v1′ reviewed against every validated finding — and the ONE cell nobody occupies

`MEASURED (ours)` + `PUBLISHED` (banked) · **T0 throughout** · written 2026-08-20
at the PI's request, after I twice asserted things about REF-A that the repo
already answered.

⛔ **First, a correction.** I told the PI *"`refa.py` in-tree is still the OLD
geometry, I have not verified whether v1 is implemented or only documented."*
**REF-A v1 IS implemented**: `refa_v1.py` (32 KB), `refa_v1_plan.py` (15 KB),
`refa_v1p.py` (8 KB), plus `REFA_V1_DESIGN.md` and `REFA_V1_DETAILED_REPORT.md`.
`refa.py` is the **superseded** arm, not the current one. I read the wrong file.

---

## 1. REF-A v1 against the validated findings — it holds up well

| # | v1's choice | validated finding | verdict |
|---|---|---|---|
| 1 | frozen **DINOv3 ViT-L/16**, d=1024 | FROST-Drive: *frozen+weak is the WORST arm* (7.39) — encoder strength is axis 3 | ✅ |
| 2 | **640 tokens, 120° HFOV**, 256×640 | axis 4, *"cheapest to fix"*; REF-A's 256 tok / 51.39° | ✅ |
| 3 | **no bottleneck**, adapter ≥ 1024 | FROST-Drive interface width 5120-d 8.17 vs 256-d 7.68 | ✅ |
| 4 | objective = **predict future PATCH features (L2)** | axis 1 *prime suspect* — successful frozen systems propagate features forward | ✅ |
| 5 | **patch tokens only**, never CLS | DINO-WM ablation: global/CLS *"significantly degrades"* | ✅ |
| 6 | behaviour from **iCEM + MPC** | axis 2 *prime suspect* — CEM/MPC or head+WM-aux, never feed-forward alone | ✅ *(but see §3)* |
| 7 | hierarchy kept (FiLM str→tac→op) | PI directive | ✅ |
| 8 | goals enter the planning **COST** | v3 direction / D-033 | ✅ |
| 9 | 6 s horizon, three rates | PI directive | ✅ |
| — | action = **broadcast-concat** | ⭐ **E-ACTSTREAM-2 at v1's OWN geometry**: token−concat **+0.000186 [+0.000152,+0.000222]**, SEPARATED — concat is right here | ✅ |

⇒ **v1 addresses all four ranked axes.** It is not REF-A-with-a-bigger-encoder;
it changes what the objective asks and where behaviour comes from.

### 1.1 ⭐ The property that matters most, and v1 has it by construction

**v1's prediction target is the FROZEN encoder's future patch features.** The
target is *external and cannot drift*. Contrast the superseded `refa.py`, whose
target lived in **adapter space** — trainable, therefore driftable — which is
exactly why its docstring needed *"collapse-to-easy-targets is the known failure
mode … the inverse-dynamics head + SigReg-on-predictions provide the
anti-collapse pressure."*

⇒ **v1 has NO `InverseDynamicsHead`, NO SigReg, and needs neither** — a fixed
target makes collapse-to-easy-targets structurally impossible. That is a
*correct* simplification, not an omission. (Verified: 0 occurrences of
`InverseDynamics`, `sigreg`, `detach` in `refa_v1.py`.)

## 2. ⛔ v1′ — the defect, and it is real

`refa_v1p.py` carries the action as **tokens in the shared attention stream**,
and cites **E-ACTSTREAM-1** for it (token beat concat 5.9–9.9× on **16-token v6
cell fields**). It contains **zero mentions of E-ACTSTREAM-2** — the transfer
test at **v1's real 640-token geometry**, which **inverts** that result:

| arm | MSE | vs C-PERSIST |
|---|---|---|
| `add` | **0.037732** | ✅ beats |
| `concat` | 0.037956 | ✅ beats |
| **`token`** | **0.038141** | ✅ beats — but **worst of the three** |

`token − concat` **+0.000186 [+0.000152, +0.000222] SEPARATED**.

⚠️ **The nuance that saves v1′ from deletion but not from its docstring.**
E-ACTSTREAM-2's own consequences table says *"`refa_v1p.py` (v1′) **kept** — it
is the right arm for the **small-token regime**"*, and flags the open question:
the **tactical** predictor runs on **64 `tac_queries`**, an order of magnitude
closer to where tokenisation won.

⇒ **v1′ is parked at the WRONG geometry relative to its own evidence.** Its
docstring quotes the 640-token cost (*"642 vs 640 — +0.6 %"*) as if the scheme
applied there, while the measurement says token is the worst option at exactly
640. **The defensible v1′ is action-tokens on the 64-query TACTICAL predictor,
not on the 640-token operative field**, and the docstring must carry
E-ACTSTREAM-2 or it will be quoted as though E-ACTSTREAM-1 still governed.

## 3. ⛔ The blocker v1 imports, stated in its own docstring

*"the recipe imports our known-worst component (the action search) — which is why
`refa_v1_plan.py` carries a structural floor **and** a cost-fidelity gate rather
than trust."* Axis 2 is satisfied **by design**, not yet by measurement. iCEM+MPC
is the same family as the CEM planner measured broken (C101). **The planner is
part of the arm under test** and v1's result is uninterpretable until the search
is shown to work.

## 4. ⭐ How this meets tonight's E-TRUNK results — they corroborate independently

E-ACTSTREAM-2, §"the more important finding":

> On v6 cell fields, **nothing** beat C-PERSIST across four configurations.
> On DINOv3 fields, **every arm does**, separated. ⇒ **the binding constraint on
> every previous readout result was the REPRESENTATION, not the predictor.**

That is **the same conclusion E-TRUNK-2/3 reached by a completely different
instrument** — a decodability probe rather than a dynamics predictor. Two
independent methods, same verdict: **the v6 representation is the constraint;
DINOv3's is not.**

## 5. ⭐⭐ THE 2×2 — and the empty cell IS the trained-WM recommendation

The axis that matters is **not** frozen-vs-trained. It is **whether the encoder
can move its own prediction target**:

| | target = **own** latent (drifts) | target = **external** frozen features (fixed) |
|---|---|---|
| **encoder TRAINABLE** | ⛔ **v6 S-W today** — `z_true_steps`, *"detached by the caller"*; detaching stops gradient *through* the target but the encoder still **decides what it will be** next step | ⭐⭐ **EMPTY — this is the recommendation** |
| **encoder FROZEN** | (degenerate — nothing to train) | ✅ **REF-A v1**, DINO-WM, V-JEPA 2-AC |

**The recommendation for the trained WM, stated plainly:** *keep our encoder
trainable, and replace the self-target with an external one — predict **future
frozen DINOv3 patch features** instead of our own future latent.*

* The encoder stays **ours and trainable** ⇒ `D-003`'s from-scratch main track is
  untouched; this is **not** adopting a frozen encoder.
* The target becomes **ungameable** ⇒ the encoder can no longer satisfy the loss
  by making its own outputs easy to predict. To succeed it must build a state
  from which *someone else's* features are predictable.
* It is **DeepSight's measured recipe** (`2605.10564`: frozen encoder + MSE
  world-state loss vs DINOv3 future BEV feats, Bench2Drive DS 86.23), with the
  one change that our encoder trains.
* **Nobody in this programme has run this cell.** v6 sits top-left; REF-A v1 sits
  bottom-right.

**Second, cheap, independent:** wire `InverseDynamicsHead` (already in-tree) into
S-W. v6 has **zero** grounding — `aux` is `masked_cells` + `sigreg`, both
label-free. REF-A's runs reached `aux_speed_r2` **0.9825**; v6 reads **−0.005**.

## 6. ⚠️ What this review does NOT establish

* **No T1 claim.** Every number here is T0. Whether the empty cell drives better
  is unmeasured — that is exactly the tier error of C129 and it is not repeated.
* **v1 is unmeasured.** Its axis coverage is *design*, not result; §3's planner
  blocker stands.
* **The 2×2 is a framing, not a theorem.** It predicts that anchoring the target
  fixes what E-TRUNK-2 measured. That prediction is testable — E-TRUNK-2 is
  0-GPU on banked features — and must be **pre-registered** before it decides a
  GPU-day.
* I have **not** reviewed `refa_v1_plan.py`'s iCEM in depth, and §3 makes it the
  load-bearing risk. That is the next read.
