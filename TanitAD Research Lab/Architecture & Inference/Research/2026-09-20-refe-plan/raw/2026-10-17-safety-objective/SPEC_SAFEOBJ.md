# SPEC SAFEOBJ: constrained EPO (`--epo-con g5`). Math, flags, argv, gate, reachability, checkpoint, cost

Companion to `PREREG_SAFEOBJ.md`, whose values are binding.
- Line numbers refer to `PK/refe/scorer_finetune.py` at `daa4568` (blob `7c8071ea…`), where PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- MEASURED 2026-10-17: reversing `test_rl_r0c.R0C_FILE_REVERSALS` on that blob gives `d34d5cf6…` = `R0_TRAINER_BLOB`. So the trainer has not moved since R0c's patch, and R0c's bytes are the base this patch reverses to.

## 1. Math

### 1.1 Notation

One micro-batch has b sets (b = 8) and M = 64 proposals per set.
- Arm logits: `sxB`, `sxA` are [b, M, 7]; the lane head is on, so `native7`.
- Labels: `tg` is [b, M, >= 6] in `rl_reward.COMPONENTS` order (`nc, dac, ep, ttc, c, ddc`).
- `y^TTC = tg[..., 3]` and `y^NC = tg[..., 0]`. The column indices are READ from `rl_reward.COMPONENTS.index(...)`, never typed.

The policy is exactly the EPO policy (`epo_policy`, :188-191), with `rule_fn = sel_score` (:750-752) and T = `a.epo_temp` (0.1):

    pi^B_{s,i} = softmax_i( log(sel(sigmoid(sxB_s)) + 1e-6) / T )
    pi^A_{s,i} = the same on sxA.detach()     (reference = arm A, detached; no gradient can reach A)

### 1.2 The constraint term (added to arm B's loss only)

    c^k_X = (1/b) * sum_s sum_i pi^X_{s,i} * y^k_{s,i}                  k in {TTC, NC},  X in {A, B};  float32
    g^k   = c^k_A - eps - c^k_B                                           (> 0 = violated;  c_A carries no graph)
    L_con = sum_k lambda_k * g^k                                          lambda_k: a Python float, constant within one optimiser update
    lB   <- lB + L_con                                                     (after the EPO term, inside train_losses)

- The loop divides `(lA + lB)` by `accum` (:1380), so L_con is scaled exactly like the EPO term.
- L_con is NOT multiplied by `lam_pg`. The ramp governs the ascent; the dual governs the constraint.
- At update 0, λ = λ0 = 0, so the term contributes nothing.

### 1.3 The gradient reaches the progress rows (why this lever can act where a safety head cannot)

    d c_B / d log sel_{s,i} = (1/(b T)) * pi_{s,i} * (y_{s,i} - sum_j pi_{s,j} y_{s,j})
    d log sel / d x_EP   = 5 s'(x_EP) / (5 s(EP) + 5 s(TTC) + 2 s(C))    > 0      (s = sigmoid)
    d log sel / d x_C    = 2 s'(x_C)  / (same)                             > 0
    d log sel / d x_lane = s'(x_lane) / s(x_lane)                          > 0      (also NC, DAC: multiplicative gates)

Take a proposal with y^TTC = 0 in a set whose c > 0. `-λ ∇c_B` LOWERS its EP, C and lane logits, and raises those of the TTC-safe proposals.

⇒ The dual acts on the channel that A2 located (51-70 % of the loss, `PK/raw/2026-10-16-safety-head/RESULT.md` §1.1). It also acts on the safety rows.

### 1.4 The dual (projected ascent, one step per optimiser update)

    gbar^k_u   = mean over the accum micro-batches of update u of g^k     (detached floats)
    lambda_k  <- min(lam_max, max(0.0, lambda_k + eta * gbar^k_u))         after o.step() (:1394), before `upd += 1` (:1402)

- Registered values: eps = 0.0, eta = 4.0, λ_max = 10.0, λ0 = 0.0, for both terms.
- λ used during update u is the value from before u's step.

### 1.5 Relation to the fixed penalty (b) = A1's `R' = R - 0.25 (1 - ttc)`

    -lam_pg * J(R') = -lam_pg * J(R) + lam_pg * 0.25 - lam_pg * 0.25 * c^TTC_B      (per set; epo_loss means over sets as c does)

Its gradient therefore equals that of the EPO term plus L_con with λ_TTC = 0.25·lam_pg and λ_NC = 0 (**identity I2**). After the ramp (lam_pg = 1), (b) is this mechanism with η = 0 and λ0_TTC = 0.25.

## 2. New trainer flags. The switch's default DISABLES the mechanism

| flag | type / choices | default | role |
|---|---|---|---|
| `--epo-con` | `{off, g5}` | **off** | the switch. `g5` = constrain TTC and NC against arm A |
| `--epo-con-eps` | float ≥ 0 | 0.0 | slack, in probability units |
| `--epo-con-lr` | float ≥ 0 | 4.0 | η, the dual step |
| `--epo-con-max` | float > 0 | 10.0 | λ_max |
| `--epo-con-lam0` | float in [0, λ_max] | 0.0 | initial λ, the same for both terms |

**Refusals** go in a NEW function `epo_con_refusals(a, ap)`. `epo_refusals` stays byte-unchanged (pinned). It refuses:
- any `--epo-con*` named without `--epo`;
- any `--epo-con-*` named while `--epo-con off`, because the weight would belong to a term that is OFF;
- `--epo-con g5` with η = 0 AND λ0 = 0. That configuration is **inert by construction**: a wired mechanism that can never act.

The gate calls it beside `epo_refusals` (launch_gate.py :388).

**Defaults:**
- The numeric defaults are the registered values. They do not enable anything; only `--epo-con g5` does.
- ⚠️ The profile still requires all five flags **NAMED** on R0d and R1d, and `--epo-con off` NAMED on R1o. A parser default never carries a lever, per CLAUDE.md "check the DEFAULT".

## 3. Argv of each arm (profile `refe_rls1d`, file `eval/rls1d_arms.json`)

**BASE** is R0's banked token argv (HF `Sayood/tanitad-refe-rls1-r0` `final/PASS_a120418a870f.json`, sha256 `24451b62…`). Derive it exactly as `eval/rls1c_arms.json` did: R0c's argv without `--epo-ref armA` and with its own `--out` / `--dump-scores`:

    --ckpt /workspace/data/rls1/base/model_sft4_B_u400.pt --backbone vitl16 --targets /workspace/data/refe_navtrain/train_grow
    --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/rls1/sets_train
    --heldout /workspace/data/rls1/sets_val --heldout-logs /workspace/data/rls1/rl_val_logs.txt --out /workspace/data/rls1/arms/<ARM>
    --pdm-labels-train /workspace/data/rls1/pdm_train --pdm-labels-heldout /workspace/data/rls1/pdm_val --require-pdm --b-mode listnet --lam 0 --lr 1e-4
    --max-updates 1000 --epochs 4 --workers 3 --seed <S> --eval-every 250 --dump-scores /workspace/data/rls1/arms/<ARM>/scores --lane-head
    --teacher-lane-train /workspace/data/rls1/tlane_train --teacher-lane-heldout /workspace/data/rls1/tlane_heldout
    --epo --epo-terms-train /workspace/data/rls1/terms_train --epo-terms-heldout /workspace/data/rls1/terms_val

| arm | argv | seed |
|---|---|---|
| **R0d** | BASE + `--epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0` | 0 |
| **R1d** | the same as R0d | 1 |
| **R1o** | BASE + `--epo-con off` | 1 |

**Self-check in the arms file:** BASE with ARM = R0 and S = 0 must hash to R0's token sha256 `24451b62…`. This is the rls1c_arms.json recipe.

## 4. Reachability trace, flag to backward (CLAUDE.md "built, tested, unreachable"). Each hop is checked by a test

1. **argparse**: `ap.add_argument("--epo-con", …)` after `--epo-ref` (:575-576) in `build_parser` (:521).
2. **namespace**: `a.epo_con*` is read in `main()`, and `epo_con_refusals` is called beside :591. `G-DVB` "consumed" is not enough; these are `built` kind (§5).
3. **build**: `dual = EpoDual(comps=("ttc", "nc") if a.epo_con == "g5" else (), eps, lr, max, lam0)` sits after the reward config (:682-690). The data event records `dual.built_facts()` (:704-707).
4. **loss**: inside `train_losses` (:792-800), right after `lB = lB + ep` (:799), comes `if dual.active: ct = epo_con_term(sxB, sxA.detach(), tg, sel_score, a.epo_temp, a.epo_con_eps, dual.lam, con_last); lB = lB + ct`.
   - `train_losses` keeps its 4-tuple return. Its parts go to a `con_last` dict, as `epo_last` does.
   - So the loop, the speed probe (:1342) and the G11 probe are unchanged callers. They all exercise the term.
5. **backward**: `((lA + lB) / a.accum).backward()` (:1380), unchanged.
6. **dual step**: `dual.step(con_acc)` after the optimiser loop (:1390-1395) and before `upd += 1` (:1402). `con_acc` accumulates `con_last` per micro-batch, beside `epo_acc` (:1385-1386).
7. **observe**: the train rows every 25 updates (:1404-1416) gain `con`, `con_lam_ttc`, `con_lam_nc`, `con_g_ttc`, `con_g_nc`, `con_cA_ttc`, `con_cB_ttc`, `con_cA_nc`, `con_cB_nc`, `con_p1_gap_ttc` and `con_cap_hits`.

## 5. Gate entries (`launch_gate.py`, `gate_checks.py`)

### 5.1 G-DVB

- `gate_checks.DVB_KINDS_SCORER` gains `"epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0"`, all of kind **built**.
- A new `SCORER_EPO_CON_KEYS = ("mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref")` names the data event's `epo_con` block, which every `--epo` run writes, R1o's OFF included.
- `scorer_dvb_rows` compares the argv with BUILT facts, never with an echo of the namespace:
  - **mode**: `"off"` or `"g5"`, derived from the built `comps` tuple.
  - **cols**: must equal `(COMPONENTS.index("ttc"), COMPONENTS.index("nc")) == (3, 0)` when on.
  - **ref**: read by calling the reference selector on sentinels, as `epo_reference_built` does. It must read `"armA_detached"`.
  - **eps, lr, max, lam0**: read off the `EpoDual` object.
- The regression arms `unknown_flag` and `stale_registry` must FAIL on a sixth `--epo-con-*` flag with no entry.

### 5.2 G-PREFLIGHT: new gate G13 `epo_con_live_and_identity`

- Add it to `SCORER_GATES`.
- `scorer_required_gates` requires it iff `--epo` AND `--epo-con != off`. A G13 present on R1o is a stray, i.e. another configuration.
- `judge_g13` re-judges it from raw fields. Every condition below must hold:
  - **on**: the L_con-only backward at λ = (1, 1) through **`train_losses`**, from a fresh forward, gives a finite, non-zero gradient on every scorer module of arm B.
  - **zero controls**: λ = (0, 0) and the switch off both give gradients of EXACTLY 0.0 on every module, and a term value of EXACTLY 0.0.
  - **A untouched**: arm A's per-module gradient from L_con is EXACTLY 0.0. A's own BCE gradient in the same probe is non-zero, which proves the A probe can read non-zero.
  - **identity at step 0**: |c_A - c_B| ≤ `kl_tol`, the tolerance G11 already uses (0.0, or 1e-2 for the non-native lane case), so g = -eps.
  - **dual reach**: a synthetic sequence fed through the run's own `dual.step` gives λ = 0.3125 exactly (violation 2^-7, η 4, 10 steps). A negative violation floors λ at 0.0 exactly, and a large one caps it at λ_max exactly.
  - **checkpoint**: the `EPO_CON` block round-trips through `torch.save` / `torch.load` bit-equal.
  - **no regression switch**: the gate was not produced under `REFE_RL_MUTATE_CON_*`.

### 5.3 G-LIVE: the metric rows must emit the terms

- R0d / R1d: `RLS1_EPO_TERMS + ("con", "con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc")`.
- R1o: `RLS1_EPO_TERMS`. The `con*` columns are absent, and their presence on R1o is a FAIL.

### 5.4 Profile `refe_rls1d`

It is `refe_rls1` with these changes:
- arm table `{R0d: 0, R1d: 1, R1o: 1}`, all of class R;
- per-arm registered EPO values: RLS1 plus `epo_con g5 / 0 / 4 / 10 / 0` for R0d and R1d, and `epo_con off` for R1o;
- per-arm must-name lists;
- `RLS1D_SUITE = RLS1C_SUITE + [test_rl_safeobj.py ZZTEST_PASS]`.

`refe_rls1`, `refe_rls1b` and `refe_rls1c` must now **refuse** any `--epo-con` value other than off. Every argv they accepted before stays accepted.

## 6. Identities, controls and mutations (`refe/test_rl_safeobj.py`, CPU, prints a RED table)

| id | assertion | strength |
|---|---|---|
| I1 | `--epo-con g5` with λ ≡ 0 (λ0 0, η 0; called directly, since the CLI refuses it as inert) gives a loss and gradient on arm B **bit-identical** to `--epo-con off` on the same micro-batch | analytic identity |
| I2 | at lam_pg = 1, the gradient of [EPO with R' = R - 0.25 (1 - y^TTC)] equals [EPO with R] + [L_con at λ_TTC 0.25, λ_NC 0] to ≤ 1e-6 relative (fp32) | ties (b) to the same code path |
| I3 | the gradient at λ = 0.5 is exactly half the gradient at λ = 1 (a power-of-two scale, so exact) | a mask that admits only all or nothing is a switch; this proves linear dependence |
| I4 | the dual arithmetic of §5.2 is exact (0.3125, floor 0.0, cap λ_max) | exact |
| I5 | the patched trainer with SAFEOBJ_FILE_REVERSALS reversed has git blob `7c8071ea…` (the R0c-tip trainer), and through R0C_FILE_REVERSALS gives `d34d5cf6…` | pinned / additive |
| C1 | arm A's gradient from L_con is EXACTLY 0.0; arm A's BCE gradient is non-zero | control that can read non-zero |

**Mutations. Each must turn its check RED:**

| mutation | what it breaks | checks that go RED |
|---|---|---|
| M1 `REFE_RL_MUTATE_CON_DETACH=1` | c_B is computed from `sxB.detach()` | G13 "on" (zero gradient) |
| M2 `REFE_RL_MUTATE_CON_ATTACH_A=1` | arm A is not detached | C1 / G13 "A untouched" |
| M3 source | delete `lB = lB + ct` in `train_losses` | G13 "on" (the probe runs through the caller) |
| M4 source | delete the `dual.step(...)` call in the loop | dual reach via the loop closure |
| M5 source | switch default `g5` instead of `off` | I5 and the R0c/S1b pins; the gate's must-name check on R1o |
| M6 source | column 2 (EP) instead of 3 | G-DVB `cols` and the `COMPONENTS` read |
| M7 gate | drop `epo_con` from `DVB_KINDS_SCORER` | G-DVB `stale_registry` |
| M8 gate | `refe_rls1d` suite without `test_rl_safeobj.py` | G-SUITE |
| M9 chain | R1o's K3 monitor without `--observe-only` | test_rls1_chain: the control would be killable |

## 7. Checkpoint and resume

* **MEASURED by reading source: `scorer_finetune.py` has NO resume path.**
  * The loop (:1375-1428) starts at update 0.
  * `scorer_states_last.pt` holds the A and B states plus `update`, with no optimiser state.
  * So a crash restarts the arm from update 0, and λ restarts at λ0. That is consistent. K6 is unchanged.
* **The dual state is still written:**
  * to `scorer_states_last.pt` (:1420-1423), as `"EPO_CON": {"lam": {"ttc": .., "nc": ..}, "update": u, "cap_hits": n}`;
  * to every full checkpoint's `meta["epo_con"]` (`save_full`, :981-1001).
  * Purpose: the reads can audit λ at each dump, and any future resume must restore it. The G13 checkpoint field pins the round trip.
* **Inference is unchanged.**
  * The constraint is training-only.
  * The deployed scorer is arm B, read by the unchanged `ckpt_io` / planner loader.
  * The `scores_{tag}_{A,B}.npz` dumps keep their format, so `rl_val_eval.py`, the K3 monitor and the navtest scorer need no change.

## 8. Cost per arm (A40-h)

| item | value | class / source |
|---|---|---|
| 1,000 updates | 3.4 A40-h | MEASURED, R0 (`PK/raw/2026-10-14-rls1-r0-kill/README.md:44`) |
| C0 and R0 | 3.4-3.5 h | INHERITED, `PK/eval/PREREG_RL_S1.md` §10.3 |
| R0c pace | 10.63 s/update average through u500, one 250-check evaluation included | MEASURED, `PK/raw/2026-10-16-r0c-read/sft.jsonl:26` |
| constraint overhead | two log-softmaxes over [8, 64] plus two weighted sums per micro-batch: < 1 % | ESTIMATED; MEASURED by the preflight speed probe (`s_per_microbatch`) before launch |
| **planned arm** | **3.5 A40-h** each for R0d, R1d and R1o; about 1.8 if K3-killed after u500 | — |
| **total** | **≤ 10.5 A40-h** | — |
| optional R0o | +3.5 | PI decision |
| navtest reads | dev box, 0 A40-h | cost UNVERIFIED |
