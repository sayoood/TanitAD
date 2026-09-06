# RESULT — `--no-strategic` is WIRED, TESTED and PRE-REGISTERED (2026-09-06, 0 GPU)

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-strategic-bypass/`
**Deliverable** a durable flag + its proofs + a pre-registered arm. **Not** a trained model.
**Compute** 0 GPU. ⛔ **The A40 (refcv5 → ≈2026-09-08 07:33 UTC) was NOT touched, and nothing was
written into any `refcv5-*` path.**

**Answer to the question asked: YES — the next experiment can run without the strategic layer.**
The flag is wired end to end, the bypass is proven in both directions, and the launch command is
`SPEC.md` §8. It is gated on **A40 availability only**.

---

## 1. What was built

`--no-strategic` — a **pure forward gate with ZERO new parameters**, spanning three files:

| file | change |
|---|---|
| `stack/tanitad/refs/refc.py` | `RefCConfig.no_strategic` (new field, default `False`); `ctx` withheld from the decoder (**S-BYPASS-1**, operative seam); `route_prior` forced `None` (**S-BYPASS-3**, selection seam) |
| `stack/tanitad/refs/refc_v3.py` | one guard read from `cfg.core` (one source of truth); the E4 strategic-goal FiLM on `z_tac` skipped (**S-BYPASS-2**, tactical seam); the E15 goal-point block gated (**S-BYPASS-4**) |
| `stack/scripts/refc_v3_train.py` | the `--no-strategic` CLI flag; pinned onto the config for BOTH arms; **stamped into `config.json`** as `no_strategic` **and** `goal_str_loss_applied`; the strategic goal aux loss dropped; the LAN preflight skipped **with its reason printed**; a preflight line naming the arm |

⭐ **Why zero new parameters is the load-bearing design choice.** A flag that can only *remove a
term from a sum* makes both required properties **provable rather than asserted**: ON and OFF builds
have **bit-identical `state_dict` keys and shapes**, so every banked checkpoint loads strictly into
both; and OFF cannot perturb a today-arm because no branch it takes is new.

⛔ **BYPASS, NEVER DELETE.** `hierarchy` stays `True`; `StrategicCtx`, `ctx_to_cond`,
`str_goal_head`, `gstr_embed`, `gstr_film`, `nav_to_str`, `ego_to_str` are all still constructed and
still in `state_dict`. §4 measures that the delete route still breaks strict loads today.

---

## 2. ⛔ The record can name the arm — because `hierarchy` alone no longer can

MEASURED (`raw/` and the preflight logs), through the trainer's own `_seam_stamp`:

| | `hier` | `hierarchy` | `no_strategic` | `goal_str` | `goal_str_loss_applied` |
|---|---|---|---|---|---|
| **OFF (refcv4b today)** | true | **true** | **false** | true | **true** |
| **ON (`--no-strategic`)** | true | **true** | **true** | true | **false** |

⭐ **`hierarchy` reads `true` in BOTH rows — that is the bypass-never-delete design working, and it
is exactly why the two new keys had to exist.** Without them a finished run could not answer *"did
the strategic layer reach the plan?"* from its own record — the precise failure `SEAM_STATE.md`
MEASURED for six pre-existing seams, where the live arm was reconstructible only by knowing what
`refc_v3.py` forces.

Preflight, MEASURED verbatim (`raw/PREFLIGHT_no_strategic_{ON,OFF}.log`, both exit 0, both `PASS`):

```
ON : strategic_layer=BYPASSED (--no-strategic) | ctx->decoder=OFF | g_str->tactical_FiLM=OFF |
     route_readout->selection=OFF | goal_str_loss=NOT APPLIED | nav->operative=ON(measurement) |
     nav->tactical=ON(E13)
OFF: strategic_layer=ACTIVE | ctx->decoder=ON | g_str->tactical_FiLM=ON |
     route_readout->selection=OFF | goal_str_loss=APPLIED | nav->operative=ON(measurement) |
     nav->tactical=ON(E13)
```

In the ON run `goal_str` is **absent from the loss dict** and the LAN preflight is skipped with its
reason printed; in the OFF run the LAN preflight runs normally and its own control fires
(`goal_str=0.4641 live / 0.0000 under the +inf-guard`, *"control able to fail: True"*).

---

## 3. ⭐⭐ THE MUTATION PROOF — both directions, with a control that MUST be able to fail

⛔ **Guards need MUTATION, not inspection.** Everything below perturbs the strategic parameters and
reads the plan; nothing below asserts a branch by reading the source.

**Method.** Tiny hier build, `model.eval()` (the decoder's noise is `zeros_like` at eval and
`ego_dropout` is training-only, so the forward is deterministic). Record the plan; overwrite **all
18 strategic tensors (5,710 elements)** with `N(0, 3)`; record again; compare **bitwise**.

**The compared surface is 28 tensors**, and it deliberately includes the tactical state — the first
version of this probe listed only the core-side keys, passed, and **never looked at `z_tac`, the
thing E4 exists to move**. A proof that does not look at the seam it is proving is not a proof.
Surface: `traj`, `wp_seq`, `sel_idx`, `sel_idx_base`, `sel_score`, `sel_score_v3`, `anchor_logits`,
`refined_logits`, `offset`, `anchor_traj`, `traj_base`, `goal_dist`, `goal_gate_value`,
`goal_score_absmean`, `maneuver_logits`, `lat_logits`, `lon_logits`, `lat_decision`, `lon_decision`,
`maneuver_decision`, **`z_tac`**, **`lat_logits_tac`**, **`lon_logits_tac`**, **`g_tac`**,
`g_tac_delta`, `goal_point_tac`, `goal_point_free`, `bank_speed_pred`.

| | mutation of the 18 strategic tensors | verdict |
|---|---|---|
| **P1 — ON (`no_strategic=True`)** | **0 of 28 surfaces changed — BIT-IDENTICAL** | ⭐ **PASS** |
| **P1c — OFF (control, same mutation)** | **20 of 28 surfaces changed** — `z_tac` max\|Δ\| **156.14**, `traj` **44.51**, `g_tac` **128.85**, `lat_logits_tac` **67.73**, `sel_score_v3` **13.56**, `sel_idx` moves by 6 | ⭐ **PASS — the probe is PROVEN able to fail** |

⭐ **P1c is the half that makes P1 mean anything.** Without it, "nothing changed" is
indistinguishable from a probe that cannot see anything — the failure mode that has produced
confident wrong numbers in this programme repeatedly.

**P2 — under the bypass, does the nav command still reach the planners?**

| | result | verdict |
|---|---|---|
| **P2a OPERATIVE** | `nav_cmd 0 → 2` at init moves **11 surfaces** (`traj`, `anchor_logits`, `sel_score`, `sel_score_v3`, `offset`, `goal_dist`, …) — nav reaches the decoder condition through the measurement vector, live from step 0 | ⭐ **PASS** |
| **P2b TACTICAL** | E13's `nav_to_tac` is **ZERO-INIT**, so nav→tactical is bit-inert at init; made non-zero, `nav_cmd 0 → 2` moves **`z_tac`, `lat_logits_tac`, `lon_logits_tac`, `g_tac`, `g_tac_delta`, `goal_point_tac`, `bank_speed_pred`** | ⭐ **PASS — pathway present** |

⚠️ **Stated honestly:** P2b proves the **pathway exists**, not that a trained arm uses it. Because
`nav_to_tac` is zero-init, "nav reaches tactical" is a claim about the **trained** model, and an
untrained one is evidence for neither side.

Artifacts: `raw/MUTATION_PROOF.log`, `raw/MUTATION_PROOF.json`, `scripts/mutation_proof.py` (exit 0).

---

## 4. ⭐⭐ STRICT LOAD ON THE REAL BANKED CHECKPOINT — with the negative control

Artifact: **`refcv4b-b1-v72-40k` `ckpt_step9500.pt`**, 1,285,301,425 B, dev-box copy at
`C:\Users\Admin\navcomp\ckpt\`, own `step` field **9500**, **551 tensors**. The model was rebuilt
through the trainer's **own** `build_parser()` + `_pin_trainer_cfg` driven by the checkpoint's own
`config.json['argv']` (verified to match `MODEL_REGISTRY.md` §4.6 byte for byte), so this is the
build the trainer would make, not a hand-assembled one.

| build | `hierarchy` | `no_strategic` | state_dict keys | params | strict load |
|---|---|---|---|---|---|
| **OFF (today)** | true | false | **551** | **107,058,488** | **missing 0, unexpected 0 — PASS** |
| **ON (`--no-strategic`)** | true | true | **551** | **107,058,488** | **missing 0, unexpected 0 — PASS** |

**Key sets identical ON vs OFF: symmetric difference 0.**

⛔ **NEGATIVE CONTROL — the DELETE route still breaks, today.** Building with `hierarchy=False`
(the tempting "just turn the hierarchy off") **FAILS** `load_state_dict(..., strict=True)` with
`Unexpected key(s) in state_dict: "goal_gate", "phi_tac.in_proj.weight", …`. ⇒ the requirement is
**real and currently binding**, and the bypass is not passing it trivially.

Artifacts: `raw/STRICT_LOAD.log`, `raw/STRICT_LOAD.json`, `scripts/strict_load.py` (exit 0).

---

## 5. ⭐⭐ OFF IS BYTE-IDENTICAL TO TODAY — measured against the pre-patch source, not asserted

Two **separate trees**, two **separate interpreters**, the **same** `dump_forward.py`
(md5 `7b979f016c52267444333541171c4aee` in both — checked):

* **BASELINE** = the pre-patch files, restored from the live repo and md5-verified
  (`refc.py` `0543ad0e…`, `refc_v3.py` `d1548c57…`, `refc_v3_train.py` `4801b648…`;
  `grep -c no_strategic` = **0**).
* **PATCHED** = the same tree with the three patched files
  (`refc.py` `c912896e…`, `refc_v3.py` `77d46171…`, `refc_v3_train.py` `94de6f2a…`;
  `grep -c no_strategic` = **3** in `refc.py`).

Same seed, same inputs, flag left at its default (OFF), forward run at **all four `nav_cmd`
values**, and the whole `state_dict` compared as well — because if the RNG stream had moved, "byte
identical" would be measuring the wrong thing.

| | |
|---|---|
| keys only in BASE / only in PATCHED | **0 / 0** |
| compared | **341 tensors** = 193 `state_dict` entries + 148 forward outputs over 4 nav values |
| **BITWISE DIFFERENT** | ⭐ **0** |

⭐ **P3 PASS.** With the flag off, the patched source is bit-for-bit the pre-patch source.

Artifact: `raw/BYTE_IDENTITY.log`, `scripts/byte_identity.py`, `scripts/dump_forward.py` (exit 0).

---

## 6. The suite, as a CONTROLLED comparison

The same pytest selection (`-k "refc or v3 or lan or select or tactical or nav"`) run on **both**
trees, failure/error IDs sorted and diffed **in both directions**:

| tree | failure/error IDs |
|---|---|
| BASELINE (pre-patch) | **45** |
| PATCHED | **45** |
| **diff** | ⭐ **EMPTY — no new failures, none fixed** |

On the focused refc/v3 set the result is **106 passed, 1 failed** on both trees, the one failure
being the pre-existing `tests/test_refc_tactical.py::test_tactical_probe_runs_end_to_end`
(a `ModuleNotFoundError`, unrelated). ⚠️ The 45 count includes collection errors from tests that
reach outside `stack/` (`taniteval/`, repo-root paths) which the off-Drive test tree does not carry
— **identical on both sides**, which is the point of running the comparison rather than a single
suite.

Artifacts: `raw/SUITE_IDS_baseline.txt`, `raw/SUITE_IDS_patched.txt`, `raw/SUITE_ID_DIFF.txt` (0 lines).

---

## 7. What tactical and operative now receive — the plain statement

* **OPERATIVE** gets the conv-map tokens, the tactical latent (E7), the tactical goal (E9), and the
  measurement vector `m = measurement([v0/10, nav_onehot(4), ego_valid, (nav_known)])`.
  **The nav command is inside `m` and the bypass does not touch it.** It loses exactly one additive
  term on the condition: `ctx_to_cond(ctx)`.
* **TACTICAL** gets the pooled window features, E13's `nav_to_tac(nav_embedding(nav_cmd))` added
  straight into `z_tac_raw`, and E11′'s ego block. It loses exactly the E4 FiLM by `g_str`.
* ⛔⛔ **THE NAV TOKEN CARRIES NO RANGE AND NO TIME.** It is a 4-way categorical — a one-hot at the
  core, a 64-d embedding row at E13. Nothing multiplies it by a metre or a second; there is no
  arc-length and no time-to-manoeuvre. **The lower layers are given a DIRECTION, not a goal
  constraint.** ⚠️ MEASURED on the goal-point surface, a bearing with range stripped is separated
  **WORSE by +2.3632 m**, and an oracle 3-way command's best decoding is *"go straight"* for all
  three classes — so this is the first thing to examine if the arm fails (`SPEC.md` §6, lever 1).
  *(Coordinated with the sibling establishing this from source; not duplicated here.)*

---

## 8. Scope — what this package does NOT establish

1. ⛔ **No driving claim of any kind.** Nothing was trained. Every number above is a **wiring**
   measurement on a tiny build plus one strict load of a banked checkpoint.
2. ⛔ **It does not show the bypass HELPS.** The motivating observation — route head κ **0.4852**,
   and clip `73e750eb` frames 032/034 where the head says RIGHT (0.68→0.73) while nav says FOLLOW
   and GT is LEFT — is a **hypothesis about a mechanism**, and a sibling owns the decisive
   diagnostic. This package deliberately does **not** pre-empt its verdict.
3. ⚠️ **P2b proves a pathway, not a trained behaviour** (E13 is zero-init).
4. ⛔ **The one-seed problem is NOT solved by this package.** `SPEC.md` §4 pre-commits the replicate
   arm and pre-commits the `NECESSARY-NOT-SUFFICIENT` stamp if only one arm is funded.
5. ⚠️ The dev-box test tree is an md5-verified mirror of the repo's `stack/{tanitad,scripts,tests}`
   `.py` files (805 files, 0 extras) — G: cannot run the stack in place.

---

## 9. Deliverable manifest

| artifact | location |
|---|---|
| `refc.py` (S-BYPASS-1/3 + the config field) | **repo** `stack/tanitad/refs/refc.py` |
| `refc_v3.py` (S-BYPASS-2/4) | **repo** `stack/tanitad/refs/refc_v3.py` |
| `refc_v3_train.py` (CLI, pin, stamp, loss, preflight) | **repo** `stack/scripts/refc_v3_train.py` |
| `SPEC.md` (pre-registration + launch command) | **repo** this package |
| `RESULT.md` (this file) | **repo** this package |
| mutation / strict-load / byte-identity / dump scripts | **repo** this package `scripts/` |
| proof logs + JSON, suite ID lists + diff | **repo** this package `raw/` |
| the banked checkpoint used for the strict load | **dev box** `C:\Users\Admin\navcomp\ckpt\ckpt_step9500.pt` (not copied into the repo — 1.29 GB) |
| the two off-Drive test trees | **dev box** `C:\Users\Admin\tanitad-stratbypass{,-baseline}\` (working copies, not deliverables) |

⭐ **Escalation, not a note in a README:** the arm in `SPEC.md` §8 needs **A40 time after
≈2026-09-08 07:33 UTC** (~44.5 h), and **~89 h if the PI funds the replicate arm that `SPEC.md` §4
requires for a lever claim**. That is a **PI compute decision** and it is the only thing standing
between this flag and its result.
