# comma2k19 heading — closing the live defect that was the DEFAULT

**Date:** 2026-07-27 · **Agent:** `heading-default` · **HEAD at start:** `8ab5327`
**Host:** dev box only. **pod1 (training, ~21,650/30,000), pod2 (small validation, cgroup 53.9/55.0 GB)
and pod3 were NOT touched** — no SSH, no GPU, no eval on a training host.
**Reads with:** `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` §8 escalations **#1 and #2**,
`Project Steering/RETRACTION_LOG.md` class **C29**, `…/2026-07-27-idm-v3/IDM_V3.md` §4.

**Evidence class + tier on every number.** **Label protocol is stated beside every number**
(`heading_repair` on/off, `v_min`) — two numbers that differ only by protocol are otherwise
indistinguishable on the page, which is the whole reason C29 existed for five days.
**Estimator, wherever an interval is quoted:** `taniteval.ci` episode-cluster bootstrap, B = 2000.
⛔ **`overlapping_holdout_se` is not used anywhere in this work.**

---

## 0. Headline

> **The repair is now the default, and the broken label can no longer be obtained silently.**
> `stack/tanitad/data/comma2k19.py` `DEFAULT_HEADING_MODE` moved
> `enu_velocity` → **`enu_velocity_hold_v1`**, and `enu_velocity` (legacy) now raises
> `LegacyHeadingRefused` unless the caller passes **both** `allow_legacy=True` **and a written
> reason**. Every accident path — a `None`, a missing config key, an unset default — resolves to
> the **repair**. **MEASURED, this file's own artifact:** the legacy path on a standstill fixture
> produces **61.78 rad/s** of yaw rate (physically impossible; the corpus's measured max is
> 15.53 rad/s); the default now produces **0.0**.
>
> **No existing cache changed meaning.** The legacy regime contributes **no cache-key fragment**, so
> every comma cache dir minted before today keeps its **exact** name (`4bce7a330c31` before = after);
> the repaired regime contributes one (`79a902d541f6`), so a repaired build is *structurally* unable
> to land in a legacy dir. Acknowledged-legacy output is **bit-identical** (`max_abs_delta = 0.0`) to
> an **independent reimplementation** of the pre-flip formula.
>
> ⚠️ **And the second job is NOT DONE.** The `idm_head_v1` re-score on its own 9,420-window val
> could not be run on the dev box — §5 states exactly why, and exactly what it needs. **The card's
> `yaw R² 0.010433` stays STALE-PENDING. Nothing was inferred in its place.**

---

## 1. The choice, and why it is BOTH shapes rather than one

The brief offered two defensible shapes. **Neither is sufficient alone**, and the reason is
symmetric — so the implementation is both.

| shape | what it fixes | ⛔ what it leaves open |
|---|---|---|
| **(a) flip the default to the repair** | new builds stop getting broken labels — the live defect | a script whose job is to **reproduce a committed cache** keeps running, **silently**, on *repaired* labels. The two results differ **only by label protocol**, so nothing on the page distinguishes them |
| **(b) keep LEGACY reachable but never silent** | the deliberate case becomes loud and self-documenting | if LEGACY stays the **default**, a caller who passes nothing **still gets broken labels**. Making the default *raise* is not "reachable but not silent", it is *unreachable* — it breaks every existing caller |

**(a)'s residual risk is the program's most-logged failure class**, not a hypothetical: C29 itself
(a number stale for five days because its protocol was not on the page), the `heldout`/`full_set`
blast radius (**27 arms, −6.67 % to +11.69 %, bidirectional**, including a sign flip), the
`observed_frac` correction. Every one is *the same number measured under a different protocol,
with the protocol invisible*.

**⇒ Implemented: (a) AND (b).**

1. `DEFAULT_HEADING_MODE = HEADING_MODE_HOLD`. Silence is now **safe**.
2. `HEADING_MODE_LEGACY` requires `allow_legacy=True` **and** a non-empty written `reason`. This is
   the `models.vision_rank.resolve_vision_rank` discipline, copied deliberately: *a boolean can be
   flipped absent-mindedly, a sentence cannot.* The module ships `LEGACY_HEADING_REASON` so a
   legitimate reproduction does not have to invent wording — but passing it is still an **explicit
   act at the call site**, and it lands in the run log.
3. The **cache key** separates the regimes (`label_params`, the `physicalai.label_params`
   construction). Without this, (a) actively *fails*: a repaired build would load — or overwrite —
   the existing legacy-keyed dir, and the cache would silently stop meaning what it meant.

**Why the opposite default from `physicalai.WHEELBASE_MODE` is admissible here.** PhysicalAI's
legacy regime must stay the default because it *is* the parity key: `physicalai-train-e438721ae894`,
2376 episodes, skip-hash `f09e44db`. **comma2k19 is an explicitly NON-PARITY corpus** —
`stack/tanitad/data/parity.py` classifies it as *"unregistered corpus (comma / cosmos / OOD)"* and
returns `parity=False`. 🔒 **Parity is untouched by this change**: no PhysicalAI code path, cache key
or episode selection is modified. `pytest -q` includes `test_wheelbase_regime.py`, which pins
`label_params() == {}` on the PhysicalAI side, and it is green.

### 1.1 What changed, by file

| file | change |
|---|---|
| `stack/tanitad/data/comma2k19.py` | `DEFAULT_HEADING_MODE` → `HEADING_MODE_HOLD`; new `LegacyHeadingRefused`, `LEGACY_HEADING_REASON`, `resolve_heading_mode()`, `label_params()`, `cache_build_params()`; `actions_and_poses` / `build_episode` / `Comma2k19Dataset` take `heading_mode=None` + `allow_legacy_heading` + `legacy_heading_reason` |
| `stack/tanitad/train/train_worldmodel.py` | the comma branch builds its cache params through `cache_build_params` (not a bare dict), passes the resolved mode to `build_episode`, **prints the protocol into the run log**, and exposes `--comma-heading-mode` / `--comma-legacy-heading-reason`. `_build_datasets` / `train()` forward both; the `mix` and `realmix` recursions forward them too |
| `stack/tests/test_comma_heading_regime.py` | **new, 23 tests** — the flip, the guard firing, the cache-key separation, the reproducibility pin, and the trainer end-to-end |
| `stack/tests/test_comma2k19.py` | `test_heading_mode_default_is_byte_identical` → `test_heading_mode_default_is_the_repair` (its old assertion was *"the default equals legacy"*, which is now false by design) |

`Comma2k19Dataset` resolves **before decoding anything**: a refused legacy request dies in the first
millisecond, not after 40 minutes of video decode. That is asserted with a spy decoder
(`test_dataset_refuses_legacy_BEFORE_decoding_anything`: `assert calls == []`).

---

## 2. ⭐ The guard, DEMONSTRATED FIRING

⛔ **A guard that cannot fire is worse than none** (RETRACTION_LOG class **C13** — several have
shipped here). Every refusal below is exercised **on the input that must trigger it**, and the
result is read out of a JSON artifact produced by *running the shipped code*, not from a test name.

**Raw:** `raw/heading_default_guard.json` · **producer:** `code/heading_default_demo.py`
**Evidence class:** `MEASURED (ours; dev box)`. **Tier:** instrument-grade — a synthetic fixture that
**reproduces** the measured corpus defect; it is not itself a corpus measurement.

### 2.1 The failing direction — the defect reproduced

Fixture: 6 standstill frames (`|v| = 0.01 m/s`, random direction — GNSS noise, not motion) then 24
frames at 10 m/s, `dt = 0.05 s`. This is the real shape of the defect: comma's heading is
`arctan2` of the ENU **velocity**, so it goes wild exactly where the vehicle is *not moving*.

| quantity | `heading_repair` **OFF** (legacy) | `heading_repair` **ON** (the new default) |
|---|---:|---:|
| max \|yaw_rate\| over the standstill run | **61.78 rad/s** (3 540 °/s) | **0.0** |
| physically possible? (\|ω\| ≤ 1.5 rad/s) | ⛔ **no** | ✅ yes |
| moving-part heading | — | **bit-identical** to legacy |
| speed channel | — | **bit-identical** to legacy |
| frames changed | — | **6**, of which **0** are above `v_min` 0.5 m/s |

**The repair repairs; it does not smooth.** Not one frame at or above the measured threshold moves —
which matters, because the threshold is itself MEASURED (`labels_v3.json → yaw_audit_by_speed`:
**26.27 %** impossible below 0.5 m/s, **0.000 %** in every bin above it).

### 2.2 Every accident mode and every half-acknowledgement

| what a caller does | result |
|---|---|
| `resolve_heading_mode("enu_velocity")` | ⛔ `LegacyHeadingRefused` |
| `…, allow_legacy=True` (flag, no reason) | ⛔ `LegacyHeadingRefused` |
| `…, allow_legacy=True, reason="   "` (blank) | ⛔ `LegacyHeadingRefused` |
| `…, reason="I want the old labels"` (reason, no flag) | ⛔ `LegacyHeadingRefused` |
| `cache_build_params(base, "enu_velocity")` — **the trainer's own call** | ⛔ `LegacyHeadingRefused` |
| `actions_and_poses(…, heading_mode="enu_velocity")` | ⛔ `LegacyHeadingRefused` |
| a typo — `"enu_velocty"` | ⛔ `ValueError` (**not** quietly treated as legacy) |
| `None` / missing config key / unset default | ✅ resolves to **`enu_velocity_hold_v1`** |
| `allow_legacy=True` **and** a written reason | ✅ returns `enu_velocity` |

The refusal message carries the measured mechanism with it — 26.27 % below 0.5 m/s, 0.000 % above,
PhysicalAI zero in every bin, and the deployed head's **+0.0114 → +0.3308** — so the person who hits
it does not have to go find out why. Same construction as `zeros_naive`'s priced-trap refusal.

### 2.3 The trainer, end-to-end

`test_trainer_default_build_does_NOT_reuse_the_legacy_cache_dir` drives the **real**
`train_worldmodel._build_datasets` comma branch (video decode mocked) twice against the **same**
root — once with the default, once with acknowledged legacy — and asserts the two occupy **different
cache dirs**, with the legacy dirs carrying the **unchanged pre-flip key**.
`test_trainer_refuses_legacy_without_a_written_reason` asserts the trainer refuses.

---

## 3. ⭐ The deployed path still reproduces — pinned

Committed comma results were measured on **LEGACY**. Two pins, both green:

| pin | assertion | result |
|---|---|---|
| **the label** | acknowledged legacy == an **INDEPENDENT reimplementation** of the pre-flip formula (`arctan2` of ENU velocity, no standstill handling), at strides 1 and 2 | **bit-identical**, `max_abs_delta = 0.0` |
| **the cache dir** | `label_params(LEGACY) == {}` ⇒ the params dict is unperturbed ⇒ the key is unchanged | `4bce7a330c31` **before = after** |
| **the separation** | the repaired regime cannot collide | `79a902d541f6` **≠** `4bce7a330c31` |

The reimplementation is deliberate: calling the same code path on both sides would let one bug
satisfy the pin twice. `test_build_episode_legacy_path_reproduces_bit_identically` extends it
through `build_episode`, i.e. including the `n_stack` alignment.

⚠️ **What the pin does NOT claim.** It pins the *loader*. It does **not** re-verify any committed
comma number end-to-end — those were produced by pod-side pipelines (`idm2_lib`, `idmval_run`) that
build their own windows. Their reproducibility is inherited, not re-measured here. **UNVERIFIED.**

---

## 4. Suite

| suite | result | vs the brief's baseline |
|---|---|---|
| `stack/` `pytest -q` | **1557 passed, 12 skipped**, 2 warnings | baseline 1534/12 → **+23 = exactly the new file**, **zero new skips** |
| `taniteval/` `pytest -q` | see §7 | — |

Run with the project venv (`C:\Users\Admin\venvs\tanitad`); system `python` 3.14 has no pytest.

---

## 5. ⛔ Job 2 — the `idm_head_v1` re-score: NOT DONE, and why

*(filled in below — see §5 in the final revision)*

---

## 6. Stale-pending: what this pass resolves and what it does not

*(filled in below)*

---

## 7. Deliverable manifest

*(filled in below)*
