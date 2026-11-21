# Mutation log — the census must be able to go RED, on every leg

**Evidence class: MEASURED (ours).** Every run below is reproducible from
`stack/scripts/v7_vocab_reach_census.py` + `scripts/gradreach_probe.py` in this
package. Rig: dev box, `C:/Users/Admin/venvs/tanitad` (torch 2.11.0+cu128), CPU,
`PYTHONPATH=C:/Users/Admin/tanitad-wt/stack;…/taniteval`, mirror byte-identical
to the repo for every consumer file (md5-verified before and after — see
`RESULT.md` §Manifest).

> ⛔ **A census that cannot go RED proves nothing.** Three legs decide a reach
> verdict, so there are three mutations, one per leg, plus the *other direction*
> — a fix-forward mutation proving the census can also go GREEN when the gap is
> actually closed. A detector stuck at "unreached" would keep reporting
> `D-TACGOAL-1` open after somebody fixed it, which is the same worthlessness
> in the opposite sign.

---

## Baseline (GREEN — the state of the repo at `cb84c07`)

```
class counts: {'TRAINING_SIGNAL': 23, 'INFERENCE_INPUT': 3,
               'AUDIT_OR_METRIC_ONLY': 18, 'UNREACHED': 0, 'NOT_EMITTED': 8}
consumer assertions:
  goal_audit        CONSUMED
  nav_input         CONSUMED
  str_action_ce_v6  CONSUMED
  str_goal_ce_v6    CONSUMED
  tac_lat_ce        CONSUMED
  tac_lon_ce        CONSUMED
  str_ce_refc       NO_CONSUMER  missing=['HEADS["str_goal"]']
  tac_action_ce_v6  NO_CONSUMER  missing=['cross_entropy(out["a_lat"]']
  tac_goal_bce      NO_CONSUMER  missing=['TacGoalEmitter', 'tac_goal_loss']
```
Artifact: `raw/census.json`.

---

## MUT-C0 / MUT-C0' — leg 3 (GRADIENT), on the LIVE arm's own argv

`scripts/gradreach_probe.py` builds the real `RefCV3Model` through
`refc_v3_train.build_parser()` + `_pin_trainer_cfg()` from **the recorded argv of
the live refcv5-v2 run**, runs the real `compute_losses_v3`, backwards the
trainer's **own** `losses["loss"]` (not a sum of everything that carries grad —
that would add gradient paths the run does not have), and reads `p.grad` per
module.

**GREEN — as the live arm actually is:**
```
  lat_head_tac          params=      264 grad_none=0/2 |g|=0.312936  GRADIENT REACHES
  nav_to_tac            params=    2,080 grad_none=0/2 |g|=115.17    GRADIENT REACHES
  tac_goal_tok_head     params=      726 grad_none=2/2 |g|=0         NOT WIRED (all grads None)
```
**RED — `MUTATE=1`, a loss term now touches the head's own logits:**
```
  tac_goal_tok_head     params=      726 grad_none=0/2 |g|=1452      GRADIENT REACHES
```
Artifacts: `raw/gradreach_live.json`, `raw/gradreach_mutated.json`.

⚠️ `params=726` is the **smoke** config used to keep the probe on CPU. The live
`base` config's own `config.json` records `param_breakdown.tac_goal_tok_head =
11286`, and `seams.tac_goal_tok_head = {requested: true, cfg: true, built:
true}`. The *wiring* verdict is a property of the loss graph, not of the width.

---

## MUT-A — leg 2 (CONSUMER): detach a head from its loss

Historical defect re-introduced in a scratch copy of `refc_v3_train.py`:

```python
- F.cross_entropy(out["lat_logits_tac"], lat_t)
+ torch.zeros((), device=device)   # MUTATED: CE removed
```

```
tac_lat_ce           NO_CONSUMER  missing=['F.cross_entropy(out["lat_logits_tac"], lat_t)']
class counts: {'TRAINING_SIGNAL': 16, 'INFERENCE_INPUT': 3,
               'AUDIT_OR_METRIC_ONLY': 22, 'UNREACHED': 3, 'NOT_EMITTED': 8}
```

**7 tokens changed class** — and the three that fall all the way to `UNREACHED`
are exactly the lateral actions with no second surface:

| token | baseline | mutant |
|---|---|---|
| `LANE_KEEP` | TRAINING_SIGNAL | **UNREACHED** |
| `NUDGE_L` | TRAINING_SIGNAL | **UNREACHED** |
| `NUDGE_R` | TRAINING_SIGNAL | **UNREACHED** |
| `TURN_L` | TRAINING_SIGNAL | AUDIT_OR_METRIC_ONLY |
| `TURN_R` | TRAINING_SIGNAL | AUDIT_OR_METRIC_ONLY |
| `LANE_CHANGE_L` | TRAINING_SIGNAL | AUDIT_OR_METRIC_ONLY |
| `LANE_CHANGE_R` | TRAINING_SIGNAL | AUDIT_OR_METRIC_ONLY |

⭐ The four that land on `AUDIT_OR_METRIC_ONLY` rather than `UNREACHED` are the
tokens whose **string** is both a tactical GOAL and a lateral ACTION. The census
keeps them apart because the surfaces are separate — which is the distinction
`tactical_label_census.py` had to spell out in prose (*"NO (name also a tac_lat
ACTION class)"*).

**Specificity control:** `tac_lon_ce` stayed `CONSUMED` under this mutation, so
the detector is not merely noisy. Pinned by
`test_the_consumer_detector_actually_fires`.

Artifact: `raw/census_MUT_A.json`.

---

## MUT-B — leg 1 (PROJECTION): detach the head from its vocabulary tuple

In the **consumer** module (`v7_labels`), not the vocabulary:

```python
v7l.HEADS["tac_lat"] = tuple(V7.TACTICAL_LON_ACTIONS_V7)   # was TACTICAL_LAT_ACTIONS_V7
```

```
class counts: {'TRAINING_SIGNAL': 9, 'INFERENCE_INPUT': 3,
               'AUDIT_OR_METRIC_ONLY': 22, 'UNREACHED': 10, 'NOT_EMITTED': 8}
```

**14 tokens changed class.** The seven longitudinal actions fall too, and that is
correct rather than sloppy: `tactical_class_ids` projects **both** axes in one
call, so a lat tuple that cannot index the record's own `a_tac.lat` raises before
the lon id is produced. The blast radius is the real one.

| token | baseline | mutant |
|---|---|---|
| `LANE_KEEP`, `NUDGE_L`, `NUDGE_R` | TRAINING_SIGNAL | **UNREACHED** |
| `TURN_L`, `TURN_R`, `LANE_CHANGE_L`, `LANE_CHANGE_R` | TRAINING_SIGNAL | AUDIT_OR_METRIC_ONLY |
| `ACCELERATE`, `ADAPT_SPEED_FOR_CURVE`, `BRAKE_TO`, `CREEP`, `CRUISE`, `FOLLOW`, `HOLD` | TRAINING_SIGNAL | **UNREACHED** |

Artifact: `raw/census_MUT_B.json`.

---

## MUT-C1 / MUT-C2 — leg 3 isolated, and the fix-forward direction

A scratch trainer in which the two calls a real `D-TACGOAL-1` fix must make
(`tac_goal_loss`, `TacGoalEmitter`) are present, run against each gradient
census:

| run | leg 2 | leg 3 (`tac_goal_tok_head`) | class counts |
|---|---|---|---|
| **MUT-C1** | CONSUMED | `NOT WIRED` (the real one) | `TRAINING_SIGNAL 23 · AUDIT 18` |
| **MUT-C2** | CONSUMED | `GRADIENT REACHES` (mutated) | `TRAINING_SIGNAL 41 · AUDIT 0` |

⇒ **leg 3 is load-bearing**: with the consumer wired but no gradient, the 18
tactical-goal tokens stay `AUDIT_OR_METRIC_ONLY`. And **the census can go
GREEN**: when both hold, all 18 move to `TRAINING_SIGNAL` in the same run.

Artifacts: `raw/census_MUT_C1.json`, `raw/census_MUT_C2.json`.

---

## The test's own mutation controls

`stack/tests/test_v7_vocab_reach_census.py` re-runs MUT-A and the fix-forward
mutation on `tmp_path` copies at every `pytest` invocation, plus one more the
census needs on this mount:

* `test_a_failed_read_is_INCONCLUSIVE_and_never_an_absence` — break only the
  **same-breath control** marker and leave the real markers alone. The surface
  must report `INCONCLUSIVE`, never `NO_CONSUMER`. ⛔ On the G: mount a search
  tool returns *"no matches"* for a file it could not open, so a missing marker
  and an unreadable file are indistinguishable without this control.

11 tests, all green (`pytest -q tests/test_v7_vocab_reach_census.py`).
