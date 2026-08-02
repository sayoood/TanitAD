# v1's training loss is UNBUILDABLE from HEAD — and it changes how v1arch must be read

**Resolved 2026-08-02** (backlog item 5). The open question was: `--jerk-weight 0.02` and
`--aux-accel` appear in v1's registry command but not in the committed trainer's arg list — what
was v1 *actually* trained with?

**Answer: the flags never existed in the tracked trainer, the jerk term exists but is reachable
ONLY through `--v2`, and `aux-accel` does not exist in the flagship trainer at all.**

---

## 1. The flags were never in the tracked trainer

**MEASURED** — `git log -S` over the full history of `stack/scripts/train_flagship4b.py`:

| search | result |
|---|---|
| `jerk-weight` in the trainer | **no commits** |
| `aux-accel` in the trainer | **no commits** |
| `jerk-weight` anywhere in the repo | only **prose** commits (registry, prereg, research notes) |

⇒ They were flags of a **pod-side trainer that was never committed**, exactly as
`MODEL_REGISTRY.md` line 177 already warned. That part is confirmed, not new.

## 2. ⭐ What IS new: the jerk penalty exists — locked behind `--v2`

The committed loss *does* implement a jerk penalty (`flagship_losses.py:303-311`), a 3rd-difference
penalty on predicted waypoint paths. But its weight is read as:

```python
w_jerk = float(getattr(cfg, "v2_traj_jerk", 0.0))     # flagship_losses.py:306
```

and `config.py:201` defaults `v2_traj_jerk: float = 0.0`. The **only** site in the entire stack
that sets it non-zero is `train_flagship4b.py:356`:

```python
cfg.v2_ego_dropout = 0.25
cfg.v2_fa_dropout  = 0.3
cfg.v2_goal_decode = True
cfg.v2_nav_dropout = 0.5
cfg.v2_traj_jerk   = 0.02        # <-- HERE, inside the --v2 block
cfg.v2_gated_intent = True
cfg.v2_anchor_tactical = True
```

⇒ **The jerk penalty is reachable only by taking the whole ten-lever `--v2` pack.**

## 3. `aux-accel` is not in the flagship trainer at all

**MEASURED**, second-location probe as the absence rule requires. `aux_accel` appears only in:

- `stack/tanitad/refs/refb.py:127` — **REF-B's** own auxiliary head, a different model
- `stack/experiments/reset-speed4b/refa_train_plus.py` — an **archived** REF-A experiment

⇒ Nothing in the flagship path. The v1 pod-side trainer had an aux longitudinal-accel head that
**has no counterpart in HEAD**.

---

## 4. ⛔ THE CONSEQUENCE — v1arch is not the matched rebuild we described it as

`flagship-v1arch-v2bal-30k` was launched to answer the PI's question — *"train the original v1 with
more data"* — with `--v2` deliberately **omitted** so the corpus would be the only axis that moved.
That omission is correct for excluding the nine architecture levers. But it also **removes the jerk
penalty v1 had**, and there is no way to add it back without re-adding all ten.

**MEASURED** — v1arch's own `config.json` loss weights contain **no jerk and no accel term**:

```
pred 1.0 · tacpred 0.5 · roll 0.5 · goal 0.5 · wp 1.0 · man 0.5 · route 0.5
route_vis 0.3 · invdyn 2.0 · fwd 1.0 · sigreg 0.1 · inv 0.5 · decorr 0.05
```

⇒ **v1arch = "v1's architecture, MINUS the jerk penalty, MINUS aux-accel, on the v2bal corpus."**
Not "v1 with more data".

⭐ **It is a trilemma with no clean corner**, and this should be stated plainly rather than
discovered later:

| build | jerk | the nine other v2 levers | is it v1? |
|---|---|---|---|
| no `--v2` (**v1arch, running**) | ❌ off | ❌ off | closest available, but loss ≠ v1 |
| `--v2` (**v2corpus**) | ✅ 0.02 | ⚠️ all on | no — that is the v2 line |
| v1's actual config | ✅ 0.02 | ❌ off | ⛔ **not buildable from HEAD** |

⚠️ **This is not a reason to stop or restart v1arch.** It remains the cleanest corpus contrast we
can construct, and the nine levers it correctly excludes are the ones that demonstrably broke the
tactical brain (κ 0.253 → 0.0072). **It is a reason to label the arm honestly.**

### Does the missing jerk term matter numerically?

It plausibly does, because we **measure** the quantity it targets. `tms` (trajectory smoothness) is
in every eval line: v1 **0.080** · RR-CTL **0.099** · v2corpus **0.150** · RR-20 **0.269**. A
smoothness penalty present in v1 and absent in v1arch is a live confound on any smoothness or
lateral comparison between them. ⚠️ **HYPOTHESIS** — the direction is not measured.

---

## 5. What this closes, and what it opens

✅ **CLOSED:** "what was v1 trained with?" — a pod-side trainer with a jerk penalty and an
aux-accel head, neither reproducible from HEAD as a standalone option.

⭐ **The registry's existing warning is UPGRADED**: it said the flags are *"not in the committed arg
list."* The stronger true statement is **"v1's loss cannot be rebuilt from HEAD at all,"** because
the one term that does exist is welded to nine others.

**OPENED — a cheap, decidable engineering item:** expose `v2_traj_jerk` as a **standalone
`--traj-jerk` flag**, independent of `--v2`. That is a few lines, it is exactly the DATA-vs-ARCH
separation principle applied to a loss term, and it would make a true v1-matched arm buildable for
the first time. ⛔ **Not done in this pass** — it changes the trainer that produced our published
arms, so it belongs in its own scoped change with the sha256 pin updated deliberately.

## Evidence class

| claim | class |
|---|---|
| `--jerk-weight` / `--aux-accel` never in the tracked trainer | **MEASURED (ours)** — `git log -S` over full history, 2026-08-02 |
| jerk penalty exists, gated on `v2_traj_jerk`, default 0.0 | **MEASURED** — `flagship_losses.py:303-311`, `config.py:201` |
| the only non-zero setter is inside the `--v2` block | **MEASURED** — `train_flagship4b.py:356`, full-stack grep |
| `aux_accel` absent from the flagship path | **MEASURED** — two-location probe: only `refs/refb.py` and archived `reset-speed4b/` |
| v1arch's weights carry no jerk/accel term | **MEASURED** — the run's own `config.json`, rescued from newpod |
| "the missing jerk term shifts `tms`" | ⚠️ **HYPOTHESIS** — plausible, direction unmeasured |
