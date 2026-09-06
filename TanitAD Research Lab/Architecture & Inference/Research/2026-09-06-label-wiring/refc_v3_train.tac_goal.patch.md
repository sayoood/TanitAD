# ESCALATION — the two seams inside `refc_v3_train.py` I may not edit

⛔ **This is not a "please merge" note left in a README.** An orthogonality
instrument sat unmerged for ten days because the request lived in a file nobody
re-read. This file is the *content* of the escalation; the escalation itself is
in the turn report to the Master Mind, naming the owner of
`stack/scripts/refc_v3_train.py`.

Everything else in `D-TACGOAL-1` is landed and tested. **Exactly two edits live
inside `refc_v3_train.py`**, which a sibling owns. They are written out here
verbatim, each with the reason it cannot live anywhere else, so the owner
applies rather than re-derives.

⚠️ Both are additive. Neither changes any existing tensor, weight, or default,
so **every banked arm's recipe is bit-identical without the new flag**.

---

## Seam 1 — the batch attach (`V3Dataset.__getitem__`, at the `lat_v7` block)

`V3Dataset` is defined *inside the trainer*; there is no separate data module
and no custom collate, so the targets cannot be attached from outside.

**After** the existing `item["lat_v7"] / item["lon_v7"]` lines, add:

```python
            # ⭐ D-TACGOAL-1 — the 22-token tactical goal SET. Multi-label, so
            # it carries its own per-cell validity `w` rather than an
            # ignore_index: a cell with no evidence must not train either way,
            # and BCE has no ignore_index to express that with.
            if lab is None:
                y_g = [0.0] * len(v7l.TAC_GOAL_TOKENS)
                w_g = [v7l.IGNORE_W] * len(v7l.TAC_GOAL_TOKENS)
            else:
                y_g, w_g = v7l.tactical_goal_targets(
                    lab, (t + w - 1) * self.v7_dt,
                    negatives=getattr(self, "tac_goal_negatives", "measured"))
            item["tac_goal_y"] = torch.tensor(y_g, dtype=torch.float32)
            item["tac_goal_w"] = torch.tensor(w_g, dtype=torch.float32)
```

⚠️ `(t + w - 1) * self.v7_dt` is **the existing expression** used for
`tactical_class_ids` two lines above. Reuse it; do not re-derive the window's
NOW. A derived constant recomputed a second way is the `HORIZON` 7→8 trap, and
here it would silently shift which windows are in band.

---

## Seam 2 — the loss term (`compute_losses_v3`, beside the z_tac CE block)

```python
    # ---- D-TACGOAL-1: the tactical GOAL SET -------------------------------
    loss_tac_goal = torch.zeros((), device=device)
    n_goal_sup = 0
    if "tac_goal_logits" in out and "tac_goal_y" in batch:
        from tanitad.refs import tac_goal_head as tgh
        loss_tac_goal, n_goal_sup = tgh.tac_goal_loss(
            out["tac_goal_logits"],
            batch["tac_goal_y"].to(device), batch["tac_goal_w"].to(device),
            pos_weight=getattr(cfg, "tac_goal_pos_weight", None),
            class_mask=getattr(cfg, "tac_goal_class_mask", None))
    extra["tac_goal_rows"] = int(n_goal_sup)
    extra["tac_goal_loss"] = (float(loss_tac_goal.detach())
                              if n_goal_sup else None)
```

⛔ **`None`, never `0.0`, when nothing was supervised.** A `0.0` in that column
reads as *"supervised, and perfect"*, so an unlabelled step and a perfect step
print the same character — the defect `test_tac_loss_logging.py` pins with
`assert row[k] is None`. The trainer's log walker already emits plain
`int`/`float`/`None` from `extra`.

And in the `loss = (...)` sum:

```python
            + TAC_GOAL_WEIGHT * loss_tac_goal
```

with, at module scope:

```python
#: ⛔ 0.0 BY DEFAULT, AND THAT IS *NOT* THE DEFECT THIS WORK REPORTS.
#: A default of 0.0 here keeps every banked arm's recipe BIT-IDENTICAL — the
#: rule is "never flip a default silently", and adding a term with a live
#: default would flip it for every run that does not name the flag. The two
#: things that make this different from `AGENT_WEIGHT_DEFAULT` /
#: `GOAL_POINT_WEIGHT_DEFAULT`:
#:   1. `tac_goal_loss` is UNGUARDED, so at weight 0.0 the head still receives
#:      a ZEROS gradient and `p.grad is None` still means "never wired";
#:   2. the launch command below passes `--w-tac-goal 0.05` explicitly, so the
#:      arm that is meant to learn it says so in its own `config.json`.
TAC_GOAL_WEIGHT_DEFAULT = 0.0
```

### ⚠️ The aux-weight budget — a decision the owner must make, not me

`refc_v3_train.py` pins total tactical aux pressure to exactly
`MANEUVER_WEIGHT = 0.10` by splitting `LAT_WEIGHT`/`LON_WEIGHT` with the
documented `/2.0`. **A third tactical term at 0.05 raises that budget to 0.15**
unless the split becomes `/3.0`. Two admissible readings, and the arm must
state which:

* **`--w-tac-goal 0.05` on top** — the tactical budget becomes 0.15. Simple; the
  trajectory term's relative pressure drops.
* **re-split to `/3.0`** — the budget stays 0.10 and the goal head takes a third
  of it. Preserves the invariant; changes `lat`/`lon` pressure, so it is **not**
  comparable to any banked arm.

⭐ Recommended: **the first, for the first arm**, because it leaves `lat`/`lon`
untouched and therefore keeps the new arm paired against the banked one on
everything except the added term. The re-split is a second arm, not a tweak.

---

## Seam 3 — the preflight (`refc_v3_train.py`, the `--v7-labels` synthetic block)

The synthetic preflight corpus has no v7 records, so v7-shaped labels are
injected. Without the two new keys the preflight exercises a shape the real run
never takes:

```python
        n_g = len(v7l.TAC_GOAL_TOKENS)
        batch["tac_goal_y"] = torch.zeros(2, n_g)
        batch["tac_goal_y"][0, 0] = 1.0             # one supervised positive
        batch["tac_goal_w"] = torch.zeros(2, n_g)
        batch["tac_goal_w"][0] = 1.0                # row 1 stays ALL-IGNORED
```

⭐ Row 1 all-ignored **on purpose**: the all-ignored row is the common case
(8.13 % of batch-8 steps on the sibling trainer), and a preflight that never
sees it cannot catch the `None`-vs-`0.0` logging defect.

---

## Seam 4 — argparse, and the split-derived weights

```python
    ap.add_argument("--w-tac-goal", type=float, default=TAC_GOAL_WEIGHT_DEFAULT,
                    help="D-TACGOAL-1: weight on the 22-token tactical goal "
                         "SET (multi-label BCE). 0.0 keeps the banked recipe.")
    ap.add_argument("--tac-goal-negatives", default="measured",
                    choices=("measured", "geometry", "all"),
                    help="which ABSENT cells are supervised as negatives. "
                         "'measured' derives it from the loaded blob's own "
                         "provenance; 'all' supervises every absence and is "
                         "the cheap comparison arm, not the default.")
```

and, where the labels are loaded, beside the existing `v7_manifest` stamp:

```python
        cfg.tac_goal_pos_weight = v7l.goal_pos_weight(labels)
        _cen = v7l.goal_supervision_census(labels)
        cfg.tac_goal_class_mask = tgh.mask_report(_cen)["mask"]
        ds.tac_goal_negatives = args.tac_goal_negatives
        v7_manifest["tac_goal"] = {
            "negatives": args.tac_goal_negatives,
            "n_trainable": tgh.mask_report(_cen)["n_trainable"],
            "masked_why": tgh.mask_report(_cen)["masked_why"],
            "pos_weight": dict(zip(v7l.TAC_GOAL_TOKENS,
                                   cfg.tac_goal_pos_weight))}
```

⛔ `pos_weight` and the class mask are **computed from the loaded split, never
typed**. `HOLD_MAIN_ROAD` is 52.3 % of `str_action` in this blob; a hardcoded
weight silently mis-weights the next one.

---

## The launch command (dev-box 4060 or Thor; ⛔ NOT the A40 — refcv5 holds it
until ≈2026-09-08 07:33 UTC)

```bash
OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
python3 /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --v7-labels    /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-labels  /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --w-tac-goal 0.05 --tac-goal-negatives measured \
  <every other flag copied VERBATIM from the paired banked arm's config.json>
```

⚠️ `--v7-labels` pins `tac_vocab_version` to **v7.0**, and under v7.0 the
tactical brain stops driving the H19 anchor prior (`man5 = None`). That is a
property of the v7.0 pin, not of this head; the paired baseline must carry the
same pin or the comparison is confounded.
