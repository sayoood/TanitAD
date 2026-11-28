"""Apply the D-TACGOAL-TRAINER-SEAM-OPEN wiring to refc_v3_train.py.

FIVE ADDITIVE EDITS. No existing default is changed; no existing loss is
rebalanced. Every anchor is asserted UNIQUE before substitution, and the
result is re-parsed with `ast` before it is written.
"""
from __future__ import annotations

import ast
import io
import sys

SRC = sys.argv[1]
DST = sys.argv[2]

with io.open(SRC, "r", encoding="utf-8", newline="") as fh:
    s = fh.read()

EDITS: list[tuple[str, str, str]] = []

# --------------------------------------------------------------------------- #
# EDIT 1 -- REFC_WEIGHT_GATES: the row this head has never had.               #
# --------------------------------------------------------------------------- #
A1 = '''    "agent_w_project": {
        "flag": "--agent-w-project", "term": "agent projection consistency",'''
B1 = '''    # ⭐⭐ D-TACGOAL-TRAINER-SEAM-OPEN (2026-09-09). This row is HALF THE FIX.
    # MEASURED 2026-09-07 on the live refcv5-v2 argv: `tac_goal_tok_head`
    # took `p.grad is None` on BOTH tensors over a 40,284-step run -- 11,286
    # parameters built, stamped and unable to learn -- and BOTH standing
    # guards were green and NEITHER was wrong. `assert_seams_are_built` asks
    # *"is it built?"* (it was). This audit enumerates DECLARED LOSS WEIGHTS
    # and asks which build a graph -- and the head HAD NO WEIGHT FLAG, so it
    # produced NO ROW AT ALL. ⛔ An instrument that enumerates weights is
    # structurally blind to a head that has none, which is why giving the head
    # a real weight flag is not packaging around the fix: it IS part of it.
    "w_tac_goal": {
        "flag": "--w-tac-goal",
        "term": "D-TACGOAL 22-token tactical goal SET (multi-label BCE)",
        # `refc_v3.py:1015` builds the head IFF `_vv != "kin3" AND
        # cfg.tac_goal_tok_head`, and `:1320` is the only writer of
        # `cache["tac_goal_logits"]`; the target comes from the v7.2 join
        # (`V3Dataset.tac_goal_targets`), so BOTH are required.
        "gate": lambda a: (bool(getattr(a, "tac_goal_tok_head", False))
                           and bool(getattr(a, "v7_labels", None)),
                           "--w-tac-goal needs `--tac-goal-tok-head` (no head "
                           "=> no `tac_goal_logits` in `out`) AND `--v7-labels` "
                           "(no join => no `tac_goal_y`/`tac_goal_w` target)"),
        "mask": None,
        "already": "NOTHING, until 2026-09-09 -- this is the first weight this "
                   "head has ever had, and until it existed the head was "
                   "invisible to this audit by construction "
                   "(D-TACGOAL-TRAINER-SEAM-OPEN).",
    },
    "agent_w_project": {
        "flag": "--agent-w-project", "term": "agent projection consistency",'''
EDITS.append(("E1 REFC_WEIGHT_GATES row", A1, B1))

# --------------------------------------------------------------------------- #
# EDIT 2 -- V3Dataset: the class attributes that switch the target channel on. #
# --------------------------------------------------------------------------- #
A2 = '''    v7_by_sid: dict | None = None
    v7_dt: float = 0.1
'''
B2 = '''    v7_by_sid: dict | None = None
    v7_dt: float = 0.1
    #: --w-tac-goal (D-TACGOAL-TRAINER-SEAM-OPEN): when True every item carries
    #: ``tac_goal_y``/``tac_goal_w`` ``[22]`` from
    #: :func:`v7_labels.tactical_goal_targets`. ⛔ Default False keeps the batch
    #: BYTE-IDENTICAL for a recipe that does not ask for the term -- the same
    #: discipline as ``nav_from_v7`` above, and the reason the OFF path can be
    #: proven bit-identical to the pre-wiring trainer at a fixed seed.
    tac_goal_targets: bool = False
    #: ⛔ NOT a literal in the loss. The negative policy is a property of THE
    #: LOADED SPLIT (`v7_labels.tactical_goal_targets.__doc__`: 3,574 of 4,572
    #: clips were never ASKED the traffic-light question, so their absence is
    #: NOT-PROBED rather than negative). Surfaced as ``--tac-goal-negatives``
    #: so the cheap arm is comparable rather than hidden.
    tac_goal_negatives: str = "measured"
'''
EDITS.append(("E2 V3Dataset attrs", A2, B2))

# --------------------------------------------------------------------------- #
# EDIT 3 -- V3Dataset.__getitem__: emit the target beside lat_v7/lon_v7.       #
# --------------------------------------------------------------------------- #
A3 = '''            item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)
            item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)
'''
B3 = '''            item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)
            item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)
            # ---- --w-tac-goal: the 22-token goal SET target ---------------
            # ⛔ ALWAYS EMITTED WHEN THE CHANNEL IS ON, including for a clip
            # with NO record -- as an explicitly all-ignored row, never a
            # missing key. That is the `nav_args` convention three blocks
            # down, and for the same measured reason: a batch that sometimes
            # carries the key would make the loss-time refusal fire at random
            # instead of at launch.
            # ⚠️ The window's NOW is `(t + w - 1) * v7_dt`, the SAME expression
            # `tactical_class_ids` is called with above. Passing the record's
            # own anchor instead would make `window_in_band` a tautology.
            if self.tac_goal_targets:
                if lab is None:
                    _n_tg = len(v7l.TAC_GOAL_TOKENS)
                    _tg_y = (0.0,) * _n_tg
                    _tg_w = (v7l.IGNORE_W,) * _n_tg
                else:
                    _tg_y, _tg_w = v7l.tactical_goal_targets(
                        lab, (t + w - 1) * self.v7_dt,
                        negatives=self.tac_goal_negatives)
                item["tac_goal_y"] = torch.tensor(_tg_y, dtype=torch.float32)
                item["tac_goal_w"] = torch.tensor(_tg_w, dtype=torch.float32)
'''
EDITS.append(("E3 V3Dataset.__getitem__ target", A3, B3))

# --------------------------------------------------------------------------- #
# EDIT 4 -- compute_losses_v3: THE CALL THAT WAS MISSING.                      #
# --------------------------------------------------------------------------- #
A4 = '''    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,'''
B4 = '''    # ---- D-TACGOAL: the 22-token tactical goal SET ------------------------
    # ⭐⭐ THIS IS THE SEAM THAT WAS OPEN. Until 2026-09-09 this file contained
    # ZERO occurrences of `tac_goal_loss` and ZERO of `TacGoalEmitter`, so
    # `refc_v3.py:1320` wrote `cache["tac_goal_logits"]` into a dict nothing
    # read: the head was built (`--tac-goal-tok-head`), stamped
    # (`_seams["tac_goal_tok_head"]["built"]`), forward-run, and had NO
    # GRADIENT PATH. refcv5-v2 trained it for 40,284 steps at grad_abs_sum
    # exactly 0.0.
    #
    # ⛔ THE `w <= 0` BRANCH IS AN ABSENCE, NOT A MULTIPLICATION BY ZERO. The
    # term never enters the graph, so a recipe that does not pass
    # `--w-tac-goal` is BIT-IDENTICAL to the pre-wiring trainer at a fixed
    # seed. That identity is the whole reason this may land beside a live
    # recipe without a rebalancing decision (which is the PI's, queue item 10,
    # and explicitly NOT taken here).
    #
    # ⚠️ AND THE AMBIGUITY THAT COSTS -- named, because `tac_goal_head.py`
    # warns about it in the opposite direction. A guarded term makes
    # `p.grad is None`, which reads IDENTICALLY to "never wired". Here that
    # confusion cannot survive, because the head now HAS A WEIGHT and
    # therefore a ROW in `effective_weights_stamp_v3`: `w_tac_goal = 0.0` in
    # config.json is a POSITIVE RECORD that the term was switched off, which
    # is exactly the evidence the 2026-09-07 census did not have. Inside the
    # channel nothing is guarded -- `tac_goal_loss` divides by a clamped
    # denominator on purpose, so an all-ignored batch still returns a real
    # zero WITH `n_supervised`, and the head still receives a gradient tensor.
    _w_tg = float(getattr(model, "_w_tac_goal", 0.0) or 0.0)
    if _w_tg > 0.0:
        # ⛔⛔ REFUSE, DO NOT SKIP -- the `--w-bev-aux` guard immediately above,
        # and the `--w-agent` guard below it, for the one measured reason:
        # a run that trains, converges, writes a checkpoint and STAMPS
        # `w_tac_goal` while the head was never supervised would read as
        # "the tactical goal set does not help". That is the failure this
        # whole package exists to close, re-manufactured one layer up.
        if "tac_goal_logits" not in out:
            raise SystemExit(
                "[v3] ⛔ --w-tac-goal > 0 but `out` carries no "
                "`tac_goal_logits`: the 22-token head was never built. Pass "
                "--tac-goal-tok-head (with --v7-labels, which pins the v7.0 "
                "tactical vocabulary), or --w-tac-goal 0.")
        if "tac_goal_y" not in batch or "tac_goal_w" not in batch:
            raise SystemExit(
                "[v3] ⛔ --w-tac-goal > 0 but the batch carries no "
                "`tac_goal_y`/`tac_goal_w`: this dataset has no goal-set "
                "target wired (`ds.tac_goal_targets` is False), so the loss "
                "would be SILENTLY SKIPPED while config.json stamps the "
                "weight -- the `w_agent` defect verbatim. Pass --v7-labels, "
                "or --w-tac-goal 0.")
        _tg_loss, _tg_n = _tac_goal_head.tac_goal_loss(
            out["tac_goal_logits"],
            batch["tac_goal_y"].to(device), batch["tac_goal_w"].to(device),
            pos_weight=getattr(model, "_tac_goal_pos_weight", None),
            class_mask=getattr(model, "_tac_goal_class_mask", None))
        loss = loss + _w_tg * _tg_loss
        extra["tac_goal"] = _tg_loss
        # ⭐ n PER TERM, IN THE LOG ROW, ALWAYS. `tac_goal_loss` returns
        # `n_supervised` precisely so a 0.0 can be read as "nothing was in
        # band" rather than "the head is broken" -- the route-head trap its
        # own docstring names (an apparent zero gradient that was a validity
        # MASK, 0.0 -> 0.687 when forced). A bare 0.0 without this number
        # reads as "supervised, and perfect".
        extra["tac_goal_n_supervised"] = float(_tg_n)
        extra["tac_goal_n_pos"] = float(
            ((batch["tac_goal_y"] > 0.5) & (batch["tac_goal_w"] > 0)).sum())

    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,'''
EDITS.append(("E4 compute_losses_v3 term", A4, B4))

# --------------------------------------------------------------------------- #
# EDIT 5 -- train(): carry the weight (and the split-derived constants).       #
# --------------------------------------------------------------------------- #
A5 = '''    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))   # WP-D
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
'''
B5 = '''    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))   # WP-D
    model._bev_shuffle = bool(getattr(args, "bev_aux_shuffle", False))
    # D-TACGOAL: same carrier, same reason. ⛔ `_tac_goal_pos_weight` and
    # `_tac_goal_class_mask` are set from THE LOADED SPLIT further down (never
    # from a literal -- `goal_pos_weight.__doc__` is explicit, and it is the
    # derived-constant trap that moved HORIZON 7 -> 8). They stay None here so
    # a w>0 arm that somehow reached the loop without a join trains UNWEIGHTED
    # and UNMASKED rather than silently using another split's constants.
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
'''
EDITS.append(("E5 train() weight carrier", A5, B5))

# --------------------------------------------------------------------------- #
# EDIT 6 -- train(): fit pos_weight / class_mask on the loaded split.          #
# --------------------------------------------------------------------------- #
A6 = '''        if getattr(args, "max_speed_input", False):
            max_speed_stats = ds.enable_max_speed(
                manifest, str(getattr(args, "max_speed_mode",
                                      msi.DEFAULT_MODE)))
'''
B6 = '''        # ---- ⭐⭐ D-TACGOAL: the goal-SET target channel, and the two
        # constants that MUST come from this split rather than a literal.
        # ⛔ `goal_pos_weight` is n_neg/n_pos capped at 50 and `mask_report`
        # switches off every class with positives but NO supervised negative
        # (5 of 22 on the v7.2 train blob) -- an unmasked logit there can only
        # ever be pushed towards 1, which degrades the shared trunk and
        # inflates any pooled score. Both are stamped into config.json, so the
        # run record says what the head was actually trained on.
        if float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0:
            ds.tac_goal_targets = True
            ds.tac_goal_negatives = str(getattr(args, "tac_goal_negatives",
                                                "measured"))
            tac_goal_census = v7l.goal_supervision_census(labels)
            tac_goal_mask = _tac_goal_head.mask_report(tac_goal_census)
            tac_goal_pw = v7l.goal_pos_weight(labels)
            model._tac_goal_pos_weight = torch.tensor(tac_goal_pw,
                                                      dtype=torch.float32)
            model._tac_goal_class_mask = torch.tensor(tac_goal_mask["mask"],
                                                      dtype=torch.float32)
            tac_goal_stats = {
                "negatives": ds.tac_goal_negatives,
                "n_trainable": int(tac_goal_mask["n_trainable"]),
                "n_total": int(tac_goal_mask["n_total"]),
                "trainable": list(tac_goal_mask["trainable"]),
                "masked_why": tac_goal_mask["masked_why"],
                "pos_weight": [float(x) for x in tac_goal_pw],
                "census": tac_goal_census,
            }
            print(f"[v3] tac_goal: {tac_goal_stats['n_trainable']}"
                  f"/{tac_goal_stats['n_total']} classes trainable "
                  f"(negatives={ds.tac_goal_negatives}), "
                  f"w={float(args.w_tac_goal)}", flush=True)
            # ⛔ A CHANNEL THAT IS ON AND TEACHES NOTHING IS THE DEFECT THIS
            # PACKAGE CLOSES, ONE LAYER UP. Refuse at LAUNCH, not after a
            # GPU day.
            if tac_goal_stats["n_trainable"] == 0:
                raise SystemExit(
                    "[v3] ⛔ --w-tac-goal > 0 but ZERO of "
                    f"{tac_goal_stats['n_total']} goal classes are trainable "
                    "on this split (every class lacks positives or lacks a "
                    "supervised negative). The head would be built, stamped "
                    "and supervised by an all-masked target.")
        if getattr(args, "max_speed_input", False):
            max_speed_stats = ds.enable_max_speed(
                manifest, str(getattr(args, "max_speed_mode",
                                      msi.DEFAULT_MODE)))
'''
EDITS.append(("E6 train() split-derived constants", A6, B6))

# --------------------------------------------------------------------------- #
# EDIT 7 -- argparse: the two knobs. BOTH are consumed above.                  #
# --------------------------------------------------------------------------- #
A7 = '''    ap.add_argument("--v2-lru", type=int, default=6,'''
B7 = '''    ap.add_argument("--w-tac-goal", type=float, default=0.0,
                    help="D-TACGOAL: weight on the 22-token tactical-goal SET "
                         "loss (multi-label BCE, tac_goal_head.tac_goal_loss). "
                         "⛔ DEFAULT 0.0 AND THE TERM IS THEN ABSENT FROM THE "
                         "GRAPH -- not multiplied by zero -- so a recipe that "
                         "does not pass it is bit-identical to the pre-wiring "
                         "trainer at a fixed seed. Needs --tac-goal-tok-head "
                         "and --v7-labels; REFC_WEIGHT_GATES refuses a weight "
                         "whose term can never be reached. ⚠️ This weight is "
                         "ADDITIVE: nothing is taken from MANEUVER_WEIGHT and "
                         "no existing term is rebalanced -- that is a PI "
                         "decision (queue item 10), not this flag's.")
    ap.add_argument("--tac-goal-negatives", default="measured",
                    choices=["measured", "geometry", "all"],
                    help="what an ABSENT goal token means. 'measured' "
                         "(default) reads provenance PER TOKEN FROM THE "
                         "LOADED SPLIT and supervises a negative only for "
                         "geometry-emitted tokens. ⛔ 'all' supervises every "
                         "absent cell: MEASURED, 3,574 of 4,572 clips were "
                         "never ASKED the traffic-light question, so 'all' "
                         "teaches the head that ~78 %% of the corpus has no "
                         "traffic light on no evidence. 'geometry' uses the "
                         "frozen declaration instead of the blob, kept so the "
                         "declaration/data divergence stays measurable.")
    ap.add_argument("--v2-lru", type=int, default=6,'''
EDITS.append(("E7 argparse knobs", A7, B7))

# --------------------------------------------------------------------------- #
# EDIT 8 -- the import.                                                        #
# --------------------------------------------------------------------------- #
A8 = '''from tanitad import effective_weights as _ew  # noqa: E402'''
B8 = '''from tanitad import effective_weights as _ew  # noqa: E402
from tanitad.refs import tac_goal_head as _tac_goal_head  # noqa: E402'''
EDITS.append(("E8 import", A8, B8))


# --------------------------------------------------------------------------- #
for tag, a, b in EDITS:
    n = s.count(a)
    if n != 1:
        raise SystemExit(f"⛔ ANCHOR NOT UNIQUE for {tag}: found {n} "
                         f"occurrences, expected exactly 1")
    s = s.replace(a, b, 1)
    print(f"applied {tag}")

# ⛔ ASSERT ON THE ARTIFACT: the result must PARSE, and the two markers the
# defect was defined by must now be present.
ast.parse(s)
# ⛔ Written as LITERAL expectations, each naming the ONE site it checks --
# never as a bare `>= 1`, which would pass on any stray mention.
assert s.count("_tac_goal_head.tac_goal_loss(") == 1, "the CALL did not land"
assert s.count('ap.add_argument("--w-tac-goal"') == 1, "the FLAG did not land"
assert s.count('"flag": "--w-tac-goal"') == 1, "the GATE ROW did not land"
assert s.count('ap.add_argument("--tac-goal-negatives"') == 1, "no negatives knob"
assert s.count("ds.tac_goal_targets = True") == 1, "the channel is never opened"

with io.open(DST, "w", encoding="utf-8", newline="") as fh:
    fh.write(s)
print(f"WROTE {DST} lines={s.count(chr(10))}")
