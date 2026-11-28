# The `refc_v3_train.py` patch — exact before/after, for its owner to apply

⛔ **I DO NOT OWN `stack/scripts/refc_v3_train.py` AND DID NOT TOUCH IT.** Everything below
is stated as a whole-line BEFORE/AFTER so it can be applied and verified without reading my
diff. Verify with `git hash-object` against `git rev-parse HEAD:<path>` afterwards, not with
`git add`'s exit code.

⚠️ **The modules are already landed and are backward compatible.** With NONE of this patch
applied the trainer still runs; what it loses is (a) the `lift_valid` mask on the map loss —
the D-3 fix has no caller — and (b) the operator flags plus the D-4 corpus-line derivation.
⛔ **The D-1/D-2 fix is ALREADY LIVE without the patch**, because `box3d_loss_row`'s default
is now `visible_filter=True` and the trainer calls it without a keyword. That is the
behaviour change, and hunk 5 is what gives the operator a name for turning it off.

⛔⛔ **HUNK 4 AND HUNK 6 MUST LAND TOGETHER OR NEITHER.** Hunk 4 lets the operator choose the
filter; hunk 6 makes the class-weight population follow that choice. Landing 4 alone lets a
filtered arm train on raw-join frequencies — 1.746× on `other_vehicle`, 0.663× on
`stroller`, **2.63× end to end** — which is the `anchors.pt` units defect in a frequency
costume and is SILENT. (Landing neither is also consistent: the default is filtered and the
default population then stays `raw_join`, which is a *declared* mismatch. See §"If you
apply nothing" at the bottom.)

⛔⛔ **A SOURCE-ADJACENCY BUDGET BINDS HUNKS 1 AND 2. MEASURED 2026-09-23, and my first
draft of this patch BROKE IT.** `stack/tests/test_map_iou_drivable.py:127-131` asserts

```python
i = SRC.index('extra["n_map_cells"] = _mrow["n_map_cells"]')
j = SRC.index('extra["map_iou_drivable"]')
assert 0 < j - i < 2500
```

— a **character** distance in the trainer's source, so that the IoU is computed in the same
block as the loss from the same tensors. **Current `j - i` = 1,877; headroom = 623
characters.** My first draft added ≈ 685 and would have gone RED on a test that has nothing
to do with these fixes.

⇒ **The hunks below are written to respect it**, by two rules:
1. **Everything Hunk 1 adds goes BEFORE the `extra["n_map_cells"]` line.** Text inserted
   before the marker shifts `i` and `j` equally and costs **zero** headroom.
2. **Hunk 2's comment is deliberately terse** — the reasoning lives in `PERCEPTION_FIXES.md`
   §3, not in the trainer.

⭐ **NOT ARITHMETIC — MEASURED.** Hunks 1 and 2 were applied to a **scratch COPY** of the
trainer (whole-line equality, one match each) and the copy read **`j − i` = 2,070, headroom
430**, and `ast.parse` clean. The real file's `sha256[:16]` was **`40ed2ee52601efa8` before
and after** the exercise — untouched. That measurement *is* the assertion
`test_map_iou_drivable.py` makes, so a clean copy means a clean test.

⚠️ **Re-measure after applying**, don't trust this arithmetic:
```bash
python -c "import pathlib;S=pathlib.Path('stack/scripts/refc_v3_train.py').read_text(encoding='utf-8');\
i=S.index('extra[\"n_map_cells\"] = _mrow[\"n_map_cells\"]');j=S.index('extra[\"map_iou_drivable\"]');\
print('j-i',j-i,'(must be < 2500)')"
```

---

## Hunk 1 — the MAP loss gets the lift mask (D-3)

`refc_v3_train.py` ~`:4286`

**BEFORE**
```python
                _mrow = _perc.map_loss_row(
                    _pout["map_logits"].index_select(0, _msel),
                    batch["map_frac"].to(device).index_select(0, _msel),
                    batch["map_seen"].to(device).index_select(0, _msel))
                loss = loss + _w_map * _mrow["loss"]
                extra["map"] = _mrow["loss"]
                extra["n_map_cells"] = _mrow["n_map_cells"]
```

**AFTER**
```python
                # ⛔⛔ D-3 (MEASURED 2026-09-22): `map_seen` is a CLIP-LIFETIME mask. The
                # SAM3 artifact declares itself `non_causal` on 135/135 files, and
                # 90.088 % of the 590 cells that lie outside the rig's ±60° at EVERY
                # instant are labelled `seen`. Against the mask the lift ALREADY computes,
                # 2,170,570 of 19,647,460 supervised cells (11.048 %) sit on cells
                # `BEVLift.forward` has zeroed and replaced with a learned `unobserved`
                # constant -- the loss was asking the head to name a class from a constant.
                # ⭐ `map_valid` is emitted by the branch (one reduction, no parameter);
                # `--no-map-lift-valid-mask` is the deliberate-regression arm.
                _mvalid = _pout.get("map_valid")
                if _mvalid is not None and getattr(args, "map_lift_valid_mask", True):
                    _mvalid = _mvalid.index_select(0, _msel)
                else:
                    _mvalid = None
                _mrow = _perc.map_loss_row(
                    _pout["map_logits"].index_select(0, _msel),
                    batch["map_frac"].to(device).index_select(0, _msel),
                    batch["map_seen"].to(device).index_select(0, _msel),
                    lift_valid=_mvalid)
                loss = loss + _w_map * _mrow["loss"]
                extra["map"] = _mrow["loss"]
                # ⭐ BOTH counts, and they go BEFORE the `n_map_cells` line ON PURPOSE:
                # `test_map_iou_drivable.py` measures a CHARACTER distance from that line
                # to `map_iou_drivable`, and text inserted before it costs no headroom.
                extra["n_map_cells_seen"] = _mrow["n_map_cells_seen"]
                extra["n_map_cells_unobserved"] = _mrow["n_map_cells_unobserved"]
                extra["n_map_cells"] = _mrow["n_map_cells"]
```

⛔ **Do not reorder those three `extra[...]` lines.** `extra["n_map_cells"]` must stay last;
the other two are the fix's visible effect and must not push it further from
`map_iou_drivable`.

## Hunk 2 — the drivable-IoU companion must score the SAME cells

`refc_v3_train.py` ~`:4312`

**BEFORE**
```python
                    _dch = _sem_map.CHANNELS.index("drivable")
                    _sn = batch["map_seen"].to(device).index_select(0, _msel)
```

**AFTER**
```python
                    _dch = _sem_map.CHANNELS.index("drivable")
                    _sn = batch["map_seen"].to(device).index_select(0, _msel)
                    # ⛔ SAME CELL SET AS THE LOSS (PERCEPTION_FIXES.md §3).
                    if _mvalid is not None:
                        _sn = _sn & _mvalid
```

⛔ **The comment is one line on purpose** — see the source-adjacency budget above. The
reasoning it replaces: an IoU scored on cells the loss does not supervise is two rules, and
the IoU is the number a collision gate reads. ⚠️ **The 0.3412 no-information drivable-IoU
floor was measured on the UNMASKED cell set** — requote it against the masked set before
using it as a gate on a masked arm.

## Hunk 3 — the BOX loss states its filter explicitly

`refc_v3_train.py` ~`:4388`

**BEFORE**
```python
                _brow = _perc.box3d_loss_row(
                    _s3, _t3,
                    cls_class_weight=getattr(model, "_cls_class_weight", None))
```

**AFTER**
```python
                # ⛔⛔ D-1/D-2 (MEASURED 2026-09-22 on v7-B1, 28,958,699 boxes): until
                # 2026-09-23 this path applied NO field cut, so 59.805 % of its targets
                # were outside the camera and 50.038 % were BEHIND THE EGO, and
                # `match_slots`' nearest-N then spent 50.076 % of its query slots there.
                # The default is now FILTERED; naming it here keeps the arm's choice in
                # argv and therefore in `config.json`.
                _brow = _perc.box3d_loss_row(
                    _s3, _t3,
                    cls_class_weight=getattr(model, "_cls_class_weight", None),
                    visible_filter=bool(getattr(args, "box3d_visible_filter", True)))
```

## Hunk 4 — the two new flags

`refc_v3_train.py` ~`:8575`, immediately after the `--w-box3d` entry in group `g6`

**BEFORE**
```python
    g6.add_argument("--map-gt-root", default=None,
```

**AFTER**
```python
    # ⛔ DEFAULT ON, AND THAT IS A BEHAVIOUR CHANGE (2026-09-23). Until then the refcv6
    # box path applied no field cut at all while the v6 seam one module over always has
    # (`refc_agents.agent_losses(filter_visible=True)`). `--no-box3d-visible-filter` is
    # the deliberate-regression arm and must be asked for by name.
    # ⛔ It also selects the CLASS-WEIGHT POPULATION (see `_cls_weight_stamp`): the
    # frequencies the `cls` term meets move by up to 1.746x under the filter.
    g6.add_argument("--box3d-visible-filter", dest="box3d_visible_filter",
                    action="store_true", default=True,
                    help="apply `refc_agents.visible_target_filter` to the 3-D box "
                         "targets BEFORE the query budget (DEFAULT). MEASURED on v7-B1: "
                         "59.805 %% of raw targets are outside the 120 deg field, "
                         "50.038 %% are behind the ego, and the nearest-N budget spent "
                         "50.076 %% of its slots on them.")
    g6.add_argument("--no-box3d-visible-filter", dest="box3d_visible_filter",
                    action="store_false",
                    help="DELIBERATE REGRESSION: train the 3-D box head on every box in "
                         "the join, including the ones behind the ego. Requires "
                         "--agent-cls-weight to resolve to a raw_join vector.")
    g6.add_argument("--map-lift-valid-mask", dest="map_lift_valid_mask",
                    action="store_true", default=True,
                    help="narrow the map loss's `seen` mask to cells the camera reaches "
                         "at THIS instant, using the lift's own `valid` (DEFAULT). "
                         "MEASURED: 11.048 %% of map supervision sits on cells BEVLift "
                         "has already replaced with its `unobserved` embedding.")
    g6.add_argument("--no-map-lift-valid-mask", dest="map_lift_valid_mask",
                    action="store_false",
                    help="DELIBERATE REGRESSION: supervise every `seen` cell, including "
                         "the ones no camera reached in this frame.")
    g6.add_argument("--map-gt-root", default=None,
```

## Hunk 5 — `--agent-cls-weight` gains the `off`-safe population, derived from the ARM

`refc_v3_train.py` ~`:4993` (`_cls_weight_stamp`)

**BEFORE**
```python
    mode = str(getattr(args, "agent_cls_weight", "off"))
    if mode == "off":
        return {"requested": "off", "mode": "off", "built": None}
    _name, _line = CLS_WEIGHT_CHOICES[mode]
    _, st = _agent_slots.load_cls_class_weight(_name, expect_corpus_line=_line)
    return dict(st, requested=mode, mode=mode, built=None)
```

**AFTER**
```python
    mode = str(getattr(args, "agent_cls_weight", "off"))
    if mode == "off":
        return {"requested": "off", "mode": "off", "built": None}
    _name, _line = CLS_WEIGHT_CHOICES[mode]
    _line, _pop, _src = _cls_weight_expectation(args, _line)
    _, st = _agent_slots.load_cls_class_weight(_name, expect_corpus_line=_line,
                                               target_population=_pop)
    return dict(st, requested=mode, mode=mode, built=None,
                corpus_line_source=_src)
```

…and, immediately **above** `_cls_weight_stamp`, the new helper:

```python
def _cls_weight_expectation(args, key_line: str) -> tuple[str, str, str]:
    """-> (expected corpus line, target population, where the line came from).

    ⛔⛔ D-4 (MEASURED 2026-09-22, `raw/p5_cls_weight_guard.json`): with the expectation
    and the artifact selected by the SAME key, the guard **cannot go red on the operator
    error it names** -- a B1 arm launched with `--agent-cls-weight train2400` loaded the
    PARITY vector and nothing refused (`guard_blocks_the_operator_error: false`). The
    arm's ACTUAL corpus was never an input. Two independent sources is the whole fix:
    `CLS_WEIGHT_CHOICES` says what the FILE claims, `corpus_line_for_join` says what the
    ARM trains, and they must agree.

    ⛔ The POPULATION comes from the box loss's own filter flag, so a filtered loss cannot
    silently run raw-join frequencies (1.746x `other_vehicle`, 0.663x `stroller`,
    2.63x end to end).
    """
    _arm = _agent_slots.corpus_line_for_join(getattr(args, "agent_join", None))
    if _arm is None:
        src = "weight-key-only (the join is not a line this table names)"
        print("[v3] ⚠️ agent cls weight: the corpus line could not be DERIVED from "
              "--agent-join %r, so the only source is the --agent-cls-weight key. The "
              "guard is single-sourced on this arm and cannot catch an artifact/corpus "
              "mismatch." % (getattr(args, "agent_join", None),), flush=True)
        line = key_line
    elif str(_arm) != str(key_line):
        raise SystemExit(
            "[v3] ⛔ --agent-cls-weight %r declares corpus line %r but --agent-join is "
            "the %r line. MEASURED 2026-09-22: the two lines share 4.09 %% of their "
            "clips and their vectors differ by up to 1.857x. Refusing." %
            (str(getattr(args, "agent_cls_weight", "off")), key_line, _arm))
    else:
        src = "derived-from-join (agrees with the weight key)"
        line = _arm
    pop = (_agent_slots.TARGET_POPULATION_VISIBLE
           if bool(getattr(args, "box3d_visible_filter", True))
           else _agent_slots.TARGET_POPULATION_RAW)
    return line, pop, src
```

⚠️ **Note the `elif` raises before `src` is bound** — that is intentional; the branch never
returns. Keep the ordering.

## Hunk 6 — the model attach reads the same expectation

`refc_v3_train.py` ~`:6270`

**BEFORE**
```python
        _cwname, _cwline = CLS_WEIGHT_CHOICES[_cwmode]
        _cw, _cws = _agent_slots.load_cls_class_weight(_cwname, expect_corpus_line=_cwline)
```

**AFTER**
```python
        _cwname, _cwline = CLS_WEIGHT_CHOICES[_cwmode]
        # ⛔ ONE SPELLING of the expectation. Two call sites deriving it separately is how
        # the stamp and the tensor end up describing different arms.
        _cwline, _cwpop, _cwsrc = _cls_weight_expectation(args, _cwline)
        _cw, _cws = _agent_slots.load_cls_class_weight(
            _cwname, expect_corpus_line=_cwline, target_population=_cwpop)
```

…and the print one line below gains the population, so the console states it:

**BEFORE**
```python
        print("[v3] agent cls weight: %s on corpus line %s (%d classes, imbalance %s:1, "
              "digest %s)" % (_cwmode, _cws["corpus_line"], len(_cws["weights"]),
                              _cws["imbalance_majority_to_rarest"], _cws["digest"]), flush=True)
```

**AFTER**
```python
        print("[v3] agent cls weight: %s on corpus line %s, population %s (%s) "
              "(%d classes, imbalance %s:1, digest %s)"
              % (_cwmode, _cws["corpus_line"], _cws["target_population"], _cwsrc,
                 len(_cws["weights"]), _cws["imbalance_majority_to_rarest"],
                 _cws["digest"]), flush=True)
```

---

## If you apply nothing

The default becomes: **box loss FILTERED** (hunk 3's behaviour is the module default) with
**`raw_join` class weights** (the loader's default). That combination is the one the review
names as a new scope error — but it is now **declared** rather than silent: the stamp says
`target_population: "raw_join"` while `_visible_filter` is True. ⛔ It is still wrong, and
hunks 4+5+6 are what make it inexpressible. If you cannot land them this session, the
correct interim arm is `--agent-cls-weight off` (bit-identical to every pre-flag arm) or
landing hunks 5+6 alone, which makes the population follow the (defaulted-True) flag even
without the flag existing on the parser — `getattr(args, "box3d_visible_filter", True)`
returns True and the visible vector is selected.

## What to assert afterwards (artifacts, not exit codes)

```bash
git show HEAD:stack/scripts/refc_v3_train.py | grep -c "map_valid"            # expect >= 2
git show HEAD:stack/scripts/refc_v3_train.py | grep -c "box3d_visible_filter"  # expect >= 4
git show HEAD:stack/scripts/refc_v3_train.py | grep -c "_cls_weight_expectation"  # expect 3
PYTHONPATH=stack python -m pytest -q stack/tests/test_cls_weight_stamp.py \
    stack/tests/test_refcv6_perception_supervision_fixes.py
```
⚠️ `test_cls_weight_stamp.py::test_every_choice_maps_to_an_artifact_that_declares_the_expected_line`
asserts `set(CLS_WEIGHT_CHOICES) == {"train2400", "b1"}` and that each maps to a
`(name, line)` pair. **This patch deliberately keeps that shape** — no new choice key — so
that test stays green untouched.
