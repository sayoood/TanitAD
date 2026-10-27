"""Apply the refcv9 warm-start edits to the LF working copy of refc_v3_train.py; every anchor must match EXACTLY once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
EDITS = []


def edit(old, new):
    EDITS.append((old, new))


# 1. imports
edit('''from tanitad.train import refcv9_alloc_force as _r9af  # noqa: E402  refcv9: --r9-alloc-force-schedule (OFF: nothing)
''', '''from tanitad.train import refcv9_alloc_force as _r9af  # noqa: E402  refcv9: --r9-alloc-force-schedule (OFF: nothing)
from tanitad.train import map_precision as _mfp  # noqa: E402  refcv9 warm start: --w-map-fp (OFF: nothing)
''')
# 2. G-DVB registration
edit('''# PI D-R9-MAPHEAD-PRECISION (2026-10-26): --r8-train-only-map is a `built` lever (the trainable set read off the model)
_mof.register_dvb_levers(_dvb.register)
''', '''# PI D-R9-MAPHEAD-PRECISION (2026-10-26): --r8-train-only-map is a `built` lever (the trainable set read off the model);
# refcv9 warm start (PI 2026-10-27): --r9-train-only too (the trainable sets read off the model)
_mof.register_dvb_levers(_dvb.register)
# refcv9 warm start: --w-map-fp is a `loss` lever (the weight the 10 cm map term reads, model._w_map_fp)
_mfp.register_dvb_levers(_dvb.register)
''')
# 3. argv pin
edit('''    _mof.pin_args(args)          # PI D-R9-MAPHEAD-PRECISION: --r8-train-only-map's refusals (a no-op when off)
''', '''    _mof.pin_args(args)          # PI D-R9-MAPHEAD-PRECISION: --r8-train-only-map's refusals (a no-op when off)
    #                              refcv9 warm start: --r9-train-only's refusals (the same machinery; a no-op when off)
    _mfp.pin_args(args)          # refcv9 warm start: --w-map-fp's refusals (a no-op at the default 0)
''')
# 4. the effective-weight registry
edit('''    "w_r9_prog": {
        "flag": "--w-r9-prog",
''', '''    "w_map_fp": {
        "flag": "--w-map-fp",
        "term": "refcv9 map precision: mean over GT-drivable near-field corridor cells (x 0-20 m, |y| < 3 m) of the "
                "10 cm head's p_sidewalk + p_edge",
        # the term reads the 10 cm head's logits and the batch's map_fine (the CE's labelled windows): every condition
        # of `w_map_hires`, plus that weight itself being live
        "gate": lambda a: (
            str(getattr(a, "map_hires", "off")) == "on"
            and float(getattr(a, "w_map_hires", 0.0) or 0.0) > 0.0
            and REFC_WEIGHT_GATES["w_map_hires"]["gate"](a)[0],
            "--w-map-fp needs `--map-hires on` AND a live --w-map-hires (every condition of the 10 cm map term): it "
            "reads that head's logits and that term's labelled windows"),
        "mask": None,
        "already": "map_precision.pin_args (needs the 10 cm head; refused under a freeze that does not train the map) "
                   "+ G-DVB `w_map_fp` (model._w_map_fp is the declared weight; absent at 0)",
    },
    "w_r9_prog": {
        "flag": "--w-r9-prog",
''')
# 5. the map-hires loss: the precision term (absent at w 0)
edit('''    if not int(lab.sum()):
        # a COUNTED zero on the branch's own graph, never an absent key
        term = logits.sum() * 0.0
        extra["map_hires"] = term
        extra["n_map_hires_cells"] = 0.0
        return term
''', '''    _w_fp = float(getattr(model, _mfp.MODEL_ATTR, 0.0) or 0.0)    # refcv9 --w-map-fp (absent: 0, nothing below)
    if not int(lab.sum()):
        # a COUNTED zero on the branch's own graph, never an absent key
        term = logits.sum() * 0.0
        extra["map_hires"] = term
        extra["n_map_hires_cells"] = 0.0
        if _w_fp > 0.0:
            extra[_mfp.LOSS_KEY] = logits.sum() * 0.0
            extra[_mfp.COUNT_KEY] = 0.0
        return term
''')
edit('''    extra["map_hires"] = row["loss"]
    for k, v in row.items():
        if k != "loss":
            extra[k] = v
    return w * row["loss"]
''', '''    extra["map_hires"] = row["loss"]
    for k, v in row.items():
        if k != "loss":
            extra[k] = v
    if _w_fp > 0.0:
        # ⭐ refcv9 --w-map-fp (PI 2026-10-27): the near-field corridor's false-positive mass on GT-drivable cells, on
        # the SAME labelled windows and codes as the CE (tanitad/train/map_precision.py); logged every step
        _fp = _mfp.map_fp_loss(logits.index_select(0, sel), batch["map_fine"].to(device).index_select(0, sel),
                               x_max_m=float(br.cfg.x_max_m), y_half_m=float(br.cfg.y_half_m))
        extra[_mfp.LOSS_KEY] = _fp["loss"]
        extra[_mfp.COUNT_KEY] = float(_fp["n_cells"])
        return w * row["loss"] + _w_fp * _fp["loss"]
    return w * row["loss"]
''')
# 6. the model attribute + the stamp (only when w > 0)
edit('''        if _hcts is not None:                 # absent (not null) for a default run: config.json unchanged
            map_hires_stamp["class_thresholds"] = {''', '''        if _mfp.is_on(args):                 # refcv9 --w-map-fp: absent (no attribute, no key) for a default run
            setattr(model, _mfp.MODEL_ATTR, _mfp.weight(args))
            map_hires_stamp["map_fp"] = _mfp.stamp(args, float(_hext.x_max_m), float(_hext.y_half_m),
                                                   tuple(_hcfg.out_hw))
            print("[v3] refcv9 map precision: --w-map-fp %g on corridor rows %s cols %s (%d cells)" % (
                _mfp.weight(args), map_hires_stamp["map_fp"]["corridor_rows"],
                map_hires_stamp["map_fp"]["corridor_cols"], map_hires_stamp["map_fp"]["n_corridor_cells"]), flush=True)
        if _hcts is not None:                 # absent (not null) for a default run: config.json unchanged
            map_hires_stamp["class_thresholds"] = {''')
# 7. the freeze
edit('''    _mof_rec = _mof.apply_freeze(model) if _mof.is_on(args) else None
''', '''    _mof_rec = (_mof.apply_freeze(model) if _mof.is_r8(args) else
                # refcv9 warm start (PI 2026-10-27): --r9-train-only <sets> -- the same machinery, the named sets
                _mof.apply_train_only(model, _mof.r9_sets(args)) if _mof.is_r9(args) else None)
    _mof_mods = None if _mof_rec is None else tuple(_mof_rec["modules"])      # the TRAINABLE modules (None: no freeze)
    _mof_tag = _mof.freeze_tag(_mof_rec)
''')
edit('''    if _mof_rec is not None:
        print("[map-only] FROZEN all but %s: %d tensors / %d params trainable, %d tensors / %d params frozen, "
              "%d running-stat norms pinned" % (list(_mof.MAP_ONLY_MODULES), _mof_rec["n_trainable_tensors"],
                                                 _mof_rec["n_trainable_params"], _mof_rec["n_frozen_tensors"],
                                                 _mof_rec["n_frozen_params"], _mof_rec["frozen_norms_pinned"]),
              flush=True)
''', '''    if _mof_rec is not None:
        print("[%s] FROZEN all but %s: %d tensors / %d params trainable, %d tensors / %d params frozen, "
              "%d running-stat norms pinned" % (_mof_tag, list(_mof_mods), _mof_rec["n_trainable_tensors"],
                                                 _mof_rec["n_trainable_params"], _mof_rec["n_frozen_tensors"],
                                                 _mof_rec["n_frozen_params"], _mof_rec["frozen_norms_pinned"]),
              flush=True)
        if _mof_rec.get("per_set"):
            print("[r9-train-only] sets %s: %s" % (_mof_rec["sets"], {s: (v["n_tensors"], v["n_params"])
                                                                      for s, v in _mof_rec["per_set"].items()}),
                  flush=True)
''')
# 8. the source, the verification, the guard
edit('''    if _mof_rec is not None:
        _mof_src, _mof_rec["init_from"] = _mof.load_source(args.init_from)
        _mof_rec["init_from"]["verify"] = _mof.verify_against_source(model, _mof_src, fresh=not ck.exists())
        _mof_guard = _mof.FrozenStateGuard(_mof_src)
        print("[map-only] --init-from %s (md5 %s, kind %s, step %s): %s over %d keys" % (
            args.init_from,''', '''    if _mof_rec is not None:
        _mof_src, _mof_rec["init_from"] = _mof.load_source(args.init_from)
        _mof_rec["init_from"]["verify"] = _mof.verify_against_source(model, _mof_src, fresh=not ck.exists(),
                                                                     modules=_mof_mods)
        _mof_guard = _mof.FrozenStateGuard(_mof_src, modules=_mof_mods)
        print("[%s] --init-from %s (md5 %s, kind %s, step %s): %s over %d keys" % (
            _mof_tag, args.init_from,''')
# 9. config.json
edit('''    if _mof_rec is not None:              # PI D-R9-MAPHEAD-PRECISION: the flag, the trainable list + count, the source
        _run_config["map_only_ft"] = _mof_rec
    (out_dir / "config.json").write_text(json.dumps(_run_config, indent=1),''', '''    if _mof_rec is not None:              # PI D-R9-MAPHEAD-PRECISION: the flag, the trainable list + count, the source
        _run_config["r9_train_only" if _mof.is_r9(args) else "map_only_ft"] = _mof_rec
    (out_dir / "config.json").write_text(json.dumps(_run_config, indent=1),''')
# 10. the EMA
edit('''        _ema = _wema.WeightEMA(model, float(args.ema_decay),
                               exclude=(_mof.ema_exclude if _mof_rec is not None else None))
''', '''        _ema = _wema.WeightEMA(model, float(args.ema_decay),
                               exclude=(_mof.ema_exclude_for(_mof_mods) if _mof_rec is not None else None))
''')
edit('''        _mof_bad = (_mof.ema_problems(_ema, model, _mof_src, fresh=not ck.exists()) if _ema is not None else [])
        if _mof_bad:
            raise SystemExit("[map-only] ⛔ the EMA is not the map-only EMA:\\n  - " + "\\n  - ".join(_mof_bad))
        _mof_guard.assert_ok(model, "start (step %d)" % step)
        _mof_rec["ema_check"] = ("no EMA (--ema-decay 0)" if _ema is None else
                                 "shadow = %d map-only tensors; %s" % (
                                     len(_ema.shadow), "fresh: raw == EMA == --init-from, bit for bit"
                                     if not ck.exists() else "resumed"))
        _run_config["map_only_ft"] = _mof_rec
''', '''        _mof_bad = (_mof.ema_problems(_ema, model, _mof_src, fresh=not ck.exists(), modules=_mof_mods)
                    if _ema is not None else [])
        if _mof_bad:
            raise SystemExit("[%s] ⛔ the EMA is not the %s EMA:\\n  - " % (_mof_tag, _mof_tag)
                             + "\\n  - ".join(_mof_bad))
        _mof_guard.assert_ok(model, "start (step %d)" % step)
        _mof_rec["ema_check"] = ("no EMA (--ema-decay 0)" if _ema is None else
                                 "shadow = %d %s tensors; %s" % (
                                     len(_ema.shadow), "map-only" if _mof_tag == "map-only" else "trainable",
                                     "fresh: raw == EMA == --init-from, bit for bit"
                                     if not ck.exists() else "resumed"))
        _run_config["r9_train_only" if _mof.is_r9(args) else "map_only_ft"] = _mof_rec
''')
# 11. the optimizer recipe check message
edit('''    if _mof.is_on(args):
        if model is None:
            raise SystemExit("[v3] ⛔ --r8-train-only-map: the optimizer recipe check needs the model")
        _bad = _mof.optimizer_problems(opt, model, args)
        if _bad:
            raise SystemExit("[v3] ⛔ THE OPTIMIZER THAT WILL RUN IS NOT THE MAP-ONLY RECIPE:\\n  - "
                             + "\\n  - ".join(_bad))
        return {"opt": "dd", "type": type(opt).__name__, "map_only": True,''', '''    if _mof.is_on(args):
        if model is None:
            raise SystemExit("[v3] ⛔ %s: the optimizer recipe check needs the model"
                             % ("--r9-train-only" if _mof.is_r9(args) else "--r8-train-only-map"))
        _bad = _mof.optimizer_problems(opt, model, args)
        if _bad:
            raise SystemExit("[v3] ⛔ THE OPTIMIZER THAT WILL RUN IS NOT THE %s RECIPE:\\n  - "
                             % ("TRAIN-ONLY" if _mof.is_r9(args) else "MAP-ONLY") + "\\n  - ".join(_bad))
        if _mof.is_r9(args):
            return {"opt": "dd", "type": type(opt).__name__, "train_only": list(_mof.r9_sets(args)),
                    "groups": [{"name": g.get("name", "?"), "lr": float(g["lr"]),
                                "weight_decay": float(g.get("weight_decay", 0.0)),
                                "n_tensors": len(g["params"])} for g in opt.param_groups]}
        return {"opt": "dd", "type": type(opt).__name__, "map_only": True,''')
# 12. the parser
edit('''    g9 = ap.add_argument_group("refcv9 R9-PF (clean-split, regime-factored progress head; REFCV9_INMODEL_DESIGN.md)")
''', '''    g8.add_argument("--r9-train-only", default=None, metavar="SET[,SET]",
                    help="refcv9 WARM START (PI 2026-10-27): train ONLY the named module sets, freeze everything else "
                         "(perception) with --r8-train-only-map's machinery (requires_grad, one AdamW group "
                         "`train_only` at --lr, pinned norms, the class-prior write skipped, the EMA exclusion, the "
                         "shipped-EMA warm start checked bit for bit, the frozen-state guard on every logged / saved "
                         "step). Sets: `planner` = core.decoder (186 tensors / 15,365,801 values on refcv8), `map` = "
                         "the three map-only modules (= --r8-train-only-map). Needs --refcv8 --opt dd --init-from = the "
                         "SHIPPED EMA model --conflict-detector off. config.json['r9_train_only'] records the sets, the "
                         "trainable list and the counts.")
    g8.add_argument("--w-map-fp", type=float, default=0.0,
                    help="refcv9 map precision term (PI 2026-10-27): + w x mean over GT-drivable cells of the "
                         "near-field corridor (x 0-20 m, |y| < 3 m) of the 10 cm head's p_sidewalk + p_edge. 0 (the "
                         "default) = no term, no key, the tip's graph. Logged as `map_fp` every training step.")
    g9 = ap.add_argument_group("refcv9 R9-PF (clean-split, regime-factored progress head; REFCV9_INMODEL_DESIGN.md)")
''')

for old, new in EDITS:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:300]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(EDITS), "edits")
