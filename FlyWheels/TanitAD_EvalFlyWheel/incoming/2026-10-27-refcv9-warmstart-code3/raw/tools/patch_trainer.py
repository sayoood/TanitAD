"""refc_v3_train.py (LF working copy), round 3: the declared-frozen subtrees through the guard / source check / EMA, the
`tactical` set in the help, and `--r8-alloc-emit-start 0` ADMITTED under `--r9-train-only` (a warm start from refcv8's
EMA, whose extras were EMITTED at its end -- emission from step 0 IS its step-0 identity). Default absent: unchanged."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
E = []


def edit(old, new):
    E.append((old, new))


edit('''    _mof_mods = None if _mof_rec is None else tuple(_mof_rec["modules"])      # the TRAINABLE modules (None: no freeze)
''', '''    _mof_mods = None if _mof_rec is None else tuple(_mof_rec["modules"])      # the TRAINABLE modules (None: no freeze)
    # the declared grad-unreachable subtrees INSIDE them stay frozen (--r9-train-only records them; r8: none)
    _mof_decl = () if _mof_rec is None else tuple(_mof_rec.get("declared_frozen") or ())
''')
edit('''        _mof_rec["init_from"]["verify"] = _mof.verify_against_source(model, _mof_src, fresh=not ck.exists(),
                                                                     modules=_mof_mods)
        _mof_guard = _mof.FrozenStateGuard(_mof_src, modules=_mof_mods)''', '''        _mof_rec["init_from"]["verify"] = _mof.verify_against_source(model, _mof_src, fresh=not ck.exists(),
                                                                     modules=_mof_mods, declared=_mof_decl)
        _mof_guard = _mof.FrozenStateGuard(_mof_src, modules=_mof_mods, declared=_mof_decl)''')
edit('''                               exclude=(_mof.ema_exclude_for(_mof_mods) if _mof_rec is not None else None))''',
     '''                               exclude=(_mof.ema_exclude_for(_mof_mods, _mof_decl) if _mof_rec is not None
                                        else None))''')
edit('''        _mof_bad = (_mof.ema_problems(_ema, model, _mof_src, fresh=not ck.exists(), modules=_mof_mods)
                    if _ema is not None else [])''', '''        _mof_bad = (_mof.ema_problems(_ema, model, _mof_src, fresh=not ck.exists(), modules=_mof_mods,
                                      declared=_mof_decl)
                    if _ema is not None else [])''')
edit('''                         "step). Sets: `planner` = core.decoder (186 tensors / 15,365,801 values on refcv8), `map` = "
                         "the three map-only modules (= --r8-train-only-map). Needs --refcv8 --opt dd --init-from = the "
                         "SHIPPED EMA model --conflict-detector off. config.json['r9_train_only'] records the sets, the "
                         "trainable list and the counts.")''', '''                         "step). Sets: `planner` = core.decoder (178 trained tensors / 15,254,091 values on refcv8; "
                         "its declared grad-unreachable subtrees stay frozen), `tactical` = the eleven tactical-branch "
                         "modules (map_only_ft.TACTICAL_MODULES; 113 / 5,476,306), `map` = the three map-only modules "
                         "(= --r8-train-only-map). Needs --refcv8 --opt dd --init-from = the SHIPPED EMA model "
                         "--conflict-detector off. config.json['r9_train_only'] records the sets, the trainable list, "
                         "the declared-frozen subtrees and the counts.")''')
# MM ruling Q1's two refusals keep their meaning for every run EXCEPT the refcv9 warm start, which declares 0
edit('''    _es = getattr(args, "r8_alloc_emit_start", None)
    if _es is not None:
        if int(_es) < 1:''', '''    _es = getattr(args, "r8_alloc_emit_start", None)
    # ⭐ refcv9 WARM START (MM 2026-10-27, R9W-EMIT-START): the run warm-starts from refcv8's EMA model, whose extra
    # candidates were EMITTED from refcv8 step 1,000 to its end -- so emission from step 0 IS the step-0 identity of
    # the emitted plan (I-2 read against the model actually loaded), and a start >= 1 would HIDE them. Declared as an
    # explicit `--r8-alloc-emit-start 0`, admitted ONLY with --r9-train-only; absent / every other run: unchanged.
    _es0_r9 = _es is not None and int(_es) == 0 and _mof.is_r9(args)
    if _es is not None:
        if int(_es) < 1 and not _es0_r9:''')
edit('''    if getattr(args, "init_from", None) and r8train.emits_at_step0(r):
        raise SystemExit("[refcv8] ⛔ a WARM-STARTED run (--init-from) that emits extra candidates from step 0 "''',
     '''    if getattr(args, "init_from", None) and r8train.emits_at_step0(r) and not _es0_r9:
        raise SystemExit("[refcv8] ⛔ a WARM-STARTED run (--init-from) that emits extra candidates from step 0 "''')

for old, new in E:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:400]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(E), "edits")
