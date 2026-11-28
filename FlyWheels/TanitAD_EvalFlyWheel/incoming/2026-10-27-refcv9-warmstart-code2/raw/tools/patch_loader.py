"""refcv7_loader.py (LF working copy): replay --r9-train-only's freeze and --w-map-fp's model attribute, as train() does."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
EDITS = [
    ('''    model._w_map_hires = float(args.w_map_hires)                        # train():8029
''', '''    model._w_map_hires = float(args.w_map_hires)                        # train():8029
    _mfp = getattr(tr, "_mfp", None)                                    # train(): refcv9 --w-map-fp (absent at 0)
    if _mfp is not None and _mfp.is_on(args):
        setattr(model, _mfp.MODEL_ATTR, _mfp.weight(args))
'''),
    ('''    if bool(getattr(args, "r8_train_only_map", False)):
        rec["map_only_ft"] = tr._mof.apply_freeze(model)
''', '''    if bool(getattr(args, "r8_train_only_map", False)):
        rec["map_only_ft"] = tr._mof.apply_freeze(model)
    elif callable(getattr(tr._mof, "is_r9", None)) and tr._mof.is_r9(args):
        # refcv9 warm start (PI 2026-10-27): `--r9-train-only <sets>` -- the same replay, the named sets
        rec["r9_train_only"] = tr._mof.apply_train_only(model, tr._mof.r9_sets(args))
'''),
]
for old, new in EDITS:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:300]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(EDITS), "edits")
