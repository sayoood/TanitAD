"""TRAP 2 SWEEP: does any flag stamp a capability it does not verify?

_seam_stamp reads seam values with getattr(core.decoder, X, <default>).
If X is not a real dataclass field, the stamp records an AD-HOC ATTRIBUTE
(set by the trainer's own pin) or the DEFAULT -- i.e. a truthful-looking
provenance record for a mechanism that does not exist.
"""
import ast, dataclasses, re, sys
from pathlib import Path
from tanitad.refs import refc

STACK = Path(r"C:/Users/Admin/tanitad-review-20260906/stack")
src = (STACK/"scripts/refc_v3_train.py").read_text(encoding="utf-8", errors="replace")
print("CONTROL: refc_v3_train.py read, %d bytes, %d 'def ' (must be non-zero)"
      % (len(src), src.count("def ")))

tree = ast.parse(src)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "_seam_stamp")

dec_fields = {f.name for f in dataclasses.fields(refc.DecoderConfig)}
core_fields = {f.name for f in dataclasses.fields(refc.CoreConfig)} \
    if hasattr(refc, "CoreConfig") else set()
print("DecoderConfig fields (%d): %s" % (len(dec_fields), sorted(dec_fields)))

rows = []
for node in ast.walk(fn):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"):
        continue
    obj, attr = node.args[0], node.args[1]
    if not isinstance(attr, ast.Constant):
        continue
    tgt = ast.unparse(obj)
    name = attr.value
    if tgt.endswith("decoder"):
        real = name in dec_fields
        rows.append((tgt, name, real, "DecoderConfig"))
    elif tgt in ("core", "cfg.core"):
        real = (name in core_fields) if core_fields else None
        rows.append((tgt, name, real, "CoreConfig"))

print("\n%-18s %-22s %-8s %s" % ("read from", "attribute", "REAL?", "verdict"))
print("-"*78)
phantom = []
for tgt, name, real, where in sorted(rows, key=lambda r: (r[0], r[1])):
    v = ("REAL field" if real else
         "⛔ PHANTOM — no such field; getattr falls back to an ad-hoc attr/default"
         if real is False else "unknown")
    if real is False:
        phantom.append(name)
    print("%-18s %-22s %-8s %s" % (tgt, name, str(real), v))

print("\nPHANTOM SEAM KEYS: %d -> %s" % (len(phantom), phantom))

# ---- the direct behavioural proof: build a model and ask it ---------------
print("\nBEHAVIOURAL CHECK — set every phantom on a DecoderConfig instance:")
dc = refc.DecoderConfig()
for p in phantom:
    try:
        setattr(dc, p, "STAMPED_BUT_UNREAD")
    except Exception as e:
        print("   %s: setattr refused (%s)" % (p, type(e).__name__)); continue
    print("   %-18s setattr ACCEPTED (dataclass not frozen) -> value is stampable "
          "and nothing reads it" % p)
# CONVERSE CONTROL: a real field must round-trip AND be a declared field
print("   CONVERSE CONTROL feasible_decode in fields = %s (a real seam exists, "
      "so the sweep is not vacuous)" % ("feasible_decode" in dec_fields))
