import sys, pathlib
sys.path.insert(0, str(pathlib.Path("C:/Users/Admin/tanitad-wt/stack/scripts")))
import torch
import refc_v3_train as t
from tanitad.refs import refc_v3 as v3

def cfg(v0=True, hier=True):
    c = v3.refc_v3_sized_config("tiny", hier=hier)
    c.core.anchors.v0_conditioned = v0
    return c

def args(**kw):
    base = ["--arm", "hier", "--out", "/tmp/x"]
    for k, v in kw.items():
        f = "--" + k.replace("_", "-")
        if v is True: base.append(f)
        elif v not in (False, None): base += [f, str(v)]
    return t.build_parser().parse_args(base)

print("=== CONTROL: a HONEST default run must PASS (or the guard is vacuous) ===")
c = cfg(); a = args(); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c); st = t._seam_stamp(c, a)
t.assert_seams_are_built(m, st)
print("  PASS  sampler=%r cross_agent=%r agents=%r" % (st["sampler"], st["cross_agent"], st["agents"]))

print()
print("=== CONTROL 2: an HONEST ddim run must PASS ===")
c = cfg(); a = args(sampler="ddim", w_u0=0.5); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c); st = t._seam_stamp(c, a)
t.assert_seams_are_built(m, st)
print("  PASS  sampler=%r control_head=%s" % (st["sampler"], type(m.core.decoder.control_head).__name__))

print()
print("=== MUTATION 1: THE EXACT PRE-FIX DEFECT -- stamp 'ddim' on a model with no denoiser ===")
c = cfg(); a = args(); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c)                      # built with sampler='none'
st = t._seam_stamp(c, a); st["sampler"] = "ddim"; st["w_u0"] = 0.5   # <- the lie
try:
    t.assert_seams_are_built(m, st); print("  !!! GUARD DID NOT FIRE -- INSPECTION, NOT A GUARD")
except SystemExit as e:
    print("  REJECTED:"); print("   ", str(e).replace("\n", "\n    ")[:600])

print()
print("=== MUTATION 2: a LIVE sampler absent from the record (the other direction) ===")
c = cfg(); a = args(sampler="ddim", w_u0=0.5); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c); st = t._seam_stamp(c, a); st["sampler"] = "none"; st["w_u0"] = 0.0
try:
    t.assert_seams_are_built(m, st); print("  !!! GUARD DID NOT FIRE")
except SystemExit as e:
    print("  REJECTED:", str(e).split("\n")[-1].strip()[:180])

print()
print("=== MUTATION 3: agents stamped, no head built ===")
c = cfg(); a = args(); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c); st = t._seam_stamp(c, a)
st["agents"] = {"enable": True, "oracle": True}; st["cross_agent"] = True
try:
    t.assert_seams_are_built(m, st); print("  !!! GUARD DID NOT FIRE")
except SystemExit as e:
    print("  REJECTED with %d findings" % (str(e).count("  - ")))
    for line in str(e).split("\n")[1:]:
        print("   ", line.strip()[:150])

print()
print("=== MUTATION 4: w_u0 > 0 with no control_head (loss silently skipped) ===")
c = cfg(); a = args(); t._pin_refcv5_seams(c, a)
m = v3.RefCV3Model(c); st = t._seam_stamp(c, a); st["w_u0"] = 1.0
try:
    t.assert_seams_are_built(m, st); print("  !!! GUARD DID NOT FIRE")
except SystemExit as e:
    print("  REJECTED:", str(e).split("\n")[-1].strip()[:180])
