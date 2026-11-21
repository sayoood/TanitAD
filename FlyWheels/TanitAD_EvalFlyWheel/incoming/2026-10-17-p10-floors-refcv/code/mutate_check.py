import sys, numpy as np, importlib
sys.path.insert(0, "E:/Projects/TanitAD/taniteval/tests")
import test_floors_kin as T
from taniteval import floors_kin as fk
def red(name, fn_test, patch):
    orig = {}
    for k, v in patch.items():
        orig[k] = getattr(fk, k); setattr(fk, k, v)
    try:
        fn_test(); return f"{name}: STAYED GREEN (BAD)"
    except (AssertionError, Exception) as e:
        return f"{name}: RED as required ({type(e).__name__})"
    finally:
        for k, v in orig.items(): setattr(fk, k, v)
print(red("damp50 0.6/0.4", T.test_damp50_is_half_ha0_plus_half_ha0_ext_exactly, {"damp50": lambda a,b: 0.6*np.asarray(a)+0.4*np.asarray(b)}))
print(red("ha0_ext swaps a<->k", T.test_constant_velocity_input_reproduces_ha0_exactly,
          {"ha0_ext_paths": lambda kin, dt, n, action_units="kappa": fk.const_control_paths(np.asarray(kin)[:,0], np.asarray(kin)[:,0]*0.1, np.asarray(kin)[:,2], dt, n, action_units=action_units)}))
orig_retime = fk.retime
def bad_retime(path, lon_src, chunk=2048):
    r = orig_retime(path, lon_src, chunk=chunk); return r * 1.01
print(red("retime +1%", T.test_retime_vectorised_equals_original_scalar_retime, {"retime": bad_retime}))
print(red("kin_features drops a*k", T.test_kin_features_known_value, {"kin_features": lambda kin: fk.kin_features.__wrapped__(kin) if hasattr(fk.kin_features,'__wrapped__') else np.zeros((1,8))}))
print(red("t0_kin forward diff", T.test_nothing_after_t0_enters_the_features,
          {"t0_kinematics": lambda v, steer, t0, stride, dt: np.stack([np.asarray(v)[np.asarray(t0)], (np.asarray(v)[np.asarray(t0)+1]-np.asarray(v)[np.asarray(t0)])/dt, np.asarray(steer)[np.asarray(t0)]], -1)}))
