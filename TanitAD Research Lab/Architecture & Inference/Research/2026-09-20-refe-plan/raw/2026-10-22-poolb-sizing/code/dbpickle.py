"""Shim unpickler for the nuPlan DB `camera` table blobs: the three nuplan data_types classes become inert stubs that record their state;
numpy array globals are the only other thing allowed. Lets a numpy-only venv read intrinsic / distortion / translation / rotation without importing nuplan
and without executing arbitrary pickle code."""
import pickle, io
import numpy as np
class Stub(list):
    """inert stand-in: a list (the pickle may APPEND items to it) that also records __setstate__ / constructor args"""
    def __init__(self, *a, **k): super().__init__(); self.args = a; self.kw = k
    def __setstate__(self, s): self.state = s
class DBUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        core = getattr(np, "_core", None) or np.core
        if module == "nuplan.database.common.data_types" and name in ("CameraIntrinsic", "Translation", "Rotation"):
            return Stub
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = core.multiarray._reconstruct
            allowed[prefix, "scalar"] = core.multiarray.scalar
        for prefix in ("numpy.core.numeric", "numpy._core.numeric"):
            allowed[prefix, "_frombuffer"] = core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported class: %s.%s" % (module, name))
        return allowed[module, name]
def flat(obj):
    if isinstance(obj, Stub):
        if len(obj): return np.concatenate([flat(v) for v in obj])
        s = getattr(obj, "state", None)
        if s is None: s = obj.args
        return flat(s)
    if isinstance(obj, dict): return np.concatenate([flat(v) for v in obj.values()]) if obj else np.zeros(0)
    if isinstance(obj, (list, tuple)): return np.concatenate([flat(v) for v in obj]) if len(obj) else np.zeros(0)
    return np.asarray(obj, dtype=float).ravel()
def loads(b): return flat(DBUnpickler(io.BytesIO(b)).load())
