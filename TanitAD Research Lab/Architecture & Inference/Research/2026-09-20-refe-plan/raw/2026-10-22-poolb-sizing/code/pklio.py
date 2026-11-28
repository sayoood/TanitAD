"""Restricted OpenScene-pkl reader: numpy array globals only; anything else raises (the pkl is third-party data and is never executed).
Same whitelist as the prior probe's profile_logs.ArrayOnlyUnpickler (which copies TanitAstra's). Needs numpy (use the opencv-ref-454 venv)."""
import pickle, io
import numpy as np

class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        core = getattr(np, "_core", None) or np.core
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = core.multiarray._reconstruct
            allowed[prefix, "scalar"] = core.multiarray.scalar
        for prefix in ("numpy.core.numeric", "numpy._core.numeric"):
            allowed[prefix, "_frombuffer"] = core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported class: %s.%s" % (module, name))
        return allowed[module, name]

def load_pkl_bytes(b):
    return ArrayOnlyUnpickler(io.BytesIO(b)).load()
