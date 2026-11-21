#!/usr/bin/env python3
"""families6.py with ONE change of environment, none of code: its ``np.load`` returns the seam's arrays read ONCE.

WHY (MEASURED 2026-11-16, P7 read stage 3, ``stage3/p7_full_families.log``): families6.py line 72 builds
``[z["poses"][idx[t]] for t in toks]`` on a ``NpzFile``. Every ``z["poses"]`` re-reads the WHOLE array from the
zip, and the row view it returns keeps that whole copy alive until ``np.asarray`` runs: 12,146 navtest tokens x
1.11 MiB = ~13.2 GB of live copies. On the memory-tight dev box that raised ``_ArrayMemoryError`` and stage 3
lost its families block.

families6.py is sha256-pinned (``eval/harness_pins.py``) and the SFT-4 / C2 / P6 / P7 results cite it, so it is
NOT edited. This wrapper loads the pinned file unchanged as a module and gives it a numpy whose ``load`` returns an
object caching every array once. Every value families6 computes is the same value (the same bytes are read); only
the number of copies changes. ``check_families6_lowmem.py`` proves output identity against the unwrapped file.

    python families6_lowmem.py --families6 <path to the pinned families6.py> <families6's own arguments...>
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types

import numpy as np


class _CachedNpz:
    """the subset of NpzFile families6 uses: ``.files``, ``[key]``, ``in``; each array read exactly once"""

    def __init__(self, z):
        self.files = list(z.files)
        self._d = {k: z[k] for k in self.files}
        z.close()

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d


def _cached_load(path, *a, **k):
    z = np.load(path, *a, **k)
    return _CachedNpz(z) if isinstance(z, np.lib.npyio.NpzFile) else z


class _NumpyProxy(types.ModuleType):
    """numpy with ``load`` replaced; every other attribute is numpy's own"""

    def __getattr__(self, k):
        return getattr(np, k)


def load_families6(path):
    spec = importlib.util.spec_from_file_location("families6_pinned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # __name__ != "__main__": main() is not run on import
    proxy = _NumpyProxy("numpy_cached_load")
    proxy.load = _cached_load
    mod.np = proxy
    return mod


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--families6" not in argv:
        raise SystemExit("--families6 <path to the pinned families6.py> is required")
    i = argv.index("--families6")
    f6 = argv[i + 1]
    del argv[i:i + 2]
    sha = hashlib.sha256(open(f6, "rb").read()).hexdigest()
    print(f"families6_lowmem: families6={f6} sha256={sha}", flush=True)
    return load_families6(f6).main(argv)


if __name__ == "__main__":
    sys.exit(main())
