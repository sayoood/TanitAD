"""SHIM of ``scipy.ndimage.binary_dilation`` for the E1 read (see ``scipy/__init__.py``). numpy only.

Implements the one call ``r9_e1_eval.nearfield_per_file`` makes -- a 2-D boolean mask, scipy's DEFAULT structuring element (the
4-connected cross, ``generate_binary_structure(2, 1)``), ``iterations >= 1``, zero border -- and REFUSES (NotImplementedError)
every other argument rather than approximating it: an unsupported call must fail loud, not return a plausible mask.
"""
import sys

import numpy as np

IS_SHIM = True
print("[scipy-shim] scipy.ndimage.binary_dilation is the numpy shim (the venv has no scipy)", file=sys.stderr, flush=True)


def binary_dilation(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
    if structure is not None or mask is not None or output is not None or border_value != 0 or origin != 0:
        raise NotImplementedError("scipy shim: only binary_dilation(input, iterations=n) with the default structure is implemented")
    if int(iterations) != iterations or iterations < 1:
        raise NotImplementedError("scipy shim: iterations must be an integer >= 1 (scipy's 0 = 'until stable' is not implemented)")
    a = np.asarray(input).astype(bool)
    if a.ndim != 2:
        raise NotImplementedError("scipy shim: 2-D input only")
    for _ in range(int(iterations)):
        b = a.copy()
        b[1:, :] |= a[:-1, :]
        b[:-1, :] |= a[1:, :]
        b[:, 1:] |= a[:, :-1]
        b[:, :-1] |= a[:, 1:]
        a = b
    return a
