# -*- coding: utf-8 -*-
"""Replace the single-run cost table with the two-run range + conservative bound."""
import pathlib

T = pathlib.Path(r"C:\Users\Admin\tanitad-perbatch\stack\tanitad\rl\refc_adapter.py")
s = T.read_text(encoding="utf-8")

OLD = """    \u26d4 AND THE COST WAS MEASURED, NOT ASSERTED (CPU, this box, min of 7x2000 reps):

    ======================================  ==========  ==================  ===================
    path                                    per call    vs smoke forward    vs deployed forward
    ======================================  ==========  ==================  ===================
    ``conditioning_requirements`` ONCE        2.535 us         0.0629 %            0.00058 %
    ``ConditioningContract.check`` EVERY      0.184 us         0.0046 %            0.00004 %
    ``assert_conditioning`` UNCACHED          3.071 us         0.0762 %            0.00070 %
    ``RefCV3Model.forward`` smoke B=1         4.030 ms            \u2014                    \u2014
    ``RefCV3Model.forward`` default B=1     436.692 ms            \u2014                    \u2014
    ======================================  ==========  ==================  ===================

    \u26a0\ufe0f REPORTED HONESTLY: the once-only optimisation **was never paying for itself**.
    One check is **1/21,937 of the *smoke* forward** \u2014 the smallest forward in the repo,
    used here as a conservative floor \u2014 and 1/2,377,009 of the deployed-scale one. Even
    the fully UNCACHED call would have been 0.076 % of that smoke forward. The cache is
    kept because it is strictly cheaper (16.7x on the per-batch path) and because resolving
"""

NEW = """    \u26d4 AND THE COST WAS MEASURED, NOT ASSERTED. CPU-only, this box, TWO independent
    runs reported as a range (min of 7 timed blocks; 20,000 reps for the per-batch path,
    2,000 for the resolve paths, 5 for the forwards):

    ========================================  ==================  =====================
    path                                      per call            share of one forward
    ========================================  ==================  =====================
    ``conditioning_requirements`` ONCE        2.514 - 2.535 us    0.063 - 0.078 % smoke
    ``ConditioningContract.check`` EVERY      0.181 - 0.184 us    0.0046 - 0.0056 % smoke
    ``assert_conditioning`` UNCACHED          3.071 - 3.124 us    0.076 - 0.097 % smoke
    ``RefCV3Model.forward`` smoke B=1         3.219 - 4.030 ms    \u2014
    ``RefCV3Model.forward`` default B=1     435.296 - 436.692 ms  \u2014
    ========================================  ==================  =====================

    \u2b50 THE CONSERVATIVE BOUND (slowest observed check against the fastest observed
    forward, which is also the SMALLEST forward in the repo): one per-batch check is
    **1/17,495 of a smoke forward** and **1/2,365,739 of a deployed-scale one**.

    \u26a0\ufe0f REPORTED HONESTLY: the once-only optimisation **was never paying for itself**.
    Even the fully UNCACHED call is at most 0.097 % of that smoke forward. The cache is
    kept because it is strictly cheaper (17x on the per-batch path) and because resolving
"""

assert OLD in s, "cost-table anchor not found"
T.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
print("cost table replaced with the two-run range")
