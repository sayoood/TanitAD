"""Did a named pytest control PASS? Read from pytest's `-rA` summary, whatever prefix the node id carries.

MEASURED 2026-10-04 (step 50,400): K0 PASSED on CUDA for both warmup and navtest. The runners matched only the
literal "PASSED tests/test_model_seam7.py::test_K0", but pytest printed the node id as
"PASSED D:/…/navsim/tests/test_model_seam7.py::test_K0…" (and once as "PASSED ::test_K0…"). The prefix depends on
pytest's rootdir, the caller's cwd, and D: being a `subst` of E: (the real path does not relativise against
the ini's dir). So the runner read a PASSED determinism control as FAILED and REFUSED CUDA, and both splits ran
on CPU fp32 (~17 h for navtest instead of ~3 h). The pytest.ini rootdir pin did not fix it.
Rule: a control passes iff a summary line reads `PASSED <anything>test_model_seam7.py::<test>`, or
`PASSED ::<test>`. A FAILED or ERROR line never counts, and neither does a mention elsewhere.
"""
from __future__ import annotations

import re


def control_passed(txt: str, test_prefix: str, module: str = "test_model_seam7.py") -> bool:
    """True iff pytest's summary has a PASSED line for a test whose name starts with `test_prefix`."""
    t = txt.replace("\\", "/")
    pat = re.compile(r"^PASSED (?:\S*/)?(?:" + re.escape(module) + r")?::" + re.escape(test_prefix),
                     re.MULTILINE)
    return pat.search(t) is not None
