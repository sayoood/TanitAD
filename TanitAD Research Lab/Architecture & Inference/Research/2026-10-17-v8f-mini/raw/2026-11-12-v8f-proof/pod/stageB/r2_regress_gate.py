"""r2_regress_gate.py -- control r2's deliberate-regression arm (HIERREP_DEFS R16).  Runs the trainer's OWN main() with the trainer's own argv (the HIER argv + --selfcheck-only,
--o-interface absent = full) after replacing the trainer module's make_config by a wrapper that builds the config with o_interface="cut".  The model is then BUILT with the cut while the
argv DECLARES full: the trainer's declared-vs-built gate must refuse (rc != 0, 'ZZDVB-REFUSEDZZ ... o_interface').  The chain asserts that; rc 0 here means the gate did not see the
mismatch and the chain stops.  No file of the trainer is edited.   usage: TREE=/path/to/tree python r2_regress_gate.py <trainer argv...>"""
import dataclasses
import os
import sys

tree = os.environ.get("TREE", "/workspace/v8proof/tree")
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
import v8f_mini_train as TR  # noqa: E402

_orig = TR.make_config


def _wrong(a):
    return dataclasses.replace(_orig(a), o_interface="cut")


TR.make_config = _wrong
rc = TR.main(sys.argv[1:])
print("R2-REGRESSION: the gate PASSED a model built with cut under an argv declaring full (rc %s)" % rc)
sys.exit(0 if rc in (0, None) else rc)
