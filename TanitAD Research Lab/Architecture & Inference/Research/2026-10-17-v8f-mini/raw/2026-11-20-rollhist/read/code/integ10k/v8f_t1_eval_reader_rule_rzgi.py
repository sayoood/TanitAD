"""v8f_t1_eval_reader_rule_rzgi.py -- ROLLHIST read package adapter: lets the unedited readout wrapper v8f_t1_eval_reader_rule_rzg.py run as the `--inner` of the tier-of-record executor
`v8f_t1_eval_hb.py --kappa-max 0.2`.  Written because the card's form `--inner v8f_t1_eval_reader_rule_rzg` cannot compose as written (MEASURED, 8-window smoke 2026-11-21):
  1. hb REFUSES: rzg loads the pinned tool inside main() (W.load_pinned), so at hb's check sys.modules has no `v8f_t1_eval`;
  2. even if it did, W.load_pinned would re-execute the tool file into a NEW module object and drop hb's heading bound (a silent no-op).
This adapter imports the pinned tool from the tree ONCE at import (as the module `v8f_t1_eval`, the object hb patches), imports rzg unedited, and makes W.load_pinned return that same object
(after rzg's own check that `tanitad` comes from the same tree).  Everything else is rzg's.  Tree from the environment variable RH_TREE."""
import importlib.util
import os
import sys

_tree = os.environ.get("RH_TREE")
if not _tree:
    raise SystemExit("REFUSED: RH_TREE is not set (the tree the pinned tool is imported from)")
_path = os.path.abspath(os.path.join(_tree, "stack", "scripts", "v8f_t1_eval.py"))
_spec = importlib.util.spec_from_file_location("v8f_t1_eval", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["v8f_t1_eval"] = _mod
_spec.loader.exec_module(_mod)

import v8f_t1_eval_reader_rule_rzg as _R  # noqa: E402  (unedited)

_root = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(_path), "..", "..")))
if not os.path.normcase(os.path.abspath(_mod.M.__file__)).startswith(_root):
    raise SystemExit("REFUSED: tanitad was not imported from the tree " + _root)


def _load_pinned_reuse(tree):
    if os.path.normcase(os.path.abspath(os.path.join(tree, "stack", "scripts", "v8f_t1_eval.py"))) != os.path.normcase(_path):
        raise SystemExit("REFUSED: --tree differs from RH_TREE")
    if sys.modules.get("v8f_t1_eval") is not _mod:
        raise SystemExit("REFUSED: the pinned tool module was replaced")
    return _mod, _path


_R.W.load_pinned = _load_pinned_reuse
main = _R.main
