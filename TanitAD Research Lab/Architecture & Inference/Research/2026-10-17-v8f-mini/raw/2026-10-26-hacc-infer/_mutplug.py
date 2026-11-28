"""pytest plugin for mutate_hacc_infer.py: replaces ``tanitad.models.v8f_mini`` by the module compiled from the file named in ``$MUT_SRC`` BEFORE any test imports it
(``pytest -p _mutplug``). The tests under mutation therefore run against the mutant, and nothing in the repo tree is edited."""
import importlib.util
import os
import sys

import tanitad.models  # noqa: F401  (the package, so the attribute below lands on it)

_src = os.environ["MUT_SRC"]
_spec = importlib.util.spec_from_file_location("tanitad.models.v8f_mini", _src)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tanitad.models.v8f_mini"] = _mod
_spec.loader.exec_module(_mod)
setattr(sys.modules["tanitad.models"], "v8f_mini", _mod)
