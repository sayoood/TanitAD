#!/usr/bin/env python3
"""Run repo test files against the STAGED trainer (stage/stack/scripts/v8f_mini_train.py) without touching the read-only tip tree: the staged module is imported FIRST under the name
`v8f_mini_train`, so every `import v8f_mini_train as TR` in the tests gets it (subprocess-spawned trainers, if any, still run the tip file). Usage: python run_pytest_staged.py <pytest args...>"""
import importlib.util, sys
STAGED = "E:/Projects/TanitAD-artifacts/oaccflat_check/stage/stack/scripts/v8f_mini_train.py"
spec = importlib.util.spec_from_file_location("v8f_mini_train", STAGED)
m = importlib.util.module_from_spec(spec)
sys.modules["v8f_mini_train"] = m
spec.loader.exec_module(m)
assert "o_interface" in m.INIT_CKPT_MATCH, "the staged trainer is not the fixed one"
import pytest
sys.exit(pytest.main(sys.argv[1:]))
