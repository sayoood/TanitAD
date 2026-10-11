"""Run the WP-RL mirror tests: main-repo `tanitad` + the mirror's tanitad/rl overlaid on tanitad.rl.__path__.

The mirror ships only tanitad/rl/ddv2_refcv7.py (no __init__.py), so a plain PYTHONPATH cannot
overlay it on the main repo's regular package. Usage: python run_mirror_tests.py <mirror stack> [pytest args]
"""
import sys
from pathlib import Path

mirror = Path(sys.argv[1]).resolve()
sys.path.insert(0, "E:/Projects/TanitAD/stack")
import tanitad.rl  # noqa: E402

tanitad.rl.__path__.insert(0, str(mirror / "tanitad" / "rl"))
import tanitad.rl.ddv2_refcv7 as R  # noqa: E402

assert Path(R.__file__).resolve().parent == (mirror / "tanitad" / "rl").resolve(), R.__file__
print("[runner] ddv2_refcv7 from", R.__file__)
print("[runner] tanitad from", tanitad.__file__)
import pytest  # noqa: E402

sys.exit(pytest.main(["-p", "no:cacheprovider", *sys.argv[2:]]))
