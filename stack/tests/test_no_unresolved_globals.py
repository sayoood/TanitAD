"""⛔⛔ Every global name a function on the refcv6 path LOADS must exist -- a NameError is not a bug
that waits for the right input, it is a bug that waits for the FIRST real step.

⭐ MEASURED 2026-09-23 by the first real refcv6 step at 416x1024 on Thor: ``compute_losses_v3``
read ``getattr(args, "map_lift_valid_mask", True)`` and ``getattr(args, "box3d_visible_filter",
True)`` -- and ``compute_losses_v3`` has no ``args``. The first died with ``NameError`` on the first
map loss. No test reached either line: the fake perception branch carries no lift, so ``map_valid``
was ``None`` and the expression short-circuited before touching ``args``; the box path was never
reached with live targets. Two lines, both green through a full suite, both fatal on step 1.

⭐ This asks the question that does not depend on reaching the line: for every function (and
every nested code object) defined in a module on the refcv6 import path, does every
``LOAD_GLOBAL`` resolve in THAT function's own ``__globals__`` or in builtins? Resolving against
the function's own globals is what makes it exact: dataclass-generated methods are compiled
against the ``dataclasses`` namespace and would otherwise read as 400+ false positives.

The POSITIVE CONTROL plants the historical defect in a synthetic module and must be flagged.
"""
from __future__ import annotations

import builtins
import dis
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_B = set(dir(builtins))


def unresolved_globals(mod) -> list[tuple[str, str, int | None]]:
    """``(owner, name, line)`` for every LOAD_GLOBAL that resolves nowhere."""
    out: list = []

    def walk(co, owner, g):
        for ins in dis.get_instructions(co):
            if ins.opname == "LOAD_GLOBAL" and ins.argval not in g and ins.argval not in _B:
                out.append((owner, ins.argval,
                            ins.positions.lineno if ins.positions else None))
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                walk(c, owner, g)

    def fn(f, owner):
        if isinstance(f, types.FunctionType) and f.__globals__ is vars(mod):
            walk(f.__code__, owner, f.__globals__)

    for name, obj in list(vars(mod).items()):
        if isinstance(obj, types.FunctionType) and obj.__module__ == mod.__name__:
            fn(obj, name)
        elif isinstance(obj, type) and obj.__module__ == mod.__name__:
            for an, av in vars(obj).items():
                if isinstance(av, (staticmethod, classmethod)):
                    fn(av.__func__, f"{name}.{an}")
                elif isinstance(av, property):
                    for acc in (av.fget, av.fset):
                        fn(acc, f"{name}.{an}")
                else:
                    fn(av, f"{name}.{an}")
    return out


def test_POSITIVE_CONTROL_the_sweep_flags_the_historical_defect():
    """⛔ Without this, a sweep that looked at nothing would read 0 forever."""
    m = types.ModuleType("synthetic_trainer")
    src = ("def compute_losses_v3(model, batch):\n"
           "    return getattr(args, 'map_lift_valid_mask', True)\n"
           "def fine(model):\n"
           "    return getattr(model, '_map_lift_valid_mask', True)\n")
    exec(compile(src, "synthetic_trainer.py", "exec"), vars(m))
    got = unresolved_globals(m)
    assert [(o, n) for o, n, _ in got] == [("compute_losses_v3", "args")]


def test_the_refcv6_import_path_has_NO_unresolved_globals():
    import refc_v3_train  # noqa: F401 -- pulls in the whole refcv6 import closure
    mods = sorted((m for n, m in sys.modules.items()
                   if m is not None and getattr(m, "__file__", None)
                   and (n.startswith("tanitad") or n in ("refc_v3_train",
                                                          "train_p8_occupancy",
                                                          "build_obstacle_join",
                                                          "refb_train", "refc_train"))),
                  key=lambda m: m.__name__)
    assert len(mods) >= 40, len(mods)          # the sweep really saw the import path
    bad = [(m.__name__,) + x for m in mods for x in unresolved_globals(m)]
    assert bad == [], bad
