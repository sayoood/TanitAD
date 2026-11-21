"""G-HYG census: a pytest plugin that RECORDS every assignment of an UNDECLARED attribute on a
non-frozen config dataclass anywhere in `tanitad`, while the suite runs unchanged.

Load it with `-p hyg_recorder_plugin` (this directory on PYTHONPATH). It changes no behaviour:
the original `__setattr__` still runs; the plugin only writes one JSON line per distinct
(class, attribute, caller file:line) to $HYG_RECORD (default ./hyg_record.jsonl).

Scope: every dataclass defined in a `tanitad.*` module whose class name ends with Config,
Cfg or Flags (the objects the trainer pins and the models read), imported through the trainer
so the census sees the same module set a launch does.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

_SEEN: set = set()
_OUT = os.environ.get("HYG_RECORD", "hyg_record.jsonl")
_PATCHED: list = []


def _record(cls_name, mod, attr, frame):
    code = frame.f_code
    key = (cls_name, attr, code.co_filename, frame.f_lineno)
    if key in _SEEN:
        return
    _SEEN.add(key)
    with open(_OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"class": cls_name, "module": mod, "attr": attr,
                             "file": code.co_filename, "line": frame.f_lineno,
                             "func": code.co_name}) + "\n")


def _patch(cls):
    params = getattr(cls, "__dataclass_params__", None)
    if params is None or params.frozen or "__setattr__" in cls.__dict__:
        return
    orig = cls.__setattr__

    def __setattr__(self, name, value, _orig=orig, _cls=cls):
        fields = getattr(type(self), "__dataclass_fields__", {})
        if name not in fields and not (name.startswith("__") and name.endswith("__")):
            _record(type(self).__name__, type(self).__module__, name, sys._getframe(1))
        _orig(self, name, value)
    cls.__setattr__ = __setattr__
    _PATCHED.append(f"{cls.__module__}.{cls.__name__}")


def pytest_configure(config):
    try:
        import refc_v3_train  # noqa: F401  -- the launch's own import graph
    except Exception as e:                         # pragma: no cover
        print(f"[hyg] trainer import failed: {e!r}", file=sys.stderr)
    _patch_all()


def pytest_collection_finish(session):
    # modules a test file imports lazily exist only after collection
    _patch_all()


def _patch_all():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("tanitad") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                    and obj.__module__ == name
                    and obj.__name__.endswith(("Config", "Cfg", "Flags"))):
                _patch(obj)
    with open(_OUT + ".patched.json", "w", encoding="utf-8") as fh:
        json.dump(sorted(_PATCHED), fh, indent=1)
