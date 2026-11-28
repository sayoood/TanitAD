"""minipytest.py -- a plain-python stand-in for the parts of pytest the lever tests use (the pod has no pytest).

Provides, as the module `pytest` (installed into sys.modules by `install()`): fixture (module scope cached), raises(exc, match=), mark.parametrize, approx-free.
Built-in fixtures: tmp_path_factory (mktemp), tmp_path, monkeypatch (setattr / setenv / undo).
`run_module(path, names=None, tmp_root=...)` imports the test file, runs the named test functions (default all test_*) and returns {name: {"ok": bool, "err": str}}.
"""
import contextlib
import importlib.util
import inspect
import itertools
import os
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

_CNT = itertools.count()


def fixture(*a, **kw):
    def deco(fn):
        fn._is_fixture = True
        fn._scope = kw.get("scope", "function")
        return fn
    if a and callable(a[0]) and not kw:
        return deco(a[0])
    return deco


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("DID NOT RAISE %r" % (self.exc,))
        if not issubclass(et, self.exc):
            return False
        if self.match is not None and not re.search(self.match, str(ev)):
            raise AssertionError("raised %r but message did not match %r: %s" % (et, self.match, str(ev)[:200]))
        return True


class _Mark:
    def parametrize(self, names, values):
        def deco(fn):
            fn._param = (names, list(values))
            return fn
        return deco

    def __getattr__(self, n):
        return lambda *a, **k: (lambda f: f)


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value=None, raising=True):
        if isinstance(obj, str):
            mod, _, attr = obj.rpartition(".")
            value, name = name, attr
            obj = importlib.import_module(mod)
        self._undo.append((obj, name, getattr(obj, name, None), hasattr(obj, name)))
        setattr(obj, name, value)

    def setitem(self, d, k, v):
        self._undo.append((d, k, d.get(k), k in d))
        d[k] = v

    def setenv(self, k, v):
        self._undo.append((os.environ, k, os.environ.get(k), k in os.environ))
        os.environ[k] = str(v)

    def undo(self):
        for obj, name, old, had in reversed(self._undo):
            if isinstance(obj, dict) or obj is os.environ:
                (obj.__setitem__(name, old) if had else obj.pop(name, None))
            elif had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self._undo = []


class _Factory:
    def __init__(self, root):
        self.root = Path(root)

    def mktemp(self, name, numbered=True):
        p = self.root / ("%s%d" % (name, next(_CNT)))
        p.mkdir(parents=True, exist_ok=True)
        return p


def install():
    m = types.ModuleType("pytest")
    m.fixture, m.raises, m.mark = fixture, _Raises, _Mark()
    sys.modules["pytest"] = m
    return m


def run_module(path, names=None, tmp_root=None):
    install()
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="minipytest_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("mp_testmod_%d" % next(_CNT), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fixtures = {n: f for n, f in vars(mod).items() if getattr(f, "_is_fixture", False)}
    cache = {}
    factory = _Factory(tmp_root)

    def resolve(name, mp):
        if name == "tmp_path_factory":
            return factory
        if name == "tmp_path":
            return factory.mktemp("t")
        if name == "monkeypatch":
            return mp
        if name in fixtures:
            f = fixtures[name]
            if f._scope == "module" and name in cache:
                return cache[name]
            kw = {p: resolve(p, mp) for p in inspect.signature(f).parameters}
            v = f(**kw)
            if f._scope == "module":
                cache[name] = v
            return v
        raise KeyError("no fixture " + name)

    tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    if names:
        tests = [(n, f) for n, f in tests if n in names]
    res = {}
    for n, f in tests:
        mp = _MonkeyPatch()
        params = [(None, None)]
        if hasattr(f, "_param"):
            pn, pv = f._param
            params = [(pn, v) for v in pv]
        for i, (pn, pv) in enumerate(params):
            key = n if pn is None else "%s[%d]" % (n, i)
            try:
                kw = {}
                for p in inspect.signature(f).parameters:
                    if pn is not None and p in [x.strip() for x in pn.split(",")]:
                        names_ = [x.strip() for x in pn.split(",")]
                        kw[p] = pv[names_.index(p)] if len(names_) > 1 else pv
                    else:
                        kw[p] = resolve(p, mp)
                f(**kw)
                res[key] = {"ok": True, "err": ""}
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                res[key] = {"ok": False, "err": (type(e).__name__ + ": " + str(e))[:300]}
            finally:
                mp.undo()
    return res
