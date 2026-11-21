"""Minimal stand-in for the pieces of pytest the A2 tests use (importorskip, approx, raises, fixture, mark.parametrize no-op).  Only put on sys.path by run_tests.py when
the real pytest is not importable (the driverl-eval venv has none).  Not a pytest replacement."""
import contextlib
import importlib
import math


class Skipped(Exception):
    pass


def importorskip(name):
    try:
        return importlib.import_module(name)
    except Exception as e:                                           # noqa: BLE001
        raise Skipped(f"missing {name}: {e}")


class approx:
    def __init__(self, v, rel=1e-6, abs=1e-12):
        self.v, self.rel, self.abs = v, rel, abs

    def __eq__(self, o):
        return math.isclose(float(o), float(self.v), rel_tol=self.rel, abs_tol=max(self.abs, 1e-9))

    def __repr__(self):
        return f"approx({self.v})"


@contextlib.contextmanager
def raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None:
            import re
            assert re.search(match, str(e)), f"{match!r} not in {e}"
        return
    raise AssertionError(f"DID NOT RAISE {exc}")


def fixture(*a, **k):
    if a and callable(a[0]):
        return a[0]
    return lambda f: f


class _Mark:
    def __getattr__(self, name):
        return lambda *a, **k: (lambda f: f)


mark = _Mark()
