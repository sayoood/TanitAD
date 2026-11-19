"""A minimal stand-in for pytest, for a pod whose Python has no pytest (nothing is installed into its venv).

It covers only what the three test files of the DSCALE-LEAD kit use: ``pytest.fixture(scope="module")``, the built-in fixtures ``tmp_path`` and
``tmp_path_factory``, ``pytest.raises``, ``pytest.skip``, ``pytest.main`` and the command line ``python -m pytest [-q] [-x] [-p ...] FILE [--basetemp DIR] [-k EXPR]``.
It never deletes anything (real pytest clears ``--basetemp``; this runner only creates folders). Exit code 0 = every selected test passed, 1 = a test failed,
5 = no test selected (as pytest). A run under this stand-in is said to be one wherever its result is quoted.
"""
import importlib.util
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

__version__ = "0.0-standin"


class Skipped(Exception):
    pass


def skip(reason=""):
    raise Skipped(reason)


def fixture(*args, scope="function", **kw):
    def deco(fn):
        fn._fx_scope = scope
        return fn
    if len(args) == 1 and callable(args[0]) and not kw:
        return deco(args[0])
    return deco


class _Raises:
    def __init__(self, exc):
        self.exc, self.value = exc, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("DID NOT RAISE %r" % (self.exc,))
        if issubclass(et, self.exc):
            self.value = ev
            return True
        return False


def raises(exc):
    return _Raises(exc)


class _TmpFactory:
    def __init__(self, base):
        self.base, self.n = Path(base), 0

    def mktemp(self, basename, numbered=True):
        p = self.base / ("%s%d" % (basename, self.n) if numbered else basename)
        self.n += 1
        p.mkdir(parents=True, exist_ok=False)
        return p

    def getbasetemp(self):
        return self.base


def _match(expr, name):
    """the -k subset: substrings joined by `or` / `and`, each optionally preceded by `not` (case-insensitive)."""
    if not expr:
        return True
    name = name.lower()

    def term(t):
        t = t.strip()
        if t.startswith("not "):
            return t[4:].strip().lower() not in name
        return t.lower() in name
    return any(all(term(t) for t in part.split(" and ")) for part in expr.split(" or "))


def _resolve(name, mod, cache, base, test_name, gens):
    if name == "tmp_path":
        p = Path(base) / test_name
        k = 0
        while p.exists():
            k += 1
            p = Path(base) / ("%s_%d" % (test_name, k))
        p.mkdir(parents=True)
        return p
    if name == "tmp_path_factory":
        if "__factory__" not in cache:
            cache["__factory__"] = _TmpFactory(base)
        return cache["__factory__"]
    fn = getattr(mod, name, None)
    if fn is None or not hasattr(fn, "_fx_scope"):
        raise RuntimeError("stand-in runner: no fixture named %r" % name)
    shared = fn._fx_scope in ("module", "session")
    if shared and name in cache:
        return cache[name]
    args = [_resolve(a, mod, cache, base, test_name, gens) for a in inspect.signature(fn).parameters]
    val = fn(*args)
    if inspect.isgenerator(val):
        g = val
        val = next(g)
        gens.append(g)
    if shared:
        cache[name] = val
    return val


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    files, kexpr, base, stop_first = [], "", None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-k":
            kexpr, i = argv[i + 1], i + 1
        elif a == "--basetemp":
            base, i = argv[i + 1], i + 1
        elif a == "-p":
            i += 1
        elif a == "-x":
            stop_first = True
        elif a.startswith("-"):
            pass
        else:
            files.append(a)
        i += 1
    if base is None:
        raise SystemExit("stand-in runner: --basetemp DIR is required (nothing is written to a default temp folder)")
    os.makedirs(base, exist_ok=True)
    n_pass = n_fail = n_skip = n_sel = 0
    t0 = time.time()
    for f in files:
        f = os.path.abspath(f)
        d = os.path.dirname(f)
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location(Path(f).stem, f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        cache, shared_gens = {}, []
        tests = [(n, o) for n, o in vars(mod).items() if n.startswith("test_") and inspect.isfunction(o) and o.__module__ == mod.__name__]
        tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
        for name, fn in tests:
            if not _match(kexpr, name):
                continue
            n_sel += 1
            gens = []
            try:
                args = [_resolve(a, mod, cache, base, name, gens) for a in inspect.signature(fn).parameters]
                fn(*args)
                n_pass += 1
            except Skipped as e:
                n_skip += 1
                print("SKIPPED %s: %s" % (name, e))
            except BaseException as e:      # a test that exits (SystemExit) outside pytest.raises has failed
                if isinstance(e, KeyboardInterrupt):
                    raise
                n_fail += 1
                print("FAILED %s::%s" % (os.path.basename(f), name))
                traceback.print_exc(limit=6)
                if stop_first:
                    break
            finally:
                for g in gens:
                    if g in shared_gens:
                        continue
                    try:
                        next(g)
                    except StopIteration:
                        pass
        if n_fail and stop_first:
            break
    print("%d passed, %d failed, %d skipped in %.1f s (stand-in runner, not pytest)" % (n_pass, n_fail, n_skip, time.time() - t0))
    sys.stdout.flush()
    if n_fail:
        return 1
    return 0 if n_sel else 5
