"""Run a test module WITHOUT pytest — Thor's venvs are JetPack-only by policy
(the two-venv rule) and neither carries pytest. A minimal shim supplies the two
pytest surfaces this module uses (`raises`, `fixture`-free tmp_path), so the real
ASSERTIONS are executed here; the authoritative `pytest -q` still runs in the repo.
"""
import sys, types, traceback, tempfile, contextlib, re, os
from pathlib import Path

shim = types.ModuleType("pytest")
class _Raises:
    def __init__(self, exc, match=None): self.exc, self.match = exc, match
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None: raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(t, self.exc): return False
        if self.match and not re.search(self.match, str(v)):
            raise AssertionError(f"raised {t.__name__} but message {str(v)!r} "
                                 f"does not match {self.match!r}")
        return True
shim.raises = lambda exc, match=None: _Raises(exc, match)
shim.skip = lambda *a, **k: None
sys.modules["pytest"] = shim

repo = Path.home() / "TanitAD"
sys.path[:0] = [str(repo / "stack"), str(repo / "taniteval"),
                str(repo / "stack" / "scripts")]
os.chdir(repo / "stack")

import importlib
mod = importlib.import_module(sys.argv[1])
names = [n for n in dir(mod) if n.startswith("test_")]
ok = fail = 0
for n in sorted(names):
    fn = getattr(mod, n)
    kw = {}
    if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
        kw["tmp_path"] = Path(tempfile.mkdtemp())
    try:
        fn(**kw); print(f"PASS  {n}"); ok += 1
    except Exception:
        print(f"FAIL  {n}"); traceback.print_exc(); fail += 1
print(f"\n{ok} passed, {fail} failed  ({len(names)} collected)")
sys.exit(1 if fail else 0)
