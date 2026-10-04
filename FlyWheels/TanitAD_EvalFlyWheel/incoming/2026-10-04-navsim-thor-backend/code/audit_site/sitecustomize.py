"""File-open audit, injected into a scorer process via PYTHONPATH (Thor has no strace).

Active ONLY when ``THOR_AUDIT_OPENS_OUT`` is set; otherwise this module does nothing. Records, for
the whole life of the interpreter that imported it (the devkit wrapper process), every PEP 578
``open`` / ``os.listdir`` / ``os.scandir`` event, plus the paths handed to ``pyogrio.read_dataframe``
/ ``pyogrio.list_layers`` (GDAL opens the map ``.gpkg`` from C, invisible to the audit hook). Writes
one JSON per process (``<out>.<pid>.json``) at exit, summarised per data root.
Drivers add this directory to PYTHONPATH when ``THOR_AUDIT_SITE`` names it (score_arm_thor.py /
score_navtest_thor.py); production runs never set it.
"""
import os

_OUT = os.environ.get("THOR_AUDIT_OPENS_OUT")
if _OUT:
    import atexit
    import json
    import sys

    _ROOTS = [r for r in os.environ.get("THOR_AUDIT_ROOTS", "/dev/shm/navsim/exp:/dev/shm/navsim/data").split(":") if r]
    _opened, _listed = {}, {}

    def _hook(event, args):
        try:
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                p = os.fsdecode(args[0])
                _opened[p] = _opened.get(p, 0) + 1
            elif event in ("os.listdir", "os.scandir"):
                p = os.fsdecode(args[0]) if args and args[0] is not None else "."
                _listed[p] = _listed.get(p, 0) + 1
        except Exception:                                              # noqa: BLE001
            pass

    def _dump():
        def under(r, d):
            return sorted(p for p in d if os.path.abspath(p).startswith(r.rstrip("/") + "/"))
        rep = {"argv": sys.argv, "pid": os.getpid(), "roots": _ROOTS,
               "per_root": {r: {"n_files_opened": len(under(r, _opened)), "n_dirs_listed": len(under(r, _listed)),
                                "files_opened": under(r, _opened), "dirs_listed": under(r, _listed)}
                            for r in _ROOTS},
               "n_open_events_total": sum(_opened.values()),
               "note": "PEP 578 audit hook (Python-level opens) + pyogrio entry points (GDAL .gpkg opens)"}
        try:
            with open(f"{_OUT}.{os.getpid()}.json", "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=1)
        except Exception:                                              # noqa: BLE001
            pass

    sys.addaudithook(_hook)
    atexit.register(_dump)

    def _wrap_pyogrio():
        try:
            import pyogrio
        except Exception:                                              # noqa: BLE001
            return

        def _w(fn):
            def w(path, *a, **k):
                try:
                    p = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path)
                    _opened[p] = _opened.get(p, 0) + 1
                except Exception:                                      # noqa: BLE001
                    pass
                return fn(path, *a, **k)
            return w
        pyogrio.read_dataframe = _w(pyogrio.read_dataframe)
        pyogrio.list_layers = _w(pyogrio.list_layers)

    _wrap_pyogrio()
