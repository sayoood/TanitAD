"""pytest plugin (loaded with ``-p zz_probe``, lives OUTSIDE both trees): records, from INSIDE the test run, which file every package the suite imports was loaded from."""
import json
import os
import sys


def _where():
    out = {}
    for name in ("tanitad", "taniteval", "v8f_mini_train", "v8f_kscore", "v8f_dagger", "v8f_mini_eval_kindx"):
        m = sys.modules.get(name)
        out[name] = getattr(m, "__file__", None) if m is not None else None
    for name in ("tanitad.models.v8f_mini", "tanitad.data.v8f_mini_data", "tanitad.data.v8f_dagger"):
        m = sys.modules.get(name)
        out[name] = getattr(m, "__file__", None) if m is not None else None
    return out


def pytest_report_header(config):
    try:
        import tanitad  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"zz_probe: import tanitad FAILED: {e!r}"
    return f"zz_probe: cwd={os.getcwd()} tanitad.__file__={sys.modules['tanitad'].__file__}"


def pytest_sessionfinish(session, exitstatus):
    tree = os.path.normcase(os.path.abspath(os.getcwd()))
    w = _where()
    inside = {k: (v is not None and os.path.normcase(os.path.abspath(v)).startswith(tree)) for k, v in w.items() if v}
    rec = {"cwd": os.getcwd(), "exitstatus": int(exitstatus), "loaded_from": w, "inside_tree": inside, "all_inside_tree": all(inside.values()) if inside else None,
           "sys_path_head": sys.path[:6]}
    path = os.environ.get("ZZ_PROBE_OUT")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
