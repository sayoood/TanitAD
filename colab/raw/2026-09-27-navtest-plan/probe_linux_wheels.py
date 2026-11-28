"""Can the two eval venvs be rebuilt on a Linux Colab VM from the SAME pins? Metadata only (PyPI JSON + the
PyTorch index pages; no wheel is downloaded).

Inputs: the two `uv pip freeze` listings taken on the dev box (driverl_eval_venv_freeze.txt, Python 3.11.13;
navsim_crun_venv_freeze.txt, Python 3.9.25). For every pinned distribution: is there a wheel for Linux x86_64 at
that exact version for that CPython (cpXY manylinux/musllinux-excluded, abi3, or py3-none-any), else an sdist?
Editable installs and Windows-only distributions are listed separately. Writes linux_wheels.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import truststore

truststore.inject_into_ssl()
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "tanitad-navtest-colab-plan/1.0 (metadata probe)"}
WINDOWS_ONLY = {"colorama": "Windows console colours (also pure python, harmless)", "pywin32": "Windows API",
                "pywinpty": "Windows PTY"}
TORCH_INDEX = {"cu128": "https://download.pytorch.org/whl/cu128", "cpu": "https://download.pytorch.org/whl/cpu"}


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def parse_freeze(p):
    pins, edit = [], []
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("-e "):
            edit.append(ln[3:])
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", ln)
        if m:
            pins.append((m.group(1), m.group(2)))
    return pins, edit


def linux_ok(fn: str, cp: str) -> bool:
    if not fn.endswith(".whl"):
        return False
    f = fn[:-4].split("-")
    if len(f) < 5:
        return False
    py, abi, plat = f[-3], f[-2], f[-1]
    plat_ok = plat == "any" or ("manylinux" in plat and "x86_64" in plat) or plat == "linux_x86_64"
    py_ok = (py in (cp, "py3", "py2.py3") or cp in py.split(".") or
             (abi == "abi3" and py.startswith("cp3") and int(py[3:]) <= int(cp[3:])))
    return plat_ok and py_ok


def check(name, ver, cp):
    base = ver.split("+")[0]
    local = ver[len(base) + 1:] if "+" in ver else ""
    if name.lower() == "torch" or local:
        idx = TORCH_INDEX.get(local or "cpu")
        st, body = get(f"{idx}/{name.lower()}/")
        files = re.findall(r">([^<]+\.whl)<", body.decode("utf-8", "replace"))
        want = [f for f in files if f.startswith(f"{name.lower()}-{ver.replace('+', '%2B')}-") or
                f.startswith(f"{name.lower()}-{ver}-")]
        ok = [f for f in want if linux_ok(f.replace("%2B", "+"), cp)]
        return {"name": name, "version": ver, "index": idx, "status": st, "linux_wheels": ok[:3],
                "verdict": "wheel" if ok else "MISSING"}
    st, body = get(f"https://pypi.org/pypi/{name}/{base}/json")
    if st != 200:
        return {"name": name, "version": ver, "status": st, "verdict": "NOT_ON_PYPI"}
    urls = json.loads(body)["urls"]
    whl = [u["filename"] for u in urls if linux_ok(u["filename"], cp)]
    sdist = [u["filename"] for u in urls if u.get("packagetype") == "sdist"]
    size = max((u["size"] for u in urls if u["filename"] in whl), default=None)
    return {"name": name, "version": ver, "status": st, "linux_wheel": whl[0] if whl else None,
            "wheel_bytes": size, "sdist": sdist[0] if sdist else None,
            "verdict": "wheel" if whl else ("sdist_only" if sdist else "MISSING")}


def survey(freeze, cp):
    pins, edit = parse_freeze(os.path.join(HERE, freeze))
    todo = [(n, v) for n, v in pins if n.lower() not in WINDOWS_ONLY]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda nv: check(nv[0], nv[1], cp), todo))
    bad = [r for r in res if r["verdict"] != "wheel"]
    return {"freeze": freeze, "python_tag": cp, "n_pins": len(pins), "n_checked": len(todo),
            "windows_only_skipped": [n for n, _ in pins if n.lower() in WINDOWS_ONLY],
            "editables": edit, "n_linux_wheel": sum(r["verdict"] == "wheel" for r in res),
            "not_wheel": bad, "wheel_bytes_sum_pypi": sum(r.get("wheel_bytes") or 0 for r in res),
            "results": res}


def main() -> int:
    t0 = time.time()
    out = {"probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "driverl_eval": survey("driverl_eval_venv_freeze.txt", "cp311"),
           "navsim_crun": survey("navsim_crun_venv_freeze.txt", "cp39")}
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "linux_wheels.json"), "w", encoding="utf-8"), indent=1)
    for k in ("driverl_eval", "navsim_crun"):
        s = out[k]
        print(f"{k}: {s['n_linux_wheel']}/{s['n_checked']} have a Linux {s['python_tag']} wheel; "
              f"not-wheel: {[(r['name'], r['version'], r['verdict']) for r in s['not_wheel']]}; "
              f"editables {s['editables']}; skipped {s['windows_only_skipped']}")
    print(f"ZZWHEELS_OK {out['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
