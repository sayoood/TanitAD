"""Shared helpers for the 2026-09-26 refcv6 frozen-trunk audit (A16).

Every instrument in this package imports this first. It
  * pins the code tree to the clean tip archive (C:/Users/Admin/cfull_tip) and ASSERTS that
    `tanitad` really imported from there (the MSYS/editable-install trap);
  * refuses to start a RAM-heavy job while the box has < MIN_FREE_GB available (brief rule 6);
  * forces CPU (CUDA_VISIBLE_DEVICES="") and caps OMP threads at 4 (brief rule 6);
  * scrubs raw clip UUIDs from anything written to disk (brief rule 8: sha12 only);
  * writes JSON ASCII-safe (cp1252 console trap).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# ⚠️ MEASURED 2026-09-26: on Windows an EMPTY env value DELETES the variable (putenv "K="), so
# CUDA_VISIBLE_DEVICES="" left the RTX 4060 visible and bootstrap()'s CPU-only assertion fired.
# "-1" is a non-empty invalid index that hides every device.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# the code tree under test is READ-ONLY (brief): no __pycache__ written into it
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TIP = Path(os.environ.get("AUDIT_TIP", "C:/Users/Admin/cfull_tip"))
STACK = TIP / "stack"
SCRIPTS = STACK / "scripts"
TANITEVAL = TIP / "taniteval"
KIT = Path("D:/refcv6_eval_kit")
CONFIG_JSON = KIT / "ckpt" / "config.json"
PKG = Path(__file__).resolve().parent.parent
#: where the READ-ONLY Thor pulls live (not banked: they carry raw clip ids).
#: Set AUDIT_SCRATCH to the directory that holds `thor_pull/`.
SCRATCH = Path(os.environ.get("AUDIT_SCRATCH", str(Path.home() / "refcv6_audit_scratch"))) / "thor_pull"
RAW = PKG / "raw"

MIN_FREE_GB = float(os.environ.get("AUDIT_MIN_FREE_GB", "8.0"))

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def scrub(obj):
    """Replace every raw clip UUID in a JSON-able object by its sha12 (brief rule 8)."""
    if isinstance(obj, str):
        return _UUID.sub(lambda m: "sha12:" + sha12(m.group(0)), obj)
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def bootstrap(stdout_utf8: bool = True) -> None:
    if stdout_utf8:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for p in (str(STACK), str(TANITEVAL), str(SCRIPTS), str(TANITEVAL / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad  # noqa: F401
    here = os.path.normcase(os.path.abspath(tanitad.__file__))
    want = os.path.normcase(os.path.abspath(str(STACK)))
    if not here.startswith(want):
        raise SystemExit(f"[audit] tanitad imported from {here}, not from {want}")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if torch.cuda.is_available():
        raise SystemExit("[audit] CUDA is visible -- this audit is CPU-only (brief rule 6)")


def ram_available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def ram_guard(what: str, need_gb: float | None = None) -> float:
    """Refuse (exit 3) when available RAM is below the brief's floor."""
    avail = ram_available_gb()
    floor = MIN_FREE_GB if need_gb is None else max(MIN_FREE_GB, need_gb)
    if avail < floor:
        print(f"[audit:RAM] REFUSED {what}: available {avail:.2f} GB < floor {floor:.1f} GB",
              flush=True)
        raise SystemExit(3)
    print(f"[audit:RAM] ok for {what}: available {avail:.2f} GB", flush=True)
    return avail


def write_json(name: str, obj) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    txt = json.dumps(scrub(obj), indent=1, ensure_ascii=True, default=str)
    if _UUID.search(txt):
        raise SystemExit(f"[audit] refusing to write {name}: a raw UUID survived the scrub")
    p.write_text(txt, encoding="utf-8")
    print(f"[audit] wrote {p} ({len(txt)} B)", flush=True)
    return p


def load_config() -> dict:
    with open(CONFIG_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def trainer_module():
    """refc_v3_train.py by path, under the SAME module name the battery loader uses."""
    import importlib.util
    name = "refc_v3_train_for_arm"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
