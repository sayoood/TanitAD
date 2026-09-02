#!/usr/bin/env python
"""E-ARCH-TSC-2 arm launcher (dev box, RTX 4060).

Runs ``stack/scripts/refa_v1_train.py::main`` IN-PROCESS and UNMODIFIED from the private
HEAD-blob mirror ``C:/Users/Admin/tsc2/stack`` (refa_v1.py = HEAD blob 8264ded8, NOT the
Task-C worktree edit), and banks the run's provenance into ``<out>/config.json``:
the exact trainer argv, the git blob ids of the three load-bearing files, which ``tanitad``
tree was imported, torch/GPU, wall time, and ``torch.cuda.max_memory_allocated()`` — the
only admissible device-memory number on this programme.

Nothing in stack/ is edited. ``--mem-fraction`` (optional) caps the caching allocator so a
run that would silently over-commit into WDDM shared host memory fails as a real OOM
instead (the 2026-09-02 step-profile trap).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

STACK = r"C:\Users\Admin\tsc2\stack"
COLAB = r"C:\Users\Admin\tanitad-wt\colab"
for p in (os.path.join(STACK, "scripts"), COLAB, STACK):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import tanitad  # noqa: E402

_imported = os.path.normcase(os.path.abspath(tanitad.__file__))
if not _imported.startswith(os.path.normcase(STACK)):
    raise SystemExit(f"WRONG TREE IMPORTED: tanitad from {tanitad.__file__}, want {STACK}")

LOAD_BEARING = {
    "refa_v1.py": os.path.join(STACK, "tanitad", "refs", "refa_v1.py"),
    "refa_v1_train.py": os.path.join(STACK, "scripts", "refa_v1_train.py"),
    "refav1_loader.py": os.path.join(STACK, "tanitad", "data", "refav1_loader.py"),
}


def blob_ids(path: str) -> dict:
    raw = open(path, "rb").read()
    lf = raw.replace(b"\r\n", b"\n")
    def _sha(b):
        return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    try:
        git = subprocess.run(["git", "hash-object", path], capture_output=True,
                             text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        git = f"ERR {e!r}"
    return {"git_hash_object": git, "sha1_blob_lf": _sha(lf), "sha1_blob_raw": _sha(raw),
            "bytes": len(raw)}


def gpu_apps() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="B_prime | E | A_prime | R7 | fit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--note", default="")
    a, trainer_argv = ap.parse_known_args()
    trainer_argv = [x for x in trainer_argv if x != "--"]   # the launcher/trainer separator
    if "--out" in trainer_argv:
        raise SystemExit("pass --out to the launcher only")
    trainer_argv = list(trainer_argv) + ["--out", a.out]

    os.makedirs(a.out, exist_ok=True)
    if a.mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    torch.cuda.reset_peak_memory_stats()

    rec = {
        "arm": a.arm, "note": a.note,
        "spec": "E-ARCH-TSC-2 (TanitAD Research Lab/Architecture & Inference/Research/"
                "2026-09-02-refav1-ema-inflation/SPEC.md)",
        "tier": "T0 (mechanism probe; 20 EVAL clips as training data; model discarded)",
        "trainer": "stack/scripts/refa_v1_train.py::main (in-process, unmodified)",
        "trainer_argv": trainer_argv,
        "launcher_argv": sys.argv,
        "stack_root": STACK, "tanitad_file": tanitad.__file__,
        "blobs": {k: blob_ids(v) for k, v in LOAD_BEARING.items()},
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mem_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9
                             if torch.cuda.is_available() else None),
        "mem_fraction_cap": a.mem_fraction,
        "host": platform.node(), "platform": platform.platform(),
        "gpu_compute_apps_before": gpu_apps(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONIOENCODING",
                                                "OMP_NUM_THREADS")},
    }
    (lambda: open(os.path.join(a.out, "config.json"), "w", encoding="utf-8")
     .write(json.dumps(rec, indent=1)))()

    import refa_v1_train  # noqa: E402  (stack/scripts on sys.path)
    t0 = time.time()
    status, err = "running", None
    try:
        rc = refa_v1_train.main(trainer_argv)
        status = f"done rc={rc}"
    except SystemExit as e:
        status, err = f"SystemExit code={e.code!r}", traceback.format_exc()
        rc = 1 if e.code not in (0, None) else 0
    except BaseException as e:  # noqa: BLE001
        status, err = f"EXC {type(e).__name__}: {e}", traceback.format_exc()
        rc = 2
    finally:
        wall = time.time() - t0
        try:
            mm = torch.cuda.max_memory_allocated() / 1e9
            mr = torch.cuda.max_memory_reserved() / 1e9
        except Exception:  # noqa: BLE001
            mm = mr = None
        rec.update({"status": status, "error": err, "wall_s": round(wall, 1),
                    "cuda_max_memory_allocated_gb": mm,
                    "cuda_max_memory_reserved_gb": mr,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        open(os.path.join(a.out, "config.json"), "w", encoding="utf-8").write(
            json.dumps(rec, indent=1))
        print(f"[run_arm] {a.arm}: {status}; wall {wall:.1f}s; "
              f"cuda_max_memory_allocated {mm} GB; reserved {mr} GB", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
