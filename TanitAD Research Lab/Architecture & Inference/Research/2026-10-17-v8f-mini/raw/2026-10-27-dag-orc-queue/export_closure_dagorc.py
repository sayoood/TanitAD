#!/usr/bin/env python3
"""What does box 2's DAG-ORC queue READ from the repo?  The DAG2 build's static import closure (E:/Projects/TanitAD-artifacts/dag2q/export_closure.py, imported, unchanged) over the two entry points the DAG-ORC stages run on box 2
(stack/scripts/v8f_mini_train.py for gatedag / traindag, stack/scripts/v8f_mini_eval.py for evaldag; no bank generator, no bank tool, no w_acc stage), computed on the python files of the tip 3b7b404e, plus the repo DATA paths
those programs open (dag2's list, unchanged).

    python export_closure_dagorc.py TREE_ROOT     (TREE_ROOT = an extraction of the tip containing stack/ and taniteval/)
writes export_closure_dagorc.json beside this file: {"files": [...], "data": [...], ...}
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ec", "E:/Projects/TanitAD-artifacts/dag2q/export_closure.py")
EC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EC)


def main():
    root = Path(sys.argv[1])
    EC.ENTRIES = ["stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_mini_eval.py"]
    files, external, relative = EC.closure(root)
    data = [p for p, _, _ in EC.DATA_READS]
    size = sum((root / f).stat().st_size for f in files)
    (HERE / "export_closure_dagorc.json").write_text(json.dumps({"entries": EC.ENTRIES, "n_files": len(files), "bytes": size, "files": files, "data": data, "external_top_level_modules": external}, indent=1), encoding="utf-8")
    print(f"dagorc: {len(files)} python files {size / 1e6:.2f} MB + {len(data)} data paths; external {external}")


if __name__ == "__main__":
    main()
