#!/usr/bin/env python3
"""Call the package's ``run_bridge8.main(argv)`` UNCHANGED, then record what Thor admits as a memory number
(in-process ``torch.cuda.max_memory_allocated``; CLAUDE.md: free/tegrastats/mem_get_info lie on unified memory) and the
wall time, into ``<out>/bridge_wrap_record.json``.

    python r8ns_bridge_wrap.py <pkg_code_dir> -- <run_bridge8 argv ...>
"""
import json
import os
import sys
import time

code = sys.argv[1]
argv = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, code)
t0 = time.time()
import run_bridge8 as RB8  # noqa: E402

t_import = time.time() - t0
rc = RB8.main(argv)
import torch  # noqa: E402

out = argv[argv.index("--out") + 1]
rec = {"rc": rc, "argv": argv, "import_s": round(t_import, 2), "wall_s": round(time.time() - t0, 2),
       "cuda_max_memory_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                        if torch.cuda.is_available() else None),
       "torch": torch.__version__, "host": os.uname().nodename if hasattr(os, "uname") else None}
with open(os.path.join(out, "bridge_wrap_record.json"), "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1)
print("ZZWRAP " + json.dumps({k: rec[k] for k in ("rc", "wall_s", "cuda_max_memory_allocated_gb")}), flush=True)
sys.exit(rc)
