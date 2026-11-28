"""mem_amend.py -- AMENDMENT 1 (memory guard only) of PREREAD_V8F_TOPK_PAIR control 5 and PREREAD_V8F_CONSORC control 6 (AMENDMENT_1_MEMORY.md).

Runs a frozen read script UNCHANGED (its sha256 is printed; it must still equal the card's .sha256 line) with one change: the module's
wait_free(gb, max_s) thresholds. A call with gb >= 10 (per arm) uses ARM_GB; any other call (per chunk, smoke) uses min(gb, CHUNK_GB).
Usage: python -I mem_amend.py <frozen_script.py> [its args ...]
"""
import hashlib
import importlib.util
import os
import sys

ARM_GB, CHUNK_GB = 4.0, 3.0

script = os.path.abspath(sys.argv[1])
with open(script, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(f"[mem_amend] {os.path.basename(script)} sha256 {digest}; guards: arm {ARM_GB} GB, chunk {CHUNK_GB} GB", flush=True)
name = os.path.splitext(os.path.basename(script))[0]
spec = importlib.util.spec_from_file_location(name, script)
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
sys.argv = [script] + sys.argv[2:]
spec.loader.exec_module(mod)
_orig = mod.wait_free


def wait_free(gb, max_s):
    return _orig(ARM_GB if gb >= 10.0 else min(gb, CHUNK_GB), max_s)


mod.wait_free = wait_free
mod.main()
