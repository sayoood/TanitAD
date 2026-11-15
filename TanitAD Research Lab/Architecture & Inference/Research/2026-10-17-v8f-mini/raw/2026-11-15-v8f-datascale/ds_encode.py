#!/usr/bin/env python3
"""DATASCALE pod-side ENCODE: runs /root/identity_pod/cache_pod.py (the corpus encoder of record, UNEDITED) against a datascale ledger / v2ep dir / out dir
by overriding its module constants (no copy of its code). Needs env_lora (PYTHONPATH, offline HF cache, V8F_LORA_BUILDER).
  python ds_encode.py --set pilot|ctl
No address, port or pod id in this file."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

B = "/root/datascale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pilot", "ctl"], required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cache_pod", "/root/identity_pod/cache_pod.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    led = Path(f"{B}/ledger_{a.set}.jsonl")
    cp.LEDGER = led
    cp.BUILD_LOG = Path(f"{B}/build_{a.set}.log")
    cp.V2EP = Path(f"{B}/v2ep_{a.set}/train").parent
    cp.BOX2_PULLED = Path(f"{B}/ctl_ref") if a.set == "ctl" else Path(f"{B}/NO_SUCH_REF")
    cp.N_EXPECT = sum(1 for ln in open(led, encoding="utf-8") if ln.strip())
    sys.argv = ["cache_pod.py", "--out", f"{B}/cache_{a.set}", "--poll", "5", "--threads", "2"]
    return cp.main()


if __name__ == "__main__":
    sys.exit(main())
