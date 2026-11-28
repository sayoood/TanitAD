#!/usr/bin/env python3
"""DATASCALE-10K per-window encode wrapper: /root/identity_pod/cache_pod.py UNEDITED (same override technique as ds_encode.py). usage: ds_encode_w.py <windir>
Needs env_lora. No address, port or pod id in this file."""
import importlib.util
import sys
from pathlib import Path

W = sys.argv[1]
spec = importlib.util.spec_from_file_location("cache_pod", "/root/identity_pod/cache_pod.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
led = Path(f"{W}/ledger_build.jsonl")
led.touch()
cp.LEDGER = led
cp.BUILD_LOG = Path(f"{W}/build.log")
cp.V2EP = Path(f"{W}/v2ep")
cp.BOX2_PULLED = Path(f"{W}/NO_SUCH_REF")
cp.N_EXPECT = sum(1 for ln in open(led, encoding="utf-8") if ln.strip())
sys.argv = ["cache_pod.py", "--out", f"{W}/cache", "--poll", "5", "--threads", "2"]
sys.exit(cp.main())
