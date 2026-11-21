#!/usr/bin/env python3
"""Block until psutil available RAM >= --min-gb (a SCRIPT poll, never a model turn).  exit 0 when met; exit 75 when --max-wait elapsed.  Used before the RAM-fragile labelling step (L1 STATE: step 5)."""
import argparse
import sys
import time

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--min-gb", type=float, default=8.0)
ap.add_argument("--max-wait", type=float, default=7200.0)
a = ap.parse_args()
t0 = time.time()
while True:
    av = psutil.virtual_memory().available / 2**30
    if av >= a.min_gb:
        print(f"ZZRAM_OK avail={av:.1f} GiB", flush=True)
        sys.exit(0)
    if time.time() - t0 > a.max_wait:
        print(f"ZZRAM_TIMEOUT avail={av:.1f} GiB", flush=True)
        sys.exit(75)
    time.sleep(20)
