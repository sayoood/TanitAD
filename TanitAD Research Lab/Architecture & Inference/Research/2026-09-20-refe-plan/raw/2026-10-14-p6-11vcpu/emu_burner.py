#!/usr/bin/env python3
"""A duty-cycle CPU burner (trainer / DataLoader stand-in for the dev-box emulation). python emu_burner.py <duty 0..1> <cpus e.g. 0-7,12-14>"""
import sys
import time
import psutil

duty, spec = float(sys.argv[1]), sys.argv[2]
cpus = []
for part in spec.split(","):
    a, _, b = part.partition("-")
    cpus += list(range(int(a), int(b or a) + 1))
psutil.Process().cpu_affinity(cpus)
slot = 0.05
while True:
    t = time.perf_counter()
    while time.perf_counter() - t < slot * duty:
        pass
    if duty < 1.0:
        time.sleep(slot * (1.0 - duty))
