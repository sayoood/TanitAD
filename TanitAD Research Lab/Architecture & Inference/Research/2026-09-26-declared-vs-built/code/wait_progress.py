"""Block until the gate chain has completed K more per-file results, or finished, or T seconds
passed (bounded foreground wait for the agent; prints one summary line on return).

    python wait_progress.py <K> <T>
"""
import json
import pathlib
import sys
import time

G = pathlib.Path(__file__).resolve().parents[1] / "raw" / "gate"
END = "ZZGATESUITES" + "DONEZZ"


def n_done() -> tuple[int, list]:
    n, last = 0, []
    for f in sorted(G.glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("file") != "__header__":
                n += 1
                last.append(f"{f.stem}:{r.get('file')}:{r.get('status')}:{r.get('counts')}")
    return n, last[-3:]


def finished() -> bool:
    d = G / "gate_chain_driver.log"
    return d.exists() and END in d.read_text(encoding="utf-8", errors="replace")


k, t = int(sys.argv[1]), float(sys.argv[2])
n0, _ = n_done()
t0 = time.time()
while time.time() - t0 < t:
    n, last = n_done()
    if n >= n0 + k or finished():
        break
    time.sleep(20)
n, last = n_done()
print(f"done={n} (+{n - n0}) finished={finished()} waited={time.time() - t0:.0f}s last={last}")
