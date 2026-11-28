"""Emit one compact line per NEW per-file result in raw/gate/*.jsonl, and a final line when the
chain's driver log carries its end marker. For a Monitor: stdout lines are the events."""
import json
import pathlib
import time

G = pathlib.Path(__file__).resolve().parents[1] / "raw" / "gate"
seen: dict = {}
END = "ZZGATESUITES" + "DONEZZ"          # built, so this file never contains the literal token
while True:
    for f in sorted(G.glob("*.jsonl")):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in lines[seen.get(f.name, 0):]:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("file") == "__header__":
                continue
            c = r.get("counts", {})
            bad = r.get("failed", []) + r.get("errors", [])
            print(f"{f.stem} {r.get('status')} {r.get('file')} "
                  f"p{c.get('passed', 0)} f{c.get('failed', 0)} e{c.get('error', 0)} "
                  f"s{c.get('skipped', 0)}" + (f" BAD {bad[:3]}" if bad else ""), flush=True)
        seen[f.name] = len(lines)
    drv = G / "gate_chain_driver.log"
    if drv.exists() and END in drv.read_text(encoding="utf-8", errors="replace"):
        print("GATE CHAIN FINISHED", flush=True)
        break
    time.sleep(30)
