"""Events for the final gate chain: the tau file appearing (or its log ending), the dev-box quick
run finishing, each new per-file result, and the chain's end marker. One stdout line per event."""
import json
import pathlib
import time

PK = pathlib.Path(__file__).resolve().parents[1]
G = PK / "raw" / "gate"
END = "ZZGATESUITES" + "DONEZZ"
seen: dict = {}
flags = {"tau": False, "tau_log": False, "quick": False}
while True:
    tau = PK / "raw" / "nav_compliance_tau_train.json"
    if not flags["tau"] and tau.exists():
        try:
            r = json.loads(tau.read_text(encoding="utf-8"))
            print(f"TAU tau={r.get('tau')} status={r.get('status')} n_pos={r.get('n_pos')} "
                  f"n_neg={r.get('n_neg')} J={r.get('youden_j')}", flush=True)
            flags["tau"] = True
        except ValueError:
            pass
    tl = G / "nav_compliance_tau_train.log"
    if not flags["tau_log"] and tl.exists() and "# exit" in tl.read_text(encoding="utf-8", errors="replace"):
        last = tl.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
        print(f"TAU-LOG {last}", flush=True)
        flags["tau_log"] = True
    ql = PK / "raw" / "dev_quick_new_tests.log"
    if not flags["quick"] and ql.exists() and "# exit" in ql.read_text(encoding="utf-8", errors="replace"):
        tail = [x for x in ql.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        print(f"QUICK {' | '.join(tail[-3:])[:400]}", flush=True)
        flags["quick"] = True
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
            if bad or r.get("status") != "OK" or f.stem.startswith(("FIXNEW", "TIPRED")):
                print(f"{f.stem} {r.get('status')} {r.get('file')} p{c.get('passed', 0)} "
                      f"f{c.get('failed', 0)} e{c.get('error', 0)} s{c.get('skipped', 0)}"
                      + (f" BAD {bad[:4]}" if bad else ""), flush=True)
        seen[f.name] = len(lines)
    drv = G / "gate_chain_driver.log"
    if drv.exists() and END in drv.read_text(encoding="utf-8", errors="replace"):
        print("GATE CHAIN FINISHED", flush=True)
        break
    time.sleep(30)
