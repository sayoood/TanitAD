"""The chain-active marker (Master Mind + Research Lab, 2026-09-28; a scheduling rule, not a SPEC change):
while a refcv7 milestone chain runs, `C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active` exists
(JSON: chain, pid, started). Long foreign jobs must not acquire the GPU lock while it exists; REFe snapshot
seams (<= 20 min) may. Created when the chain first holds the lock; removed on EVERY exit path, and ONLY if
its content is still this chain's.

    python chain_marker.py create --chain refcv7-milestone-step5000 --pid <pid>
    python chain_marker.py remove --chain refcv7-milestone-step5000 --pid <pid>
"""
import argparse
import json
import os
import time

MARK = os.environ.get("REFCV7_CHAIN_MARKER", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("create", "remove", "show"))
    ap.add_argument("--chain", default=None)
    ap.add_argument("--pid", default=None)
    a = ap.parse_args()
    cur = None
    if os.path.exists(MARK):
        try:
            cur = json.load(open(MARK, encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            cur = {"unreadable": True}
    if a.cmd == "show":
        print(json.dumps({"marker": cur}))
        return
    if a.cmd == "create":
        if cur is not None and not (isinstance(cur, dict) and cur.get("chain") == a.chain
                                    and str(cur.get("pid")) == str(a.pid)):
            print(json.dumps({"created": False, "why": "a marker of another chain exists", "marker": cur}))
            return
        tmp = MARK + ".tmp"
        json.dump({"chain": a.chain, "pid": a.pid, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                  open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, MARK)
        print(json.dumps({"created": True}))
        return
    if cur is None:
        print(json.dumps({"removed": False, "why": "no marker"}))
    elif isinstance(cur, dict) and cur.get("chain") == a.chain and str(cur.get("pid")) == str(a.pid):
        os.remove(MARK)
        print(json.dumps({"removed": True}))
    else:
        print(json.dumps({"removed": False, "why": "the marker is not this chain's", "marker": cur}))


if __name__ == "__main__":
    main()
