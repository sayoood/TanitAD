#!/usr/bin/env python3
"""train_final.txt by the card's replacement rule (section 2): walk train_order.txt (positions 1..520); a clip that is not staged
is replaced by the next of the order; the first 500 staged clips are the list, the first 450 `fit`, the last 50 `sel`.
Reads the stage ledger (last line per clip must be ok). Writes <out>/train_final.txt (`<clip> <role>`) and
<out>/TRAIN_FINAL_RECORD.json (skipped clips with reasons and positions, shortfall if any). Pure Python; no pod access needed."""
import argparse
import json
import os
import sys

N_FINAL, N_FIT, MAX_POS = 500, 450, 520


def ledger_last(path):
    d = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    d[r["clip"]] = r
                except Exception:                                    # noqa: BLE001
                    pass
    return d


def pick(order, ledger, n_final=N_FINAL, n_fit=N_FIT, max_pos=MAX_POS):
    """order: [(position, clip)]; ledger: {clip: row with ok}. -> (final [(clip, role)], skipped [dict])"""
    final, skipped = [], []
    for pos, c in order:
        if pos > max_pos or len(final) >= n_final:
            break
        r = ledger.get(c)
        if r is not None and r.get("ok"):
            final.append(c)
        else:
            skipped.append({"position": pos, "clip": c, "reason": (r or {}).get("reason", "not staged (no ledger line)")})
    roles = [(c, "fit" if i < n_fit else "sel") for i, c in enumerate(final)]
    return roles, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--order", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    order = []
    for ln in open(a.order, encoding="utf-8"):
        t = ln.split()
        if len(t) == 2:
            order.append((int(t[0]), t[1]))
    roles, skipped = pick(order, ledger_last(a.ledger))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_final.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for c, r in roles:
            fh.write(f"{c} {r}\n")
    rec = {"n_final": len(roles), "n_fit": sum(1 for _, r in roles if r == "fit"), "n_sel": sum(1 for _, r in roles if r == "sel"),
           "shortfall": max(0, N_FINAL - len(roles)), "max_position_used": max([p for p, c in order if c in {x for x, _ in roles}] or [0]),
           "skipped": skipped}
    with open(os.path.join(a.out, "TRAIN_FINAL_RECORD.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "skipped"}), f"skipped={len(skipped)}")
    return 0 if len(roles) >= 490 else 3


if __name__ == "__main__":
    sys.exit(main())
