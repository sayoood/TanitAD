#!/usr/bin/env python3
"""gbo_table.py -- print a G-BOX-OVERFIT record as the per-criterion table + the trajectory (markdown).

usage: python gbo_table.py <record.json> [--arm main] [--paused-s S]
Reads only the record; computes nothing the harness did not write except the paused-corrected s/step (when a
pause interval is given) and the presence ratio already in criterion 6.
"""
from __future__ import annotations

import argparse
import json
import math


def f(v, nd=3):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(f(x, nd) for x in v) + "]"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--arm", default="main")
    ap.add_argument("--paused-s", type=float, default=0.0)
    a = ap.parse_args()
    rec = json.load(open(a.record, encoding="utf-8"))
    print(f"record: binding={rec.get('binding')} | commit={rec.get('commit')!r}")
    print(f"prereg_md5_ok={rec.get('prereg_md5_ok')} frameset_md5={rec.get('frameset_md5')} "
          f"trunk_compile={rec.get('trunk_compile', 'not stamped')} RESULT={rec.get('RESULT', 'not written')}")
    c3 = rec.get("C3") or {}
    print(f"C3: pass={c3.get('pass')} totals={c3.get('totals')} want={c3.get('want_totals')}")
    for k, v in (rec.get("C1_C2") or {}).items():
        print(f"{k}: {json.dumps(v)}")
    arm = (rec.get("arms") or {}).get(a.arm)
    if arm is None:
        print(f"arm {a.arm!r} not in the record (arms: {list((rec.get('arms') or {}))})")
        return 1
    crit = arm["criteria"]
    print(f"\n### arm {a.arm}: verdict {arm['verdict']}  (wall {arm.get('wall_s')} s, "
          f"{arm.get('n_trainable_params')} trainable params in {arm.get('n_trainable_tensors')} tensors)\n")
    print("| # | criterion | value at the last step | pass |")
    print("|---|---|---|---|")
    for k in sorted(x for x in crit if x != "PASS"):
        print(f"| {k} | {crit[k]['what']} | {f(crit[k]['value'])} | {'PASS' if crit[k]['pass'] else 'FAIL'} |")
    print(f"| all | | | {'PASS' if crit['PASS'] else 'FAIL'} |")
    rows = arm["rows"]
    keys = [("step", 0), ("ap2m", 3), ("prec", 3), ("rec", 3), ("n_conf", 0), ("auroc_matched", 3),
            ("pr_equal_gate_informative", 3), ("pr_equal_prec_informative", 3), ("pr_equal_rec_informative", 3),
            ("centre_p50_m", 3), ("size_p50_m", 3), ("z_p50_m", 3), ("cls_acc", 3), ("presence", 4),
            ("loss_mean_since_last", 4), ("s_per_step", 2), ("peak_mem_gb", 2)]
    keys = [(k, nd) for k, nd in keys if any(k in r for r in rows)]
    print("\n| " + " | ".join(k for k, _ in keys) + " | gpu_shared_with |")
    print("|" + "---|" * (len(keys) + 1))
    for r in rows:
        shared = r.get("gpu_shared_with")
        sh = "-" if shared is None else (str(len(shared)) + " job(s)" if shared else "none")
        print("| " + " | ".join(f(r.get(k), nd) for k, nd in keys) + f" | {sh} |")
    last = rows[-1]
    sps = last.get("s_per_step")
    if sps is not None and a.paused_s:
        n = int(last["step"])
        print(f"\ns/step (wall, incl. pause) {sps:.2f}; paused {a.paused_s:.0f} s -> "
              f"{(sps * n - a.paused_s) / n:.2f} s/step excluding the pause")
    shared = sorted({s for r in rows for s in (r.get("gpu_shared_with") or [])})
    print(f"\ngpu_shared_with (union over rows): {shared if shared else 'none'}")
    print(f"peak_mem_gb (torch.cuda.max_memory_allocated): {f(arm.get('peak_mem_gb'), 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
