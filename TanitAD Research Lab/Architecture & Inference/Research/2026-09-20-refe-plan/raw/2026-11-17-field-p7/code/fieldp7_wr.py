#!/usr/bin/env python3
"""FIELDP7: the committed gradient-norm balance (PREREG_AUX_FIELD_LOSS_DRAFT.md "Weight rule") on P7's model_final, CPU, real train rows.

It reuses the committed tool `refe/weight_rule.py` UNCHANGED (build_instrumented = the real train.run on the arm's argv up to the optimiser build; measure = the rule's 8 seeded
micro-batches, pooled gradients, r(w)).  The only addition is memory: every parameter outside `traj_head` is switched to requires_grad=False BEFORE measure().  The rule takes
d(loss)/d(traj_head) only, and that gradient does not depend on whether upstream parameters track gradients, so the numbers are the same; the autograd graph through the 24-block
ViT-L (about 15 GB on this box at batch 1) is simply never recorded.  (The reach script fieldp7_reach.py runs the full-graph path, with activation checkpointing, on the real trainer.)

    python fieldp7_wr.py --refe <dir of refe/> --args "<train.py argv>" --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--args", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refe = Path(a.refe).resolve()
    sys.path.insert(0, str(refe))
    import weight_rule as W
    targv = a.args.split()
    T, ns, model, ds, dev = W.build_instrumented(refe, targv, None)
    head_ids = {id(p) for p in model.traj_head.parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(False)
    res = W.measure(T, ns, model, ds, dev)
    rule = W.decide({w: res["r_by_w"][repr(w)] for w in W.CANDIDATES})
    res.update({"rule": rule, "argv_sha256": hashlib.sha256("\x00".join(targv).encode()).hexdigest(), "init_from": ns.init_from, "field_margin": ns.field_margin,
                "field_interp": ns.field_interp, "label_store": ns.field_labels, "device": str(dev), "others_frozen_for_memory": True,
                "n_bank_rows": len(ds)})
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"FIELDP7_WR pooled ||grad l_traj|| = {res['pooled']['norm_grad_traj_loss']:.6g}  ||grad L_field|| (w=1) = {res['pooled']['norm_grad_field_loss_w1']:.6g}  r(1) = {res['pooled']['r1']:.6g}")
    print("  r(w): " + "  ".join(f"{w:g}:{res['r_by_w'][repr(w)]:.4g}" for w in W.CANDIDATES))
    print(f"  coverage {res['label_coverage_of_the_8_batches']:.3f} in-extent {res['field_inext_mean']:.3f} violating {res['field_viol_mean']:.3f}")
    print(f"  RULE {rule['status']} w_field={rule['w_field']} -- {rule['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
