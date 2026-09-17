"""PER-HEAD GRADIENT REACH, from the live run's own metrics.jsonl.

⛔ THE NUMBER THIS PACKAGE EXISTS FOR. At tip ``837c308`` every parameter of
the behaviour decoder sat at ``grad_abs_sum`` EXACTLY 0, because nothing read
its outputs. A head that still reads 0 after this wiring is a FAILURE to report
loudly, not a detail.

⛔⛔ AND A ZERO IS NOT READ AS A FAILURE WITHOUT ITS ``n_supervised``. The
22-token goal target and the 8+8 action targets are supervised ONLY inside the
record's ±2 s band (``v7_labels.window_in_band``; the goal targets return
all-``IGNORE_W`` outside it, ``v7_labels.py:1019``). So an out-of-band step
legitimately produces loss 0.0 and gradient 0.0 — and that is INDISTINGUISHABLE
from a dead head unless ``n_supervised`` is printed beside it. This script
therefore splits the rows and reports BOTH populations.

⭐ ``n_grad_none`` is the third column and the one that separates the two
failure modes outright: ``p.grad is None`` means NEVER WIRED, while a zeros
gradient means wired and unsupervised. At tip the head had no gradient path at
all; here it must read 0 Nones on every step.

Usage:  python grad_reach.py <run-dir> [<run-dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HEADS = ("tac_decoder_v6", "tac_decoder_v6.validity_head",
         "tac_decoder_v6.conf_head", "tac_decoder_v6.lat_head",
         "tac_decoder_v6.lon_head", "tac_decoder_v6.queries",
         "tac_decoder_v6.layers", "tac_decoder_v6.agent_in")

out = {"runs": []}
for d in sys.argv[1:]:
    p = Path(d) / "metrics.jsonl"
    if not p.is_file():
        out["runs"].append({"run": d, "verdict": "INCONCLUSIVE",
                            "why": "no metrics.jsonl"})
        continue
    rows = [json.loads(x) for x in
            p.read_text(encoding="utf-8").strip().splitlines() if x.strip()]
    if not rows:
        out["runs"].append({"run": d, "verdict": "INCONCLUSIVE",
                            "why": "metrics.jsonl is empty"})
        continue
    inb = [r for r in rows
           if float(r.get("tacv6_n_supervised_goal_cells", 0) or 0) > 0]
    oob = [r for r in rows
           if float(r.get("tacv6_n_supervised_goal_cells", 0) or 0) == 0]
    heads = {}
    for h in HEADS:
        k = f"gp_{h}_grad_abs_sum"
        if k not in rows[-1]:
            continue
        v_in = [float(r[k]) for r in inb if k in r]
        v_oob = [float(r[k]) for r in oob if k in r]
        nn = [float(r.get(f"gp_{h}_n_grad_none", -1)) for r in rows]
        heads[h] = {
            "n_params": rows[-1].get(f"gp_{h}_n_params"),
            "n_tensors": rows[-1].get(f"gp_{h}_n_tensors"),
            # ⛔ `p.grad is None` on ANY step is the "never wired" signature.
            "n_grad_none_max": max(nn) if nn else None,
            "in_band_steps": len(v_in),
            "grad_abs_sum_in_band_max": max(v_in) if v_in else None,
            "grad_abs_sum_in_band_min": min(v_in) if v_in else None,
            "out_of_band_steps": len(v_oob),
            "grad_abs_sum_out_of_band_max": max(v_oob) if v_oob else None,
            "verdict": (
                "INCONCLUSIVE — no in-band step was sampled" if not v_in
                else ("REACHES" if min(v_in) > 0.0
                      else "⛔ STILL ZERO ON AN IN-BAND STEP")),
        }
    out["runs"].append({
        "run": d,
        "n_steps": len(rows),
        "n_in_band_steps": len(inb),
        "n_out_of_band_steps": len(oob),
        "in_band_frac": round(len(inb) / max(len(rows), 1), 4),
        "losses_on_a_representative_in_band_step": (
            {k: inb[0][k] for k in sorted(inb[0]) if k.startswith("tacv6")}
            if inb else None),
        "heads": heads,
        "⚠️": ("an out-of-band step legitimately reads loss 0.0 / grad 0.0; "
               "`n_supervised` is the only thing that distinguishes it from a "
               "dead head, and `n_grad_none` separates 'unsupervised' from "
               "'never wired'"),
    })

print(json.dumps(out, indent=1, ensure_ascii=False))
