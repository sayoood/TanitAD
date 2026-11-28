#!/usr/bin/env python3
"""hqs_bench.py -- the DECODER BENCH (non-binding, informative): the one-frame test on the box head ALONE, on the box
memory's inputs captured from the launch config (``capture_oneframe.py``), trunk and BEV frozen by construction.

The ladder showed the one-frame failure with a frozen trunk too, so the question "does HQS stabilise the assignment?"
is a box-head question. Modes: ``learned`` (MAIN's head, the baseline -- must reproduce the ladder's failure),
``heatmap`` (HQS), ``anchors_removed`` (A14's red arm: learned reference points). The loss is the landed
``box3d_loss_row`` (with the heat term under HQS); the optimiser is the launch's HEAD group (AdamW lr 1e-4, weight
decay 1e-4) with clip 10 (A13). Metrics every 25 steps, A14's literals at the end.

usage: python hqs_bench.py <capture.pt> <learned|heatmap|anchors_removed> [steps 500] [every 25] [device]
"""
import json
import sys
import time

import numpy as np
import torch

from tanitad.data.bev_raster import GRID_DEFAULT
from tanitad.models import agent_slots as AS
from tanitad.models import box3d_head as B3
from tanitad.models import refcv6_perception_branch as PB
from tanitad.models import slot_query_select as SQS


def main():
    path, mode = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    dev = sys.argv[5] if len(sys.argv) > 5 else ("cuda" if torch.cuda.is_available() else "cpu")
    cap = torch.load(path, map_location="cpu", weights_only=False)
    box_mem, box_dec = cap["box_mem"].to(dev), cap["box_dec"].to(dev)
    fmap, bev = (t.to(dev) for t in cap["mem_in"][:2])
    tgt = {k: v.to(dev) for k, v in cap["tgt"].items()}
    kw = {k: ({a: (b.to(dev) if torch.is_tensor(b) else b) for a, b in v.items()} if isinstance(v, dict)
              else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in cap["kw"].items()}
    params = list(box_mem.parameters()) + list(box_dec.parameters())
    heat = qpos = ref = None
    if mode in ("heatmap", "anchors_removed"):
        torch.manual_seed(0)
        heat = SQS.BEVHeatHead(int(bev.shape[1])).to(dev)
        qpos = SQS.AnchorPosEmbed(int(box_dec.d_model), GRID_DEFAULT).to(dev)
        params += list(heat.parameters()) + list(qpos.parameters())
        if mode == "anchors_removed":
            g = torch.Generator().manual_seed(0)
            k = int(box_dec.n_queries)
            ref = torch.nn.Parameter(torch.stack([torch.rand(k, generator=g) * GRID_DEFAULT.x_fwd_m,
                                                  (torch.rand(k, generator=g) * 2 - 1) * GRID_DEFAULT.y_half_m],
                                                 -1).to(dev))
            params.append(ref)
    elif mode != "learned":
        raise SystemExit(f"mode {mode!r}")
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    calls = []
    o_ms, o_b3 = AS.match_slots, B3.match_slots

    def ms(pred, t, *, presence_cost="sigmoid"):
        m = o_ms(pred, t, presence_cost=presence_cost)
        calls.append(m)
        return m
    AS.match_slots, B3.match_slots = ms, ms
    rows, prev, prev_anc, t0 = [], None, None, time.time()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            calls.clear()
            mem = box_mem(fmap, bev)
            if heat is None:
                slots = box_dec(mem)
            else:
                h = heat(bev)
                if ref is None:
                    anc, _ = SQS.select_anchors(h, int(box_dec.n_queries), SQS.heat_grid_of(GRID_DEFAULT, h))[:2]
                else:
                    anc = ref[None].expand(int(h.shape[0]), -1, -1)
                slots = SQS.anchored_forward(box_dec, mem, anc, qpos(anc))
                slots["heat_logits"] = h
                slots["heat_grid"] = SQS.heat_grid_of(GRID_DEFAULT, h)
            row = PB.box3d_loss_row(slots, tgt, **kw)
            row["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            if step % every:
                continue
            m = calls[-1]
            cur = [{int(c): int(r) for r, c in zip(R.tolist(), C.tolist())} for R, C in zip(m["rows"], m["cols"])]
            keep = None
            if prev is not None:
                tot = sum(len(b) for b in cur)
                keep = sum(int(prev[i].get(k) == v) for i, b in enumerate(cur) for k, v in b.items()) / max(tot, 1)
            anc_keep = None
            if "anchors" in slots:
                a0 = slots["anchors"].detach().cpu()
                cur_a = [{k: (round(float(a0[b, v, 0]), 2), round(float(a0[b, v, 1]), 2)) for k, v in cur[b].items()}
                         for b in range(len(cur))]
                if prev_anc is not None:
                    tot = sum(len(b) for b in cur_a)
                    anc_keep = sum(int(prev_anc[i].get(k) == v) for i, b in enumerate(cur_a)
                                   for k, v in b.items()) / max(tot, 1)
                prev_anc = cur_a
            prev = cur
            p = torch.sigmoid(slots["presence_logit"].detach().float())[0].cpu()
            idx = torch.tensor(sorted(cur[0].values()), dtype=torch.long)
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask[idx] = True
            r = {"step": step, "loss": float(row["loss"].detach()),
                 "heat": float(row["box3d_heat"].detach()) if "box3d_heat" in row else None,
                 "n_matched": int(mask.sum()), "p_matched_median": float(p[mask].median()),
                 "p_matched_max": float(p[mask].max()), "p_unmatched_max": float(p[~mask].max()),
                 "n_conf": int((p >= 0.5).sum()), "keep": keep, "anchor_keep": anc_keep}
            rows.append(r)
            print(json.dumps(r), flush=True)
    finally:
        AS.match_slots, B3.match_slots = o_ms, o_b3
    fin = [r["keep"] for r in rows if r["keep"] is not None and r["step"] > steps - 100]
    verdict = {"i_p_matched_median": rows[-1]["p_matched_median"], "i_pass": rows[-1]["p_matched_median"] >= 0.5,
               "ii_keep_final100": float(np.mean(fin)) if fin else None,
               "ii_pass": bool(fin) and float(np.mean(fin)) >= 0.8}
    verdict["PASS"] = bool(verdict["i_pass"] and verdict["ii_pass"])
    print("VERDICT", mode, json.dumps(verdict), f"wall {time.time() - t0:.1f}s on {dev}", flush=True)


if __name__ == "__main__":
    main()
