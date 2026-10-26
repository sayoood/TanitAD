#!/usr/bin/env python3
"""HiCAP latency-budget model: what backbone size does a latency criterion admit? (pure arithmetic)

Replaces the retired 'sub-300 M parameter' cap (PI ruling 2026-09-29: "if the inference time of the whole
system is very low, the system can have far more than 300 M parameters"). Evidence class of EVERY output
here: ESTIMATED -- closed-form arithmetic anchored on NVIDIA-PUBLISHED Jetson Thor rows (TensorRT-Edge-LLM
0.10.0, batch 1, ViT FP16, LLM NVFP4) as transcribed in RA_vlm_backbones.md section 2. NOTHING here is a
HiCAP timing and nothing here is a measurement of ours; H-HC11 / hicap_backbone_latency_bench.py measures.

Model (all conventions inherited from sec_eff.tex Table 'thor' so the four measured rows are reproduced):
  T_r(P, n, images) = images * ViT_ms_per_image + T_prefill(P, n)
  ViT_ms_per_image  = published ViT ms on a ~265-token (1,064-patch) COCO image, NOT scaled down to our
                      640-patch crop -> conservative upper bound (same as Table 'thor').
  T_prefill         = linear-in-n interpolation of the two published rows (n=292 and n=2,048), extrapolated
                      linearly below 292 (flagged). For a model WITHOUT rows a BRACKET is given:
        optimistic  = max( 2*P_body*n / 272 TFLOPS , weight_bytes / (0.7*273 GB/s) )
        pessimistic = max( t_8B(n) * P_body/P_8B , weight floor )    # the 8B row's effective TFLOPS
      (only for P_body > 6.95 B; for P_body <= 6.95 B the time is interpolated between the measured 2B/4B/8B
      rows at the same n, because a smaller model runs at LOWER effective throughput, so scaling the 8B row
      down would be optimistic, not pessimistic)
      valid only if effective throughput is non-decreasing in model size within the Qwen3-VL dense family
      (3 measured points: 65/95/126 TFLOPS at n=292 and 206/239/272 at n=2048; NOT true across families --
      the Qwen3.5 hybrid is ~1.5x slower per token). Beyond 2x the largest measured body (13.9 B) the bracket
      is all that is offered: no fitted exponent, no extrapolated point estimate.
  Weight floor: a prefill must stream every weight once; NVFP4 = 4.5 bit/param (4-bit + one FP8 scale / 16).
  Budget: refresh duty  D = f_r * (T_r + overhead)  <=  0.5 (the rest is for the 10 Hz fast path and other
  tenants -- an ASSUMPTION);  overhead = 15 ms (pooling, engine hand-off, pre-processing; ASSUMPTION, RA sec 5.3
  says the published rows exclude 10-20 ms).  Token age (what the fusion transformer's age embedding sees)
  worst case  A = T_r + overhead + 1/f_r.
Run:  python3 hicap_latency_budget.py   ->  hicap_latency_budget_result.json + tables on stdout.
"""
import json
import os
import sys

THOR_BW = 273e9            # B/s, PUBLISHED spec
BW_EFF = 0.70              # achievable share of peak for a streaming read (ASSUMPTION; consistent with the
                           # measured 8B / 4B / 2B floors below: weight_floor < measured prefill in every row)
TFLOPS_MAX = 272e12        # highest effective LLM prefill rate implied by a published Thor row (8B @ 2,048)
NVFP4_B = 4.5 / 8          # bytes / param
OVERHEAD_MS = 15.0         # ASSUMPTION
DUTY_MAX = 0.5             # ASSUMPTION
TOK_PER_CAM = 160          # 256x640 crop, patch 16, 2x2 merge (arithmetic; Qwen3-VL-lineage processors)
TOK_PROMPT = 64            # ego / query / prompt tokens (HYPOTHESIS)
THOR_MEM_GB = 128.0
VLM_MEM_BUDGET_GB = 64.0   # ASSUMPTION: half of unified memory for the VLM; PI to confirm
TICKS_2HZ = 188640         # 26.2 h at 2 Hz (sec_eff)
POOL_TOK = 128

# measured rows: (ViT ms per ~265-token image, prefill ms @292, prefill ms @2048)  -- RA_vlm_backbones.md sec 2
MEASURED = {
    "Qwen3-VL-2B":  dict(vit=11.4, p292=12.7, p2048=28.0,  body=1.41, total=2.1),
    "Qwen3-VL-4B":  dict(vit=11.6, p292=22.3, p2048=62.2,  body=3.63, total=4.4),
    "Qwen3.5-4B":   dict(vit=10.9, p292=31.6, p2048=95.2,  body=3.5,  total=4.5),   # hybrid; body ESTIMATED
    "Qwen3-VL-8B":  dict(vit=15.7, p292=32.1, p2048=104.5, body=6.95, total=8.3),
}
# rows used only as bracket inputs (no Thor row): name -> (kind, body_active_B, total_B, vit_ms, hidden, note)
UNMEASURED = {
    "Cosmos3-Edge reasoner (2.4B)":            ("dense", 1.41, 2.4, 15.7, 2048, "Nemotron-dense 2B + SigLIP2-SO400M; eager-BF16 row exists (P), no TRT row"),
    "Cosmos-Reason2-8B (= Qwen3-VL-8B arch)":  ("dense", 6.95, 8.3, 15.7, 4096, "architecture equivalence from the HF card (INHERITED); timing = 8B row"),
    "Alpamayo-1.5 VLM (8.2B)":                 ("dense", 6.95, 8.3, 15.7, 4096, "Cosmos-Reason2 backbone (P: recipes), taken as the 8B row (INHERITED equivalence); 2.3B expert not counted"),
    "Cosmos-Reason2-32B (= Qwen3-VL-32B arch)": ("dense", 31.2, 33.0, 15.7, 5120, "body/hidden from config arithmetic, UNVERIFIED; ViT assumed SO400M"),
    "Qwen3-Omni-30B-A3B (MoE, 3.3B active)":   ("moe",   3.3,  30.5, 15.7, 2048, "Thinker MoE; text Qwen3-30B-A3B NVFP4 pf@2048=137.7 ms (P); ViT/audio not in that row"),
}
MOE_PF2048 = 137.7         # ms, PUBLISHED (text-only Qwen3-30B-A3B NVFP4)


def lin(n, a, ta, b, tb):
    return ta + (n - a) * (tb - ta) / (b - a)


def prefill_measured(m, n):
    return lin(n, 292, m["p292"], 2048, m["p2048"])


def weight_floor_ms(total_b):
    return total_b * 1e9 * NVFP4_B / (BW_EFF * THOR_BW) * 1e3


def tokens(cams, frames):
    return cams * frames * TOK_PER_CAM + TOK_PROMPT


def t_measured(name, cams, frames):
    m = MEASURED[name]
    n = tokens(cams, frames)
    vit = cams * frames * m["vit"]
    return dict(n=n, vit=round(vit, 1), prefill=round(prefill_measured(m, n), 1),
                total=round(vit + prefill_measured(m, n), 1), extrapolated_below_292=n < 292)


def prefill_dense_interp(body, n):
    """Inside the measured body range (1.41 .. 6.95 B): linear interpolation in body size of the measured
    prefill time at the same n (the three measured Qwen3-VL rows). Below 1.41 B: the 2B row (floor)."""
    pts = [(MEASURED[k]["body"], prefill_measured(MEASURED[k], n)) for k in ("Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B")]
    if body <= pts[0][0]:
        return pts[0][1]
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if body <= b1:
            return t0 + (body - b0) * (t1 - t0) / (b1 - b0)
    return None


def vit_ms_for(body):
    """Published ViT ms per image follows the family: SigLIP2-L (11.6) for the 2B/4B rows, SO400M (15.7) for 8B."""
    return 11.6 if body <= 4.0 else 15.7


def t_bracket(kind, body, total, vit_ms, cams, frames):
    n = tokens(cams, frames)
    vit = cams * frames * vit_ms
    floor = weight_floor_ms(total)
    if kind == "moe":
        opt = max(2 * body * 1e9 * n / TFLOPS_MAX * 1e3, floor)
        pes = max(MOE_PF2048, opt)                  # cannot be slower than the published 2,048-token row
    else:
        m8 = MEASURED["Qwen3-VL-8B"]
        opt = max(2 * body * 1e9 * n / TFLOPS_MAX * 1e3, floor)
        inside = prefill_dense_interp(body, n)
        if inside is not None:                      # inside the measured range: interpolation, no bracket
            opt = pes = max(inside, floor)
        else:
            pes = max(prefill_measured(m8, n) * body / m8["body"], floor, opt)
    return dict(n=n, vit=round(vit, 1), prefill_opt=round(opt, 1), prefill_pes=round(pes, 1),
                total_opt=round(vit + opt, 1), total_pes=round(vit + pes, 1),
                weight_floor_ms=round(floor, 1), beyond_2x_measured=body > 2 * MEASURED["Qwen3-VL-8B"]["body"])


SCEN = [(1, 1), (3, 1), (7, 1), (3, 3)]     # (cameras, frames per camera in the VLM input)


def verdict(t_ms, f_r):
    """PASS iff duty <= DUTY_MAX at refresh rate f_r (incl. overhead)."""
    return (t_ms + OVERHEAD_MS) * 1e-3 * f_r <= DUTY_MAX


def main():
    out = {"evidence_class": "ESTIMATED", "assumptions": dict(
        BW_EFF=BW_EFF, OVERHEAD_MS=OVERHEAD_MS, DUTY_MAX=DUTY_MAX, TOK_PER_CAM=TOK_PER_CAM,
        TOK_PROMPT=TOK_PROMPT, VLM_MEM_BUDGET_GB=VLM_MEM_BUDGET_GB), "measured_models": {}, "bracketed_models": {},
        "max_dense_body_B": {}, "cache_GB": {}, "memory_GB": {}}

    # 1. reproduce Table 'thor' of sec_eff.tex (regression check on the interpolation convention)
    expect = {("Qwen3-VL-2B", 1): 23.5, ("Qwen3-VL-2B", 3): 49.1, ("Qwen3-VL-2B", 7): 100.3,
              ("Qwen3-VL-4B", 1): 32.4, ("Qwen3-VL-4B", 3): 62.8, ("Qwen3-VL-4B", 7): 123.8,
              ("Qwen3.5-4B", 1): 40.0, ("Qwen3.5-4B", 3): 73.4, ("Qwen3.5-4B", 7): 140.2,
              ("Qwen3-VL-8B", 1): 45.0, ("Qwen3-VL-8B", 3): 89.6, ("Qwen3-VL-8B", 7): 178.8}
    for (name, cams), want in expect.items():
        got = t_measured(name, cams, 1)["total"]
        assert abs(got - want) <= 0.15, (name, cams, got, want)
    out["regression_vs_sec_eff_table_thor"] = "PASS (12 cells within 0.15 ms)"

    # 2. measured-anchored models, four scenarios, three refresh rates
    for name in MEASURED:
        rows = {}
        for cams, frames in SCEN:
            t = t_measured(name, cams, frames)
            t["age_worst_ms_at_2Hz"] = round(t["total"] + OVERHEAD_MS + 500, 0)
            t["duty_ok"] = {f"{hz}Hz": verdict(t["total"], hz) for hz in (2, 5, 10)}
            rows[f"{cams}cam x {frames}f"] = t
        out["measured_models"][name] = rows

    # 3. bracketed models
    for name, (kind, body, total, vit, hidden, note) in UNMEASURED.items():
        rows = {}
        for cams, frames in SCEN:
            t = t_bracket(kind, body, total, vit, cams, frames)
            t["duty_ok_opt"] = {f"{hz}Hz": verdict(t["total_opt"], hz) for hz in (2, 5, 10)}
            t["duty_ok_pes"] = {f"{hz}Hz": verdict(t["total_pes"], hz) for hz in (2, 5, 10)}
            t["age_worst_ms_at_2Hz_pes"] = round(t["total_pes"] + OVERHEAD_MS + 500, 0)
            rows[f"{cams}cam x {frames}f"] = t
        out["bracketed_models"][name] = dict(kind=kind, body_active_B=body, total_B=total, note=note, rows=rows)
        for prec, b in (("NVFP4", NVFP4_B), ("FP8", 1.0), ("BF16", 2.0)):
            out["memory_GB"].setdefault(name, {})[prec] = round(total * 1e9 * b / 1e9, 1)
        out["cache_GB"][name] = round(TICKS_2HZ * POOL_TOK * hidden * 3 / 1e9, 0)

    # 4. how large may a DENSE Qwen3-VL-lineage body be? (pessimistic = 8B-row rate, optimistic = 272 TFLOPS)
    grid = [round(0.5 + 0.1 * i, 1) for i in range(0, 600)]      # 0.5 .. 60.4 B
    for hz in (2, 5, 10):
        for cams, frames in SCEN:
            best = {}
            for tag in ("pes", "opt"):
                ok = 0.0
                for p in grid:
                    total_b = p * 1.06                              # embeddings + ViT overhead (ESTIMATED)
                    t = t_bracket("dense", p, total_b, vit_ms_for(p), cams, frames)
                    if verdict(t["total_" + tag], hz):
                        ok = p
                best[tag] = ok
            best["beyond_2x_measured_range"] = best["pes"] > 2 * MEASURED["Qwen3-VL-8B"]["body"]
            out["max_dense_body_B"][f"{hz}Hz_{cams}cam_x_{frames}f"] = best
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hicap_latency_budget_result.json"), "w") as f:
        json.dump(out, f, indent=1)

    # stdout tables
    print("== measured-anchored (ms per refresh, ViT+prefill, no overhead) ==")
    for name, rows in out["measured_models"].items():
        print(f"{name:14s}", "  ".join(f"{k}: {v['total']:6.1f}" for k, v in rows.items()))
    print("== bracketed (opt-pes, ms) ==")
    for name, d in out["bracketed_models"].items():
        print(f"{name[:44]:44s}", "  ".join(f"{k}: {v['total_opt']:.0f}-{v['total_pes']:.0f}" for k, v in d["rows"].items()),
              f"| floor {list(d['rows'].values())[0]['weight_floor_ms']} ms | NVFP4 {out['memory_GB'][name]['NVFP4']} GB | cache {out['cache_GB'][name]:.0f} GB")
    print("== max dense body (B params) meeting duty <= 0.5 ==")
    for k, v in out["max_dense_body_B"].items():
        print(f"{k:22s} pes {v['pes']:5.1f}  opt {v['opt']:5.1f}  beyond2x={v['beyond_2x_measured_range']}")


if __name__ == "__main__":
    sys.exit(main())
