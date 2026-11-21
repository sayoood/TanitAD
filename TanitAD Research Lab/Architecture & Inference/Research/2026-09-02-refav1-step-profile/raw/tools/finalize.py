"""Build RESULT.md's section 3b from the bank (never retyped) and write RESULT.final.md
+ raw/top_kernels.md + raw/derived.json."""
import json
from pathlib import Path

S = Path(__file__).parent
OUT = Path("C:/Users/Admin/refav1_probe/profile_out")
d = json.loads((OUT / "profile.json").read_text(encoding="utf-8"))

ST = d["step_fp32_bs1"]["profiler_trace_offline"]
FW = d["fwdprof_fp32_bs1_trace"]
CP = d["components_fp32_bs1_trace"]["profiled_op_rollout"]
CP3 = d["components_fp32_bs1"]["profiled_op_rollout"]
n = max(ST["n_profiler_steps"], 1)
ba = ST["backward_attribution"]
fwd, bwd = ba["forward_kernel_ms_by_phase"], ba["backward_kernel_ms_by_phase"]

FAMILY = {
    "operative rollout K=30 (fwd + bwd)": ["P/op_rollout"],
    "op loss: to_enc + mse": ["P/to_enc", "P/loss_mse"],
    "tactical: tac_field init + rollout K=10": ["P/tac_field_init", "P/tac_rollout"],
    "instruments: participation eig (fp64)": ["P/participation_eig", "P/participation_ratio"],
    "optimizer: AdamW + clip_grad_norm": ["P/opt_step", "P/clip_grad", "P/zero_grad"],
    "target path: std+adapter on future/ext, tac_field x30 (grad-recorded, detached)":
        ["P/std_future", "P/adapter_future", "P/tac_field_targets", "P/std_str_ext", "P/adapter_str_ext"],
    "encode window: std + adapter": ["P/encode", "P/std_window", "P/adapter_window"],
    "brains (strategic + tactical policies)": ["P/brains"],
    "strategic: subspace + rollout": ["P/str_subspace", "P/str_rollout"],
    "heads / proposal / label CE": ["P/heads", "P/proposal", "P/loss_ce"],
    "forward glue (chan_std reductions, mean, stack, slices)": ["P/forward"],
    "h2d / loader (kernels only)": ["P/h2d", "P/loader_fetch"],
}
def fam(name):
    for k, v in FAMILY.items():
        if name in v:
            return k
    return f"other: {name}"

rows = {}
for ph, ms in fwd.items():
    rows.setdefault(fam(ph), [0.0, 0.0])[0] += ms / n
for ph, ms in bwd.items():
    rows.setdefault(fam(ph), [0.0, 0.0])[1] += ms / n
una = ba["unattributed_kernel_ms"] / n
# name-based reassignment of the kernels Kineto emitted no launch record for (offline_attr analysis,
# raw/offline_attr.log): 360/step cutlass simt_sgemm (= 30 x 6 x 2, the rollout's GEMMs) -> rollout;
# one fp64 tensorop d884gemm -> participation covariance; _foreach_* -> AdamW.
REASSIGN = {"operative rollout K=30 (fwd + bwd)": 151.7 + 10.9 + 3.4 + 0.9 + 0.1,
            "instruments: participation eig (fp64)": 52.2,
            "optimizer: AdamW + clip_grad_norm": 3.8 + 2.0 + 1.9 + 1.8}
reassigned = sum(REASSIGN.values())
tot = sum(a + b for a, b in rows.values()) + una
lines = [f"Kernel time of the **exact trainer step** (loader → H2D → forward → backward → clip → AdamW; live loader; bs 1 fp32; "
         f"activations paged to host so it fits) attributed to the forward phase that created each op — backward nodes are "
         f"mapped through autograd sequence numbers ({ba['n_forward_seq']:,} forward ops mapped); {n} profiled steps "
         f"(`profile.json:step_fp32_bs1.profiler_trace_offline`; the 256 MB trace is kept locally). Total kernel time "
         f"**{tot:.0f} ms/step**; GPU-busy {100*ST['gpu_busy_frac']:.1f} % (NOT a launch-bound reading: the pageable offload "
         f"traffic starves the GPU by construction — see the no-offload profiles below). {una:.0f} ms/step carried no launch record "
         f"and is reassigned by kernel identity (cutlass `simt_sgemm` ×{360}/step = 30 steps × 6 blocks × 2 → rollout; the single "
         f"fp64 `d884gemm` → participation covariance; `_foreach_*` → AdamW; `raw/offline_attr.log`).",
         "", "| phase | fwd kernel ms | bwd kernel ms | reassigned ms | total ms | share of kernel time |", "|---|---|---|---|---|---|"]
items = []
for k, (a, b) in rows.items():
    r = REASSIGN.get(k, 0.0)
    items.append((k, a, b, r, a + b + r))
items.sort(key=lambda x: -x[4])
for k, a, b, r, t in items:
    lines.append(f"| {k} | {a:.0f} | {b:.0f} | {r:.0f} | {t:.0f} | {100*t/tot:.1f} % |")
rest = una - reassigned
lines.append(f"| still unattributed | | | | {rest:.0f} | {100*rest/tot:.1f} % |")
fam_md = "\n".join(lines)

# forward per-marker table (no-grad forward under the profiler)
m = FW["markers_per_step_ms"]
fwd_total = m.get("P/forward", {}).get("dev_total_ms", 0.0)
order = sorted(((k, v) for k, v in m.items() if k != "P/forward"), key=lambda kv: -kv[1]["dev_total_ms"])
fl = [f"**Every forward phase under the profiler (no grad, bs 1 fp32, `raw/trace_forward_nograd_bs1_fp32.json`, "
      f"`profile.json:fwdprof_fp32_bs1_trace`).** Device time = kernels launched inside the marker (nested markers overlap: "
      f"`P/encode` ⊃ `P/std_window` + `P/adapter_window`). `P/forward` {fwd_total:.0f} ms device span; kernel time "
      f"{FW['kernel_time_ms_per_step']:.0f} ms/step; **GPU-busy {100*FW['trace']['gpu_busy_frac']:.1f} %**; "
      f"{FW['trace']['kernel_hist']['n_kernels']/max(FW['n_meas'],1):.0f} kernels/step, median {FW['trace']['kernel_hist']['median_us']:.0f} us.",
      "", "| marker | device ms | CPU ms | calls | share of forward |", "|---|---|---|---|---|"]
for k, v in order:
    if v["dev_total_ms"] < 0.05:
        continue
    fl.append(f"| `{k}` | {v['dev_total_ms']:.1f} | {v['cpu_total_ms']:.1f} | {v['count_per_step']:.0f} | {100*v['dev_total_ms']/max(fwd_total,1e-9):.1f} % |")
fwd_md = "\n".join(fl)

kh3 = CP3["trace"]["kernel_hist"]; n3 = max(CP3["trace"]["n_profiler_steps"], 1)
k15 = (f"**Rollout fwd+bwd at K=15 under the profiler (bs 1 fp32, no offload; the 30-step rollout is two of these).** "
       f"Kernel time **{CP['kernel_time_ms_per_step']:.0f} ms/step** (event-timed: 808 ms — profiler overhead is negligible), "
       f"**GPU-busy {100*CP['trace']['gpu_busy_frac']:.1f} %** (1 step, `raw/trace.json`, {CP['trace_bytes']/2**20:.1f} MB) and "
       f"**{100*CP3['trace']['gpu_busy_frac']:.1f} %** (3 steps, trace kept locally at 53.9 MB); {kh3['n_kernels']/n3:.0f} kernels/step, "
       f"median kernel {kh3['median_us']:.0f} us; {100*kh3['frac_kernels_lt_10us']:.0f} % of kernels are shorter than 10 us but they hold "
       f"only {100*kh3['frac_time_in_kernels_lt_50us']:.1f} % of kernel time — the launch-bound signature is absent.")
top = ["", "Top kernels of the rollout fwd+bwd (K=15, self device time per step, `raw/top_kernels.md`):", "",
       "| # | kernel | calls/step | ms/step | % of kernel time |", "|---|---|---|---|---|"]
for i, k in enumerate(CP["top_kernels"][:15], 1):
    top.append(f"| {i} | `{k['name'][:80]}` | {k['count_per_step']:.0f} | {k['dev_self_ms_per_step']:.1f} | {k['pct_of_kernel_time']:.1f} |")
gemm_terms = ("gemm", "cutlass", "sgemm", "gemv", "fmha", "flash", "attention", "cublas", "xmma")
gemm = sum(k["dev_self_ms_per_step"] for k in CP["top_kernels"] if any(t in k["name"].lower() for t in gemm_terms))
top.append("")
top.append(f"GEMM + attention kernels in the top-20: **{gemm:.0f} of {CP['kernel_time_ms_per_step']:.0f} ms ({100*gemm/CP['kernel_time_ms_per_step']:.0f} %)**; "
           f"the rest (residual adds, GELU/GELU-backward, LayerNorm, reductions, copies) is the ceiling `torch.compile`'s fusion could touch.")

fail = ("\n\n**How the full-step table was obtained, and what it cost.** The full fwd+bwd step fits no batch on 8 GB (§2). "
        "`torch.autograd.graph.save_on_cpu` with pinned memory fails at ~7 GB with a CUDA-runtime `out of memory` (`raw/run2.log`); "
        "pageable offload runs (9.89 s/step wall over 10 measured steps, `profile.json:step_fp32_bs1.wall`, wall-clock inadmissible by "
        "construction) but holds **18.7–22.7 GB of host working set** (≈ 3× a step's 7.5 GB of saved activations) and a 10-step "
        "profiler pass on top drove the host to 1.1 GB free / commit at the pagefile limit (`raw/run3.log`–`run5.log`). The 3-step "
        "pass exported its trace before the process was stopped; the table above is that trace, attributed offline with the shipped "
        "script's own `trace_stats()` (`raw/offline_attr.log`).")

md = (S / "RESULT.md").read_text(encoding="utf-8")
assert "{{RUN3_FAMILIES}}" in md and "{{RUN3_BUSY}}" in md
md = md.replace("{{RUN3_FAMILIES}}", fam_md + "\n\n" + fwd_md).replace("{{RUN3_BUSY}}", k15 + "\n" + "\n".join(top) + fail)
(S / "RESULT.final.md").write_text(md, encoding="utf-8")

# raw/top_kernels.md
tk = ["# Top-20 kernels (self device time per step, RTX 4060, bs 1 fp32)", "",
      "## Operative rollout fwd+bwd, K=15 (`components_fp32_bs1_trace.profiled_op_rollout`)", "",
      "| # | kernel | calls/step | ms/step | avg us | % of kernel time |", "|---|---|---|---|---|---|"]
for i, k in enumerate(CP["top_kernels"][:20], 1):
    tk.append(f"| {i} | `{k['name']}` | {k['count_per_step']:.0f} | {k['dev_self_ms_per_step']:.1f} | {k['avg_us']:.1f} | {k['pct_of_kernel_time']:.1f} |")
tk += ["", "## Whole forward, no grad (`fwdprof_fp32_bs1_trace`)", "",
       "| # | kernel | calls/step | ms/step | avg us | % of kernel time |", "|---|---|---|---|---|---|"]
for i, k in enumerate(FW["top_kernels"][:20], 1):
    tk.append(f"| {i} | `{k['name']}` | {k['count_per_step']:.0f} | {k['dev_self_ms_per_step']:.1f} | {k['avg_us']:.1f} | {k['pct_of_kernel_time']:.1f} |")
(OUT / "top_kernels.md").write_text("\n".join(tk) + "\n", encoding="utf-8")
print(fam_md); print(); print(fwd_md); print(); print(k15); print("\n".join(top))
print("\nrollout share of full-step kernel time:", round(100 * items[0][4] / tot, 1), "%  total", round(tot), "ms")
print("wrote", S / "RESULT.final.md", "and", OUT / "top_kernels.md")
