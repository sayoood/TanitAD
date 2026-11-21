"""Render markdown tables from profile.json (schema = refa_v1_profile.py). Prints to stdout."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_probe/profile_out/profile.json")
d = json.loads(p.read_text(encoding="utf-8"))


def f(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


print("## env"); e = d.get("env", {})
print(f"torch {e.get('torch')} cuda {e.get('cuda')} gpu {e.get('gpu')} {f(e.get('gpu_total_gb'),2)} GB, SMs {e.get('sm_count')}, "
      f"allocator cap {f(e.get('allocator_cap_gb'),2)} GB, tf32 {e.get('matmul_allow_tf32')}")
for k, v in (e.get("snapshot") or {}).items():
    print(f"  {k}: git-blob(LF) {v['git_blob_sha1_lf']} bytes {v['bytes']} crlf={v['had_crlf']}")

if "fit" in d:
    print("\n## fit\n| precision | bs | fits | max_mem_gb | first_step_s | note |\n|---|---|---|---|---|---|")
    for r in d["fit"]["rows"]:
        print(f"| {r['precision']} | {r['bs']} | {r['fits']} | {f(r.get('max_mem_gb'),2)} | {f(r.get('first_step_s'),2)} | {r.get('error','')[:60]} |")

if "loader" in d:
    L = d["loader"]
    print("\n## loader")
    print(f"hot batch(8) median {f(L['batch_lru_hot_s']['median']*1000)} ms; cold batch(8) median {f(L['batch_lru_cold_s']['median']*1000)} ms "
          f"(misses/batch {L['batch_lru_cold_s']['misses_per_batch']})")
    m = L["miss_cost_warm_pagecache_s"]
    print(f"miss cost warm page-cache: fp8 file {m['fp8_cache_file']['bytes']/2**20:.1f} MB {f(m['fp8_cache_file']['median']*1000)} ms; "
          f"v2ep {m['v2ep_file']['bytes']/2**20:.1f} MB {f(m['v2ep_file']['median']*1000)} ms; per window {f(m['per_window_total_median']*1000)} ms; x8 {f(m['per_batch_of_8_est_s'],3)} s")
    for r in L.get("cold_pagecache_reads", []):
        print(f"cold read {r['file'][:8]} {r['bytes']/2**20:.0f} MB first {f(r['first_read_s'],3)} s ({f(r['first_read_MBps'],0)} MB/s) second {f(r['second_read_s'],3)} s")
    s = L["slice_dequant_per_window_s"]
    print(f"slice+dequant per window {f(s['median']*1000,2)} ms ({s['dtype_on_disk']}); x8 {f(s['per_batch_of_8_est_s']*1000)} ms")

if "eig" in d:
    E = d["eig"]
    print("\n## eig / instruments (ms, median)")
    for k in ("live_gpu_fp64", "live_call_cpu", "fp32_variant_gpu", "chan_std_op", "chan_std_tac", "chan_std_str", "adapter_std_logstep"):
        if k in E:
            print(f"  {k}: {f(E[k]['median_ms'],2)} ms  {E[k].get('shape','')}")
    print(f"  rows {E['eig_rows']} d {E['d']} (z rows total {E['z_rows_total']}); participation {E.get('participation_value')}")

if "sweep" in d:
    S = d["sweep"]
    for key in ("forward_full", "op_rollout"):
        print(f"\n## sweep {key} (no_grad, median ms)\n| precision | bs1 | bs2 | bs4 | bs8 | bs8/bs1 |\n|---|---|---|---|---|---|")
        by = {}
        for r in S[key]:
            by.setdefault(r["precision"], {})[r["bs"]] = r
        for prec, rows in by.items():
            cells = []
            for bs in (1, 2, 4, 8):
                r = rows.get(bs)
                cells.append("OOM" if r is None or "oom" in r else f"{r['median_ms']:.0f} ({r['max_mem_gb']:.1f}GB)")
            r1, r8 = rows.get(1), rows.get(8)
            ratio = (r8["median_ms"] / r1["median_ms"]) if r1 and r8 and "median_ms" in r1 and "median_ms" in r8 else float("nan")
            print(f"| {prec} | " + " | ".join(cells) + f" | {ratio:.2f} |")

if "graph" in d:
    print("\n## graph (30 operative steps forward, median ms)\n| arm | eager | replay | speedup | max|diff| |\n|---|---|---|---|---|")
    for k, r in d["graph"].items():
        if "error" in r:
            print(f"| {k} | ERROR {r['error'][:80]} | | | |")
        else:
            print(f"| {k} | {f(r['eager']['median_ms'])} | {f(r['graph_replay']['median_ms'])} | {f(r['speedup'],2)}x | {r['max_abs_diff_vs_eager']:.2e} |")

for key in [k for k in d if k.startswith("step_")]:
    S = d[key]
    print(f"\n## {key}")
    for mode, w in S.get("wall", {}).items():
        print(f"  [{mode}] wall {f(w['wall_s_per_step'],3)} s/step  cpu-side {f(w['cpu_side_s_per_step_median'],3)} s  loader_cpu {f(w['loader_cpu_s_median'],3)} s  "
              f"maxmem {f(w['max_mem_gb'],2)} GB spilled={w.get('spilled_over_vram_budget')} offload={w.get('offload_activations')}")
        ph = w["phases_per_step"]
        print("  | phase | gpu_span_ms | cpu_s | calls |\n  |---|---|---|---|")
        for name, v in sorted(ph.items(), key=lambda kv: -kv[1]["gpu_span_ms"]):
            print(f"  | {name} | {f(v['gpu_span_ms'])} | {f(v['cpu_s'],3)} | {f(v['n_calls'],1)} |")
    P = S.get("profiler")
    if P:
        print(f"  profiler: kernel time {f(P['kernel_time_ms_per_step'])} ms/step; wall under profiler {f(P['wall_under_profiler']['wall_s_per_step'],3)} s; "
              f"GPU busy {f(P['trace']['gpu_busy_frac']*100)} %; kernels/step ~{P['trace']['kernel_hist']['n_kernels']/max(P['trace']['n_profiler_steps'],1):.0f}, "
              f"median kernel {f(P['trace']['kernel_hist']['median_us'])} us, frac kernels <10us {f(P['trace']['kernel_hist']['frac_kernels_lt_10us']*100)} %, "
              f"frac time in kernels <50us {f(P['trace']['kernel_hist']['frac_time_in_kernels_lt_50us']*100)} %")
        print("  | marker | dev_total_ms | dev_self_ms | cpu_total_ms | calls |\n  |---|---|---|---|---|")
        for name, v in sorted(P["markers_per_step_ms"].items(), key=lambda kv: -kv[1]["dev_total_ms"]):
            print(f"  | {name} | {f(v['dev_total_ms'])} | {f(v['dev_self_ms'])} | {f(v['cpu_total_ms'])} | {f(v['count_per_step'],1)} |")
        ba = P["trace"].get("backward_attribution", {})
        print(f"  backward attribution: {ba.get('status')}")
        if ba.get("status") == "ok":
            n = max(P["trace"]["n_profiler_steps"], 1)
            print("  | phase | fwd kernel ms/step | bwd kernel ms/step |\n  |---|---|---|")
            keys = sorted(set(ba["forward_kernel_ms_by_phase"]) | set(ba["backward_kernel_ms_by_phase"]),
                          key=lambda k: -(ba["forward_kernel_ms_by_phase"].get(k, 0) + ba["backward_kernel_ms_by_phase"].get(k, 0)))
            for k in keys:
                print(f"  | {k} | {f(ba['forward_kernel_ms_by_phase'].get(k,0)/n)} | {f(ba['backward_kernel_ms_by_phase'].get(k,0)/n)} |")
            print(f"  unattributed {f(ba['unattributed_kernel_ms']/n)} ms/step")
        print("  top kernels:")
        for i, k in enumerate(P["top_kernels"][:20], 1):
            print(f"   {i:2d}. {k['dev_self_ms_per_step']:7.1f} ms  {k['pct_of_kernel_time']:5.1f}%  x{k['count_per_step']:6.0f}  avg {k['avg_us']:8.1f} us  {k['name'][:80]}")

for key in [k for k in d if k.startswith("components_")]:
    C = d[key]
    print(f"\n## {key} (bs {C['bs']}, {C['precision']})\n| item | fwd_ms | bwd_ms | max_mem_gb | note |\n|---|---|---|---|---|")
    for name, r in C["items"].items():
        print(f"| {name} | {f(r.get('fwd_ms', r.get('ms')))} | {f(r.get('bwd_ms'))} | {f(r.get('max_mem_gb'),2)} | {r.get('oom','')[:50]} |")
