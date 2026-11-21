'''Is NVFP4 real and USEFUL on Jetson Thor, at OUR tensor sizes?

CONTEXT: Thor reports sm_110 (Blackwell) and torch exposes float4_e2m1fn_x2, so the format is
plausibly native. NVFP4 is E2M1 with a per-block E4M3 scale plus a per-tensor FP32 scale;
published claims are up to ~4x over BF16 with ~1%% accuracy loss ON LLMs.

⚠️ THE REASON THIS NEEDS MEASURING RATHER THAN ADOPTING: today we measured bf16 make our
PREDICTOR ROLL *SLOWER* (0.86x) because its tensors are 1x8x2048 — far too small to repay
tensor-core setup. NVFP4 has the same failure mode, only more so: it adds block-scale packing
work. Published LLM speedups come from GEMMs orders of magnitude larger than ours.

So: (1) does an NVFP4 GEMM actually EXECUTE here, and (2) at what SIZE does it start winning —
above or below the sizes our model actually uses?
'''
import json, os, time
import torch

out = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
       'cuda': torch.version.cuda}
p = torch.cuda.get_device_properties(0)
out['sm'] = f'sm_{p.major}{p.minor}'
out['has_fp4_dtype'] = hasattr(torch, 'float4_e2m1fn_x2')
out['has_fp8_dtype'] = hasattr(torch, 'float8_e4m3fn')
out['scaled_mm_present'] = hasattr(torch, '_scaled_mm')

def bench_mm(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return round(ts[len(ts)//2], 4)

# our actual shapes vs LLM-scale shapes
SIZES = [(8, 2048, 2048, 'OUR predictor step'),
         (128, 2048, 2048, 'our batched'),
         (1024, 2048, 2048, 'medium'),
         (4096, 4096, 4096, 'LLM-scale')]
rows = {}
for M, K, N, tag in SIZES:
    a16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b16 = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    r = {'tag': tag, 'shape': f'{M}x{K}x{N}'}
    r['bf16_ms'] = bench_mm(lambda: a16 @ b16)
    # fp8 as the intermediate reference point
    try:
        af = a16.to(torch.float8_e4m3fn); bf = b16.t().contiguous().t().to(torch.float8_e4m3fn)
        s = torch.tensor(1.0, device='cuda')
        r['fp8_ms'] = bench_mm(lambda: torch._scaled_mm(af, bf, scale_a=s, scale_b=s,
                                                        out_dtype=torch.bfloat16))
        r['fp8_speedup_x'] = round(r['bf16_ms']/r['fp8_ms'], 2)
    except Exception as e:
        r['fp8'] = f'FAILED {type(e).__name__}: {str(e)[:110]}'
    rows[f'{M}x{K}x{N}'] = r
out['gemm'] = rows

# does an NVFP4 path execute at all?
try:
    from torch.nn.functional import scaled_mm  # newer API surface
    out['nvfp4_api'] = 'torch.nn.functional.scaled_mm present'
except Exception:
    out['nvfp4_api'] = 'no functional.scaled_mm'
try:
    x = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)
    q = x.to(torch.float4_e2m1fn_x2)
    out['fp4_cast'] = {'ok': True, 'dtype': str(q.dtype), 'bytes': q.element_size(),
                       'shape': tuple(q.shape)}
except Exception as e:
    out['fp4_cast'] = {'ok': False, 'err': f'{type(e).__name__}: {str(e)[:160]}'}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(os.path.expanduser('~/thor_nvfp4.json'),'w'), indent=1)
