"""First-contact smoke for a Colab VM: what hardware we got, whether CUDA really works, and how fast the VM pulls.

    & colab\\colab.ps1 exec -s <name> --timeout 600 -f colab\\gpu_smoke.py

Prints one `ZZSMOKE {json}` line and writes the same JSON to /content/gpu_smoke.json after EVERY section (a client
timeout must never cost the parts already measured -- RUNNER.md section 9, trap 4). ASCII-only on purpose: the CLI
reads this file with the dev box's locale codec. Nothing here needs a token.
"""
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request

OUT = "/content/gpu_smoke.json"
R = {"t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1)
    os.replace(tmp, OUT)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


# 1) the machine
R["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
R["cuda_driver"] = sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")
R["cpus"] = os.cpu_count()
R["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
du = shutil.disk_usage("/content")
R["disk_content_free_gb"] = round(du.free / 1e9, 1)
R["python"] = platform.python_version()
save()

# 2) CUDA for real: a conv2d (cuDNN) and a matmul (cuBLAS) -- `import torch` proves neither (CLAUDE.md torch trap)
try:
    import torch
    R["torch"] = torch.__version__
    R["cuda_available"] = torch.cuda.is_available()
    if R["cuda_available"]:
        d = torch.device("cuda")
        x = torch.randn(8, 3, 224, 224, device=d)
        w = torch.randn(16, 3, 3, 3, device=d)
        y = torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        R["conv2d_ok"] = bool(torch.isfinite(y).all().item())
        for dt, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")):
            a = torch.randn(8192, 8192, device=d, dtype=dt)
            b = torch.randn(8192, 8192, device=d, dtype=dt)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize()
            t0 = time.time()
            n = 20
            for _ in range(n):
                a @ b
            torch.cuda.synchronize()
            R["matmul_tflops_" + name] = round(2 * 8192 ** 3 * n / (time.time() - t0) / 1e12, 1)
            del a, b
        R["max_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
except Exception as e:  # a smoke reports, it does not crash
    R["torch_error"] = f"{type(e).__name__}: {e}"
save()

# 3) how fast the VM pulls from HF (public file, streamed for ~8 s, no token)
try:
    url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(url, timeout=20) as r:
        while time.time() - t0 < 8:
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
    R["hf_pull_mb_s"] = round(n / 1e6 / (time.time() - t0), 1)
except Exception as e:
    R["hf_pull_error"] = f"{type(e).__name__}: {e}"
R["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("ZZSMOKE " + json.dumps(R))
