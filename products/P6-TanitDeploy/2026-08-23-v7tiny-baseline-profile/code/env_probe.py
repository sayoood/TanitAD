"""Deploy-env probe. READ ONLY - installs nothing, allocates <100MB.

Verifies BY CONTENT: a real conv2d on CUDA, not `import torch`.
(CLAUDE.md: cuBLAS/matmul can succeed while cuDNN/conv is broken.)
"""
import sys, json, platform

out = {"python": sys.version.split()[0], "executable": sys.executable,
       "platform": platform.platform()}

try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device_name"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        # CONTENT check: a real conv2d on CUDA (cuDNN path), not just import.
        try:
            m = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
            x = torch.randn(2, 3, 32, 32, device="cuda")
            y = m(x)
            torch.cuda.synchronize()
            out["conv2d_cuda"] = {"ok": True, "out_shape": list(y.shape),
                                  "finite": bool(torch.isfinite(y).all().item()),
                                  "abs_mean": float(y.abs().mean().item())}
        except Exception as e:
            out["conv2d_cuda"] = {"ok": False, "error": repr(e)[:300]}
        # fp16 + bf16 matmul content checks
        for dt, nm in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            try:
                a = torch.randn(64, 64, device="cuda", dtype=dt)
                b = torch.randn(64, 64, device="cuda", dtype=dt)
                c = a @ b
                torch.cuda.synchronize()
                out[nm + "_matmul"] = {"ok": True,
                                       "finite": bool(torch.isfinite(c).all().item())}
            except Exception as e:
                out[nm + "_matmul"] = {"ok": False, "error": repr(e)[:200]}
except Exception as e:
    out["torch"] = "FAIL: " + repr(e)[:300]

for mod in ["tensorrt", "onnx", "onnxruntime", "torch_tensorrt", "polygraphy",
            "torchvision", "numpy", "triton"]:
    try:
        m = __import__(mod)
        out[mod] = str(getattr(m, "__version__", "present-no-version"))
    except Exception as e:
        out[mod] = "MISSING (" + type(e).__name__ + ")"

print(json.dumps(out, indent=1))
