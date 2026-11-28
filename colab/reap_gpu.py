"""Reap orphaned Colab kernels that still hold GPU memory.

⛔ WHY THIS EXISTS (MEASURED 2026-08-19, cost one failed launch and a wrong
diagnosis). `colab restart-kernel` starts a NEW kernel but does NOT always reap
the old one. The dead kernel keeps its CUDA context: after a restart the T4
reported **8,741 MiB used / 6,172 MiB free with nothing running**, and the next
4-bit load of Qwen3.5-9B — a model that had loaded fine at 8.0 GB minutes
earlier — died with:

    ValueError: Some modules are dispatched on the CPU or the disk.
    Make sure you have enough GPU RAM to fit the quantized model.

That message names the MODEL, so it reads as "the model is too big" and invites
the wrong fix (smaller model, more quantisation, a bigger GPU). The real state
was a leak, and the honest probe is `--query-compute-apps`, which named the
orphan (PID 4969, 8,566 MiB) immediately. Reaping it restored 14,738 MiB free.

Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps in
CLAUDE.md: **a symptom read as its own root cause.**

⚠️ TWO RULES THIS FILE ENCODES, both learned the hard way:
1. **Kill by EXPLICIT PID.** A pattern kill self-matches the killing process.
2. **Match BOTH kernel names.** The Colab launcher is `colab_kernel_launcher`,
   NOT `ipykernel`. Matching only `ipykernel` reported "not an ipykernel" for a
   kernel and skipped the very process holding the memory — absence found at one
   name is not absence.

Usage:  colab exec -s <session> -f colab/reap_gpu.py --timeout 240
        (with PYTHONUTF8=1 — see RUNNER.md)
"""
import os
import subprocess
import time

KERNEL_NAMES = ("ipykernel", "colab_kernel_launcher")


def compute_apps():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, mb = [x.strip() for x in line.split(",")]
        yield int(pid), int(mb)


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return "(gone)"


def main():
    me = os.getpid()
    killed = 0
    for pid, mb in compute_apps():
        cmd = cmdline(pid)
        if pid == me:
            print(f"  KEEP {pid} ({mb} MiB) - this kernel")
        elif not any(k in cmd for k in KERNEL_NAMES):
            print(f"  SKIP {pid} ({mb} MiB) not a kernel: {cmd[:80]}")
        else:
            print(f"  KILL {pid} ({mb} MiB): {cmd[:80]}")
            os.kill(pid, 9)
            killed += 1
    if killed:
        time.sleep(4)
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                          "--format=csv"], capture_output=True, text=True).stdout)
    print(f"reaped {killed} orphaned kernel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
