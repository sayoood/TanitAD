"""MM 2026-11-18: run the OPTB trainer tests ON THE DEV-BOX GPU (device auto -> cuda), under the shared dev-box GPU lock (stage2_sft protocol).
Arm F (fixed hjepa.py) must be green. Arm R (the pre-fix hjepa.py swapped in, then restored by sha256) must go RED on the gate test: proof the test reaches the device path.
usage: gpu_test.py  (writes F:/TanitAD-scratch/actrsp/optb/devfix/gpu_test.json; last stdout line ZZ OPTB GPUTEST PASS|FAIL)"""
import hashlib, json, os, shutil, subprocess, sys, time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
ME = f"optb-gputest MM pid {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}"
D = "F:/TanitAD-scratch/actrsp/optb/devfix"
T = "F:/TanitAD-scratch/actrsp/optb/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
HJ = f"{T}/stack/tanitad/models/hjepa.py"
OLD = "F:/TanitAD-scratch/actrsp/optb/hjepa.before_devfix.py"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else 10 ** 9


def take_lock():
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, ME.encode()); os.close(fd); return
            except FileExistsError:
                pass
        print(f"waiting for the GPU lock {time.strftime('%T')}", flush=True); time.sleep(60)


def release():
    try:
        if open(LOCK, encoding="utf-8").read() == ME:
            os.remove(LOCK); print(f"GPU lock released {time.strftime('%T')}", flush=True)
    except OSError:
        pass


def pytest(tag, kexpr=None):
    env = dict(os.environ); env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", TMP=f"{D}/tmp", TEMP=f"{D}/tmp",
               PYTHONPATH=f"{T}/stack;{T}/taniteval".replace("/", "\\"))
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={D}/tmp/bt_{tag}", f"{T}/stack/tests/test_v8f_optb.py"] + (["-k", kexpr] if kexpr else [])
    t = time.time()
    r = subprocess.run(cmd, cwd=f"{T}/stack", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    open(f"{D}/pytest_{tag}.log", "w", encoding="utf-8").write(r.stdout + r.stderr)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"rc": r.returncode, "tail": tail, "min": round((time.time() - t) / 60, 2)}


def main():
    os.makedirs(f"{D}/tmp", exist_ok=True)
    fixed_sha = sha(HJ); old_sha = sha(OLD)
    assert fixed_sha != old_sha
    res = {"fixed_hjepa_sha256": fixed_sha, "old_hjepa_sha256": old_sha}
    take_lock()
    try:
        for _ in range(30):
            if used_mib() < 3500:
                break
            time.sleep(20)
        res["used_mib_at_start"] = used_mib()
        if res["used_mib_at_start"] >= 3500:
            res["problem"] = "GPU not idle under the lock (>= 3,500 MiB)"; return res
        chk = subprocess.run([PY, "-c", "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"], capture_output=True, text=True, encoding="utf-8")
        res["cuda"] = chk.stdout.strip()
        res["F_fixed_full_file"] = pytest("F")
        shutil.copyfile(OLD, HJ)
        try:
            res["R_prefix_gate"] = pytest("R", "gate_passes_for_the_rsp_arm")
        finally:
            shutil.copyfile(f"{D}/hjepa.fixed.py", HJ)
        res["restored_sha256"] = sha(HJ)
    finally:
        release()
    return res


if __name__ == "__main__":
    shutil.copyfile(HJ, f"{D}/hjepa.fixed.py")
    r = main()
    ok = (r.get("restored_sha256") == r["fixed_hjepa_sha256"] and r.get("F_fixed_full_file", {}).get("rc") == 0
          and r.get("R_prefix_gate", {}).get("rc") not in (0, None) and str(r.get("cuda", "")).startswith("True"))
    r["verdict"] = "PASS" if ok else "FAIL"
    json.dump(r, open(f"{D}/gpu_test.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(r, indent=1))
    print(f"ZZ OPTB GPUTEST {r['verdict']}")
