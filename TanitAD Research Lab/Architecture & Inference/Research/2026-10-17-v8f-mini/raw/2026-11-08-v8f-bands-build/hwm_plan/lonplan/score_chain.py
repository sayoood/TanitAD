"""LON-PLAN SCORE chain (AMEND_HWMPLAN_3 section 4; MM 2026-11-16). Waits for A0_s0's sentinel, then runs the remaining planning arms ONE AT A
TIME (the dev box ran out of commit at 18:02Z with jobs stacked), then C-READBACK on every folder. λ* from the Log entry (landed 9ef0c72a):
λ*_0 = 300, λ*_1 = 100; R at -max(λ*_k, 1). Sentinel (last line of score_chain.log): ZZ LONSCORE CHAIN DONE ZZ | ZZ LONSCORE CHAIN FAIL ZZ <why>.
"""
import os
import subprocess
import sys
import time

L = "F:/TanitAD-scratch/actrsp/lonplan"
B = L + "/build"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CH = {"s0": dict(win=F"F:/TanitAD-scratch/actsens/read/out/C7_R35_s0/win/windows.npz",
                 wsha="371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt",
                 cksha="d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf", lam=300.0),
      "s1": dict(win=L + "/score_windows_s1/windows.npz",
                 wsha="8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt",
                 cksha="a97e6017cffb13830dacd04e1ea108fa8534bc3ab3007b8f536f01ac0846f881", lam=100.0)}
ARMS = [("A0_s1", "s1", 0.0, [0, 1]), ("A1_s0", "s0", None, [0, 1]), ("A1_s1", "s1", None, [0, 1]),
        ("R_s0", "s0", "neg", [0]), ("R_s1", "s1", "neg", [0])]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           WMACT_CODE_ROOT="F:/TanitAD-scratch/ladder_tree",
           PYTHONPATH=r"F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fail(why):
    print(f"ZZ LONSCORE CHAIN FAIL ZZ {why}", flush=True)
    sys.exit(1)


def lam_of(k, spec):
    if spec is None:
        return CH[k]["lam"]
    if spec == "neg":
        return -max(CH[k]["lam"], 1.0)
    return float(spec)


def plan(name, k, lam, seeds):
    out = f"{L}/score/{name}"
    if os.path.exists(f"{out}/DONE"):
        log(name, "skip (DONE exists)")
        return out
    c = CH[k]
    argv = [PY, "lonplan_run.py", "plan", "--windows-src", c["win"], "--windows-sha256", c["wsha"], f"--declare-lam={lam}",
            "--declare-seed-mode", "hold", "--ckpt-sha256", c["cksha"], "--",
            "--tag", f"R35_{k}", "--ckpt", c["ck"], "--train-latents", f"F:/TanitAD-scratch/hwm_plan/latents_train_R35_{k}.pt",
            "--train-tag", f"R35_{k}", "--seed-mode", "hold", f"--lam={lam}", "--arms", "hwm_oracle", "--seeds", *map(str, seeds),
            "--workers", "1", "--out", out]
    log(name, "start lam", lam, "seeds", seeds)
    t = time.time()
    with open(f"{L}/score_{name}.log", "w", encoding="utf-8") as f:
        r = subprocess.run(argv, cwd=B, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(name, "rc", r.returncode, round((time.time() - t) / 60, 1), "min")
    if r.returncode != 0:
        fail(f"{name} plan rc {r.returncode} (score_{name}.log)")
    open(f"{out}/DONE", "w").write("ok\n")
    return out


def main():
    a0 = f"{L}/score_A0_s0.log"
    for _ in range(240):
        txt = open(a0, encoding="utf-8", errors="replace").read() if os.path.exists(a0) else ""
        if "ZZ LONSCORE A0_s0 DONE ZZ" in txt:
            break
        if "ZZ LONSCORE A0_s0 FAIL ZZ" in txt:
            fail("A0_s0 failed (score_A0_s0.log)")
        time.sleep(60)
    else:
        fail("A0_s0 sentinel not seen in 4 h")
    open(f"{L}/score/A0_s0/DONE", "w").write("ok\n")
    outs = {"A0_s0": f"{L}/score/A0_s0"}
    for name, k, spec, seeds in ARMS:
        outs[name] = plan(name, k, lam_of(k, spec), seeds)
    for name, out in outs.items():
        with open(f"{L}/score_readback_{name}.log", "w", encoding="utf-8") as f:
            r = subprocess.run([PY, "lonplan_run.py", "readback", "--out", out], cwd=B, env=env, stdout=f, stderr=subprocess.STDOUT)
        log(name, "readback rc", r.returncode)
        if r.returncode != 0:
            fail(f"C-READBACK {name} rc {r.returncode}")
    print("ZZ LONSCORE CHAIN DONE ZZ", flush=True)


if __name__ == "__main__":
    main()
