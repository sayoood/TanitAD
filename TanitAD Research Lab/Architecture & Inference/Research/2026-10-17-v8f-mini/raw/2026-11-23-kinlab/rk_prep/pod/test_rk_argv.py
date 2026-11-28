"""test_rk_argv.py -- REVISION 2 (Amendment 1). Dev-box test of rk_argv.py on RL_s0's REAL argv (read from the pod, pasted below). stdlib only, no pod.

The well-formed render must differ from RL's in exactly the label value, --out and the two appended tokens `--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` (token for token) and be refused/accepted as below; each deliberate regression
(a FOURTH difference, a wrong label, a wrong bank, a wrong dose, a moved token, the cache path of the OTHER seed's bank, the cache missing / moved / duplicated / of another kind) must REFUSE with the stated reason. Usage: python -I test_rk_argv.py <rk_argv.py>"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("rk_argv", sys.argv[1])
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

BSHA = "e5e44c0156ebc774e3551a54f589dc4fec19a68dc85c2c9838f7ecc0592662cf"
RL = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "/root/rollhist/runs/RL_s0/train", "--cache-dir", "/root/data/cache_union_dscale",
      "--v9-train", "/root/dscale10k/merged/v9_labels_union.npz", "--v9-train-md5", "9881b170a889f743c5b0894f9ad13f1d", "--lead-train", "/root/dscale10k/merged/lead_union.npz",
      "--speed-train", "/root/dscale10k/merged/speed_union.jsonl", "--join-train", "/root/dscale10k/merged/join_union.npz", "--p10", "/root/strat1/inputs/p10/p10_prior.pt",
      "--kin-mode", "yaw", "--tracks", "/root/dscale10k/merged/native_tracks_union.pt", "--batch", "64", "--steps-mult", "4", "--device", "cuda",
      "--init-ckpt", "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2",
      "--ego-perturb-p", "0", "--dagger-p", "0.1", "--dagger-bank", "/root/rollhist/bank/s0", "--dagger-bank-sha256", BSHA, "--dagger-seed", "0", "--w-acc", "49.6969",
      "--detach-upper", "--fit-join", "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2", "--dagger-label", "speed_v1"]
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(name, "PASS" if cond else "FAIL", extra)


def repl(t, old, new):
    t = list(t)
    t[t.index(old)] = new
    return t


# 1. the well-formed render
CACHE0 = "/root/rollhist/kin_cache/s0_kin_v1.npz"
rk, bad = M.build(RL, 0, "/root/rollhist", BSHA)
check("WELL-FORMED build", not bad and len(rk) == len(RL) + 2 and rk[-2:] == ["--dagger-label-cache", CACHE0] and rk[-4:-2] == ["--dagger-label", "kin_v1"] and "/root/rollhist/runs/RK_s0/train" in rk, bad)
pos = [i for i in range(len(RL)) if RL[i] != rk[i]]
check("WELL-FORMED differs from RL in exactly 2 token positions plus the 2 appended cache tokens", len(pos) == 2 and RL[pos[0] - 1] == "--out" and RL[pos[1] - 1] == "--dagger-label" and rk[:len(RL)][:pos[0]] == RL[:pos[0]], pos)
check("WELL-FORMED verify (diff_problems) clean", not M.diff_problems(RL, rk, 0, "/root/rollhist"))
rk1, bad1 = M.build([("/root/rollhist/bank/s1" if t == "/root/rollhist/bank/s0" else "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt" if t == "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt" else "/root/rollhist/runs/RL_s1/train" if t == "/root/rollhist/runs/RL_s0/train" else t) for t in RL], 1, "/root/rollhist", BSHA)
check("WELL-FORMED seed-1 cache path is s1's", not bad1 and rk1[-2:] == ["--dagger-label-cache", "/root/rollhist/kin_cache/s1_kin_v1.npz"], rk1[-2:])
check("WELL-FORMED seed-1 form (own init, bank, --out)", not M.build([("/root/rollhist/bank/s1" if t == "/root/rollhist/bank/s0" else "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt" if t == "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt" else "/root/rollhist/runs/RL_s1/train" if t == "/root/rollhist/runs/RL_s0/train" else t) for t in RL], 1, "/root/rollhist", BSHA)[1])

# 2. deliberate regressions of the RK argv (verify): each must be refused
V = {
    "V1 label left speed_v1": repl(rk, "kin_v1", "speed_v1"),
    "V2 label human": repl(rk, "kin_v1", "human"),
    "V3 a third flag appended": rk + ["--detach-lower"],
    "V4 a third value changed (--steps)": repl(rk, "3000", "3001"),
    "V5 another bank (bank_k)": repl(rk, "/root/rollhist/bank/s0", "/root/rollhist/bank_k/s0"),
    "V6 --out left RL's": repl(rk, "/root/rollhist/runs/RK_s0/train", "/root/rollhist/runs/RL_s0/train"),
    "V7 dose 0.3": repl(rk, "0.1", "0.3"),
    "V8 dagger-mode add appended": rk + ["--dagger-mode", "add"],
    "V9 bank sha changed": repl(rk, BSHA, "0" * 64),
    "V10 label token moved (not at its RL position)": rk[:5] + ["--dagger-label", "kin_v1"] + rk[5:-4] + ["--dagger-label-cache", CACHE0],
    "V11 init changed": repl(rk, "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt"),
    "V12 --out of RA": repl(rk, "/root/rollhist/runs/RK_s0/train", "/root/rollhist/runs/RA_s0/train"),
    # Amendment 1: the cache token pair
    "V13 cache path names the OTHER seed's bank (s1)": repl(rk, CACHE0, "/root/rollhist/kin_cache/s1_kin_v1.npz"),
    "V14 cache flag missing (RK = RL + two changes only)": rk[:-2],
    "V15 cache path of another kind (speed_v1)": repl(rk, CACHE0, "/root/rollhist/kin_cache/s0_speed_v1.npz"),
    "V16 cache path in another folder": repl(rk, CACHE0, "/tmp/s0_kin_v1.npz"),
    "V17 cache flag not last (a fourth flag after it)": rk + ["--detach-lower"],
    "V18 cache flag duplicated": rk + ["--dagger-label-cache", CACHE0],
    "V19 cache pair moved before the label (label not at its RL position)": rk[:-4] + ["--dagger-label-cache", CACHE0, "--dagger-label", "kin_v1"],
    "V20 cache path has no value": rk[:-1],
    "V21 cache path left as the other seed AND label speed_v1": repl(repl(rk, CACHE0, "/root/rollhist/kin_cache/s1_kin_v1.npz"), "kin_v1", "speed_v1"),
}
for k, t in V.items():
    b = M.diff_problems(RL, t, 0, "/root/rollhist")
    check(k, bool(b), b[:1])

# 3. deliberate regressions of the SOURCE (build): RL's own argv is wrong
B = {
    "B1 RL label human (not speed_v1)": repl(RL, "speed_v1", "human"),
    "B2 RL label not the last two tokens": RL + ["--w-extra", "1"],
    "B3 RL dose 0.3": repl(RL, "0.1", "0.3"),
    "B4 RL bank sha not bank.json's": repl(RL, BSHA, "1" * 64),
    "B5 RL already has dagger-mode": RL + ["--dagger-mode", "swap"],
    "B6 RL init is another seed's": repl(RL, "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt"),
    "B7 RL readout is causal_g1": repl(RL, "causal_g1x", "causal_g1"),
    "B8 RL already names a cache": RL[:-2] + ["--dagger-label-cache", CACHE0, "--dagger-label", "speed_v1"],
}
for k, t in B.items():
    b = M.build(t, 0, "/root/rollhist", BSHA)[1]
    check(k, bool(b), b[:1])
print("ZZ RK-ARGV TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
