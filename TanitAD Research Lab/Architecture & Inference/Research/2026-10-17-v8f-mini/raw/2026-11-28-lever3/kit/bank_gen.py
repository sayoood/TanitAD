#!/usr/bin/env python3
"""bank_gen.py -- the dev-box DAgger bank of card LEVER3 (CPU). Port of the pod's rh_bank.sh + the chain's mk_bank step (rh_s3_chain.sh:152-181).
Flags exactly as the pod's s2 bank (bank/s2/PROVENANCE.json, MEASURED): `--readout causal_g --acc-gain 1 --path-exec --kappa-max 0.2`, generator ckpt = UE_sK/A, N shards in parallel, merge, stock BANK_CHECK
(rh_pilot_check.py bank, readout causal_g1x), then l3_bank_check.py (port of s3_bank_check.py, ckpt sha named).
Commands:
  join  : build the join npz ONCE with the trainer's own build_data under --fit-join (loads the v9 union npz, ~2 GB); the shards then LOAD it (they must not rewrite it).
  gen   : shards in parallel (+ merge + checks). Resumable: a shard whose npz + provenance exist is skipped. Writes <out>/BANK_DONE (a plain `>`) only after every check passed.
Memory guard: the box is shared and its free COMMIT is small; gen refuses to start unless free commit >= n_shards x --commit-per-shard-gib + 1.5 GiB (default 1.7 GiB per shard: measured 1.6 GiB on a 20-clip slice, see REPORT.md).
Windows paths with forward slashes. CPU only: CUDA_VISIBLE_DEVICES=-1 (an EMPTY value still shows the GPU).
usage: bank_gen.py gen --seed 0 --cache C:/Users/Admin/lever3/cache_union --join C:/Users/Admin/lever3/merged/join_union.npz --tracks .../native_tracks_union.pt --out C:/Users/Admin/lever3/bank/s0
       [--shards 4] [--threads 4] [--limit-clips N] [--expect-clips 14512 --expect-windows 985048]"""
import argparse, ctypes, glob, hashlib, json, os, subprocess, sys, time

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = os.environ.get("L3_TREE", "C:/Users/Admin/lever3/tree_cdb")
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PIL = KIT + "/rh_pilot_check.py"
UEA = KIT + "/inputs/UE_s0.A.json"            # the stage-A argv (the TRAIN list: cache, labels, join); seed is not read by build_data
GRID = KIT + "/inputs/eval_grid_windows.json"  # sha256 659c3b65... = the grid the pod's s2 bank recorded
V9 = "C:/Users/Admin/lever3/merged/v9_labels_union.npz"
P10 = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt"
INIT = "C:/Users/Admin/lever3/init/UE_s{seed}/A/ckpt_last.pt"
UE_SHA = {0: "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914", 1: "f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc"}   # sha256 of UE_s0/A, UE_s1/A ckpt_last.pt (MEASURED 2026-11-28)


def env(threads):
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_DISABLE_XET": "1", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
              "PYTHONPATH": ";".join([TREE + "/stack", TREE + "/stack/scripts", TREE]), "RH_TREE": TREE})
    return e


def free_commit_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def m2b_argv(a, fit_join):
    t = json.load(open(UEA))["argv"]
    ov = {"--cache-dir": a.cache, "--v9-train": a.v9, "--lead-train": a.lead, "--speed-train": a.speed, "--join-train": a.join, "--p10": P10, "--tracks": a.tracks,
          "--out": a.out + "/m2b_unused", "--device": "cpu", "--seed": str(a.seed)}
    for k, v in ov.items():
        t[t.index(k) + 1] = v
    if not fit_join:
        t.remove("--fit-join")
    return t


def cmd_join(a):
    os.makedirs(a.out, exist_ok=True)
    argv = m2b_argv(a, fit_join=True)
    p = a.out + "/m2b_argv_join.json"
    json.dump({"argv": argv}, open(p, "w"))
    code = ("import sys,json;import v8f_mini_train as TR;from tanitad.data import v8f_mini_data as D;"
            "ns=TR.build_parser().parse_args(json.load(open(sys.argv[1]))['argv']);cfg=TR.make_config(ns);"
            "store,J,wins,clips,info=TR.build_data(ns,cfg);print('join written',ns.join_train,len(clips),'clips',len(wins),'windows')")
    r = subprocess.run([PY, "-c", code, p], env=env(4), cwd=TREE)
    return r.returncode


def shard_done(out, k, n):
    stem = f"{out}/shard_{k:03d}of{n:03d}"
    return os.path.isfile(stem + ".npz") and os.path.isfile(stem + ".provenance.json")


def cmd_gen(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(out + "/BANK_DONE") and os.path.getsize(out + "/BANK_DONE") > 0:
        print("BANK_DONE exists; nothing to do")
        return 0
    need = a.shards * a.commit_per_shard_gib + 1.5
    fc = free_commit_gib()
    print(f"free commit {fc:.1f} GiB, need {need:.1f} GiB for {a.shards} shards")
    if fc < need:
        print(f"REFUSED: free commit {fc:.1f} GiB < {need:.1f} GiB (shrink --shards or wait); nothing started")
        return 3
    if not os.path.isfile(a.join):
        print(f"REFUSED: no join file {a.join}: run `bank_gen.py join` first")
        return 3
    ck = INIT.format(seed=a.seed)
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    if sha != UE_SHA[a.seed]:
        print(f"REFUSED: {ck} sha256 {sha} != {UE_SHA[a.seed]}")
        return 3
    margv = out + "/m2b_argv.json"
    json.dump({"argv": m2b_argv(a, fit_join=False)}, open(margv, "w"))
    base = [PY, TREE + "/stack/scripts/v8f_dagger_bank.py", "gen", "--ckpt", ck, "--m2b-argv", margv, "--cache-dir", a.cache, "--tracks", a.tracks, "--join-train", a.join,
            "--eval-grid", GRID, "--out-dir", out, "--n-shards", str(a.shards), "--device", "cpu", "--threads", str(a.threads),
            "--readout", "causal_g", "--acc-gain", "1", "--path-exec", "--kappa-max", "0.2"]
    if a.limit_clips:
        base += ["--limit-clips", str(a.limit_clips)]
    else:
        base += ["--expect-clips", str(a.expect_clips), "--expect-windows", str(a.expect_windows)]
    t0 = time.time()
    procs = {}
    for k in range(a.shards):
        if shard_done(out, k, a.shards):
            print(f"shard {k}: done already, skipped")
            continue
        procs[k] = subprocess.Popen(base + ["--shard", str(k)], env=env(a.threads), cwd=TREE, stdout=open(f"{out}/gen_{k}.log", "w"), stderr=subprocess.STDOUT)
    fails = 0
    for k, p in procs.items():
        rc = p.wait()
        if rc != 0:
            fails += 1
            print(f"shard {k} rc={rc}: {open(f'{out}/gen_{k}.log', errors='replace').read()[-300:]}")
    print(f"gen wall_s {time.time() - t0:.0f} failed_shards {fails}")
    if fails:
        return 1
    r = subprocess.run([PY, TREE + "/stack/scripts/v8f_dagger_bank.py", "merge", "--out-dir", out], env=env(a.threads), cwd=TREE)
    if r.returncode != 0 or not os.path.isfile(out + "/bank.json"):
        print("BANK_GEN FAIL merge / no bank.json (a control failed: see PROVENANCE.json)")
        return 1
    bj = json.load(open(out + "/bank.json"))
    ew = a.expect_windows if not a.limit_clips else bj["common"]["train"]["n_windows"]
    r = subprocess.run([PY, PIL, "bank", "--bank", out, "--readout", "causal_g1x", "--expect-windows", str(ew), "--out", out + "/BANK_CHECK.json"], env=env(a.threads), cwd=TREE)
    if r.returncode != 0:
        ok_partial = False
        if a.limit_clips and os.path.isfile(out + "/BANK_CHECK.json"):       # a smoke bank is partial by design: only check e (complete train cover) may fail
            ck = json.load(open(out + "/BANK_CHECK.json"))["checks"]
            ok_partial = all(v.get("pass") for k, v in ck.items() if not k.startswith("e_"))
        if not ok_partial:
            print("BANK_GEN FAIL BANK_CHECK (stock check)")
            return 1
    cmd = [PY, KIT + "/l3_bank_check.py", "--bank", out, "--expect-windows", str(ew), "--ue-sha", UE_SHA[a.seed], "--out", out + "/L3_BANK_CHECK.json"]
    if a.limit_clips:
        cmd.append("--allow-partial")
    r = subprocess.run(cmd, env=env(a.threads), cwd=TREE)
    if r.returncode != 0:
        print("BANK_GEN FAIL L3_BANK_CHECK")
        return 1
    bsha = hashlib.sha256(open(out + "/bank.json", "rb").read()).hexdigest()      # the --dagger-bank-sha256 of the arm argv
    open(out + "/BANK_DONE", "w").write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} bank_sha256 {bsha} partial {bj.get('partial')}\n")
    print("ZZ BANK_GEN DONE ZZ", bsha, "total_wall_s", int(time.time() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["join", "gen"])
    ap.add_argument("--seed", type=int, required=True, choices=[0, 1])
    ap.add_argument("--cache", required=True, help="the cache root (holds train/clips and train/manifest.jsonl)")
    ap.add_argument("--join", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v9", default=V9)
    ap.add_argument("--lead", default="C:/Users/Admin/lever3/merged/lead_union.npz")
    ap.add_argument("--speed", default="C:/Users/Admin/lever3/merged/speed_union.jsonl")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--expect-clips", type=int, default=14512)
    ap.add_argument("--expect-windows", type=int, default=985048)
    ap.add_argument("--commit-per-shard-gib", type=float, default=1.7)
    a = ap.parse_args()
    sys.exit(cmd_join(a) if a.cmd == "join" else cmd_gen(a))


main()
