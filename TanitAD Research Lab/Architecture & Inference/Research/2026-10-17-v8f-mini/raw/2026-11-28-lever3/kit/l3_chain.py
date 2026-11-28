#!/usr/bin/env python3
"""l3_chain.py -- the LEVER3 chain: bank s0 -> bank s1 -> argv render + mints (gate, P3) -> D3K_s0 -> D9K_s0 -> D3K_s1 -> D9K_s1 (card PREREG_V8F_LEVER3_V2). NEVER launched by the builder; the MM launches it detached.
Resumable: every step writes a done-file in <root>/done_l3/ with a plain write AFTER its checks; a relaunch skips finished steps. rc checked at every step; first failure stops the chain with `ZZ L3 CHAIN FAIL ZZ <step>`.
Steps: join (trainer's build_data under --fit-join, once) -> bank_s0 -> bank_s1 -> render (l3_argv.py render with the two banks' sha256) -> verify (control b4) -> gate_<arm> (mint: kit/l3_gate_mint.py, NOT BUILT, P3) -> arm_<arm> (arm_run.py).
Banks run one after the other (the box cannot hold two sets of shards: commit headroom); arms run one at a time under the GPU lock (arm_run.py).
usage: l3_chain.py [--root C:/Users/Admin/lever3] [--shards 4] [--threads 4]"""
import argparse, hashlib, os, subprocess, sys, time
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="C:/Users/Admin/lever3")
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
R = a.root
DONE = f"{R}/done_l3"
os.makedirs(DONE, exist_ok=True)


def say(m):
    print(time.strftime("%FT%TZ", time.gmtime()), m, flush=True)


def fail(step, why=""):
    say(f"ZZ L3 CHAIN FAIL ZZ {step} {why}")
    sys.exit(1)


def step(name, fn):
    if os.path.isfile(f"{DONE}/{name}.done"):
        say(f"step {name}: done, skipped")
        return
    say(f"step {name}")
    rc = fn()
    if rc != 0:
        fail(name, f"rc={rc}")
    open(f"{DONE}/{name}.done", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")


def run(args):
    return subprocess.run([PY] + args).returncode


bank = lambda s: ["bank_gen.py", "gen", "--seed", str(s), "--cache", f"{R}/cache_union", "--join", f"{R}/merged/join_union.npz", "--tracks", f"{R}/merged/native_tracks_union.pt",
                  "--out", f"{R}/bank/s{s}", "--shards", str(a.shards), "--threads", str(a.threads)]
if not os.path.isfile(f"{R}/cache_union/train/manifest.jsonl"):
    fail("precheck", "no cache_union/train/manifest.jsonl (the MM writes it when the build ends)")
step("join", lambda: 0 if os.path.isfile(f"{R}/merged/join_union.npz") else run([KIT + "/bank_gen.py", "join", "--seed", "0", "--cache", f"{R}/cache_union", "--join", f"{R}/merged/join_union.npz",
                                                                              "--tracks", f"{R}/merged/native_tracks_union.pt", "--out", f"{R}/bank/join_build"]))
for s in (0, 1):
    step(f"bank_s{s}", lambda s=s: run([KIT + "/" + bank(s)[0]] + bank(s)[1:]))
sha = {s: hashlib.sha256(open(f"{R}/bank/s{s}/bank.json", "rb").read()).hexdigest() for s in (0, 1)}
step("render", lambda: run([KIT + "/l3_argv.py", "render", "--root", R, "--bank-sha0", sha[0], "--bank-sha1", sha[1]]))
step("verify_b4", lambda: run([KIT + "/l3_argv.py", "verify", "--root", R]))
for arm in ("D3K_s0", "D9K_s0", "D3K_s1", "D9K_s1"):
    step(f"gate_{arm}", lambda arm=arm: 0 if os.path.isfile(f"{R}/runs/{arm}/gate/gate_gatetok.json") else (fail(f"gate_{arm}", "no gate token (P3: mint it with the gate checker; never by hand)") or 1))
    step(f"arm_{arm}", lambda arm=arm: run([KIT + "/arm_run.py", arm]))
say("ZZ L3 CHAIN DONE ZZ")
