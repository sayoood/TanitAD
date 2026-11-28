"""s0p_chain_tool.py -- the small python helpers of the S0-P waiter chain (s0p_chain.sh).  No network, no box access, standalone (numpy only).  Python run with -I.

  mkjob   --mode dry|shard --gid G [--i I --k K] --out FILE     the box-2 job .cmd (CRLF; every backslash comes from chr(92): bash printf mangles \\b / \\U)
  mklaunch --chain SH --console LOG --ssh P --sftp P --out FILE                  the WMI-detached start line (not started here)
  check-dry --log FILE [--n-pos N --n-neg N]                    the box-2 --dry-run log must show the banked pool, disjoint clips, the sample counts; prints the sample sha256
  verify-shard --dir D --i I --k K --sample-sha S [--n-total N] a pulled shard dir: MANIFEST.sha256 (every file's sha256), meta.json (shard i/K, bounds, sample sha), array shapes / dtypes
"""
import argparse
import hashlib
import json
import os
import re
import sys

BS = chr(92)
SHARD_FILES = ("E.npy", "v40.npy", "v160.npy", "mean.npy", "y.npy", "clip.npy", "ws.npy", "eid.npy", "meta.json", "sample.json")
B2 = {"root": "D:" + BS + "tanitad", "py": "D:" + BS + "tanitad" + BS + "venv" + BS + "Scripts" + BS + "python.exe"}
SD = B2["root"] + BS + "stopperc" + BS + "s0p"            # shipped scripts
OUT = B2["root"] + BS + "stopperc" + BS + "s0p_out"       # --out-dir (shard_<i> below it)
MD = B2["root"] + BS + "stopperc" + BS + "s0p_jobs"       # markers
CACHE = B2["root"] + BS + "cache_full"
JOIN = B2["root"] + BS + "runs" + BS + "v8f_mini_full" + BS + "join_train.npz"
TRACKS = CACHE + BS + "native_tracks.pt"
REPO = B2["root"] + BS + "repo"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def shard_bounds(n, i, k):
    return (i * n) // k, ((i + 1) * n) // k


def mkjob(mode, gid, i, k):
    assert mode in ("dry", "shard")
    tag = "DRY" if mode == "dry" else f"SH{i}"
    args = f'--tree {REPO} --cache-dir {CACHE} --split train --join {JOIN} --tracks {TRACKS} --out-dir "{OUT}" --eval-cache-dir {CACHE}'
    args += " --dry-run" if mode == "dry" else f" --shard {i}/{k}"
    ok = f'"{MD}{BS}{tag}_{gid}.done"'
    fail = f'"{MD}{BS}{tag}_{gid}.fail"'
    L = [
        "@echo off",
        f"rem S0PJOB mode={mode} i={i if mode == 'shard' else 0} k={k if mode == 'shard' else 0} gid={gid}",
        "rem S0-P train-side extraction on box 2 (instantiated by s0p_chain.sh via s0p_chain_tool.py mkjob). CPU only; writes only under D:" + BS + "tanitad" + BS + "stopperc. No labels, no goto.",
        "setlocal enabledelayedexpansion",
        'set "OMP_NUM_THREADS=2"',
        'set "CUDA_VISIBLE_DEVICES=-1"',
        f'if not exist "{MD}" mkdir "{MD}"',
        f"echo ===== S0P-JOB {tag} {gid} %DATE% %TIME%",
        f'"{B2["py"]}" "{SD}{BS}s0p_extract_train.py" {args}',
        'set "RC=!ERRORLEVEL!"',
        "echo ===== S0P-JOB-END rc=!RC! %DATE% %TIME%",
        f"if !RC! NEQ 0 (echo rc=!RC!> {fail} & exit /b !RC!)",
    ]
    if mode == "shard":
        L.append(f'if not exist "{OUT}{BS}shard_{i}{BS}MANIFEST.sha256" (echo no-manifest> {fail} & exit /b 1)')
    L += [f"echo done> {ok}", "exit /b 0"]
    return "\r\n".join(L) + "\r\n"


def mklaunch(chain, console, ssh, sftp):
    bash = "C:" + BS + "Program Files" + BS + "Git" + BS + "bin" + BS + "bash.exe"
    w = lambda p: p.replace("/", BS)
    return "@echo off\r\n" + f'set "S0P_SSH={w(ssh)}"\r\nset "S0P_SFTP={w(sftp)}"\r\n' + f'"{bash}" -l "{w(chain)}" > "{w(console)}" 2>&1\r\n'


def check_dry(text, n_pos, n_neg):
    text = text.replace("\x00", "")
    need = [(r'"pool_equals_banked_prevalence":\s*true', "pool_equals_banked_prevalence"), (r'"overlap":\s*0\b', "disjoint overlap 0"),
            (rf'"n_pos_sample":\s*{n_pos}\b', f"n_pos_sample {n_pos}"), (rf'"n_neg_sample":\s*{n_neg}\b', f"n_neg_sample {n_neg}"), (rf'"n_sample":\s*{n_pos + n_neg}\b', f"n_sample {n_pos + n_neg}")]
    miss = [name for rx, name in need if not re.search(rx, text)]
    m = re.search(r'"sample_sha256":\s*"([0-9a-f]{64})"', text)
    if not m:
        miss.append("sample_sha256")
    if miss:
        raise SystemExit("dry-run log check failed, missing: " + ", ".join(miss))
    return m.group(1)


def verify_shard(d, i, k, sample_sha, n_total):
    import numpy as np
    man = {}
    for line in open(os.path.join(d, "MANIFEST.sha256"), encoding="utf-8"):
        if line.strip():
            h, n = line.split()
            man[n] = h
    assert set(man) == set(SHARD_FILES), f"manifest names {sorted(man)} != {sorted(SHARD_FILES)}"
    for n, h in man.items():
        assert sha256_file(os.path.join(d, n)) == h, f"{n}: sha256 differs from MANIFEST.sha256"
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    sh = meta.get("shard") or {}
    assert (sh.get("i"), sh.get("k")) == (i, k), f"meta shard {sh} != {i}/{k}"
    assert meta["sample_sha256"] == sample_sha, "meta sample_sha256 differs from the dry-run's"
    n_total = n_total or meta["n_sample"]
    lo, hi = shard_bounds(n_total, i, k)
    assert (sh["lo"], sh["hi"], sh["n_total"]) == (lo, hi, n_total) and meta["n_shard"] == hi - lo, f"meta bounds {sh} != ({lo},{hi},{n_total})"
    n = hi - lo
    want = {"E.npy": ((n, 18), "float32"), "v40.npy": ((n, 40, 1024), "uint8"), "v160.npy": ((n, 160, 1024), "uint8"), "mean.npy": ((n, 1024), "float16"), "y.npy": ((n,), "uint8"),
            "ws.npy": ((n,), "int64"), "eid.npy": ((n,), "int64"), "clip.npy": ((n,), None)}
    for name, (shape, dt) in want.items():
        a = np.load(os.path.join(d, name), mmap_mode="r")
        assert a.shape == shape and (dt is None or a.dtype.name == dt), f"{name}: {a.shape} {a.dtype} != {shape} {dt}"
    sj = json.load(open(os.path.join(d, "sample.json"), encoding="utf-8"))
    assert sj["sha256"] == sample_sha and len(sj["y"]) == n, "shard sample.json is not this shard's slice of the sample"
    assert [int(x) for x in sj["y"]] == np.load(os.path.join(d, "y.npy")).tolist(), "sample.json y != y.npy"
    return {"n": n, "lo": lo, "hi": hi, "manifest_sha256": sha256_file(os.path.join(d, "MANIFEST.sha256"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("mkjob", "mklaunch", "check-dry", "verify-shard"))
    ap.add_argument("--mode"), ap.add_argument("--gid"), ap.add_argument("--i", type=int, default=0), ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out"), ap.add_argument("--chain"), ap.add_argument("--console"), ap.add_argument("--log"), ap.add_argument("--dir"), ap.add_argument("--sample-sha"), ap.add_argument("--ssh"), ap.add_argument("--sftp")
    ap.add_argument("--n-pos", type=int, default=4000), ap.add_argument("--n-neg", type=int, default=12000), ap.add_argument("--n-total", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "mkjob":
        assert re.fullmatch(r"[0-9A-Za-z_]+", a.gid or ""), "gid"
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(mkjob(a.mode, a.gid, a.i, a.k))
    elif a.cmd == "mklaunch":
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(mklaunch(a.chain, a.console, a.ssh, a.sftp))
    elif a.cmd == "check-dry":
        print(check_dry(open(a.log, encoding="utf-8", errors="replace").read(), a.n_pos, a.n_neg))
    else:
        print(json.dumps(verify_shard(a.dir, a.i, a.k, a.sample_sha, a.n_total)))


if __name__ == "__main__":
    main()
