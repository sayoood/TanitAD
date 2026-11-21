"""Helpers of the STOPPERC dry run (dev box only): a throw-away git repo that carries the stand-in trainer and the REAL W script blob, a fake box 2 folder, a fake BAL-NULL_s0 checkpoint. Nothing here opens a network connection.
    dry_helpers.py mkrepo <dir> [good|badw]   -> prints the 40-hex commit.  `badw` commits a W script whose sha256 is NOT the pinned one.
    dry_helpers.py mkbox <dir> <python> <gpu MiB>
    dry_helpers.py mkbal <path> equal|differs -> a torch checkpoint whose model tensors equal / differ from NULL_s0's; prints its sha256"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import stopperc_check as SC  # noqa: E402
import plant_trainer as PT  # noqa: E402

CK = SC.B
MIRROR = "C:/Users/Admin/tanitad-push"
WREL = "stack/scripts/stop_aux_w_from_smoke.py"


def git(repo, *a, text=True):
    r = subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", repo, *a], capture_output=True, check=True)
    return r.stdout.decode("utf-8").strip() if text else r.stdout


def wscript_blob():
    """the W script at the code of record, raw bytes, from the read-only mirror (a read: no write to the mirror)"""
    b = git(MIRROR, "show", f"{SC.COMMIT}:{WREL}", text=False)
    assert hashlib.sha256(b).hexdigest() == "064baba6e7385ee1df475e9e3206eb5cf74360bf6b3c8d3637823ba27bd4de63", "the mirror's W script is not the pinned blob"
    return b


def mkrepo(d, variant="good"):
    os.makedirs(d, exist_ok=True)
    git(d, "init", "-q")
    files = {CK.TRAINER_REL: open(os.path.join(HERE, "plant_trainer.py"), "rb").read(),
             CK.MODEL_REL: ("# " + "  ".join(CK.NEEDLES[CK.MODEL_REL]) + "\n").encode(), CK.DATA_REL: ("# " + "  ".join(CK.NEEDLES[CK.DATA_REL]) + "\n").encode(),
             "stack/tanitad/__init__.py": b"", "stack/tanitad/models/__init__.py": b"", "stack/tanitad/data/__init__.py": b""}
    w = wscript_blob()
    allf = dict(files)
    allf[WREL] = w + (b"\n# tampered\n" if variant == "badw" else b"")
    for rel, data in allf.items():
        p = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)
    open(os.path.join(d, "export_list.txt"), "w", newline="\n").write("\n".join(files) + "\n")
    git(d, "add", "-A")
    git(d, "-c", "user.name=dry", "-c", "user.email=dry@example.invalid", "commit", "-q", "-m", "dry-run tree")
    print(git(d, "rev-parse", "HEAD"))


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def tsave(obj, p):
    import torch
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save(obj, p)


def mkbox(box, py, gpu_mib):
    BS = chr(92)
    root = os.path.join(box, "D", "tanitad")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    v9 = os.path.join(root, "data", "v9", "v9_labels_train.npz")
    put(v9, b"v9 labels")
    shas = {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        tsave({"model": PT.to_torch(PT.model_state(s))}, p)
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    bshas = {}
    for s in (0, 1):
        bp = os.path.join(root, "runs", "v8f_mini_full", "dagger_bank", f"s{s}", "bank.json")
        put(bp, f"bank {s}".encode())
        bshas[str(s)] = hashlib.sha256(open(bp, "rb").read()).hexdigest()
    os.makedirs(os.path.join(root, "stopperc"), exist_ok=True)
    put(os.path.join(box, "gpu_mib.txt"), str(gpu_mib).encode())
    v9md5 = hashlib.md5(open(v9, "rb").read()).hexdigest()
    w = lambda p: p.replace("/", "\\")                                                   # noqa: E731
    env = ["@echo off", f'set "PY={w(py)}"', 'set "RUNSAFE="', f'set "STOPBAL_ROOT={w(root)}"', 'set "STOPBAL_POLL_S=0.05"', 'set "STOPBAL_FLOOR_BYTES=1000"', 'set "CUDA_VISIBLE_DEVICES=-1"',
           'set "STOPPERC_SKIP_EVALBUILD=1"',                                            # the real model code is not in the throw-away tree: eval_build_problems is NOT covered by this dry run
           f'set "STOPBAL_FAKE_INIT_SHA_FILE={w(root)}{BS}init_sha.json"', f'set "STOPBAL_FAKE_V9_MD5={v9md5}"', f'set "STOPBAL_FAKE_BANK_SHA_FILE={w(root)}{BS}bank_sha.json"', f'set "PLANT_CONTROL={w(box)}{BS}plant_control.json"', ""]
    put(os.path.join(root, "env.cmd"), "\r\n".join(env).encode())
    put(os.path.join(box, "plant_control.json"), b"{}")
    json.dump(shas, open(os.path.join(root, "init_sha.json"), "w"))
    json.dump(bshas, open(os.path.join(root, "bank_sha.json"), "w"))
    print(json.dumps({"root": root}))


def mkbal(path, how):
    st = PT.trained_state(0, 0)                                                          # = what NULL_s0 (init 0, seed 0) leaves
    if how == "differs":
        st["head_o.w"] = st["head_o.w"] + 1e-3
    tsave({"model": PT.to_torch(st), "argv_sha": "balnullbalnullba", "note": "dry-run fake BAL-NULL_s0 (never the real 5169ea95 checkpoint)"}, path)
    print(hashlib.sha256(open(path, "rb").read()).hexdigest())


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "mkrepo":
        mkrepo(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "good")
    elif c == "mkbox":
        mkbox(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == "mkbal":
        mkbal(sys.argv[2], sys.argv[3])
