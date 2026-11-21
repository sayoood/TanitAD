"""Helpers of the dry run (dev box only): a throw-away git repo that carries the stand-in trainer, and a fake box 2 folder. Nothing here opens a network connection."""
import hashlib
import json
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import egowm_check as CK  # noqa: E402
import plant_trainer as PT  # noqa: E402


def git(repo, *a):
    return subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", repo, *a], capture_output=True, encoding="utf-8", check=True).stdout.strip()


def mkrepo(d):
    os.makedirs(d, exist_ok=True)
    git(d, "init", "-q")
    files = {CK.TRAINER_REL: open(os.path.join(HERE, "plant_trainer.py"), encoding="utf-8").read(),
             CK.MODEL_REL: "# def built_speed  def ego_stack_cls  reflect_integrate  anchored_speeds\n", CK.DATA_REL: "# with_v\n", CK.WINDOWS_REL: "# self.ego\n",
             "stack/tanitad/__init__.py": "", "stack/tanitad/models/__init__.py": "", "stack/tanitad/data/__init__.py": ""}
    for rel, txt in files.items():
        p = os.path.join(d, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(txt)
    open(os.path.join(d, "export_list.txt"), "w", newline="\n").write("\n".join(files) + "\n")
    git(d, "add", "-A")
    git(d, "-c", "user.name=dry", "-c", "user.email=dry@example.invalid", "commit", "-q", "-m", "dry-run tree")
    print(git(d, "rev-parse", "HEAD"))


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def mkbox(box, py, gpu_mib):
    root = os.path.join(box, "D", "tanitad")
    for rel in ("cache_full/native_tracks.pt", "data/v9/lead_train4572.npz", "data/v9/refcv6_speed_max_v8_train.jsonl", "runs/v8f_mini_full/join_train.npz", "runs/v8f_mini_full/p10/p10_prior.pt"):
        put(os.path.join(root, *rel.split("/")), rel.encode())
    v9 = os.path.join(root, "data", "v9", "v9_labels_train.npz")
    put(v9, b"v9 labels")
    shas = {}
    for s in (0, 1):
        p = os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt")
        put(p, pickle.dumps({"model": PT.model_state(s)}))
        shas[str(s)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    for s_ in (0, 1):
        os.makedirs(os.path.join(root, "zcache", f"M2b_s{s_}", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "egowm"), exist_ok=True)
    put(os.path.join(box, "gpu_mib.txt"), str(gpu_mib).encode())
    v9md5 = hashlib.md5(open(v9, "rb").read()).hexdigest()
    w = lambda p: p.replace("/", "\\")                                                   # noqa: E731
    env = ["@echo off", f'set "PY={w(py)}"', 'set "RUNSAFE="', f'set "EGOWM_ROOT={w(root)}"', 'set "EGOWM_POLL_S=0.05"', 'set "EGOWM_FLOOR_BYTES=1000"', 'set "EGOWM_CKPT_LOADER=pickle"',
           f'set "EGOWM_FAKE_INIT_SHA_FILE={w(root)}\\init_sha.json"', f'set "EGOWM_FAKE_V9_MD5={v9md5}"', f'set "PLANT_CONTROL={w(box)}\\plant_control.json"',
           'set "EGOWM_HOME=' + w(root) + '\\egowm"', 'set "EGOWM_REPO=' + w(root) + '\\egowm\\repo"', 'set "EGOWM_ASIDE_ROOT=' + w(root) + '\\egowm"', ""]
    put(os.path.join(root, "env.cmd"), "\r\n".join(env).encode())
    put(os.path.join(box, "plant_control.json"), b"{}")
    json.dump(shas, open(os.path.join(root, "init_sha.json"), "w"))
    print(json.dumps({"root": root}))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "mkrepo":
        mkrepo(sys.argv[2])
    elif c == "mkbox":
        mkbox(sys.argv[2], sys.argv[3], sys.argv[4])
