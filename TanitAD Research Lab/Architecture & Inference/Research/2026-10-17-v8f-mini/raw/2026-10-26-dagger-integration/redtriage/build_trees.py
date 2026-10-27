"""Build redtriage/pristine (git archive of the live tip) and redtriage/dagger (the same + the 5 DAgger code files from di/tree).

  python build_trees.py <tip> [extra raw subdir ...]

Verification: every extracted file is hashed as a git blob and compared with `git ls-tree -r` of the same pathspecs (count, ids); the 5 overlay files are cmp'd against di/tree.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile

MIRROR = "C:/Users/Admin/tanitad-push"
RT = "E:/Projects/TanitAD-artifacts/redtriage"
DI_TREE = "E:/Projects/TanitAD-artifacts/di/tree"
PKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
HJ = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
RAW_DIRS = ["2026-10-25-oacc", "2026-10-25-hblend", "2026-10-25-hspec", "2026-10-25-khyp", "2026-10-25-kindx", "2026-10-25-mixo", "2026-10-25-lora", "full", "eval", "p10", "p10_yaw"]
RAW_FILES = ["eval141_kin.npz", "eval_grid_windows.json", "join_check.json", "join_eval141.npz", "label_join_report.json"]
OVERLAY = ["stack/scripts/v8f_dagger_bank.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_dagger.py", "stack/tests/test_v8f_dagger.py", "stack/tests/test_v8f_dagger_train.py"]
GIT = ["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", MIRROR]


def pathspecs(extra_raw):
    ps = ["stack", "taniteval", "Project Steering", "CLAUDE.md", f"{PKG}/code", f"{HJ}/code"]
    ps += [f"{PKG}/raw/{d}" for d in RAW_DIRS + list(extra_raw)] + [f"{PKG}/raw/{f}" for f in RAW_FILES]
    return ps


def lp(p):
    p = os.path.abspath(p)
    return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def extract(tip, ps, dst):
    shutil.rmtree(lp(dst), ignore_errors=True)
    os.makedirs(lp(dst), exist_ok=True)
    p = subprocess.Popen(GIT + ["archive", "--format=tar", tip, "--"] + ps, stdout=subprocess.PIPE)
    n = 0
    got = {}
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            tgt = os.path.join(dst, m.name)
            if m.isdir():
                os.makedirs(lp(tgt), exist_ok=True)
            elif m.isfile():
                os.makedirs(lp(os.path.dirname(tgt)), exist_ok=True)
                data = tf.extractfile(m).read()
                with open(lp(tgt), "wb") as fh:
                    fh.write(data)
                got[m.name] = blob_id(data)
                n += 1
            else:
                print("SKIPPED (not a regular file):", m.name, m.type)
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"git archive rc {rc}")
    return n, got


def expected(tip, ps):
    out = subprocess.run(GIT + ["ls-tree", "-r", "-z", tip, "--"] + ps, capture_output=True, check=True).stdout.split(b"\0")
    exp = {}
    for rec in out:
        if not rec:
            continue
        meta, path = rec.decode("utf-8").split("\t", 1)
        mode, typ, oid = meta.split()
        if typ == "blob" and mode != "160000":
            exp[path] = (oid, mode)
    return exp


def main():
    tip = sys.argv[1]
    extra = sys.argv[2:]
    ps = pathspecs(extra)
    exp = expected(tip, ps)
    print("expected blobs:", len(exp), flush=True)
    for name in ("pristine", "dagger"):
        n, got = extract(tip, ps, f"{RT}/{name}")
        miss = sorted(set(exp) - set(got))
        extra_f = sorted(set(got) - set(exp))
        bad = sorted(k for k in got if k in exp and got[k] != exp[k][0])
        syml = sorted(k for k, v in exp.items() if v[1] == "120000")
        print(f"{name}: extracted {n} files; missing {len(miss)}; unexpected {len(extra_f)}; blob-id mismatches {len(bad)}; symlink-mode entries {len(syml)}", flush=True)
        for k in (miss + extra_f + bad)[:10]:
            print("   ", k)
        if miss or extra_f or bad:
            raise SystemExit(2)
    # overlay the 5 DAgger files on dagger/
    for f in OVERLAY:
        src, dst = f"{DI_TREE}/{f}", f"{RT}/dagger/{f}"
        old = open(lp(dst), "rb").read() if os.path.exists(lp(dst)) else None
        shutil.copyfile(lp(src), lp(dst))
        same = open(lp(src), "rb").read() == open(lp(dst), "rb").read()
        print(f"overlay {f}: cmp_equal={same}  existed_at_tip={old is not None}  differs_from_tip={old is not None and old != open(lp(dst), 'rb').read()}  sha256={hashlib.sha256(open(lp(dst), 'rb').read()).hexdigest()[:16]}", flush=True)
        if not same:
            raise SystemExit(3)


main()
