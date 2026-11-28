"""Sync the code the harness runs OFF the G: Drive, and PROVE what was synced.

G: cannot RUN the stack (Errno 22 mid-import), so the harness runs from a local
mirror.  A mirror is only trustworthy if you can say WHICH bytes it holds, so
this writes a per-file md5 manifest beside it.  ``pod_currency_audit`` exists
because presence proves transfer, md5 proves bytes, and an import proves loading
-- none of them proves currency.  The manifest is the currency evidence.
"""
import glob as _glob
import hashlib, io, json, os, shutil, sys, time

SRC = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DST = r"C:\Users\Admin\refcv5cmp\repo"
TREES = [
    ("stack/tanitad",     None),
    ("stack/scripts",     None),
    ("taniteval/taniteval", None),
    ("taniteval/tools",   None),
    ("taniteval/tests",   None),
]
#: top-level ``taniteval/*.py`` too. ⛔ NOT cosmetic: without them
#: ``test_estimator_closeout.py`` fails on the mirror while passing in the
#: repo -- a mirror that fails a test the repo passes is exactly the
#: "is the box running the repo?" trap, and it must be closed by making the
#: mirror faithful, never by excusing the failure.
FILES = (["stack/setup.py", "stack/pyproject.toml", "taniteval/conftest.py"]
         + ["taniteval/" + _f.replace("\\", "/").rsplit("/", 1)[-1]
            for _f in _glob.glob(os.path.join(SRC, "taniteval", "*.py"))])
SKIP_DIR = {"__pycache__", ".pytest_cache", ".git"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_tree(rel):
    s, d = os.path.join(SRC, rel.replace("/", os.sep)), os.path.join(DST, rel.replace("/", os.sep))
    man = {}
    for root, dirs, files in os.walk(s):
        dirs[:] = [x for x in dirs if x not in SKIP_DIR]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            sp = os.path.join(root, f)
            rp = os.path.relpath(sp, s)
            dp = os.path.join(d, rp)
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            for attempt in range(4):        # G: flaps; retry rather than believe a failure
                try:
                    shutil.copy2(sp, dp)
                    break
                except OSError as e:
                    if attempt == 3:
                        raise
                    time.sleep(2.0)
            man["%s/%s" % (rel, rp.replace(os.sep, "/"))] = md5(dp)
    return man


def main():
    man = {}
    for rel, _ in TREES:
        man.update(copy_tree(rel))
    for rel in FILES:
        s = os.path.join(SRC, rel.replace("/", os.sep))
        if not os.path.exists(s):
            continue
        d = os.path.join(DST, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        man[rel] = md5(d)
    # a 0-file sync means the mount flapped -- REFUSE, do not believe it
    if len(man) < 200:
        sys.exit("[sync] REFUSING: only %d files copied -- the G: mount almost "
                 "certainly flapped. Retry; do not believe this number." % len(man))
    out = os.path.join(DST, "_SYNC_MANIFEST.json")
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"src": SRC, "dst": DST, "n_files": len(man),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "files": man}, fh, indent=1)
    print("[sync] %d files -> %s" % (len(man), DST))
    for k in ("taniteval/tools/refcv3_arm.py", "taniteval/taniteval/four_families.py",
              "taniteval/taniteval/ci.py"):
        print("       %s  %s" % (man.get(k, "MISSING"), k))


main()
