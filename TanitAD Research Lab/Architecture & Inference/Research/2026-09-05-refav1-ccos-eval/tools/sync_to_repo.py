"""Copy files INTO the G: repo with retries until the mount is back, then verify by md5.

Usage: python sync_to_repo.py <deadline_minutes> <src_root> (<rel1> [<rel2> ...] | --list <file>)
  <src_root> is where the files are read from (the mirror or a local package copy);
  each <rel> is copied to <REPO>/<rel>. Directories are created as needed. With --list the
  relative paths are read one per line (paths with spaces survive).
G: goes fully down (reads AND writes, Errno 22) for minutes at a time; an exit code
from a copy on this mount is not evidence, so every file is re-hashed at the destination.
"""
import hashlib, os, shutil, sys, time

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    deadline_min = float(sys.argv[1]); src_root = sys.argv[2]; rels = sys.argv[3:]
    if rels and rels[0] == "--list":
        with open(rels[1], encoding="utf-8") as fh:
            rels = [l.strip("\r\n") for l in fh if l.strip()]
    want = {r: md5(os.path.join(src_root, r)) for r in rels}
    pending = dict(want)
    t_end = time.time() + 60 * deadline_min
    n = 0
    while pending and time.time() < t_end:
        n += 1
        for r in list(pending):
            src, dst = os.path.join(src_root, r), os.path.join(REPO, r)
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    if md5(dst) == want[r]:
                        print(f"OK (already) {r} {want[r]}", flush=True); del pending[r]; continue
                except OSError:
                    pass
                shutil.copyfile(src, dst)
                got = md5(dst)
                if got == want[r]:
                    print(f"OK   {r} {got}", flush=True); del pending[r]
                else:
                    print(f"MISMATCH {r} {got} != {want[r]}", flush=True)
            except OSError as e:
                if n % 10 == 1:
                    print(f"  retry {n} {r}: {e}", flush=True)
        if pending:
            time.sleep(20)
    print("ALL_SYNCED" if not pending else f"PENDING {sorted(pending)}", flush=True)
    sys.exit(0 if not pending else 1)


if __name__ == "__main__":
    main()
