"""Copy a local bank tree onto the (flapping) G: mount with retries + sha256 read-back.

usage: python bank_copy.py <src_root> <dst_root>
Every file under src_root is copied to the same relative path under dst_root; each copy
is verified by re-reading the destination and comparing sha256 (never by presence).
Prints one line per file and a final manifest; exits non-zero if any file is unverified.
"""
import hashlib
import os
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def retry(fn, what, n=12, wait=8.0):
    last = None
    for i in range(n):
        try:
            return fn()
        except OSError as ex:                      # errno 22 / Invalid request code / EEXIST
            last = ex
            print(f"    transient on {what}: {ex} (retry {i + 1}/{n})", flush=True)
            time.sleep(wait)
    raise SystemExit(f"GAVE UP on {what}: {last}")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows, bad = [], []
    for root, _dirs, files in os.walk(src):
        for f in sorted(files):
            sp = os.path.join(root, f)
            rel = os.path.relpath(sp, src)
            dp = os.path.join(dst, rel)
            want = sha(sp)
            retry(lambda: os.makedirs(os.path.dirname(dp), exist_ok=True), f"mkdir {rel}")

            def _copy():
                with open(sp, "rb") as a:
                    data = a.read()
                with open(dp, "wb") as b:
                    b.write(data)
                return True
            ok = False
            for k in range(6):
                retry(_copy, f"copy {rel}")
                got = retry(lambda: sha(dp), f"readback {rel}")
                if got == want:
                    ok = True
                    break
                print(f"    sha mismatch on {rel} (pass {k + 1}); recopying", flush=True)
                time.sleep(5)
            rows.append((rel, os.path.getsize(sp), want, ok))
            if not ok:
                bad.append(rel)
            print(f"  {'OK ' if ok else 'BAD'} {rel} {os.path.getsize(sp)} B sha256 {want[:16]}", flush=True)
    print(f"\n{len(rows)} files, {len(bad)} unverified: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
