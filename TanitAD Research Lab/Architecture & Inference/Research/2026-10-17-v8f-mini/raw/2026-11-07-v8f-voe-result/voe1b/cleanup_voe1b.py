"""PREREG_V8F_VOE1B cleanup (CLAUDE.md 'delete what is not needed'): deletes this package's OWN re-creatable temp data by literal absolute path and logs each deletion
(path, size) to CLEANUP_VOE1B.log. Runs only if VOE1B_RUN.done reads rc=0 and RESULT_VOE1B.json and RESULT_VOE1B.md exist (the run and its report are banked in this directory).
Nothing here is the only copy: the bank is rebuilt by build_bank_voe1b.py (about 2 minutes); the smoke directory is a plumbing test of the same scripts on 30 episodes."""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "CLEANUP_VOE1B.log")
BANK_FILE = "F:/TanitAD-scratch/voe1b_bank/bank_voe1b.pt"
BANK_DIR = "F:/TanitAD-scratch/voe1b_bank"
SMOKE_DIR = "F:/TanitAD-scratch/voe1b_smoke"
PYCACHE = os.path.join(HERE, "__pycache__").replace("\\", "/")


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def main():
    lines = [f"cleanup started {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    done = os.path.join(HERE, "VOE1B_RUN.done")
    rj, rm = os.path.join(HERE, "RESULT_VOE1B.json"), os.path.join(HERE, "RESULT_VOE1B.md")
    ok = os.path.exists(done) and open(done).read().strip() == "rc=0" and os.path.exists(rj) and os.path.getsize(rj) > 0 and os.path.exists(rm)
    if not ok:
        lines.append("preconditions not met (run rc != 0, or RESULT_VOE1B.json / RESULT_VOE1B.md missing): nothing deleted")
    else:
        for p in (BANK_FILE, BANK_DIR, SMOKE_DIR, PYCACHE):
            allowed = p.startswith("F:/TanitAD-scratch/voe1b_") or p == PYCACHE
            if not allowed:
                lines.append(f"REFUSED (not a literal voe1b path): {p}")
                continue
            if not os.path.exists(p):
                lines.append(f"absent: {p}")
                continue
            sz = size_of(p)
            if os.path.isdir(p):
                if p == BANK_DIR and os.listdir(p):
                    lines.append(f"kept (not empty): {p}")
                    continue
                shutil.rmtree(p)
            else:
                os.remove(p)
            lines.append(f"deleted {p} ({sz:,} bytes)")
    open(LOG, "a", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
    sys.exit(0)
