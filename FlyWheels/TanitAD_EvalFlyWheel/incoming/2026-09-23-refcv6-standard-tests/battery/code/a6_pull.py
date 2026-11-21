"""SPEC A6 step 2: READ-ONLY pull of the 139 selected train clips (scp) with md5 checked against Thor.

usage: python a6_pull.py   -> D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6/<clip>.v2ep.pt
Refuses (exit 2) unless every local md5 equals `ssh -n md5sum` on Thor. Resumable: a file already
present with the right md5 is not pulled again. Writes raw/a6/pull_record.json (sha12 only).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = "tanitad-thor-wifi"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", H]
TRAIN_DIR = "/home/nvidia/data/refcv6-b1-416x1024-train"
DST = Path("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6")
OUT = HERE.parent / "raw" / "a6"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    clips = json.load(open(OUT / "train_clips.PRIVATE.json", encoding="utf-8"))
    remote = {}
    for i in range(0, len(clips), 25):
        chunk = clips[i:i + 25]
        cmd = "md5sum " + " ".join(f"{TRAIN_DIR}/{c}.v2ep.pt" for c in chunk)
        r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=900)
        for line in r.stdout.splitlines():
            m, path = line.split(None, 1)
            remote[Path(path.strip()).name[: -len(".v2ep.pt")]] = m
    missing = [c for c in clips if c not in remote]
    if missing:
        raise SystemExit(f"[a6] {len(missing)} selected clips have no remote md5 -- refusing")
    t0, n_pulled, bad, total = time.time(), 0, [], 0
    for c in clips:
        loc = DST / f"{c}.v2ep.pt"
        if loc.exists() and md5(loc) == remote[c]:
            total += loc.stat().st_size
            continue
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{H}:{TRAIN_DIR}/{c}.v2ep.pt", str(loc)],
                       check=False, timeout=1800)
        n_pulled += 1
        if not loc.exists() or md5(loc) != remote[c]:
            bad.append(hashlib.sha256(c.encode()).hexdigest()[:12])
        else:
            total += loc.stat().st_size
    rec = {"tool": "a6_pull.py", "n_selected": len(clips), "n_pulled_now": n_pulled, "n_md5_bad": len(bad),
           "bad_sha12": bad, "bytes": total, "wall_s": round(time.time() - t0, 1), "dst": str(DST),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "verdict": "PASS" if not bad else "FAIL"}
    json.dump(rec, open(OUT / "pull_record.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec))
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
