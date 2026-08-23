"""STAGED, NOT RUN — mirror the nine pod4-only rescue checkpoints to HF.

⛔ DO NOT RUN THIS WITHOUT TWO THINGS:

  1. **The PI's authorisation.** These are nine checkpoints that have never been
     published. `Sayood/tanitad-archive-pod2-2026-08` already exists and already
     holds exactly this class of artifact (pod2 rescue checkpoints), which is an
     argument that this is inside the established migration precedent -- but it
     is an argument, not a decision, and publishing is the PI's call.

  2. **`pod4` must not be training.** As of 2026-08-03 it is running
     `flagship-v1arch-v2bal-30k`. This script reads, hashes and uploads ~28 GB
     FROM that pod: sustained disk and network load on a live training run. The
     standing rule ("never add load to a pod that is training") is absolute and
     no durability argument outranks it. Check first:

         ssh tanitad-pod4 'ps -eo pid,etime,cmd | grep [t]rain_flagship'

WHY THESE NINE
--------------
MEASURED 2026-08-03: no repo under `Sayood/` holds a checkpoint whose size
matches any of these. They were rescued off pod2 before its termination, and
pod4 -- itself a rented A40 -- is now their only location. Zero durable copies.

⚠️ The "no HF copy" finding is from SIZE comparison against the full `Sayood/`
file tree, NOT sha256. A mirrored checkpoint would have exactly that size, so it
is a strong negative signal, but it is not proof. This script re-checks by
sha256 before uploading and SKIPS anything that turns out to already be there --
so a wrong size-based conclusion costs a hash, not a duplicate upload.

RUN IT **ON pod4**, not through this dev box: the dev-box relay is 0.92 MB/s.
HF upload from a pod is 368-377 MB/s MEASURED.

    scp mirror_pod4_rescue_to_hf.py tanitad-pod4:/tmp/
    ssh tanitad-pod4 'cd /tmp && python3 mirror_pod4_rescue_to_hf.py --check'
    ssh tanitad-pod4 'cd /tmp && nohup python3 mirror_pod4_rescue_to_hf.py \
                      --go > /tmp/mirror.log 2>&1 &'

Logs go to /tmp = LOCAL disk. A logger writing to a failing filesystem cannot
report that the filesystem is failing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "Sayood/tanitad-archive-pod2-2026-08"
RESCUE = Path("/workspace/rescue/experiments")

#: arm -> the files worth mirroring. `ckpt.pt` is the artifact; the sidecars are
#: what make it reproducible, and they are tiny.
ARMS = [
    "flagship4b-v3enc-30k",
    "flagship4b-v3enc-expA-nodrop-2k",
    "flagship-v2corpus-30k",
    "refb-speed-30k",
    "refb-refbpatch-30k",
    "refb-phase0-30k",
    "finetune_traj",
    "ft_trial",
    "axis6-relaxed",
]
SIDECARS = ("config.json", "metrics.json", "summary.json", "train_log.jsonl")


def token() -> str:
    """Read in place. Never printed, never placed in argv."""
    for p in (Path.home() / ".hf_token", Path("/workspace/.hf_token")):
        if p.exists():
            return p.read_text().strip()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    raise SystemExit("no HF token found (~/.hf_token or $HF_TOKEN)")


def sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            n += len(b)
            h.update(b)
    return h.hexdigest(), n


def training_guard() -> list[str]:
    """Return a list of reasons NOT to run. Empty list = clear."""
    reasons = []
    try:
        import subprocess
        # Discover trainers from the live process table. NEVER `pgrep -f` --
        # it self-matches the probing command.
        out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True,
                             timeout=30).stdout.decode("utf-8", "replace")
        for line in out.splitlines():
            low = line.lower()
            if ("train" in low and "python" in low
                    and "mirror_pod4_rescue" not in low):
                reasons.append(f"a trainer appears to be running: {line.strip()[:110]}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"could not read the process table ({exc}) -- "
                       f"UNKNOWN is not a clear signal, refusing")
    return reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="hash locally + compare to HF; upload nothing")
    ap.add_argument("--go", action="store_true", help="actually upload")
    ap.add_argument("--force-despite-training", action="store_true",
                    help="override the trainer guard (you had better be sure)")
    args = ap.parse_args(argv)
    if not (args.check or args.go):
        ap.error("pass --check or --go")

    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi

    tok = token()
    api = HfApi(token=tok)

    existing: dict[str, dict] = {}
    for f in api.list_repo_tree(REPO_ID, repo_type="model", recursive=True,
                                expand=True, token=tok):
        if getattr(f, "size", None) is None:
            continue
        lfs = getattr(f, "lfs", None)
        existing[f.path] = {"size": f.size,
                            "sha256": getattr(lfs, "sha256", None) if lfs else None}

    plan = []
    for arm in ARMS:
        d = RESCUE / arm
        if not d.is_dir():
            plan.append({"arm": arm, "status": "SOURCE_MISSING"})
            continue
        for name in ("ckpt.pt",) + SIDECARS:
            src = d / name
            if not src.exists():
                continue
            dest = f"{arm}/{name}"
            rec = {"arm": arm, "src": str(src), "dest": dest,
                   "bytes": src.stat().st_size}
            if name == "ckpt.pt":
                rec["sha256"], _ = sha256(src)
            hit = existing.get(dest)
            if hit and (rec.get("sha256") is None
                        or hit["sha256"] == rec.get("sha256")):
                rec["status"] = "ALREADY_ON_HF"
            elif hit:
                rec["status"] = "DIFFERS_ON_HF"   # do not clobber; escalate
            else:
                rec["status"] = "TO_UPLOAD"
            plan.append(rec)

    print(json.dumps({"repo": REPO_ID, "plan": plan}, indent=1))
    todo = [p for p in plan if p.get("status") == "TO_UPLOAD"]
    print(json.dumps({
        "to_upload": len(todo),
        "bytes": sum(p["bytes"] for p in todo),
        "already": sum(1 for p in plan if p.get("status") == "ALREADY_ON_HF"),
        "differs": [p["dest"] for p in plan if p.get("status") == "DIFFERS_ON_HF"],
    }, indent=1))

    if args.check:
        return 0

    blockers = [] if args.force_despite_training else training_guard()
    if blockers:
        for b in blockers:
            print(f"REFUSING: {b}")
        print("pod4 must be idle. Re-run when the arm has finished.")
        return 2

    t0 = time.time()
    for p in todo:
        api.upload_file(path_or_fileobj=p["src"], path_in_repo=p["dest"],
                        repo_id=REPO_ID, repo_type="model", token=tok)
        print(json.dumps({"uploaded": p["dest"], "bytes": p["bytes"]}), flush=True)

    # Verify by re-reading the remote tree: exit code is not evidence.
    after = {f.path: getattr(getattr(f, "lfs", None), "sha256", None)
             for f in api.list_repo_tree(REPO_ID, repo_type="model",
                                         recursive=True, expand=True, token=tok)}
    bad = [p["dest"] for p in todo
           if p.get("sha256") and after.get(p["dest"]) != p["sha256"]]
    print(json.dumps({"done": True, "seconds": round(time.time() - t0, 1),
                      "sha_mismatch": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
