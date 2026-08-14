#!/usr/bin/env python3
"""Weights-only fp16 snapshot of a v6 checkpoint — the cheap survivable backup.

⛔ WHY THIS EXISTS. MEASURED 2026-08-14: `v6F-SW-30k/ckpt.pt` (3.53 GB) had
**NEVER** pushed to HF — zero successes across a whole day of a live push loop.
Every attempt aborted at ~1 % (23.9 MB / 3.53 GB) with a `BadRequestError` whose
message the push log **truncated to "Bad request for commit endp"**, so the loop
looked healthy: it kept reporting `pushed 6 [...]` for the small JSONs while the
one artifact that makes a pod handover survivable silently never landed.

The full error, recovered by re-running the upload with the traceback captured:

    Bad request for commit endpoint:
    Private repository storage limit reached, please upgrade your plan
    to increase your private storage limit

⚠️ **The lesson is the truncated log, not the quota.** A push loop that reports
successes while its largest artifact fails every cycle is the monitoring-echo
trap in a new costume: the summary line counted the files it *attempted*, so a
100 %-failing upload rendered as a healthy heartbeat. ⇒ **A backup is not
verified by its writer's log. Verify from the FAR SIDE** — list the repo and
check the file is there with the expected size (`verify_remote()` below).

WHAT THIS BUYS. `ckpt.pt` is `stack` (573 tensors) + `opt` + `step` + `config`.
The optimiser state is roughly two thirds of the bytes and is needed only for
bit-exact resumption. Dropping it and casting weights to fp16:

    3.53 GB  ->  0.67 GB   (5.3x smaller, MEASURED, 336 559 305 params)

That is enough for the P-battery, any eval, and `--init-from`. It is NOT enough
for `--resume auto` to continue with optimiser state — that is the deliberate
trade, and it must be stated wherever a snapshot is used as a restart point.

⚠️ `config` is carried INTO the snapshot on purpose. `load_resume` does a strict
state-dict load, so a checkpoint restores only into the same architecture; a
snapshot that travels without its config is a refused restart on the new pod.

Usage:
    python3 ckpt_fp16_snapshot.py <src ckpt.pt> <dst weights_fp16.pt>
    python3 ckpt_fp16_snapshot.py <src> <dst> --push <repo_id> <path_in_repo>
"""
import os
import sys

import torch

# The real top-level keys, MEASURED from a v6 checkpoint — do NOT guess these.
# An earlier version of this script assumed "model"/"state_dict", fell through
# to the whole checkpoint dict, and produced a 3.53 GB "snapshot" containing
# zero tensors while exiting 0. `params=0` in the output is that bug's signature.
STATE_KEY = "stack"
CARRY = ("step", "config")


def snapshot(src: str, dst: str) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if STATE_KEY not in ck:
        raise KeyError(
            f"{src}: no '{STATE_KEY}' key; found {sorted(ck)}. Refusing to "
            "guess — a wrong key silently produces an empty snapshot."
        )
    sd = ck[STATE_KEY]
    half = {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in sd.items()
    }
    n = sum(v.numel() for v in half.values() if torch.is_tensor(v))
    if n == 0:
        raise ValueError(f"{src}: snapshot holds 0 parameters — refusing to write")
    meta = {k: ck[k] for k in CARRY if k in ck}
    torch.save({"model": half, "_meta": meta, "_fp16_weights_only": True}, dst)
    return {
        "params": n,
        "src_gb": os.path.getsize(src) / 1e9,
        "dst_gb": os.path.getsize(dst) / 1e9,
        "step": meta.get("step"),
    }


def verify_remote(repo_id: str, path_in_repo: str, expect_bytes: int) -> bool:
    """Confirm from the FAR SIDE that the file landed at the expected size.

    This is the check the push loop did not have. `upload_file` returning
    without raising is not evidence; the repo listing is.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    for f in info.siblings:
        if f.rfilename == path_in_repo:
            got = f.size or 0
            ok = abs(got - expect_bytes) <= max(1024, expect_bytes // 1000)
            print(f"VERIFY {path_in_repo} remote={got}B expect={expect_bytes}B "
                  f"{'OK' if ok else 'SIZE_MISMATCH'}")
            return ok
    print(f"VERIFY {path_in_repo} ABSENT from {repo_id}")
    return False


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    st = snapshot(src, dst)
    print("SNAP_OK params=%(params)d src_gb=%(src_gb).2f dst_gb=%(dst_gb).2f "
          "step=%(step)s" % st)

    if "--push" in sys.argv:
        i = sys.argv.index("--push")
        repo_id, path_in_repo = sys.argv[i + 1], sys.argv[i + 2]
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=dst, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="model")
        return 0 if verify_remote(repo_id, path_in_repo,
                                  os.path.getsize(dst)) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
