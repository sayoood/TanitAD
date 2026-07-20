"""Strict-parity skipset verdict for the pod1 REF-B train build.

Authority is the coordinator's deployed /workspace/parity_skipset.sh (canonical
serialization: ",".join(sorted(skip_clip_ids)) -> sha256). This helper also does
a self-contained reconstruction using the IDENTICAL method as a backstop (so the
gate still computes the canonical hash if the script sub-invocation fails), and
cross-checks the actual built episode count.

Exit 0 iff: (canonical script hash OR backstop hash) == EXPECT_HASH, and
skips == 24, and built ep_*.pt == 2376. Else exit 3 -> PARITY_HOLD.
"""
import glob
import hashlib
import os
import re
import subprocess
import sys

import pandas as pd
import torch

ROOT = os.environ.get("TANITAD_PHYSICALAI_ROOT",
                      "/workspace/data/physicalai_phase0")
KEY = "physicalai-train-e438721ae894"
EXPECT_HASH = ("f09e44db000407bb472f19201dc673238cb098db"
               "1746a21282c8935a79e85457")
EXPECT_USABLE = 2376
EXPECT_SKIPS = 24
SCRIPT = "/workspace/parity_skipset.sh"


def backstop():
    """Mirror parity_skipset.sh EXACTLY: sorted(sel ids) -> randperm(seed0) ->
    train -> skip_ positions -> comma-join(sorted) -> sha256."""
    sel = pd.read_parquet(f"{ROOT}/r0/r0_selection.parquet")
    clip_ids = sorted(sel["clip_id"].astype(str).tolist())
    if len(clip_ids) != 3000:
        return "", -1, []
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(clip_ids), generator=g).tolist()
    n_val = max(1, int(len(clip_ids) * 0.2))
    val_i = set(perm[:n_val])
    train = [c for i, c in enumerate(clip_ids) if i not in val_i]
    tdirs = sorted(glob.glob(f"{ROOT}/_epcache/physicalai-train-*"))
    if not tdirs:
        return "", -1, []
    tdir = tdirs[-1]
    idx = sorted(int(os.path.basename(p)[5:10])
                 for p in glob.glob(f"{tdir}/skip_*"))
    skip_clipids = [train[i] for i in idx]
    ids = ",".join(sorted(skip_clipids))
    return hashlib.sha256(ids.encode()).hexdigest(), len(idx), sorted(skip_clipids)


def run_script():
    if not os.path.exists(SCRIPT):
        return "", -1, -1, ""
    try:
        r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True,
                           timeout=300)
        out = r.stdout + r.stderr
    except Exception as e:
        return "", -1, -1, f"script error: {e}"
    mh = re.search(r"PARITY_HASH[^=]*=\s*([0-9a-f]{64})", out)
    ms = re.search(r"skips=(\d+)", out)
    mu = re.search(r"usable_train=(\d+)", out)
    return (mh.group(1) if mh else "",
            int(ms.group(1)) if ms else -1,
            int(mu.group(1)) if mu else -1, out)


def main():
    n_built = len(glob.glob(f"{ROOT}/_epcache/{KEY}/ep_*.pt"))
    s_hash, s_skips, s_usable, s_out = run_script()
    b_hash, b_skips, b_ids = backstop()

    print(f"BUILT {n_built}")
    print(f"SCRIPT_HASH {s_hash or 'NA'} script_skips={s_skips} "
          f"script_usable={s_usable}")
    print(f"BACKSTOP_HASH {b_hash or 'NA'} backstop_skips={b_skips}")
    print(f"EXPECT_HASH {EXPECT_HASH}")
    if b_ids:
        print("SKIP_IDS " + ",".join(b_ids))

    skips = s_skips if s_skips >= 0 else b_skips
    hash_ok = (s_hash == EXPECT_HASH) or (b_hash == EXPECT_HASH)
    counts_ok = (n_built == EXPECT_USABLE and skips == EXPECT_SKIPS)
    verdict = "MATCH" if (hash_ok and counts_ok) else "MISMATCH"
    print(f"VERDICT {verdict} hash_ok={hash_ok} counts_ok={counts_ok} "
          f"built={n_built} skips={skips}")
    sys.exit(0 if verdict == "MATCH" else 3)


if __name__ == "__main__":
    main()
