"""Pull the PUBLIC rollout-recovery arms to Thor. NO TOKEN — the repo is public, and an
empty Bearer header is worse than none (it raises LocalProtocolError, which is how the
first attempt failed)."""
import os
from huggingface_hub import snapshot_download
p = snapshot_download("Sayood/tanitad-rollout-recovery", repo_type="model",
                      local_dir="/home/nvidia/models/rollout-recovery")
print("PULLED ->", p, flush=True)
for r, _, fs in os.walk(p):
    for f in fs:
        fp = os.path.join(r, f)
        if os.path.getsize(fp) > 1e6:
            print("  %s  %.0f MB" % (os.path.relpath(fp, p), os.path.getsize(fp)/1e6), flush=True)
