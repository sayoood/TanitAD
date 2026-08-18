#!/bin/bash
# Pull the PUBLIC rollout-recovery arms to Thor for closed-loop + four-family work.
# Public repo => no gating stall; RR-20/RR-CTL/REF-C-junction are exactly the arms the PI
# wants compared, and REF-C's mainline weights are stranded on the stopped pod3 volume.
export HF_TOKEN=
~/venvs/tanitad-edge/bin/python - <<'EOP'
import os
from huggingface_hub import snapshot_download
p = snapshot_download('Sayood/tanitad-rollout-recovery', repo_type='model',
                      token=os.environ.get('HF_TOKEN'),
                      local_dir='/home/nvidia/models/rollout-recovery')
print('PULLED ->', p, flush=True)
for r, _, fs in os.walk(p):
    for f in fs:
        fp = os.path.join(r, f)
        if os.path.getsize(fp) > 1e6:
            print(f'  {os.path.relpath(fp, p)}  {os.path.getsize(fp)/1e6:.0f} MB', flush=True)
EOP
