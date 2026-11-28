# pull_ue.py : runs ON THE POD (staging of card PREREG_V8F_PROOF_ADD46). Pulls the two frozen 10k checkpoints from the private dataset
# and prints their sizes and sha256. The token is read in place from the pod's token file and is never printed.
# Usage: HF_HUB_DISABLE_XET=1 /workspace/venv/bin/python pull_ue.py <out dir>
import hashlib
import os
import sys

from huggingface_hub import hf_hub_download

out = sys.argv[1]
tok = open("/root/.hf_token").read().strip()
WANT = {"UE_s0/A/ckpt_last.pt": "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914",
        "UE_s1/A/ckpt_last.pt": "f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc"}
ok = True
for f, want in WANT.items():
    try:
        p = hf_hub_download(repo_id="Sayood/tanitad-v8f-dscale10k-runs", repo_type="dataset", filename=f, token=tok, local_dir=out)
    except Exception as ex:  # noqa: BLE001  (the message may not carry the token; print the class only)
        print("PULL-FAILED", f, type(ex).__name__)
        ok = False
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    print("pulled", f, os.path.getsize(p), got[:16], "SHA-OK" if got == want else "SHA-MISMATCH")
    ok = ok and got == want
print("ZZ PULL-UE DONE" if ok else "ZZ PULL-UE STOP")
