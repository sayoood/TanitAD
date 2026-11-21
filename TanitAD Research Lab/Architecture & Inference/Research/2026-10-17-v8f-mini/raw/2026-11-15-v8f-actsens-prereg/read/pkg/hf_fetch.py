"""Fetch the phase-1 plans tar of R35 (private HF dataset) and verify its sha256. Token is read in place by publish_context_labels.authenticated() and never printed."""
import os, sys, hashlib, tarfile, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
import importlib.util
HERE = "F:/TanitAD-scratch/actsens/read"
spec = importlib.util.spec_from_file_location("pcl", HERE + "/publish_context_labels.py")
pcl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pcl)
api, tok, who = pcl.authenticated()
from huggingface_hub import hf_hub_download
REV = "aa2d801477f429be909c6b6dedb5c602eb2bc0b7"
WANT = "abff3a388ad8b22aa15b7ad9a26da828692cfe5f752c5bfe4b95787cda12cc4a"
p = hf_hub_download("Sayood/tanitad-v8f-plantest3-runs", "hwmplan/hwmplan_phase1_R35.tar", repo_type="dataset", revision=REV, token=tok, local_dir=HERE + "/hf_phase1/dl")
h = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
print(p, h.hexdigest(), "OK" if h.hexdigest() == WANT else "MISMATCH")
assert h.hexdigest() == WANT
with tarfile.open(p) as t:
    t.extractall(HERE + "/hf_phase1/R35")
print(sorted(os.listdir(HERE + "/hf_phase1/R35")))
