"""Fail-closed input check for the ACTSENS read. usage: check_tree.py   (env from env.sh)
Asserts: tanitad comes from the ACTSENS tree; V8FMiniConfig has wm_gamma_levels; code-of-record shas (contrast_l1full 'b7661635', decode_check '027da9ee'); R35 reference files exist; the HF tar members' sha256 equal HF_BANK per_file_sha256."""
import hashlib, os, sys, json
tree = os.environ["ACT_TREE"]
import tanitad
from tanitad.models import v8f_mini as M
f = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
assert f.startswith(os.path.abspath(tree).replace("\\", "/").lower()), tanitad.__file__
import dataclasses
assert "wm_gamma_levels" in {x.name for x in dataclasses.fields(M.V8FMiniConfig)}, "tree lacks wm_gamma_levels: not the ACTSENS commit"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for p, want in (("F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py", "b7661635"), ("F:/TanitAD-scratch/hwm_plan/decode_check/decode_check.py", "027da9ee")):
    assert sha(p).startswith(want), (p, sha(p))
ref = os.environ["R35REF"]
bank = json.load(open(os.environ["RD"] + "/hf_bank_phase1.json"))
for rel, want in bank["per_file_sha256"].items():
    p = f"{ref}/{rel}".replace("/s0/", "/s0/").replace("/s1/", "/s1/")
    assert os.path.exists(p), p
    assert sha(p) == want, ("R35 reference differs from the HF bank", p)
print("CHECK_TREE_OK", tanitad.__file__, "ref files", len(bank["per_file_sha256"]))
