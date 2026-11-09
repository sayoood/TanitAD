"""Beside PREREG_V8F_PLANTEST3 (Master Mind, 2026-11-09): do the ladder world models see anything M2b's world model does not? Tensor-by-tensor comparison of each finished ladder checkpoint with its M2b initialiser. CPU, no eval window read.
z1 = wm.proj(adapter tokens, mean-pooled), so if every adapter tensor is equal, a ladder z1 is a learned re-projection of the same 256-D pooled token and its readable information is bounded by that token's.
Usage: python compare_ladder_adapter.py   (writes COMPARE_LADDER_ADAPTER.json next to this file)"""
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts"
M2B = {"s0": f"{ART}/v8f-mini-hacc/M2b_s0_hacc.pt", "s1": f"{ART}/v8f-mini-hacc/M2b_s1_hacc.pt"}
PULLS = "C:/Users/Admin/t2v8f_pulls"


def main():
    out = {}
    for p in sorted(glob.glob(f"{PULLS}/ladder_*/ckpt_last.pt")):
        tag = os.path.basename(os.path.dirname(p))[len("ladder_"):]
        seed = tag.rsplit("_", 1)[1]
        m = torch.load(M2B[seed], map_location="cpu", weights_only=False)["model"]
        c = torch.load(p, map_location="cpu", weights_only=False)
        d = c["model"]
        adapter = [k for k in d if k.startswith("adapter.") or k == "pos"]
        proj = [k for k in d if k.startswith("wm.proj")]
        assert adapter and proj
        out[tag] = {"adapter_tensors": len(adapter), "adapter_equal": int(sum(torch.equal(m[k], d[k]) for k in adapter)),
                    "proj_tensors": len(proj), "proj_equal": int(sum(torch.equal(m[k], d[k]) for k in proj)),
                    "wm_strides": list(c["cfg"]["wm_strides"]), "hmax": c["cfg"]["hmax"]}
    json.dump(out, open(os.path.join(HERE, "COMPARE_LADDER_ADAPTER.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
