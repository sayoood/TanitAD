#!/usr/bin/env python3
"""Compose a FULL REFe checkpoint (what planner.py loads, strict) from the base (u400) and a yaw_ft.py arm's final.pt. Pattern: eval/build_sft_ckpt.py.

Only the tensors in final.pt["state"] replace the base's (all must exist with the same shape). Meta: the base's meta, plus -- for an arm that was FED the yaw (arm B) --
  ego_yaw = "nuplan_angular_rate_z", ego_yaw_scale = 10.0, ego_yaw_slot = 4, ego_yaw_table_sha256 = <train table>
and for arm C (yaw 0 at train AND eval) NONE of those keys (a C checkpoint must not make YT's shim feed it a yaw). Both carry meta["yaw_ft"] = provenance.
GATES before the file is moved into place (a red gate deletes the .tmp and exits 3):
  K1 every tensor NOT in final.pt["state"] is torch.equal to the base's (scorer, backbone, registers, ... untouched)
  K2 R1 arms: the ONLY changed tensor is ego_enc.0.weight and ONLY its column 4 differs (columns != 4 torch.equal)
  K3 arm C at R1: the whole state_dict is torch.equal to the base (C == A0 bit for bit, MM gate)
  K4 the composed first-layer column 4 equals final.pt["col4"] exactly
    python build_yaw_ckpt.py --final <arm>/final.pt --out <model.pt> [--base u400.pt]
"""
import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC  # noqa: E402
sys.path.insert(0, str(YC.REFE_DIR))
import ckpt_io  # noqa: E402


def compose(base_sd: dict, fin: dict, base_path: str):
    """New state dict (shallow copy of the base's, trained tensors replaced) + meta. The base tensors are never mutated."""
    st = dict(base_sd["model"])
    for k, v in fin["state"].items():
        assert k in st, f"final.pt carries a key the base lacks: {k}"
        assert st[k].shape == v.shape, (k, st[k].shape, v.shape)
        st[k] = v.to(st[k].dtype).clone()
    meta = dict(base_sd.get("meta", {}) or {})
    cfgd = fin["config"]
    fed = cfgd["yaw_source"] == "table"
    if fed:
        meta.update({"ego_yaw": YC.YAW_TAG, "ego_yaw_scale": float(cfgd["yaw_scale"]), "ego_yaw_slot": YC.EGO_YAW_SLOT, "ego_yaw_table_sha256": cfgd["yaw_table_sha256"]})
    else:
        for k in ("ego_yaw", "ego_yaw_scale", "ego_yaw_slot", "ego_yaw_table_sha256"):
            meta.pop(k, None)
    meta["yaw_ft"] = {"arm": fin["arm"], "rung": fin["rung"], "seed": cfgd["args"]["seed"], "base": base_path, "base_sha256": cfgd["base_sha256"], "steps": cfgd["steps"],
                      "lr_col": cfgd["args"]["lr_col"], "lr_base": cfgd["lr_base"], "cache_manifest_sha256": cfgd["cache_manifest_sha256"], "yaw_scale_source": cfgd["yaw_scale_source"],
                      "changed_keys": fin["changed_keys"], "yaw_ft_code_sha256": cfgd["code_sha256"].get("yaw_ft.py")}
    return st, meta, fed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(YC.U400))
    a = ap.parse_args()
    base_sd = torch.load(a.base, map_location="cpu", weights_only=False)
    base = base_sd["model"]
    fin = torch.load(a.final, map_location="cpu", weights_only=False)
    st, meta, fed = compose(base_sd, fin, a.base)
    gates = {"K1_untouched_equal": all(torch.equal(st[k], base[k]) for k in st if k not in fin["state"])}
    changed = [k for k in st if not torch.equal(st[k], base[k])]
    gates["changed_tensors"] = changed
    if fin["rung"] == "R1":
        W, W0, c = st["ego_enc.0.weight"], base["ego_enc.0.weight"], YC.EGO_YAW_SLOT
        other = [i for i in range(W.shape[1]) if i != c]
        gates["K2_only_col4_changed"] = bool(set(changed) <= {"ego_enc.0.weight"} and torch.equal(W[:, other], W0[:, other]))
    gates["K3_C_equals_A0"] = bool(not changed) if (fin["rung"] == "R1" and not fed) else None
    gates["K4_col4_equal_final"] = bool(torch.equal(st["ego_enc.0.weight"][:, YC.EGO_YAW_SLOT], fin["col4"].to(st["ego_enc.0.weight"].dtype)))
    ok = all(v for k, v in gates.items() if k.startswith("K") and v is not None)
    YC.jdump(gates, str(a.out) + ".gates.json")
    print("gates", gates)
    if not ok:
        print("ZZYAWCKPT_GATE_RED")
        return 3
    n = ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": st, "meta": meta}, a.out)
    print(f"ZZYAWCKPT_OK {a.out} bytes {n} arm {fin['arm']} rung {fin['rung']} yaw_fed {fed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
