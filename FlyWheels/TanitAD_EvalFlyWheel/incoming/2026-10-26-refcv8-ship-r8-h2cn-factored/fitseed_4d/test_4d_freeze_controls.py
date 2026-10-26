"""test_4d_freeze_controls.py -- AMENDMENT 4d: the new freeze comparators can see a difference (deliberate-regression arms).  CPU, seconds, no fit, no eval-side data.
  * the registered stage vs itself  -> every tensor bit-equal (identity must read GREEN)
  * one weight perturbed by 1 ulp   -> tensors_equal must read RED and name exactly that tensor (the control can see a single-ulp difference)
  * the K-SEED-MOVES criterion on a copy with an unchanged state_dict but a changed 'st' stat -> NOT a parameter difference (must stay RED for that purpose)
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h2cn_freeze as H  # noqa: E402
import ship_r8_sub_common as C  # noqa: E402


def main() -> int:
    blob = torch.load(H.REG_STAGE_PT, map_location="cpu", weights_only=True)
    same = H.tensors_equal(blob, copy.deepcopy(blob))
    assert all(v["bit_equal"] and v["n_differ"] == 0 and v["n_tensors"] > 0 for v in same.values()), same
    b2 = copy.deepcopy(blob)
    w = b2["surfaces"]["S6"]["state_dict"]["2.weight"]
    flat = w.view(-1)
    flat[3] = torch.nextafter(flat[3], torch.tensor(float("inf"), dtype=flat.dtype))
    pert = H.tensors_equal(blob, b2)
    assert pert["S6"]["differ"] == ["state_dict.2.weight"] and not pert["S6"]["bit_equal"] and pert["S2"]["bit_equal"], pert
    b3 = copy.deepcopy(blob)
    b3["surfaces"]["S2"]["st"]["xmu"] = b3["surfaces"]["S2"]["st"]["xmu"] + 1e-3
    st_only = H.tensors_equal(blob, b3)
    assert st_only["S2"]["differ"] == ["st.xmu"] and not any(d.startswith("state_dict.") for d in st_only["S2"]["differ"]), st_only
    print("ZZ4DFREEZECONTROLSOKZZ", {"identity_n_tensors": {s: v["n_tensors"] for s, v in same.items()}, "ulp_perturbation_named": pert["S6"]["differ"],
                                      "st_only_named": st_only["S2"]["differ"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
