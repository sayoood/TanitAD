"""CPU proof 1: the head the adapter builds is byte-equal to the aux_drv entry of P7's model_final.pt (fresh torch.load of the file, not the adapter's object).
Writes proof_head_weights.json next to this script's out dir (argv[1])."""
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch  # noqa: E402

import p7_adapter as AD  # noqa: E402  (imports the instrument; asserts its module path)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
ck = torch.load(AD.P7_CKPT, map_location="cpu", weights_only=False)["aux_drv"]
res = {"ckpt": AD.P7_CKPT, "ckpt_sha256": AD.ANF.sha256_file(AD.P7_CKPT), "instrument_file": AD.ANF.__file__, "aux_head_module": AD.AUX.__file__}
for mode, path in (("P7", AD.P7_CKPT), ("ZERO", "ZERO::" + AD.P7_CKPT), ("FRESH", "FRESH::" + AD.P7_CKPT)):
    head, info = AD.load_p7_head(path)
    sd = head.state_dict()
    ck_head = {k[len("head."):]: v for k, v in ck.items() if k.startswith("head.")}
    non_head = [k for k in ck if not k.startswith("head.")]
    r = {"n_ckpt_tensors": len(ck), "n_ckpt_non_head": len(non_head), "n_head_state_tensors": len(sd), "keys_equal": set(sd) == set(ck_head), "n_params": info["n_params"]}
    eq = []
    for k in sorted(sd):
        a, b = sd[k].detach().cpu().contiguous(), ck_head[k].contiguous()
        same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        bytes_eq = hashlib.sha256(a.numpy().tobytes()).hexdigest() == hashlib.sha256(b.numpy().tobytes()).hexdigest()
        eq.append((k, bool(same), bool(bytes_eq), float(a.abs().sum())))
    r["n_tensors_equal"] = int(sum(1 for e in eq if e[1])); r["n_tensors_bytes_equal"] = int(sum(1 for e in eq if e[2]))
    r["abs_sum_total"] = float(sum(e[3] for e in eq))
    res[mode] = r
res["P7_byte_equal"] = bool(res["P7"]["keys_equal"] and res["P7"]["n_tensors_equal"] == res["P7"]["n_head_state_tensors"] == res["P7"]["n_tensors_bytes_equal"] and res["P7"]["n_ckpt_non_head"] == 0)
res["ZERO_all_zero"] = bool(res["ZERO"]["abs_sum_total"] == 0.0)
res["FRESH_differs_from_P7"] = bool(res["FRESH"]["n_tensors_equal"] < res["FRESH"]["n_head_state_tensors"])
json.dump(res, open(os.path.join(out, "proof_head_weights.json"), "w", encoding="utf-8"), indent=1, sort_keys=True)
print("PROOF_HEAD", json.dumps({k: res[k] for k in ("P7_byte_equal", "ZERO_all_zero", "FRESH_differs_from_P7")}), flush=True)
sys.exit(0 if (res["P7_byte_equal"] and res["ZERO_all_zero"]) else 3)
