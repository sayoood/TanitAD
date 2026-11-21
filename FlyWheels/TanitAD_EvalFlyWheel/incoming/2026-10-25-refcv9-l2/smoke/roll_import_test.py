import sys, json, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
cal = "C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/calibrator_30k_DEV.json"
sys.argv = ["l2_roll.py", "--arm", "cal", "--calibrator", cal, "--trace-out", "C:/Users/Admin/refcv8_pf/refcv9_l2/smoke/trace_test.npz", "--require-seal", "--",
            "--ckpt", "x", "--config", "y", "--seed", "0", "--dump-dir", "z", "--windows-json", "w", "--device", "cpu", "--out-json", "o", "--os-only"]
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_roll as R
import torch
print("imported; seals", {k: (v[:12] if v else v) for k, v in R.SEALS.items()})
lg = torch.randn(1, 8)
R.A.arm = "cal"; out = R._arm_logits(lg, (0, 5))
b8 = torch.tensor(R.CAL["b8"]); T = R.CAL["T"]
print("cal ok", bool(torch.allclose(out, (lg + b8) / T)), "T", T, "b8", [round(x, 3) for x in R.CAL["b8"]])
R.A.arm = "ident"; print("ident bit-equal", bool(torch.equal(R._arm_logits(lg, (0, 5)), lg)))
R.A.arm = "const_cruise"; o = R._arm_logits(lg, (0, 5)); print("const argmax", int(o.argmax()), "other", float(o[0, 0]))
R.A.arm = "oracle"; R.LABELS = {(0, 5): 3, (0, 6): 2}
print("oracle labelled col", int(R._arm_logits(lg, (0, 5)).argmax()), "masked -> None:", R._arm_logits(lg, (0, 6)) is None, "unlabelled -> None:", R._arm_logits(lg, (9, 9)) is None)
print("counters", R.ST["c"])
