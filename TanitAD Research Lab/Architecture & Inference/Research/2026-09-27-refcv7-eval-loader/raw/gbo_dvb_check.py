"""Does G-DVB refuse a map-hires branch built WITHOUT the R2/R3 fields (g_box_overfit's replay)?"""
import dataclasses, sys, types, os
sys.path.insert(0, "C:/lgt/r7ldr/stack"); sys.path.insert(0, "C:/lgt/r7ldr/stack/scripts"); sys.path.insert(0, "C:/lgt/r7ldr/stack/tests")
os.environ["REFCV6_REPO"] = "C:/lgt/r7ldr"
import torch
import test_refcv7_eval_loader as TM
from pathlib import Path
d = Path("C:/lgt/r7ldr_scratch/gbo_dvb"); d.mkdir(exist_ok=True)
TM._synth_inputs(d); argv = TM._argv(d)
T = TM._load_module("refc_v3_train_gbo_ref", TM.STACK / "scripts" / "refc_v3_train.py")
cap = TM.LG.run_trainer_until(T, TM.LG.set_flag(argv, "--trunk-compile", None), "model")
ck = d / "ckpt.pt"; torch.save({"model": cap["model"].state_dict(), "step": 0}, ck)
L = TM._load_module("refcv7_loader_gbo", TM.LOADER_PATH)
orig = L.map_hires_config
L.map_hires_config = lambda tr, args, sha: dataclasses.replace(orig(tr, args, sha), near_lift_x_m=0.0, near_refine_blocks=0)
try:
    L.build_model(TM._record(T, argv), str(ck), device="cpu", strict=False, remap={})
    print("ZZNOT_REFUSEDZZ")
except SystemExit as e:
    print("ZZREFUSEDZZ", str(e).replace("\n", " | ")[:700])
