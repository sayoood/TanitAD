import sys, types
sys.path.insert(0, "C:/lgt/r7ldr/stack/tests")
from pathlib import Path
import test_g_box_overfit_near_lift as TM
d = Path("C:/lgt/r7ldr_scratch/gbo_green_vs_tip"); d.mkdir(exist_ok=True)
TM._synth_inputs(d); argv = TM._argv(d)
T = TM._load("refc_v3_train_gvt", TM.STACK / "scripts" / "refc_v3_train.py")
cap = TM.LG.run_trainer_until(T, argv, "model")
G_tip = TM._load("gbo_tip_t", Path("C:/lgt/r7ldr_scratch/gbo_tip_lf.py"))
rig = types.SimpleNamespace(d=d, argv=argv, T=T, model_t=cap["model"], G=G_tip)
try:
    TM.test_the_harness_replay_IS_the_launch_model_with_the_near_lift(rig)
    print("ZZGREEN_ON_TIP (BAD)ZZ")
except BaseException as e:
    print("ZZRED_ON_TIPZZ", type(e).__name__, str(e).replace("\n", " | ")[:300])
