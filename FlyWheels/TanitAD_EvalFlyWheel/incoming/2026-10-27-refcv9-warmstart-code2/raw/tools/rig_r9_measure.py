import sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "tests")
import torch
import test_map_only_ft as R
import refc_v3_train as T
from tanitad.train import map_only_ft as M
from tanitad.train import weight_ema as W
torch.set_num_threads(4)
tmp = Path(tempfile.mkdtemp(dir="F:/c_overflow_2026-10-26/tmp"))
model, cfg = R._build()
src_path = tmp / "src.pt"; R._save_source(model, src_path)
args = R._args(r8_train_only_map=False, r9_train_only="planner,map", init_from=str(src_path))
rec = M.apply_train_only(model, ("planner", "map"))
mods = tuple(rec["modules"])
opt = T.build_optimizer(model, args)
print("assert recipe:", T.assert_optimizer_matches_recipe(opt, args, model=model))
src, srec = M.load_source(str(src_path))
print("verify", M.verify_against_source(model, src, fresh=True, modules=mods))
ema = W.WeightEMA(model, 0.9, exclude=M.ema_exclude_for(mods))
print("ema problems", M.ema_problems(ema, model, src, fresh=True, modules=mods))
guard = M.FrozenStateGuard(src, modules=mods)
before = R._clone_state(model)
keys = R._steps(model, cfg, opt, ema, n=2)
print("grad keys", keys)
after = model.state_dict()
moved = [k for k in before if not M.bit_equal(before[k], after[k])]
print("moved outside", [k for k in moved if not M.is_trainable_under(k, mods)])
print("moved dec", len([k for k in moved if k.startswith("core.decoder.")]), "of", len([k for k in before if k.startswith("core.decoder.")]))
print("dec not moved", [k for k in before if k.startswith("core.decoder.") and k not in moved])
print("moved map", len([k for k in moved if M.is_map_only(k)]))
print("guard", guard.changed(model))
print("grad None outside", all(p.grad is None for n, p in model.named_parameters() if not M.is_trainable_under(n, mods)))
# mutation: perception tensor trainable
model2, cfg2 = R._build()
M.apply_train_only(model2, ("planner", "map"))
opt2 = T.build_optimizer(model2, args)
ema2 = W.WeightEMA(model2, 0.9, exclude=M.ema_exclude_for(mods))
g2 = M.FrozenStateGuard(src, modules=mods)
p = dict(model2.named_parameters())["core.encoder.net.conv1.weight"]
p.requires_grad_(True); opt2.param_groups[0]["params"].append(p)
print("opt problems", M.optimizer_problems(opt2, model2, args))
print("dvb r9", [str(m) for m in M.dvb_check_r9(model2, args)])
b2 = R._clone_state(model2)
R._steps(model2, cfg2, opt2, ema2, n=2)
print("guard mut", g2.changed(model2))
try:
    g2.assert_ok(model2, "step 2")
except SystemExit as e:
    print("RAISED", str(e)[:300])
