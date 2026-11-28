import sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "tests")
import torch
import _refcv8_rig as R8
from tanitad.train import map_only_ft as M
from tanitad.train import weight_ema as W
from tanitad.train import declared_vs_built as D
torch.set_num_threads(4)
T = R8.trainer()
RIG_ARGV = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
            "--n-anchors", "20", "--ego-history", "--residual-prior", "ha0_ext_pose", "--out", "X"] + R8.F1_F6
cfg, m = R8.build(T, True, seed=0, n_alloc=8, w_alloc_l1=1.0, w_sat=0.1)
tmp = Path(tempfile.mkdtemp(dir="F:/c_overflow_2026-10-26/tmp"))
src_path = tmp / "src.pt"
torch.save({"model": {k: v.detach().clone() for k, v in m.state_dict().items()}, "step": 48852, "kind": "ema_model_only"}, src_path)
args = T.build_parser().parse_args(RIG_ARGV + ["--r9-train-only", "planner,tactical", "--init-from", str(src_path), "--lr", "1e-3", "--opt", "dd"])
rec = M.apply_train_only(m, M.r9_sets(args))
mods, dec = tuple(rec["modules"]), tuple(rec["declared_frozen"])
print("gu", [str(x) for x in D.check_grad_unreachable(m, args)])
print("dvb_r9", [str(x) for x in M.dvb_check_r9(m, args)])
opt = T.build_optimizer(m, args)
print(T.assert_optimizer_matches_recipe(opt, args, model=m))
src, srec = M.load_source(str(src_path))
print(M.verify_against_source(m, src, fresh=True, modules=mods, declared=dec))
ema = W.WeightEMA(m, 0.9, exclude=M.ema_exclude_for(mods, dec))
print("ema", M.ema_problems(ema, m, src, fresh=True, modules=mods, declared=dec))
guard = M.FrozenStateGuard(src, modules=mods, declared=dec)
eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=120)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=60, channels=cfg.core.encoder.in_channels)
ds.ego_history = True
n = len(ds)
batch = torch.utils.data.default_collate([ds[i] for i in range(0, n, max(1, n // 3))][:3])
v7l = T.v7l
batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0]); batch["lon_v7"] = torch.tensor([0, v7l.IGNORE_ID, 0])
batch["tac_goal_y"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS)); batch["tac_goal_w"] = torch.zeros(3, len(v7l.TAC_GOAL_TOKENS))
m._w_tac_v6 = 1.0
m.train()
before = {k: v.detach().clone() for k, v in m.state_dict().items()}
for i in range(2):
    torch.manual_seed(5 + i)
    L = T.compute_losses_v3(m, batch, "cpu", mode="diffusion")
    if i == 0: print("grad keys", sorted(k for k, v in L.items() if torch.is_tensor(v) and v.requires_grad))
    opt.zero_grad(set_to_none=True); L["loss"].backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0); opt.step(); ema.update(m)
after = m.state_dict()
moved = [k for k in before if not M.bit_equal(before[k], after[k])]
print("moved outside", [k for k in moved if not M.is_trained(k, mods, dec)])
for mod in mods:
    tens = [n for n, _ in m.named_parameters() if M.is_trainable_under(n, (mod,)) and not M.is_trainable_under(n, dec)]
    mv = [k for k in moved if M.is_trainable_under(k, (mod,))]
    print(f"  {mod}: {len(mv)} moved of {len(tens)} params")
print("not moved trained params", [n for n, p in m.named_parameters() if p.requires_grad and n not in moved])
print("guard", guard.changed(m))
