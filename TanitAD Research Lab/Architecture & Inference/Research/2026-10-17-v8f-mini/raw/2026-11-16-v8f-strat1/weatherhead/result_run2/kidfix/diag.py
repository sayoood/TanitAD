import sys, os, json, numpy as np, torch
sys.path.insert(0, "F:/TanitAD-scratch/strat1/whead/build")
import wh_lib as W
import tanitad.models.v8f_mini as M
arm, bs_list = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
dt = sys.argv[3] if len(sys.argv) > 3 else "f32"
ck = torch.load(f"F:/TanitAD-scratch/strat1/whead/run2/fit/{arm}/ckpt_wh.pt", map_location="cpu", weights_only=False)
model = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), W.p10_zero()); model.load_state_dict(ck["model"])
if dt == "f64": model = model.double()
join = dict(np.load(W.pinned("join_eval141.npz")))
ew = W.eval_windows(W.pinned("eval_grid_windows.json"), join["clip"])
rs = {"ci": ew["ci"], "t": ew["t"], "row": np.zeros(len(ew["ci"]), np.int64)}
src = W.StreamSource(W.EVAL_CACHE, [str(c) for c in join["clip"]], rs)
ev = np.load(f"F:/TanitAD-scratch/strat1/whead/run2/t0/eval_WH-{arm}_real.npz", allow_pickle=True)["ctx_pool_logits"]
kid = np.load(f"F:/TanitAD-scratch/strat1/whead/run2/kid/kid_{arm}.npz")["ctx_pool_logits"]
print("threads", torch.get_num_threads(), "n", len(ev), "eval-vs-banked-kid max", np.abs(ev-kid).max())
for bs in bs_list:
    lg = W.predict(model, src, np.arange(len(ev)), torch.device("cpu"), bs=bs) if dt=="f32" else None
    d = np.abs(lg - ev)
    print(f"bs={bs}: max {d.max():.3e} n>1e-5 {(d.max(1)>1e-5).sum()} median-win-max {np.median(d.max(1)):.2e}; at worst |logit|={abs(ev[np.unravel_index(d.argmax(),d.shape)]):.2f} col {np.unravel_index(d.argmax(),d.shape)[1]}; ev absmax {np.abs(ev).max():.2f}")
    np.save(f"tmp/lg_{arm}_bs{bs}.npy", lg)
