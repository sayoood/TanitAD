"""T2-A prep: build refcv7-50,400 on CPU through the launch gate's own G-EVAL loader and dump the
facts the closed-loop policy needs (window, horizons, image_hw, which input seams are BUILT), plus the
shape/dtype contract of ONE real eval item. CPU ONLY.

usage:  python cpu_model_smoke.py <out.json> [ckpt] [config]
Runtime on the dev box CPU: ~3-5 min (build ~180 s measured 2026-09-28 for refcv7; item contract ~1-2 min).
Writes <out.json> progressively (a crash after the build still leaves the build facts).
"""
import os
import sys

# ---- CPU ONLY. "-1" (NOT "") hides the GPU on this box; an empty string does not. ---------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("REFCV6_KIT", "D:/refcv6_eval_kit")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json   # noqa: E402
import time   # noqa: E402

ROOT = os.environ.get("TANITAD_ROOT", "E:/Projects/TanitAD")
for p in (f"{ROOT}/stack", f"{ROOT}/taniteval", f"{ROOT}/stack/scripts", f"{ROOT}/taniteval/tools"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402

assert not torch.cuda.is_available(), "GPU visible: this prep job is CPU-ONLY (RTX 4060 belongs to another session)"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

out_path = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else "E:/refcv7_eval_kit/ckpt/ckpt_50400.pt"
cfgp = sys.argv[3] if len(sys.argv) > 3 else "E:/refcv7_eval_kit/ckpt/config.json"
res = {"cuda_available": torch.cuda.is_available(), "ckpt": ckpt, "config": cfgp, "t_start": time.strftime("%FT%T")}


def dump():
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


dump()
from tanitad.eval import refcv7_loader as L  # noqa: E402

config = L.load_config(cfgp)
t0 = time.time()
model, cfg, args, rec = L.build_model(config, ckpt, device="cpu")
res["build_s"] = round(time.time() - t0, 1)
core = model.core
res["build_s_done"] = True
dump()


def _g(fn):
    try:
        return fn()
    except BaseException as e:  # noqa: BLE001
        return f"ERR {e!r}"[:200]


facts = {
    "state_dict": {k: rec["state_dict"][k] for k in ("strict", "n_keys", "step")},
    "state_dict_missing": rec["state_dict"]["missing"], "state_dict_unexpected": rec["state_dict"]["unexpected"],
    "param_breakdown_equal": rec["param_breakdown"]["equal"],
    "param_total": rec["param_breakdown"]["rebuilt"].get("total"),
    "declared_vs_built_mismatches": rec["declared_vs_built"]["mismatches"],
    "mode": rec["mode"], "decoder_steps": rec["decoder_steps"], "sampler": rec["sampler"],
    "image_hw": _g(lambda: list(cfg.core.encoder.image_hw())),
    "encoder_in_channels": _g(lambda: int(cfg.core.encoder.in_channels)),
    "horizons": _g(lambda: [int(h) for h in cfg.core.trajectory.horizons]),
    "hier": bool(cfg.hier),
    "cfg_ego_state_inject": bool(cfg.ego_state_inject),
    "cfg_nav_args_inject": bool(cfg.nav_args_inject),
    "cfg_max_speed_input": bool(getattr(cfg, "max_speed_input", False)),
    "cfg_max_speed_onehot_v6": bool(getattr(cfg, "max_speed_onehot_v6", False)),
    "core_ego_hist_built": getattr(core, "ego_hist", None) is not None,
    "core_agents": _g(lambda: None if getattr(core, "agents", None) is None else {
        "enable": bool(core.agents.enable), "oracle": bool(core.agents.oracle)}),
    "residual_prior": _g(lambda: str(getattr(core.decoder, "residual_prior", None))),
    "has_perception": getattr(model, "_perception", None) is not None,
    "has_map_hires": getattr(model, "_map_hires", None) is not None,
    "has_lift_bank_hires": getattr(model, "_lift_bank_hires", None) is not None,
    "has_vis1": getattr(model, "_vis1", None) is not None,
    "cfg_window_attr": {k: getattr(args, k, None) for k in ("window", "n_stack", "ego_history", "no_strategic",
                                                           "residual_prior", "bev_source", "map_hires")},
    "departures": rec["departures"],
    "anchor_file_vs_ckpt_max_abs_diff": _g(lambda: {k: v["max_abs_diff"] for k, v in rec["anchor_file_vs_ckpt_buffers"].items()}),
}
res["model_facts"] = facts
res["t_build_done"] = time.strftime("%FT%T")
dump()
print("BUILD-DONE", res["build_s"], flush=True)

# ---- the shape/dtype contract of ONE real eval item (what the closed loop must reproduce) ----------- #
try:
    t1 = time.time()
    ds, _eps, _drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=False)
    res["eval_dataset_build_s"] = round(time.time() - t1, 1)
    res["eval_dataset_len"] = len(ds)
    res["eval_dataset_window"] = int(getattr(ds, "window", -1))
    res["cfg_core_window"] = int(cfg.core.window)
    it = ds[0]
    res["eval_item_index0"] = [int(x) for x in ds.index[0]]
    contract = {}
    for k, v in it.items():
        if hasattr(v, "shape"):
            contract[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            contract[k] = {"py": type(v).__name__}
    res["eval_item_contract"] = contract
    for k in ("pose_last", "pose_hist", "nav_cmd", "nav_valid", "v_max_ms", "v_max_valid", "map_ep"):
        if k in it:
            res.setdefault("eval_item_values", {})[k] = it[k].tolist() if hasattr(it[k], "tolist") else it[k]
    # the lift bank: which ids it covers and what extrinsics type it keys on
    bank = getattr(model, "_lift_bank_hires", None)
    res["lift_bank"] = {"len": len(bank), "type": type(bank).__name__,
                        "frame": str(getattr(bank, "frame", None))[:300]}
    import inspect
    from tanitad.models import refcv6_perception_branch as _pb
    res["lift_bank"]["for_episodes_sig"] = str(inspect.signature(_pb.LiftGeometryBank.for_episodes))
    res["lift_bank"]["init_sig"] = str(inspect.signature(_pb.LiftGeometryBank.__init__))
except BaseException as e:  # noqa: BLE001  -- never lose the build facts to a dataset failure
    res["eval_item_error"] = repr(e)[:600]
res["t_end"] = time.strftime("%FT%T")
dump()
print("SMOKE-DONE", flush=True)
