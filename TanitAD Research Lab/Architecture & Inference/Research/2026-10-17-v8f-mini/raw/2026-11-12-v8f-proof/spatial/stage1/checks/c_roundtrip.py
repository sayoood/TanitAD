"""Round trip: a GRID checkpoint written by the synthetic trainer (3 steps) is loaded by the model builder of
stack/scripts/v8f_t1_eval.py (cmd_run lines 505-509, reproduced verbatim), and a FINE-TUNE from it (--init-ckpt) is accepted for the same lever and refused for another."""
import json
import os
import shutil
import sys

TREE = "F:/TanitAD-scratch/v8proof/spatial/stage1/tree_patch"
S = "F:/TanitAD-scratch/v8proof/spatial/stage1"
import torch
import tanitad

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(TREE).lower())
sys.path.insert(0, TREE + "/stack/scripts")
import v8f_mini_train as TR
import v8f_t1_eval as T1                                    # the module itself imports; its builder lines are reproduced below
from tanitad.models import v8f_mini as M

assert os.path.abspath(T1.__file__).lower().startswith(os.path.abspath(TREE).lower())
res = {}
base = ["--synthetic", "--arm", "M2b", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0"]
for tag, extra in (("grid", ["--wm-state", "grid"]), ("mean640", ["--wm-dim", "640"]), ("mean", [])):
    out = f"{S}/tmp/rt_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    rc = TR.main([*base, *extra, "--out", out, "--steps", "3", "--log-every", "1"])
    ck = torch.load(f"{out}/ckpt_last.pt", map_location="cpu", weights_only=False)
    # ---- v8f_t1_eval.cmd_run lines 506-509 (verbatim)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    b = model.built_levers()
    res[tag] = {"rc": rc, "ckpt_cfg_has_wm_state": "wm_state" in ck["cfg"], "cfg_wm_state": cfg.wm_state, "strict_load": "ok",
                "built_wm_state": b["wm_state"], "l1_dim": b["wm_l1_dim"], "l1_pred_width": b["wm_l1_pred_width"]}
    # ---- fine-tune start: the same lever is accepted, another is refused
    fa = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *extra])
    fcfg = TR.make_config(fa)
    fcfg = M.V8FMiniConfig(**{**fcfg.__dict__, **{k: ck["cfg"][k] for k in ("d", "ff", "heads", "n_q", "depth_s", "depth_t", "depth_o", "nb", "hmax", "hist")}})
    fm = M.V8FMini(fcfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
    rec = TR.load_init_ckpt(f"{out}/ckpt_last.pt", fm, fcfg)
    res[tag]["init_ckpt_same_lever"] = {"n_mismatch_after_load": rec["n_mismatch_after_load"], "n_tensors": rec["n_tensors"]}
    for other_tag, other in (("grid", ["--wm-state", "grid"]), ("mean640", ["--wm-dim", "640"]), ("mean", [])):
        if other_tag == tag:
            continue
        oa = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", *other])
        ocfg = TR.make_config(oa)
        ocfg = M.V8FMiniConfig(**{**ocfg.__dict__, **{k: ck["cfg"][k] for k in ("d", "ff", "heads", "n_q", "depth_s", "depth_t", "depth_o", "nb", "hmax", "hist")}})
        om = M.V8FMini(ocfg, M.P10Prior.synthetic(0), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
        try:
            TR.load_init_ckpt(f"{out}/ckpt_last.pt", om, ocfg)
            res[tag][f"init_ckpt_into_{other_tag}"] = "ACCEPTED (BAD)"
        except ValueError as e:
            res[tag][f"init_ckpt_into_{other_tag}"] = "REFUSED: " + str(e)[:160]
res["PASS"] = bool(all(res[t]["rc"] == 0 and res[t]["strict_load"] == "ok" for t in ("grid", "mean640", "mean"))
                   and res["grid"]["cfg_wm_state"] == "grid" and res["grid"]["built_wm_state"] == "grid" and not res["mean"]["ckpt_cfg_has_wm_state"]
                   and all("REFUSED" in v for t in res if t != "PASS" for k, v in res[t].items() if k.startswith("init_ckpt_into_")))
json.dump(res, open(f"{S}/checks/c_roundtrip.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
