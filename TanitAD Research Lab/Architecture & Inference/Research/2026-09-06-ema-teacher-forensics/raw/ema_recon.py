"""C3/C4 gating controls: is the EMA teacher present, and is this the right ARM?"""
import json, os, sys, torch

ASSETS = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
ARMS = ["v7tiny_emao14_30k", "v7tiny_emao14_30k_tauramp", "v7tiny_o14fut30k",
        "v7tiny_postrain30k", "v7tiny_rdw8p30k", "v7tiny_splitp30k",
        "v7tiny_k8clip05p30k", "v7tiny_k60clip05p30k", "v7tiny_postrain30k_freeze"]

out = {}
for arm in ARMS:
    d = os.path.join(ASSETS, arm)
    ck = os.path.join(d, "ckpt.pt")
    rec = {"arm": arm, "ckpt": ck, "exists": os.path.exists(ck)}
    if not rec["exists"]:
        out[arm] = rec
        continue
    rec["bytes"] = os.path.getsize(ck)
    try:
        obj = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:
        rec["load_error"] = "%s: %s" % (type(e).__name__, e)
        out[arm] = rec
        continue
    rec["top_keys"] = sorted(list(obj.keys())) if isinstance(obj, dict) else str(type(obj))
    # locate the model state dict
    sd = None
    for k in ("model", "state_dict", "model_state", "stack"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            rec["sd_key"] = k
            break
    if sd is None and isinstance(obj, dict) and all(hasattr(v, "shape") for v in list(obj.values())[:5]):
        sd = obj
        rec["sd_key"] = "<root>"
    if sd is None:
        rec["sd_key"] = None
        out[arm] = rec
        continue
    keys = list(sd.keys())
    rec["n_tensors"] = len(keys)
    # C3: teacher present?
    ema_enc = [k for k in keys if k.startswith("ema_o5_enc.")]
    ema_ro = [k for k in keys if k.startswith("ema_o5_ro.")]
    rec["C3_ema_o5_enc_n"] = len(ema_enc)
    rec["C3_ema_o5_ro_n"] = len(ema_ro)
    rec["C3_ema_present"] = (len(ema_enc) + len(ema_ro)) > 0
    rec["ema_prefixes_any"] = sorted(set(k.split(".")[0] for k in keys if "ema" in k.lower()))
    rec["top_level_modules"] = sorted(set(k.split(".")[0] for k in keys))
    if ema_enc:
        rec["ema_enc_numel"] = int(sum(sd[k].numel() for k in ema_enc))
        rec["ema_ro_numel"] = int(sum(sd[k].numel() for k in ema_ro))
        rec["ema_sample_keys"] = ema_enc[:5]
    # C4: the ARM's own record
    cfgj = os.path.join(d, "config.json")
    cfg = None
    if os.path.exists(cfgj):
        try:
            cfg = json.load(open(cfgj, "r", encoding="utf-8"))
            rec["config_json"] = "present"
        except Exception as e:
            rec["config_json"] = "unreadable %s" % e
    if cfg is None and isinstance(obj, dict):
        for k in ("config", "cfg", "args"):
            if k in obj:
                cfg = obj[k] if isinstance(obj[k], dict) else vars(obj[k])
                rec["config_json"] = "from-ckpt[%s]" % k
                break
    if isinstance(cfg, dict):
        argv = cfg.get("argv")
        rec["C4_argv"] = argv if isinstance(argv, (list, str)) else None
        for f in ("o5_target", "ema_decay", "lr", "wd", "weight_decay", "steps", "run",
                  "stage", "horizons", "o5_k", "o5_form", "seed"):
            if f in cfg:
                rec["C4_" + f] = cfg[f]
        if "weights" in cfg and isinstance(cfg["weights"], dict):
            rec["C4_weights"] = cfg["weights"]
        if "freeze" in cfg and isinstance(cfg["freeze"], dict):
            rec["C4_freeze_n_trainable"] = cfg["freeze"].get("n_trainable")
    if isinstance(obj, dict):
        for k in ("step", "global_step", "steps"):
            if k in obj and isinstance(obj[k], int):
                rec["ckpt_step"] = obj[k]
    out[arm] = rec
    del obj, sd

print(json.dumps(out, indent=1, default=str))
