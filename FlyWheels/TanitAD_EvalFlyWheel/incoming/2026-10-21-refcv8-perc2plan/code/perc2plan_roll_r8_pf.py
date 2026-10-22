"""perc2plan_roll_r8_pf.py -- PERC2PLAN-R8 (PREREG_PERC2PLAN.md, sha256 8baeedd8...): ONE os-only inference-seed roll of
refcv8 EMA-48,852 with the per-frame nav feed, with ONE eval-time perception intervention installed.

HOW (a WRAPPER, the fan_roll_r8_pf.py pattern -- it edits no file):
  1. optionally re-points `r8_repoint.REMAP_OVERRIDES` at a Thor remap JSON (a PATH, not a number; recorded);
  2. imports `roll_seed_r8_pf` as a module (its module-level code = the registered nav-feed overlay check, the r8
     re-pointing, the overlay install, the --nav-feed perframe_full argv append -- all UNCHANGED);
  3. wraps `r7_roll.Refcv6Proxy.__call__` ONLY to (a) publish the current window key (ctx.index) and (b) install the
     hooks on the first call -- it then calls the original UNCHANGED;
  4. hooks (the arm decides what they do; BASE = identity, the original objects pass through):
       AGENT: `AgentTokenEmbed.register_forward_hook` replaces (tokens, pad)  -> decoder cross-agent + scene hook;
       BEV:   the instance attribute `model._bev_hook` is wrapped; the returned dict's `bev_tokens`, `bev_feats`
              (and `bev_pad`) are replaced                                     -> scene hook + decoder BEV coupling;
       a forward PRE-hook on the core decoder asserts (identity) that its `agent_tokens` / `bev` kwargs ARE the hooked
       tensors (C4 reach);
  5. calls `roll_seed_r8_pf.main()` with sys.argv = EXACTLY the battery child argv for that seed (+ --os-only).

SHUF donor (PREREG s2 + AMENDMENT 1): window j of episode e receives the UN-intervened tensors of window
min(j, L_prev - 1) of the previously processed episode; the FIRST episode's donor is the LAST episode of the same-seed
BASE roll, banked by that roll (`--donor-bank-out`) and loaded read-only (`--donor-bank-in`).

    python perc2plan_roll_r8_pf.py --arm AGT_ZERO --pf-code <hold_r8pf/code> [--remap-json <thor remap>] --ckpt CK \
        --config CFG --seed 1 --dump-dir D --windows-json W --out-json RJ --meta-json M --device cuda --os-only \
        [--donor-bank-in B.pt] [--donor-bank-out B.pt] [--smoke-cpu-fp32-trunk]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass

#: arm -> (agent mode, bev mode); mode None = identity
ARMS = {"BASE": (None, None), "AGT_ZERO": ("zero", None), "AGT_SHUF": ("shuf", None),
        "BEV_ZERO": (None, "zero"), "BEV_SHUF": (None, "shuf"), "BOTH_ZERO": ("zero", "zero")}
BEV_KEYS = ("bev_tokens", "bev_feats", "bev_pad")

ST = {"arm": None, "agent_mode": None, "bev_mode": None, "key": None, "installed": False,
      "cur_clip": None, "cur_list": [], "prev_list": None, "prev_clip": None, "bank_in": None,
      "episodes_seen": [], "last_agent_out": None, "last_bev_feats_out": None, "census": {},
      "c": {"proxy_calls": 0, "agent_fwd": 0, "agent_orig_nonzero": 0, "agent_changed": 0, "agent_max_abs_new": 0.0,
            "agent_max_abs_diff": 0.0, "agent_pad_changed": 0,
            "bev_fwd": 0, "bev_orig_nonzero": 0, "bev_changed": 0, "bev_max_abs_new": 0.0, "bev_max_abs_diff": 0.0,
            "bevtok_changed": 0, "bevfeat_changed": 0,
            "dec_calls": 0, "dec_agent_is_hooked": 0, "dec_bev_is_hooked": 0, "dec_agent_none": 0, "dec_bev_none": 0,
            "donor_from_bank": 0, "donor_from_prev_episode": 0, "donor_same_clip": 0}}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rec():
    """The record for the CURRENT window (created on its first forward)."""
    if not ST["cur_list"] or ST["cur_list"][-1]["key"] != ST["key"]:
        ST["cur_list"].append({"key": ST["key"], "agent": None, "bev": None})
    return ST["cur_list"][-1]


def _donor(field):
    """The un-intervened donor tensors for the current window (cross-clip, by processing order)."""
    j = len(ST["cur_list"]) - 1
    src, tag = (ST["prev_list"], "donor_from_prev_episode")
    if src is None:
        src, tag = ST["bank_in"], "donor_from_bank"
    if not src:
        raise RuntimeError("[perc2plan] SHUF arm but no donor episode (first episode needs --donor-bank-in)")
    d = src[min(j, len(src) - 1)]
    if int(d["key"][0]) == int(ST["key"][0]):
        ST["c"]["donor_same_clip"] += 1
    if d.get(field) is None:
        raise RuntimeError(f"[perc2plan] donor window {d['key']} carries no {field} tensors")
    return d[field], tag


def _like(x, ref):
    x = x.to(device=ref.device, dtype=ref.dtype)
    if x.shape[0] != ref.shape[0]:
        x = x[:1].expand(ref.shape[0], *x.shape[1:])
    if tuple(x.shape) != tuple(ref.shape):
        raise RuntimeError(f"[perc2plan] donor shape {tuple(x.shape)} != target {tuple(ref.shape)}")
    return x.contiguous()


def _agent_hook(module, inp, out):
    import torch
    tok, pad = out
    c = ST["c"]
    c["agent_fwd"] += 1
    r = _rec()
    if r["agent"] is None:
        r["agent"] = {"tok": tok[:1].detach().clone(), "pad": pad[:1].detach().clone()}
    mode = ST["agent_mode"]
    if mode is None:
        new_tok, new_pad = tok, pad
    elif mode == "zero":
        new_tok, new_pad = torch.zeros_like(tok), pad
    elif mode == "shuf":
        d, tag = _donor("agent")
        c[tag] += 1
        new_tok, new_pad = _like(d["tok"], tok), _like(d["pad"], pad)
    else:
        raise RuntimeError(mode)
    o_nz = bool(tok.detach().abs().max() > 0)
    diff = float((new_tok.detach().float() - tok.detach().float()).abs().max())
    c["agent_orig_nonzero"] += int(o_nz)
    c["agent_changed"] += int(diff > 0)
    c["agent_max_abs_diff"] = max(c["agent_max_abs_diff"], diff)
    c["agent_max_abs_new"] = max(c["agent_max_abs_new"], float(new_tok.detach().float().abs().max()))
    c["agent_pad_changed"] += int(not torch.equal(new_pad, pad))
    ST["last_agent_out"] = new_tok
    if mode is None:
        return None                      # identity: the module's own output object flows on
    return (new_tok, new_pad)


def _wrap_bev_hook(model):
    orig_factory = model._bev_hook       # bound method

    def factory(*a, **k):
        h = orig_factory(*a, **k)
        if h is None:
            return None

        def hooked(*ha, **hk):
            import torch
            pout = h(*ha, **hk)
            c = ST["c"]
            c["bev_fwd"] += 1
            r = _rec()
            if r["bev"] is None:
                r["bev"] = {kk: (pout[kk][:1].detach().clone() if torch.is_tensor(pout.get(kk)) else None)
                            for kk in BEV_KEYS}
            mode = ST["bev_mode"]
            if "bev_tokens" not in pout or "bev_feats" not in pout:
                raise RuntimeError(f"[perc2plan] perception dict lacks bev_tokens/bev_feats: {sorted(pout)}")
            orig = {kk: pout.get(kk) for kk in BEV_KEYS}
            if mode is None:
                ST["last_bev_feats_out"] = pout["bev_feats"]
                c["bev_orig_nonzero"] += int(bool(pout["bev_tokens"].abs().max() > 0 or pout["bev_feats"].abs().max() > 0))
                return pout
            new = dict(pout)
            if mode == "zero":
                new["bev_tokens"] = torch.zeros_like(pout["bev_tokens"])
                new["bev_feats"] = torch.zeros_like(pout["bev_feats"])
            elif mode == "shuf":
                d, tag = _donor("bev")
                c[tag] += 1
                for kk in BEV_KEYS:
                    if torch.is_tensor(pout.get(kk)):
                        if d.get(kk) is None:
                            raise RuntimeError(f"[perc2plan] donor lacks {kk}")
                        new[kk] = _like(d[kk], pout[kk])
            else:
                raise RuntimeError(mode)
            dt = float((new["bev_tokens"].float() - orig["bev_tokens"].float()).abs().max())
            df = float((new["bev_feats"].float() - orig["bev_feats"].float()).abs().max())
            c["bev_orig_nonzero"] += int(bool(orig["bev_tokens"].abs().max() > 0 or orig["bev_feats"].abs().max() > 0))
            c["bevtok_changed"] += int(dt > 0)
            c["bevfeat_changed"] += int(df > 0)
            c["bev_changed"] += int(dt > 0 and df > 0)
            c["bev_max_abs_diff"] = max(c["bev_max_abs_diff"], dt, df)
            c["bev_max_abs_new"] = max(c["bev_max_abs_new"], float(new["bev_tokens"].abs().max()),
                                       float(new["bev_feats"].abs().max()))
            ST["last_bev_feats_out"] = new["bev_feats"]
            return new
        return hooked

    model._bev_hook = factory            # instance attribute shadows the bound method (refc_v3.forward: self._bev_hook)


def _dec_prehook(module, args, kwargs):
    c = ST["c"]
    c["dec_calls"] += 1
    at, bv = kwargs.get("agent_tokens"), kwargs.get("bev")
    c["dec_agent_none"] += int(at is None)
    c["dec_bev_none"] += int(bv is None)
    c["dec_agent_is_hooked"] += int(at is not None and at is ST["last_agent_out"])
    c["dec_bev_is_hooked"] += int(bv is not None and bv is ST["last_bev_feats_out"])
    return None


def _install(model):
    from tanitad.refs import refc_agents as RA
    embeds = [(n, m) for n, m in model.named_modules() if isinstance(m, RA.AgentTokenEmbed)]
    if len(embeds) != 1:
        raise RuntimeError(f"[perc2plan] expected ONE AgentTokenEmbed, found {[n for n, _ in embeds]}")
    embeds[0][1].register_forward_hook(_agent_hook)
    if not hasattr(model, "_bev_hook"):
        raise RuntimeError("[perc2plan] model has no _bev_hook")
    _wrap_bev_hook(model)
    core = getattr(model, "core", None)
    dec = getattr(core, "decoder", None)
    if dec is None:
        raise RuntimeError("[perc2plan] model.core.decoder not found")
    dec.register_forward_pre_hook(_dec_prehook, with_kwargs=True)
    ST["census"] = {"agent_embed_module": embeds[0][0], "model_class": type(model).__name__,
                    "core_class": type(core).__name__, "decoder_class": type(dec).__name__,
                    "r8_drv_is_none": getattr(model, "r8_drv", None) is None,
                    "wp_index_cfg_is_none": getattr(dec, "wp_index_cfg", None) is None,
                    "bev_coupling_params": int(getattr(dec, "bev_coupling_params", lambda: 0)()),
                    "tac_decoder_v6_built": getattr(model, "tac_decoder_v6", None) is not None,
                    "perception_attached": getattr(model, "_perception", None) is not None,
                    "map_hires_attached": getattr(model, "_map_hires", None) is not None}
    if not ST["census"]["r8_drv_is_none"]:
        raise RuntimeError("[perc2plan] the drivable critic is built: a perception channel the prereg did not register")
    ST["installed"] = True


def _install_proxy(RR):
    orig = RR.Refcv6Proxy.__call__

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        if not ST["installed"]:
            if os.environ.get("P2P_DEBUG_NOHOOK") == "1":       # C1 bisection only (smoke); never set on a real roll
                ST["installed"] = True
                ST["census"] = {"debug": "P2P_DEBUG_NOHOOK=1 -- NO hooks installed"}
            else:
                _install(self.__dict__["model"])
        key = tuple(int(x) for x in self.ctx.index)
        if key[0] != ST["cur_clip"]:
            if ST["cur_list"]:
                ST["prev_list"], ST["prev_clip"] = ST["cur_list"], ST["cur_clip"]
            ST["cur_list"], ST["cur_clip"] = [], key[0]
            ST["episodes_seen"].append(key[0])
        ST["key"] = key
        ST["c"]["proxy_calls"] += 1
        return orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)

    RR.Refcv6Proxy.__call__ = wrapped


def _bank_save(path):
    import torch

    def cpu(d):
        return None if d is None else {k: (None if v is None else v.detach().cpu()) for k, v in d.items()}
    rows = [{"key": list(r["key"]), "agent": cpu(r["agent"]), "bev": cpu(r["bev"])} for r in ST["cur_list"]]
    torch.save({"clip_index": ST["cur_clip"], "rows": rows, "arm": ST["arm"]}, path)
    return {"path": str(path), "clip_index": ST["cur_clip"], "n_rows": len(rows), "sha256": sha256_file(path)}


def _bank_load(path, dev):
    import torch
    b = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for r in b["rows"]:
        rows.append({"key": tuple(r["key"]),
                     "agent": None if r["agent"] is None else {k: v.to(dev) for k, v in r["agent"].items()},
                     "bev": None if r["bev"] is None else {k: (None if v is None else v.to(dev)) for k, v in r["bev"].items()}})
    return rows, {"path": str(path), "clip_index": b["clip_index"], "n_rows": len(rows), "sha256": sha256_file(path),
                  "banked_by_arm": b.get("arm")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--pf-code", required=True)
    ap.add_argument("--remap-json", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--meta-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--os-only", action="store_true")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true")
    ap.add_argument("--donor-bank-in", default=None)
    ap.add_argument("--donor-bank-out", default=None)
    a = ap.parse_args()
    ST["arm"] = a.arm
    ST["agent_mode"], ST["bev_mode"] = ARMS[a.arm]
    shuf = "shuf" in ARMS[a.arm]
    if shuf and not a.donor_bank_in:
        raise SystemExit("[perc2plan] a SHUF arm needs --donor-bank-in (the same-seed BASE roll's last-episode bank)")
    pf_code = Path(a.pf_code)
    remap = None
    if a.remap_json:
        r8_code = Path(os.environ["PF_R8_CODE"])
        sys.path.insert(0, str(r8_code))
        import r8_repoint  # noqa: E402
        remap = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "thor_file": a.remap_json,
                 "thor_map": json.load(open(a.remap_json, encoding="utf-8"))}
        r8_repoint.REMAP_OVERRIDES = Path(a.remap_json)
    child = [str(pf_code / "roll_seed_r8_pf.py"), "--ckpt", a.ckpt, "--config", a.config, "--seed", str(a.seed),
             "--dump-dir", a.dump_dir, "--windows-json", a.windows_json, "--device", a.device, "--out-json", a.out_json]
    if a.smoke_cpu_fp32_trunk:
        child.append("--smoke-cpu-fp32-trunk")
    if a.os_only:
        child.append("--os-only")
    sys.argv = child
    sys.path.insert(0, str(pf_code))
    import roll_seed_r8_pf as RP  # noqa: E402
    import r7_roll as RR  # noqa: E402
    import torch  # noqa: E402
    assert sys.modules["r7_roll"] is RR
    bank_in = None
    if a.donor_bank_in:
        ST["bank_in"], bank_in = _bank_load(a.donor_bank_in, torch.device(a.device))
    _install_proxy(RR)
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    print(f"[perc2plan] arm={a.arm} agent={ST['agent_mode']} bev={ST['bev_mode']} seed={a.seed} ZZP2PSTARTZZ", flush=True)
    rc = RP.main()
    wall = time.time() - t0
    bank_out = _bank_save(a.donor_bank_out) if a.donor_bank_out else None
    c = ST["c"]
    am, bm = ST["agent_mode"], ST["bev_mode"]
    # C4 (PREREG s5) -- read mechanically from the counters
    c4 = {"census_ok": bool(ST["census"].get("r8_drv_is_none") and ST["census"].get("wp_index_cfg_is_none")
                            and ST["census"].get("bev_coupling_params", 0) > 0 and ST["census"].get("tac_decoder_v6_built")),
          "decoder_sees_hooked_agent": c["dec_agent_is_hooked"] == c["dec_calls"] and c["dec_calls"] > 0,
          "decoder_sees_hooked_bev": c["dec_bev_is_hooked"] == c["dec_calls"] and c["dec_calls"] > 0}
    if am is None:
        c4["agent_identity"] = c["agent_max_abs_diff"] == 0.0 and c["agent_changed"] == 0
    elif am == "zero":
        c4["agent_zero_exact"] = c["agent_max_abs_new"] == 0.0
        c4["agent_changed_all_nonzero"] = c["agent_changed"] == c["agent_orig_nonzero"] and c["agent_fwd"] > 0
    else:
        c4["agent_changed_ge99pct"] = c["agent_fwd"] > 0 and c["agent_changed"] >= 0.99 * c["agent_fwd"]
    if bm is None:
        c4["bev_identity"] = c["bev_max_abs_diff"] == 0.0 and c["bev_changed"] == 0
    elif bm == "zero":
        c4["bev_zero_exact"] = c["bev_max_abs_new"] == 0.0
        c4["bev_changed_all_nonzero"] = c["bev_changed"] == c["bev_orig_nonzero"] and c["bev_fwd"] > 0
    else:
        c4["bev_changed_ge99pct"] = c["bev_fwd"] > 0 and c["bev_changed"] >= 0.99 * c["bev_fwd"]
    c4["all_pass"] = all(v for v in c4.values())
    meta = {"prereg": "PREREG_PERC2PLAN.md sha256 8baeedd8a22a078d74868d3b730d057a57041b8e8e35f531ed47ee72376e3b61",
            "arm": a.arm, "agent_mode": am, "bev_mode": bm, "inference_seed": a.seed, "os_only": bool(a.os_only),
            "child_rc": rc, "child_argv": child, "wall_s": round(wall, 1), "counters": c, "C4": c4,
            "census": ST["census"], "n_episodes_seen": len(ST["episodes_seen"]),
            "donor_bank_in": bank_in, "donor_bank_out": bank_out, "remap_overrides": remap,
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                              if torch.cuda.is_initialized() else None),
            "env": {k: os.environ.get(k) for k in ("R8_LAUNCH_TREE", "R8_EVAL_KIT", "PF_R8_CODE", "PF_NF_HOLD",
                                                    "R8_NO_POSE_SYNC", "OMP_NUM_THREADS", "PYTHONPATH",
                                                    "CUDA_VISIBLE_DEVICES")},
            "torch": torch.__version__, "this_script_sha256": sha256_file(__file__),
            "ckpt": a.ckpt, "config": a.config, "windows_json": a.windows_json,
            "producer": "perc2plan_roll_r8_pf.py (wrapper over roll_seed_r8_pf.main; hooks on AgentTokenEmbed + model._bev_hook)"}
    json.dump(meta, open(a.meta_json, "w", encoding="utf-8"), indent=1, default=str)
    print("[perc2plan] arm=%s rc=%s C4=%s fwd agent/bev/dec=%d/%d/%d wall=%.0fs %s" % (
        a.arm, rc, c4["all_pass"], c["agent_fwd"], c["bev_fwd"], c["dec_calls"], wall,
        "ZZP2PDONEOKZZ" if (c4["all_pass"] and not rc) else "ZZP2PDONEBADZZ"), flush=True)
    return 0 if (c4["all_pass"] and not rc) else 4


if __name__ == "__main__":
    sys.exit(main())
