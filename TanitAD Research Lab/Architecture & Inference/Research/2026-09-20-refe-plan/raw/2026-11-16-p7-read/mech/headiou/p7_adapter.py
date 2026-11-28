#!/usr/bin/env python3
"""P7 head-IoU adapter: a thin NEW wrapper around the UNEDITED instrument eval/aux_navtest_forward.py (ANF).

What it changes (all by monkeypatching ANF's module globals, nothing on disk is edited):
  1. load_aux_head: reads the head from the aux_drv entry of P7's model_final.pt (a state dict of the AuxDrivable container, keys 'head.*'), builds AUX.AuxDrivable(cfg) with
     the config P7 trained (tap visual, depth 2, feed off, w_aux 0.1, cell 2.0 m, 40x32 grid), loads STRICTLY.  The arm "path" may be
        <ckpt>            the trained P7 head
        ZERO::<ckpt>      the same head with every parameter zeroed (control: logit 0 everywhere => empty predicted mask => IoU 0)
        FRESH::<ckpt>     the same architecture freshly initialised (seed 0), never trained (the P3 floor, A2)
  2. CtxTap: after the planner is built, proves the planner's trunk equals ckpt['model'] tensor for tensor (torch.equal on every key) and writes proof_trunk_load.json;
     also hooks scene_proj (the trainer's tap: 2nd call = visual_ctx) so that forward_token can prove tap.ctx == the trainer's tap tensor.
  3. identity_report: the banked C3 table is the u400 scorer (7 components), so it cannot be P7's reference; the scorer logits of this forward are compared with P7's own
     dump (seams/stage2_p7_full.proposals.npz, 12,146 x 64 x 6) instead.  Reported, not asserted bit-identical (different sampling/grouping may differ).
  4. drive-letter paths (the dev box's D: is now E:) -- W3_EXPORT / DB root overrides.
Everything else (the planner's own code path, the lock, the ctx cache, the output contract) is ANF's.
"""
import json
import os
import sys

TREE = "F:/tanitad_scratch/p7headiou/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
EVAL = TREE + "/eval"
REFE = TREE + "/refe"
sys.path.insert(0, EVAL)
sys.path.insert(0, REFE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import aux_navtest_forward as ANF  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402

assert os.path.abspath(ANF.__file__).replace("\\", "/").startswith(EVAL), ANF.__file__
assert os.path.abspath(AUX.__file__).replace("\\", "/").startswith(REFE), AUX.__file__

P7_CKPT = "F:/TanitAD_p7read/run/model_final.pt"
P7_DUMP = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.proposals.npz"
PROOF_DIR = os.environ.get("P7_PROOF_DIR", "")
CFG = dict(tap="visual", depth=2, feed="off", w_aux=0.1, seed=0)

ANF.W3_EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
ANF.WAIT_IDLE = "E:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
ANF.CKPT = "NO-SUCH-U400"   # keep the u400 md5 guard off (it fires only when --ckpt == ANF.CKPT)
ANF.RULE = "navsim_v1"      # P7 has a 6-output scorer; the rule P7's own seam dump used (stage2_p7_full.proposals.npz rule=navsim_v1). Selection does not touch the head's ctx.
_orig_acquire =ANF.acquire_gpu_lock


def _acquire(**kw):          # acquire_gpu_lock's wait_idle default was bound at def time (a D: path that no longer exists): pass the E: one explicitly
    kw["wait_idle"] = ANF.WAIT_IDLE
    return _orig_acquire(**kw)


ANF.acquire_gpu_lock = _acquire
_PROOF = {"trunk": None, "tap_vs_scene_proj": []}


def _write_proof():
    if PROOF_DIR:
        os.makedirs(PROOF_DIR, exist_ok=True)
        with open(os.path.join(PROOF_DIR, "proof_run.json"), "w", encoding="utf-8") as f:
            json.dump(_PROOF, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------------------------------------------- head
def load_p7_head(path, device="cpu"):
    mode, ck = "P7", path
    for pre in ("ZERO::", "FRESH::"):
        if path.startswith(pre):
            mode, ck = pre[:-2], path[len(pre):]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if "aux_drv" not in sd:
        raise ANF.HeadRefused(f"{ck}: no aux_drv entry")
    meta = sd.get("aux_drv_meta") or {}
    if not (meta.get("cell") == 2.0 and meta.get("nx") == 40 and meta.get("ny") == 32 and meta.get("tap") == "visual" and meta.get("feed_built") is False and meta.get("depth") == 2):
        raise ANF.HeadRefused(f"{ck}: aux_drv_meta is not the P7 head config: {meta}")
    cfg = AUX.AuxDrivableConfig(**CFG)
    box = AUX.AuxDrivable(cfg)
    if mode == "FRESH":
        pass
    else:
        box.load_state_dict(sd["aux_drv"], strict=True)
        if mode == "ZERO":
            with torch.no_grad():
                for p in box.parameters():
                    p.zero_()
    head = box.head.to(device).eval()
    n = int(sum(p.numel() for p in head.parameters()))
    if n != int(meta["n_params"]):
        raise ANF.HeadRefused(f"{ck}: head has {n} params, aux_drv_meta says {meta['n_params']}")
    info = {"path": os.path.abspath(ck), "sha256": ANF.sha256_file(ck), "mode": mode, "update": None, "config": dict(CFG), "store_sha256": meta.get("store_sha256"),
            "frames": None, "shuffle_seed": None, "feed_present": False, "n_params": n, "aux_drv_meta": {k: meta[k] for k in meta if not isinstance(meta[k], (dict, list))}}
    return head, info


ANF.load_aux_head = load_p7_head


# ---------------------------------------------------------------------------------------------------------------- trunk proof + tap
class P7CtxTap(ANF.CtxTap):
    def __init__(self, model):
        super().__init__(model)
        ck = torch.load(P7_CKPT, map_location="cpu", weights_only=False)["model"]
        sd = model.state_dict()
        keys_ck, keys_m = set(ck), set(sd)
        bad = []
        for k in sorted(keys_ck & keys_m):
            a, b = sd[k].detach().cpu(), ck[k]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                bad.append(k)
        _PROOF["trunk"] = {"n_ckpt_keys": len(keys_ck), "n_model_keys": len(keys_m), "only_in_ckpt": sorted(keys_ck - keys_m)[:10], "only_in_model": sorted(keys_m - keys_ck)[:10],
                           "n_unequal": len(bad), "unequal_first": bad[:10], "all_equal": (not bad and keys_ck == keys_m), "ckpt_sha256": ANF.sha256_file(P7_CKPT)}
        print("TRUNK_PROOF", json.dumps(_PROOF["trunk"]), flush=True)
        if not _PROOF["trunk"]["all_equal"]:
            _write_proof()
            raise RuntimeError("the planner's trunk is NOT byte-equal to P7's model_final.pt")
        self._sp = []
        self._h2 = model.scene_proj.register_forward_hook(lambda m, i, o: self._sp.append(o.detach()))
        _write_proof()


ANF.CtxTap = P7CtxTap
_orig_forward_token = ANF.forward_token


def forward_token(planner, ego, img, tap, swap=None):
    tap._sp.clear()
    out = _orig_forward_token(planner, ego, img, tap, swap)
    # the trainer's tap = scene_proj's 2nd output of ONE forward (visual_ctx); the head here reads tap.ctx (score_dec's input, detached)
    if len(tap._sp) >= 2:
        a, b = tap._sp[1].float(), tap.ctx.float()
        _PROOF["tap_vs_scene_proj"].append({"n_calls": len(tap._sp), "equal": bool(a.shape == b.shape and torch.equal(a, b)), "max_abs": float((a - b).abs().max()) if a.shape == b.shape else None})
    else:
        _PROOF["tap_vs_scene_proj"].append({"n_calls": len(tap._sp), "equal": None})
    _write_proof()
    return out


ANF.forward_token = forward_token

# ---------------------------------------------------------------------------------------------------------------- identity vs P7's own dump
_dump = None


def identity_report(new, ref, tokens):
    global _dump
    if _dump is None:
        z = np.load(P7_DUMP, allow_pickle=False)
        _dump = ({str(t): i for i, t in enumerate(z["token"])}, np.asarray(z["logits"], np.float32))
    idx, lg = _dump
    new = np.asarray(new, np.float32)
    miss = [t for t in tokens if str(t) not in idx]
    if miss or new.shape[1:] != lg.shape[1:]:
        return {"PASS": False, "n_tokens": int(new.shape[0]), "n_tokens_bit_identical": 0, "max_abs_delta": float("inf"), "note": f"p7 dump mismatch: missing {len(miss)}, new {new.shape}, dump {lg.shape}"}
    r = lg[[idx[str(t)] for t in tokens]]
    d = np.abs(new.astype(np.float64) - r.astype(np.float64)).reshape(new.shape[0], -1).max(1)
    return {"PASS": bool(np.array_equal(new, r)), "n_tokens": int(new.shape[0]), "n_tokens_bit_identical": int((d == 0).sum()), "max_abs_delta": float(d.max()),
            "median_token_max_delta": float(np.median(d)), "ref_actual": P7_DUMP, "worst_tokens": [(str(tokens[i]), float(d[i])) for i in np.argsort(-d)[:5]],
            "note": "reference = P7's own dump (stage2_p7_full.proposals.npz logits, 6 comps), NOT the u400 C3 table"}


ANF.identity_report = identity_report

if __name__ == "__main__":
    import ego_yaw as EY  # P7 is a yaw-trained checkpoint: the P7 read fed it the navtest yaw table (stage2_sft --ego-yaw-table); the planner alone would feed ego[4]=0
    assert os.path.abspath(EY.__file__).replace("\\", "/").startswith(REFE), EY.__file__
    EY.patch_planner_class("E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz")
    try:
        import refe_navtest_seam as SEAM  # noqa: E402
        SEAM.W3_EXPORT = ANF.W3_EXPORT
        SEAM.TEST_DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
    except Exception as e:  # the seam is re-imported inside ANF.run; a failure here is reported, not hidden
        print("SEAM_PATCH_WARN", repr(e), flush=True)
    sys.exit(ANF.main())
