#!/usr/bin/env python3
"""YAX checkpoint tools (CPU, CUDA_VISIBLE_DEVICES=-1).  Pattern: YF2 build_yaw_ckpt.py + YF3 yf3_identity.py.
  build    : compose a FULL REFe checkpoint (what planner.py loads, strict) from u400 and an ax_ft.py final.pt.  Gates before the file is moved into place (red -> rc 3, ZZAXCKPT_GATE_RED):
             K1 final.pt state keys are a subset of the DEC keys, all exist in u400 with the same shape; every tensor NOT in final.pt is torch.equal to u400's (scorer, backbone, ... untouched)
             K2 changed tensors are a subset of the DEC keys   K3 AXOFF: column 2 of ego_enc.0.weight is EXACTLY 0.0   K4 composed DEC tensors equal final.pt's   K5 meta policy == arm policy
             meta: ego_ax_policy "zero"|"real", ego_ax_slot 2, ax_ft = provenance.
  identity : reload the composed FILE and check (GA-GAIN / G-CKPT):  AXOFF: decode at ax in {0, +0.5, -0.5, real} all torch.equal on N VAL rows; POSITIVE CONTROL u400 decodes differently at ax=+0.5 vs 0;
             AXON: decode at real ax differs from ax=0 (the probe sees a gain);  both: the file's DEC tensors equal final.pt's, meta policy matches.   Success token ZZAXID_OK.
    python ax_ckpt_tools.py build --final <arm>/final.pt --out <model.pt>
    python ax_ckpt_tools.py identity --ckpt <model.pt> --final <arm>/final.pt --out <json> [--frames 512] [--base u400.pt]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import yaw_ft_ref as FT  # noqa: E402
import yaw_common as YC  # noqa: E402
import ax_ft  # noqa: E402
import ckpt_io  # noqa: E402

AX = 2


def dec_key(k):
    return k.startswith(YC.DEC)


def compose(base_sd, fin, base_path):
    st = dict(base_sd["model"])
    for k, v in fin["state"].items():
        assert k in st, f"final.pt carries a key the base lacks: {k}"
        assert st[k].shape == v.shape, (k, st[k].shape, v.shape)
        st[k] = v.to(st[k].dtype).clone()
    meta = dict(base_sd.get("meta", {}) or {})
    meta["ego_ax_policy"], meta["ego_ax_slot"] = fin["ax_policy"], AX
    c = fin["config"]
    meta["ax_ft"] = {"arm": fin["arm"], "seed": c["args"]["seed"], "steps": c["args"]["steps"], "lr": c["args"]["lr"], "base": base_path, "base_sha256": c["base_sha256"],
                     "cache_manifest_sha256": c["cache_manifest_sha256"], "changed_keys": fin["changed_keys"], "ax_ft_code_sha256": c["code_sha256"].get("ax_ft.py")}
    return st, meta


def cmd_build(a):
    base_sd = torch.load(a.base, map_location="cpu", weights_only=False)
    base = base_sd["model"]
    fin = torch.load(a.final, map_location="cpu", weights_only=False)
    st, meta = compose(base_sd, fin, a.base) if all(k in base and dec_key(k) for k in fin["state"]) else (None, None)
    g = {"K1_state_keys_declared": st is not None}
    if st is not None:
        g["K1_untouched_equal"] = all(torch.equal(st[k], base[k]) for k in st if k not in fin["state"])
        changed = [k for k in st if not torch.equal(st[k], base[k])]
        g["changed_tensors"] = changed
        g["K2_changed_subset_of_DEC"] = all(dec_key(k) for k in changed)
        g["K3_AXOFF_col2_exactly_zero"] = bool((st["ego_enc.0.weight"][:, AX] == 0).all()) if fin["ax_policy"] == "zero" else None
        g["K4_composed_equals_final"] = all(torch.equal(st[k], v.to(st[k].dtype)) for k, v in fin["state"].items())
        g["K5_meta_policy"] = meta["ego_ax_policy"] == fin["ax_policy"]
    ok = all(v for k, v in g.items() if k.startswith("K") and v is not None)
    YC.jdump(g, str(a.out) + ".gates.json")
    print("gates", g)
    if not ok:
        print("ZZAXCKPT_GATE_RED")
        return 3
    n = ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": st, "meta": meta}, a.out)
    print(f"ZZAXCKPT_OK {a.out} bytes {n} arm {fin['arm']}")
    return 0


def cmd_identity(a):
    model, cfg, _ = FT.build_decoder(a.base)
    base_model, _, _ = FT.build_decoder(a.base)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    fin = torch.load(a.final, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v for k, v in sd["model"].items() if dec_key(k)}, strict=False)
    meta = sd.get("meta", {})
    man = json.load(open(a.manifest, encoding="utf-8"))
    cache = FT.Cache(a.cache)
    rows = FT.load_rows(cache, man)["VAL"][:a.frames]
    ids = list(range(len(rows)))
    sc, er, go, _ = ax_ft.make_batch(rows, ids, cache, "real")
    pol = fin["ax_policy"]
    res = {"ckpt": str(a.ckpt), "policy": pol, "n_frames": len(rows), "meta_policy": meta.get("ego_ax_policy"), "meta_slot": meta.get("ego_ax_slot")}
    res["G_CKPT_file_equals_final"] = all(torch.equal(sd["model"][k], v.to(sd["model"][k].dtype)) for k, v in fin["state"].items())
    res["G_CKPT_meta"] = bool(meta.get("ego_ax_policy") == pol and meta.get("ego_ax_slot") == AX)
    res["col2_abs_sum"] = float(sd["model"]["ego_enc.0.weight"][:, AX].abs().sum())

    def dec(m, ax):
        e = er.clone()
        if ax is not None:
            e[:, AX] = ax
        with torch.no_grad():
            return FT.original_decode(m, cfg, sc, e, go)

    base0, basep = dec(base_model, 0.0), dec(base_model, 0.5)
    res["positive_control_u400_gain_visible"] = bool(not torch.equal(base0, basep))
    res["positive_control_u400_max_abs_diff"] = float((basep - base0).abs().max())
    d0 = dec(model, 0.0)
    if pol == "zero":
        res["gain_zero_exact"] = {str(ax): bool(torch.equal(d0, dec(model, ax))) for ax in (0.5, -0.5, None)}
        res["GA_GAIN"] = bool(all(res["gain_zero_exact"].values()) and res["col2_abs_sum"] == 0.0)
    else:
        res["gain_visible_real_vs_zero"] = bool(not torch.equal(d0, dec(model, None)))
        res["max_abs_diff_real_vs_zero"] = float((dec(model, None) - d0).abs().max())
        res["GA_GAIN"] = bool(res["gain_visible_real_vs_zero"])               # AXON must still USE ax (else AXON == AXOFF and the control is empty)
    ok = bool(res["GA_GAIN"] and res["G_CKPT_file_equals_final"] and res["G_CKPT_meta"] and res["positive_control_u400_gain_visible"])
    res["pass"] = ok
    YC.jdump(res, a.out)
    print(json.dumps(res))
    print("ZZAXID_OK" if ok else "ZZAXID_RED")
    return 0 if ok else 3


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--final", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--base", default=str(YC.U400))
    i = sp.add_parser("identity")
    i.add_argument("--ckpt", required=True)
    i.add_argument("--final", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--frames", type=int, default=512)
    i.add_argument("--base", default=str(YC.U400))
    i.add_argument("--cache", default=str(HERE.parent / "YF2" / "cache_u400_4k"))
    i.add_argument("--manifest", default=str(HERE.parent / "YF2" / "manifest_yaw_4k.json"))
    a = ap.parse_args()
    return cmd_build(a) if a.cmd == "build" else cmd_identity(a)


if __name__ == "__main__":
    sys.exit(main())
