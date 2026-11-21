"""WP-D s4 — exercise the trainer wiring WITHOUT a GPU: refusals, one loss step.

⛔ Every refusal is asserted to FIRE, and the happy path is asserted to produce a
NON-ZERO loss with a NON-ZERO supervised count. A guard that is present and
never reachable is the defect this programme has logged four times in one night;
a happy path that reports success on an all-zero target is the poisoned-bank
trap. Both are checked here, on CPU, before any GPU-day is requested.
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
from pathlib import Path


def load(path: str):
    spec = iu.spec_from_file_location("refc_v3_train_wpd", path)
    m = iu.module_from_spec(spec)
    sys.modules["refc_v3_train_wpd"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    sys.argv = ["refc_v3_train"]
    T = load(a.trainer)
    import torch
    from tanitad.data import bev_aux as BA
    from tanitad.refs import refc_bev_aux as RB

    P = T.build_parser()
    res = {"refusals": [], "flags": [], "happy": {}}

    def parse(extra):
        return P.parse_args(["--out", "_x", "--arm", "hier"] + extra)

    # ---- 0. the flags exist and default OFF/inert --------------------------
    d = parse([])
    for k, v in (("bev_aux", "off"), ("w_bev_aux", 0.0),
                 ("bev_aux_occlusion", "mask"), ("bev_aux_detach", False),
                 ("bev_aux_rng", 24), ("bev_aux_rmax", 60.0),
                 ("bev_aux_pos_weight", 30.61)):
        got = getattr(d, k)
        assert got == v, (k, got, v)
        res["flags"].append({"flag": k, "default": got})

    cfg0 = T.v3.refc_v3_config() if hasattr(T.v3, "refc_v3_config") else None

    def fresh_cfg():
        import copy
        c = T.v3.RefCV3Config()
        return copy.deepcopy(c)

    # ---- 1. the refusals must FIRE ----------------------------------------
    cases = [
        ("bev_aux_on_zero_weight",
         ["--bev-aux", "col", "--w-bev-aux", "0"], "ZERO gradient"),
        ("bev_aux_on_no_join",
         ["--bev-aux", "col", "--w-bev-aux", "0.1"], "NO LABELS"),
        ("weight_without_head",
         ["--bev-aux", "off", "--w-bev-aux", "0.5"], "SILENTLY SKIPPED"),
    ]
    for name, extra, needle in cases:
        args = parse(extra)
        cfg = fresh_cfg()
        try:
            T._pin_refcv5_seams(cfg, args)
        except SystemExit as e:
            msg = str(e)
            assert needle in msg, (name, msg[:400])
            res["refusals"].append({"case": name, "fired": True,
                                    "needle": needle})
            continue
        raise AssertionError(f"{name}: the refusal did NOT fire")

    # ---- 2. the happy path builds a head and a real loss -------------------
    args = parse(["--bev-aux", "col", "--w-bev-aux", "0.1",
                  "--agent-join", "SOME/join.jsonl.xz"])
    cfg = fresh_cfg()
    cfg.core.encoder.image_size = 64
    cfg.core.encoder.image_width = 640
    T._pin_refcv5_seams(cfg, args)
    assert cfg.core.bev_aux is not None and cfg.core.bev_aux.enable
    assert cfg.core.bev_aux.n_az == cfg.core.encoder.grid_shape[1] == 20
    res["happy"]["bev_aux_cfg"] = cfg.core.bev_aux.to_dict()

    # a real head, a real target, a real backward
    torch.manual_seed(0)
    head = RB.BEVAuxHead(48, (8, 20), cfg.core.bev_aux)
    spec = BA.PolarBEVSpec(n_az=20, n_rng=cfg.core.bev_aux.n_rng)
    scene = [{"cx": 10.0, "cy": 0.0, "yaw": 0.0, "l": 12.0, "w": 2.6},
             {"cx": 30.0, "cy": 5.0, "yaw": 0.0, "l": 4.5, "w": 2.0}]
    occ, msk = BA.build_target(scene, spec=spec, occlusion="mask")
    # ⚠️ CONTENT ASSERTION on the target before it is used for anything.
    assert float(occ.sum()) > 0.0, "the target is all zeros — POISONED"
    assert bool(msk.any()) and not bool(msk.all()), "the mask is degenerate"
    trunk = torch.nn.Conv2d(9, 48, 3, padding=1)
    fmap = trunk(torch.randn(2, 9, 8, 20))
    o = torch.from_numpy(occ)[None].repeat(2, 1, 1)
    m = torch.from_numpy(msk)[None].repeat(2, 1, 1)
    out = RB.bev_aux_loss(head(fmap), o, m, cfg.core.bev_aux.pos_weight)
    assert out["n_supervised"] > 0 and out["n_pos"] > 0
    out["loss"].backward()
    g = sum(float(p.grad.abs().sum()) for p in trunk.parameters()
            if p.grad is not None)
    assert g > 0.0, "no gradient reached the trunk"
    res["happy"].update(
        loss=float(out["loss"]), n_supervised=out["n_supervised"],
        n_pos=out["n_pos"], base_rate=out["base_rate"],
        trunk_grad_absnorm=g, head_params=head.n_params(),
        target_occ_sum=float(occ.sum()),
        target_frac_ignored=float((~msk).mean()))

    # ---- 3. the parity guard, on the real numbers --------------------------
    par = RB.assert_loss_parity(0.1, float(out["loss"]), 0.5)
    res["happy"]["loss_parity"] = par

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"[s4] ALL CHECKS PASSED — wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
