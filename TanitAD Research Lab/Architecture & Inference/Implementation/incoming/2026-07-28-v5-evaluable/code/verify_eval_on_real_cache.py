"""⭐ THE EVALUATOR'S V2 PATH, PROVEN ON THE REAL pod2 CACHES — not on fixtures.

Runs on pod2 against the two REGISTERED v5 caches. Six legs, each able to
report a different failure:

  L1  precondition   the val cache is PNG at 256x640 and REGISTERED (a lossy or
                     unregistered cache would make every leg below meaningless)
  L2  the seam       `eval_flagship_v4.build_v2_val_episodes` hands back the
                     SUB-frame; the encoder's token count moves with it
  L3  vs the loader  bit-identical to the same loader with NO frame, sliced by
                     hand — the slice CLAIM itself, on real PNG bytes
  L4  vs the trainer the evaluator and the trainer produce the IDENTICAL tensors
                     for the same clip (they must, or the gate is not scoring
                     what was trained)
  L5  ⛔ NEG         the frame withheld from the loader -> REFUSED
  L6  ⛔ NEG         the `ego=` shape: a run that trained at 176x624 scored
                     without --v2-subframe -> REFUSED, naming the flag value

⛔ NOTHING IS WRITTEN to either cache. The registered dirs are opened read-only;
the negative legs monkeypatch in-process.

🔒 No clip id is printed or stored.

usage:  PYTHONPATH=<stack> python3 verify_eval_on_real_cache.py \
            --val <cache dir> --sub 176x624 --n 6 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="/workspace/v5eval/stack")
    ap.add_argument("--val", required=True)
    ap.add_argument("--sub", default="176x624")
    ap.add_argument("--n", type=int, default=6, help="clips to compare")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    sys.path.insert(0, str(Path(a.stack) / "scripts"))
    import torch

    from tanitad.config import flagship4b_config
    from tanitad.data import parity
    import tanitad.data.v2_dataset as V2
    import eval_flagship_v4 as E
    import train_flagship_v4 as T

    out: dict = {"host": "pod2", "val_cache": a.val, "subframe": a.sub,
                 "evidence_class": "MEASURED (ours; artifact = this JSON)"}

    # ---------------- L1: precondition ------------------------------------- #
    files = sorted(Path(a.val).glob("*.v2ep.pt"))
    probe = torch.load(str(files[0]), map_location="cpu", weights_only=False)
    key = parity.corpus_key_of(a.val)
    ent = parity.manifest_entry(key) if key else None
    out["L1_precondition"] = {
        "n_clips": len(files),
        "codec": str(probe.get("codec")),
        "raster": [int(probe["image_h"]), int(probe["image_w"])],
        "stored_frame": probe.get("frame"),
        "corpus_key_resolved": key,
        "registered": ent is not None,
        "registered_episode_count": (ent or {}).get("episode_count"),
        "registered_geometry_observed_frac": (
            (((ent or {}).get("provenance") or {}).get("geometry") or {})
            .get("geometry_check", {}).get("observed_frac")),
    }
    del probe

    argv = ["--ckpt", "x", "--key", "k", "--out", "o",
            "--v2-val-cache", a.val,
            "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
            "--projection", "cylindrical", "--v2-subframe", a.sub,
            "--require-parity"]
    args = _parse(argv)

    # ---------------- L2: the seam ----------------------------------------- #
    cfg = flagship4b_config()
    cache_frame, model_frame = E.resolve_eval_frames(args, cfg)
    eps, prov = E.build_v2_val_episodes(args, cache_frame=cache_frame,
                                        train_frame=model_frame, verbose=True)
    gh, gw = cfg.encoder.token_grid()
    out["L2_seam"] = {
        "n_providers": len(eps),
        "provider_shape": [int(x) for x in eps[0].frames.shape],
        "encoder_hw": list(cfg.encoder.image_hw()),
        "token_grid": [gh, gw], "n_tokens": gh * gw,
        "cache_frame": cache_frame.to_dict(), "model_frame": model_frame.to_dict(),
        "sliced_from": prov["geometry_binding"].get("sliced_from"),
        "val_parity": {k: prov["val_parity"].get(k) for k in
                       ("corpus_key", "parity", "episodes_loaded",
                        "episode_uid_sha256", "content_check")},
    }

    # ---------------- L3: vs the same loader, sliced by hand ---------------- #
    rs = prov["geometry_binding"]["sliced_from"]["rows"]
    cs = prov["geometry_binding"]["sliced_from"]["cols"]
    ref = V2.build_v2_providers(a.val, verbose=False)
    n = min(a.n, len(eps))
    worst, rows = 0, 0
    for i in range(n):
        T_out = int(eps[i].frames.shape[0])
        got = eps[i].frames[0:T_out]
        want = ref[i].frames[0:T_out][:, :, rs[0]:rs[1], cs[0]:cs[1]]
        assert got.shape == want.shape, (got.shape, want.shape)
        d = int((got.int() - want.int()).abs().max())
        worst = max(worst, d)
        rows += T_out
    out["L3_vs_hand_slice"] = {"clips": n, "stacked_rows": rows,
                               "max_abs_diff": worst,
                               "bit_identical": worst == 0}

    # ---------------- L4: vs the TRAINER's own seam ------------------------- #
    targv = ["--v2-train-cache", a.val, "--v2-val-cache", a.val,
             "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
             "--projection", "cylindrical", "--v2-subframe", a.sub,
             "--from-scratch"]
    ta = T.build_parser().parse_args(targv)
    tcfg = flagship4b_config()
    tcache, tmodel = T.resolve_v2_frames(ta, tcfg)
    t_train, _ = T.build_v2_data(ta, {"train_parity": {}, "val_parity": {}},
                                 cache_frame=tcache, train_frame=tmodel,
                                 verbose=False)
    worst4 = 0
    for i in range(n):
        T_out = int(eps[i].frames.shape[0])
        worst4 = max(worst4, int((eps[i].frames[0:T_out].int()
                                  - t_train[i].frames[0:T_out].int())
                                 .abs().max()))
    out["L4_vs_trainer_seam"] = {
        "clips": n, "max_abs_diff": worst4, "identical": worst4 == 0,
        "trainer_frame": tmodel.to_dict(), "eval_frame": model_frame.to_dict(),
        "frames_agree": tmodel == model_frame}

    # ---------------- L5: NEG — the frame withheld from the loader ---------- #
    real = V2.build_v2_providers
    V2.build_v2_providers = lambda dirs, **kw: real(
        dirs, **{k: v for k, v in kw.items() if k != "frame"})
    try:
        E.build_v2_val_episodes(args, cache_frame=cache_frame,
                                train_frame=model_frame, verbose=False)
        out["L5_neg_loader_not_told"] = {"refused": False,
                                         "VERDICT": "GUARD IS INERT"}
    except parity.ParityViolation as ex:
        m = str(ex)
        out["L5_neg_loader_not_told"] = {
            "refused": True,
            "names_the_case": "SUB-FRAME WAS DECLARED BUT NEVER APPLIED" in m,
            "names_the_call": "build_v2_providers" in m}
    finally:
        V2.build_v2_providers = real

    # ---------------- L6: NEG — the `ego=` shape ---------------------------- #
    run_cfg = {"geometry": model_frame.report(),
               "geometry_cache": {**cache_frame.report()}}
    try:
        parity.assert_eval_frame_matches_run(run_cfg, cache_frame,
                                             label="--ckpt vs eval frame",
                                             cache_frame=cache_frame)
        out["L6_neg_scored_on_parent"] = {"refused": False,
                                          "VERDICT": "GUARD IS INERT"}
    except parity.ParityViolation as ex:
        m = str(ex)
        out["L6_neg_scored_on_parent"] = {
            "refused": True,
            "names_the_case": "WAS NOT TRAINED ON" in m,
            "names_the_flag_value": f"--v2-subframe {a.sub}" in m}
    # and the POSITIVE twin, so L6 is not a one-sided assertion
    ok = parity.assert_eval_frame_matches_run(run_cfg, model_frame,
                                              label="--ckpt vs eval frame",
                                              cache_frame=cache_frame)
    out["L6_pos_matching_frame"] = {"checked": ok["checked"],
                                    "trained_frame_tag": ok.get("trained_frame_tag"),
                                    "cache_frame_matches_run":
                                        ok.get("cache_frame_matches_run")}

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def _parse(argv):
    """Build eval_flagship_v4's parser surface without invoking main()."""
    import argparse as _ap

    import goal_modes
    from tanitad.geometry import add_geometry_args
    p = _ap.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-cache", default=None)
    p.add_argument("--v2-val-cache", default=None, nargs="+")
    p.add_argument("--v2-lru", type=int, default=64)
    p.add_argument("--v2-subframe", default=None)
    p.add_argument("--require-parity", action="store_true")
    add_geometry_args(p)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--goal-mode", choices=goal_modes.GOAL_MODES, default="oracle")
    p.add_argument("--select-rule", default="as-trained")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
