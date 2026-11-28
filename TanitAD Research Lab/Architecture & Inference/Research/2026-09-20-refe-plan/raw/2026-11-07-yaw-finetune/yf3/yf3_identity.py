#!/usr/bin/env python3
"""YF3 IDENTITY GATES on a composed checkpoint (CPU, CUDA_VISIBLE_DEVICES=-1), independent of the trainer's YawDecoder path:
  kind C : the checkpoint's FULL state_dict is torch.equal to u400's and its meta has no ego_yaw key (R1: C == A0 bit for bit, MM gate)  -> C_state_dict_equals_A0
           (R2 / R3: C trains the rest of ego_enc / the decoder with yaw 0, so equality with u400 is not expected; reported as such, not gated)
  kind B : G-CKPT -- the trained decode (trainer path: YawDecoder with the leaf col4) equals, torch.equal, the decode through the COMPOSED checkpoint's own modules (reloaded from the file on disk),
           on N VAL rows with ego[4] = 10 x omega(token); and the meta carries ego_yaw / scale 10.0 / slot 4 / the pinned train-table sha
  kind B0: "B before training": the composed checkpoint with col4 = 0 and everything else u400 decodes, torch.equal, identically to A0's deployed decode (ego[4] = 0), both with omega = 0 and omega != 0
    python yf3_identity.py --kind B|C|B0 --ckpt <model.pt> [--final <arm>/final.pt] --out <json> [--frames 512]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402

sys.path.insert(0, str(C.Y2))
import torch  # noqa: E402
import yaw_common as YC  # noqa: E402
import yaw_ft as FT  # noqa: E402


def load_ckpt_decoder(path):
    """decoder-side REFe holding the DEC tensors of the checkpoint FILE (not the trainer's in-memory copy)"""
    model, cfg, _ = FT.build_decoder(str(C.U400))
    sd = torch.load(path, map_location="cpu", weights_only=False)
    want = {k: v for k, v in sd["model"].items() if k.startswith(YC.DEC)}
    model.load_state_dict(want, strict=False)
    return model, cfg, sd


def val_batch(n):
    cache = FT.Cache(C.CACHE)
    rows = FT.load_rows(cache, C.jload(C.MANIFEST))["VAL"]
    rows = rows[:n]
    table, _, _ = YC.load_yaw_table(C.TABLE_TRAIN)
    FT.check_rows(rows, table)
    ids = list(range(len(rows)))
    return FT.make_batch(rows, ids, cache, table, "table", C.YAW_SCALE), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("B", "C", "B0"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--regress", choices=("none", "default_col"), default="none", help="TESTS ONLY: B0 with u400's random column 4 left in place (must be RED)")
    a = ap.parse_args()
    base_sd = torch.load(C.U400, map_location="cpu", weights_only=False)
    base = base_sd["model"]
    res = {"kind": a.kind, "ckpt": a.ckpt, "frames_requested": a.frames}
    (sc, eg, go, tg), n = val_batch(a.frames)
    res["n_frames"] = n
    eg0 = eg.clone()
    eg0[:, YC.EGO_YAW_SLOT] = 0.0
    omodel, cfg, _ = FT.build_decoder(str(C.U400))
    with torch.no_grad():
        a0 = FT.original_decode(omodel, cfg, sc, eg0, go)                                  # A0 as deployed: ego[4] = 0
    if a.kind == "C":
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        st = sd["model"]
        fin = torch.load(a.final, map_location="cpu", weights_only=False)
        res["rung"] = fin.get("rung")
        res["C_state_dict_equals_A0"] = bool(set(st) == set(base) and all(torch.equal(st[k], base[k]) for k in st))
        res["meta_has_no_ego_yaw"] = not any(k in (sd.get("meta") or {}) for k in ("ego_yaw", "ego_yaw_scale", "ego_yaw_slot", "ego_yaw_table_sha256"))
        m, cfg2, _ = load_ckpt_decoder(a.ckpt)
        with torch.no_grad():
            c_ck = FT.original_decode(m, cfg2, sc, eg0, go)
            res["decode_equals_A0_yaw0"] = bool(torch.equal(a0, c_ck))
            # the trainer's own path (YawDecoder, yaw 0) vs the composed checkpoint's modules: the G-CKPT leg for arm C
            tm, tcfg, _ = FT.build_decoder(str(C.U400))
            tm.load_state_dict({k: v for k, v in fin["state"].items() if k.startswith(YC.DEC)}, strict=False)
            res["G_CKPT_trainer_decode_equals_composed_ckpt_decode"] = bool(torch.equal(FT.YawDecoder(tm, tcfg, fin["col4"])(sc, eg0, go), c_ck))
        if fin.get("rung") == "R1":
            res["ok"] = bool(res["C_state_dict_equals_A0"] and res["meta_has_no_ego_yaw"] and res["decode_equals_A0_yaw0"])
            res["note"] = "R1: C == A0 is GATED (state_dict torch.equal to u400)"
        else:
            res["ok"] = bool(res["meta_has_no_ego_yaw"] and res["G_CKPT_trainer_decode_equals_composed_ckpt_decode"])
            res["note"] = "R2/R3: C trains the rest with yaw 0, equality with u400 is not expected and not gated; G-CKPT and 'no ego_yaw meta' are"
    elif a.kind == "B0":
        # B before training: the u400 decoder with column 4 of ego_enc.0.weight replaced by zeros (the zero-init), composed exactly as build_yaw_ckpt composes a B checkpoint
        m, cfg2, _ = FT.build_decoder(str(C.U400))
        with torch.no_grad():
            if a.regress != "default_col":
                m.ego_enc[0].weight[:, YC.EGO_YAW_SLOT] = 0.0
            res["B0_equals_A0_yaw0"] = bool(torch.equal(a0, FT.original_decode(m, cfg2, sc, eg0, go)))
            res["B0_equals_A0_yaw_nonzero(10 x omega fed)"] = bool(torch.equal(a0, FT.original_decode(m, cfg2, sc, eg, go)))
        res["yaw_nonzero_frames"] = int((eg[:, YC.EGO_YAW_SLOT] != 0).sum())
        res["ok"] = bool(res["B0_equals_A0_yaw0"] and res["B0_equals_A0_yaw_nonzero(10 x omega fed)"] and res["yaw_nonzero_frames"] > 0)
    else:
        fin = torch.load(a.final, map_location="cpu", weights_only=False)
        m, cfg2, sd = load_ckpt_decoder(a.ckpt)
        meta = sd.get("meta") or {}
        # trainer path: the SAME base model for everything but the trained tensors, col4 from final.pt
        tm, tcfg, _ = FT.build_decoder(str(C.U400))
        tm.load_state_dict({k: v for k, v in fin["state"].items() if k.startswith(YC.DEC)}, strict=False)
        dec = FT.YawDecoder(tm, tcfg, fin["col4"])
        with torch.no_grad():
            t_train = dec(sc, eg, go)
            t_ckpt = FT.original_decode(m, cfg2, sc, eg, go)
        res["G_CKPT_trainer_decode_equals_composed_ckpt_decode"] = bool(torch.equal(t_train, t_ckpt))
        res["col4_in_ckpt_equals_final"] = bool(torch.equal(sd["model"]["ego_enc.0.weight"][:, YC.EGO_YAW_SLOT], fin["col4"].to(sd["model"]["ego_enc.0.weight"].dtype)))
        res["meta_ok"] = bool(meta.get("ego_yaw") == YC.YAW_TAG and float(meta.get("ego_yaw_scale", -1)) == 10.0 and int(meta.get("ego_yaw_slot", -1)) == YC.EGO_YAW_SLOT
                              and meta.get("ego_yaw_table_sha256") == C.TABLE_TRAIN_SHA)
        res["moved_from_A0_when_yaw_fed"] = bool(not torch.equal(t_ckpt, FT.original_decode(omodel, cfg, sc, eg0, go)))
        res["col4_norm"] = float(fin["col4"].norm())
        res["ok"] = bool(res["G_CKPT_trainer_decode_equals_composed_ckpt_decode"] and res["col4_in_ckpt_equals_final"] and res["meta_ok"])
    C.jdump(res, a.out)
    print(json.dumps(res))
    print("ZZYF3ID_OK" if res["ok"] else "ZZYF3ID_RED")
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
