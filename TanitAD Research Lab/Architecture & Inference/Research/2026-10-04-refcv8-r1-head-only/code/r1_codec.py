"""R1 codec study (DESIGN.md sec. 5 rule: "if only I-S fails, the codec is changed before any arm").

I-S FAILED its pre-stated p99-fan bar on the first 2,048 GPU eval windows (p99 fan |d| 0.078 m > 0.05 m; picks 4 /
2,048 within the 1 % bar; I-X exact 0.0). This study measures, on the same model and the same head path, how much of the
storage error each tensor's codec causes, so the replacement codec is chosen by measurement. Windows: a seeded (seed 1)
sample of the R1 scored eval windows. Per window: the FULL forward (capture), then the head path fed each variant;
seed pinned before every forward. Nothing is cached; prints and writes <out>/codec_study.json.

Decision rule (fixed BEFORE this study runs): among the variants whose stored bytes keep the R1 cache (11,634 scored +
train windows) <= 15 GB, choose the cheapest one with p99 fan |d| <= 0.05 m AND differing E9 picks <= 1 %. If none
qualifies, report the measured table and stop for a Master-Mind decision (no arm runs on a failed codec).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402

BYTES_FIXED = 32768 + 13200 + 2000          # pooled fp16 + slots + small keys (per window)
N_STORED = 11634                            # 4,634 scored eval + 7,000 train windows (SPEC_R1 sec. 2)


def q8_channel(x):
    q, lo, sc = L1.quant_u8(x)
    return L1.dequant_u8(q, lo, sc), x.shape[1] * x.shape[2] * x.shape[3] + 8 * x.shape[1]


def q8_row(x):
    """int8 affine per (channel, X row): lo/scale [B, C, X]."""
    import torch
    lo = x.amin(dim=3, keepdim=True)
    hi = x.amax(dim=3, keepdim=True)
    sc = ((hi - lo) / 255.0).clamp_min(1e-12)
    q = torch.round((x - lo) / sc).clamp(0, 255)
    return lo + q * sc, x.shape[1] * x.shape[2] * x.shape[3] + 8 * x.shape[1] * x.shape[2]


def q16_channel(x):
    import torch
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    sc = ((hi - lo) / 65535.0).clamp_min(1e-12)
    q = torch.round((x - lo) / sc).clamp(0, 65535)
    return lo + q * sc, 2 * x.shape[1] * x.shape[2] * x.shape[3] + 8 * x.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt")
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()
    import r1_heads as H
    from r1_capture import select_windows
    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device=a.device, strict=True)

    class R1DS(tr.V3Dataset):
        def __getitem__(self, i):
            item = super().__getitem__(i)
            e_i, t = self.index[i]
            item["map_ep"] = torch.tensor(int(self.episodes[e_i].episode_id), dtype=torch.long)
            item["r1_i"] = torch.tensor(int(i), dtype=torch.long)
            return item
    e_ds, e_eps, _ = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=False, dataset_cls=R1DS)
    from tanitad.data import v2_dataset as v2
    m_ = v2.load_or_build_manifest(args.eval_cache, verbose=False)
    sha_of_eid = {int(v2.stable_episode_id(str(c))): L1.sha12(str(c)) for c in m_["clip_id"]}
    idx, info, _gt = select_windows(e_ds, "eval", sha_of_eid, 0, 10, 5)
    scored = [i for i, r in zip(idx, info) if ("turn" in r["flags"] or "sample" in r["flags"])]
    rng = np.random.default_rng(1)
    pick = sorted(rng.choice(len(scored), min(a.n, len(scored)), replace=False))
    win = [scored[j] for j in pick]
    CAP = {"phase": None}
    # ⛔ forward hooks must RETURN NONE: a hook that returns a value REPLACES the module's output (a lambda around
    # dict.setdefault did exactly that on the first run and fed pooled_seq in place of the strategic ctx)
    def h_out(_m, _i, o):
        CAP.setdefault("out_" + str(CAP["phase"]), o)

    def h_pool(_m, i, _o):
        if CAP["phase"] == "full":
            CAP.setdefault("pooled", i[0].detach().clone())

    def h_kv(_m, _i, o):
        if CAP["phase"] == "full":
            CAP.setdefault("kv", o.detach().clone())

    def h_perc(_m, _i, o):
        if CAP["phase"] == "full":
            CAP.setdefault("pout", o)

    def h_slots(_m, _i, o):
        if CAP["phase"] == "full":
            CAP.setdefault("slots", o)
    model.register_forward_hook(h_out)
    model.core.strategic.register_forward_hook(h_pool)
    model.core.decoder.feat_proj.register_forward_hook(h_kv)
    model._perception.register_forward_hook(h_perc)
    model.core.agent_head.register_forward_hook(h_slots)
    feed = H.Feed()

    def run(name, b):
        CAP["phase"] = name
        torch.manual_seed(a.seed)
        with torch.no_grad():
            H.forward_like_trainer(model, b, a.device, args.mode, tr)
        CAP["phase"] = None
        o = CAP["out_" + name]
        return {"fan": o["anchor_traj"].float(), "sel": o["sel_idx"], "s_e9": o["sel_score_v3"].float(),
                "p_lat": o["tacv6_lat_logits"].float().softmax(-1)}
    variants = ("exact", "kv16", "bev_q8c", "bev_q8row", "bev_q16c", "bev_fp16", "slots16",
                "prod_q8c", "cand_q8row", "cand_q16c", "cand_fp16")
    rows = {v: [] for v in variants}
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, win), batch_size=1, shuffle=False, num_workers=2)
    t0 = time.time()
    for k, b in enumerate(dl):
        CAP.clear()
        CAP["phase"] = None
        f = run("full", b)
        kv, pooled, bev, slots = CAP["kv"], CAP["pooled"], CAP["pout"]["bev_feats"].detach(), H.slots_to_cache(CAP["slots"])
        s16 = {x: (v.float() if x == "presence_logit" else v.half().float()).contiguous() for x, v in slots.items()}
        b8c, _ = q8_channel(bev)
        b8r, _ = q8_row(bev)
        b16, _ = q16_channel(bev)
        bh = bev.half().float()
        cfgs = {"exact": (kv, bev, slots), "kv16": (kv.half().float(), bev, slots), "bev_q8c": (kv, b8c, slots),
                "bev_q8row": (kv, b8r, slots), "bev_q16c": (kv, b16, slots), "bev_fp16": (kv, bh, slots),
                "slots16": (kv, bev, s16), "prod_q8c": (kv.half().float(), b8c, s16),
                "cand_q8row": (kv.half().float(), b8r, s16), "cand_q16c": (kv.half().float(), b16, s16),
                "cand_fp16": (kv.half().float(), bh, s16)}
        saved = H.install(model, feed)
        try:
            for v, (kv_, bev_, sl_) in cfgs.items():
                feed.cur = {"kv": kv_.contiguous(), "pooled": pooled.half().float().contiguous() if v != "exact" else pooled,
                            "bev": bev_.contiguous(), "slots": sl_, "bev_pad": None}
                s = run("v_" + v, b)
                rows[v].append({"fan": float((s["fan"] - f["fan"]).abs().max()),
                                "s_e9": float((s["s_e9"] - f["s_e9"]).abs().max()),
                                "p_lat": float((s["p_lat"] - f["p_lat"]).abs().max()),
                                "sel_eq": bool(int(s["sel"]) == int(f["sel"]))})
        finally:
            H.uninstall(model, saved)
        if k % 16 == 0:
            print(f"[codec] {k + 1}/{len(win)} {time.time() - t0:.0f}s", flush=True)
    bev_bytes = {"bev_q8c": 96 * 120 * 64 + 8 * 96, "bev_q8row": 96 * 120 * 64 + 8 * 96 * 120,
                 "bev_q16c": 2 * 96 * 120 * 64 + 8 * 96, "bev_fp16": 2 * 96 * 120 * 64}
    out = {"n_windows": len(win), "device": a.device, "seed": a.seed, "variants": {}}
    for v, rr in rows.items():
        fan = np.array([r["fan"] for r in rr])
        out["variants"][v] = {"n_sel_differ": int(sum(not r["sel_eq"] for r in rr)),
                              "fan_p99": float(np.percentile(fan, 99)), "fan_max": float(fan.max()),
                              "fan_p50": float(np.percentile(fan, 50)),
                              "s_e9_p99": float(np.percentile([r["s_e9"] for r in rr], 99)),
                              "p_lat_max": float(max(r["p_lat"] for r in rr))}
    for v, bb in (("prod_q8c", "bev_q8c"), ("cand_q8row", "bev_q8row"), ("cand_q16c", "bev_q16c"),
                  ("cand_fp16", "bev_fp16")):
        per = 416 * 384 * 2 + bev_bytes[bb] + BYTES_FIXED
        out["variants"][v]["bytes_per_window"] = per
        out["variants"][v]["cache_GB_for_11634"] = round(per * N_STORED / 1e9, 2)
    ok = [v for v in ("prod_q8c", "cand_q8row", "cand_q16c", "cand_fp16")
          if out["variants"][v]["cache_GB_for_11634"] <= 15.0 and out["variants"][v]["fan_p99"] <= 0.05
          and out["variants"][v]["n_sel_differ"] <= 0.01 * len(win)]
    out["qualifying_in_budget"] = ok
    out["choice"] = (min(ok, key=lambda v: out["variants"][v]["bytes_per_window"]) if ok else None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "codec_study.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
