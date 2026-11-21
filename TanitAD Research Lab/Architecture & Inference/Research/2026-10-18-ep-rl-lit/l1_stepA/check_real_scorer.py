#!/usr/bin/env python3
"""Evidence script (CPU, read-only): (1) P2-s0's real scorer loads strictly and answers through the LANDED score_trajectories; (2) the SET-DEPENDENCE premise (prereg G2 control) on
real weights; (3) CPU cost per scored set; (4) the banked dump stores the REPAIRED plan only (last heading), so the raw scorer input must be re-captured; (5) trunk fingerprints P2 vs base
(a visual_ctx cache of the base trunk is invalid for P2-s0); (6) the one on-disk visual_ctx cache (M5) vs the fit universe.
    python check_real_scorer.py --refe-dir <refe> --p2 <p2/model_final.pt> --base <model_final.pt> --proposals <proposals.npz> --m5 <manifest.json,...> --tokens <txt> --out <json>"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe-dir", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--m5", nargs="*", default=[])
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import real_scorer as RS
    import stepA as A
    out = {}
    z = np.load(a.proposals, allow_pickle=True)
    p20 = z["proposals20"]
    out["dump_last_two_headings_equal_everywhere"] = bool(np.array_equal(p20[..., -1, 2], p20[..., -2, 2]))
    out["dump_shape"] = list(p20.shape)
    ns, info = RS.load_scorer(a.p2, a.refe_dir, "cpu")
    out["scorer_info"] = info
    fan = torch.from_numpy(np.ascontiguousarray(p20[0], np.float32))[None]
    g = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, 7680, 256, generator=g) * 0.5
    t0 = time.time()
    with torch.no_grad():
        full = ns.score_trajectories(fan, ctx)
    out["cpu_seconds_first_set"] = round(time.time() - t0, 2)
    t0 = time.time()
    with torch.no_grad():
        full = ns.score_trajectories(fan, ctx)
        keep = [i for i in range(64) if i != 17]
        part = ns.score_trajectories(fan[:, keep], ctx)
    out["cpu_seconds_two_sets"] = round(time.time() - t0, 2)
    out["set_dependence_max_abs_change_on_other_63"] = float((full[0, keep] - part[0]).abs().max())
    out["logits_shape"] = list(full.shape)
    # mutation check on the real path: perturbing ONE proposal changes the scorer output of THAT proposal
    f2 = fan.clone()
    f2[0, 5, :, 0] += 1.0
    with torch.no_grad():
        o2 = ns.score_trajectories(f2, ctx)
    out["moving_proposal_5_by_1m_changes_its_own_logits"] = float((o2[0, 5] - full[0, 5]).abs().max())
    out["moving_proposal_5_changes_others_via_set_attention"] = float((o2[0, [i for i in range(64) if i != 5]] - full[0, [i for i in range(64) if i != 5]]).abs().max())
    # trunk fingerprints
    spec_dir = a.refe_dir
    sys.path.insert(0, spec_dir)
    import ctx_cache as CC
    fps = {}
    for name, path in (("p2_s0", a.p2), ("base_model_final", a.base)):
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        fps[name] = CC.trunk_fingerprint(ck["model"])
    out["trunk_fingerprint"] = fps
    out["trunk_fingerprint_equal"] = fps["p2_s0"] == fps["base_model_final"]
    # the M5 cache vs the fit universe
    ho = set(open(a.tokens).read().split())
    m5 = {}
    for p in a.m5:
        m = json.load(open(p))
        keys = [k.split("|") for k in m["keys"]]
        m5[os.path.relpath(p, os.path.dirname(os.path.dirname(p)))] = {"checkpoint": m.get("checkpoint"), "checkpoint_md5": m.get("checkpoint_md5"), "n": m["n"], "tokens_in_fit_universe": sum(1 for k in keys if k[1] in ho)}
    out["m5_cache"] = m5
    json.dump(out, open(a.out, "w"), indent=1)
    print("ZZREALCHECK_DONE", json.dumps({k: out[k] for k in out if k not in ("scorer_info", "m5_cache")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
