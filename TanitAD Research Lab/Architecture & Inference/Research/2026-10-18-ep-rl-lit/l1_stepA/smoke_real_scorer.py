#!/usr/bin/env python3
"""CPU smoke of the REAL scoring path (`stepA.py score` CLI -> real_scorer -> the landed REFe.score_trajectories on P2-s0's weights) on a SYNTHETIC bundle: real P2 fans (the banked,
repaired proposals20 with a synthetic corrupted last heading standing in for the raw one) and a RANDOM visual_ctx.  It proves the plumbing and the gates G2a / G2c / G3 on real weights and
measures the CPU cost; it proves NOTHING about scorer accuracy (the ctx is noise).
    python smoke_real_scorer.py --refe-dir <refe> --ckpt <p2/model_final.pt> --proposals <proposals.npz> --out <json> [--tokens 2]"""
import argparse, json, os, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stepA as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=2)
    a = ap.parse_args()
    z = np.load(a.proposals, allow_pickle=True)
    n = a.tokens
    fans = np.array(z["proposals20"][:n], np.float32)
    fans[..., -1, 2] = np.random.default_rng(0).uniform(-3.5, 3.5, fans[..., -1, 2].shape)          # a corrupted raw last heading (A7), synthetic
    rng = np.random.default_rng(1)
    ctx = (rng.standard_normal((n, 7680, 256)) * 0.5).astype(np.float32)
    bits = np.ascontiguousarray(ctx.view(np.uint32) >> 16).astype(np.uint16)                          # float32 -> bf16 bits (truncation)
    d = tempfile.mkdtemp(prefix='l1smoke_')                                                           # not TemporaryDirectory: a memmap keeps ctx.npy open on Windows
    if True:
        np.save(os.path.join(d, "ctx.npy"), bits)
        np.savez(os.path.join(d, "b.npz"), token=np.array([str(t) for t in z["token"][:n]]), log=np.array([str(t) for t in z["log"][:n]]), traj_raw=fans, logits=np.zeros((n, 64, 6), np.float32),
                 pick=np.zeros(n, np.int64), sub=np.zeros((n, 64, 6)), pdms=np.zeros((n, 64)), valid=np.ones((n, 64), bool))
        # logits must be the scorer's own for choose_anchors: score once with the real path, then perturb from those
        import real_scorer
        sf = real_scorer.make_score_fn(a.ckpt, os.path.join(d, "ctx.npy"), np.arange(n), "cpu", a.refe_dir, "none")
        lg = np.zeros((n, 64, 6), np.float32)
        for i in range(n):
            sf.bind(i)
            lg[i] = sf(fans[i])
        np.savez(os.path.join(d, "b.npz"), token=np.array([str(t) for t in z["token"][:n]]), log=np.array([str(t) for t in z["log"][:n]]), traj_raw=fans, logits=lg,
                 pick=np.argmax(A.aggregate(lg), 1), sub=np.zeros((n, 64, 6)), pdms=np.zeros((n, 64)), valid=np.ones((n, 64), bool))
        t0 = time.time()
        assert A.main(["perturb", "--bundle", os.path.join(d, "b.npz"), "--out", os.path.join(d, "p.npz")]) == 0
        rc = A.main(["score", "--bundle", os.path.join(d, "b.npz"), "--perts", os.path.join(d, "p.npz"), "--out", os.path.join(d, "s.npz"), "--ckpt", a.ckpt, "--ctx", os.path.join(d, "ctx.npy"),
                     "--device", "cpu", "--refe-dir", a.refe_dir, "--gate-tokens", str(n)])
        sec = time.time() - t0
        gates = json.load(open(os.path.join(d, "s.npz.gates.json")))
        s = A.load_npz(os.path.join(d, "s.npz"))
        pe = A.load_npz(os.path.join(d, "p.npz"))
        moved = float(np.mean(np.abs(s["p_logit"] - s["a_logit"][np.arange(n)[:, None], pe["anchor"]]).max(-1) > 0))
    out = {"rc": rc, "tokens": n, "sets_scored": int(n * (1 + pe["anchor"].shape[1])), "seconds": round(sec, 1), "seconds_per_set": round(sec / (n * (1 + pe["anchor"].shape[1]) + 3 * n), 3),
           "G2a_all_pass": all(g["passed"] for g in gates["G2a"]), "G3_all_pass": all(g["passed"] for g in gates["G3"]), "G2c_premise_true": [g["premise_true"] for g in gates["G2c"]],
           "G2c_max_change": [g["max_abs_change_on_other_63"] for g in gates["G2c"]], "share_of_perturbed_rows_whose_logits_differ_from_the_anchor": moved,
           "NOTE": "random visual_ctx: plumbing + gates only, no accuracy claim"}
    json.dump(out, open(a.out, "w"), indent=1)
    print("ZZSMOKE", json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
