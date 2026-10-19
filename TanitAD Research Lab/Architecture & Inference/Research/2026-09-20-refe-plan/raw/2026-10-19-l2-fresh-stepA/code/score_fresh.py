#!/usr/bin/env python3
"""L2-FRESH-STEPA step 6c: EP scores of P2's head and of BOTH final L2 heads on the fresh run's exported EP-head input tokens, plus C-FEAT.  Reads NO label; computes NO A-quantity.

Inputs (run dir W): feats/feats.npz + feats/p2_score_head.npz (merge_feats.py), scores.npz (step 6: P2's banked logits of THIS run), bundle/bundle.npz, perts.npz.
C-FEAT (the L2 prereg section 2 gate, on the fresh run): P2's EP head applied to the exported tokens reproduces step 6's EP scores; max abs diff <= 1e-4 on >= 99 % of rows, in-fan and perturbed;
the shards were made from THIS bundle/perts (sha256).  A failed C-FEAT exits 3 and writes no score file.
Writes W/fresh_ep_scores.npz {ep_in_<arm> [N,64], ep_p_<arm> [N,Q]} for arm in P2, L2_s0, L2_s1, and W/fresh_ep_scores.json (the gate record + sha256 of every input and of the heads).

    python score_fresh.py --run-dir <W> --heads <heads dir with final_head_s{0,1}.npz + final_heads.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
import l2lib  # noqa: E402
import stepA  # noqa: E402  (via l2lib's L1 path)

EP = l2lib.EP


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def c_feat(tok_in, tok_p, W, b, a_logit, p_logit, tol=1e-4, share_req=0.99) -> dict:
    """P2's EP head on the exported tokens vs step 6's banked EP scores.  Pure (tested, with a RED mutation)."""
    rows, ok = {}, True
    for name, tok, banked in (("in_fan", tok_in, a_logit), ("perturbed", tok_p, p_logit)):
        d = np.abs(l2lib.ep_apply(tok, W, b) - np.asarray(banked, np.float64)[..., EP])
        share = float((d <= tol).mean())
        rows[name] = {"n_rows": int(d.size), "max_abs": float(d.max()), "share_le_tol": share, "p99_abs": float(np.quantile(d, 0.99))}
        ok = ok and share >= share_req
    return {"gate": "C-FEAT", "tolerance": tol, "required_share": share_req, "rows": rows, "pass": bool(ok)}


def check_head(name, Wh, bh, W2, b2):
    """a final head must differ from P2 ONLY in the EP row (rows 0,1,3,4,5 bit-identical) AND must actually differ there (a head equal to P2 would score the 'L2' arm as P2: the built-but-unreachable class)"""
    frozen = [r for r in range(6) if r != EP]
    if not (np.array_equal(Wh[frozen], W2[frozen]) and np.array_equal(bh[frozen], b2[frozen])):
        raise SystemExit(f"final head {name}: a frozen row differs from P2's")
    if np.array_equal(Wh[EP], W2[EP]) and np.array_equal(bh[EP], b2[EP]):
        raise SystemExit(f"final head {name}: the EP row equals P2's (the L2 arm would be P2)")


def head_arrays(path):
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["weight"], np.float32), np.asarray(z["bias"], np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--feats", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    W_ = a.run_dir
    fp = a.feats or f"{W_}/feats/feats.npz"
    z = np.load(fp, allow_pickle=False)
    h = np.load(os.path.join(os.path.dirname(fp), "p2_score_head.npz"))
    W2, b2 = np.asarray(h["weight"], np.float32), np.asarray(h["bias"], np.float32)
    sc = stepA.load_npz(f"{W_}/scores.npz")
    rec = c_feat(z["infan_tok"], z["pert_tok"], W2, b2, sc["a_logit"], sc["p_logit"])
    rec["bundle_sha256_match"] = str(z["bundle_sha256"]) == sha_file(f"{W_}/bundle/bundle.npz")
    rec["perts_sha256_match"] = str(z["perts_sha256"]) == sha_file(f"{W_}/perts.npz")
    rec["pass"] = bool(rec["pass"] and rec["bundle_sha256_match"] and rec["perts_sha256_match"])
    if not rec["pass"]:
        json.dump(rec, open(os.path.join(W_, "fresh_ep_scores.cfeat_FAIL.json"), "w"), indent=1)
        print("ZZCFEAT_FAIL", json.dumps(rec), flush=True)
        return 3
    fh = json.load(open(os.path.join(a.heads, "final_heads.json")))["heads"]
    arms = {"P2": (W2, b2)}
    heads_rec = {}
    for s in ("s0", "s1"):
        p = os.path.join(a.heads, f"final_head_{s}.npz")
        if sha_file(p) != fh[s]["sha256"]:
            raise SystemExit(f"final head {s}: sha256 differs from final_heads.json (the head changed after it was banked)")
        Wh, bh = head_arrays(p)
        check_head(s, Wh, bh, W2, b2)
        arms[f"L2_{s}"] = (Wh, bh)
        heads_rec[s] = {"path": p, "sha256": fh[s]["sha256"]}
    out = {}
    for name, (Wh, bh) in arms.items():
        out[f"ep_in_{name}"] = l2lib.ep_apply(z["infan_tok"], Wh, bh)
        out[f"ep_p_{name}"] = l2lib.ep_apply(z["pert_tok"], Wh, bh)
    op = a.out or os.path.join(W_, "fresh_ep_scores.npz")
    np.savez(op, **out)
    rec.update(arms=list(arms), heads=heads_rec, p2_head_sha256=sha_file(os.path.join(os.path.dirname(fp), "p2_score_head.npz")), feats_sha256=sha_file(fp), out_sha256=sha_file(op),
               note="scores only: NO label read, NO A-quantity computed")
    json.dump(rec, open(op.replace(".npz", ".json"), "w"), indent=1)
    print("ZZCFEAT_OK", json.dumps({k: (v["max_abs"], v["share_le_tol"]) for k, v in rec["rows"].items()}), flush=True)
    print("ZZSCORES_DONE", op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
