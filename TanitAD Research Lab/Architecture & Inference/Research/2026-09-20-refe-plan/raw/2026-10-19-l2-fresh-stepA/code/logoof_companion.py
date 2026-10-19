#!/usr/bin/env python3
"""L2-FRESH-STEPA, item 2 (MM ruling): the LOG-OOF COMPANION on the B carve.  REPORTED-ONLY, NEVER GATED.

What it is.  L2's 5-fold-by-LOG out-of-fold read (L2_RESULT.json, cells H_s0 / H_s1) scored every Step A token with the head trained on the OTHER four folds (fold = rank mod 5 over the 24 logs sorted by
name).  Those five fold heads per seed were NOT banked (arms/H_s*.npz holds only the OUT-OF-FOLD SCORES).  This module REFITS them with the sealed recipe, through the sealed code path
(`l2lib.fold_split` -> `l2lib.fit_head`, exactly as `l2lib.oof_scores` does) and demands EXACT reproduction before they may be used:

  IDENTITY   the OOF scores assembled from the refit fold heads equal the BANKED arms/H_s{s}.npz `ep_in` / `ep_p` BIT FOR BIT (np.array_equal, both arrays, both seeds);
  RED ARM 1  a refit with seed + 1 must NOT reproduce them (so the identity is not vacuous: the seed reaches the result);
  RED ARM 2  fold heads applied to the WRONG folds (rolled by one) must NOT reproduce them (so the fold assignment reaches the result).

If the identity fails, the companion is OMITTED and the reason is recorded (`logoof_heads.json` -> `reproduction.pass == false`); `analyse_fresh.py` then prints "companion omitted: <reason>".

How it is used.  Each B token is scored by the fold head whose held-out fold contains that token's LOG (`score_by_fold`).  The B carve shares Step A's 24 logs, so the fold of a B token is the fold of its log
in the SAME 24-log ranking; `compatible_logs` refuses (companion omitted) if the carve's log set differs from the heads' log set.  The heads never saw the token's log, so this is a log-held-out score
for the head; it is NOT a fresh-log test for P2 or for the scene (B stays "token-fresh, LOG-SHARED").

    CUDA_VISIBLE_DEVICES=-1 python logoof_companion.py build --out <dir> [--threads 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
import l2lib  # noqa: E402

FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
RECIPE = "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json"
BANKED_ARMS = "D:/Projects/TanitAD-artifacts/l2_offprop/run/arms"
SEEDS = l2lib.TRAIN_SEEDS            # (0, 1)
N_FOLDS = l2lib.N_FOLDS              # 5
HEADS_NPZ = "logoof_heads.npz"
HEADS_JSON = "logoof_heads.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def fit_fold_heads(D, rec, seed: int) -> list:
    """the N_FOLDS fold heads of the 'H' arm at `seed`: the SAME calls `l2lib.oof_scores` makes (fold_split's structural C-LEAK checks run before any step; fit_head(... steps=None, extra=0, w_pair=1.0))"""
    arm = l2lib.ArmSpec(f"H_s{seed}", "H", seed=seed)
    Xi, Xp = D.tok_in.reshape(-1, D.tok_in.shape[-1]), D.tok_p.reshape(-1, D.tok_p.shape[-1])
    out = []
    for f in range(N_FOLDS):
        tr, te, ts, chk = l2lib.fold_split(D, f, arm, None)
        fit = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, arm.seed, rec, steps=arm.steps, extra=0, w_pair=1.0)
        out.append({"W": fit["W"], "b": fit["b"], "steps": int(fit["steps"]), "n_train_tokens": int(len(tr)), "n_test_tokens": int(len(te)), "n_pairs": int(len(ts.ia)),
                    "min_ep_grad_abs_sum": fit["min_ep_grad_abs_sum"], "leak_check": chk})
    return out


def score_by_fold(tok_in, tok_p, fold_of_token, heads: list):
    """EP score of every token by the head of ITS fold.  tok_in [N,64,D], tok_p [N,Q,D], fold_of_token [N] -> (ep_in [N,64], ep_p [N,Q]), float64 (l2lib.ep_apply, as oof_scores)."""
    fold_of_token = np.asarray(fold_of_token, np.int64)
    ep_in = np.zeros(tok_in.shape[:2], np.float64)
    ep_p = np.zeros(tok_p.shape[:2], np.float64)
    seen = np.zeros(len(fold_of_token), np.int64)
    for f, h in enumerate(heads):
        te = np.nonzero(fold_of_token == f)[0]
        ep_in[te] = l2lib.ep_apply(tok_in[te], h["W"], h["b"])
        ep_p[te] = l2lib.ep_apply(tok_p[te], h["W"], h["b"])
        seen[te] += 1
    if not (seen == 1).all():
        raise RuntimeError(f"log-OOF companion: {int((seen != 1).sum())} tokens were not scored by exactly one fold head (fold ids outside 0..{len(heads) - 1}?)")
    return ep_in, ep_p


def compatible_logs(carve_log_names, heads_log_names) -> tuple:
    """-> (ok, reason).  The fold of a carve token is defined only if its log is one of the logs the heads' folds were built from."""
    a, b = sorted(str(x) for x in carve_log_names), sorted(str(x) for x in heads_log_names)
    if a == b:
        return True, "carve logs == the 24 logs the fold heads were built from"
    extra, miss = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    return False, f"carve logs differ from the fold heads' logs ({len(extra)} not in the heads' logs, {len(miss)} heads' logs absent from the carve): the fold of a token is undefined"


def _equal(x, y) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    return {"bit_equal": bool(x.shape == y.shape and np.array_equal(x, y)), "max_abs_diff": float(np.abs(x - y).max()) if x.shape == y.shape else None, "n": int(x.size)}


def reproduction_record(D, rec, banked_dir=BANKED_ARMS, seeds=SEEDS, fitted=None) -> tuple:
    """IDENTITY + the two RED arms.  Returns (record, {seed: heads}).  `fitted` lets a caller pass already-fitted heads."""
    heads_by_seed, rec_out, ok = {}, {"seeds": {}, "banked_arms_dir": banked_dir}, True
    for s in seeds:
        z = np.load(os.path.join(banked_dir, f"H_s{s}.npz"), allow_pickle=False)
        hs = fitted[s] if fitted and s in fitted else fit_fold_heads(D, rec, s)
        heads_by_seed[s] = hs
        ei, ep = score_by_fold(D.tok_in, D.tok_p, D.fold, hs)
        ident = {"ep_in": _equal(ei, z["ep_in"]), "ep_p": _equal(ep, z["ep_p"])}
        ident["pass"] = bool(ident["ep_in"]["bit_equal"] and ident["ep_p"]["bit_equal"])
        # RED ARM 1: the same machinery at seed + 1 must NOT reproduce the banked scores
        hs1 = fit_fold_heads(D, rec, s + 1)
        ei1, ep1 = score_by_fold(D.tok_in, D.tok_p, D.fold, hs1)
        red1 = {"ep_in": _equal(ei1, z["ep_in"]), "ep_p": _equal(ep1, z["ep_p"])}
        red1["must_differ_and_does"] = bool((not red1["ep_in"]["bit_equal"]) and (not red1["ep_p"]["bit_equal"]))
        # RED ARM 2: fold heads applied to the WRONG folds (rolled by one) must NOT reproduce them
        rolled = [hs[(f + 1) % N_FOLDS] for f in range(N_FOLDS)]
        ei2, ep2 = score_by_fold(D.tok_in, D.tok_p, D.fold, rolled)
        red2 = {"ep_in": _equal(ei2, z["ep_in"]), "ep_p": _equal(ep2, z["ep_p"])}
        red2["must_differ_and_does"] = bool((not red2["ep_in"]["bit_equal"]) and (not red2["ep_p"]["bit_equal"]))
        rec_out["seeds"][f"s{s}"] = {"banked_arm_sha256": sha_file(os.path.join(banked_dir, f"H_s{s}.npz")), "identity": ident, "red_seed_plus_1": red1, "red_rolled_folds": red2,
                                     "steps_per_fold": [h["steps"] for h in hs], "n_pairs_per_fold": [h["n_pairs"] for h in hs]}
        ok = ok and ident["pass"] and red1["must_differ_and_does"] and red2["must_differ_and_does"]
    rec_out["pass"] = bool(ok)
    return rec_out, heads_by_seed


def load_heads(path_npz: str, path_json: str) -> dict:
    """-> {"manifest": ..., "heads": {seed: [ {W,b} x 5 ]}}.  Refuses a heads file whose sha256 is not the manifest's (a head changed after it was banked), and a manifest whose reproduction did not pass."""
    man = json.load(open(path_json))
    if sha_file(path_npz) != man["heads_npz_sha256"]:
        raise SystemExit("log-OOF companion: logoof_heads.npz sha256 differs from logoof_heads.json (the heads changed after they were banked)")
    if not man["reproduction"]["pass"]:
        raise SystemExit("log-OOF companion: the banked OOF scores were NOT reproduced exactly; the companion is omitted")
    z = np.load(path_npz, allow_pickle=False)
    heads = {int(s): [{"W": z["W"][i, f], "b": z["b"][i, f]} for f in range(N_FOLDS)] for i, s in enumerate(man["seeds"])}
    return {"manifest": man, "heads": heads}


def build(out_dir: str, threads: int = 2) -> int:
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rec = l2lib.Recipe.from_json(RECIPE)
    D = l2lib.load_real(FEATS)
    repro, hbs = reproduction_record(D, rec)
    man = {"what": "log-OOF companion: the 5 fold heads x seeds 0/1 of L2's out-of-fold read, REFIT with the sealed recipe through the sealed code path; reported-only, never gated",
           "seeds": list(SEEDS), "n_folds": N_FOLDS, "torch_threads": threads, "recipe_sha256": sha_lf(RECIPE), "feats_sha256": sha_file(FEATS), "log_names": list(D.log_names),
           "fold_of_log": [int(x) for x in D.fold_of_log], "code_sha256_lf": {"l2lib.py": sha_lf(os.path.join(L2_CODE, "l2lib.py")), "logoof_companion.py": sha_lf(os.path.abspath(__file__))},
           "reproduction": repro, "evidence": "MEASURED (ours); SEEN-SCENE, SELECTION tier"}
    if repro["pass"]:
        Ws = np.stack([np.stack([h["W"] for h in hbs[s]]) for s in SEEDS]).astype(np.float32)          # [2,5,6,256]
        bs = np.stack([np.stack([h["b"] for h in hbs[s]]) for s in SEEDS]).astype(np.float32)          # [2,5,6]
        steps = np.array([[h["steps"] for h in hbs[s]] for s in SEEDS])
        p = os.path.join(out_dir, HEADS_NPZ)
        np.savez(p, W=Ws, b=bs, steps=steps, seeds=np.array(SEEDS), fold_of_log=D.fold_of_log)
        man["heads_npz_sha256"] = sha_file(p)
        man["heads_npz"] = HEADS_NPZ
        frozen = [r for r in range(6) if r != l2lib.EP]
        man["frozen_rows_equal_P2_bitwise"] = bool(all(np.array_equal(Ws[i, f][frozen], np.asarray(D.W6, np.float32)[frozen]) and np.array_equal(bs[i, f][frozen], np.asarray(D.b6, np.float32)[frozen])
                                                       for i in range(len(SEEDS)) for f in range(N_FOLDS)))
    else:
        man["heads_npz_sha256"] = None
        man["omitted_because"] = "exact reproduction of the banked OOF scores failed (see reproduction)"
    man["seconds"] = round(time.time() - t0, 1)
    json.dump(man, open(os.path.join(out_dir, HEADS_JSON), "w"), indent=1)
    print("ZZLOGOOF_BUILD", json.dumps({"reproduction_pass": repro["pass"], "heads_npz_sha256": man["heads_npz_sha256"], "seconds": man["seconds"]}), flush=True)
    return 0 if repro["pass"] else 4


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return build(a.out, a.threads)


if __name__ == "__main__":
    sys.exit(main())
