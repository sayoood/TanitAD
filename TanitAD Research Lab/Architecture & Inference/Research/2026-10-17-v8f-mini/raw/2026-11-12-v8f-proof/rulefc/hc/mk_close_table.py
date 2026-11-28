"""mk_close_table.py -- card PREREG_V8F_PROOF_ADD152 (HEADCLOSE10K): the closing table of the new form, run ON box 2 from D:/tanitad/leadspeed (beside leadspeed.py,
tokstore.py and train_full_box2.py, which are imported and not edited). Tier T0 (heads on cached tokens of logged frames); never driving performance.

  python mk_close_table.py --out D:/tanitad/rhacc10k/inputs/headclose [--device cuda]

Reads the two row-217 heads (sha256 checked) on every eval-141 clip with train_full_box2.head_outputs, and writes
  HEAD_TABLE.npz  clip_index, frame, est = (c148 + c149) / 2 (NaN where the heads have no window: frames 0 .. 8), and beside c148, c149, v
  ZERO_TABLE.npz  the same rows with est = 0.0 (controls t2 / t3)
  TABLE.json      control t1 (row 217's eval-141 closing r through train_full_box2.read_quantities with the eval labels of train_full_box2), sha256 of both tables.
clip_index = the position of the clip in the join's `clip` array (the s.ci of the rollout). Prints QUEUE-COMPLETE at the end in every case (the chain's end marker).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LS_DIR = os.environ.get("LEADSPEED_DIR", HERE)
if LS_DIR not in sys.path:
    sys.path.insert(0, LS_DIR)

CARD = "PREREG_V8F_PROOF_ADD152"
HEADS = (("c148", "D:/tanitad/leadspeed/all/head_30000.pt", "83faaa6d355562f7341b946d7c2ccee4b3c39b349bd848c4b94af9d4f0cc41d5"),
         ("c149", "D:/tanitad/leadspeed/all149/head_30000.pt", "4ae87b88ab53ec3907a801e8971e84ba690944241595a1babef99593cff6e5bf"))
WANT_S = {"c148": 0.4255526531822995, "c149": 0.4445759283818997}     # box2_all/FULL.json and box2_all149/FULL.json, eval141.S
WANT_NS = 3430
S_TOL = 0.002
H_FIRST = 9                                                             # leadspeed.H - 1: the first frame with a full 10-frame window


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def eval_labels(L, J, ci, T):
    """The eval labels of train_full_box2.main (lab_eval), for join clip index ci and T token frames."""
    o, n = int(J["off"][ci]), int(J["n"][ci])
    lv = (np.asarray(J["lead_valid"][o:o + n]) == 1).astype(int)
    hv = np.asarray(J["has"][o:o + n]).astype(int)
    return L.token_labels(2 * np.arange(n), hv, lv, np.asarray(J["lead_gap"][o:o + n], np.float64), T)


def build_tables(per_clip):
    """per_clip: list of dict(ci, c148 [T], c149 [T], v [T]) -> (head table dict, zero table dict). est = mean of the two heads, NaN where either is NaN."""
    cols = {k: [] for k in ("clip_index", "frame", "est", "c148", "c149", "v")}
    for d in per_clip:
        T = len(d["c148"])
        a, b = np.asarray(d["c148"], np.float64), np.asarray(d["c149"], np.float64)
        if len(b) != T or len(d["v"]) != T:
            raise ValueError("clip %d: columns of different lengths" % d["ci"])
        est = np.where(np.isfinite(a) & np.isfinite(b), 0.5 * (a + b), np.nan)
        cols["clip_index"].append(np.full(T, int(d["ci"]), np.int64))
        cols["frame"].append(np.arange(T, dtype=np.int64))
        cols["est"].append(est)
        cols["c148"].append(a)
        cols["c149"].append(b)
        cols["v"].append(np.asarray(d["v"], np.float64))
    head = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    zero = {"clip_index": head["clip_index"].copy(), "frame": head["frame"].copy(), "est": np.zeros(len(head["est"]), np.float64)}
    return head, zero


def check_rows(head, n_join, T_of):
    """Control t1, row part: every join clip has rows 0 .. T-1 exactly once; est NaN exactly at frames 0 .. 8 and finite at 9 .. T-1. -> dict with PASS."""
    ci, fr, est = head["clip_index"], head["frame"], head["est"]
    seen = sorted(set(int(x) for x in ci))
    ok_clips = seen == list(range(n_join))
    ok_rows, ok_nan, bad = True, True, []
    for c in seen:
        m = ci == c
        f = fr[m]
        T = int(T_of[c])
        if not np.array_equal(np.sort(f), np.arange(T)):
            ok_rows = False
            bad.append(("rows", c))
        e = est[m][np.argsort(f, kind="stable")]
        want_nan = np.arange(len(e)) < H_FIRST
        if not np.array_equal(~np.isfinite(e), want_nan):
            ok_nan = False
            bad.append(("nan", c))
    return {"clips": len(seen), "join_clips": int(n_join), "all_clips": ok_clips, "rows_once": ok_rows, "nan_exactly_before_first_window": ok_nan,
            "bad": bad[:10], "PASS": bool(ok_clips and ok_rows and ok_nan)}


def check_s(S, nS):
    """Control t1, r part: S of c148 and c149 reproduce row 217 within S_TOL with n_S = 3,430."""
    per = {k: {"S": S[k], "want": WANT_S[k], "n_S": nS[k], "PASS": bool(abs(S[k] - WANT_S[k]) <= S_TOL and nS[k] == WANT_NS)} for k in WANT_S}
    return {"per_head": per, "tol": S_TOL, "PASS": all(v["PASS"] for v in per.values())}


def save_npz(path, d):
    with open(path, "wb") as fh:
        np.savez(fh, **d)
    return sha_file(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evalc", default="D:/tanitad/cache/eval")
    ap.add_argument("--join", default="D:/tanitad/rhacc10k/pins/join_eval141.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--heads", default="", help="tests only: comma list name=path=sha")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"card": CARD, "tier": "T0 (heads on cached tokens of logged frames); never driving performance", "controls": {}, "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rc = 1
    try:
        import torch
        import leadspeed as L
        import tokstore
        import train_full_box2 as TF
        res["sha256"] = {n: sha_file(os.path.join(LS_DIR, n)) for n in ("leadspeed.py", "tokstore.py", "train_full_box2.py")}
        res["sha256"]["mk_close_table.py"] = sha_file(os.path.abspath(__file__))
        heads = HEADS if not a.heads else tuple(tuple(x.split("=")) for x in a.heads.split(","))
        dev = torch.device(a.device)
        models, hk = {}, {}
        for name, path, want in heads:
            got = sha_file(path)
            hk[name] = {"path": path, "sha256": got, "PASS": got == want}
            if got != want:
                raise RuntimeError("checkpoint %s sha256 %s is not %s" % (path, got[:16], want[:16]))
            h = L.make_head(d_in=1024, width=64, temporal=True).to(dev)
            h.load_state_dict(torch.load(path, map_location=dev))
            h.eval()
            models[name] = h
        res["controls"]["t1_ckpt"] = {"heads": hk, "PASS": all(v["PASS"] for v in hk.values())}
        J = np.load(a.join, allow_pickle=True)
        jclip = [str(c) for c in J["clip"]]
        evm = TF.jl_lines(os.path.join(a.evalc, "manifest.jsonl"))
        if sorted(r["clip"] for r in evm) != sorted(jclip):
            raise RuntimeError("eval manifest clips (%d) differ from the join clips (%d)" % (len(evm), len(jclip)))
        per_clip, quant = [], {k: [] for k in WANT_S}
        quant["mean"] = []
        T_of = {}
        v_eq = True
        for r in evm:
            pth = os.path.join(a.evalc, r["file"])
            d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
            v = d["v"].numpy().astype(np.float64)
            del d
            tok = tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", TF.SHAPE))
            T = tok.shape[0]
            v_eq &= len(v) == T
            ci = jclip.index(r["clip"])
            T_of[ci] = T
            out = {"ci": ci, "v": v}
            outs = {}
            for name, h in models.items():
                outs[name] = TF.head_outputs(torch, h, tok, dev)
                out[name] = outs[name][2]
            pres, gap, clos = eval_labels(L, J, ci, T)
            mean = np.where(np.isfinite(out["c148"]) & np.isfinite(out["c149"]), 0.5 * (out["c148"] + out["c149"]), np.nan)
            for name, col in (("c148", out["c148"]), ("c149", out["c149"]), ("mean", mean)):
                src = outs.get(name, outs["c148"])
                quant[name].append({"v": v, "clos_lab": clos, "clos_head": col, "pres_lab": pres, "pres_head": src[0], "gap_lab": gap, "gap_head": src[1]})
            per_clip.append(out)
            del tok
        rq = {k: TF.read_quantities(v) for k, v in quant.items()}
        res["eval141"] = {k: {"S": rq[k]["S"], "n_S": rq[k]["n_S"], "A": rq[k]["A"], "n_A": rq[k]["n_A"]} for k in rq}
        res["controls"]["t1_r"] = check_s({k: rq[k]["S"] for k in WANT_S}, {k: rq[k]["n_S"] for k in WANT_S})
        head, zero = build_tables(sorted(per_clip, key=lambda d: d["ci"]))
        res["controls"]["t1_rows"] = check_rows(head, len(jclip), T_of)
        res["controls"]["t1_v"] = {"v_len_equals_token_frames": bool(v_eq), "PASS": bool(v_eq)}
        res["tables"] = {"head": save_npz(os.path.join(a.out, "HEAD_TABLE.npz"), head), "zero": save_npz(os.path.join(a.out, "ZERO_TABLE.npz"), zero),
                         "rows": int(len(head["est"])), "finite_est": int(np.isfinite(head["est"]).sum())}
        res["controls_all_pass"] = all(v["PASS"] for v in res["controls"].values())
        rc = 0 if res["controls_all_pass"] else 2
    except BaseException as ex:                                                             # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:600])
        res["traceback"] = traceback.format_exc()[-3000:]
        res["controls_all_pass"] = False
        rc = 3
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(a.out, "TABLE.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
    print("TABLE controls_all_pass %s rc %d" % (res.get("controls_all_pass"), rc), flush=True)
    print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
