"""Fidelity gate + slicing for the P6 video (replaces u400's inference gate; no model is run, no GPU).

Reads P6's second full-navtest pass (proposals_p6final.npz: token, proposals [N,64,8,3] executed NAVSIM poses, logits [N,64,6], pick) and, for the
1,090 video tokens (video_tokens.json order), requires:
  (1) every token is present, once;
  (2) proposals[pick] equals the banked plan in seams/refe_p6final.npz within 1e-3 m (every component; xy and heading reported separately);
  (3) argmax of the rule navsim_v1 over the logits equals the recorded pick;
  (4) markers/dump.DONE reports poses_bit_identical true (and tokens_equal true).
PASS -> writes p6_video_proposals.npz (the slice the renderer reads) and FIDELITY_GATE.json.  FAIL -> GATE_FAIL.json and exit 2.

    python gate_fidelity.py
"""
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz"
MARK = os.environ.get("P6_GATE_MARK", "F:/TanitAD_p6read_work/markers/dump.DONE")
BANK = "E:/Projects/TanitAD/data/refe_navtest/seams/refe_p6final.npz"
CKPT_SHA = "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1"
TOL = 1e-3
V1_W = (5.0, 5.0, 2.0)


def agg_v1(logits):
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return p[..., 0] * p[..., 1] * (p[..., 2] * V1_W[0] + p[..., 3] * V1_W[1] + p[..., 4] * V1_W[2]) / sum(V1_W)


def fail(why, extra=None):
    json.dump({"failed": why, "extra": extra, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(HERE, "GATE_FAIL.json"), "w"), indent=1)
    print("FIDELITY GATE FAILED:", why, extra, flush=True)
    return 2


def main():
    dump_path = os.environ.get("P6_GATE_DUMP", DUMP)
    marker = json.load(open(MARK))
    if marker.get("poses_bit_identical") is not True or marker.get("tokens_equal") is not True:
        return fail("dump.DONE does not report poses_bit_identical true / tokens_equal true", marker)
    toks = json.load(open(os.path.join(HERE, "video_tokens.json")))["tokens"]
    assert len(toks) == int(os.environ.get("P6_GATE_NTOK", 1090)) and len(set(toks)) == len(toks), len(toks)
    D = np.load(dump_path)
    B = np.load(BANK)
    if str(D["rule"]) != "navsim_v1":
        return fail("dump rule is not navsim_v1", str(D["rule"]))
    if D["logits"].shape[-1] != 6:
        return fail("dump logits are not 6 outputs", list(D["logits"].shape))
    di = {str(t): i for i, t in enumerate(D["token"])}
    bi = {str(t): i for i, t in enumerate(B["token"])}
    if len(di) != len(D["token"]):
        return fail("duplicate tokens in the dump")
    missing = [t for t in toks if t not in di or t not in bi]
    if missing:
        return fail("tokens missing from the dump or the banked seam", {"n": len(missing), "first": missing[:5]})
    rows = np.array([di[t] for t in toks])
    brow = np.array([bi[t] for t in toks])
    props, logits, pick = D["proposals"][rows], D["logits"][rows], D["pick"][rows].astype(np.int64)
    sel = props[np.arange(len(toks)), pick]                      # [1090, 8, 3]
    bank = B["poses"][brow].astype(np.float64)
    dlt = np.abs(sel.astype(np.float64) - bank)
    dxy, dh = float(dlt[..., :2].max()), float(dlt[..., 2].max())
    dmax = float(dlt.max())
    ident = int((dlt.reshape(len(toks), -1).max(1) == 0).sum())
    am = agg_v1(logits).argmax(-1)
    n_pick_mismatch = int((am != pick).sum())
    ov = {"n": len(toks), "pass": bool(dmax <= TOL and n_pick_mismatch == 0), "max_xy_delta_m": dxy, "max_heading_delta_rad": dh,
          "max_abs_delta_any_component": dmax, "tokens_identical": ident, "tolerance_m": TOL, "argmax_rule_ne_pick": n_pick_mismatch,
          "rule": "navsim_v1 (NC*DAC*(5EP+5TTC+2C)/12, no lane factor)"}
    rep = {"overall": ov, "dump": dump_path, "dump_sha256": hashlib.sha256(open(dump_path, "rb").read()).hexdigest(), "banked_seam": BANK,
           "banked_seam_sha256": hashlib.sha256(open(BANK, "rb").read()).hexdigest(), "dump_DONE": marker, "ckpt_sha256_expected": CKPT_SHA,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not ov["pass"]:
        json.dump(rep, open(os.path.join(HERE, "GATE_FAIL_detail.json"), "w"), indent=1)
        return fail("selected trajectory != banked plan within 1e-3 m, or argmax(rule) != pick", ov)
    out = os.path.join(HERE, "p6_video_proposals.npz")
    np.savez(out, token=np.array(toks), proposals=props.astype(np.float32), logits=logits.astype(np.float32), pick=pick, rule=np.array("navsim_v1"),
             ckpt_sha256=np.array(CKPT_SHA))
    json.dump(rep, open(os.path.join(HERE, "FIDELITY_GATE.json"), "w"), indent=1)
    print("FIDELITY GATE PASS", ov, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
