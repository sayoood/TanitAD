#!/usr/bin/env python
"""v8f_t1_eval_rule_qhead.py -- card PREREG_V8F_PROOF_ADD42 (QHEAD): the vision-fed distance rule of arm QC of card ADD31 (presence threshold P*, the head's closing-speed read), with the gap

  g_used = min(g_head, q_head)

where q_head is the output of the trained quantile head (qhead_fit.py) read from the SAME state input as the banked lead head, at every tick:
the 5 states z1 = wm.z1(adapter(tokens) + pos) of the last 5 frames. The banked head, the model and the pinned eval tool are not changed (the tools of cards ADD29 / ADD31 are imported, not edited).

  python v8f_t1_eval_rule_qhead.py --head-p-min P (--qhead FILE | --q-equals-head) --tree TREE --mix-presence head --mix-gap head --mix-closing head
         [--rule-log NPZ] [--floor {card,cap-only}] run <the argv of the pinned tool's `run`>

  --qhead FILE       the head file written by qhead_fit.py; its record next to it (FILE minus .npz plus .json) must carry the same checkpoint sha256 as --ckpt of the run
  --q-equals-head    control rollout LSQHI_s0: q_head is the head read itself, so g_used = g_head and the dump equals the all-head mix of card ADD29 (control h1)
Per call of the planner a side file `<rule-log minus .npz>_qh.npz` records head gap, quantile output, gap used, presence probability and the all-frames-real flag, row for row with the rule log (control h3).
Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance).
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISTRULE = os.environ.get("DISTRULE_DIR", "")
if not DISTRULE:
    for cand in (os.path.join(HERE, "..", "distrule"), HERE):
        if os.path.isfile(os.path.join(cand, "v8f_t1_eval_rule_mix.py")):
            DISTRULE = os.path.normpath(cand)
            break
for p in (DISTRULE, HERE):
    if p and p not in sys.path:
        sys.path.insert(0, p)
import qhead_model as QM  # noqa: E402
import v8f_t1_eval_rule_mix as X  # noqa: E402

W = X.W
STATE = {"mode": None, "head": None, "head_dev": None, "cols": {k: [] for k in ("g_head", "q_head", "g_used", "p_head", "ok")}}
_ORIG_HEAD_VALUES = W.head_values
SIDE_KEYS = tuple(STATE["cols"])


def read_q(model, feats, hist, g_head):
    """q_head in metres for each row of the batch: the quantile head on the same z1 as the banked head (mode head), or the head read itself (mode equals-head)."""
    if STATE["mode"] == "equals-head":
        return np.array(g_head, copy=True)
    import torch
    dev = feats.device
    if STATE["head_dev"] != str(dev):
        STATE["head"] = STATE["head"].to(dev).eval()
        STATE["head_dev"] = str(dev)
    if STATE["head"].hist != hist:
        raise SystemExit("REFUSED: the quantile head reads %d frames, the banked head %d" % (STATE["head"].hist, hist))
    with torch.no_grad():
        f = feats[:, -hist:]
        z1 = model.wm.z1(model.adapter(f.float()) + model.pos[None, None])      # the expression of the banked read (v8f_t1_eval_rule.head_values)
        o = STATE["head"](z1)[:, 0, 0].float().cpu().numpy()
    return (o * QM.GAP_SCALE_M).astype(np.float32)


def head_values_qhead(model, feats, frame_ok=None, hist=None):
    """The wrapper's head read with the gap replaced by min(g_head, q_head); presence, closing speed and the all-frames-real flag stay the banked values."""
    p, g, c, ok = _ORIG_HEAD_VALUES(model, feats, frame_ok, hist)
    h = int(model.wm.lead_head.hist if hist is None else hist)
    q = read_q(model, feats, h, g)
    used = QM.used_gap(g, q)
    for k, v in (("g_head", g), ("q_head", q), ("g_used", used), ("p_head", p), ("ok", ok)):
        STATE["cols"][k].append(np.asarray(v))
    return p, used, c, ok


def save_side(path, meta):
    cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in STATE["cols"].items()}
    cols["ok"] = cols["ok"].astype(bool)
    with open(path, "wb") as fh:
        np.savez(fh, **cols, **{k: np.asarray(v) for k, v in meta.items()})


def arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--head-p-min", type=float, required=True)
    ap.add_argument("--qhead", default="")
    ap.add_argument("--q-equals-head", action="store_true")
    mine, rest = ap.parse_known_args(argv)
    if not (0.0 < mine.head_p_min < 1.0):
        raise SystemExit("REFUSED: --head-p-min must lie strictly between 0 and 1")
    if bool(mine.qhead) == bool(mine.q_equals_head):
        raise SystemExit("REFUSED: give exactly one of --qhead FILE and --q-equals-head")
    if "--mix-presence" not in rest or "--mix-gap" not in rest or "--mix-closing" not in rest:
        raise SystemExit("REFUSED: name the mix (--mix-presence, --mix-gap, --mix-closing)")
    mix = {k: arg_value(rest, "--mix-" + k) for k in ("presence", "gap", "closing")}
    if mix != {"presence": "head", "gap": "head", "closing": "head"}:
        raise SystemExit("REFUSED: this card reads the all-head mix only, got %s" % json.dumps(mix))
    meta = {"card": "PREREG_V8F_PROOF_ADD42", "head_p_min": mine.head_p_min, "q_mode": "equals-head" if mine.q_equals_head else "head"}
    if mine.qhead:
        if not os.path.isfile(mine.qhead):
            raise SystemExit("REFUSED: no head file " + mine.qhead)
        if QM.is_eval_path(mine.qhead):
            raise SystemExit("REFUSED: a head file of the evaluation split: " + mine.qhead)
        rec_p = mine.qhead[:-4] + ".json" if mine.qhead.endswith(".npz") else mine.qhead + ".json"
        if not os.path.isfile(rec_p):
            raise SystemExit("REFUSED: no fit record " + rec_p)
        rec = json.load(open(rec_p, encoding="utf-8"))
        ck = arg_value(rest, "--ckpt")
        if ck and os.path.isfile(ck) and rec.get("states_record", {}).get("ckpt_sha256"):
            sha = QM.sha256_file(ck)
            if sha != rec["states_record"]["ckpt_sha256"]:
                raise SystemExit("REFUSED: the head was fitted on states of checkpoint %s, this run is %s" % (rec["states_record"]["ckpt_sha256"], sha))
        sha_head = QM.sha256_file(mine.qhead)
        if rec.get("head_file_sha256") and sha_head not in rec["head_file_sha256"].values():
            raise SystemExit("REFUSED: the head file does not match the sha256 of its fit record")
        STATE["head"] = QM.load_head(mine.qhead)
        STATE["head_dev"] = "cpu"
        meta.update(qhead=mine.qhead, qhead_sha256=sha_head, fit_record=rec_p)
    STATE["mode"] = meta["q_mode"]
    W.HEAD_P_MIN = float(mine.head_p_min)
    W.head_values = head_values_qhead
    print("qhead constants: head_p_min=%.4f q_mode=%s head=%s" % (W.HEAD_P_MIN, meta["q_mode"], mine.qhead or "-"), flush=True)
    rl = arg_value(rest, "--rule-log")
    try:
        return X.main(rest)
    finally:
        if rl and (STATE["cols"]["g_head"]):
            side = (rl[:-4] if rl.endswith(".npz") else rl) + "_qh.npz"
            save_side(side, meta)
            print("wrote", side, flush=True)


if __name__ == "__main__":
    sys.exit(main())
