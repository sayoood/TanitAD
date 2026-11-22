"""n1dm_kownsplit.py -- K-OWN-SPLIT of PREREG_V8F_N1DM section 2.3 (iii): the READ-SIDE diagnostic.  NO BAR, DECIDES NO OUTCOME.  CPU, 2 threads, zero training.

The card's words: "the route head's accuracy on own-code vs GT-code rows of a probe batch; a large gap names the shortcut.  ... if it fires, the next arm masks the decoder loss through the branches to own-code rows."
WHY: in N1DM the 3 branches train on ALL rows.  On a teacher-forced row (probability p_gt, 0.2) O's plan feature h holds the GT tactical code (future-derived) and the soft route posterior
p = softmax(route_head(h)).detach() is read from that h (v8f_mini.py:1121-1122).  If the head is much more accurate on GT-code rows, the branches see a cleaner route signal in training than they ever get at inference.
SOURCE FACTS (tree the-N1DM-commit, stack/tanitad/models/v8f_mini.py; quoted by line):
  :1156  _forward(b, p_gt, use_labels, gen=None, code_override=None)         :1189 own = all True
  :1195-1197 (code_mix == "whole") use_t = (rand < p_gt) & lab_t.has ; code = where(use_t, GT code, own) ; own = ~use_t
  :1214  out["route_logits"], out["route_own"]                               :1397  CE mask rm = route_ok & route_own  (the head is trained on OWN-code rows only)
  :317-318 cfg: aux_route != "off" requires code_mix == "whole" (a launch refusal), so the mask is the per-sample use_t draw in N1DH and N1DM alike; this tool refuses a model whose cfg.code_mix is not "whole".
PAIRED FORM (the same labelled windows, model.eval(), torch.no_grad()):  arm OWN = _forward(b, 0.0, True) (every row reads T's own code; route_own all True);  arm GT = _forward(b, GT_P=1.0, True) (every row with lab_t.has reads
the GT code, route_own False there).  Metric = the route head's argmax accuracy vs route_y on rows with route_ok & lab_t.has.  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, per checkpoint:
it answers ONLY "would another draw of EPISODES say this?" (never another training run, never another inference run).  Caveat: at p_gt 1 the S-checkpoint substitution (lab_s) is also on, as in training; the arm
"gt_tcode_only" (code_override = GT code on has rows, own elsewhere, S untouched) isolates the T-code effect and is reported beside.  Beside the pair, no bar: the NATURAL p_gt draw (the trainer's own, p_gt 0.2, seeded generator)
split by route_own.  Labels = the trainer's own route_stage (v8f_mini_train.route_stage, the K-LABEL path, unedited) on the collated batch.  A model without a route head is a REFUSAL, never a number.
Controls: K-ID (own arm twice: logits bit-identical, difference exactly 0.0), K-MASK (route_own False exactly on the has rows at p_gt 1, all True at p_gt 0; has rows exist), K-PERM (labels permuted across rows read the
chance level), K-UNIF (route_head weight and bias zeroed: constant argmax, same class share, gap exactly 0), K-INFER (the OWN arm's logits equal forward_infer's, bit for bit), and on a real run K-ALIGN / K-LABEL-RUN.
Tier: T0 (a training-format forward, teacher-forced where the arm says so; a WM diagnostic, never driving performance).
Usage: python -I n1dm_kownsplit.py --tree TREE --ckpt CKPT --profile ho|e141 --max-windows 480 --out OUT.json [--work DIR]"""
import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)

GT_P = 1.0                      # the GT arm's teacher-forcing probability (K-MASK pins what it does); a mutation that sets 0.0 turns K-MASK / K-GAP-LIVE RED
NATURAL_P = 0.2                 # the card's p_gt
N_BOOT, SEED = 2000, 0


def _ci_module(tree):
    spec = importlib.util.spec_from_file_location("ci_kown", tree + "/taniteval/taniteval/ci.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arm_forward(model, b, p_gt, use_labels, seed=0):
    """one forward of the training entry on a collated batch; returns the model's out dict (route_logits, route_own, code ...)"""
    import torch
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return model._forward(b, p_gt, use_labels, gen)


SLICE = 64                      # rows per forward (a 480-row batch of the modes model segfaulted natively on the dev box; 64-row slices are the size the tests cover)


def _slice(v, i, j, B):
    import torch
    if isinstance(v, dict):
        return {k: _slice(x, i, j, B) for k, x in v.items()}
    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
        return v[i:j]
    return v


def _slices(batches):
    for b in batches:
        B = int(b["feats"].shape[0])
        for i in range(0, B, SLICE):
            yield {k: _slice(v, i, min(i + SLICE, B), B) for k, v in b.items()}


def run_arms(model, batches, seed=0):
    """per slice of <= SLICE rows, the forwards (the natural draw's generator seed = seed + slice index); returns dict of concatenated numpy arrays"""
    import numpy as np
    import torch
    acc = {k: [] for k in ("own", "gt", "tonly", "nat", "own2", "infer", "own_own_mask", "gt_own_mask", "nat_own_mask", "has")}
    for ci, b in enumerate(_slices(batches)):
        has = b["lab_t"]["has"].bool()
        o1, o2 = arm_forward(model, b, 0.0, True, seed), arm_forward(model, b, 0.0, True, seed)
        g = arm_forward(model, b, GT_P, True, seed)
        nat = arm_forward(model, b, NATURAL_P, True, seed + ci)
        with torch.no_grad():
            inf = model.forward_infer(b)
            override = torch.where(has[:, None], model._code_gt(b["lab_t"]).to(o1["code"].dtype), o1["code"])
            to = model._forward(b, 0.0, False, None, override)
        for k, o in (("own", o1), ("own2", o2), ("gt", g), ("nat", nat), ("tonly", to), ("infer", inf)):
            acc[k].append(o["route_logits"].detach().float().numpy())
        acc["own_own_mask"].append(o1["route_own"].numpy())
        acc["gt_own_mask"].append(g["route_own"].numpy())
        acc["nat_own_mask"].append(nat["route_own"].numpy())
        acc["has"].append(has.numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def _argmax_counts(lg, m, n_class=3):
    import numpy as np
    pr = lg.argmax(-1)[m]
    return [int((pr == c).sum()) for c in range(n_class)]


def _acc_block(corr, m, eid, ci, n_boot):
    """accuracy + episode-cluster CI of one arm on the rows of mask m (descriptive; the paired block carries the claim)"""
    import numpy as np
    if not m.any():
        return {"n_rows": 0, "acc": None}
    r = ci.episode_cluster_bootstrap(corr[m].astype(np.float64), eid[m], n_boot=n_boot, seed=SEED)
    return {"n_rows": int(m.sum()), "n_episodes": int(len(set(eid[m].tolist()))), "acc": float(corr[m].mean()), "ci_lo": r.get("lo"), "ci_hi": r.get("hi"), "estimator": r.get("estimator")}


def read_split(model, batches, y, ok, eid, ci, n_boot=N_BOOT, seed=SEED, perm_tol=None):
    """the K-OWN-SPLIT record on aligned arrays (y [N] int, ok [N] bool, eid [N]); `batches` = list of collated batches in row order.  Returns the result dict with its controls."""
    import numpy as np
    if getattr(model, "route_head", None) is None:
        return {"status": "REFUSED: the model has no route head (cfg.aux_route off): K-OWN-SPLIT is undefined, not zero"}
    if getattr(model.cfg, "code_mix", "whole") != "whole":
        return {"status": f"REFUSED: cfg.code_mix is {model.cfg.code_mix!r}: the own-code mask of the route CE is the per-sample use_t draw only under 'whole' (v8f_mini.py:317-318, :1195-1197)"}
    model.eval()
    A = run_arms(model, batches, seed)
    y, ok, eid = np.asarray(y), np.asarray(ok, bool), np.asarray(eid)
    has = A["has"]
    m = ok & has
    corr = {k: (A[k].argmax(-1) == y) for k in ("own", "gt", "tonly", "nat", "infer")}
    res = {"model_cfg": {"aux_route": model.cfg.aux_route, "code_mix": model.cfg.code_mix, "o_interface": model.cfg.o_interface, "detach_upper": bool(model.cfg.detach_upper)},
           "n_rows": int(len(y)), "n_route_ok": int(ok.sum()), "n_has": int(has.sum()), "n_pair_rows": int(m.sum()), "n_episodes_pair": int(len(set(eid[m].tolist()))) if m.any() else 0,
           "class_share_route_y_pair_rows": [float((y[m] == c).mean()) if m.any() else None for c in range(3)],
           "majority_share_pair_rows": float(max((y[m] == c).mean() for c in range(3))) if m.any() else None}
    res["argmax_class_counts_pair_rows"] = {k: _argmax_counts(A[k], m) for k in ("own", "gt", "tonly")}
    res["arms"] = {k: _acc_block(corr[k], m, eid, ci, n_boot) for k in ("own", "gt", "tonly")}
    if m.any():
        for k, nm in (("gt", "gt_minus_own"), ("tonly", "gt_tcode_only_minus_own")):
            r = ci.paired_episode_cluster_bootstrap(corr[k][m].astype(np.float64), corr["own"][m].astype(np.float64), eid[m], n_boot=n_boot, seed=seed)
            res[nm] = r
    else:
        res["gt_minus_own"] = None
    res["max_abs_logit_diff_gt_vs_own"] = float(np.abs(A["gt"] - A["own"]).max())
    res["max_abs_logit_diff_gt_vs_own_on_has_rows"] = float(np.abs(A["gt"] - A["own"])[has].max()) if has.any() else None
    # natural p_gt 0.2 draw, split by route_own (no bar)
    nown = A["nat_own_mask"]
    res["natural_draw"] = {"p_gt": NATURAL_P, "note": "the trainer's own draw (seeded generator, one per chunk): rows split by route_own; route_ok rows only; descriptive, not paired (different rows)",
                           "own_code_rows": _acc_block(corr["nat"], ok & nown, eid, ci, n_boot), "gt_code_rows": _acc_block(corr["nat"], ok & ~nown, eid, ci, n_boot),
                           "share_gt_rows_among_route_ok": float((ok & ~nown).sum() / max(int(ok.sum()), 1))}
    # ---- controls
    C = {}
    d_id = float(np.abs(A["own"] - A["own2"]).max())
    paired_id = ci.paired_episode_cluster_bootstrap((A["own"].argmax(-1) == y)[m].astype(np.float64), (A["own2"].argmax(-1) == y)[m].astype(np.float64), eid[m], n_boot=200, seed=seed) if m.any() else {"delta": None}
    C["K-ID"] = {"max_abs_logit_diff_two_own_runs": d_id, "paired_delta": paired_id["delta"], "pass": bool(d_id == 0.0 and paired_id["delta"] == 0.0)}
    mask_ok = bool(np.array_equal(A["gt_own_mask"], ~has) and A["own_own_mask"].all() and has.any())
    C["K-MASK"] = {"gt_arm_own_equals_not_has": bool(np.array_equal(A["gt_own_mask"], ~has)), "own_arm_all_true": bool(A["own_own_mask"].all()), "n_has": int(has.sum()), "n_rows": int(len(has)), "pass": mask_ok,
                   "note": "a model whose probe batch has no lab_t.has row would make the GT arm the OWN arm: that reads FAIL here, never a gap of 0"}
    rs = np.random.RandomState(0)
    yp = y[rs.permutation(len(y))]
    pa = {}
    for k in ("own", "gt"):
        pred = A[k].argmax(-1)
        chance = float(sum(((pred == c) & m).sum() / max(m.sum(), 1) * ((y == c) & m).sum() / max(m.sum(), 1) for c in range(3)))
        acc = float((pred == yp)[m].mean()) if m.any() else float("nan")
        tol = perm_tol if perm_tol is not None else max(0.05, 3 * math.sqrt(0.25 / max(int(m.sum()), 1)))
        pa[k] = {"permuted_label_acc": acc, "chance_expected": chance, "abs_diff": abs(acc - chance), "tol": tol, "pass": bool(m.any() and abs(acc - chance) <= tol)}
    C["K-PERM"] = {**pa, "pass": bool(pa["own"]["pass"] and pa["gt"]["pass"])}
    uni = copy.deepcopy(model)
    with __import__("torch").no_grad():
        uni.route_head.weight.zero_()
        uni.route_head.bias.zero_()
    U = run_arms(uni, batches, seed)
    cu = {k: (U[k].argmax(-1) == y) for k in ("own", "gt")}
    cnt = {k: _argmax_counts(U[k], m) for k in ("own", "gt")}
    gap_u = float(cu["gt"][m].mean() - cu["own"][m].mean()) if m.any() else float("nan")
    C["K-UNIF"] = {"argmax_class_counts_own": cnt["own"], "argmax_class_counts_gt": cnt["gt"], "gap_gt_minus_own": gap_u, "max_abs_logit": float(max(abs(U["own"]).max(), abs(U["gt"]).max())),
                   "pass": bool(cnt["own"] == cnt["gt"] and sum(1 for c in cnt["own"] if c) == 1 and gap_u == 0.0)}
    d_inf = float(np.abs(A["own"] - A["infer"]).max())
    C["K-INFER"] = {"max_abs_logit_diff_own_arm_vs_forward_infer": d_inf, "pass": bool(d_inf == 0.0)}
    res["controls"] = C
    bad = [k for k, v in C.items() if not v["pass"]]
    res["status"] = "VALID" if not bad else "INVALID (controls failed: " + ", ".join(bad) + ")"
    return res


class FullCapture:
    """wraps TL.Capture and, on top of its collate wrapper, keeps a deep clone of EVERY collated batch INCLUDING its label dicts (lab, lab_t, lab_s), which TL.Capture's tensor-only clone leaves out"""

    def __init__(self, TL):
        self.base = TL.Capture(False)
        self.fulls = []

    def install(self, M, D):
        self.base.install(M, D)
        inner, cap = M.V8FMini.forward_infer, self

        def clone(v):
            if hasattr(v, "detach"):
                return v.detach().clone()
            if isinstance(v, dict):
                return {k: clone(x) for k, x in v.items()}
            return v

        def forward_infer(model, batch):                                # after TL's wrapper: the collated batch of THIS call (TL.Capture.last_full), label dicts included
            n0 = len(cap.base.calls)
            out = inner(model, batch)
            if len(cap.base.calls) > n0:
                cap.fulls.append({k: clone(v) for k, v in cap.base.last_full.items()})
            return out
        M.V8FMini.forward_infer = forward_infer

    def uninstall(self, M, D):
        self.base.uninstall(M, D)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", choices=["ho", "e141"], default="ho")
    ap.add_argument("--max-windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=HERE + "/kown_work")
    ap.add_argument("--hb-dir", default=None)
    for k in ("tracks", "cache", "grid", "join", "navtext", "rb"):
        ap.add_argument("--ho-" + k, default=None)
    a = ap.parse_args(argv)
    import n1dm_t0route_lib as TL
    tree = TL.setup_paths(a.tree, a.hb_dir)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(tree + "/stack"), tanitad.__file__
    import tanitad.models.v8f_mini as M
    from tanitad.data import v8f_mini_data as D
    import v8f_t1_eval_hb as HB
    import v8f_mini_train as TR
    import n1dh_klim as K
    import n1dm_ho_read as R
    import argparse as _ap
    R.setup(_ap.Namespace(tree=tree, inp=HERE + "/in", cmp_in=HERE + "/in"))
    ci = _ci_module(tree)
    t0 = time.time()
    res = {"tool": "n1dm_kownsplit.py", "card": "PREREG_V8F_N1DM.md section 2.3 (iii) K-OWN-SPLIT; code of record the-N1DM-commit", "profile": a.profile, "ckpt": a.ckpt, "tier": "T0 (training-format forward; a WM diagnostic, not driving performance)",
           "evidence_class": ("MEASURED (ours; checkpoint forward on D1-HO windows, zero training)" if a.profile == "ho" else "MEASURED (ours; checkpoint forward) on the eval-141 probe windows, NOT D1-HO; the D1-HO profile runs where its clip caches are"),
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0; answers only 'another draw of EPISODES' (never another training run / inference run)", "max_windows": a.max_windows, "report_only": True}
    model = K.load_model(a.ckpt)
    if getattr(model, "route_head", None) is None:
        res["status"] = "REFUSED: the checkpoint's model has no route head (cfg.aux_route off): K-OWN-SPLIT is undefined, not zero"
        json.dump(res, open(a.out, "w"), indent=1)
        print(res["status"])
        return 1
    cap = FullCapture(TL)
    cap.install(M, D)
    bs = a.batch or a.max_windows
    try:
        rc = HB.main(TL.tool_args(a.profile, a.ckpt, "KOWN-probe", f"{a.work}/probe", "logged", a.max_windows, bs, a))
    finally:
        cap.uninstall(M, D)
    if rc not in (0, None):
        raise RuntimeError(f"the T1 tool returned {rc}")
    dump = TL.load_dump(f"{a.work}/probe/dump_KOWN-probe")
    perm, kal = TL.align(cap.base, dump)
    res["controls_run"] = {"K-ALIGN": kal}
    if perm is None or not kal["pass_"] or len(cap.fulls) != len(cap.base.calls):
        res["status"] = "INVALID (K-ALIGN failed or captured batch count differs)"
        json.dump(res, open(a.out, "w"), indent=1)
        return 1
    ys, oks = zip(*[TR.route_stage(b) for b in cap.fulls])
    y, ok = torch.cat(ys).numpy(), torch.cat(oks).numpy()
    glat, _ = TL.kin_labels(dump["g"])
    y_ref = np.empty(len(glat), np.int64)
    y_ref[perm] = glat
    eid = np.empty(len(glat), dtype=dump["eid"].dtype)
    eid[perm] = dump["eid"]
    res["controls_run"]["K-LABEL-RUN"] = {"route_stage_equals_dump_kinematic_label_on_ok_rows": bool((y[ok] == y_ref[ok]).all()), "n_ok": int(ok.sum()), "n_rows": int(len(y)),
                                         "pass": bool((y[ok] == y_ref[ok]).all() and ok.sum() > 0)}
    rec = read_split(model, cap.fulls, y, ok, eid, ci)
    res.update(rec)
    if "controls" not in rec:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(res["status"])
        return 1
    if not res["controls_run"]["K-LABEL-RUN"]["pass"] and res["status"] == "VALID":
        res["status"] = "INVALID (controls failed: K-LABEL-RUN)"
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    g = res.get("gt_minus_own") or {}
    print(json.dumps({"n_pair_rows": res["n_pair_rows"], "n_has": res["n_has"], "n_episodes": res["n_episodes_pair"], "acc_own": res["arms"]["own"]["acc"], "acc_gt": res["arms"]["gt"]["acc"],
                      "gap_gt_minus_own": g.get("delta"), "ci": [g.get("lo"), g.get("hi")], "status": res["status"][:70]}))
    return 0 if res["status"] == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
