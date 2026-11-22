"""DSCALE10K G-LIVE reach probe + G-HYG record (pod side, GPU). Written by the pod agent 2026-11-16; NOT part of the frozen build report.

usage: reach_probe.py <argv.json> <out.json> <release.manifest.json> [--reach]
  always: builds the trainer's own cfg / data / model from the argv and records n_trainable (parameter count) and the cfg (G-HYG evidence).
  --reach: three batch kinds (corpus-only, 10k-only, mixed) from the union train windows + a corpus TL-positive control batch; per term: finite, value, grad (sum |d term / d param|).
The term names are the checker's (STAGE_A_TERMS). Their mapping onto the trainer's loss terms is stated in ``mapping`` of the output and is the probe author's:
  ckpt<-T.ckpt, tac_lat<-T.lat, tac_lon<-T.lon, traj<-ade/3+0.5*fde/6, pen<-sum pen_*, pen_self<-sum pens_*, wm<-T.wm, acc<-T.acc;
  goal_vlm / goal_geom<-the goal BCE split by the goal classes whose release GOAL_PROV contains `vlm` / not; cons_lc<-the Huber columns lc_*, cons_lat<-columns 0..9, cons_lon<-the rest.
"""
import copy
import dataclasses
import json
import os
import random
import sys

TREE = "/root/dscale10k/tree"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, "/root/dscale10k/study"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import v8f_mini_train as T  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import dscale10k_check as C  # noqa: E402

argv = json.load(open(sys.argv[1]))["argv"]
outp, relman = sys.argv[2], sys.argv[3]
do_reach = "--reach" in sys.argv
argv = [t for t in argv if t != "--fit-join"]                       # the join file exists (written by the dry build); the probe must not rewrite it
a = T.build_parser().parse_args([str(t) for t in argv])
a.out = os.path.join("/root/dscale10k/tmp", "probe2_" + os.path.basename(sys.argv[1]).replace(".json", ""))
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = T.make_config(a)
assert cfg.launch_intent_w == 0.0 and cfg.level_head != "on" and cfg.lead_head != "on", "the probe maps only the flag-off optional heads"
store, J, wins, clips, info = T.build_data(a, cfg)
prior = M.load_p10(a.p10)
mu, sd = M.cons_stats(J)
torch.manual_seed(int(a.seed))
model = M.V8FMini(cfg, prior, mu, sd).to(dev)
n_tr = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
res = {"argv": sys.argv[1], "n_clips": int(info["n_clips"]), "train_sha256": info["train_sha256"], "arm": {"n_trainable": n_tr, "cfg": json.loads(json.dumps(dataclasses.asdict(cfg), default=str))}}
if do_reach:
    tenk = {json.loads(l)["clip"] for l in open("/root/dscale10k/in/manifest_10k.jsonl") if l.strip()}
    clipname = lambda w: str(J["clip"][w[0]])
    wc = [w for w in wins if clipname(w) not in tenk]
    wt = [w for w in wins if clipname(w) in tenk]
    res["n_windows"] = {"corpus": len(wc), "tenk": len(wt)}
    rng = random.Random(0)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start, stok=getattr(store, "stok_store", None))
    man = json.load(open(relman))
    vm_bits = C.vlm_mask_from_manifest(man)
    vm = torch.tensor([float((vm_bits >> i) & 1) for i in range(22)], device=dev)
    lcn = [M.CI["lc_t_cross_s"], M.CI["lc_d_cross_m"]]
    lat_cols = list(range(0, 10))
    cmask = {"cons_lc": lcn, "cons_lat": [i for i in lat_cols if i not in lcn], "cons_lon": [i for i in range(len(M.CONS_NAMES)) if i not in lat_cols and i not in lcn]}
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    def batch(sel):
        return T.to_dev(D.collate([ds.item(*w) for w in sel]), dev)

    def terms_of(b):
        out = model.loss(b, p_gt=0.5, gen=torch.Generator(device=dev).manual_seed(0))
        Tm = out["terms"]
        lt = b["lab_t"]
        w = lt["goal_w"].float() * lt["has"][:, None].float()
        bce = F.binary_cross_entropy_with_logits(out["goal_logits"], lt["goal_y"], reduction="none")
        den = w.sum().clamp_min(1.0)
        t = {"ckpt": Tm["ckpt"], "tac_lat": Tm["lat"], "tac_lon": Tm["lon"], "goal_vlm": (bce * w * vm).sum() / den, "goal_geom": (bce * w * (1 - vm)).sum() / den,
             "traj": Tm["ade"] / 3.0 + 0.5 * Tm["fde"] / 6.0, "acc": Tm["acc"]}
        tgt = (lt["cons"] - model.cons_mu) / model.cons_sd
        if cfg.cons_absent == "zero":
            m = lt["has"][:, None].expand_as(tgt)
            tgt = torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))
        else:
            m = torch.isfinite(tgt) & lt["has"][:, None]
        hub = F.huber_loss(out["cons_std"], torch.nan_to_num(tgt), reduction="none", delta=1.0)
        mf = m.float()
        for nm, cols in cmask.items():
            cm = torch.zeros(hub.shape[-1], device=dev)
            cm[cols] = 1.0
            t[nm] = (hub * mf * cm).sum() / (mf * cm).sum().clamp_min(1.0)
        if cfg.consistency:
            t["pen"] = sum(Tm[f"pen_{r}"] for r in M.RULES)
            t["pen_self"] = sum(Tm[f"pens_{r}"] for r in M.RULES)
        if "wm" in Tm:
            t["wm"] = Tm["wm"]
        return t

    def table(b):
        t = terms_of(b)
        rec = {}
        for k, v in t.items():
            fin = bool(torch.isfinite(v).all())
            if v.requires_grad:
                g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                gs = float(sum(float(x.abs().sum()) for x in g if x is not None))
            else:
                gs = 0.0
            rec[k] = {"finite": fin, "value": float(v.detach()), "grad": gs}
        return rec

    NB = 24
    res["corpus"] = table(batch(rng.sample(wc, NB)))
    if wt:                                                # a B-set arm has no 10k windows (gate patch 2026-11-17): only the corpus kind and the control
        res["tenk"] = table(batch(rng.sample(wt, NB)))
        res["mixed"] = table(batch(rng.sample(wc, NB // 2) + rng.sample(wt, NB // 2)))
    pos = []
    for _ in range(80):
        cand = rng.sample(wc, 64)
        bb = D.collate([ds.item(*w) for w in cand])
        lt = bb["lab_t"]
        hit = ((lt["goal_w"].float().cpu() * vm.cpu()).sum(1) > 0) & lt["has"].cpu()
        pos += [cand[i] for i in range(len(cand)) if bool(hit[i])]
        if len(pos) >= NB:
            break
    res["control_n_positive_windows"] = len(pos)
    if pos:
        ctl = table(batch(pos[:NB]))
        res["control"] = {"goal_vlm_grad": ctl["goal_vlm"]["grad"], "goal_vlm_value": ctl["goal_vlm"]["value"]}
    else:
        res["control"] = {"goal_vlm_grad": None}
    res["mapping"] = __doc__.split("The term names")[1].strip()
    res["cons_absent"] = cfg.cons_absent
    res["vlm_mask_bits"] = int(vm_bits)
    kinds = ("corpus", "tenk", "mixed") if wt else ("corpus",)
    res["problems_stageA"] = C.reach_problems({k: res[k] for k in kinds + ("control",)}, False, kinds)
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(res, open(outp, "w"), indent=1, default=str)
print("PROBE-DONE", outp, "n_trainable", n_tr, "problems", res.get("problems_stageA"))
