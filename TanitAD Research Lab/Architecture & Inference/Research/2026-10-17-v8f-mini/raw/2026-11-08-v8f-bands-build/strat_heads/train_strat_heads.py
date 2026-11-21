"""Head-only trainer for the three strategic heads (head_ctx, head_ckpt, head_gstr) on the frozen ladder world model and the per-frame latent cache. See strat_heads.py and IMPL_CHOICES_STRATHEADS.md.

  python train_strat_heads.py <levers> --selfcheck-only      fixed-seed self-check, writes selfcheck.json + SELFCHECK_TOKEN.json (only here, never by hand); refuses on any mismatch
  python train_strat_heads.py <levers> --token <file>        train under the token (the argv minus --token must hash to the token's argv_sha16)
  python train_strat_heads.py <levers> --eval-only --heads F  evaluate (STRATEGIC PAUSED banner, per-attribute n / macro-F1 / confusion, episode-cluster bootstrap, context-swap numbers)
  python train_strat_heads.py <levers> --dump-dir D --heads F write per-clip stok [T, D] for streams B and D
Exit codes: 0 ok, 2 refused (selfcheck / token / lever mismatch). CPU or cuda; the WM never receives a gradient and its sha256 is re-checked at the end."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import strat_heads as S  # noqa: E402

CODE_ROOT = os.environ.get("STRAT_CODE_ROOT", "F:/TanitAD-scratch/ladder_tree")
CODE_FILES = ("strat_heads.py", "train_strat_heads.py")


def refuse(msg: str):
    print("REFUSED: " + msg, flush=True)
    raise SystemExit(2)


# ---------------------------------------------------------------------------------------------------- args
def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--wm-ckpt", required=True)
    a("--zcache-train", default=None)
    a("--zcache-eval", default=None)
    a("--release-train", default=None, help="release dir whose frames/*.parquet carry the context + goal columns of the TRAIN clips")
    a("--release-eval", default=None)
    a("--geom-train", default=None, help="stream A strategic_geom_train.npz (checkpoint labels)")
    a("--geom-eval", default=None)
    a("--side-train", default=None, help="dir of per-clip sidecars (act, v, nav) made by act_extract.py")
    a("--side-eval", default=None)
    a("--out", required=True)
    a("--seed", type=int, default=0)
    a("--steps", type=int, default=6000)
    a("--batch", type=int, default=128)
    a("--lr", type=float, default=1e-3)
    a("--wd", type=float, default=1e-2)
    a("--warmup", type=int, default=100)
    a("--ctx-hidden", type=int, default=256)
    a("--ck-hidden", type=int, default=256)
    a("--g-hidden", type=int, default=256)
    a("--nav-dim", type=int, default=8)
    a("--w-ctx", type=float, default=1.0)
    a("--w-ck", type=float, default=1.0)
    a("--w-g", type=float, default=1.0)
    a("--ck-act-mix", type=float, default=0.5, help="share of samples whose checkpoint rollout uses the neutral zero action (the only mode at inference)")
    a("--country-w-pow", type=float, default=1.0)
    a("--other-w-pow", type=float, default=0.5)
    a("--device", default="cpu")
    a("--limit-clips", type=int, default=0)
    a("--eval-stride", type=int, default=5)
    a("--n-boot", type=int, default=1000)
    a("--log-every", type=int, default=100)
    a("--save-every", type=int, default=500)
    a("--selfcheck-only", action="store_true")
    a("--token", default=None)
    a("--eval-only", action="store_true")
    a("--heads", default=None)
    a("--dump-dir", default=None)
    a("--resume", action="store_true")
    return p


def argv_sha16(argv: list) -> str:
    toks, skip = [], False
    for t in argv:
        if skip:
            skip = False
            continue
        if t == "--token":
            skip = True
            continue
        if t == "--selfcheck-only":
            continue
        toks.append(t)
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()[:16]


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- world model
def import_code():
    for p in (CODE_ROOT + "/stack", CODE_ROOT + "/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    root = os.path.abspath(CODE_ROOT).replace("\\", "/").lower()
    got = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    if not got.startswith(root):
        refuse(f"tanitad imported from {got}, not from the pinned code root {root}")
    from tanitad.models import v8f_mini as M
    return M


def load_wm(path: str, device: str = "cpu"):
    M = import_code()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    wm = M.WMAux(cfg)
    sd = {k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}
    r = wm.load_state_dict(sd, strict=True)
    fz = S.FrozenWM(wm.to(device))
    return fz, cfg, {"ckpt_sha256": S.file_sha(path), "cfg_d": int(cfg.d), "wm_dim": int(cfg.wm_dim), "wm_strides": list(cfg.wm_strides), "wm_k": int(cfg.wm_k)}


# ---------------------------------------------------------------------------------------------------- build
class Mix:
    """The loss weights the trainer's total really uses (read back by the selfcheck)."""

    def __init__(self, w_ctx, w_ck, w_g):
        self.w = {"ctx": float(w_ctx), "ck": float(w_ck), "g": float(w_g)}

    def total(self, lc, lk, lg):
        return self.w["ctx"] * lc + self.w["ck"] * lk + self.w["g"] * lg


def build(args, fz, train_clips):
    d_lat = train_clips[0].z.shape[1]
    heads = S.StratHeads(d_lat, fz.d3, args.nav_dim, args.ctx_hidden, args.ck_hidden, args.g_hidden, step_s=fz.step_s)
    heads.to(args.device)
    params = [p for p in heads.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    smp = S.Sampler(train_clips, args.batch, args.seed, args.ck_act_mix, S.HIST_OFFSETS, args.nav_dim)
    smp.seed = args.seed
    wts = {"country": S.country_weights(train_clips, args.country_w_pow).to(args.device)}
    for i, (n, c) in enumerate(S.CTX_ATTRS):
        wts[n] = S.class_weights(train_clips, "ctx4", i, len(c), args.other_w_pow).to(args.device)
    gw = S.class_weights(train_clips, "g_tok", None, len(S.STR_TOKENS), args.other_w_pow).to(args.device)
    wi = np.concatenate([c.lab["g_wi"][np.isfinite(c.lab["g_wi"])] for c in train_clips] + [np.array([], np.float32)])
    ws = float(np.percentile(wi, 95)) if wi.size else 300.0
    heads.gstr.within_scale.fill_(max(ws, 1.0))
    mix = Mix(args.w_ctx, args.w_ck, args.w_g)
    return heads, opt, smp, wts, gw, mix


def dev(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_losses(heads, fz, batch, wts, gw, step_s, zero_act=None):
    z = batch["zero_act"] if zero_act is None else zero_act
    z3, rolled = S.prepare_wm(fz, batch, z)
    out = heads.forward_all(batch, z3, rolled)
    lc, tc = S.loss_ctx(out["ctx"], batch, wts)
    mask = S.ck_mask(batch, z, step_s)
    lk, per_h = S.loss_ck(out["ck"], batch, heads.ckpt.horizons_s, mask)
    lg, tg = S.loss_g(out["g"], batch, gw, float(heads.gstr.within_scale))
    return out, (lc, lk, lg), {"ctx": tc, "g": tg, "ck_per_h": per_h, "ck_n": int(mask.sum())}


# ---------------------------------------------------------------------------------------------------- self-check
def group_grads(heads, fz, loss) -> dict:
    """abs-sum of the gradient a single loss puts on each group; the WM reads 0.0 when it has no .grad at all."""
    heads.zero_grad(set_to_none=True)
    for p in fz.wm.parameters():
        p.grad = None
    loss.backward(retain_graph=True)
    out = {}
    for name, mod in (("ctx", heads.ctx), ("ckpt", heads.ckpt), ("gstr", heads.gstr)):
        out[name] = float(sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None))
    out["wm"] = float(sum(float(p.grad.abs().sum()) for p in fz.wm.parameters() if p.grad is not None))
    return out


def check_grad_matrix(heads, fz, batch, wts, gw, step_s) -> dict:
    out, (lc, lk, lg), _ = forward_losses(heads, fz, batch, wts, gw, step_s)
    G = {"ctx": group_grads(heads, fz, lc), "ck": group_grads(heads, fz, lk), "g": group_grads(heads, fz, lg)}
    own = {"ctx": "ctx", "ck": "ckpt", "g": "gstr"}
    bad = []
    for loss, row in G.items():
        if not row[own[loss]] > 0.0:
            bad.append(f"{loss} loss puts no gradient on its own head ({row[own[loss]]})")
        for grp in ("ctx", "ckpt", "gstr", "wm"):
            if grp != own[loss] and row[grp] != 0.0:
                bad.append(f"{loss} loss puts {row[grp]:.3e} on {grp} (must be exactly 0.0)")
    if bad:
        raise AssertionError("; ".join(bad))
    return {"matrix": G, "rolled_requires_grad": False}


class _NoDetachHeads(S.StratHeads):
    def ctx_for_gstr(self, out):                                       # MUTATION: the detach is removed
        return torch.cat([F.softmax(out[n], -1) for n in self.ctx.names], -1)


class _EgoCtx(S.CtxHead):
    def forward(self, batch):                                          # MUTATION 1: names an ego input
        o = super().forward(batch)
        o["weather"] = o["weather"] + batch["act_f"][:, :1, 0] * 0.0 + batch["act_f"][:, 0, :1].sum(-1, keepdim=True)
        return o


class _EgoCtxHidden(S.CtxHead):
    def forward(self, batch):                                          # MUTATION 2: the same read, name hidden from the static scan (caught by the invariance probe)
        o = super().forward(batch)
        key = "a" + "ct_f"
        o["weather"] = o["weather"] + batch[key][:, 0, :1].sum(-1, keepdim=True)
        return o


class _SituationGstr(S.GstrHead):
    def forward(self, batch, z3_now, ctx_p):                           # MUTATION: the goal path reads a situation-classifier output
        o = super().forward(batch, z3_now, ctx_p)
        o["tok"] = o["tok"] + batch["situation"]
        return o


def must_fail(fn, label: str) -> str:
    try:
        fn()
    except AssertionError as e:
        return f"RED as required ({label}): {str(e)[:160]}"
    raise AssertionError(f"MUTATION NOT DETECTED: {label} (the check it targets is dead)")


def edge_effect(heads, fz, batch, kind: str) -> float:
    """Max abs change of head_gstr's outputs when the incoming edge is replaced: 'ctx_swap' (another row's posteriors), 'ctx_cut' (zeros), 'nav' (another row's nav)."""
    heads.eval()
    with torch.no_grad():
        z3 = fz.z3_of(batch["zhist"][:, 0])
        c = heads.ctx(batch)
        p = heads.ctx_for_gstr(c)
        o0 = heads.gstr(batch, z3, p)
        if kind == "ctx_swap":
            o1 = heads.gstr(batch, z3, p.roll(1, 0))
        elif kind == "ctx_cut":
            o1 = heads.gstr(batch, z3, torch.zeros_like(p))
        else:
            b2 = dict(batch)
            b2["nav"] = batch["nav"].roll(1, 0) + 1.0
            o1 = heads.gstr(b2, z3, p)
    heads.train()
    return float(max((o0[k] - o1[k]).abs().max() for k in o0))


def check_edges(heads, fz, batch) -> dict:
    r = {k: edge_effect(heads, fz, batch, k) for k in ("ctx_swap", "ctx_cut", "nav")}
    bad = [k for k, v in r.items() if not v > 0.0]
    if bad:
        raise AssertionError(f"edge(s) {bad} do not move head_gstr: {r}")
    return r


def check_chain(fz, clip) -> dict:
    """The per-frame lift and the macro-action chain reproduce HJEPAStack.encode on a 31-frame window (tick frames 0, 15, 30)."""
    z = torch.as_tensor(clip.z[:31])[None]
    rng = np.random.default_rng(5)
    act = torch.as_tensor(rng.normal(0, 1, (1, 30, 2)).astype(np.float32))
    with torch.no_grad():
        z1 = fz.wm.z1_from_mean(z)
        enc = fz.stack.encode(z1, fz.wm.scale_actions(act))
        mine = fz.z3_of(z[0, [0, 15, 30]])
        mac = fz.macro3(act)
    d_state = float((mine - enc.z[2][0]).abs().max())
    d_mac = float((mac[0] - enc.act[2][0]).abs().max())
    if not (d_state < 1e-4 and d_mac < 1e-4):
        raise AssertionError(f"per-frame lift / macro chain differ from HJEPAStack.encode: state {d_state}, macro {d_mac}")
    return {"max_abs_state": d_state, "max_abs_macro": d_mac, "tol": 1e-4, "step_s": fz.step_s, "strides": [fz.s2, fz.s3]}


def check_wm_frozen(heads, fz, batch, wts, gw, step_s, args) -> dict:
    """One optimizer step on the heads leaves the WM sha256 unchanged (bit-equal); the detector is proved alive by a mutation: an optimizer that DOES include a copy's WM parameters changes the sha."""
    before = fz.sha()
    h2 = copy.deepcopy(heads)
    opt = torch.optim.AdamW(h2.parameters(), lr=1e-2)
    out, (lc, lk, lg), _ = forward_losses(h2, fz, batch, wts, gw, step_s)
    (lc + lk + lg).backward()
    opt.step()
    after = fz.sha()
    if before != after:
        raise AssertionError("the WM state changed after a head step")
    wm2 = copy.deepcopy(fz.wm)
    for p in wm2.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.SGD(wm2.parameters(), lr=1e-1)
    (sum(p.sum() for p in wm2.parameters()) * 1.0).backward()
    opt2.step()
    mutated = S.tensor_sha(wm2.state_dict())
    if mutated == before:
        raise AssertionError("MUTATION NOT DETECTED: an optimizer over the WM parameters left the sha unchanged (the bit-equal check is dead)")
    return {"wm_sha256": before, "unchanged_after_head_step": True, "mutation": "optimizer over WM parameters changes sha256 as required"}


def readback(args, heads, fz, opt, smp, wts, mix, clips) -> tuple:
    """(declared, built) lever dicts. Built values are read from the BUILT objects, never from args."""
    ctx_dims = [o.out_features for o in heads.ctx.out]
    declared = {"ctx_hidden": args.ctx_hidden, "ck_hidden": args.ck_hidden, "g_hidden": args.g_hidden, "nav_dim": args.nav_dim, "ctx_dims": list(S.CTX_DIMS),
                "horizons_s": list(S.HORIZONS_S), "wm_step_s": 3.0, "wm_strides": [3, 5], "ck_k_index": list(range(1, 10)), "ck_act_mix": args.ck_act_mix, "hist_offsets": list(S.HIST_OFFSETS),
                "batch": args.batch, "lr": args.lr, "wd": args.wd, "seed": args.seed, "w": {"ctx": args.w_ctx, "ck": args.w_ck, "g": args.w_g},
                "country_w": S.country_weights(clips, args.country_w_pow).tolist(), "wm_trainable_params": 0, "opt_params_in_wm": 0, "detach_ctx_into_gstr": True}
    wm_ids = {id(p) for p in fz.wm.parameters()}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    built = {"ctx_hidden": heads.ctx.trunk[0].out_features, "ck_hidden": heads.ckpt.mlp[0].out_features, "g_hidden": heads.gstr.trunk[0].out_features, "nav_dim": heads.gstr.nav_dim,
             "ctx_dims": ctx_dims, "horizons_s": [float(x) for x in heads.ckpt.horizons_s], "wm_step_s": round(fz.step_s, 6), "wm_strides": [fz.s2, fz.s3],
             "ck_k_index": [int(k) for k in heads.ckpt.k_index], "ck_act_mix": smp.ck_act_mix, "hist_offsets": list(smp.hist_offsets), "batch": smp.batch,
             "lr": opt.param_groups[0]["lr"], "wd": opt.param_groups[0]["weight_decay"], "seed": smp.seed, "w": dict(mix.w), "country_w": wts["country"].cpu().tolist(),
             "wm_trainable_params": sum(1 for p in fz.wm.parameters() if p.requires_grad), "opt_params_in_wm": len(wm_ids & opt_ids)}
    return declared, built


def selfcheck(args, heads, fz, opt, smp, wts, gw, mix, clips, wm_info) -> dict:
    torch.manual_seed(20261108)
    rec = {"seed": 20261108, "tanitad_file": None}
    import tanitad
    rec["tanitad_file"] = tanitad.__file__
    declared, built = readback(args, heads, fz, opt, smp, wts, mix, clips)
    bad = [f"{k}: declared {declared[k]!r}, built {built.get(k)!r}" for k in built if k in declared and declared[k] != built[k]]
    if bad:
        refuse("declared levers differ from the BUILT heads: " + "; ".join(bad))
    rec["levers"] = {"declared": {k: declared[k] for k in declared if k not in ("country_w",)}, "built_equal": True}
    items = smp.draw(min(32, args.batch))
    batch = dev(smp.assemble(items, zero_mask=np.arange(len(items)) % 2 == 0), args.device)
    heads.train()
    try:
        rec["chain_vs_encode"] = check_chain(fz, clips[0])
        rec["ctx_vision_only"] = S.assert_ctx_vision_only(heads.ctx, batch)
        z3 = fz.z3_of(batch["zhist"][:, 0])
        rec["gstr_goal_path"] = S.assert_gstr_goal_path(heads.gstr, batch, z3, heads.ctx_for_gstr(heads.ctx(batch)))
        ck_src = S.source_problems(type(heads.ckpt).forward, S.CKPT_FORBIDDEN)
        if ck_src:
            raise AssertionError(f"head_ckpt code names forbidden inputs: {ck_src}")
        rec["ckpt_source_scan"] = {"forbidden_hits": []}
        rec["grad_matrix"] = check_grad_matrix(heads, fz, batch, wts, gw, fz.step_s)
        rec["detach_measured"] = rec["grad_matrix"]["matrix"]["g"]["ctx"] == 0.0
        rec["edges_live"] = check_edges(heads, fz, batch)
        rec["wm_frozen"] = check_wm_frozen(heads, fz, batch, wts, gw, fz.step_s, args)
        mu = {}
        mu["no_detach"] = must_fail(lambda: check_grad_matrix(_clone_as(heads, _NoDetachHeads), fz, batch, wts, gw, fz.step_s), "detach removed from the ctx -> gstr edge")
        h_ego = _clone_as(heads, S.StratHeads)
        h_ego.ctx.__class__ = _EgoCtx
        mu["ctx_ego_named"] = must_fail(lambda: S.assert_ctx_vision_only(h_ego.ctx, batch), "ego input named in head_ctx")
        h_hid = _clone_as(heads, S.StratHeads)
        h_hid.ctx.__class__ = _EgoCtxHidden
        mu["ctx_ego_hidden"] = must_fail(lambda: S.assert_ctx_vision_only(h_hid.ctx, batch), "ego input read under a computed key name (invariance probe only)")
        h_sit = _clone_as(heads, S.StratHeads)
        h_sit.gstr.__class__ = _SituationGstr
        mu["gstr_situation"] = must_fail(lambda: S.assert_gstr_goal_path(h_sit.gstr, {**batch, "situation": torch.zeros(1, 8)}, z3, heads.ctx_for_gstr(heads.ctx(batch))), "situation-classifier read in the goal path")
        h_cut = _clone_as(heads, S.StratHeads)
        with torch.no_grad():
            h_cut.gstr.trunk[0].weight[:, h_cut.gstr.norm.normalized_shape[0] + h_cut.gstr.nav_dim:] = 0.0
        mu["ctx_edge_cut"] = must_fail(lambda: check_edges(h_cut, fz, batch), "context edge cut (ctx columns of gstr zeroed)")
        rec["mutations"] = mu
    except AssertionError as e:
        refuse("selfcheck: " + str(e))
    rec["pass"] = True
    rec["wm_info"] = wm_info
    return rec


def _clone_as(heads, cls):
    h = cls(heads.ctx.norm.normalized_shape[0] // 2, heads.gstr.norm.normalized_shape[0], heads.gstr.nav_dim, heads.ctx.trunk[0].out_features, heads.ckpt.mlp[0].out_features,
            heads.gstr.trunk[0].out_features, step_s=3.0)
    h.load_state_dict(heads.state_dict())
    return h


# ---------------------------------------------------------------------------------------------------- evaluation
def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pr = cm.sum(0).astype(float)
    f = [2 * tp[i] / (sup[i] + pr[i]) for i in range(len(tp)) if sup[i] > 0 and (sup[i] + pr[i]) > 0]
    return float(np.mean(f)) if f else float("nan")


@torch.no_grad()
def evaluate(heads, fz, clips, args, step_s) -> dict:
    from taniteval import ci
    heads.eval()
    rec = {}
    n_cls = {"country": len(S.COUNTRIES), **{n: len(c) for n, c in S.CTX_ATTRS}}
    for cid_i, c in enumerate(clips):
        T = len(c.z)
        ts = [t for t in range(0, T, args.eval_stride) if c.ok[t]]
        if not ts:
            continue
        smp = S.Sampler([c], 256, 0, 0.0, S.HIST_OFFSETS, args.nav_dim)
        cm = {n: np.zeros((k, k), np.int64) for n, k in n_cls.items()}
        ck = {m: {"h": np.zeros((S.N_HZ, 3)), "cv": np.zeros((S.N_HZ, 3)), "n": np.zeros(S.N_HZ)} for m in ("zero", "logged")}
        g = {v: {"cm": np.zeros((8, 8), np.int64), "bt": [0.0, 0], "wi": [0.0, 0], "con": [0.0, 0.0, 0], "tokp": []} for v in ("plain", "swap")}
        partner = clips[(cid_i + 1) % len(clips)]
        psmp = S.Sampler([partner], 256, 0, 0.0, S.HIST_OFFSETS, args.nav_dim)
        for s in range(0, len(ts), 256):
            items = [(0, t) for t in ts[s:s + 256]]
            batch = dev(smp.assemble(items, zero_mask=np.zeros(len(items), bool)), args.device)
            first_ok = int(np.flatnonzero(partner.ok)[0])
            pitems = [(0, min(t, len(partner.z) - 1)) for _, t in items]
            pitems = [(0, t) if partner.ok[t] else (0, first_ok) for _, t in pitems]
            pbatch = dev(psmp.assemble(pitems, zero_mask=np.zeros(len(items), bool)), args.device)
            c_out = heads.ctx(batch)
            lab = {"country": batch["country"], **{n: batch["ctx4"][:, i] for i, (n, _) in enumerate(S.CTX_ATTRS)}}
            for n in n_cls:
                m = lab[n] >= 0
                if m.any():
                    np.add.at(cm[n], (lab[n][m].cpu().numpy(), c_out[n].argmax(-1)[m].cpu().numpy()), 1)
            for mode in ("zero", "logged"):
                zero = torch.full((len(items),), mode == "zero", dtype=torch.bool, device=args.device)
                z3, rolled = S.prepare_wm(fz, batch, zero)
                pred = heads.ckpt(z3, rolled)
                mask = S.ck_mask(batch, zero, step_s)
                if mode == "logged" and not c.side:
                    mask = mask & False
                sc = S.ck_scale(heads.ckpt.horizons_s)[None]
                phys = pred * sc
                y = batch["ck_y"]
                err = torch.stack([torch.hypot(phys[..., 0] - y[..., 0], phys[..., 1] - y[..., 1]), S.wrap(phys[..., 2] - y[..., 2]).abs()], -1)
                tt = heads.ckpt.horizons_s[None]
                cvx, cvy = batch["v0"][:, None] * tt, torch.zeros_like(tt)
                cv = torch.stack([torch.hypot(cvx - y[..., 0], cvy - y[..., 1]), S.wrap(0.0 - y[..., 2]).abs()], -1)
                mk = mask.float()
                ck[mode]["h"][:, 0] += (err[..., 0] * mk).sum(0).cpu().numpy()
                ck[mode]["h"][:, 1] += (err[..., 1] * mk).sum(0).cpu().numpy()
                ck[mode]["cv"][:, 0] += (cv[..., 0] * mk).sum(0).cpu().numpy()
                ck[mode]["cv"][:, 1] += (cv[..., 1] * mk).sum(0).cpu().numpy()
                ck[mode]["n"] += mk.sum(0).cpu().numpy()
            z3 = fz.z3_of(batch["zhist"][:, 0])
            p_plain = heads.ctx_for_gstr(c_out)
            p_swap = heads.ctx_for_gstr(heads.ctx(pbatch))
            o = {"plain": heads.gstr(batch, z3, p_plain), "swap": heads.gstr(batch, z3, p_swap)}
            for v, ov in o.items():
                m = batch["g_m"]
                if m.any():
                    np.add.at(g[v]["cm"], (batch["g_tok"][m].cpu().numpy(), ov["tok"].argmax(-1)[m].cpu().numpy()), 1)
                ma = m & torch.isfinite(batch["g_bt"]) & torch.isfinite(batch["g_wi"]) & (batch["g_tok"] != 0)
                if ma.any():
                    g[v]["bt"][0] += float((ov["reg"][ma, 0] * S.BT_SCALE - batch["g_bt"][ma]).abs().sum())
                    g[v]["bt"][1] += int(ma.sum())
                    g[v]["wi"][0] += float((ov["reg"][ma, 1] * heads.gstr.within_scale - batch["g_wi"][ma]).abs().sum())
                    g[v]["wi"][1] += int(ma.sum())
                mc = batch["con_m"]
                if mc.any():
                    sc2 = torch.tensor(S.CON_SCALE, device=args.device)
                    d = (ov["con"][mc] * sc2 - batch["con_y"][mc]).abs()
                    g[v]["con"][0] += float(d[:, 0].sum())
                    g[v]["con"][1] += float(d[:, 1].sum())
                    g[v]["con"][2] += int(mc.sum())
            g["plain"]["tokp"].append(float((F.softmax(o["plain"]["tok"], -1) - F.softmax(o["swap"]["tok"], -1)).abs().sum(-1).sum()))
            g["plain"].setdefault("n_rows", 0)
            g["plain"]["n_rows"] += len(items)
        rec[c.cid] = {"cm": cm, "ck": ck, "g": g}
    heads.train()
    return summarise(rec, args, ci)


def _ci(items, eid, stat, args, ci):
    return ci.stat_cluster_bootstrap(items, eid, stat, n_boot=args.n_boot, seed=0)


def summarise(rec: dict, args, ci) -> dict:
    cids = sorted(rec)
    out = {"strategic_status": "PAUSED (PI item 25)", "tier": "T0 (teacher-forced vision latents on the cache; checkpoints under logged actions are T0 diagnostics)", "evidence_class": "MEASURED",
           "estimator": "episode-cluster bootstrap over clips (taniteval/ci.py stat_cluster_bootstrap), n_boot=%d" % args.n_boot, "n_clips": len(cids), "ctx": {}, "ckpt": {}, "gstr": {}}
    for n in ("country",) + tuple(a for a, _ in S.CTX_ATTRS):
        items = [rec[c]["cm"][n] for c in cids]
        tot = sum(items)
        classes = S.COUNTRIES if n == "country" else dict(S.CTX_ATTRS)[n]
        out["ctx"][n] = {"n_frames": int(tot.sum()), "n_clips_with_label": int(sum(1 for i in items if i.sum() > 0)), "macro_f1": _ci(items, cids, lambda it: macro_f1(sum(it)), args, ci),
                         "accuracy": float(np.trace(tot) / max(tot.sum(), 1)), "classes": list(classes), "support": tot.sum(1).tolist(), "confusion_true_by_pred": tot.tolist()}
    for mode in ("zero", "logged"):
        res = {}
        for h, hz in enumerate(S.HORIZONS_S):
            items = [(rec[c]["ck"][mode]["h"][h, 0], rec[c]["ck"][mode]["cv"][h, 0], rec[c]["ck"][mode]["n"][h], rec[c]["ck"][mode]["h"][h, 1], rec[c]["ck"][mode]["cv"][h, 1]) for c in cids]
            n = sum(i[2] for i in items)
            if n == 0:
                res[f"{hz:g}s"] = {"n": 0, "note": "no valid label / no logged actions"}
                continue

            def stat(it):
                nn = max(sum(i[2] for i in it), 1.0)
                hd, cv = sum(i[0] for i in it) / nn, sum(i[1] for i in it) / nn
                return {"head_xy_m": hd, "cv_xy_m": cv, "delta_head_minus_cv_m": hd - cv, "head_heading_rad": sum(i[3] for i in it) / nn, "cv_heading_rad": sum(i[4] for i in it) / nn}
            res[f"{hz:g}s"] = {"n": int(n), **_ci(items, cids, stat, args, ci)}
        out["ckpt"][mode] = res
    for v in ("plain", "swap"):
        items = [rec[c]["g"][v] for c in cids]
        cm = sum(i["cm"] for i in items)

        def gstat(it):
            cmm = sum(i["cm"] for i in it)
            bt, wi, con = sum(i["bt"][0] for i in it), sum(i["wi"][0] for i in it), sum(i["con"][0] for i in it)
            nb, nc = max(sum(i["bt"][1] for i in it), 1), max(sum(i["con"][2] for i in it), 1)
            return {"token_macro_f1": macro_f1(cmm), "by_time_mae_s": bt / nb, "within_mae_m": wi / nb, "speed_margin_mae": con / nc,
                    "headway_margin_mae_s": sum(i["con"][1] for i in it) / nc}
        out["gstr"][v] = {"n_frames": int(cm.sum()), "n_args_frames": int(sum(i["bt"][1] for i in items)), **_ci(items, cids, gstat, args, ci), "token_classes": list(S.STR_TOKENS),
                          "support": cm.sum(1).tolist(), "confusion_true_by_pred": cm.tolist()}
    nrows = sum(rec[c]["g"]["plain"].get("n_rows", 0) for c in cids)
    out["gstr"]["swap_effect_mean_l1_token_probs"] = sum(sum(rec[c]["g"]["plain"]["tokp"]) for c in cids) / max(nrows, 1)
    return out


def print_report(r: dict):
    print("STRATEGIC PAUSED (PI item 25): the numbers below are diagnostics at tier T0, never a pass.", flush=True)
    for n, d in r["ctx"].items():
        m = d["macro_f1"]
        print(f"  ctx {n:13s} n={d['n_frames']:7d} clips={d['n_clips_with_label']:4d} macro-F1 {m['point']:.3f} [{m.get('lo', float('nan')):.3f}, {m.get('hi', float('nan')):.3f}] acc {d['accuracy']:.3f}", flush=True)
    for mode, res in r["ckpt"].items():
        for hz, d in res.items():
            if d.get("n", 0) == 0:
                continue
            print(f"  ckpt[{mode:6s}] {hz:>4s} n={d['n']:7d} head {d['head_xy_m']['point']:8.1f} m  cv {d['cv_xy_m']['point']:8.1f} m  delta {d['delta_head_minus_cv_m']['point']:+8.1f} [{d['delta_head_minus_cv_m'].get('lo', float('nan')):+.1f}, {d['delta_head_minus_cv_m'].get('hi', float('nan')):+.1f}]", flush=True)
    for v in ("plain", "swap"):
        d = r["gstr"][v]
        print(f"  gstr[{v:5s}] n={d['n_frames']:7d} token macro-F1 {d['token_macro_f1']['point']:.3f}  by_time MAE {d['by_time_mae_s']['point']:.2f} s  within MAE {d['within_mae_m']['point']:.1f} m  "
              f"speed-margin MAE {d['speed_margin_mae']['point']:.3f}  headway MAE {d['headway_margin_mae_s']['point']:.3f} s", flush=True)
    print(f"  gstr context-swap effect: mean L1 change of token probabilities {r['gstr']['swap_effect_mean_l1_token_probs']:.4f}", flush=True)


# ---------------------------------------------------------------------------------------------------- dump
STOK_KEYS = (("ctx_country", len(S.COUNTRIES)), ("ctx_illumination", 4), ("ctx_weather", 4), ("ctx_surface", 3), ("ctx_visibility", 3),
             ("ck_zero", S.N_HZ * 3), ("g_tok_p", 8), ("g_by_time_norm", 1), ("g_within_norm", 1), ("g_constraints_norm", 2))


@torch.no_grad()
def dump_clip(heads, fz, clip, args, step_s, path: str, meta: dict):
    heads.eval()
    T = len(clip.z)
    D = sum(n for _, n in STOK_KEYS)
    stok = np.zeros((T, D), np.float32)
    ckl = np.zeros((T, S.N_HZ * 3), np.float32)
    idx = [t for t in range(T) if clip.ok[t]]
    smp = S.Sampler([clip], 256, 0, 0.0, S.HIST_OFFSETS, args.nav_dim)
    for s in range(0, len(idx), 256):
        ts = idx[s:s + 256]
        b = dev(smp.assemble([(0, t) for t in ts], zero_mask=np.ones(len(ts), bool)), args.device)
        z3, rolled = S.prepare_wm(fz, b, b["zero_act"])
        out = heads.forward_all(b, z3, rolled)
        probs = [F.softmax(out["ctx"][n], -1) for n in heads.ctx.names]
        g = out["g"]
        row = torch.cat(probs + [out["ck"].reshape(len(ts), -1), F.softmax(g["tok"], -1), g["reg"], g["con"]], -1)
        stok[ts] = row.cpu().numpy()
        if clip.side:
            zl = torch.zeros(len(ts), dtype=torch.bool, device=args.device)
            z3l, rl = S.prepare_wm(fz, b, zl)
            ckl[ts] = heads.ckpt(z3l, rl).reshape(len(ts), -1).cpu().numpy()
    keys, o = [], 0
    for n, k in STOK_KEYS:
        keys.append([n, o, o + k])
        o += k
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, stok=stok, stok_ok=clip.ok.astype(bool), stok_dim=np.int64(D), stok_keys=np.array(json.dumps(keys)), ck_logged_diagnostic=ckl, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)
    heads.train()


# ---------------------------------------------------------------------------------------------------- main
def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    f = (step - args.warmup) / max(args.steps - args.warmup, 1)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))


def load_data(args):
    tr_lab = S.Labels(args.release_train, args.geom_train) if args.release_train else None
    tr = S.load_clipset(args.zcache_train, tr_lab, args.side_train, limit=args.limit_clips) if args.zcache_train and tr_lab else []
    ev_lab = S.Labels(args.release_eval, args.geom_eval) if args.release_eval else None
    ev = S.load_clipset(args.zcache_eval, ev_lab, args.side_eval, limit=args.limit_clips) if args.zcache_eval and ev_lab else []
    return tr, ev


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = make_parser().parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(args.out, exist_ok=True)
    fz, cfg, wm_info = load_wm(args.wm_ckpt, args.device)
    wm_sha0 = fz.sha()
    train, ev = load_data(args)
    if not train and not (args.eval_only or args.dump_dir):
        refuse("no training clips (check --zcache-train / --release-train / --side-train)")
    if args.eval_only or args.dump_dir:
        if not args.heads:
            refuse("--heads is required with --eval-only / --dump-dir")
        ck = torch.load(args.heads, map_location="cpu", weights_only=False)
        if ck["wm_sha256"] != wm_sha0:
            refuse("the heads were trained on another world model (wm sha256 differs)")
        a2 = copy.copy(args)
        for k in ("ctx_hidden", "ck_hidden", "g_hidden", "nav_dim"):
            setattr(a2, k, ck["args"][k])
        heads = S.StratHeads((ev or train)[0].z.shape[1], fz.d3, a2.nav_dim, a2.ctx_hidden, a2.ck_hidden, a2.g_hidden, step_s=fz.step_s).to(args.device)
        heads.load_state_dict(ck["heads"])
        if args.eval_only:
            r = evaluate(heads, fz, ev, a2, fz.step_s)
            r["heads_sha256"] = S.file_sha(args.heads)
            print_report(r)
            json.dump(r, open(os.path.join(args.out, "eval_strat_heads.json"), "w", encoding="utf-8", newline="\n"), indent=1)
        if args.dump_dir:
            meta = {"heads_sha256": S.file_sha(args.heads), "wm_sha256": wm_sha0, "tier": "T0", "ck_mode": "zero-action rollout (deployable)", "ck_norm": "x,y / (10 m/s * horizon), heading / 1 rad",
                    "strategic_status": "PAUSED (PI item 25)"}
            for split, cl in (("train", train), ("eval", ev)):
                for c in cl:
                    dump_clip(heads, fz, c, a2, fz.step_s, os.path.join(args.dump_dir, split, c.cid + ".npz"), meta)
            print(f"dumped {len(train)} train + {len(ev)} eval clips to {args.dump_dir}", flush=True)
        return 0

    heads, opt, smp, wts, gw, mix = build(args, fz, train)
    rec = selfcheck(args, heads, fz, opt, smp, wts, gw, mix, train, wm_info)
    sc_path = os.path.join(args.out, "selfcheck.json")
    json.dump(rec, open(sc_path, "w", encoding="utf-8", newline="\n"), indent=1)
    token = {"argv_sha16": argv_sha16(argv), "selfcheck_sha256": S.file_sha(sc_path), "wm_sha256": wm_sha0, "code_sha256": code_sha()}
    if args.selfcheck_only:
        json.dump(token, open(os.path.join(args.out, "SELFCHECK_TOKEN.json"), "w", encoding="utf-8", newline="\n"), indent=1)
        print("SELFCHECK PASS; token written", flush=True)
        return 0
    if not args.token or not os.path.exists(args.token):
        refuse("no --token: run with --selfcheck-only first")
    tk = json.load(open(args.token, encoding="utf-8"))
    for k in ("argv_sha16", "wm_sha256", "code_sha256"):
        if tk.get(k) != token[k]:
            refuse(f"token {k} {tk.get(k)} does not match this run ({token[k]})")
    step = 0
    last = os.path.join(args.out, "heads_last.pt")
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        smp.rng = np.random.default_rng(args.seed + step)
    t0 = time.time()
    heads.train()
    while step < args.steps:
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, args)
        batch = dev(smp.assemble(smp.draw()), args.device)
        out, (lc, lk, lg), aux = forward_losses(heads, fz, batch, wts, gw, fz.step_s)
        loss = mix.total(lc, lk, lg)
        if not torch.isfinite(loss):
            refuse(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            print(f"step {step} loss {float(loss):.4f} ctx {float(lc):.4f} ck {float(lk):.4f} g {float(lg):.4f} n_ck {aux['ck_n']} {(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            _save(last, heads, opt, step, args, wm_sha0)
    wm_sha1 = fz.sha()
    final = {"wm_sha256_before": wm_sha0, "wm_sha256_after": wm_sha1, "wm_bit_equal": wm_sha0 == wm_sha1, "steps": step, "s_per_step": (time.time() - t0) / max(step, 1)}
    json.dump(final, open(os.path.join(args.out, "train_done.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    if wm_sha0 != wm_sha1:
        refuse("the world model changed during training")
    if ev:
        r = evaluate(heads, fz, ev, args, fz.step_s)
        print_report(r)
        json.dump(r, open(os.path.join(args.out, "eval_strat_heads.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print("TRAIN DONE", flush=True)
    return 0


def _save(path, heads, opt, step, args, wm_sha):
    tmp = path + ".tmp"
    torch.save({"heads": heads.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args), "wm_sha256": wm_sha}, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
