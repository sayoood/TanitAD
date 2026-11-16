"""Goal head (G4, G5): head-only training of the predicted level-3 goal latent of HWM-PLAN (SPEC_HWM sections 4 and 9). CPU or GPU; the world model is frozen and is never written.
Inputs at inference (hwm_plan.GoalInputs): level-1 latents up to NOW, the navigation command, the measured ego history and the vision-only strategic token (ctx, detached, optional). Nothing else.
Target: the encoded true-future level-3 latent (pointwise encoders of the frozen stack) at the observable level-3 steps j = 1 .. n_steps (frame NOW + j * s2 * s3); a step whose frame is not in the clip is masked and contributes exactly 0.0.
Prediction: base + delta, base = the current latent encoded to level 3 (the stub goal), delta = the head's output. Wired into the planner as the PredictedGoal fn (make_goal_fn, level 3).
Admissibility: the module is scanned for situation-classifier identifiers by test_hwm_plan.py; the strategic token enters detached (DETACH_CTX), so the head's loss puts exactly 0.0 gradient on whatever produced it.
RECOVER-NEXT style: train() reads back what was BUILT and probes the gradients before step 1 (selfcheck), and refuses to start when a check is red. Subcommands: selfcheck | build-data | train."""
import argparse, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn as nn
import hwm_plan as H

DETACH_CTX = True          # the strategic token is a detached input; the selfcheck reads this back


class GoalHeadSelfcheckError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- the head
class GoalHead(nn.Module):
    """delta = head(GoalInputs), (B, n_steps, d3). Input vector: the last latent, its change over the history, the flattened ego history (a, yaw rate) of the last `hist` frames, the navigation command; the strategic token enters through
    its own projection, added to the first hidden layer, after a detach. ctx_dim 0 builds the head without a context path (the off arm)."""

    def __init__(self, d1: int, d3: int, n_steps: int, nav_dim: int = 5, ctx_dim: int = 0, hist: int = 1, hidden: int = 512, depth: int = 2):
        super().__init__()
        self.d1, self.d3, self.n_steps, self.nav_dim, self.ctx_dim, self.hist = int(d1), int(d3), int(n_steps), int(nav_dim), int(ctx_dim), int(hist)
        self.cfg = dict(d1=self.d1, d3=self.d3, n_steps=self.n_steps, nav_dim=self.nav_dim, ctx_dim=self.ctx_dim, hist=self.hist, hidden=int(hidden), depth=int(depth))
        self.inp = nn.Linear(2 * self.d1 + 2 * self.hist + self.nav_dim, hidden)
        self.ctx_proj = nn.Linear(self.ctx_dim, hidden, bias=False) if self.ctx_dim > 0 else None
        self.mid = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(0, depth - 1))])
        self.out = nn.Linear(hidden, self.n_steps * self.d3)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, gi: H.GoalInputs) -> torch.Tensor:
        z = gi.z_hist
        b = z.shape[0]
        zl, dz = z[:, -1], z[:, -1] - z[:, 0]
        if gi.ego_hist is None:
            ego = z.new_zeros(b, self.hist * 2)
        else:
            e = gi.ego_hist[:, -self.hist:]
            if e.shape[1] < self.hist:
                e = torch.cat([e[:, :1].expand(-1, self.hist - e.shape[1], -1), e], 1)
            ego = e.reshape(b, -1).to(z.dtype)
        nav = z.new_zeros(b, self.nav_dim) if gi.nav is None else gi.nav.to(z.dtype)
        h = self.inp(torch.cat([zl, dz, ego, nav], -1))
        if self.ctx_proj is not None and gi.ctx is not None:
            c = gi.ctx.detach() if DETACH_CTX else gi.ctx
            h = h + self.ctx_proj(c.to(z.dtype))
        h = torch.relu(h)
        for lin in self.mid:
            h = torch.relu(lin(h))
        return self.out(h).reshape(b, self.n_steps, self.d3)


# ---------------------------------------------------------------------------------------------- targets, loss
def encode_now(st, z1):
    """The level-1 latent encoded to level 3 by the frozen pointwise encoders (no gradient): the stub goal and the base of the prediction."""
    with torch.no_grad():
        return st.initial_latents(z1, 3)[2]


def encode_targets(st, z_tgt):
    """z_tgt (B, n, D1) true-future level-1 latents at the level-3 steps -> (B, n, D3), no gradient."""
    b, n, d = z_tgt.shape
    with torch.no_grad():
        return st.initial_latents(z_tgt.reshape(b * n, d), 3)[2].reshape(b, n, -1)


def goal_loss(pred, tgt, valid):
    """Mean over the valid steps of the per-step mean squared error. A masked step contributes exactly 0.0 to the loss and to the gradient."""
    per = (pred - tgt).pow(2).mean(-1)
    contrib = per * valid.to(per.dtype)
    return contrib.sum() / valid.sum().clamp_min(1), contrib


def inputs_of(batch):
    return H.GoalInputs(batch["z_hist"], batch.get("nav"), batch.get("ego"), batch.get("ctx"))


def predict(head, st, gi):
    return encode_now(st, gi.z_hist[:, -1])[:, None] + head(gi)


def batch_loss(head, st, batch):
    gi = inputs_of(batch)
    pred = predict(head, st, gi)
    loss, contrib = goal_loss(pred, encode_targets(st, batch["z_tgt"]), batch["valid"])
    return loss, contrib, pred


def make_goal_fn(head, st, T3: int):
    """The PredictedGoal fn: fn(GoalInputs) -> (B, D3), the prediction at level-3 step T3. level 3: the planner takes it as the top goal as it is."""
    assert 1 <= T3 <= head.n_steps, (T3, head.n_steps)

    def fn(gi):
        with torch.no_grad():
            return predict(head, st, gi)[:, T3 - 1]

    fn.level = 3
    return fn


# ---------------------------------------------------------------------------------------------- frozen world model: fingerprint, freeze, trainable set
def wm_fingerprint(st) -> str:
    """sha256 over every tensor of the stack's state (parameters and buffers, names sorted): the world model must be bit-equal before and after a run."""
    h = hashlib.sha256()
    for k, v in sorted(st.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_wm(st):
    flags = [(p, p.requires_grad) for p in st.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)

    def restore():
        for p, f in flags:
            p.requires_grad_(f)

    return restore


def trainable_params(head, st):
    return list(head.parameters())


# ---------------------------------------------------------------------------------------------- the selfcheck (before step 1)
def _grads(head):
    return {n: (None if p.grad is None else p.grad.detach().abs().sum().item()) for n, p in head.named_parameters()}


def selfcheck(head, st, batch):
    """Read back what was BUILT and probe the gradients, before the first optimiser step. Returns the readback dict; raises GoalHeadSelfcheckError listing every red check.
    1 built levers: steps, context path, target level, detach flag, source scan.  2 non-zero gradient: every head parameter that the batch reaches has a non-zero gradient.  3 exact-0.0 control: with every step masked the loss and every head
    gradient read exactly 0.0.  4 world model: no gradient reaches it, no parameter of it is trainable in this run.  5 the strategic token: the loss puts no gradient on it (ctx.grad is None) and the world model's state is unchanged."""
    red, rb = [], {}
    fp0 = wm_fingerprint(st)
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    from tanitad.models.v8f_mini import scan_source_for_forbidden
    rb["built"] = dict(head_params=sum(p.numel() for p in head.parameters()), n_steps=head.n_steps, target_level=3, ctx_path=head.ctx_proj is not None, ctx_dim=head.ctx_dim, hist=head.hist,
                       detach_ctx=bool(DETACH_CTX), wm_trainable=sum(p.numel() for p in st.parameters() if p.requires_grad), source_scan=scan_source_for_forbidden(src))
    if rb["built"]["source_scan"]:
        red.append(("source scan", rb["built"]["source_scan"]))
    if rb["built"]["wm_trainable"] != 0:
        red.append(("world model is trainable", rb["built"]["wm_trainable"]))
    if head.ctx_proj is not None and not DETACH_CTX:
        red.append(("ctx path is not detached", None))
    # stale .grad on the stack's parameters is read, reported and cleared first (hwm_plan.decode_macro calls backward() through the macro encoder and leaves .grad on it; the weights are not touched), so that check 4 reads the PROBE only
    rb["wm_stale_grads_before_probe"] = sum(p.grad is not None for p in st.parameters())
    for p_ in st.parameters():
        p_.grad = None
    # 2 non-zero gradient
    b = dict(batch)
    if b.get("ctx") is not None:
        b["ctx"] = b["ctx"].detach().clone().requires_grad_(True)
    head.zero_grad(set_to_none=True)
    loss, _, _ = batch_loss(head, st, b)
    loss.backward()
    g = _grads(head)
    dead = [n for n, v in g.items() if not (n.startswith("ctx_proj") and b.get("ctx") is None) and (v is None or v == 0.0)]
    rb["probe"] = dict(loss=float(loss.detach()), min_param_grad_abs_sum=min((v for v in g.values() if v is not None), default=None), n_params_with_zero_grad=len(dead))
    if dead:
        red.append(("zero gradient on a head parameter", dead))
    if not torch.isfinite(loss):
        red.append(("probe loss not finite", float(loss.detach())))
    if b.get("ctx") is not None:
        rb["ctx_grad_is_none"] = b["ctx"].grad is None
        if b["ctx"].grad is not None:
            red.append(("the loss reached the strategic token", float(b["ctx"].grad.abs().sum())))
    if any(p.grad is not None for p in st.parameters()):
        red.append(("a gradient reached the world model", None))
    # 3 exact-0.0 control
    c = dict(b)
    c["valid"] = torch.zeros_like(batch["valid"])
    head.zero_grad(set_to_none=True)
    loss0, _, _ = batch_loss(head, st, c)
    loss0.backward()
    g0 = _grads(head)
    rb["control_exact_zero"] = dict(loss=float(loss0.detach()), max_grad_abs_sum=max((v for v in g0.values() if v is not None), default=0.0), n_params_none=sum(v is None for v in g0.values()))
    if float(loss0.detach()) != 0.0 or any(v != 0.0 for v in g0.values() if v is not None):
        red.append(("the exact-0.0 control reads non-zero", rb["control_exact_zero"]))
    head.zero_grad(set_to_none=True)
    # 5 world model unchanged by the probe
    if wm_fingerprint(st) != fp0:
        red.append(("the probe changed the world model", None))
    rb["wm_fingerprint"] = fp0
    rb["red"] = [r[0] for r in red]
    if red:
        raise GoalHeadSelfcheckError(json.dumps(rb["red"]) + " :: " + json.dumps([str(r[1])[:200] for r in red]))
    return rb


# ---------------------------------------------------------------------------------------------- training
def train(head, st, batch_fn, steps: int, lr: float = 1e-3, wd: float = 1e-4, seed: int = 0, log=print):
    """batch_fn(step) -> batch dict. The world model is frozen for the run and restored; its fingerprint is asserted equal at the end. selfcheck() runs first on batch_fn(0)."""
    torch.manual_seed(seed)
    fp0 = wm_fingerprint(st)
    restore = freeze_wm(st)
    hist = []
    try:
        rb = selfcheck(head, st, batch_fn(0))
        log("SELFCHECK " + json.dumps(rb))
        opt = torch.optim.AdamW(trainable_params(head, st), lr=lr, weight_decay=wd)
        head.train()
        for s in range(steps):
            loss, _, _ = batch_loss(head, st, batch_fn(s))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            hist.append(float(loss.detach()))
            if s % 50 == 0 or s == steps - 1:
                log(f"step {s} loss {float(loss.detach()):.6f}")
        head.eval()
    finally:
        restore()
    fp1 = wm_fingerprint(st)
    if fp1 != fp0:
        raise GoalHeadSelfcheckError("the world model changed during head-only training")
    return dict(selfcheck=rb, loss_history=hist, wm_fingerprint_before=fp0, wm_fingerprint_after=fp1)


@torch.no_grad()
def controls(head, st, batch, donor=None):
    """Eval-time controls on one batch (mean masked loss; lower is better): the head, the stub goal (the current latent encoded to level 3), a SHUFFLED goal (the head's predictions of other windows), a CONTEXT SWAP (the strategic token of other windows,
    only with a context path) and the head without its context. A head that reads vision, nav and context beats the stub; the shuffled goal and a context swap must be worse than the head where the context carries information."""
    b = batch["z_hist"].shape[0]
    donor = torch.roll(torch.arange(b), 1) if donor is None else torch.as_tensor(donor)
    tgt, valid = encode_targets(st, batch["z_tgt"]), batch["valid"]
    gi = inputs_of(batch)
    pred = predict(head, st, gi)
    out = dict(head=float(goal_loss(pred, tgt, valid)[0]), stub=float(goal_loss(encode_now(st, gi.z_hist[:, -1])[:, None].expand_as(pred), tgt, valid)[0]),
               shuffled=float(goal_loss(pred[donor], tgt, valid)[0]))
    if head.ctx_proj is not None and gi.ctx is not None:
        out["ctx_swap"] = float(goal_loss(predict(head, st, gi._replace(ctx=gi.ctx[donor])), tgt, valid)[0])
        out["no_ctx"] = float(goal_loss(predict(head, st, gi._replace(ctx=None)), tgt, valid)[0])
    return out


# ---------------------------------------------------------------------------------------------- data (windows at NOW, targets at the level-3 steps)
def build_goal_data(model, store, eids, nows, c3: int, n_steps: int, batch: int = 16):
    """Windows (eid, NOW cache step) from a ClipStore (eval or train split): z_now (N, D1), z_tgt (N, n_steps, D1) the level-1 latents at NOW + j * c3, valid (N, n_steps) (frame inside the clip and ok), ego (N, 1, 2) the measured
    (a, yaw rate) at NOW. History is the NOW frame only (the harness hands the head z_hist of length 1). The adapter, pos and wm.z1 are those of the GIVEN model."""
    Zn, Zt, V, E, keep_i = [], [], [], [], []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        f = torch.from_numpy(np.stack(buf)).float()
        with torch.no_grad():
            z = model.wm.z1(model.adapter(f) + model.pos[None, None])
        Zn.append(z[:, 0]); Zt.append(z[:, 1:])
        buf.clear()

    for i, (e, g) in enumerate(zip(eids, nows)):
        d = store.get(str(e))
        T = len(d["pose"])
        if g < 0 or g >= T:
            continue
        idx = np.array([g + j * c3 for j in range(n_steps + 1)])
        inside = idx < T
        ic = np.clip(idx, 0, T - 1)
        ok = (np.isfinite(d["p10f"][ic]).all(1) | (ic < 1)) & inside
        if not ok[0]:
            continue
        buf.append(np.asarray(d["feats"][ic]))
        V.append(ok[1:]); E.append(np.array([[np.nan_to_num(d["a"][g]), d["ch1"][g]]], np.float32)); keep_i.append(i)
        if len(buf) >= batch:
            flush()
    flush()
    return dict(z_now=torch.cat(Zn).numpy(), z_tgt=torch.cat(Zt).numpy(), valid=np.stack(V), ego=np.stack(E), eid=np.asarray(eids)[keep_i].astype("U36"), now=np.asarray(nows)[keep_i])


def split_by_episode(eids, frac=0.8, salt="goalhead-v1"):
    """Stable hash split of the episodes: (train indices, val indices); no episode in both."""
    uniq = sorted(set(str(e) for e in eids))
    rank = sorted(uniq, key=lambda e: (hashlib.sha256(f"{salt}|{e}".encode()).hexdigest(), e))
    tr = set(rank[:int(len(rank) * frac)])
    m = np.array([str(e) in tr for e in eids])
    return np.nonzero(m)[0], np.nonzero(~m)[0]


class NpzBatches:
    """Seeded batches from a build-data npz (optionally with nav (N, nav_dim) and stok (N, C) aligned row for row)."""

    def __init__(self, path_or_dict, idx=None, batch=256, seed=0):
        d = dict(np.load(path_or_dict, allow_pickle=False)) if isinstance(path_or_dict, str) else dict(path_or_dict)
        self.d, self.batch, self.seed = d, int(batch), int(seed)
        self.idx = np.arange(len(d["z_now"])) if idx is None else np.asarray(idx)

    def get(self, step, sel=None):
        if sel is None:
            r = np.random.default_rng(self.seed * 1000003 + int(step))
            sel = r.choice(self.idx, size=min(self.batch, len(self.idx)), replace=False)
        t = lambda k: None if k not in self.d else torch.from_numpy(np.asarray(self.d[k][sel])).float()
        return dict(z_hist=torch.from_numpy(self.d["z_now"][sel]).float()[:, None], z_tgt=t("z_tgt"), valid=torch.from_numpy(self.d["valid"][sel]).bool(), ego=t("ego"), nav=t("nav"), ctx=t("stok"))


def save_head(path, head, extra):
    torch.save(dict(cfg=head.cfg, state=head.state_dict(), extra=extra), path)


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = GoalHead(**ck["cfg"])
    h.load_state_dict(ck["state"])
    return h.eval(), ck["extra"]


# ---------------------------------------------------------------------------------------------- CLI
def _load_stack(ckpt, train_latents=None, tag=None):
    import hwm_data as D
    m, cfg, ck = D.load_model(ckpt)
    return m, m.wm, m.wm.stack


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build-data")
    b.add_argument("--ckpt", required=True); b.add_argument("--out", required=True); b.add_argument("--n-steps", type=int, default=6)
    b.add_argument("--split", default="eval", help="ClipStore split: eval (the local eval cache; the harness windows) or train (box 2: --store-root points at cache_full)")
    b.add_argument("--store-root", default=None); b.add_argument("--clips", default=None, help="json list of clip ids (train); default: the harness's eval clips")
    b.add_argument("--starts", type=int, default=17); b.add_argument("--n", type=int, default=0)
    t = sp.add_parser("train")
    t.add_argument("--ckpt", required=True); t.add_argument("--data", required=True); t.add_argument("--out", required=True)
    t.add_argument("--n-steps", type=int, default=6); t.add_argument("--steps", type=int, default=3000); t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--seed", type=int, default=0); t.add_argument("--hidden", type=int, default=512); t.add_argument("--depth", type=int, default=2)
    t.add_argument("--ctx", choices=["off", "on"], default="off", help="on: the data npz carries stok (N, C) and the head has a context path")
    t.add_argument("--quiet", action="store_true", help="do not print the loss lines")
    s = sp.add_parser("selfcheck")
    s.add_argument("--ckpt", required=True); s.add_argument("--data", required=True); s.add_argument("--n-steps", type=int, default=6); s.add_argument("--ctx", choices=["off", "on"], default="off")
    a = ap.parse_args(argv)
    m, wm, st = _load_stack(a.ckpt)
    c3 = int(st.cfg[1].stride) * int(st.cfg[2].stride)
    if a.cmd == "build-data":
        import hwm_data as D
        import wmact_lib as W
        if a.split == "eval" and not a.store_root:
            store = W.open_windows(16)[0].store
            eids, nows = D.plantest_pairs()
        else:
            tracks = W.D.load_native_tracks(os.path.join(a.store_root, "native_tracks.pt"))
            store = W.D.ClipStore(a.store_root, a.split, lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
            clips = json.load(open(a.clips, encoding="utf-8"))
            eids, nows = [], []
            for c in clips:
                T = len(store.get(c)["pose"]); hi = max(0, T - 1 - a.n_steps * c3)
                for g in np.unique(np.round(np.linspace(0, hi, a.starts)).astype(int)):
                    eids.append(c); nows.append(int(g))
            eids, nows = np.asarray(eids), np.asarray(nows)
        if a.n:
            sel = np.unique(np.linspace(0, len(eids) - 1, a.n).astype(int)); eids, nows = eids[sel], nows[sel]
        d = build_goal_data(m, store, eids, nows, c3, a.n_steps)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez(a.out, **d)
        print("build-data rows", len(d["eid"]), "valid steps", d["valid"].sum(0).tolist(), flush=True)
        return
    nb = NpzBatches(a.data)
    d1, d3 = int(st.cfg[0].latent_dim), int(st.cfg[2].latent_dim)
    ctx_dim = int(nb.d["stok"].shape[1]) if a.ctx == "on" else 0
    head = GoalHead(d1, d3, a.n_steps, ctx_dim=ctx_dim, nav_dim=int(nb.d["nav"].shape[1]) if "nav" in nb.d else 5, hidden=getattr(a, "hidden", 512), depth=getattr(a, "depth", 2))
    if a.cmd == "selfcheck":
        restore = freeze_wm(st)
        try:
            print(json.dumps(selfcheck(head, st, nb.get(0)), indent=1))
        finally:
            restore()
        return
    tr, va = split_by_episode(nb.d["eid"])
    ntr, nva = NpzBatches(nb.d, tr, a.batch, a.seed), NpzBatches(nb.d, va, 4096, a.seed)
    t0 = time.time()
    res = train(head, st, ntr.get, a.steps, a.lr, seed=a.seed, log=(lambda x: None) if a.quiet else print)
    ctl = controls(head, st, nva.get(0)) if len(va) else {}
    save_head(a.out, head, dict(selfcheck=res["selfcheck"], wm_fingerprint=res["wm_fingerprint_after"], controls_val=ctl, steps=a.steps, seconds=time.time() - t0, ckpt=os.path.basename(a.ckpt), n_train_rows=int(len(tr)), n_val_rows=int(len(va))))
    print("saved", a.out, "wm_fingerprint_equal", res["wm_fingerprint_before"] == res["wm_fingerprint_after"], flush=True)


if __name__ == "__main__":
    main()
