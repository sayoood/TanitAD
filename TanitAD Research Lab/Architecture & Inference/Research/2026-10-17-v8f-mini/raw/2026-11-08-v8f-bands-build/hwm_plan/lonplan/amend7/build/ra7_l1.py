"""AMEND_HWMPLAN_7 section 6.1: the RA pass with RA7 in place of RA. Everything is imported from AMEND_5's ra_l1 unchanged except two things, forked here:
  MLPReadout / load_readout7   the readout RA7([z0, z0 + D]) (frozen, eval), the sha256 check BEFORE parsing (C-RA7-SHA)
  plan_arrays7                 ra_l1.plan_arrays whose only changes are that every readout call passes the batch's z0 (RAL1, NEG, SHUF and ORACLE paths) and that `built` records kind `mlp`
amend7_env installs both into ra_l1 (load_readout, plan_arrays) so ra_l1.run_arm / load_ctx use them. Import amend7_env FIRST.
Tier T0. CPU only."""
import hashlib, io, os, sys
sys.dont_write_bytecode = True
import numpy as np
import torch
import amend5_stage as ST
import ra_l1 as R
import plantest_lib as P

ACC, KS = R.ACC, R.KS


class MLPReadout:
    """ra7(Dz | z0) = MLP([z0, z0 + Dz]), standardised by the fit rows' mean / sd. Frozen, eval mode. Dz (..., D); z0 (..., D) or lower rank (broadcast to Dz's shape)."""
    def __init__(self, net, mu, sd, sha256, path, dtype=torch.float32, lam=0.0):
        self.net = net.to(dtype).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mu, self.sd = torch.as_tensor(mu).to(dtype), torch.as_tensor(sd).to(dtype)
        self.dtype = dtype
        self.sha256, self.path, self.lam = sha256, path, lam
        self.d = int(self.mu.shape[0]) // 2
        self.perm = np.random.RandomState(0).permutation(self.d)         # the permutation of ra_l1.py:75
        self.inv = torch.as_tensor(np.argsort(self.perm))                  # D'[i] = D[inv[i]]: for a readout linear in D this equals RA-SHUF's W[perm]
        self.W = torch.zeros(1, dtype=dtype)                              # dtype handle for code that asks ra.W.dtype (ra_l1 / amend5_controls)

    def read(self, Dz, z0, shuf=False):
        if shuf:
            Dz = Dz[..., self.inv]
        z = z0
        while z.dim() < Dz.dim():
            z = z.unsqueeze(-2)
        z = z.expand_as(Dz)
        x = torch.cat([z, z + Dz], -1)
        return self.net((x - self.mu) / self.sd)[..., 0]


def net_from_state(state, width, d_in):
    from mlp7 import make_net
    net = make_net(d_in, width)
    net.load_state_dict(state)
    return net


def load_readout7(path, k, sha=None):
    """C-RA7-SHA: the file's sha256 must equal the frozen value (RA7_FIT_<k>.json, hashed from the bytes that are then parsed); one changed byte refuses. With no RA7_FIT record the frozen value is '0'*64: every use refuses."""
    want = ST.RA_SHA[k] if sha is None else sha
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise R.RefusedRun(f"readout {path} sha256 {got[:12]} != the frozen {want[:12]}")
    M = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if M.get("kind") != "mlp":
        raise R.RefusedRun(f"readout {path} is not an RA7 file (kind {M.get('kind')!r})")
    net = net_from_state(M["net_state"], int(M["width"]), len(M["mu"]))
    return MLPReadout(net, torch.tensor(np.asarray(M["mu"], np.float64), dtype=torch.float32), torch.tensor(np.asarray(M["sd"], np.float64), dtype=torch.float32), got, path)


def plan_arrays7(ctx, base, win, mode, src="none", mean_src="none", mutate=None, built=None, batch=P.BATCH):
    """ra_l1.plan_arrays with RA7. One base plan file (dict of arrays) -> the arm's plan file (dict). Same arguments, mutation hooks and outputs; the readout is read as ra.read(Dz, z0)."""
    built = built if built is not None else {}
    n = len(base["plan_raw"])
    assert win["z"].shape[0] == n and win["A"].shape[0] == n, (win["z"].shape, n)
    out = {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in base.items()}
    mode_run = "l1" if (mutate == "off_runs_pass" and mode == "off") else mode
    src_run = ("l1" if src in ("none", None) else src) if mutate == "off_runs_pass" else src
    if mutate == "wrong_src" and src_run in ("l1", "l2"):
        src_run = "l1" if src_run == "l2" else "l2"
    built.update(mode=mode, src="none", kind="off", mirror=False, perm=False, readout_sha256=ctx.ra.sha256, mean_src="none", src_key=None)
    if mode == "off" and mean_src == "none" and mode_run == "off":
        return out, built
    dt = ctx.ra.dtype
    z = torch.from_numpy(np.asarray(win["z"])).to(dt)
    A0 = np.asarray(win["A"], np.float64)[:, 0]
    h1 = ctx.h1
    hc_all = ctx.hold_c(A0)[:, :h1]
    a1 = np.asarray(base["a1_raw"], np.float64)
    yaw_b_all = torch.from_numpy(a1[:, :h1, 1] / ctx.ch1_scale).to(dt)
    if mean_src != "none":
        key = R.SRC_KEY[mean_src]
        built.update(kind="mean", mean_src=mean_src, src_key=key)
        hc_raw = hc_all[:, :KS, 0].numpy().astype(np.float64) * ACC
        flat = R.mean_assembly(np.asarray(base[key], np.float64)[:, :KS, 0], hc_raw, float(ctx.lo1[0]) * ACC, float(ctx.hi1[0]) * ACC)
        out["plan_raw"][:, :KS, 0] = flat
        return out, built
    built.update(kind="mlp", mode=mode_run, src=(src_run if mode_run in ("l1", "l2", "neg", "shuf") else "none"), mirror=(mode_run == "neg"), perm=(mode_run == "shuf"))
    if mode_run in ("l1", "l2", "neg", "shuf"):
        key = R.SRC_KEY[src_run]
        built["src_key"] = key
        srcf = np.asarray(base[key], np.float64)
        if srcf.shape[1] < KS:
            raise R.RefusedRun(f"{key} has {srcf.shape[1]} knots, RA7 reads 10")
    acc_out = np.zeros((n, KS)); cost_out = np.zeros(n); hit_out = np.zeros(n, bool); it_out = np.zeros(n); tgt_out = np.zeros(n)
    full12 = None
    if mutate == "write12":
        full12 = np.zeros((n, h1))
    for b0 in range(0, n, batch):
        ii = slice(b0, min(n, b0 + batch))
        z0, hc, yaw_b = z[ii, 0], hc_all[ii], yaw_b_all[ii]
        use_shuf = mode_run == "shuf"
        ra_fn = (lambda Dz, _s=use_shuf, _z0=z0: ctx.ra.read(Dz, _z0, shuf=_s))               # RA7 reads [z0, z0 + D]: the batch's z0 on every call
        built["perm"] = bool(use_shuf and not np.array_equal(ctx.ra.perm, np.arange(len(ctx.ra.perm))))
        with torch.no_grad():
            if mode_run == "oracle":
                target = ctx.ra.read(z[ii, KS] - z0, z0)
                yaw_use = hc[:, :, 1]
            else:
                yaw_use = yaw_b
                srcacc = torch.from_numpy(a1[ii, :h1, 0] / ACC).to(dt)
                srcacc[:, :KS] = torch.from_numpy(srcf[ii, :KS, 0] / ACC).to(dt)
                target = ra_fn(R.roll(ctx.st, z0, torch.stack([srcacc, yaw_use], -1).unsqueeze(1))[:, 0] - z0)
                if mode_run == "neg":
                    r_h = ctx.ra.read(R.roll(ctx.st, z0, torch.stack([hc[:, :, 0], yaw_use], -1).unsqueeze(1))[:, 0] - z0, z0)
                    target = 2 * r_h - target
        ni = 1 if mutate == "n_iters1" else ctx.n_iters
        acc, cst, nit, hit = R.ra_pass(ctx.st, ctx.lo1, ctx.hi1, ra_fn, z0, hc[:, :, 0], yaw_use, target, n_iters=ni)
        acc_out[ii] = acc[:, :KS].numpy().astype(np.float64) * ACC
        cost_out[ii], hit_out[ii], it_out[ii], tgt_out[ii] = cst.numpy(), hit.numpy(), nit, target.numpy()
        if full12 is not None:
            full12[ii] = acc.numpy().astype(np.float64) * ACC
    out["plan_raw"][:, :KS, 0] = acc_out
    if mutate == "write12":
        out["plan_raw"][:, :h1, 0] = full12
    if mutate == "write_yaw":
        out["plan_raw"][:, :h1, 1] = hc_all[:, :, 1].numpy().astype(np.float64) * ctx.ch1_scale + 1e-3
    out.update(ra_acc_raw=acc_out, ra_cost=cost_out, ra_hit_box=hit_out, ra_iters=it_out, ra_target_scaled=tgt_out)
    return out, built
