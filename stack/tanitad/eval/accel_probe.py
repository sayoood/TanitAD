"""Capacity/architecture probe instrument — "is the channel THERE, or is the head too small?"

WHY THIS EXISTS
---------------
The IDM's ``long_accel`` channel scores a NEGATIVE R² on every seed and both
domains while ``speed`` scores +0.86, and the banked baseline is **not separated
from a shuffled-latent control** (``…/2026-08-03-idm-derived-accel/``). Two
explanations fit that evidence equally well and they point at opposite work:

* **unrecoverable** — the frozen latents carry no usable information about the
  channel, so the lever is the representation or the data; or
* **unrecovered** — one head geometry, trained one way, failed to find it.

Telling them apart needs three things that a single-arm R² cannot supply, and
this module provides all three so that any future channel can be adjudicated the
same way instead of by argument:

1. **A capacity LADDER whose top rung is not a neural network.**
   :class:`DualRidge` solves ridge exactly in the dual, so the whole
   regularisation path costs ONE eigendecomposition, and an RBF kernel gives an
   effectively non-parametric arm with no optimiser, no learning rate and no
   epoch budget to blame. "The head was too small" cannot survive it.

2. **A DETECTION-SENSITIVITY control.** :func:`inject_signal` writes a known
   multiple of the target along a random direction of the latent and re-runs the
   identical probe. Sweeping the multiple converts an unfalsifiable "there is
   nothing there" into a quantitative floor: *nothing there above X % of latent
   variance, at this n*. Without it, a null result is indistinguishable from a
   probe that is simply not sensitive enough at N ~ 4.5 k with D = 2048.

3. **A shared loss with the shipped head.** :func:`probe_loss` is
   ``idm_head.idm_loss`` with a per-scalar mask added, and
   ``tests/test_accel_probe.py`` asserts the two are numerically identical when
   the mask is all-ones — so an architecture sweep is a sweep over ARCHITECTURE,
   not an accidental sweep over the objective.

Only torch + numpy, no pod paths, so every piece is unit-testable on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "r2_score", "standardize", "window_features", "FEATURE_BUILDERS",
    "DualRidge", "rbf_kernel", "inject_signal",
    "MLPHead", "GRUHead", "probe_loss", "fit_probe_head", "predict_head",
]


# --------------------------------------------------------------------------- #
# scoring                                                                     #
# --------------------------------------------------------------------------- #
def r2_score(pred, gt) -> float:
    """1 - SS_res/SS_tot in float64. A CONSTANT gt gives 0.0, not a NaN.

    The flat case is real here: one comma2k19 episode ships ``long_accel``
    identically 0.0, so a bootstrap draw can contain only flat windows. Returning
    NaN there silently drops draws and narrows the interval; returning 0.0 says
    "explained nothing", which is what a constant target means.
    """
    g = np.asarray(gt, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    sst = float(((g - g.mean()) ** 2).sum())
    if sst <= 0.0:
        return 0.0
    return 1.0 - float(((g - p) ** 2).sum()) / sst


def standardize(Xtr: Tensor, *others: Tensor, eps: float = 1e-6):
    """Zero-mean/unit-var columns using TRAIN statistics only -> (Xtr, *others)."""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(eps)
    return tuple(((X - mu) / sd) for X in (Xtr, *others))


# --------------------------------------------------------------------------- #
# feature builders over a latent window Z [N, W, D]                           #
# --------------------------------------------------------------------------- #
def window_features(Z: Tensor, kind: str) -> Tensor:
    """``Z`` [N, W, D] -> [N, F] for one of :data:`FEATURE_BUILDERS`.

    ``diff`` exists because acceleration is a DERIVATIVE: if the latent encodes
    speed at all, the information about its change lives in the difference of
    symmetric window positions, not in the centre token that the shipped linear
    readout reads. Giving a linear probe that basis is the cheapest way to find
    out whether the shipped head's failure is a basis problem.

    ``tdiff`` / ``abstdiff`` (added 2026-08-03, D-LATENT) are the ADJACENT-frame
    bases. ``diff`` differences positions symmetric about the centre, so it
    spans lags of 2, 4 and ``w-1`` steps and NEVER forms the 1-step difference;
    for a channel whose physics is a per-step derivative that is the wrong
    stencil. ``tdiff`` is the full first-difference sequence (w-1 blocks) — the
    discrete velocity of the representation — and ``abstdiff`` is its magnitude,
    i.e. a sign-blind MOTION-ENERGY basis. On a pixel substrate ``abstdiff`` is a
    crude optical-flow-magnitude proxy, which is the quantity whose own time
    derivative is longitudinal acceleration; on a latent substrate it asks
    whether the representation moves *at all* in a target-correlated way.
    """
    if Z.ndim != 3:
        raise ValueError(f"Z must be [N, W, D], got {tuple(Z.shape)}")
    n, w, d = Z.shape
    c = w // 2
    if kind == "centre":
        return Z[:, c]
    if kind == "window":
        return Z.reshape(n, w * d)
    if kind == "diff":
        offs = [o for o in (1, 2, c) if c + o < w and c - o >= 0]
        return torch.cat([Z[:, c + o] - Z[:, c - o] for o in offs], dim=1)
    if kind == "centre_diff":
        return torch.cat([window_features(Z, "centre"),
                          window_features(Z, "diff")], dim=1)
    if kind in ("tdiff", "abstdiff"):
        if w < 2:
            raise ValueError(f"{kind!r} needs W >= 2, got W={w}")
        dz = (Z[:, 1:] - Z[:, :-1]).reshape(n, (w - 1) * d)
        return dz.abs() if kind == "abstdiff" else dz
    raise KeyError(f"unknown feature kind {kind!r}; known: {FEATURE_BUILDERS}")


FEATURE_BUILDERS = ("centre", "window", "diff", "centre_diff",
                    "tdiff", "abstdiff")


# --------------------------------------------------------------------------- #
# exact ridge in the dual — the WHOLE alpha path for one eigendecomposition    #
# --------------------------------------------------------------------------- #
def rbf_kernel(A: Tensor, B: Tensor, gamma: float) -> Tensor:
    """exp(-gamma * ||a-b||^2), computed without materialising the differences."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d2 = (a2 + b2 - 2.0 * (A @ B.T)).clamp_min(0.0)
    return torch.exp(-gamma * d2)


class DualRidge:
    """Kernel ridge in the dual, with the regularisation path free after one eigh.

    Why the dual: the window feature set is 9 x 2048 = 18,432 columns against
    ~4.5 k training windows, so the primal normal equations are a 18,432^2 solve
    for a problem whose rank is at most n. The dual is exact, smaller, and its
    eigendecomposition makes EVERY alpha a single vector division — which is what
    turns "capacity" into a swept axis instead of one arbitrary setting.

    The kernel is CENTRED in feature space, so the model always carries an
    implicit intercept and a linear kernel here equals primal ridge on centred
    features (asserted in the unit test).
    """

    def __init__(self, Xtr: Tensor, Ytr: Tensor, *, kernel: str = "linear",
                 gamma: float | None = None, matmul_device: str | None = None,
                 matmul_dtype: torch.dtype = torch.float64):
        #: the Gram of an 18,432-column feature block is 7.6e11 flops. On the
        #: GPU in fp32 that is 0.2 s against ~2 min on CPU fp64, and the fp32
        #: relative error (~1e-5) is orders below the smallest ridge alpha we
        #: ever select, so it cannot move a verdict. Default stays exact CPU
        #: fp64 so the unit tests pin the reference semantics.
        self.kernel = kernel
        self.gamma = gamma
        self._mm_dev = matmul_device
        self._mm_dtype = matmul_dtype
        self.Xtr = Xtr.double()
        K = self._kernel(self.Xtr)
        n = K.shape[0]
        self.n = n
        self._krow = K.mean(0, keepdim=True)            # [1, n]
        self._kall = float(K.mean())
        Kc = K - self._krow - self._krow.T + self._kall
        self.Ymean = Ytr.double().mean(0, keepdim=True)
        Yc = Ytr.double() - self.Ymean
        w, V = torch.linalg.eigh(Kc)
        self.w = w.clamp_min(0.0)
        self.V = V
        self.VtY = V.T @ Yc

    def _kernel(self, A: Tensor, B: Tensor | None = None) -> Tensor:
        B = self.Xtr if B is None else B
        if self._mm_dev is not None:
            A = A.to(self._mm_dev, self._mm_dtype)
            B = B.to(self._mm_dev, self._mm_dtype)
        if self.kernel == "linear":
            K = A @ B.T
        elif self.kernel == "rbf":
            if self.gamma is None:
                raise ValueError("rbf kernel needs gamma")
            K = rbf_kernel(A, B, self.gamma)
        else:
            raise KeyError(f"unknown kernel {self.kernel!r}")
        return K.to("cpu", torch.float64)

    def dual_coef(self, alpha: float) -> Tensor:
        return self.V @ (self.VtY / (self.w[:, None] + float(alpha)))

    def predict(self, X: Tensor, alpha: float) -> Tensor:
        """-> [M, out] predictions at regularisation ``alpha``."""
        Kte = self._kernel(X.double())
        Ktc = Kte - Kte.mean(1, keepdim=True) - self._krow + self._kall
        return Ktc @ self.dual_coef(alpha) + self.Ymean

    def predict_train(self, alpha: float) -> Tensor:
        return self.predict(self.Xtr, alpha)

    @staticmethod
    def alpha_grid(n_per_decade: int = 2, lo: float = -4, hi: float = 8) -> list[float]:
        k = int(round((hi - lo) * n_per_decade)) + 1
        return [float(10.0 ** e) for e in np.linspace(lo, hi, k)]


# --------------------------------------------------------------------------- #
# detection-sensitivity control                                                #
# --------------------------------------------------------------------------- #
def inject_signal(Z: Tensor, y, frac: float, seed: int = 0,
                  centre_only: bool = True, standardize_y: bool = True,
                  rms: float | None = None,
                  direction: Tensor | None = None) -> tuple[Tensor, dict]:
    """Write ``frac`` of the latent's RMS magnitude as a copy of ``y`` and return it.

    ⚠️ THIS IS A CONTROL, NOT A LEAK. It never touches the targets or the split;
    it manufactures a latent in which the answer is KNOWN to be present at a
    known strength, so that a null result on the real latent can be quoted with
    a sensitivity: *not present above this strength, at this n*. A null result
    without it is unfalsifiable — indistinguishable from a probe too blunt to
    see a real but weak signal.

    ``y`` is standardised first, so ``frac`` is a pure amplitude ratio against
    the latent's own RMS element magnitude and is comparable across arms.

    ⚠️ When injecting into a TRAIN and a HELD-OUT block that must stay
    comparable, pass ``standardize_y=False`` with an already-train-standardised
    ``y`` and an explicit ``rms``. Re-standardising each block separately would
    give the two blocks different injection gains, and the probe would then be
    partly measuring that mismatch.
    """
    Z = Z.clone()
    yv = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
    if standardize_y:
        yv = (yv - yv.mean()) / yv.std().clamp_min(1e-9)
    g = torch.Generator().manual_seed(seed)
    d = Z.shape[-1]
    #: ``direction`` lets the caller BRACKET the probe's sensitivity. A random
    #: direction is the hardest case for ridge (its implicit prior favours
    #: high-variance directions); the latent's leading principal direction is the
    #: easiest. A real signal sits somewhere between, so quoting a floor from one
    #: of them alone would over- or under-state the null.
    u = (torch.randn(d, generator=g, dtype=torch.float64)
         if direction is None else direction.double().clone())
    u = u / u.norm()
    rms = float(Z.double().pow(2).mean().sqrt()) if rms is None else float(rms)
    amp = frac * rms * float(d) ** 0.5          # so ||added|| = frac*rms*sqrt(d)
    add = (amp * yv[:, None] * u[None, :]).to(Z.dtype)
    c = Z.shape[1] // 2
    if centre_only:
        Z[:, c] = Z[:, c] + add
    else:
        Z = Z + add[:, None, :]
    meta = {"frac_of_latent_rms": float(frac), "latent_rms": rms,
            "injected_rms": float(add.double().pow(2).mean().sqrt()),
            "centre_only": bool(centre_only), "seed": int(seed),
            "direction": "random" if direction is None else "supplied"}
    return Z, meta


# --------------------------------------------------------------------------- #
# neural heads sharing the shipped output contract                             #
# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    """Plain MLP on the centre token or the flattened window.

    The point of this arm is that it removes the temporal transformer entirely:
    if a 2-layer MLP on the flattened window matches the transformer, the
    transformer's inductive bias is not what is limiting the channel.
    """

    def __init__(self, state_dim: int, hidden: int = 512, depth: int = 2,
                 window: int = 9, n_scalars: int = 4, n_horizons: int = 4,
                 mode: str = "centre"):
        super().__init__()
        if mode not in ("centre", "window"):
            raise KeyError(f"mode must be centre|window, got {mode!r}")
        self.mode = mode
        self.window = window
        self.center = window // 2
        self.n_horizons = n_horizons
        f_in = state_dim if mode == "centre" else state_dim * window
        layers: list[nn.Module] = []
        d = f_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.norm = nn.LayerNorm(d)
        self.scalar_head = nn.Linear(d, n_scalars)
        self.traj_head = nn.Linear(d, 2 * n_horizons)

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        b = z.shape[0]
        x = z[:, self.center] if self.mode == "centre" else z.reshape(b, -1)
        h = self.norm(self.body(x))
        return {"scalars": self.scalar_head(h),
                "traj": self.traj_head(h).reshape(b, self.n_horizons, 2)}


class GRUHead(nn.Module):
    """Bidirectional GRU over the window, read out at the CENTRE position.

    A different sequence inductive bias from the transformer (local, recurrent,
    order-sensitive) at comparable parameter count — so "the sequence model was
    the wrong kind" is covered as well as "it was too small".
    """

    def __init__(self, state_dim: int, d_model: int = 128, layers: int = 2,
                 window: int = 9, n_scalars: int = 4, n_horizons: int = 4):
        super().__init__()
        self.window = window
        self.center = window // 2
        self.n_horizons = n_horizons
        self.in_proj = nn.Linear(state_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=layers, batch_first=True,
                          bidirectional=True)
        self.norm = nn.LayerNorm(2 * d_model)
        self.scalar_head = nn.Linear(2 * d_model, n_scalars)
        self.traj_head = nn.Linear(2 * d_model, 2 * n_horizons)

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        b = z.shape[0]
        h, _ = self.rnn(self.in_proj(z))
        x = self.norm(h[:, self.center])
        return {"scalars": self.scalar_head(x),
                "traj": self.traj_head(x).reshape(b, self.n_horizons, 2)}


# --------------------------------------------------------------------------- #
# loss + generic trainer                                                       #
# --------------------------------------------------------------------------- #
def probe_loss(pred: dict[str, Tensor], scal: Tensor, traj: Tensor, std,
               *, mask: Tensor | None = None, traj_weight: float = 1.0,
               traj_scale: float = 10.0, huber_beta: float = 1.0) -> dict:
    """``idm_head.idm_loss`` with a per-scalar MASK.

    With ``mask=None`` this is numerically IDENTICAL to the shipped loss (unit
    test). The mask exists for the single-task arm: if the four-way multitask
    objective is what starves ``long_accel``, an accel-only arm must show it, and
    that arm has to differ from the shipped one in the objective ALONE.
    """
    ps = std.norm(pred["scalars"])
    ts = std.norm(scal)
    if mask is None:
        scal_l = F.huber_loss(ps, ts, delta=huber_beta)
    else:
        m = mask.to(ps).reshape(1, -1)
        if float(m.sum()) <= 0:
            raise ValueError("scalar mask selects no channel")
        per = F.huber_loss(ps, ts, delta=huber_beta, reduction="none")
        scal_l = (per * m).sum() / (m.sum() * per.shape[0])
    traj_l = F.smooth_l1_loss(pred["traj"] / traj_scale, traj / traj_scale,
                              beta=huber_beta)
    return {"loss": scal_l + traj_weight * traj_l,
            "scalar_loss": scal_l.detach(), "traj_loss": traj_l.detach()}


def fit_probe_head(make_module, train, *, epochs: int = 12, batch: int = 256,
                   lr: float = 3e-4, wd: float = 0.01, traj_weight: float = 1.0,
                   seed: int = 0, device: str = "cpu",
                   scalar_mask=None, log=None):
    """Fit ANY module with the shipped optimiser recipe -> (module, standardizer).

    Mirrors ``idm_head.fit_head`` — AdamW(lr, wd) + cosine over epochs*steps,
    Huber on standardised scalars + scaled smooth-L1 on the trajectory — so an
    architecture sweep varies the architecture and nothing else.
    """
    from idm_head import Standardizer

    torch.manual_seed(seed)
    Ztr, Str, Ttr = train[0], train[1], train[2]
    std = Standardizer.fit(Str)
    mask = None if scalar_mask is None else torch.as_tensor(
        scalar_mask, dtype=torch.float32)
    mod = make_module().to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=wd)
    n = Ztr.shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, epochs * max(1, n // batch))
    for ep in range(epochs):
        mod.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = mod(Ztr[idx].to(device))
            ld = probe_loss(out, Str[idx].to(device), Ttr[idx].to(device), std,
                            mask=mask, traj_weight=traj_weight)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step()
            sched.step()
            tot += float(ld["loss"].detach()) * len(idx)
        if log:
            log(f"    epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    return mod, std


@torch.no_grad()
def predict_head(mod, Z: Tensor, *, device: str = "cpu", batch: int = 512):
    """-> (scalars [N,4], traj [N,H,2]) on CPU."""
    mod.eval()
    S, T = [], []
    for i in range(0, Z.shape[0], batch):
        o = mod(Z[i:i + batch].to(device))
        S.append(o["scalars"].float().cpu())
        T.append(o["traj"].float().cpu())
    return torch.cat(S), torch.cat(T)
