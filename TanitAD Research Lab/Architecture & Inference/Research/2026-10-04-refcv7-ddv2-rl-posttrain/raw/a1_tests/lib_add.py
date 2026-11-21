

# =========================================================================== #
# SPEC_RL_A1 (2026-10-11): the KL anchor to the start checkpoint, and the kill-switch
# =========================================================================== #
@contextlib.contextmanager
def swapped_params(named, ref: dict):
    """Temporarily load ``ref`` (name -> tensor) into the ``named`` parameters, IN PLACE, and restore
    the originals bitwise on exit -- also on an exception. Used to run the REFERENCE policy (the
    frozen start checkpoint, the trainer's ``base``) through the same decoder without a second copy
    of the model. ``.grad`` is never touched.

    WARNING: Callers must run every reference forward BEFORE the first theta forward of the same backward
    graph: the swap bumps the parameters' version counters, which would invalidate a graph built
    earlier.
    """
    saved = {n: p.detach().clone() for n, p in named}
    try:
        with torch.no_grad():
            for n, p in named:
                p.copy_(ref[n].to(device=p.device, dtype=p.dtype))
        yield
    finally:
        with torch.no_grad():
            for n, p in named:
                p.copy_(saved[n])


def step_kl_to_reference(x0_theta: Tensor, x0_ref: Tensor, x_t: Tensor, t: int,
                         alphas_cumprod: Tensor, *, eta: float = D.DDV2.eta,
                         consts: D.DDV2Constants = D.DDV2) -> Tensor:
    """Per-trajectory KL( pi_ref(. | x_t) || pi_theta(. | x_t) ) of ONE denoising step t -> t-1, under
    the SURROGATE likelihood density the policy gradient itself uses (``ddim_logprob_step``: a
    diagonal Gaussian with std ``max(sigma_t, likelihood_std_floor)``, SPEC A4) -> ``[B, M]``, >= 0.

    Both densities share the std, so the KL is exactly ``sum_{S,2} (mu_theta - mu_ref)^2 /
    (2 lik^2)``. It is computed THROUGH ``ddim_logprob_step`` (``lp_ref - lp_theta`` evaluated at
    ``mu_ref``) so it inherits the release's mean arithmetic -- eps from the unclamped x0, the
    optional clamp after eps, the ``sqrt(1 - abar_prev - sigma^2)`` direction -- rather than
    re-deriving it. Gradient reaches the caller only through ``x0_theta``; ``x0_ref`` is detached.
    No random number is drawn (``prev_sample`` is always given).

    WARNING: This is NOT the KL of the SAMPLER's density (multiplicative per-(B, N) exploration noise,
    floor ``explore_std_floor``): that density is degenerate in the waypoint dimension. The anchor
    regularises the same object the policy gradient moves.
    """
    kw = dict(eta=eta, explore_std_floor=consts.explore_std_floor,
              likelihood_std_floor=consts.likelihood_std_floor,
              clip_sample=consts.clip_sample, clip_sample_range=consts.clip_sample_range)
    x0_ref = x0_ref.detach()
    x_t = x_t.detach()
    _, _, mu_ref = D.ddim_logprob_step(x0_ref, x_t, int(t), int(t) - 1, alphas_cumprod,
                                       prev_sample=x_t, **kw)
    mu_ref = mu_ref.detach()
    _, lp_ref, _ = D.ddim_logprob_step(x0_ref, x_t, int(t), int(t) - 1, alphas_cumprod,
                                       prev_sample=mu_ref, **kw)
    _, lp_th, _ = D.ddim_logprob_step(x0_theta, x_t, int(t), int(t) - 1, alphas_cumprod,
                                      prev_sample=mu_ref, **kw)
    return lp_ref.detach() - lp_th


#: The per-step stats the kill-switch reads (``rl_step`` + ``cmd_train``'s ``param_delta_norm``).
HALT_KEYS = ("grad_norm", "x0_out_of_box_frac_final_a_lon", "param_delta_norm")


def halt_check(history: list, *, gn_median: float = 0.0, oob_median: float = 0.0,
               pdn: float = 0.0, window: int = 20) -> str | None:
    """SPEC_RL_A1 sec. 5 kill-switch over the per-step ``history`` (dicts carrying ``HALT_KEYS``,
    oldest first). A threshold of 0 disables its rule. -> the joined trip reasons, or ``None``.

    * K1 -- median ``grad_norm`` over the last ``window`` steps > ``gn_median``
    * K2 -- median ``x0_out_of_box_frac_final_a_lon`` over the last ``window`` steps > ``oob_median``
    * K3 -- the CURRENT ``param_delta_norm`` (L2 distance to the start checkpoint) > ``pdn``

    K1/K2 are medians so one spike cannot trip them, and they are silent until ``window`` steps
    exist. K3 is instantaneous: the distance is cumulative and does not spike.
    """
    import statistics
    if int(window) < 1:
        raise Ddv2V7Error(f"halt window {window} < 1")
    if not history:
        return None
    reasons = []
    cur = history[-1]
    if pdn > 0 and float(cur["param_delta_norm"]) > pdn:
        reasons.append(f"K3 param_delta_norm {float(cur['param_delta_norm']):.4f} > {pdn:g}")
    if len(history) >= int(window):
        tail = history[-int(window):]
        if gn_median > 0:
            m = statistics.median(float(h["grad_norm"]) for h in tail)
            if m > gn_median:
                reasons.append(f"K1 median{window} grad_norm {m:.4f} > {gn_median:g}")
        if oob_median > 0:
            m = statistics.median(float(h["x0_out_of_box_frac_final_a_lon"]) for h in tail)
            if m > oob_median:
                reasons.append(f"K2 median{window} x0_out_of_box_frac_final_a_lon {m:.4f} > "
                               f"{oob_median:g}")
    return "; ".join(reasons) or None
