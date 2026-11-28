# -*- coding: utf-8 -*-
"""Rewrite the search of `project_contact_free` as a MARGIN LADDER."""
import io

P = r"C:\Users\Admin\collproj\stack\tanitad\refs\contact_projection.py"
s = io.open(P, encoding="utf-8").read()

# 1. signature: add margin_fallback
old = "                         skip_first: bool = True,\n                         lateral: bool = True,\n"
new = ("                         skip_first: bool = True,\n"
       "                         lateral: bool = True,\n"
       "                         margin_fallback: bool = True,\n"
       "                         n_margin_rungs: int = 5,\n")
assert old in s
s = s.replace(old, new)

# 2. the r_need comment block -> the ladder
old_r = '''    #: the clearance the SEARCH must reach. The scorer flags with a strict ``<``
    #: at ``r_m``; clearing exactly ``r_m`` in float64 lands on the wrong side of
    #: an fp32 re-derivation about half the time (the MARGIN lesson of
    #: ``feasible_decode``, measured there at 0.609375). ``margin`` puts the output
    #: strictly outside by ~2 cm at m = 0 — physically nothing, ~1e5x the fp32 noise.
    r_need = (float(r_m) + float(margin_m)) * (1.0 + float(margin))
'''
new_r = '''    #: the clearance the SEARCH must reach. The scorer flags with a strict ``<``
    #: at ``r_m``; clearing exactly ``r_m`` in float64 lands on the wrong side of
    #: an fp32 re-derivation about half the time (the MARGIN lesson of
    #: ``feasible_decode``, measured there at 0.609375). ``margin`` puts the output
    #: strictly outside by ~2 cm at m = 0 — physically nothing, ~1e5x the fp32 noise.
    #
    # ⛔⛔ THE MARGIN LADDER, AND WHY THE FIRST VERSION WITHOUT IT WAS WRONG.
    # A requested margin can be UNACHIEVABLE for a given candidate while a smaller
    # one is perfectly achievable. The first version searched a single ``r_need``
    # and, failing, returned the INPUT unchanged and labelled it ``UNAVOIDABLE`` —
    # so raising the margin from 4 m to 6 m took ``fan_contact`` at the scorer's own
    # 2 m radius from **0.000000 back up to 0.007389 (227 candidates)**. MEASURED
    # 2026-09-05 on the 240 x 128 bank: asking for MORE safety delivered LESS,
    # because the guarantee was "clear at r_need OR untouched" rather than "as clear
    # as this candidate can be made". ⇒ the ladder retries at decreasing margins and
    # reports ``margin_achieved`` per candidate, so the guarantee becomes the one a
    # caller actually wants: **clear at the scorer's radius unless NO member of the
    # family clears it, and clear at the requested margin wherever that is possible.**
    rungs = [float(margin_m)]
    if margin_fallback and float(margin_m) > 0.0:
        k = max(2, int(n_margin_rungs))
        rungs = [float(margin_m) * (1.0 - i / (k - 1.0)) for i in range(k)]
        rungs = sorted({round(x, 9) for x in rungs}, reverse=True)
    r_needs = [(float(r_m) + g) * (1.0 + float(margin)) for g in rungs]
'''
assert old_r in s
s = s.replace(old_r, new_r)

# 3. the search body
i = s.index("    sig_grid = torch.linspace(0.0, 1.0, int(n_sigma)")
j = s.index("    # ---- re-integrate at the chosen (sigma, dkappa), uniformly")
body = '''    sig_grid = torch.linspace(0.0, 1.0, int(n_sigma), dtype=torch.float64, device=dev)
    G = sig_grid.numel()
    dk_grid = torch.linspace(-float(kappa_max), float(kappa_max), int(n_dkappa),
                             dtype=torch.float64, device=dev)
    K = dk_grid.numel()

    best_sigma = torch.ones(B, N, dtype=torch.float64, device=dev)
    best_dkappa = torch.zeros(B, N, dtype=torch.float64, device=dev)
    solved = torch.zeros(B, N, dtype=torch.bool, device=dev)
    via_lateral = torch.zeros(B, N, dtype=torch.bool, device=dev)
    margin_achieved = torch.full((B, N), float("nan"), dtype=torch.float64, device=dev)

    for rung, r_need in zip(rungs, r_needs):
        # ---- axis 1: the LONGITUDINAL retraction, SCANNED (never bisected) --- #
        todo = has_bn & (~solved)
        if not bool(todo.any()):
            break
        chunk = max(1, int(2_000_000 // max(1, N * G * S)))
        for b0 in range(0, B, chunk):
            b1 = min(B, b0 + chunk)
            if not bool(todo[b0:b1].any()):
                continue
            sg = sig_grid.reshape(1, 1, G)
            ss = s0[b0:b1, :, None] * sg                            # [b, N, G]
            hh = h0[b0:b1, :, None].expand_as(ss)
            aa = a0[b0:b1, :, None, :] * sg[..., None]              # [b, N, G, S-2]
            # ⛔⛔ ONE FORMULA, HERE AND IN THE FINAL RE-INTEGRATION. Writing the
            # lateral control as ``lat * sigma^2`` is algebraically identical to
            # ``kappa * v_mid^2`` ONLY while ``sigma * v_mid > MIN_SPEED``; under
            # the floor the two DIVERGE, so a scan written one way and a
            # re-integration written the other SEARCHES ONE PATH AND EMITS A
            # DIFFERENT ONE. MEASURED 2026-09-05: the scan reported clear and
            # ``assert_no_contact`` on the OUTPUT read a 1.719 m clearance against
            # a 2.0 m disc — the module would have shipped a colliding path while
            # its own bookkeeping said zero. ⭐ Caught only because the control
            # re-derives from the OUTPUT with the scorer instead of trusting the
            # search (RETRACTION #30, verbatim).
            vmg = (v_mid0[b0:b1, :, None, :] * sg[..., None]).clamp_min(min_speed)
            yy = kappa0[b0:b1, :, None, :] * vmg.pow(2)
            q = integrate_controls(ss, hh, aa, yy, dt=dt, clamp=clamp_friction,
                                   a_max=a_max, kappa_max=kappa_max, mu=mu,
                                   margin=margin, min_speed=min_speed)
            d = min_rel_distance(q, ld[b0:b1, :, None, :, :], skip_first=skip_first)
            ok = (d >= r_need) & todo[b0:b1].unsqueeze(-1)
            # the LARGEST clear sigma on the grid — exact on the grid, and no
            # monotonicity is assumed, which is the point of scanning: a lead that
            # REVERSES toward the ego makes slowing down worse, and a bisection
            # would return a colliding path in exactly that case.
            idx = torch.where(ok, sig_grid.reshape(1, 1, G),
                              torch.full_like(d, -1.0)).amax(dim=-1)
            got = idx >= 0.0
            best_sigma[b0:b1] = torch.where(got, idx.clamp_min(0.0),
                                            best_sigma[b0:b1])
            margin_achieved[b0:b1] = torch.where(
                got, torch.full_like(idx, float(rung)), margin_achieved[b0:b1])
            solved[b0:b1] = solved[b0:b1] | got

        # ---- axis 2: the LATERAL evade, on this rung's residual only -------- #
        need_lat = has_bn & (~solved)
        if lateral and bool(need_lat.any()):
            bi, ni = torch.nonzero(need_lat, as_tuple=True)
            M = bi.numel()
            sg = sig_grid.reshape(1, G, 1)
            dk = dk_grid.reshape(1, 1, K)
            step = max(1, int(400_000 // max(1, G * K * S)))
            for m0 in range(0, M, step):
                m1 = min(M, m0 + step)
                b_, n_ = bi[m0:m1], ni[m0:m1]
                mm = b_.numel()
                # ⛔ EVERY control axis is materialised to the FULL [m, G, K] grid.
                # A [m, G, 1] speed broadcasts correctly inside the arithmetic and
                # then sizes the integrator's accumulator wrongly — a shape bug a
                # looser integrator would have turned into a silent wrong answer.
                ss = (s0[b_, n_][:, None, None] * sg).expand(mm, G, K).contiguous()
                hh = h0[b_, n_][:, None, None].expand(mm, G, K).contiguous()
                aa = (a0[b_, n_][:, None, None, :]
                      * sg[..., None]).expand(mm, G, K, S - 2).contiguous()
                vm = (v_mid0[b_, n_][:, None, None, :]
                      * sg[..., None]).clamp_min(min_speed)
                kk = kappa0[b_, n_][:, None, None, :] + dk[..., None]
                yy = (kk * vm.pow(2)).expand(mm, G, K, S - 2).contiguous()
                q = integrate_controls(ss, hh, aa, yy, dt=dt, clamp=clamp_friction,
                                       a_max=a_max, kappa_max=kappa_max, mu=mu,
                                       margin=margin, min_speed=min_speed)
                d = min_rel_distance(q, ld[b_, n_][:, None, None, :, :],
                                     skip_first=skip_first)          # [m, G, K]
                ok = d >= r_need
                # ⭐ MINIMAL INTERVENTION, ORDERED: among the clear members prefer
                # the largest sigma (least progress given up), then the smallest
                # |dkappa| (least steering imposed). A pure "max sigma" rule would
                # happily pick a violent swerve to keep 1 % more speed.
                score = torch.where(ok, sg.expand_as(d) - 1e-6 * dk.abs().expand_as(d),
                                    torch.full_like(d, -1.0))
                flat = score.reshape(score.shape[0], -1).argmax(dim=-1)
                gi, ki = flat // K, flat % K
                got = ok.reshape(ok.shape[0], -1).any(dim=-1)
                best_sigma[b_, n_] = torch.where(got, sig_grid[gi], best_sigma[b_, n_])
                best_dkappa[b_, n_] = torch.where(got, dk_grid[ki], best_dkappa[b_, n_])
                margin_achieved[b_, n_] = torch.where(
                    got, torch.full_like(sig_grid[gi], float(rung)),
                    margin_achieved[b_, n_])
                via_lateral[b_, n_] = via_lateral[b_, n_] | got
                solved[b_, n_] = solved[b_, n_] | got

    # ⛔ an UNSOLVED candidate keeps its own controls and is LABELLED, never
    # silently half-retracted: a partial retraction that still collides is worse
    # than an honest "this one cannot be fixed", because it looks like a fix.
    unsolved = has_bn & (~solved)
    best_sigma = torch.where(unsolved, torch.ones_like(best_sigma), best_sigma)
    best_dkappa = torch.where(unsolved, torch.zeros_like(best_dkappa), best_dkappa)

'''
s = s[:i] + body + s[j:]

# 4. the reason block
old_reason = '''    reason = torch.full((B, N), REASON_CLEAR, dtype=torch.long, device=dev)
    reason = torch.where(~has_bn, torch.full_like(reason, REASON_NO_AGENT), reason)
    moved = has_bn & (~untouched)
    reason = torch.where(moved & any_clear,
                         torch.full_like(reason, REASON_LONGITUDINAL), reason)
    reason = torch.where(moved & (~any_clear) & lat_ok,
                         torch.full_like(reason, REASON_LATERAL), reason)
    reason = torch.where(has_bn & (~any_clear) & (~lat_ok),
                         torch.full_like(reason, REASON_UNAVOIDABLE), reason)
    info = {"reason": reason, "sigma": best_sigma.to(paths.dtype),
            "dkappa": best_dkappa.to(paths.dtype),
            "roundtrip_max_m": rt.to(paths.dtype),
            "uniform": q_uniform.to(paths.dtype),
            "r_need_m": torch.tensor(float(r_need), dtype=paths.dtype),
            "enabled": torch.ones((), dtype=torch.bool)}
    return out.to(paths.dtype), info'''
new_reason = '''    reason = torch.full((B, N), REASON_CLEAR, dtype=torch.long, device=dev)
    reason = torch.where(~has_bn, torch.full_like(reason, REASON_NO_AGENT), reason)
    moved = has_bn & (~untouched)
    reason = torch.where(moved & (~via_lateral),
                         torch.full_like(reason, REASON_LONGITUDINAL), reason)
    reason = torch.where(moved & via_lateral,
                         torch.full_like(reason, REASON_LATERAL), reason)
    reason = torch.where(unsolved,
                         torch.full_like(reason, REASON_UNAVOIDABLE), reason)
    info = {"reason": reason, "sigma": best_sigma.to(paths.dtype),
            "dkappa": best_dkappa.to(paths.dtype),
            "margin_achieved_m": margin_achieved.to(paths.dtype),
            "solved": solved, "via_lateral": via_lateral,
            "roundtrip_max_m": rt.to(paths.dtype),
            "uniform": q_uniform.to(paths.dtype),
            "margin_rungs_m": torch.tensor(rungs, dtype=paths.dtype),
            "r_need_m": torch.tensor(float(r_needs[0]), dtype=paths.dtype),
            "r_need_floor_m": torch.tensor(float(r_needs[-1]), dtype=paths.dtype),
            "enabled": torch.ones((), dtype=torch.bool)}
    return out.to(paths.dtype), info'''
assert old_reason in s
s = s.replace(old_reason, new_reason)

io.open(P, "w", encoding="utf-8").write(s)
print("PATCHED contact_projection.py with the margin ladder")
