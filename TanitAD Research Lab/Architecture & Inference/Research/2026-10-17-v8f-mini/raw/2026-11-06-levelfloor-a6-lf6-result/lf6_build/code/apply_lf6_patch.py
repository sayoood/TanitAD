"""Apply the LF6 patch (``--floor-lift-cap DELTA``, the bounded-lift floor F_B, PREREG_LEVELFLOOR_A6 sha256 fbf5de02...) to the LF5 tool (sha256 88b275a1..., lf5_build/tool). One-shot; each replacement must match exactly once.
Usage: python -I apply_lf6_patch.py <LF5 v8f_t1_eval_ri_lf5.py 88b275a1> <out v8f_t1_eval_ri.py>     (the patched tool keeps the name v8f_t1_eval_ri.py inside the read tree; level_floor.py next to it is UNCHANGED, sha 2781e23a)

F_B (card section 2) = A5's F with one more term in the floor speed:  v_f(tau) = min( kappa * v_hat (lead cap as A3), v0 + 1.0 * tau, v_O(tau) + DELTA ),  s_f(t) = integral_0^t v_f,  s' = max(s, s_f), resampled along O's polyline as in A3.
v_O is O's planned speed after S: on the interval (t_{k-1}, t_k] it is the chord of the origin-prepended polyline / 0.2 s (knot 0 = origin). The integral is exact: per interval the integrand is min(c_k, v0' + tau), c_k = min(v_f, v_O,k + DELTA), a ramp that meets a constant.
Three choices the card leaves open, fixed here and tested:
  * a row where the DELTA term is nowhere the strict minimum (at no knot is v_O,k + DELTA < v_f and < v0' + t_k) reads its s_f from level_floor.ramp_progress, A5's closed form, unchanged: F_B is F on that row, bit for bit (this is what makes DELTA = 1e9 the LF5 tool exactly);
  * s_f is clipped at s_k + DELTA * t_k (the card's own bound, s' - s <= DELTA * t, true in exact arithmetic): it only removes float round-off, so DELTA = 0 never binds (the analytic zero control) and the bound holds on the floats;
  * ``lf_vf`` (the tick's v_f) stays the SCALAR first term, kappa * v_hat with the lead cap, as in A3-A5, so every pinned control of the earlier readers reads it unchanged. New per-tick fields (only with the flag): ``lf_dterm`` (a bound tick where the DELTA term is
    the strict minimum at some knot where F lifts), ``lf_dterm_knots``, ``lf_s`` (s_k, [30]) and ``lf_sp`` (s'_k, [30]).
Absent flag = the LF5 tool bit for bit in every rollout output. ONE defect of the LF5 tool is fixed (A5_NOTE_READER_FIX): with SIGMA 0 and source ``logged`` the LF5 tool writes ``floor_vhat_noise_seed`` = None into the ticks npz (a 0-d object array the pinned loader, allow_pickle=False,
refuses); the LF6 tool omits the key when the seed is None. The result-JSON header is unchanged (it still records the seed as null)."""
import hashlib
import sys

SRC_SHA = "88b275a1ca67894be01907f72b71302b72f6e0b4fa0af067f4e2c3d5286aba77"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the LF5 tool 88b275a1"
assert b"\r" not in b, "the source tool must be LF"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. docstring ---------------------------------------------------------------------------------------------------------
rep('''and the header records ``floor_vhat_noise`` and ``floor_vhat_noise_seed``.
''', '''and the header records ``floor_vhat_noise`` and ``floor_vhat_noise_seed``.
``--floor-lift-cap DELTA`` (LF6, PREREG_LEVELFLOOR_A6 sha256 fbf5de02..., float >= 0, needs ``--level-floor on``): the bounded-lift floor F_B, v_f(tau) = min(kappa * v_hat with the lead cap, v0 + 1.0 * tau, v_O(tau) + DELTA), v_O = O's planned speed after S (chord / 0.2 s);
s' - s <= DELTA * t at every knot. Absent = the LF5 tool bit for bit. The ticks gain ``lf_dterm``, ``lf_dterm_knots``, ``lf_s``, ``lf_sp`` and ``floor_lift_cap``, the header records ``floor_lift_cap``. LF6 also omits the ticks key ``floor_vhat_noise_seed`` when the seed is None.
''')

# 2. flag checks -------------------------------------------------------------------------------------------------------
rep('''(a noise level is never drawn from an unnamed generator)")
''', '''(a noise level is never drawn from an unnamed generator)")
    lc_ = getattr(a, "floor_lift_cap", None)                                                              # LF6
    if lc_ is not None:
        if a.level_floor != "on":
            raise SystemExit("REFUSED: --floor-lift-cap belongs to --level-floor on; with the floor off it would silently do nothing")
        if not (math.isfinite(float(lc_)) and float(lc_) >= 0.0):
            raise SystemExit(f"REFUSED: --floor-lift-cap must be a finite number >= 0, got {lc_!r}")
''')

# 3. the operator F_B (module level, before LevelFloor) ----------------------------------------------------------------
rep('''class LevelFloor:
    """The floor of one run:''', '''# ------------------------------------------------------------------------------------------------ LF6 (PREREG_LEVELFLOOR_A6, sha256 fbf5de02...): the bounded-lift floor F_B
def bounded_progress(op, v_f, v0, S, delta):
    """s_f(t_k), k = 1..30, of F_B with the cap v_O(tau) + delta. ``v_f`` [B] the scalar first term (already 0 where no floor), ``v0`` [B], ``S`` [B, 31] the plan's cumulative arc length (S[:, 0] = 0). Returns ``(s_f [B, 30], dterm_k [B, 30] bool)``.
    v_O,k = (S[k] - S[k-1]) / DT on (t_{k-1}, t_k]. On that interval the integrand is min(c_k, v0' + tau), c_k = min(v_f, v_O,k + delta): the closed form G(t; c) = v0' * tr + 0.5 * a_f * tr^2 + c * (t - tr), tr = min(t, max(c - v0', 0) / a_f), differenced over the interval.
    ``dterm_k``: the delta term is the strict minimum of the three at the interval's end t_k (v_O,k + delta < v_f and < v0' + t_k); where no interval has it, the integrand is min(v_f, v0' + tau) throughout and F_B IS F. The result is clipped at S[k] + delta * t_k (the card's bound)."""
    v_f = np.atleast_1d(np.asarray(v_f, np.float64))
    v0p = np.maximum(np.nan_to_num(np.atleast_1d(np.asarray(v0, np.float64)), nan=0.0, posinf=0.0, neginf=0.0), 0.0)[:, None]
    t_hi = op.T_KNOTS[None, :]
    t_lo = (op.DT * np.arange(0, op.K, dtype=np.float64))[None, :]
    v_o = np.diff(np.asarray(S, np.float64), axis=1) / op.DT                                     # [B, K]: O's planned speed on (t_{k-1}, t_k]
    cap_o = v_o + float(delta)
    vf_c = v_f[:, None]
    c = np.minimum(vf_c, cap_o)

    def G(t):
        tr = np.minimum(t, np.maximum(c - v0p, 0.0) / op.A_F)
        return v0p * tr + 0.5 * op.A_F * tr * tr + c * (t - tr)
    s_f = np.cumsum(G(t_hi) - G(t_lo), axis=1)
    dterm_k = (cap_o < vf_c) & (cap_o < v0p + t_hi)
    return np.minimum(s_f, np.asarray(S, np.float64)[:, 1:] + float(delta) * t_hi), dterm_k


def apply_floor_batch_bounded(op, plans, v0, v_hat, pi, gap_hat, kappa, delta):
    """F_B on a batch: ``level_floor.apply_floor_batch`` with s_f from :func:`bounded_progress`. Same return and ``info`` keys as the operator, plus ``dterm`` (bound row with the delta term the strict minimum at a knot where F lifts), ``dterm_knots`` (how many), ``s`` (s_k) and ``sp`` (s'_k), [B, 30]."""
    delta = float(delta)
    if not (np.isfinite(delta) and delta >= 0.0):
        raise ValueError(f"delta must be a finite number >= 0, got {delta}")
    p64 = np.asarray(plans, np.float64)
    full, S = op.plan_arc(p64)
    B = p64.shape[0]
    v_hat, pi, gap_hat, v0 = (np.broadcast_to(np.atleast_1d(np.asarray(x, np.float64)), (B,)) for x in (v_hat, pi, gap_hat, v0))
    v_f, band, cap = op.floor_speed(v_hat, pi, gap_hat, kappa)
    v_fb = np.where(band, v_f, 0.0)
    s_ramp = op.ramp_progress(v_fb, v0)                                                          # A5's F (closed form)
    with np.errstate(invalid="ignore"):
        s_b, dterm_k = bounded_progress(op, v_fb, v0, S, delta)
    use_b = dterm_k.any(axis=1)
    s_k = S[:, 1:]
    with np.errstate(invalid="ignore"):
        s_f = np.minimum(np.where(use_b[:, None], s_b, s_ramp), s_k + delta * op.T_KNOTS[None, :])
        s_new = np.maximum(s_k, s_f)
        knot_bind = band[:, None] & np.isfinite(p64).all(axis=(1, 2))[:, None] & (s_new != s_k)
    row_bind = knot_bind.any(axis=1)
    kd = dterm_k & knot_bind
    info = {"bind": row_bind, "cap": cap, "vf": v_f, "vhat": v_hat.copy(), "pi": pi.copy(), "gap": op.GAP_SCALE_M * gap_hat, "lift": np.zeros(B), "band": band,
            "dir_fallback": np.zeros(B, bool), "short_last": np.zeros(B, bool), "bind_k10": knot_bind[:, :op.K_EXEC].any(axis=1),
            "dterm": row_bind & kd.any(axis=1), "dterm_knots": kd.sum(axis=1).astype(np.int64), "s": s_k.copy(), "sp": s_new.copy()}
    if not row_bind.any():
        return plans, info
    out = np.array(plans, copy=True)
    for b_ in np.nonzero(row_bind)[0]:
        pts, fb = op.resample_polyline(full[b_], S[b_], s_new[b_])
        new = p64[b_].copy()
        new[knot_bind[b_]] = pts[knot_bind[b_]]
        out[b_] = new.astype(out.dtype)
        info["dir_fallback"][b_] = fb
        info["short_last"][b_] = bool(np.linalg.norm(full[b_, -1] - full[b_, -2]) < op.SHORT_LAST_M)
        _, S_out = op.plan_arc(np.asarray(out[b_], np.float64)[None])
        info["lift"][b_] = (S_out[0, op.K_LIFT] - S[b_, op.K_LIFT]) / op.T_LIFT_S
    return out, info


LF6_OPERATOR_TEXT = (" || LF6 F_B (PREREG_LEVELFLOOR_A6): v_f(tau) = min(kappa * v_hat with the lead cap, v0 + 1.0 * tau, v_O(tau) + DELTA), v_O = the plan's chord speed on (t_{k-1}, t_k] after S (knot 0 = origin); "
                     "s_f(t) = integral_0^t v_f; s' = max(s, s_f) with s_f <= s + DELTA * t; the rest as F")


class LevelFloor:
    """The floor of one run:''')

# 4. LevelFloor.__init__ ----------------------------------------------------------------------------------------------
rep('''oracle=None, noise=None, noise_seed=None):''', '''oracle=None, noise=None, noise_seed=None, lift_cap=None):''')
rep('''            self.stats["noise_draws"] = 0
''', '''            self.stats["noise_draws"] = 0
        self.lift_cap = None if lift_cap is None else float(lift_cap)                               # LF6: DELTA of F_B (None = A5's F)
        if self.lift_cap is not None:
            if not (math.isfinite(self.lift_cap) and self.lift_cap >= 0.0):
                raise ValueError(f"LF6: the lift cap DELTA must be a finite number >= 0, got {lift_cap!r}")
            self.stats["dterm_bound"] = 0                                                           # bound ticks where the DELTA term is the strict minimum at a knot where F lifts
''')

# 5. LevelFloor.__call__ ----------------------------------------------------------------------------------------------
rep('''        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
''', '''        if self.lift_cap is None:
            out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        else:                                                                                        # LF6: F_B
            out, info = apply_floor_batch_bounded(self.op, plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa, self.lift_cap)
            self.stats["dterm_bound"] += int(info["dterm"].sum())
''')

# 6. meta (the ticks-file keys; the defect fix) -----------------------------------------------------------------------
rep('''        return {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256, "floor_vhat_source": self.vhat_source, **({"label": "CEILING", "floor_vhat_noise": self.noise, "floor_vhat_noise_seed": self.noise_seed} if self.vhat_source == "logged" else {})}
''', '''        m = {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256, "floor_vhat_source": self.vhat_source}
        if self.vhat_source == "logged":
            m["label"], m["floor_vhat_noise"] = "CEILING", self.noise
            if self.noise_seed is not None:                                                         # LF6 fix: a None seed is a 0-d object array that allow_pickle=False refuses (A5_NOTE_READER_FIX); absent = no draw
                m["floor_vhat_noise_seed"] = self.noise_seed
        if self.lift_cap is not None:
            m["floor_lift_cap"] = self.lift_cap
        return m
''')

# 7. detail -----------------------------------------------------------------------------------------------------------
rep('''lf_vhat_heads in the ticks file holds what the heads said on the same tick"
        return d
''', '''lf_vhat_heads in the ticks file holds what the heads said on the same tick"
        if self.lift_cap is not None:
            d["floor_lift_cap"] = self.lift_cap
            d["operator"] = self.op.OPERATOR_TEXT + LF6_OPERATOR_TEXT
            d["prereg"] = "dagland/motion_c/a6_prereg/PREREG_LEVELFLOOR_A6.md (sha256 fbf5de02...), H-LEVELFLOOR-1 (A6)"
            d["bounded_lift"] = "s' - s <= DELTA * t at every knot; lf_vf is the scalar first term kappa * v_hat (lead cap), the DELTA term acts inside s_f; lf_dterm = a bound tick where the DELTA term is the strict minimum at a knot where F lifts; lf_s / lf_sp = s_k / s'_k"
            st["dterm_share_of_bound"] = (st["dterm_bound"] / st["bind"]) if st["bind"] else None
        return d
''')

# 8. load_level_floor -------------------------------------------------------------------------------------------------
rep('''noise=getattr(a, "floor_vhat_noise", None), noise_seed=getattr(a, "floor_vhat_noise_seed", None))''', '''noise=getattr(a, "floor_vhat_noise", None), noise_seed=getattr(a, "floor_vhat_noise_seed", None), lift_cap=getattr(a, "floor_lift_cap", None))''')

# 9. TickRecorder -----------------------------------------------------------------------------------------------------
rep('''    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''', '''    LF6_FIELDS = ("lf_dterm", "lf_dterm_knots", "lf_s", "lf_sp")                                                                     # LF6 (--floor-lift-cap): the DELTA term binds, s_k and s'_k of the tick
    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''')
rep('''floor_noise: bool = False):''', '''floor_noise: bool = False, floor_bound: bool = False):''')
rep('''        self.floor_noise = bool(floor_noise) and self.floor_oracle
''', '''        self.floor_noise = bool(floor_noise) and self.floor_oracle
        self.floor_bound = bool(floor_bound) and self.level_floor
''')
rep('''(self.LF5_FIELDS if self.floor_noise else ())}''', '''(self.LF5_FIELDS if self.floor_noise else ()) + (self.LF6_FIELDS if self.floor_bound else ())}''')
rep('''                    r["lf_vlog"].append(np.float64(fi["vlog"][b])); r["lf_eps"].append(np.float64(fi["eps"][b]))
''', '''                    r["lf_vlog"].append(np.float64(fi["vlog"][b])); r["lf_eps"].append(np.float64(fi["eps"][b]))
                if self.floor_bound:
                    r["lf_dterm"].append(bool(fi["dterm"][b])); r["lf_dterm_knots"].append(int(fi["dterm_knots"][b]))
                    r["lf_s"].append(np.asarray(fi["s"][b], np.float64)); r["lf_sp"].append(np.asarray(fi["sp"][b], np.float64))
''')
rep('''floor_noise=floor is not None and floor.noise_path)''', '''floor_noise=floor is not None and floor.noise_path, floor_bound=floor is not None and floor.lift_cap is not None)''')

# 10. argparse + header -----------------------------------------------------------------------------------------------
rep('''    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''', '''    p.add_argument("--floor-lift-cap", type=float, default=None, help="LF6 (PREREG_LEVELFLOOR_A6): DELTA >= 0 in m/s, the bounded-lift floor F_B, v_f(tau) = min(kappa * v_hat, v0 + 1.0 * tau, v_O(tau) + DELTA); needs --level-floor on; default (absent) = A5's F, the LF5 tool bit for bit")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''')
rep('''        header["floor_vhat_source"] = floor.vhat_source
''', '''        header["floor_vhat_source"] = floor.vhat_source
        if floor.lift_cap is not None:
            header["floor_lift_cap"] = floor.lift_cap
''')

open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
