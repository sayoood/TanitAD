"""Apply the LF5 patch (``--floor-vhat-noise SIGMA`` and ``--floor-vhat-noise-seed N``, PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...) to the LFO tool (sha256 6744d3e4..., lfo_build/tool). One-shot; each replacement must match exactly once.
Usage: python -I apply_lf5_patch.py <LFO v8f_t1_eval_ri.py 6744d3e4> <out v8f_t1_eval_ri.py>     (the patched tool keeps the name v8f_t1_eval_ri.py inside the read tree)
Noise: only with ``--floor-vhat-source logged``. v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1), ONE numpy.random.default_rng(N) per rollout, ONE vector draw of the batch size at every floor call (= every replan of a chunk), in replan order,
taken whether or not a floor applies (NaN v_log stays NaN). The noise path is active when the seed is given (SIGMA 0 then multiplies by exp(0) = 1.0 exactly); with neither flag nothing changes: no generator, no draw, the LFO tool bit for bit."""
import hashlib
import sys

SRC_SHA = "6744d3e4686e344fb72e02ad8f659c779eb73261de13c4a3844e15243667dcc6"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the LFO tool 6744d3e4"
assert b"\r" not in b, "the source tool must be LF"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. docstring ---------------------------------------------------------------------------------------------------------
rep('''the floor off. The result JSON header records ``floor_vhat_source``''', '''the floor off.
``--floor-vhat-noise SIGMA`` / ``--floor-vhat-noise-seed N`` (LF5, PREREG_LEVELFLOOR_A5 sha256 2aac1dbe..., only with ``--floor-vhat-source logged``; refused otherwise): v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1) from ONE
``numpy.random.default_rng(N)`` per rollout, one vector draw per floor call (per replan of a chunk) in replan order, taken whether or not a floor applies; SIGMA > 0 needs the seed. Neither flag = the LFO tool bit for bit. With the
noise path active the ticks file gains ``lf_vlog`` (v_log) and ``lf_eps`` (the draw), and the header records ``floor_vhat_noise`` and ``floor_vhat_noise_seed``.
The result JSON header records ``floor_vhat_source``''')

# 2. flag checks -------------------------------------------------------------------------------------------------------
rep('''    elif src_ is None:
        a.floor_vhat_source = "heads"
''', '''    elif src_ is None:
        a.floor_vhat_source = "heads"
    nz_, nzs_ = getattr(a, "floor_vhat_noise", None), getattr(a, "floor_vhat_noise_seed", None)          # LF5
    if nz_ is not None or nzs_ is not None:
        if a.level_floor != "on" or getattr(a, "floor_vhat_source", None) != "logged":
            raise SystemExit("REFUSED: --floor-vhat-noise / --floor-vhat-noise-seed are defined only with --level-floor on --floor-vhat-source logged (the noise perturbs the logged speed; on the heads it would silently do nothing)")
        if nz_ is not None and not (math.isfinite(float(nz_)) and float(nz_) >= 0.0):
            raise SystemExit(f"REFUSED: --floor-vhat-noise must be a finite number >= 0, got {nz_!r}")
        if nzs_ is not None and int(nzs_) < 0:
            raise SystemExit(f"REFUSED: --floor-vhat-noise-seed must be an integer >= 0, got {nzs_!r}")
        if nz_ is not None and float(nz_) > 0.0 and nzs_ is None:
            raise SystemExit("REFUSED: --floor-vhat-noise SIGMA > 0 needs --floor-vhat-noise-seed N (a noise level is never drawn from an unnamed generator)")
''')

# 3. LevelFloor.__init__ ----------------------------------------------------------------------------------------------
rep('''    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str, vhat_source: str = "heads", oracle=None):''',
    '''    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str, vhat_source: str = "heads", oracle=None, noise=None, noise_seed=None):''')
rep('''        if self.vhat_source == "logged":
            self.stats["oracle_no_route"] = 0                                                       # ticks with no complete logged route = no floor
''', '''        if self.vhat_source == "logged":
            self.stats["oracle_no_route"] = 0                                                       # ticks with no complete logged route = no floor
        self.noise = 0.0 if noise is None else float(noise)                                         # LF5: SIGMA (0.0 = none)
        self.noise_seed = None if noise_seed is None else int(noise_seed)
        if (noise is not None or noise_seed is not None) and self.vhat_source != "logged":
            raise ValueError("LF5: the v_hat noise belongs to --floor-vhat-source logged")
        if self.noise > 0.0 and self.noise_seed is None:
            raise ValueError("LF5: a noise level SIGMA > 0 needs a seed")
        self.noise_path = self.noise_seed is not None                                               # the noise path runs (draws are taken) iff a seed is given; SIGMA 0 then multiplies by exp(0) = 1.0
        self.rng = np.random.default_rng(self.noise_seed) if self.noise_path else None
        if self.noise_path:
            self.stats["noise_draws"] = 0
''')

# 4. LevelFloor.__call__ ----------------------------------------------------------------------------------------------
rep('''            v_heads = v_hat
            v_hat = self.oracle_vhat(len(v_heads))
        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        if self.vhat_source == "logged":
            info["vhat_heads"] = np.asarray(v_heads, np.float64)
''', '''            v_heads = v_hat
            v_hat = self.oracle_vhat(len(v_heads))
            if self.noise_path:                                                                      # LF5: one draw per window of this replan, in replan order, whether or not a floor applies
                v_log = v_hat
                eps = self.rng.standard_normal(len(v_log))
                self.stats["noise_draws"] += int(len(eps))
                v_hat = v_log * np.exp(self.noise * eps)
        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        if self.vhat_source == "logged":
            if self.noise_path:
                info["vlog"], info["eps"] = np.asarray(v_log, np.float64), np.asarray(eps, np.float64)
            info["vhat_heads"] = np.asarray(v_heads, np.float64)
''')

# 5. meta / detail ----------------------------------------------------------------------------------------------------
rep('''**({"label": "CEILING"} if self.vhat_source == "logged" else {})}

    def detail(self):''', '''**({"label": "CEILING", "floor_vhat_noise": self.noise, "floor_vhat_noise_seed": self.noise_seed} if self.vhat_source == "logged" else {})}

    def detail(self):''')
rep('''            d["level_input"] = "logged_route_speed''', '''            d["floor_vhat_noise"], d["floor_vhat_noise_seed"] = self.noise, self.noise_seed
            if self.noise_path:
                d["noise"] = "v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1), one numpy.random.default_rng(seed) per rollout, one vector draw per floor call, in replan order, whether or not a floor applies (PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...)"
            d["level_input"] = "logged_route_speed''')

# 6. load_level_floor -------------------------------------------------------------------------------------------------
rep('''vhat_source=getattr(a, "floor_vhat_source", None) or "heads")''', '''vhat_source=getattr(a, "floor_vhat_source", None) or "heads",
                      noise=getattr(a, "floor_vhat_noise", None), noise_seed=getattr(a, "floor_vhat_noise_seed", None))''')

# 7. TickRecorder -----------------------------------------------------------------------------------------------------
rep('''    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''', '''    LF5_FIELDS = ("lf_vlog", "lf_eps")                                                                                               # LF5 (the noise path is active): v_log and the draw of the tick
    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''')
rep('''level_floor: bool = False, floor_oracle: bool = False):''', '''level_floor: bool = False, floor_oracle: bool = False, floor_noise: bool = False):''')
rep('''        self.floor_oracle = bool(floor_oracle) and self.level_floor
''', '''        self.floor_oracle = bool(floor_oracle) and self.level_floor
        self.floor_noise = bool(floor_noise) and self.floor_oracle
''')
rep('''(self.LFO_FIELDS if (level_floor and floor_oracle) else ())}''', '''(self.LFO_FIELDS if (level_floor and floor_oracle) else ()) + (self.LF5_FIELDS if self.floor_noise else ())}''')
rep('''                    r["lf_vhat_heads"].append(np.float32(fi["vhat_heads"][b]))
''', '''                    r["lf_vhat_heads"].append(np.float32(fi["vhat_heads"][b]))
                if self.floor_noise:
                    r["lf_vlog"].append(np.float64(fi["vlog"][b])); r["lf_eps"].append(np.float64(fi["eps"][b]))
''')
rep('''floor_oracle=floor is not None and floor.vhat_source == "logged")''', '''floor_oracle=floor is not None and floor.vhat_source == "logged", floor_noise=floor is not None and floor.noise_path)''')

# 8. argparse + header ------------------------------------------------------------------------------------------------
rep('''    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''', '''    p.add_argument("--floor-vhat-noise", type=float, default=None, help="LF5 (PREREG_LEVELFLOOR_A5): SIGMA, v_hat := v_log * exp(SIGMA * eps), eps ~ N(0,1); only with --floor-vhat-source logged; default (absent) = no noise; SIGMA > 0 needs --floor-vhat-noise-seed")
    p.add_argument("--floor-vhat-noise-seed", type=int, default=None, help="LF5: the seed of the ONE numpy.random.default_rng of the rollout that draws eps (one vector draw per replan, in replan order); only with --floor-vhat-source logged")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''')
rep('''        header["floor_vhat_source"] = floor.vhat_source
        if floor.vhat_source == "logged":
            header["label"] = "CEILING"''', '''        header["floor_vhat_source"] = floor.vhat_source
        if floor.vhat_source == "logged":
            header["floor_vhat_noise"], header["floor_vhat_noise_seed"] = floor.noise, floor.noise_seed
            header["label"] = "CEILING"''')

open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("OK", hashlib.sha256(s.encode("utf-8")).hexdigest())
