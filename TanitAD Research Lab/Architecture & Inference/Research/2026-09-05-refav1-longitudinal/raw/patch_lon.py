# -*- coding: utf-8 -*-
"""Apply the LONGITUDINAL ANALOGUE to refa_v1.py. Every anchor is asserted to
occur EXACTLY ONCE (or exactly twice where stated); a miss aborts without
writing. Idempotent: a file already carrying the marker is left alone."""
import io
import sys
import time

PATH = sys.argv[1]
MARK = "a_sustain"


def rd(p, tries=25):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE: could not read %s" % p)


def wr(p, t, tries=25):
    for _ in range(tries):
        try:
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            if len(rd(p)) == len(t):
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE: could not write %s" % p)


t = rd(PATH)
if MARK in t:
    print("ALREADY PATCHED (marker %r present %d times)" % (MARK, t.count(MARK)))
    raise SystemExit(0)
n0 = len(t)
# The repo tree and the off-Drive clone differ in line endings (banked memory:
# "CRLF/LF split by tree"). Normalise for anchoring, restore on write, so the
# same patcher is correct in both trees and neither is silently re-ended.
EOL = "\r\n" if "\r\n" in t else "\n"
print("  line ending detected: %r" % EOL)
t = t.replace("\r\n", "\n")


def sub(old, new, label, want=1):
    global t
    c = t.count(old)
    if c != want:
        raise SystemExit(
            "ANCHOR %s occurs %d times (need %d) -- ABORT, nothing written"
            % (label, c, want))
    t = t.replace(old, new, want)
    print("  patched %s" % label)


STAR = u"⭐"
NO = u"⛔"
WARN = u"⚠"

# ---------------- 1. canonical_controls: the a_sustain knob ---------------- #
sub("""def canonical_controls(lat: str, lon: str, v0: float, op_steps: int,
                       op_dt: float, *,
                       kappa_turn: float | None = None) -> Tensor:""",
    """def canonical_controls(lat: str, lon: str, v0: float, op_steps: int,
                       op_dt: float, *,
                       kappa_turn: float | None = None,
                       a_sustain: float | None = None) -> Tensor:""",
    "canonical_controls signature")

MAINTAIN = u"""    if a_sustain is not None and abs(v_t - float(v0)) < _MAINTAIN_EPS:
        # {S}{S} `a_sustain` -- THE LONGITUDINAL ANALOGUE OF `kappa_turn`
        # (D-REFAV1-LON-VOCAB, 2026-09-05). It touches ONLY the MAINTAIN
        # branch (`v_t == v0`), which is `CRUISE` always and
        # `ADAPT_SPEED_FOR_CURVE` while `v0 <= GOAL_CURVE_VMAX_MPS`. On that
        # branch the shipped profile is `a == 0` EXACTLY -- the vocabulary can
        # say "hold this SPEED" and has no way to say "hold this
        # ACCELERATION".
        # MEASURED (dump_wk15, n = 40 windows, ckpt 21,109; banked at
        # `.../2026-09-05-refav1-longitudinal/raw/lon_branch.txt`):
        #   * the maintain branch is **31 / 40 windows (77.5 %)** and
        #     **20 / 24 (83.3 %)** of the GT-LON stratum, so the goal is
        #     silent exactly where longitudinal action is demanded;
        #   * the vocabulary's reachable `dv` over the 2.0 s plan window is
        #     **[-2.85, +0.98] m/s** while the corpus demands p10 -2.16 /
        #     p90 **+2.34** -- the positive side is short by **2.40x**, and
        #     **14 of the 17 accelerating windows (82.4 %) are outside the
        #     reachable set entirely**. That is the `kappa in {{0, 0.08}}`
        #     defect with the sign flipped: laterally the only magnitude was
        #     6.9x too BIG, longitudinally the only positive one is 2.4x too
        #     SMALL.
        # {S} THE HINT IS MEASURABLE, WHICH IS WHY THIS IS NOT `kappa_turn`'s
        # oracle problem: `choose_kappa_level` needs a curvature the head
        # cannot supply, so M15's payoff was an ORACLE-CHOOSER bound. Here the
        # natural hint is `a0 = (v[t0] - v[t0-dt]) / dt`, a BACKWARD DIFFERENCE
        # OF PAST MEASURED SPEEDS closing at t0 (`echo_gate.ha0_ext`'s own
        # definition, "no future") -- admissible at T1 under the PI ruling of
        # 2026-09-02 that the measured state at cycle time is a legal initial
        # state. So the ORACLE column and the REALISED column COINCIDE, and
        # there is no free parameter to tune on the scored split.
        # {W} The value is clipped to `GOAL_A_MAX` like every other row.
        # {N} `None` is the SHIPPED path and skips this branch entirely, so an
        # arm that does not ask for it is BIT-IDENTICAL to the pre-2026-09-05
        # expression -- asserted by `tests/test_refa_v1_a_sustain.py`, each
        # case with a same-breath control that must differ.
        a[:] = max(-GOAL_A_MAX, min(GOAL_A_MAX, float(a_sustain)))
    else:
        v = float(v0)
        for i in range(op_steps):
            a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))
            a[i] = a_i
            v += a_i * op_dt""".format(S=STAR, N=NO, W=WARN)

sub("""    v = float(v0)
    for i in range(op_steps):
        a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))
        a[i] = a_i
        v += a_i * op_dt""", MAINTAIN, "canonical_controls maintain branch")

# ---------------- 2. the predicate constant + its provenance --------------- #
EPS = u"""#: {N} THE MAINTAIN-BRANCH PREDICATE for `a_sustain`. A token is on the
#: maintain branch when its target speed IS the measured v0 -- `CRUISE`
#: always, `ADAPT_SPEED_FOR_CURVE` while `v0 <= GOAL_CURVE_VMAX_MPS`. Counted
#: on THAT predicate, never on the decode: M28 (2) measured that a `LANE_KEEP`
#: decode is not a hold goal, and the same trap applies here -- 29/40 windows
#: decode `ADAPT_SPEED_FOR_CURVE` but only 27 of them sit below the 8 m/s cap
#: and therefore actually command `a == 0`; the other 2 command a SUSTAINED
#: BRAKE to 8 m/s, which is a different mechanism with the same token name.
_MAINTAIN_EPS = 1e-9

W_JERK = 0.02                                   #: comfort, on channel 0 only""".format(N=NO)
sub("""W_JERK = 0.02                                   #: comfort, on channel 0 only""",
    EPS, "_MAINTAIN_EPS")

# ---------------- 3. _imagine_tactical_goal: signature + use --------------- #
sub("""                               goal_kappa_levels=None,
                               goal_kappa_hint=None
                               ) -> tuple[Tensor, dict]:""",
    """                               goal_kappa_levels=None,
                               goal_kappa_hint=None,
                               a_sustain=None
                               ) -> tuple[Tensor, dict]:""",
    "_imagine_tactical_goal signature")

CTRL = u"""        # {S} `a_sustain` may be a SCALAR (one value for the batch) or one value
        # per row -- the per-row form is what a measured-a0 hint is.
        if a_sustain is None:
            asu = [None] * len(lat_i)
        else:
            asu = torch.as_tensor(a_sustain, dtype=torch.float32
                                  ).reshape(-1).tolist()
            if len(asu) == 1:
                asu = asu * len(lat_i)
            if len(asu) != len(lat_i):
                raise ValueError(f"a_sustain carries {{len(asu)}} rows for a "
                                 f"batch of {{len(lat_i)}}")
        ctrl = torch.stack([
            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,
                               kappa_turn=k, a_sustain=s)
            for i, j, v, k, s in zip(lat_i, lon_i, v0.tolist(), kt,
                                     asu)]).to(last)""".format(S=STAR)
sub("""        ctrl = torch.stack([
            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,
                               kappa_turn=k)
            for i, j, v, k in zip(lat_i, lon_i, v0.tolist(), kt)]).to(last)""",
    CTRL, "_imagine_tactical_goal ctrl stack")

sub("""                      "kappa_vocab": goal_kappa_vocab_id(goal_kappa_levels)}""",
    """                      "kappa_vocab": goal_kappa_vocab_id(goal_kappa_levels),
                      "a_sustain_used": [None if s is None else float(s)
                                         for s in asu]}""",
    "_imagine_tactical_goal goal_info stamp")

# ---------------- 4. plan(): signature ------------------------------------- #
sub("""             goal_kappa_hint=None,
             goal_keeps_seed: bool = False):""",
    """             goal_kappa_hint=None,
             a_sustain=None,
             jerk_seam_a0: float | None = None,
             goal_keeps_seed: bool = False):""",
    "plan signature")

# ---------------- 5. plan(): the two goal call sites ----------------------- #
sub("""                goal_kappa_turn=goal_kappa_turn,
                goal_kappa_levels=goal_kappa_levels,
                goal_kappa_hint=goal_kappa_hint)""",
    """                goal_kappa_turn=goal_kappa_turn,
                goal_kappa_levels=goal_kappa_levels,
                goal_kappa_hint=goal_kappa_hint,
                a_sustain=a_sustain)""",
    "plan -> _imagine_tactical_goal (2 sites)", want=2)

# ---------------- 6. the JERK SEAM ----------------------------------------- #
SEAM = u"""            if jerk_seam_a0 is None:
                jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt
            else:
                # {N}{N} THE MISSING TERM (D-REFAV1-LON-COST, 2026-09-05). The
                # shipped jerk is the diff of the plan's OWN actions, indices
                # 1..H-1 against 0..H-2 -- so the step from the car's MEASURED
                # acceleration at t0 to `controls[0]` is NOT PRICED. A plan
                # that drops instantly from a0 = -2.3 m/s^2 to a = 0 therefore
                # costs exactly the same jerk as one that continues smoothly,
                # and the all-zero control is the JOINT minimiser of BOTH
                # regularisers (`w_jerk*mean(jerk^2)` is 0 for ANY constant a,
                # including 0; `w_kappa*mean(kappa^2)` is 0 only at kappa = 0).
                # {N} And the third weight cannot oppose it: `target_speed`
                # defaults to `None` and `refav1_arm.py`'s single `.plan(...)`
                # call never passes it, so `w_vend` has contributed ZERO cost
                # to every banked refav1 window -- the longitudinal channel has
                # no cost term at all unless one is armed.
                # {S} Prepending the measured a0 as the (-1)-th action prices
                # the seam with the SAME `w_jerk` -- a repair of an incomplete
                # term, not a new weight, exactly as `ccosh` repaired an
                # undefined cost rather than re-weighting one.
                # {W} `a0` is a backward difference of PAST measured speeds
                # (`echo_gate.ha0_ext`: "no future"), admissible at T1 under
                # the 2026-09-02 PI ruling on the measured state at cycle time.
                # {N} `None` is the SHIPPED path: bit-identical to every arm
                # banked before 2026-09-05.
                a_prev = controls.new_full((controls.shape[0], 1),
                                           float(jerk_seam_a0))
                a_seq = torch.cat([a_prev, controls[:, :, 0]], dim=1)
                jerk = (a_seq[:, 1:] - a_seq[:, :-1]) / pc.dt
            c = c + w_jerk * jerk.pow(2).mean(-1)                  # comfort""".format(
    S=STAR, N=NO, W=WARN)
sub("""            jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt
            c = c + w_jerk * jerk.pow(2).mean(-1)                  # comfort""",
    SEAM, "jerk seam")

# ---------------- 7. result stamps ----------------------------------------- #
STAMP = u"""        res.goal_keeps_seed = bool(goal_keeps_seed)
        # {S} THE LONGITUDINAL LEVERS TRAVEL ON THE RESULT, like every other
        # convention: a banked window must say whether its goal could express a
        # sustained acceleration, whether the jerk seam was priced, and whether
        # `W_VEND` was armed AT ALL -- `target_speed=None` (the shipped path)
        # makes the third weight of the cost triple a DEAD TERM, and a dump
        # that does not record it cannot be told apart from one where it bound.
        res.a_sustain = (None if a_sustain is None else
                         [float(x) for x in
                          torch.as_tensor(a_sustain).flatten()])
        res.jerk_seam_a0 = (None if jerk_seam_a0 is None
                            else float(jerk_seam_a0))
        res.target_speed = (None if target_speed is None
                            else float(target_speed))
        res.w_vend_armed = target_speed is not None""".format(S=STAR)
sub("""        res.goal_keeps_seed = bool(goal_keeps_seed)""", STAMP, "result stamps")

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK  %s  %d -> %d chars  (marker %r x%d)"
      % (PATH, n0, len(t), MARK, t.count(MARK)))
