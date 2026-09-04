"""Second patch: the vocabulary's channel 1 may be LATERAL ACCELERATION.

MEASURED 2026-09-04 on the 4,823-window surface. A constant-CURVATURE family is
unflyable at speed -- a_lat = v^2 * kappa, so kappa 0.06 at 27 m/s is 43.7 m/s^2
= 4.5 g, and 104 of 117 anchors break a mu = 0.7 friction circle (peak 3.96 g)
where refcv4's fixed-path set broke NONE. Those anchors are not WRONG (the
oracle never picks one, because the GT it is scored against is flyable) but they
are WASTED BUDGET, and at high speed they waste almost all of it.

Expressing channel 1 as LATERAL ACCELERATION and deriving
kappa = a_lat / max(v0, v_floor)^2 makes the family physical by construction:
the control space becomes (a_lon, a_lat), which IS the Kamm-circle space, so a
bound on the grid IS a bound on the friction circle.

  flat kappa,  13a x 9k = 117 : ADE 0.2610  -0.0387 [-0.0652, -0.0104]  104/117 over mu=0.7 @ 27 m/s, peak 3.96 g
  a_lat 3.0,   13a x 9k = 117 : ADE 0.1987  -0.1009 [-0.1213, -0.0813]    0/117 over mu=0.7 @ 27 m/s, peak 0.68 g

a_lat = 0 <=> kappa = 0 EXACTLY, so the pinned straight-ahead control survives
the reparameterisation unchanged.
"""
import io
import sys

P = sys.argv[1]
raw = io.open(P, "rb").read()
CRLF = raw.count(b"\r\n")
NL = "\r\n" if CRLF else "\n"
src = raw.decode("utf-8").replace("\r\n", "\n")
orig = src
EDITS = []


def sub(old, new, tag):
    global src
    n = src.count(old)
    assert n == 1, "edit %r matched %d times (expected 1)" % (tag, n)
    src = src.replace(old, new)
    EDITS.append(tag)


sub("""    v0_conditioned: bool = False
    ref_speed_ms: float = 10.0""",
    """    v0_conditioned: bool = False
    ref_speed_ms: float = 10.0
    # ⭐ WHAT CHANNEL 1 OF `anchor_controls` MEANS.
    #   "kappa" — curvature 1/m, integrated as supplied (the literal control).
    #   "alat"  — LATERAL ACCELERATION m/s^2; curvature is DERIVED per window as
    #             `a_lat / max(v0, alat_v_floor_ms)^2`, clamped to `kappa_cap`.
    # MEASURED: a constant-CURVATURE family is unflyable at speed (a_lat =
    # v^2 * kappa, so kappa 0.06 at 27 m/s is 4.5 g and 104 of 117 anchors break
    # a mu = 0.7 circle). Under "alat" the control space IS the Kamm-circle
    # space, so a bound on the grid is a bound on the friction circle: the same
    # 117-anchor budget reads 0.1987 m oracle-in-vocabulary (-0.1009 [-0.1213,
    # -0.0813] vs ha) with 0/117 over mu = 0.7 and a 0.68 g peak.
    # `a_lat = 0` <=> `kappa = 0` exactly, so the pinned straight-ahead control
    # survives the reparameterisation.
    control_units: str = "kappa"
    alat_v_floor_ms: float = 4.0     # below this the clamp would explode
    kappa_cap: float = 0.12          # ~8.3 m turn radius; a parking-lot bound""",
    "cfg-alat")

sub("""                 horizons: tuple[int, ...] = (),
                 v0_conditioned: bool = False,
                 ref_speed_ms: float = 10.0):""",
    """                 horizons: tuple[int, ...] = (),
                 v0_conditioned: bool = False,
                 ref_speed_ms: float = 10.0,
                 control_units: str = "kappa",
                 alat_v_floor_ms: float = 4.0,
                 kappa_cap: float = 0.12):""",
    "init-alat-sig")

sub("""        self.anchor_v0_cond = bool(v0_conditioned)
        self.anchor_ref_speed = float(ref_speed_ms)
        self.anchor_dt = 0.1""",
    """        self.anchor_v0_cond = bool(v0_conditioned)
        self.anchor_ref_speed = float(ref_speed_ms)
        self.anchor_dt = 0.1
        if control_units not in ("kappa", "alat"):
            raise ValueError(f"control_units {control_units!r} not in "
                             f"('kappa', 'alat')")
        self.anchor_control_units = control_units
        self.anchor_alat_v_floor = float(alat_v_floor_ms)
        self.anchor_kappa_cap = float(kappa_cap)""",
    "init-alat-state")

sub("""        h = self.anchor_roll_steps
        ctrl = self.anchor_controls.to(torch.float32)
        ctrl = ctrl[None, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)""",
    """        h = self.anchor_roll_steps
        ctrl = self.anchor_controls.to(torch.float32)
        if self.anchor_control_units == "alat":
            # kappa = a_lat / v^2, clamped. The floor keeps a standing-start
            # window from asking for an infinite curvature, and the cap is a
            # geometric bound (~8.3 m radius) that no road manoeuvre needs.
            vv = v.clamp_min(self.anchor_alat_v_floor) ** 2          # [B]
            kap = (ctrl[None, :, 1] / vv[:, None]).clamp(
                -self.anchor_kappa_cap, self.anchor_kappa_cap)       # [B, N]
            ctrl = torch.stack(
                [ctrl[None, :, 0].expand(batch, n), kap], dim=-1)    # [B, N, 2]
            ctrl = ctrl[:, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)
        else:
            ctrl = ctrl[None, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)""",
    "roll-alat")

# `anchors` (the ref-speed roll) must be produced by the SAME rule, so the
# checkpoint-visible artifact is the family a withheld row actually decodes.
io.open(P, "wb").write(src.replace("\n", NL).encode("utf-8"))
print("applied %d edits: %s" % (len(EDITS), ", ".join(EDITS)))
print("delta bytes: %+d" % (len(src) - len(orig)))
