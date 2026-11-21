"""Apply the refcv4-b v0-conditioned-vocabulary edits to refc.py.

Every replacement asserts it matched EXACTLY ONCE. Exit code is not the
evidence -- the script prints a per-edit content assertion and re-greps the
result at the end.
"""
import io
import sys

P = sys.argv[1]
raw = io.open(P, "rb").read()
CRLF = raw.count(b"\r\n")
BARE = raw.count(b"\n") - CRLF
assert BARE == 0 or CRLF == 0, "mixed line endings (%d CRLF / %d LF)" % (CRLF, BARE)
NL = "\r\n" if CRLF else "\n"
src = raw.decode("utf-8").replace("\r\n", "\n")
orig = src
EDITS = []
print("line endings: %s (%d lines)" % ("CRLF" if CRLF else "LF", CRLF or BARE))


def sub(old, new, tag):
    global src
    n = src.count(old)
    assert n == 1, "edit %r matched %d times (expected 1)" % (tag, n)
    src = src.replace(old, new)
    EDITS.append(tag)


# --------------------------------------------------------------------------
# 1. decoder __init__ signature + the rolled-vocabulary state
# --------------------------------------------------------------------------
sub("""                 factored_maneuver: bool = False,
                 sel: "SelectionConfig | None" = None):
        super().__init__()
        self.cfg = cfg
        self.n_steps = n_steps""",
    """                 factored_maneuver: bool = False,
                 sel: "SelectionConfig | None" = None,
                 horizons: tuple[int, ...] = (),
                 v0_conditioned: bool = False,
                 ref_speed_ms: float = 10.0):
        super().__init__()
        self.cfg = cfg
        self.n_steps = n_steps""",
    "init-signature")

sub("""        # Anchor vocabulary — a persistent buffer (travels with the checkpoint).
        self.register_buffer("anchors", anchors)              # [N, S, 2]
""",
    """        # Anchor vocabulary — a persistent buffer (travels with the checkpoint).
        self.register_buffer("anchors", anchors)              # [N, S, 2]
        # ⭐ refcv4-b: the v0-CONDITIONED vocabulary. `anchor_controls` [N, 2]
        # = (accel m/s^2, curvature 1/m) held constant over the horizon; the
        # emitted bank is rolled per window from that window's own v0. `anchors`
        # keeps holding the SAME family rolled at `ref_speed_ms`, so it stays
        # the checkpoint-visible artifact a content check can compare, and it is
        # the surface the two param-free geometric priors fall back to.
        self.anchor_v0_cond = bool(v0_conditioned)
        self.anchor_ref_speed = float(ref_speed_ms)
        self.anchor_dt = 0.1
        self.register_buffer("anchor_controls",
                             torch.zeros(anchors.shape[0], 2), persistent=True)
        # slot index of each horizon inside a dt-tick rollout. `rollout_unicycle`
        # returns the state AFTER each step, so horizon k lands at index k - 1.
        _h = tuple(horizons) or tuple(range(1, n_steps + 1))
        if len(_h) != n_steps:
            raise ValueError(f"horizons {_h} does not match n_steps {n_steps}")
        self.register_buffer("anchor_slots",
                             torch.tensor([k - 1 for k in _h],
                                          dtype=torch.long), persistent=False)
        self.anchor_roll_steps = int(max(_h))
""",
    "init-buffers")

# --------------------------------------------------------------------------
# 2. load_anchors accepts the controls alongside the paths
# --------------------------------------------------------------------------
sub('''    def load_anchors(self, anchors: Tensor) -> None:
        """Install an externally-built anchor vocabulary (build_refc_anchors.py).
        Shape must match [N, n_steps, 2] of the constructed decoder."""
        if tuple(anchors.shape) != tuple(self.anchors.shape):
            raise ValueError(f"anchor shape {tuple(anchors.shape)} != decoder "
                             f"{tuple(self.anchors.shape)}")
        self.anchors.copy_(anchors.to(self.anchors.dtype))
''',
    '''    def load_anchors(self, anchors: Tensor,
                     controls: Tensor | None = None) -> None:
        """Install an externally-built anchor vocabulary (build_refc_anchors.py).
        Shape must match [N, n_steps, 2] of the constructed decoder.

        ``controls`` [N, 2] = the (accel, curvature) the paths were rolled from.
        Required when the decoder is ``v0_conditioned`` — without it the bank
        cannot be re-rolled at the window's speed, and a silent fallback to the
        fixed paths would be exactly the defect this vocabulary exists to fix,
        so it RAISES instead.
        """
        if tuple(anchors.shape) != tuple(self.anchors.shape):
            raise ValueError(f"anchor shape {tuple(anchors.shape)} != decoder "
                             f"{tuple(self.anchors.shape)}")
        self.anchors.copy_(anchors.to(self.anchors.dtype))
        if controls is not None:
            if tuple(controls.shape) != tuple(self.anchor_controls.shape):
                raise ValueError(
                    f"anchor controls {tuple(controls.shape)} != decoder "
                    f"{tuple(self.anchor_controls.shape)}")
            self.anchor_controls.copy_(controls.to(self.anchor_controls.dtype))
        elif self.anchor_v0_cond:
            raise ValueError(
                "v0_conditioned decoder loaded an anchor file with no "
                "`controls` [N, 2]. The bank is rolled per window from those "
                "controls; falling back to the fixed paths would silently "
                "restore the un-conditioned vocabulary this build exists to "
                "replace.")

    def roll_bank(self, v_ms: Tensor | None, ego_keep: Tensor | None,
                  batch: int, dtype: torch.dtype) -> Tensor:
        """[B, N, S, 2] — the anchor bank THIS forward decodes.

        Fixed builds expand the stored paths, which is byte-identical to the
        pre-2026-09-04 ``anchors[None].expand(...)``. v0-conditioned builds roll
        ``anchor_controls`` through the programme's OWN integrator
        (:func:`tanitad.models.kinematic.rollout_unicycle`, the one
        ``refa_v1_plan.unicycle_paths`` calls with ``action_units="kappa"``)
        from each window's measured speed.

        ⛔ ``ego_keep`` binds here for the same reason it binds on the S2 band,
        only harder: ``v_ms`` is the PRE-dropout speed, and rolling the bank
        from it on a withheld row would put the withheld channel into the
        candidate GEOMETRY. Withheld rows are rolled at ``ref_speed_ms``, so the
        dropout regime is genuinely speed-blind.
        """
        n = self.anchors.shape[0]
        if not self.anchor_v0_cond:
            return self.anchors.to(dtype)[None].expand(
                batch, n, self.n_steps, 2)
        if v_ms is None:
            v = self.anchors.new_full((batch,), self.anchor_ref_speed)
        else:
            v = v_ms.reshape(-1).to(torch.float32)
            if ego_keep is not None:
                v = torch.where(ego_keep.reshape(-1),
                                v, torch.full_like(v, self.anchor_ref_speed))
        # ⚠️ rolled in float32 regardless of the AMP dtype: 60 sequential
        # integration steps in fp16 accumulate visible drift, and the bank is
        # the geometry every anchor target is measured against.
        h = self.anchor_roll_steps
        ctrl = self.anchor_controls.to(torch.float32)
        ctrl = ctrl[None, :, None, :].expand(batch, n, h, 2).reshape(-1, h, 2)
        state0 = torch.zeros(batch * n, 4, device=ctrl.device,
                             dtype=torch.float32)
        state0[:, 3] = v[:, None].expand(batch, n).reshape(-1)
        path = rollout_unicycle(state0, ctrl, dt=self.anchor_dt)[..., :2]
        return path[:, self.anchor_slots].reshape(
            batch, n, self.n_steps, 2).to(dtype)
''',
    "load-anchors+roll-bank")

# --------------------------------------------------------------------------
# 3. the two param-free geometric priors read the bank when there is one
# --------------------------------------------------------------------------
sub("""    def _lan_anchor_prior(self, lan_dir: Tensor) -> Tensor:""",
    """    def _lan_anchor_prior(self, lan_dir: Tensor,
                          bank: Tensor | None = None) -> Tensor:""",
    "lan-prior-signature")

sub("""        a = self.anchors.to(lan_dir.dtype)                    # [N, S, 2]
        end = a[:, -1]                                        # [N, 2]
        r = torch.linalg.vector_norm(end, dim=-1).clamp_min(1e-6)
        cos_a, sin_a = end[:, 0] / r, end[:, 1] / r           # [N]
        compat = (cos_a[None] * lan_dir[:, 0:1]
                  + sin_a[None] * lan_dir[:, 1:2])            # [B, N]
        return compat * lan_dir[:, 2:3]""",
    """        if bank is None:                    # fixed vocabulary — legacy path,
            a = self.anchors.to(lan_dir.dtype)  # bit-identical to pre-refcv4-b
            end = a[:, -1]                                    # [N, 2]
            r = torch.linalg.vector_norm(end, dim=-1).clamp_min(1e-6)
            cos_a, sin_a = end[:, 0] / r, end[:, 1] / r       # [N]
            compat = (cos_a[None] * lan_dir[:, 0:1]
                      + sin_a[None] * lan_dir[:, 1:2])        # [B, N]
            return compat * lan_dir[:, 2:3]
        end = bank.to(lan_dir.dtype)[:, :, -1]                # [B, N, 2]
        r = torch.linalg.vector_norm(end, dim=-1).clamp_min(1e-6)
        cos_a, sin_a = end[..., 0] / r, end[..., 1] / r       # [B, N]
        compat = cos_a * lan_dir[:, 0:1] + sin_a * lan_dir[:, 1:2]
        return compat * lan_dir[:, 2:3]""",
    "lan-prior-body")

sub("""    def _goal_along_prior(self, dist_pref: Tensor) -> Tensor:""",
    """    def _goal_along_prior(self, dist_pref: Tensor,
                          bank: Tensor | None = None) -> Tensor:""",
    "goal-along-signature")

sub("""        end_x = self.anchors.to(dist_pref.dtype)[:, -1, 0]           # [N]
        z = (end_x - end_x.mean()) / end_x.std().clamp_min(1e-6)     # [N]
        return z[None] * dist_pref.reshape(-1, 1)                   # [B, N]""",
    """        if bank is None:              # fixed vocabulary — legacy path, exact
            end_x = self.anchors.to(dist_pref.dtype)[:, -1, 0]       # [N]
            z = (end_x - end_x.mean()) / end_x.std().clamp_min(1e-6)  # [N]
            return z[None] * dist_pref.reshape(-1, 1)               # [B, N]
        end_x = bank.to(dist_pref.dtype)[:, :, -1, 0]                # [B, N]
        z = ((end_x - end_x.mean(dim=1, keepdim=True))
             / end_x.std(dim=1, keepdim=True).clamp_min(1e-6))       # [B, N]
        return z * dist_pref.reshape(-1, 1)                          # [B, N]""",
    "goal-along-body")

# --------------------------------------------------------------------------
# 4. the fan itself
# --------------------------------------------------------------------------
sub("""        anchors = self.anchors.to(fmap.dtype)                 # [N, S, 2]
        n = anchors.shape[0]
        x0 = anchors[None].expand(b, n, self.n_steps, 2)""",
    """        anchors = self.anchors.to(fmap.dtype)                 # [N, S, 2]
        n = anchors.shape[0]
        # ⭐ refcv4-b: [B, N, S, 2]. For a fixed vocabulary this is exactly the
        # old `anchors[None].expand(...)`; for a v0-conditioned one it is the
        # per-window roll. `bank` — not `anchors` — is the geometry every
        # consumer below must read, including the anchor target in the trainer,
        # which is why it is also returned.
        bank = self.roll_bank(v_ms, ego_keep, b, fmap.dtype)
        x0 = bank
        prior_bank = bank if self.anchor_v0_cond else None""",
    "fan-x0")

sub("""            pre_keep = sl.anchor_reachability_mask(
                anchors, v_ms.to(anchors.dtype), accel_max=sel.accel_max,
                horizon_s=sel.horizon_s)""",
    """            pre_keep = sl.anchor_reachability_mask(
                bank, v_ms.to(bank.dtype), accel_max=sel.accel_max,
                horizon_s=sel.horizon_s)""",
    "prefilter-mask")

sub("""        x = anchors[None] + offset                            # [B, N, S, 2]""",
    """        x = bank + offset                                     # [B, N, S, 2]""",
    "fan-refine")

sub("""            r_terms.append(self.goal_gate * self._lan_anchor_prior(goal_dir))
        if self.goal_dist_gate is not None and goal_dist_pref is not None:
            r_terms.append(self.goal_dist_gate
                           * self._goal_along_prior(goal_dist_pref))""",
    """            r_terms.append(self.goal_gate
                           * self._lan_anchor_prior(goal_dir, prior_bank))
        if self.goal_dist_gate is not None and goal_dist_pref is not None:
            r_terms.append(self.goal_dist_gate
                           * self._goal_along_prior(goal_dist_pref,
                                                    prior_bank))""",
    "rank-goal-priors")

sub("""            terms.append(self.lan_gate * self._lan_anchor_prior(lan_dir))""",
    """            terms.append(self.lan_gate
                         * self._lan_anchor_prior(lan_dir, prior_bank))""",
    "conf-lan-prior")

# --------------------------------------------------------------------------
# 5. construction — hand the decoder its horizons and the v0 flags
# --------------------------------------------------------------------------
sub("""            factored_maneuver=cfg.factored_maneuver,
            sel=cfg.selection())""",
    """            factored_maneuver=cfg.factored_maneuver,
            sel=cfg.selection(),
            horizons=cfg.trajectory.horizons,
            v0_conditioned=cfg.anchors.v0_conditioned,
            ref_speed_ms=cfg.anchors.ref_speed_ms)""",
    "construction")

# --------------------------------------------------------------------------
# 6. the integrator import — the SAME one `refa_v1_plan.unicycle_paths` calls
#    with action_units="kappa" (which is the identity for curvature controls),
#    so an anchor rolled here and one rolled by the offline builder are the
#    same arithmetic and not two implementations.
# --------------------------------------------------------------------------
sub("""from tanitad.refs import refc_select as sl
from tanitad.refs import refc_tactical as tac""",
    """from tanitad.models.kinematic import rollout_unicycle
from tanitad.refs import refc_select as sl
from tanitad.refs import refc_tactical as tac""",
    "import-integrator")

# --------------------------------------------------------------------------
# 7. publish the bank — the trainer's anchor target MUST be measured against
#    the geometry that was actually decoded, not against `decoder.anchors`.
# --------------------------------------------------------------------------
sub('''               "anchor_traj": x, "offset": offset, "sel_score": score,''',
    '''               "anchor_traj": x, "anchor_bank": bank,
               "offset": offset, "sel_score": score,''',
    "decoder-return-bank")

sub('''               "anchor_traj": dec["anchor_traj"], "offset": dec["offset"],''',
    '''               "anchor_traj": dec["anchor_traj"],
               "anchor_bank": dec["anchor_bank"], "offset": dec["offset"],''',
    "model-return-bank")

io.open(P, "wb").write(src.replace("\n", NL).encode("utf-8"))
print("applied %d edits:" % len(EDITS))
for e in EDITS:
    print("  - " + e)
print("delta bytes: %+d" % (len(src) - len(orig)))
