"""Build integration/refc_wiring.patch by editing a COPY of refc.py and diffing.

⛔ refc.py itself is owned by another agent and is never touched: the edits land
on a copy and the deliverable is the diff.

REBASED 2026-09-16 onto `8c7d215` ("refcv6 core: ImageNet timm trunk with frame
+ ego history at 256x1024, and all of F1-F9"), which took refc.py from 3,576 to
4,035 lines and changed three of the eleven anchors the first cut used.
"""
P = "b/refc.py"
_raw = open(P, "rb").read().decode("utf-8")
CRLF = "\r\n" in _raw
s = _raw.replace("\r\n", "\n")


def sub(old, new, n=1):
    global s
    got = s.count(old)
    assert got == n, f"expected {n} occurrences, found {got}: {old[:110]!r}"
    s = s.replace(old, new, n)


# --------------------------------------------------------------------- 1 ----
sub("from tanitad.refs import refc_wp_index as wpi",
    "from tanitad.refs import refc_wp_index as wpi\n"
    "# refcv6 coupling (1): BEV features at the candidate's own waypoints.\n"
    "# Same import discipline as `wpi` above -- a leaf module whose only\n"
    "# tanitad import is `data.bev_raster`, so it cannot cycle back into refc.\n"
    "from tanitad.models import refc_bev_coupling as bevc")

# ------------------------------------------------------------------- C5 ----
# ⭐ Correction C5 (PI-recorded): WP-B is the SPEC's coupling (2), not (1).
# `SPEC_REFCV6_V2.md` §1 numbers them: (1) BEV sampled at the candidate's own
# waypoints, (2) agent slots addressed by waypoint (WP-B), (3) image tokens by
# content. All three in-file comments are corrected so the code and the
# programme record agree; no behaviour changes.
sub("# refcv5 WP-B (`E-WP-INDEX-1`) — DiffusionDrive coupling (1). Cycle-free by",
    "# refcv5 WP-B (`E-WP-INDEX-1`) — DiffusionDrive coupling (2) [C5]. Cycle-free by")
sub("    # ---- refcv5 WP-B: the waypoint index (DiffusionDrive coupling (1)) ----- #",
    "    # ---- refcv5 WP-B: the waypoint index (DD coupling (2), C5) ------------- #")
sub("        # ⭐⭐ refcv5 WP-B (`E-WP-INDEX-1`) — DiffusionDrive coupling (1).",
    "        # ⭐⭐ refcv5 WP-B (`E-WP-INDEX-1`) — DiffusionDrive coupling (2).\n"
    "        # ⚠️ CORRECTION C5 (2026-09-16): these comments said \"coupling (1)\".\n"
    "        # `SPEC_REFCV6_V2.md` §1 numbers them (1) BEV sampled at the\n"
    "        # candidate's own waypoints, (2) agent slots addressed by waypoint\n"
    "        # (this), (3) image tokens by content. A reader who trusted the old\n"
    "        # wording concluded the BEV seam already existed; it did not.")

# --------------------------------------------------------------------- 2 ----
sub("    def wp_index_params(self) -> int:",
    '''    def attach_bev_coupling(self, cfg, d_bev: int) -> int:
        """Build refcv6 coupling (1): one BEV waypoint sampler per decoder
        layer. Returns the parameter count.

        ⛔⛔ **ATTACH THIS AFTER `attach_wp_index`, AS THE LAST STATEMENT OF
        `RefCModel.__init__`.** The RNG-ordering argument in
        `attach_wp_index`'s docstring applies to this seam too, and the ORDER
        between the two matters: with the BEV sampler drawn LAST, a
        `bev-coupling on` build's WP-B heads are bit-identical to a
        `bev-coupling off` build's at the same seed, so the 2x2 ablation
        (WP-B on/off) x (BEV on/off) differs in the levers and in nothing else.
        Attached the other way round, turning the BEV coupling on would silently
        re-draw every WP-B head.

        `d_bev` is the width of the BEV FEATURE MAP the sampler reads -- the BEV
        encoder's `d_out`, which is downstream of the trunk's stride-16 map and
        so is NOT a backbone constant. ⛔ It is a required argument rather than a
        default for the same reason `Box3DMemory.image_hw` is: the stride-16
        width is 1024 on `resnet101` and 256 on `resnet34` (read from
        `feature_info` via `CNNEncoderConfig.s16_dim`), and a default here would
        pin one backbone silently.

        `n_points` is taken from `self.n_steps`: the sampler weights the
        candidate's OWN waypoints, and there are exactly `n_steps` of them.
        """
        if cfg is None or not bool(getattr(cfg, "enable", False)):
            return 0
        if int(d_bev) < 1:
            raise ValueError(
                f"d_bev must be the BEV encoder's output width, got {d_bev}. "
                f"Take it from the perception branch you built, never a "
                f"literal (PI 2026-09-16).")
        self.bev_coupling_cfg = cfg
        n = 0
        for ly in self.layers:
            ly.bev_wp = bevc.BEVWaypointSampler(
                bevc.BEVCouplingConfig(
                    enable=True, d_model=int(self.cfg.d), d_bev=int(d_bev),
                    n_points=int(self.n_steps),
                    learned_offsets=bool(getattr(cfg, "learned_offsets", False)),
                    offset_max_m=float(getattr(cfg, "offset_max_m", 2.0))))
            n += ly.bev_wp.n_params
        return n

    def bev_coupling_params(self) -> int:
        """Trainable parameters coupling (1) added -- carved out of the
        `decoder` line in :func:`param_breakdown` for the reason
        :meth:`wp_index_params` is: a lever whose cost is buried inside a 40 M
        row is a lever nobody can audit."""
        return sum(ly.bev_wp.n_params for ly in self.layers
                   if getattr(ly, "bev_wp", None) is not None)

    def bev_coupling_provenance(self) -> dict:
        """⚠️ What a trainer stamps into `config.json`: whether this build is
        the RELEASED DiffusionDrive coupling or OUR extension. The released code
        samples AT the waypoints with NO learned offsets
        (`.../2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/blocks.py`
        :80-108); an arm with offsets on that reports itself as
        "DiffusionDrive's coupling" is a mislabelled arm, not a better one."""
        ly = next((l for l in self.layers
                   if getattr(l, "bev_wp", None) is not None), None)
        return {"enabled": ly is not None,
                **({} if ly is None else ly.bev_wp.cfg.as_dict())}

    def wp_index_params(self) -> int:''')

# --------------------------------------------------------------------- 3 ----
sub("        self.wp_index: nn.Module | None = None",
    """        self.wp_index: nn.Module | None = None
        # refcv6 coupling (1). None until `attach_bev_coupling`; while it is
        # None `forward` never enters the branch, so the emitted tensors are
        # bitwise those of a build that never had this seam -- the
        # `_agent_bias` rule, that the OFF path must not reach a different
        # kernel either.
        self.bev_wp: nn.Module | None = None""")

# --------------------------------------------------------------------- 4 ----
sub("""    def forward(self, q: Tensor, kv: Tensor, cond: Tensor,
                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_index: tuple | None = None) -> Tensor:""",
    """    def forward(self, q: Tensor, kv: Tensor, cond: Tensor,
                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_index: tuple | None = None,
                bev: Tensor | None = None,
                waypoints: Tensor | None = None) -> Tensor:""")

sub("""        if self.cross_agent is not None and agent_tokens is not None:
            q = q + self.agent_gate * self._attend_agents(
                q, agent_tokens, agent_pad, agent_index)""",
    """        if self.cross_agent is not None and agent_tokens is not None:
            q = q + self.agent_gate * self._attend_agents(
                q, agent_tokens, agent_pad, agent_index)
        # ⭐ refcv6 coupling (1) -- the BEV read at the candidate's OWN
        # waypoints, recomputed on every pass because the address MOVES as the
        # trajectory is denoised. The sampler owns its zero-init gate, so this
        # line is bitwise inert at step 0 and on any build with `bev_wp is None`.
        if self.bev_wp is not None and bev is not None and waypoints is not None:
            q = self.bev_wp(q, waypoints, bev)""")

# --------------------------------------------------------------------- 5 ----
sub("""    def _decode(self, kv: Tensor, cond: Tensor, x_est: Tensor,
                t_idx: int, agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:""",
    """    def _decode(self, kv: Tensor, cond: Tensor, x_est: Tensor,
                t_idx: int, agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None,
                bev: Tensor | None = None) -> tuple[Tensor, Tensor]:""")

sub("""        index = self._agent_index(x_est, agent_pos)
        for layer in self.layers:
            q = layer(q, kv, cond, agents, agent_pad, index)""",
    """        index = self._agent_index(x_est, agent_pos)
        for layer in self.layers:
            # `x_est` IS the metric fan this pass is refining, so it is the
            # address for BOTH couplings -- the agent slots (WP-B / coupling
            # (2), through `index`) and the BEV map (coupling (1), through
            # `waypoints`).
            q = layer(q, kv, cond, agents, agent_pad, index, bev, x_est)""")

# --------------------------------------------------------------------- 6 ----
sub("""    def _decode_ctrl(self, kv: Tensor, cond: Tensor, x_path: Tensor,
                     t: Tensor, agents: Tensor | None,
                     agent_pad: Tensor | None,
                     agent_pos: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:""",
    """    def _decode_ctrl(self, kv: Tensor, cond: Tensor, x_path: Tensor,
                     t: Tensor, agents: Tensor | None,
                     agent_pad: Tensor | None,
                     agent_pos: Tensor | None = None,
                     bev: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:""")

# ⚠️ BOTH sampler loops. F3/F4 (landed in 8c7d215) split `_decode_ctrl` into a
# fast path -- taken when `cascade` and `adaln` are both off, and deliberately
# "the ONE LINE it always was" -- and a per-layer cascade path. Patching only
# the first would have left coupling (1) silently DEAD in every F3/F4 arm,
# which is exactly the arm refcv6 is for.
sub("""        if self.cascade is None and self.adaln is None:
            for layer in self.layers:
                q = layer(q, kv, cond, agents, agent_pad, index)""",
    """        if self.cascade is None and self.adaln is None:
            for layer in self.layers:
                q = layer(q, kv, cond, agents, agent_pad, index, bev, x_path)""")

sub("""        for i, layer in enumerate(self.layers):
            q = layer(q, kv, cond, agents, agent_pad, index)""",
    """        for i, layer in enumerate(self.layers):
            q = layer(q, kv, cond, agents, agent_pad, index, bev, x_path)""")

# --------------------------------------------------------------------- 7 ----
sub("""    def _sample(self, kv: Tensor, cond: Tensor, bank: Tensor,
                v_ms: Tensor | None, steps: int,
                agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor, dict]:""",
    """    def _sample(self, kv: Tensor, cond: Tensor, bank: Tensor,
                v_ms: Tensor | None, steps: int,
                agents: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None,
                bev: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor, dict]:""")

sub("""            conf, du = self._decode_ctrl(kv, cond, x_path, tt,
                                         agents, agent_pad, agent_pos)""",
    """            conf, du = self._decode_ctrl(kv, cond, x_path, tt,
                                         agents, agent_pad, agent_pos, bev)""")

# --------------------------------------------------------------------- 8 ----
# AnchoredDiffusionDecoder.forward: accept `bev` and thread it to every pass.
sub("""                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None,
                ego_hist: Tensor | None = None) -> dict:""",
    """                agent_tokens: Tensor | None = None,
                agent_pad: Tensor | None = None,
                agent_pos: Tensor | None = None,
                ego_hist: Tensor | None = None,
                bev: Tensor | None = None) -> dict:""")

sub("""                c_s, o_s = self._decode(kv, cond, xs, 0,
                                        agent_tokens, agent_pad, agent_pos)""",
    """                c_s, o_s = self._decode(kv, cond, xs, 0,
                                        agent_tokens, agent_pad, agent_pos,
                                        bev)""")

# --------------------------------------------------------------------- 9 ----
# RefCModel: the declared config field, and the attach as the LAST statement.
sub('''    wp_index: "object | None" = None''',
    '''    wp_index: "object | None" = None
    #: refcv6 coupling (1) -- `refc_bev_coupling.BEVCouplingConfig | None`, a
    #: DECLARED field for the same reason `wp_index` is one: the trainer assigns
    #: `core.decoder.bev_coupling = bcfg`, and on a dataclass without the field
    #: that assignment would be a silent no-op on a frozen instance.
    bev_coupling: "object | None" = None
    #: width of the BEV feature map coupling (1) reads -- the BEV encoder's
    #: `d_out`. ⛔ NOT a backbone constant: it is downstream of the stride-16
    #: map, whose own width is 1024 on resnet101 and 256 on resnet34 and is read
    #: from `feature_info` (`CNNEncoderConfig.s16_dim`), never written down.
    bev_coupling_d_bev: int = 96''')

sub("""        _wp = getattr(cfg.decoder, "wp_index", None)
        if _wp is not None:
            self.decoder.attach_wp_index(_wp)""",
    """        _wp = getattr(cfg.decoder, "wp_index", None)
        if _wp is not None:
            self.decoder.attach_wp_index(_wp)
        # ⭐⭐ refcv6 coupling (1) -- attached AFTER WP-B and LAST of all, so
        # that turning the BEV coupling on leaves every WP-B head bit-identical
        # (see `attach_bev_coupling`'s docstring). Anything added below this line
        # breaks BOTH removability proofs.
        _bevc = getattr(cfg.decoder, "bev_coupling", None)
        if _bevc is not None:
            self.decoder.attach_bev_coupling(
                _bevc, int(getattr(cfg.decoder, "bev_coupling_d_bev", 96)))""")

# -------------------------------------------------------------------- 10 ----
# RefCModel.forward: accept the BEV features and hand them to the decoder.
sub("""                agent_gt: dict | None = None,
                ego_poses: Tensor | None = None,
                ego_n_past: int | None = None) -> dict:""",
    """                agent_gt: dict | None = None,
                ego_poses: Tensor | None = None,
                ego_n_past: int | None = None,
                bev: Tensor | None = None) -> dict:""")

sub("""                           agent_pos=agent_pos, ego_hist=ego_vec)""",
    """                           agent_pos=agent_pos, ego_hist=ego_vec, bev=bev)""")

# -------------------------------------------------------------------- 11 ----
# ⭐ NEW IN THE REBASE. The landed timm trunk COMPUTES a stride-16 map and
# then throws it away (`TimmResNetTrunk.forward` returns only `(s32, pooled)`
# from a `forward_features` that produced `(s16, s32, pooled)`). refcv6's whole
# perception branch reads stride 16 -- an oracle on stride 32 caps at AP 0.3341
# vs 0.4713 (SPEC_REFCV6_V2.md §2) -- so without this seam the caller has to
# run the 21.8-45 M backbone a SECOND time to get a map it already built.
sub("""        b, w = frames.shape[:2]
        if self.cfg.hierarchy:
            fmap_all, pooled_all = self.encoder(
                frames.reshape(b * w, *frames.shape[2:]))
            pooled_seq = pooled_all.reshape(b, w, -1)
            pooled = pooled_seq[:, -1]
            fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
            ctx = self.strategic(pooled_seq)
        else:                                    # last frame only (same values)
            fmap, pooled = self.encoder(frames[:, -1])
            ctx = None""",
    """        b, w = frames.shape[:2]
        # ⭐⭐ refcv6 PERCEPTION SEAM. `fmap_s16` is the stride-16 map the
        # trunk ALREADY produced, exposed for the BEV lift -> map head + 3-D box
        # branch. `None` only on the in-repo REF-C trunk, which emits stride 32
        # alone (`CNNEncoderConfig.s16_dim` RAISES rather than inventing a
        # width). ⛔ Its channel count is `cfg.encoder.s16_dim`, read from
        # timm's `feature_info` (1024 on resnet101, 256 on resnet34) and never
        # written down.
        # ⚠️ BOTH branches are wired. An earlier cut left the hierarchy path
        # `None` and called that "reported rather than half-wired" -- but
        # `refc_smoke_config()` and every hierarchy arm set `hierarchy = True`,
        # so the seam would have been dead on the path actually taken. Here it
        # is the SAME last-frame reshape `fmap` already gets, one axis wider.
        fmap_s16 = None
        _s16 = hasattr(self.encoder, "forward_features")
        if self.cfg.hierarchy:
            if _s16:
                s16_all, fmap_all, pooled_all = self.encoder.forward_features(
                    frames.reshape(b * w, *frames.shape[2:]))
                fmap_s16 = s16_all.reshape(b, w, *s16_all.shape[1:])[:, -1]
            else:
                fmap_all, pooled_all = self.encoder(
                    frames.reshape(b * w, *frames.shape[2:]))
            pooled_seq = pooled_all.reshape(b, w, -1)
            pooled = pooled_seq[:, -1]
            fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
            ctx = self.strategic(pooled_seq)
        else:                                    # last frame only (same values)
            if _s16:
                # ONE trunk pass, three outputs. `fmap`/`pooled` are the SAME
                # tensors `self.encoder(...)` returns -- its `forward` is
                # literally `forward_features(...)[1:]` -- so this branch is
                # bit-identical to the line it replaces.
                fmap_s16, fmap, pooled = self.encoder.forward_features(
                    frames[:, -1])
            else:
                fmap, pooled = self.encoder(frames[:, -1])
            ctx = None""")

# and hand it back, so a caller does not have to re-run the trunk
sub("""        out = {"pooled": pooled, "traj": traj, "wp_seq": traj,""",
    """        out = {"pooled": pooled, "traj": traj, "wp_seq": traj,
               # refcv6: the perception branch's input, emitted beside the
               # plan. A READ of what the trunk already built -- adding a key
               # cannot change any existing one, and the bit-identity proof
               # covers every tensor that was there before. `None` on the
               # in-repo REF-C trunk and on the hierarchy path.
               "fmap_s16": fmap_s16,""")

open(P, "wb").write((s.replace("\n", "\r\n") if CRLF else s).encode("utf-8"))
print("all edits applied")
