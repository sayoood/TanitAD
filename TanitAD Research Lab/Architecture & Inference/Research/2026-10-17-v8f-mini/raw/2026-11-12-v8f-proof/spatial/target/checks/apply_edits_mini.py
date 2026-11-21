"""Edits of stack/tanitad/models/v8f_mini.py for EGOV and FUTTOK (run once on tree_patch). Each edit must match exactly once."""
p = "C:/Users/Admin/v8proof_c/target/tree_patch/stack/tanitad/models/v8f_mini.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# constants
rep('''MOTION_CFG_DEFAULTS = {"wm_l1_residual": "off", "wm_l1_roll_k": 0, "w_wm_roll": 0.0, "wm_l1_diag": "off"}''',
    '''WM_ACT_MODES = ("acc_yaw", "speed")                       # TARGET lever EGOV (--wm-l1-act): the level-1 action of the WM auxiliary: acc_yaw = (acceleration, ch1) today; speed = + the measured ego speed of the start frame
WM_SPEED_SCALE = 10.0                                     # TARGET lever EGOV: the third action channel is the logged ego speed [m/s] of the start frame of a transition divided by this constant
MOTION_CFG_DEFAULTS = {"wm_l1_residual": "off", "wm_l1_roll_k": 0, "w_wm_roll": 0.0, "wm_l1_diag": "off",
                       "wm_l1_act": "acc_yaw", "wm_l1_futtok_k": 0, "w_wm_futtok": 0.0}      # + TARGET levers EGOV / FUTTOK: same rule, recorded in a checkpoint cfg only when they differ from the default''')

# config fields
rep('''    wm_strides: tuple = (2, 2)
    w_ckpt: float = 1.0''', '''    wm_l1_act: str = "acc_yaw"         # TARGET lever EGOV: acc_yaw | speed (level-1 action gets a third channel: the logged ego speed of the start frame / WM_SPEED_SCALE)
    wm_l1_futtok_k: int = 0            # TARGET lever FUTTOK: K, rollout length of the token-read term of level 1 (0 = off; else 2 <= K <= hmax - 1, needs wm_state grid)
    w_wm_futtok: float = 0.0           # TARGET lever FUTTOK: W, the weight of futtok/L1 in the level-1 loss (0.0 exactly when K is 0; > 0 when K >= 2)
    wm_strides: tuple = (2, 2)
    w_ckpt: float = 1.0''')

# validation
rep('''        if self.wm_l1_roll_k > self.hmax - 1:
            raise ValueError(f"wm_l1_roll_k {self.wm_l1_roll_k} does not fit the training sequence: the WM sees hmax = {self.hmax} frames, so K <= {self.hmax - 1}")
''', '''        if self.wm_l1_roll_k > self.hmax - 1:
            raise ValueError(f"wm_l1_roll_k {self.wm_l1_roll_k} does not fit the training sequence: the WM sees hmax = {self.hmax} frames, so K <= {self.hmax - 1}")
        if self.wm_l1_act not in WM_ACT_MODES:
            raise ValueError(f"wm_l1_act must be one of {WM_ACT_MODES}, got {self.wm_l1_act!r}")
        if not (isinstance(self.wm_l1_futtok_k, int) and not isinstance(self.wm_l1_futtok_k, bool) and self.wm_l1_futtok_k >= 0):
            raise ValueError(f"wm_l1_futtok_k must be an integer >= 0, got {self.wm_l1_futtok_k!r}")
        if not (math.isfinite(self.w_wm_futtok) and self.w_wm_futtok >= 0.0):
            raise ValueError(f"w_wm_futtok must be a finite number >= 0, got {self.w_wm_futtok!r}")
        if not ((self.wm_l1_futtok_k == 0 and self.w_wm_futtok == 0.0) or (self.wm_l1_futtok_k >= 2 and self.w_wm_futtok > 0.0)):
            raise ValueError(f"wm_l1_futtok_k {self.wm_l1_futtok_k} and w_wm_futtok {self.w_wm_futtok}: need K = 0 with W = 0.0 (off) or K >= 2 with W > 0 (a token term with no weight, or a weight with no term, is a lever that does nothing)")
        if self.wm_l1_futtok_k > self.hmax - 1:
            raise ValueError(f"wm_l1_futtok_k {self.wm_l1_futtok_k} does not fit the training sequence: the WM sees hmax = {self.hmax} frames, so K <= {self.hmax - 1}")
        if self.wm_l1_futtok_k > 0 and self.wm_state != "grid":
            raise ValueError("wm_l1_futtok_k > 0 needs wm_state grid: the token read is per cell, and a mean state has no cells")
        if self.wm == "off" and (self.wm_l1_act != "acc_yaw" or self.wm_l1_futtok_k != 0):
            raise ValueError("wm_l1_act / wm_l1_futtok_k need the world-model auxiliary (wm a or b): with wm off there is no level-1 predictor")
''')

# WMAux build
rep('''        lv = [mk(latent_dim=cfg.wm_l1_dim, pred_width=cfg.wm_l1_pred_width, action_dim=2, act_embed_dim=32, **kind),''',
    '''        if cfg.wm_l1_futtok_k > 0:                                      # TARGET lever FUTTOK
            kind["futtok_k"], kind["futtok_weight"] = cfg.wm_l1_futtok_k, cfg.w_wm_futtok
        self.l1_act = cfg.wm_l1_act                                     # TARGET lever EGOV: speed = a third level-1 action channel (the stack's level 1 is built with action_dim 3)
        lv = [mk(latent_dim=cfg.wm_l1_dim, pred_width=cfg.wm_l1_pred_width, action_dim=3 if cfg.wm_l1_act == "speed" else 2, act_embed_dim=32, **kind),''')

rep('''        self.ch1_scale = CH1_SCALE[cfg.kin_mode]

    def z1(self''', '''        self.ch1_scale = CH1_SCALE[cfg.kin_mode]
        self.tok_head = None
        if cfg.wm_l1_futtok_k > 0:                                      # TARGET lever FUTTOK: built ONLY when on (no new key in the state dict when off); ONE linear map, the same weights for every cell
            self.tok_head = nn.Linear(cfg.wm_cell_dim, cfg.d_feat)
        self.tok_dim = cfg.d_feat

    def z1(self''')

# scale_actions + forward
rep('''    def scale_actions(self, act: Tensor) -> Tensor:
        """Raw action channels ``act`` [..., 2] = (a [m/s^2], ch1) -> the WM's action units. ch1 = the recorded steer (kin_mode steer) or
        the MEASURED yaw rate (kin_mode yaw, PI ruling D-PI-YAWRATE-1). ONE spelling for training and for the beta planner."""
        return torch.stack([act[..., 0] / WM_ACC_SCALE, act[..., 1] / self.ch1_scale], -1)

    def forward(self, adapt: Tensor, act: Tensor) -> dict:
        z1 = self.z1(adapt)                                                     # [B, F, D]
        a = self.scale_actions(act)[:, :-1]                                     # action of the transition j -> j+1
        if self.diag:                                                           # MOTION diagnostics: extra no-grad keys only, the terms and the total are the same
            return self.stack.loss(z1, a, diag=True)
        return self.stack.loss(z1, a)
''', '''    def scale_actions(self, act: Tensor, speed: Tensor | None = None) -> Tensor:
        """Raw action channels ``act`` [..., 2] = (a [m/s^2], ch1) -> the WM's action units. ch1 = the recorded steer (kin_mode steer) or
        the MEASURED yaw rate (kin_mode yaw, PI ruling D-PI-YAWRATE-1). ONE spelling for training and for the beta planner.
        TARGET lever EGOV: with ``wm_l1_act speed`` the result is [..., 3]: the third channel is the logged ego speed ``speed`` [...] (m/s, at the frame the channels of ``act`` belong to)
        divided by ``WM_SPEED_SCALE``; ``speed`` is REQUIRED then. With the lever off the result is [..., 2], exactly today's tensor (a ``speed`` passed anyway is not read)."""
        two = torch.stack([act[..., 0] / WM_ACC_SCALE, act[..., 1] / self.ch1_scale], -1)
        if self.l1_act != "speed":
            return two
        if speed is None:
            raise ValueError("this world-model auxiliary was built with --wm-l1-act speed (level-1 action width 3) but no `speed` was given: pass the logged ego speed of every frame "
                             "(tanitad.models.v8f_mini.wm_frame_speed(batch) = the `wm_speed` field of the batch)")
        if tuple(speed.shape) != tuple(act.shape[:-1]):
            raise ValueError(f"speed {tuple(speed.shape)} must match the frames of act {tuple(act.shape[:-1])}")
        return torch.cat([two, (speed.to(two.dtype) / WM_SPEED_SCALE)[..., None]], -1)

    def forward(self, adapt: Tensor, act: Tensor, speed: Tensor | None = None, feats: Tensor | None = None) -> dict:
        z1 = self.z1(adapt)                                                     # [B, F, D]
        a = self.scale_actions(act, speed)[:, :-1]                              # action of the transition j -> j+1 (EGOV: the speed of the START frame j)
        kw = {}
        if self.tok_head is not None:                                           # TARGET lever FUTTOK: the frozen token grid of every frame, normalised over its feature axis without parameters, never with a gradient
            if feats is None:
                raise ValueError("FUTTOK is on but WMAux.forward got no `feats` (the cached token grid): the term has no target (--encoder lora batches carry none)")
            if feats.shape[1] != adapt.shape[1] or feats.shape[2] != self.n_tok or feats.shape[-1] != self.tok_dim:
                raise ValueError(f"feats {tuple(feats.shape)} do not match the adapter tokens {tuple(adapt.shape[:3])} / token width {self.tok_dim}")
            kw = {"tok_head": self.tok_head, "tok_tgt": F.layer_norm(feats.detach().float(), (self.tok_dim,))}
        if self.diag:                                                           # MOTION diagnostics: extra no-grad keys only, the terms and the total are the same
            kw["diag"] = True
        return self.stack.loss(z1, a, **kw)
''')

# loss call + logging
rep('''            wm = self.wm(out["adapt"], batch["act"])
            T["wm"] = wm["total"]''', '''            if cfg.wm_l1_act == "speed" or cfg.wm_l1_futtok_k > 0:          # TARGET levers: the extra inputs are read ONLY when a lever is on (off = the same call as before)
                wm = self.wm(out["adapt"], batch["act"], speed=wm_frame_speed(batch) if cfg.wm_l1_act == "speed" else None,
                             feats=batch.get("feats") if cfg.wm_l1_futtok_k > 0 else None)
            else:
                wm = self.wm(out["adapt"], batch["act"])
            T["wm"] = wm["total"]
            if cfg.wm_l1_futtok_k > 0:
                T["wm_futtok"] = wm["futtok/L1"].detach()                   # TARGET lever FUTTOK: logged (and required by G-LIVE) whenever it is declared; it is inside T["wm"] already''')
rep('''                if cfg.wm_l1_roll_k > 0:
                    T["wm_roll_l1"], T["wm_roll_copy_l1"] = wm["roll/L1"].detach(), wm["diag/roll_copy_L1"]
''', '''                if cfg.wm_l1_roll_k > 0:
                    T["wm_roll_l1"], T["wm_roll_copy_l1"] = wm["roll/L1"].detach(), wm["diag/roll_copy_L1"]
                if cfg.wm_l1_futtok_k > 0:
                    T["wm_futtok_copy"] = wm["diag/futtok_copy"]
''')

# built levers
rep('''            lv["wm_l1_diag"] = "on" if self.wm.diag else "off"
''', '''            lv["wm_l1_diag"] = "on" if self.wm.diag else "off"
            lv["wm_l1_act"] = "speed" if int(self.wm.stack.act_embed[0].in_features) == 3 else "acc_yaw"       # TARGET lever EGOV: read from the BUILT level-1 action embedding (its input width: 3 channels = speed)
            lv["wm_l1_futtok_k"] = int(self.wm.stack.cfg[0].futtok_k) if self.wm.tok_head is not None else 0         # TARGET lever FUTTOK: the K and the W the stack's loss reads, and only when the head is built
            lv["w_wm_futtok"] = float(self.wm.stack.cfg[0].futtok_weight) if self.wm.tok_head is not None else 0.0
''')

# helper
rep('''class WMAux(nn.Module):''', '''def wm_frame_speed(batch_or_item) -> Tensor:
    """TARGET lever EGOV: the logged ego speed [m/s] of every cached frame, [B, F] (or [F] for one item) -- the field ``wm_speed``, built in ``v8f_mini_data.ClipStore`` from the clip's
    logged speed array (``d["v"]``, the cache's ``v``). ONE function for the trainer (``V8FMini.loss``) and the read tools: ``WMAux.scale_actions(act, wm_frame_speed(batch))``."""
    try:
        return batch_or_item["wm_speed"]
    except KeyError:
        raise KeyError("the batch carries no `wm_speed` (the logged ego speed of every cached frame); build it with v8f_mini_data.item / collate of this tree") from None


class WMAux(nn.Module):''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
