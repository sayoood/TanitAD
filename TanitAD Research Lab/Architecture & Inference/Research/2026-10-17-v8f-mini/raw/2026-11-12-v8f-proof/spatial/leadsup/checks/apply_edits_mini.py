"""Edits of stack/tanitad/models/v8f_mini.py for LEADSUP (run once on tree_patch; argv[1] = tree root). Each edit must match exactly once."""
import sys

root = sys.argv[1]
p = f"{root}/stack/tanitad/models/v8f_mini.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- constants
rep('''TARGET_CFG_DEFAULTS = {"wm_l1_act": "acc_yaw", "wm_l1_futtok_k": 0, "w_wm_futtok": 0.0}
''', '''TARGET_CFG_DEFAULTS = {"wm_l1_act": "acc_yaw", "wm_l1_futtok_k": 0, "w_wm_futtok": 0.0}
LEAD_SUP_MODES = ("off", "on")                            # LEADSUP (--lead-sup): a supervised lead-vehicle head on the level-1 grid states of the last `lead_hist` frames (train-time label only; the driving path never reads it)
LEAD_SUP_DETACH_MODES = ("off", "on")                     # LEADSUP (--lead-sup-detach): on = the head reads state.detach(): its term reaches no parameter outside the head (the zero control and the comparison arm)
LEAD_SUP_GAP_SCALE_M = 50.0                               # LEADSUP: the gap output is gap / this [m]
LEAD_SUP_CLOS_SCALE_MS = 10.0                             # LEADSUP: the closing-speed output is closing speed / this [m/s]
LEAD_SUP_HUBER_GAP = 0.1                                  # LEADSUP: Huber delta in gap / 50 m units (= 5 m); the builder's choice, a module constant (LEADSUP_DEFS.md)
LEAD_SUP_HUBER_CLOS = 0.2                                 # LEADSUP: Huber delta in closing / 10 m/s units (= 2 m/s); the builder's choice
LEAD_SUP_FRAME_DIM = 64                                   # LEADSUP: per-frame linear width (640 -> 64)
LEAD_SUP_HIDDEN = 128                                     # LEADSUP: hidden width of the read-out MLP (hist * 64 -> 128 -> 3)
#: LEADSUP config fields and their defaults: recorded in a checkpoint cfg ONLY when they differ from the default (every earlier checkpoint keeps its key set; an absent key = the default)
LEADSUP_CFG_DEFAULTS = {"lead_sup": "off", "w_lead_sup": 0.0, "lead_hist": 5, "lead_sup_detach": "off"}
''')

# ---- config fields
rep('''    wm_strides: tuple = (2, 2)
    w_ckpt: float = 1.0''', '''    lead_sup: str = "off"              # LEADSUP: off | on (a supervised lead head on the level-1 grid states; needs wm_state grid)
    w_lead_sup: float = 0.0            # LEADSUP: weight of the lead_sup term in the total (0.0 exactly when off; > 0 when on)
    lead_hist: int = 5                 # LEADSUP: number of consecutive frames (ending at t) whose level-1 states the head reads
    lead_sup_detach: str = "off"       # LEADSUP: on = the head reads state.detach() (needs lead_sup on)
    wm_strides: tuple = (2, 2)
    w_ckpt: float = 1.0''')

# ---- validation
rep('''        if self.cons_absent not in ("mask", "zero"):
            raise ValueError(f"cons_absent must be mask|zero, got {self.cons_absent!r}")
''', '''        if self.lead_sup not in LEAD_SUP_MODES:
            raise ValueError(f"lead_sup must be one of {LEAD_SUP_MODES}, got {self.lead_sup!r}")
        if self.lead_sup_detach not in LEAD_SUP_DETACH_MODES:
            raise ValueError(f"lead_sup_detach must be one of {LEAD_SUP_DETACH_MODES}, got {self.lead_sup_detach!r}")
        if not (isinstance(self.lead_hist, int) and not isinstance(self.lead_hist, bool) and self.lead_hist >= 1):
            raise ValueError(f"lead_hist must be an integer >= 1, got {self.lead_hist!r}")
        if not (math.isfinite(self.w_lead_sup) and self.w_lead_sup >= 0.0):
            raise ValueError(f"w_lead_sup must be a finite number >= 0, got {self.w_lead_sup!r}")
        if self.lead_sup == "off" and (self.w_lead_sup != 0.0 or self.lead_sup_detach != "off" or self.lead_hist != 5):
            raise ValueError("w_lead_sup / lead_sup_detach / lead_hist are set but lead_sup is off: a lever that is declared and does nothing")
        if self.lead_sup == "on":
            if not self.w_lead_sup > 0.0:
                raise ValueError(f"lead_sup on with w_lead_sup {self.w_lead_sup}: a head with no loss weight is a lever that is declared and does nothing")
            if self.wm == "off" or self.wm_state != "grid":
                raise ValueError("lead_sup on needs the world-model auxiliary with wm_state grid: the head reads the level-1 grid states")
            if self.lead_hist > self.hmax:
                raise ValueError(f"lead_hist {self.lead_hist} does not fit the training sequence (hmax {self.hmax})")
        if self.cons_absent not in ("mask", "zero"):
            raise ValueError(f"cons_absent must be mask|zero, got {self.cons_absent!r}")
''')

# ---- head module (before WMAux)
rep('''class WMAux(nn.Module):
    """M2: the H-JEPA world-model auxiliary''', '''class LeadSupHead(nn.Module):
    """LEADSUP: reads the level-1 grid states [B, F, D] of `hist` consecutive frames: a per-frame linear D -> 64, the `hist` embeddings concatenated (hist * 64), then Linear(hist * 64, 128) - GELU - Linear(128, 3).
    Output [B, F - hist + 1, 3] = (presence logit, gap / 50 m, closing speed / 10 m/s) at the END frames t = hist - 1 .. F - 1. Inputs are the states only (no ego, nav, limit, label): the labels enter the loss, never the head."""

    def __init__(self, state_dim: int, hist: int):
        super().__init__()
        self.hist = hist
        self.frame = nn.Linear(state_dim, LEAD_SUP_FRAME_DIM)
        self.mlp = nn.Sequential(nn.Linear(hist * LEAD_SUP_FRAME_DIM, LEAD_SUP_HIDDEN), nn.GELU(), nn.Linear(LEAD_SUP_HIDDEN, 3))

    def forward(self, z: Tensor) -> Tensor:
        F_ = z.shape[1]
        if F_ < self.hist:
            raise ValueError(f"lead head needs at least {self.hist} frames, got {F_}")
        e = self.frame(z)                                                       # [B, F, 64]
        w = e.unfold(1, self.hist, 1).permute(0, 1, 3, 2)                       # [B, F - hist + 1, hist, 64]: window i = frames i .. i + hist - 1
        return self.mlp(w.flatten(2)).float()


class WMAux(nn.Module):
    """M2: the H-JEPA world-model auxiliary''')

# ---- WMAux build (LAST in WMAux: every module above keeps G's init draws)
rep('''        self.tok_dim = cfg.d_feat

    def z1(self''', '''        self.tok_dim = cfg.d_feat
        self.lead_head = None
        self.lead_detach = cfg.lead_sup_detach == "on"
        if cfg.lead_sup == "on":                                        # LEADSUP: built LAST in the WM (no draw before it moves; off = no module, no key in the state dict)
            self.lead_head = LeadSupHead(cfg.n_tok * cfg.wm_cell_dim, cfg.lead_hist)

    def z1(self''')

# ---- WMAux forward
rep('''        if self.diag:                                                           # MOTION diagnostics: extra no-grad keys only, the terms and the total are the same
            kw["diag"] = True
        return self.stack.loss(z1, a, **kw)''', '''        if self.diag:                                                           # MOTION diagnostics: extra no-grad keys only, the terms and the total are the same
            kw["diag"] = True
        res = self.stack.loss(z1, a, **kw)
        if self.lead_head is not None:                                          # LEADSUP: the head reads the SAME z1 the stack's loss used (detach on: state.detach())
            res = dict(res)
            res["lead_out"] = self.lead_head(z1.detach() if self.lead_detach else z1)
        return res''')

# ---- loss term
rep('''            if cfg.wm_l1_diag == "on":                                      # MOTION diagnostics: logged only, detached, not in the total''', '''            if cfg.lead_sup == "on":                                        # LEADSUP: computed ONLY when the head is built (off = no new tensor op, no new T key = the base, bit for bit)
                T.update(lead_sup_terms(wm["lead_out"], batch, cfg.lead_hist))
            if cfg.wm_l1_diag == "on":                                      # MOTION diagnostics: logged only, detached, not in the total''')
rep('''        total = total + cfg.w_stop_aux * T["stop_aux"] if cfg.stop_aux == "on" else total       # PREREG_V8F_STOPPERC arm A: ONE line (a mutation deletes it); head off adds nothing: the tip's total
''', '''        total = total + cfg.w_stop_aux * T["stop_aux"] if cfg.stop_aux == "on" else total       # PREREG_V8F_STOPPERC arm A: ONE line (a mutation deletes it); head off adds nothing: the tip's total
        total = total + cfg.w_lead_sup * T["lead_sup"] if cfg.lead_sup == "on" else total       # LEADSUP: ONE line (a mutation deletes it); head off adds nothing: the base total
''')

# ---- the term function (before wm_frame_speed)
rep('''def wm_frame_speed(batch_or_item) -> Tensor:''', '''def lead_sup_terms(out: Tensor, batch: dict, hist: int) -> dict:
    """LEADSUP: the three parts and their sum, from the head output ``out`` [B, F - hist + 1, 3] (end frames t = hist - 1 .. F - 1) and the per-frame batch labels ``lead_pres`` / ``lead_gapm`` / ``lead_clos`` [B, F]
    (``v8f_mini_data.lead_sup_label_arrays``; NaN = no label). A window end frame counts only when ALL ``hist`` frames behind it are real (``frame_ok``). BCE(presence, every counted frame) + Huber(gap / 50, frames
    with a gap label) + Huber(closing / 10, frames with a closing label); each part is the mean over its own mask. Returns T entries: ``lead_sup`` (live), the parts and the counts (detached)."""
    for k in ("lead_pres", "lead_gapm", "lead_clos"):
        if k not in batch:
            raise ValueError(f"lead_sup on but the batch carries no `{k}`: v8f_mini_data builds it when the dataset is made with lead_sup=True (the trainer does it from --lead-sup on)")
    fe = out.shape[1]
    sl = slice(hist - 1, hist - 1 + fe)
    okw = batch["frame_ok"].unfold(1, hist, 1).all(-1)                                  # [B, fe]
    pres, gap, clos = batch["lead_pres"][:, sl].float(), batch["lead_gapm"][:, sl].float(), batch["lead_clos"][:, sl].float()
    m_p = okw.float()
    m_g = (okw & torch.isfinite(gap)).float()
    m_c = (okw & torch.isfinite(clos)).float()
    bce = F.binary_cross_entropy_with_logits(out[..., 0], pres, reduction="none")
    hg = F.huber_loss(out[..., 1], torch.nan_to_num(gap) / LEAD_SUP_GAP_SCALE_M, reduction="none", delta=LEAD_SUP_HUBER_GAP)
    hc = F.huber_loss(out[..., 2], torch.nan_to_num(clos) / LEAD_SUP_CLOS_SCALE_MS, reduction="none", delta=LEAD_SUP_HUBER_CLOS)
    t_b = (bce * m_p).sum() / m_p.sum().clamp_min(1.0)
    t_g = (hg * m_g).sum() / m_g.sum().clamp_min(1.0)
    t_c = (hc * m_c).sum() / m_c.sum().clamp_min(1.0)
    return {"lead_sup": t_b + t_g + t_c, "lead_sup_bce": t_b.detach(), "lead_sup_gap": t_g.detach(), "lead_sup_clos": t_c.detach(),
            "lead_sup_n_pres": m_p.sum().detach(), "lead_sup_n_gap": m_g.sum().detach(), "lead_sup_n_clos": m_c.sum().detach()}


def wm_frame_speed(batch_or_item) -> Tensor:''')

# ---- built_levers
rep('''            lv["wm_cell_dim"] = int(self.wm.proj[1].out_features) if pr["location_sensitive"] else None''', '''            lh = self.wm.lead_head                                      # LEADSUP: read from the BUILT module (is there a head, its history width from its read-out layer's input, and whether the wiring detaches)
            lv["lead_sup"] = "on" if lh is not None else "off"
            lv["lead_hist"] = int(lh.mlp[0].in_features // LEAD_SUP_FRAME_DIM) if lh is not None else 5
            lv["lead_sup_detach"] = "on" if (lh is not None and self.wm.lead_detach) else "off"
            lv["w_lead_sup"] = float(self.cfg.w_lead_sup) if lh is not None else 0.0
            lv["wm_cell_dim"] = int(self.wm.proj[1].out_features) if pr["location_sensitive"] else None''')

open(p, "w", encoding="utf-8", newline="").write(s)
print("OK", p)
