"""REFe -- DriveZero's camera planner, reproduced with ONE front camera and a DINOv3-only backbone.

SCOPE (PI 2026-09-20): reproduce on THEIR data (nuPlan/NAVSIM). Two deliberate differences from
DriveZero, everything else is their recipe:
  1. ONE camera (CAM_F0) instead of their four (F0, B0, L0, R0);
  2. DINOv3 alone instead of their agglomerative DriveVFM (that becomes TanitVFM in Stage 4).

THEIR STUDENT, as banked from the paper (../2026-09-20-drivezero-deep-analysis/RESULT.md):
  4 cameras @ 960x512 | output 20 steps @ 5 Hz = 4 s | 64 proposals + a predicted score each
  backbone DriveVFM ViT-L FROZEN, only rank-32 Q/V LoRA trainable
  3D position embeddings, then 16 REGISTER tokens per camera (from DrivoR)
  ego-kinematics + command -> one ego token added to M trajectory queries -> decoder cross-attends
  scene tokens; a SEPARATE scoring decoder predicts the six PDM components (BCE)
  losses: winner-takes-all L1 to the teacher rollout + a proposal-scoring loss
  338.46 M full / 18.58 M trainable (5.49 %)

BACKBONE: ViT-L, by PI decision 2026-09-20 (see REFeConfig.backbone for the rationale and its
consequence). REFe is then 316.9 M / 11.88 M trainable with ONE camera. Their 338.46 M is ViT-L with
FOUR cameras, so the counts are close but are NOT the same configuration and REFe's budget is
REPORTED rather than matched. The comparable published anchor is now their DINOv3 ViT-L = 94.55
(Table A13) rather than the ViT-S row.

The backbone is a NAMED, PLUGGABLE variant (`REFeConfig.for_backbone`), never a hand-edit. All three
DINOv3 sizes are on D: and all three load strictly -- every checkpoint tensor consumed, no target
tensor left at random init. Weights come from the ungated timm mirror; facebook/dinov3-* returns 401
without a licence acceptance.

Dependencies are kept OUT of the eval venv beyond tiny --no-deps additions: installing into it is how
this project previously replaced torch with a wheel the driver could not run, and torch was
re-verified with a real CUDA conv2d after every install here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# DINOv3 variants, all ungated on the timm mirror and all present on D:.
# The published DINOv3-only scores are the reason to care which one is used:
#   ViT-S  93.88  (their Table 7 -- and DINOv2 ViT-S scores 93.88 too, i.e. at this scale the
#                  backbone provably does NOT matter, which is a real argument against ViT-S for
#                  an experiment whose whole point is the backbone)
#   ViT-B  not published
#   ViT-L  94.55  (their Table A13)  -- +0.67 over ViT-S, but REFe then costs 316.9 M and breaks
#                  the programme's sub-300M thesis. That tension is a PI decision, not a default.
BACKBONES = {
    "vits16": dict(width=384, depth=12, heads=6, weights="dinov3-vits16", published_pdms=93.88),
    "vitb16": dict(width=768, depth=12, heads=12, weights="dinov3-vitb16", published_pdms=None),
    "vitl16": dict(width=1024, depth=24, heads=16, weights="dinov3-vitl16", published_pdms=94.55),
}


@dataclass
class REFeConfig:
    # camera / backbone
    # PI DECISION 2026-09-20: ViT-L. Rationale recorded because it has a consequence worth keeping:
    # their published DINOv3 ViT-L is 94.55 against ViT-S 93.88 (+0.67), and at ViT-S their own
    # Table 7 shows DINOv2 == DINOv3 == 93.88, i.e. at small scale the backbone provably does not
    # matter -- which would have made a DINOv3-only experiment uninformative by construction.
    # CONSEQUENCE, stated not buried: REFe is then 316.9 M and EXCEEDS the programme's sub-300M
    # thesis. The PI chose this knowing that; REFe is a reproduction on their data, not the
    # deployed product, so the thesis is not asserted to bind here.
    backbone: str = "vitl16"  # one of BACKBONES; see the table above before changing it
    img_h: int = 512
    img_w: int = 960
    patch: int = 16
    width: int = 1024         # ViT-L; use REFeConfig.for_backbone(<name>) rather than hand-editing
    depth: int = 24
    heads: int = 16
    mlp_ratio: float = 4.0
    # ⭐ PI DECISION 2026-09-20 (second ruling): FOUR CAMERAS, exactly as the paper and the
    # reference configuration. This SUPERSEDES the earlier one-front-camera departure, so REFe now
    # differs from DriveZero in ONE variable only -- the backbone (DINOv3 instead of DriveVFM).
    # ⛔ THESE FOUR ARE THE PAPER'S (Table A12, p. 30), AND THAT IS THE WHOLE JUSTIFICATION.
    # An earlier comment here claimed they were "chosen for 360 coverage from nuPlan's eight".
    # That is FALSE and our own calibration refutes it: MEASURED over all C(8,4)=70 subsets on a
    # 0.05 deg azimuth grid, this set covers 70.76 % of azimuth with TWO 52.7 deg blind sectors at
    # +/-91..144 deg -- the rear quarters, i.e. the lane-change blind spots -- and ranks 22nd of 70.
    # F0/L1/R1/B0 would cover 78.10 %. ⚠️ The paper never claims 360 coverage anywhere (zero hits
    # for "field of view", "FOV" or "surround" in 32 pages), so this is NOT a defect in the
    # reproduction -- inventing a geometric rationale for it was. What IS true and worth saying:
    # F0/L0/R0 give a contiguous forward arc of ~191 deg with no holes.
    n_cameras: int = 4
    cameras: tuple = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0')

    # their tricks
    lora_rank: int = 32       # rank-32 on Q and V only
    n_registers: int = 16     # per camera, from DrivoR
    n_proposals: int = 64     # M, winner-takes-all
    horizon_steps: int = 20   # 20 @ 5 Hz = 4 s
    traj_dim: int = 3         # x, y, yaw
    n_score_components: int = 6   # the six PDM components, BCE
    # decoder
    dec_width: int = 256
    dec_depth: int = 4   # PAPER: "a 4-layer proposal decoder". Was 3 until 2026-09-20 --
                         # an UNINTENDED divergence, not a design choice, fixed on PI
                         # instruction. Nothing in our docs had justified the 3.
    # ⚠ THE PAPER DOES NOT STATE THE SCORING DECODER'S DEPTH. It specifies "a 4-layer
    # proposal decoder" and then only "a separate scoring decoder". Reusing dec_depth for
    # both is a CHOICE, not a quotation, so it gets its own field and says so -- otherwise a
    # later reader would cite the paper for a number the paper never gave.
    score_dec_depth: int | None = None   # None => mirror dec_depth
    reg_heads: int = 16       # head_dim 64 at width 1024, matching DINOv3 ViT-L itself
    # ⭐ PI RULING 2026-09-20: ratio 1 ACCEPTED (the sizing study's recommendation). Perceiver and
    # Perceiver IO state that latent blocks "preserve the dimensionality of their inputs"; a 4x
    # expansion here was a transformer-block habit. At width 1024 the expansion alone was
    # 8,388,608 params, and ratio 4 would put trainable at ~25.2 M against the paper's 18.58 M.
    # This is a SEPARATE axis from the PI's compress-at-1024 width ruling, and is now also ruled.
    reg_mlp_ratio: int = 1
    pos3d_depth_bins: int = 2 # PETR-style analytic 3D position encoding
    n_goal_freq: int = 4      # Fourier bands per goal coordinate
    # CAM_F0 intrinsics, MEASURED over all 1,349 nuPlan log DBs (the dominant rig, 20 of 21
    # vehicles; the other reads fx 1528.712 / cx 970.460 / cy 578.414). They belong in the config
    # because the encoding is a FUNCTION of them.
    # ⛔ THESE ARE THE NATIVE-RESOLUTION INTRINSICS AND MUST BE RESCALED TO THE INPUT.
    # MEASURED: applying 1920x1120 intrinsics to a 960x512 input gave a horizontal field of view of
    # **31.34 degrees instead of 62.85**, with x and y never crossing zero -- i.e. the "frustum"
    # covered a quarter of the image and sat off-centre. `cam_native_w/h` record the geometry the
    # intrinsics were measured at, and `_frustum` scales them to `img_w/img_h`.
    cam_fx: float = 1545.0
    cam_fy: float = 1545.0
    cam_cx: float = 960.0
    cam_cy: float = 560.0
    cam_native_w: int = 1920
    cam_native_h: int = 1080          # MEASURED from the camera table; cy is 560, NOT h/2
    # ⛔ THE IMAGES ARE NOT RECTIFIED, AND UNTIL 2026-09-21 THE FRUSTUM PRETENDED THEY WERE.
    # nuPlan's `camera.distortion` is the Caltech model (k1, k2, p1, p2, k3) -- PUBLISHED in the
    # devkit's `templates.py` -- and a grep over the whole devkit finds **no `undistort` anywhere**,
    # so the JPEGs on disk carry the lens distortion. Unprojecting them with an ideal pinhole
    # (u - cx)/fx misplaces every ray, and worst at the edges where the side cameras overlap:
    # MEASURED +4.27 deg at the horizontal edge, +6.39 deg at the corner (~11 m lateral at 60 m),
    # and the same world point at ego (25, 12, 0) reconstructs ~1.7 m APART from CAM_F0 vs CAM_L0.
    # ⭐ The cross-camera disagreement is the part that matters: metric agreement between views is
    # the entire reason the frustum is lifted into the ego frame, and distortion destroys it.
    # MEASURED across 64 driverl_val14 logs: 63 share this coefficient set; one log carries a
    # second rig (-0.352686, 0.182677, -0.001237, -0.001275, -0.068048), a ~1 % difference in k1.
    # Banked with provenance in data/nuplan_cam_calib.json.
    cam_distortion: tuple = (-0.356123, 0.172545, -0.00213, 0.000464, -0.05231)
    undistort: bool = True            # set False ONLY as a declared ablation arm
    undistort_iters: int = 20         # fixed-point inverse; 20 is ~1e-7 at this k1 (MEASURED)
    # ⚠️ A DECLARED DEPARTURE FROM THE PAPER, NOT ONE OF ITS INVARIANTS. PUBLISHED (p. 8):
    # "Candidate trajectories are detached before entering the scoring branch" -- the paper
    # detaches the TRAJECTORIES. REFe additionally detaches the scoring decoder's visual CONTEXT,
    # so the PDM loss reaches neither the backbone LoRA nor `scene_proj`. Since REFe's whole
    # purpose is comparing DINOv3 against DriveVFM, how much the encoder is adapted is the
    # variable under study -- so this is the ablation switch, and it is off by default only
    # because that is the behaviour every banked result so far was produced under.
    detach_scorer_context: bool = True
    # per-camera extrinsics, MEASURED from a real nuPlan log DB and banked at
    # data/nuplan_cam_calib.json. All eight share intrinsics; only the pose differs, which is
    # exactly the information a per-token table could never carry.
    # ⚠️ SCOPE THIS CLAIM HONESTLY. The encoding is now **resolution-free** and a function of the
    # CAMERA -- it was neither before. It is NOT yet a function of the RIG: these are ONE vehicle's
    # extrinsics baked into the config, while the corpus carries ~22 distinct extrinsic sets per
    # channel. The original justification for going analytic ("a learned table CANNOT condition on
    # geometry that varies across our corpus") applies to a hardcoded tuple just as much. To
    # realise it fully the extrinsics must arrive PER SAMPLE from the target tuple.
    # ⭐ Practically small -- the 22 rigs differ by centimetres -- but the claim must say
    # "resolution-free and per-camera", never "a function of the rig".
    cam_t: tuple = ((1.671059527953005, -0.015845052723399974, 1.498692086468587), (1.6525236869932511, 0.14301533004884703, 1.5037464761814885), (1.6252642952324325, -0.16984766018479558, 1.506127136781794), (-0.5014443985418284, 0.01454315307533865, 1.4500732895699808))
    cam_q: tuple = ((-0.48907127214206575, 0.510302361005295, -0.49953659413553425, 0.5008632370590135), (-0.6669006357745871, 0.6825368654798103, -0.20513261807068844, 0.21750305346801996), (-0.21792717623664432, 0.20689410777832956, -0.67526131123783, 0.6735909259851252), (0.5025840662459249, -0.49753586562491176, -0.5009195148518135, 0.49894584717502716))
    pos3d_near_m: float = 1.0
    pos3d_far_m: float = 60.0
    dec_heads: int = 8
    # goal / ego conditioning
    n_goal_points: int = 2    # MEASURED from their released config: goal_count_probs [0.5, 0.5]
    # ⛔ RETRACTED 2026-09-21 -- THIS WAS 9 "PER TABLE A2", AND TABLE A2 IS THE WRONG TABLE.
    # Its caption reads **"Structured TEACHER input schema"** (paper p. 25), and its neighbouring
    # rows are "Actor history 15-D / frame, <=96 actors" and "Vector-map segment 10-D / token" --
    # inputs the camera-only student provably does not have. Quoting its "Ego state 9-D" for the
    # STUDENT is a true number read outside its scope, the same class this package's own advisory
    # is about. ⚠️ The paper dimensions the student's ego input NOWHERE; all it says (p. 7) is
    # "the multi-view images of the current frame, **the ego kinematics**, and a navigation
    # command", and (p. 8) "the ego kinematics and the command are embedded into one ego token".
    # ⇒ Seven real KINEMATIC scalars, and nothing else: vx, vy, ax, ay, yaw_rate, steering, speed.
    #   * the old 8th slot was a HARDCODED 0.0 -- a pad, carrying nothing;
    #   * the 9-D version filled slots 8/9 with the ego's length and width, which are GEOMETRY, not
    #     kinematics, and on nuPlan are CONSTANT (a Pacifica throughout: 5.176 x 2.297 m). Two
    #     constants cost the whole banked corpus its validity and buy exactly zero information.
    # A corpus with varied vehicles would change this, and then it is a one-line config move.
    ego_dim: int = 7          # vx, vy, ax, ay, yaw_rate, tire_steering_angle, speed

    @classmethod
    def for_backbone(cls, name: str, **kw):
        """Build a config for a named DINOv3 variant, so size is never a hand-edit."""
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
        b = BACKBONES[name]
        return cls(backbone=name, width=b["width"], depth=b["depth"], heads=b["heads"], **kw)


class LoRALinear(nn.Module):
    """Frozen base projection plus a trainable rank-r update. Only A and B carry gradient."""

    def __init__(self, in_f: int, out_f: int, rank: int, bias: bool = True):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=bias)
        self.base.weight.requires_grad_(False)
        if bias and self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays zero => identity at init
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def build_axial_rope(gh: int, gw: int, dh: int, device=None, dtype=torch.float32,
                     base: float = 100.0):
    """DINOv3's axial rotary position encoding, in ITS OWN formulation.

    ⛔ OUR FIRST VERSION WAS SELF-CONSISTENT AND WRONG, AND MEASURABLY WORSE THAN NO ENCODING AT
    ALL. Against the released reference on the real ViT-S checkpoint: no rope at all gave relative
    L2 **0.585** / cosine **0.834**; ours gave **0.760** / **0.688**. A rotation the frozen trunk
    was not pretrained under moves its features FARTHER from their learned function than simply
    omitting the encoding. Nothing errors, nothing fails to load, and no tensor-consumption check
    can see it -- DINOv3 stores no rope tensor, so there is nothing to account against.
    ⭐ Three of six elements differed (base and the exponent ladder were already right):
      coordinates  reference uses PATCH CENTRES normalised to [-1, +1]; ours used raw integer
                   indices, so the angles scaled with image size instead of being resolution-free
      scale        reference multiplies by 2*pi; ours omitted it
      layout       reference is `tile(2)`, pairing channel i with i + D/2 (a HALF SPLIT); ours used
                   `repeat_interleave(2)`, pairing 2j with 2j+1 -- a different channel pairing than
                   the pretrained Q/K were learned under
    ⭐ EVIDENCE CLASS: **PUBLISHED**, not inherited. The primary sources are now BANKED on this box
    and every element was checked against them line by line:
      `TanitAD Research Lab/Library/refs/dinov3_rope_position_encoding.py` (sha256 af8ea35a…)
          coords_h = arange(0.5, H)/H ; coords = 2*coords - 1        -> patch centres in [-1, +1]
          periods  = base ** (2*arange(D//4) / (D//2))               -> identical ladder, base 100
          angles   = 2*pi * coords / periods ; flatten(1,2).tile(2)  -> identical 2*pi and layout
      `TanitAD Research Lab/Library/refs/dinov3_attention.py`
          rope_rotate_half(x): x1, x2 = x.chunk(2, -1); cat([-x2, x1], -1)
          rope_apply(x, sin, cos): (x * cos) + (rope_rotate_half(x) * sin)
    Both match this implementation exactly. Until 2026-09-20 the formulation was INHERITED from a
    review and could not be checked here; it is now quotable.
    Returns (cos, sin) of shape [gh*gw, dh].
    """
    assert dh % 4 == 0, f"axial RoPE needs dh divisible by 4, got {dh}"
    # patch CENTRES, normalised to [-1, +1] -- resolution-free, unlike raw indices
    y = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / gh * 2.0 - 1.0
    x = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / gw * 2.0 - 1.0
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).reshape(-1, 2)  # [N, 2]
    periods = base ** (2.0 * torch.arange(dh // 4, device=device, dtype=torch.float32)
                       / (dh // 2))                                                   # [dh//4]
    ang = 2.0 * math.pi * coords[:, :, None] / periods                                # [N, 2, dh//4]
    ang = ang.flatten(1, 2).tile(2)                                                   # [N, dh]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotate_half(x):
    """Pair channel i with i + D/2, which is what `tile(2)` above assumes."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(t, cos, sin, offset: int):
    """Rotate q/k. `offset` skips the cls + register prefix, which carries NO position."""
    if cos is None:
        return t
    n_pos = cos.shape[0]
    if t.shape[-2] < offset + n_pos:
        return t
    seg = t[..., offset:offset + n_pos, :]
    seg = seg * cos + _rotate_half(seg) * sin
    return torch.cat([t[..., :offset, :], seg, t[..., offset + n_pos:, :]], dim=-2)


class Attention(nn.Module):
    """Self-attention whose Q and V are LoRA-adapted; K and the output projection stay frozen.

    MEASURED from the real DINOv3 ViT-S checkpoint (timm/vit_small_patch16_dinov3.lvd1689m):
    it stores a FUSED `attn.qkv.weight` of shape (1152, 384) and carries NO qkv bias. Q/K/V are
    kept as three separate projections here so LoRA can touch Q and V only, and the loader splits
    the fused tensor into them. bias=False matches the checkpoint exactly -- inventing biases the
    checkpoint does not have would leave them at random init and silently perturb a frozen trunk.
    """

    def __init__(self, dim: int, heads: int, rank: int, qkv_bias: bool = False):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.q = LoRALinear(dim, dim, rank, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = LoRALinear(dim, dim, rank, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        for p in (*self.k.parameters(), *self.proj.parameters()):
            p.requires_grad_(False)

    def forward(self, x, rope=None, rope_offset: int = 0):
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dh).transpose(1, 2)
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin, rope_offset)
            k = apply_rope(k, cos, sin, rope_offset)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """DINOv3 block: pre-norm attention and MLP, each scaled by a learned LayerScale vector.

    MEASURED: the real checkpoint carries `gamma_1` and `gamma_2` of shape (384,) per block.
    Omitting LayerScale would leave 2 x 384 x 12 = 9,216 checkpoint values unconsumed and change
    the trunk's forward function, so it is part of the architecture, not an optional refinement.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, rank: int, qkv_bias: bool = False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, rank, qkv_bias=qkv_bias)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))
        for m in (self.n1, self.n2, self.mlp):
            for p in m.parameters():
                p.requires_grad_(False)
        self.gamma_1.requires_grad_(False)
        self.gamma_2.requires_grad_(False)

    def forward(self, x, rope=None, rope_offset: int = 0):
        x = x + self.gamma_1 * self.attn(self.n1(x), rope, rope_offset)
        return x + self.gamma_2 * self.mlp(self.n2(x))


class VitS16(nn.Module):
    """DINOv3 ViT-S/16 architecture. FROZEN except the rank-r Q/V LoRA inside each block."""

    N_DINOV3_REG = 4     # MEASURED: the checkpoint carries reg_token of shape (1, 4, 384)

    def __init__(self, cfg: REFeConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.width, cfg.patch, cfg.patch)
        for p in self.patch_embed.parameters():
            p.requires_grad_(False)
        self.gh, self.gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
        # ⛔ `self.pos`, a trunc_normal_ frozen table, USED TO LIVE HERE AND WAS A DEFECT.
        # DINOv3 stores no positional tensor at all -- it computes axial RoPE on the fly -- so the
        # table had no checkpoint counterpart (arithmetic: 305,045,504 - 303,079,424 = 1,966,080 =
        # 1920 x 1024) and fed the frozen trunk a RANDOM vector per patch it had never seen, while
        # withholding the encoding it was pretrained with. Removed 2026-09-20; RoPE is built below
        # and applied inside Attention.
        self._rope_cache: dict = {}
        # ⛔ THE FROZEN TRUNK WAS BEING FED UN-NORMALISED [0,1] IMAGES, AND THAT IS WORSE THAN THE
        # POSITIONAL DEFECT WE ALREADY FIXED. The checkpoint's own `config.json` declares
        # `mean [0.485, 0.456, 0.406]` and `std [0.229, 0.224, 0.225]`, and a grep over every
        # refe/*.py returned ZERO normalisation. MEASURED on the real checkpoint by the third
        # review: feature displacement rel L2 **0.562** / cos 0.841 -- LARGER than deleting the
        # positional encoding entirely (0.529) and 2.9x a BGR/RGB swap (0.194).
        # ⭐ Why it matters beyond accuracy: REFe exists to compare DINOv3 against DriveVFM. A trunk
        # evaluated at the wrong operating point is not the trunk being compared, so the whole
        # backbone ablation would have measured our preprocessing rather than their encoder.
        # Registered as BUFFERS so every consumer -- trainer, planner, validator -- gets it and
        # nobody has to remember.
        self.register_buffer("img_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        # DINOv3's OWN class token and 4 register tokens. These are pretrained and the trunk's
        # attention was learned with them present, so dropping them would change the frozen
        # function -- they are NOT the same thing as DriveZero's 16 task registers per camera,
        # which are added after the trunk. Caught by the loader's no-unconsumed-tensor control.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.width), requires_grad=False)
        self.reg_token = nn.Parameter(torch.zeros(1, self.N_DINOV3_REG, cfg.width),
                                      requires_grad=False)
        self.blocks = nn.ModuleList(
            [Block(cfg.width, cfg.heads, cfg.mlp_ratio, cfg.lora_rank) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.width)
        for p in self.norm.parameters():
            p.requires_grad_(False)

    def _rope(self, gh, gw, device, dtype):
        key = (gh, gw, str(device), str(dtype))
        if key not in self._rope_cache:
            self._rope_cache[key] = build_axial_rope(gh, gw, self.cfg.width // self.cfg.heads,
                                                     device=device, dtype=dtype)
        return self._rope_cache[key]

    def forward(self, img):                      # [B, 3, H, W] -> [B, gh*gw, width]
        B = img.shape[0]
        img = (img - self.img_mean) / self.img_std      # the checkpoint's declared operating point
        pe = self.patch_embed(img)
        gh, gw = pe.shape[-2], pe.shape[-1]      # derive from the ACTUAL map, not from the config
        x = pe.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1),
                       self.reg_token.expand(B, -1, -1), x], dim=1)
        # the cls token and DINOv3's 4 registers carry NO position and must not be rotated
        rope = self._rope(gh, gw, x.device, x.dtype)
        off = 1 + self.N_DINOV3_REG
        for b in self.blocks:
            x = b(x, rope, off)
        x = self.norm(x)
        return x[:, 1 + self.N_DINOV3_REG:]      # drop cls + DINOv3 registers, keep patch tokens


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ctx: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, kdim=ctx, vdim=ctx, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, ctx):
        h = self.n1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.cross(h, ctx, ctx, need_weights=False)[0]
        return q + self.mlp(self.n3(q))


def _quat_to_mat(q):
    """w,x,y,z quaternion -> 3x3 rotation, as nuPlan stores camera rotations."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])])


class RegisterCompress(nn.Module):
    """Learnable registers COMPRESS the visual tokens into a small set of scene tokens.

    Paper §2.3, following DrivoR: "learnable registers then compress these tokens into a small set
    of scene tokens". Registers are the queries, the visual tokens are keys and values, so the
    output has exactly `n_registers * n_cameras` tokens however many patches came in -- which is
    what makes it a bottleneck rather than an append.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 1):
        super().__init__()
        # ⭐ MLP RATIO 1, NOT 4 (sizing study 2026-09-20). Perceiver and Perceiver IO state that
        # latent blocks "preserve the dimensionality of their inputs"; a 4x expansion here is a
        # transformer-block habit, not a compression-block design. At width 1024 the expansion
        # alone cost 8,388,608 of this module's 12,598,272 parameters.
        # ⭐ HEADS 16, NOT 8 -- FREE. head_dim 64 is the convention the ViT scaling work uses, and
        # DINOv3 ViT-L itself is 16 heads at 1024. Head count does not change the parameter count;
        # it changes how the attention partitions, and we were at head_dim 128 by accident.
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        h = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, reg, visual):
        h = self.n1(reg)
        reg = reg + self.cross(h, self.n2(visual), self.n2(visual), need_weights=False)[0]
        return reg + self.mlp(self.n3(reg))


class REFe(nn.Module):
    def __init__(self, cfg: REFeConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or REFeConfig()
        self.backbone = VitS16(cfg)
        n_patch = cfg.n_cameras * (cfg.img_h // cfg.patch) * (cfg.img_w // cfg.patch)
        # 16 register tokens per camera (DrivoR, adopted by DriveZero) -- trainable
        self.registers = nn.Parameter(
            torch.zeros(1, cfg.n_cameras * cfg.n_registers, cfg.width))
        nn.init.trunc_normal_(self.registers, std=0.02)
        # 3D position embedding added to patch tokens (their "3D position embeddings")
        # ⛔ THE LEARNED TABLE WAS A 2-D POSITIONAL TABLE WEARING A 3-D NAME.
        # DriveZero p.8 says the visual tokens are "enriched with 3D position embeddings [50]" and
        # **[50] is PETR**, whose encoding is an MLP over ANALYTICALLY COMPUTED camera-frustum 3D
        # coordinates -- a function of the intrinsics and extrinsics, not a per-token lookup.
        # MEASURED over all 1,349 nuPlan log DBs: CAM_F0 has 2 distinct intrinsics and 22-24
        # distinct extrinsics, so a table storing one vector per token position CANNOT condition on
        # the camera geometry that varies across our own corpus.
        # ⚠️ It was also the same 1,966,080 shape as the bogus positional table deleted from the
        # trunk earlier the same day, for the same reason: resolution- and geometry-locked.
        nd = cfg.pos3d_depth_bins
        self.pos3d_mlp = nn.Sequential(
            nn.Linear(3 * nd, cfg.width // 2), nn.ReLU(inplace=True),
            nn.Linear(cfg.width // 2, cfg.width))
        self._pos3d_cache: dict = {}
        # set True only for a deliberate single-camera ablation, never by default
        self.allow_camera_mismatch = False
        # ⭐ PI DECISION 2026-09-20: COMPRESS AT THE BACKBONE WIDTH, PROJECT AFTERWARDS.
        # The paper's two statements -- "learnable registers then compress these tokens into a
        # small set of scene tokens" and "a 256-dimensional planning representation" -- do not fix
        # WHERE the compression happens. Both readings are admissible:
        #   (a) compress at 1024, then project to 256   <- THIS ONE, chosen by the PI
        #   (b) project to 256 first, then compress     <- 16x cheaper; measured 790,272 params
        # (a) lets the registers attend over the FULL-WIDTH visual features and only then discard
        # capacity, which is the richer operation; (b) discards first and compresses what is left.
        # ⚠️ The cost is stated rather than buried: `reg_compress` is 6,303,744 trainable at 1024
        # against 790,272 at 256, and it is why REFe's trainable count exceeds the paper's 18.58 M.
        # (This read 12,598,272 until 2026-09-21 -- the pre-`reg_mlp_ratio=1` figure, left behind by
        # the ruling recorded 40 lines above. The conclusion held; the margin was double.)
        # An ablation between (a) and (b) is the experiment that would settle it; none has run.
        self.reg_compress = RegisterCompress(cfg.width, cfg.reg_heads,
                                             mlp_ratio=cfg.reg_mlp_ratio)
        self.scene_proj = nn.Linear(cfg.width, cfg.dec_width)
        # ego kinematics + GOAL -> one ego token. The goal is the augmentation input (2 points).
        # ⭐ SINUSOIDAL GOAL ENCODING (sizing study). DriveZero's own teacher encodes the goal
        # point with Fourier features rather than feeding raw metres, and a planner's goal is a
        # POSITION whose useful resolution spans centimetres to tens of metres -- exactly the range
        # a linear layer on raw coordinates handles worst. `n_goal_freq` bands per coordinate.
        self.goal_freqs = nn.Parameter(
            2.0 ** torch.arange(cfg.n_goal_freq).float(), requires_grad=False)
        goal_dim = 2 * cfg.n_goal_points * (1 + 2 * cfg.n_goal_freq)
        self.ego_enc = nn.Sequential(
            nn.Linear(cfg.ego_dim + goal_dim, cfg.dec_width), nn.GELU(),
            nn.Linear(cfg.dec_width, cfg.dec_width))
        self.queries = nn.Parameter(torch.zeros(1, cfg.n_proposals, cfg.dec_width))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.dec = nn.ModuleList([CrossBlock(cfg.dec_width, cfg.dec_heads, cfg.dec_width)
                                  for _ in range(cfg.dec_depth)])
        self.traj_head = nn.Sequential(
            nn.Linear(cfg.dec_width, cfg.dec_width), nn.GELU(),
            nn.Linear(cfg.dec_width, cfg.horizon_steps * cfg.traj_dim))
        # SEPARATE scoring decoder fed DETACHED features: their design keeps the score loss from
        # backpropagating into the trajectory features.
        sdd = cfg.dec_depth if cfg.score_dec_depth is None else cfg.score_dec_depth
        self.score_dec = nn.ModuleList([CrossBlock(cfg.dec_width, cfg.dec_heads, cfg.dec_width)
                                        for _ in range(sdd)])
        # ⛔ CONFORMANCE GAP THE SIZING STUDY SURFACED. The paper: "a separate scoring decoder then
        # ENCODES EACH CANDIDATE TRAJECTORY AS A SCORE QUERY, attends to the visual tokens, and
        # predicts the six components". Ours fed the scoring decoder the detached proposal QUERIES
        # -- the latent that produced the trajectory, not the trajectory itself. A scorer that
        # never sees the geometry it is scoring is judging the intent rather than the path, and the
        # two differ precisely where the trajectory head is imperfect.
        self.score_q_mlp = nn.Sequential(
            nn.Linear(cfg.horizon_steps * cfg.traj_dim, cfg.dec_width), nn.GELU(),
            nn.Linear(cfg.dec_width, cfg.dec_width))
        self.score_head = nn.Linear(cfg.dec_width, cfg.n_score_components)

    def baked_calib(self, n_cam: int | None = None) -> tuple:
        """The config's ONE rig as per-camera 16-vectors (`calib_table.FIELDS` order).

        This is what every sample used before per-sample calibration (R22), and what `calib=None`
        still means -- so the default path is the same arithmetic on the same numbers.
        """
        c = self.cfg
        n_cam = c.n_cameras if n_cam is None else n_cam
        assert n_cam <= len(c.cam_t), (
            f"asked for {n_cam} cameras but only {len(c.cam_t)} extrinsics are configured -- "
            f"add the calibration rather than letting the encoding silently reuse a pose")
        return tuple((c.cam_fx, c.cam_fy, c.cam_cx, c.cam_cy, *c.cam_distortion,
                      *c.cam_t[i], *c.cam_q[i]) for i in range(n_cam))

    def _frustum(self, gh, gw, device, dtype, n_cam: int | None = None, calib=None):
        """PETR-style 3D coordinates per patch, per camera, expressed IN THE EGO FRAME.

        `calib`: None (the config's baked rig) or n_cam 16-vectors in `calib_table.FIELDS` order
        -- intrinsics, Caltech distortion and extrinsics PER CAMERA. ⛔⛔ R22: PETR builds this
        from EACH SAMPLE'S calibration, and ours baked one vehicle's: MEASURED 26-30 distinct rigs
        per camera over 2,774 local logs, the baked one exact on 3.2 %, rotation off by a median
        0.84-1.52 deg and up to 2.67 deg on navtrain's vehicles.

        ⛔ THE PREVIOUS VERSION WAS A TABLE BY ANOTHER ROUTE. It unprojected into the CAMERA frame
        and never applied the extrinsics, so two builds were bit-identical and the encoding could
        not distinguish one camera from another -- the very property it was justified by. With four
        cameras that is not merely wasteful, it is wrong: the same pixel in the front and rear
        images would receive the same 3D code.
        ⭐ Lifting each camera's frustum into the EGO frame by its own rotation and translation is
        what PETR does and is what makes the code a function of the RIG.
        """
        # ⛔ BUILD FOR THE CAMERAS ACTUALLY PRESENT, NOT FOR THE CONFIG'S COUNT. A mismatch is a
        # silent shape error at best and a wrong encoding at worst: the validator passes a
        # single-camera tensor and got a 7,680-token frustum against 1,920 tokens of features.
        # The camera axis is a property of the INPUT; the config only says what the rig is.
        n_cam = self.cfg.n_cameras if n_cam is None else n_cam
        # ⛔⛔ THE KEY USED TO BE (gh, gw, device, dtype, n_cam) ONLY -- IT OMITTED EVERY QUANTITY
        # THE FRUSTUM IS A FUNCTION OF. MEASURED 2026-09-21 by the sixth review: flipping
        # `undistort`, changing `cam_distortion`, or replacing the extrinsics on a LIVE instance
        # after one forward gave **max |delta| = 0.000000e+00** -- three different ways.
        # ⚠️ That made the declared distortion ablation INERT under exactly the mutation pattern
        # this package's own `diag_guard_audit.py` uses (which is why that file has to hand-call
        # `_pos3d_cache.clear()`), and no instrument anywhere exercised `undistort`.
        # ⭐ A cache key must name EVERY input the cached value depends on, or the cache silently
        # converts a configuration change into a no-op -- and an ablation that cannot move is
        # indistinguishable from an ablation that has no effect.
        # (`train.py --no-undistort` was safe only because it sets the flag BEFORE construction.)
        c_ = self.cfg
        # ONE code path for both sources: the baked rig is just the default calibration, so the
        # per-sample arm cannot drift from it and `calib=None` stays bit-identical to before R22.
        rig = self.baked_calib(n_cam) if calib is None else \
            tuple(tuple(float(v) for v in cam) for cam in calib)
        assert len(rig) == n_cam and all(len(r) == 16 for r in rig), (
            f"calibration must be {n_cam} x 16 (calib_table.FIELDS), got "
            f"{len(rig)} x {[len(r) for r in rig]}")
        # the key names EVERY input the frustum depends on -- including the rig (see below)
        key = (gh, gw, str(device), str(dtype), n_cam,
               bool(c_.undistort), int(c_.undistort_iters),
               c_.cam_native_w, c_.cam_native_h, c_.img_h, c_.img_w,
               c_.pos3d_near_m, c_.pos3d_far_m, c_.pos3d_depth_bins, rig)
        if key in self._pos3d_cache:
            return self._pos3d_cache[key]
        c = self.cfg
        sx, sy = c.img_w / c.cam_native_w, c.img_h / c.cam_native_h
        u = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * (c.img_w / gw)
        v = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * (c.img_h / gh)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        d = torch.linspace(c.pos3d_near_m, c.pos3d_far_m, c.pos3d_depth_bins,
                           device=device, dtype=torch.float32)
        out = []
        for i in range(n_cam):
            out.append(self._lift_camera(uu, vv, d, gh, gw, sx, sy, rig[i], device))
        pts = torch.cat(out, dim=0) / c.pos3d_far_m                  # [n_cam*P, 3*nd]
        res = pts.to(dtype).unsqueeze(0)
        self._pos3d_cache[key] = res
        return res

    def _lift_camera(self, uu, vv, d, gh, gw, sx, sy, cam, device):
        """One camera's patch rays -> undistorted -> lifted into the EGO frame: [P, 3*nd]."""
        c = self.cfg
        fx0, fy0, cx0, cy0, k1, k2, p1, p2, k3, tx, ty, tz, qw, qx, qy, qz = cam
        fx, fy = fx0 * sx, fy0 * sy
        cx, cy = cx0 * sx, cy0 * sy
        # observed (DISTORTED) normalised coordinates
        xd = (uu - cx) / fx
        yd = (vv - cy) / fy
        # ⭐ INVERT THE CALTECH POLYNOMIAL to recover the IDEAL ray each pixel actually looks along.
        # The forward model maps ideal -> distorted, so there is no closed form; this is the
        # standard fixed-point iteration (the same one OpenCV's undistortPoints uses). It is inside
        # the (gh, gw, device, dtype, n_cam) cache, so the cost is paid once per geometry.
        # ⚠️ `undistort=False` is the ablation arm and reproduces the old ideal-pinhole behaviour
        # EXACTLY -- that is what makes this change measurable rather than asserted.
        if c.undistort:
            xu, yu = xd.clone(), yd.clone()
            for _ in range(c.undistort_iters):
                r2 = xu * xu + yu * yu
                radial = 1.0 / (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
                dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
                dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
                xu, yu = (xd - dx) * radial, (yd - dy) * radial
            xd, yd = xu, yu
        x = xd[..., None] * d
        y = yd[..., None] * d
        z = d.expand_as(x)
        cam_pts = torch.stack([x, y, z], dim=-1).reshape(gh * gw, c.pos3d_depth_bins, 3)
        R = _quat_to_mat(torch.tensor((qw, qx, qy, qz), device=device, dtype=torch.float32))
        t = torch.tensor((tx, ty, tz), device=device, dtype=torch.float32)
        ego_pts = cam_pts @ R.T + t                                  # [P, nd, 3] in the EGO frame
        return ego_pts.reshape(gh * gw, -1)

    def _pos3d(self, img, n_cam: int | None = None, calib=None):
        """[1 or B, n_cam*P, width]. `calib` None -> the baked rig, broadcast over the batch.

        `calib` [B, n_cam, 16] -> each sample's OWN rig (PETR). A batch holds few distinct rigs
        (one per vehicle), so each UNIQUE rig's frustum is built once (and cached across steps)
        and the MLP runs once per unique rig; samples gather their rows. MEASURED ~30 rigs per
        camera over the whole local corpus, so the cache stays small.
        """
        gh, gw = img.shape[-2] // self.cfg.patch, img.shape[-1] // self.cfg.patch
        if n_cam is None:
            n_cam = img.shape[1] if img.dim() == 5 else 1
        if calib is None:
            return self.pos3d_mlp(self._frustum(gh, gw, img.device, img.dtype, n_cam))
        if calib.dim() != 3 or calib.shape[1] != n_cam or calib.shape[2] != 16:
            raise ValueError(f"calib must be [B, {n_cam}, 16], got {tuple(calib.shape)}")
        rows = [tuple(tuple(cam) for cam in smp)
                for smp in calib.detach().to("cpu", torch.float64).tolist()]
        uniq: dict = {}
        idx = [uniq.setdefault(r, len(uniq)) for r in rows]
        fr = torch.cat([self._frustum(gh, gw, img.device, img.dtype, n_cam, calib=r)
                        for r in uniq], dim=0)                        # [U, n_cam*P, 3*nd]
        emb = self.pos3d_mlp(fr)                                      # [U, n_cam*P, width]
        return emb[torch.tensor(idx, device=emb.device)]              # [B, n_cam*P, width]

    def forward(self, img, ego, goal, detach_scorer: bool = True, calib=None):
        """`detach_scorer=False` is the DELIBERATE-REGRESSION path and exists so the validator can
        exercise the REAL forward pass with one property flipped, instead of restating it.

        ⛔ WHY THIS FLAG EXISTS. `validate_model.py`'s regression arm hand-built the forward pass
        and drifted THREE TIMES IN ONE DAY: once when the registers began to compress, once when
        `pos3d` became an analytic MLP, and once when the goal gained Fourier features. Each time it
        raised an AttributeError or silently tested a path nobody ships. A guard that duplicates the
        thing it guards will always lag it. One flag on the real function cannot.
        """
        """img [B, N_cam, 3, H, W] (or [B,3,H,W] for a single camera); ego [B, ego_dim];
        goal [B, 2*n_goal_points]."""
        if img.dim() == 4:                       # [B,3,H,W] -> [B,1,3,H,W]
            img = img.unsqueeze(1)
        B, N = img.shape[0], img.shape[1]
        # ⛔ REFUSE A CAMERA-COUNT MISMATCH RATHER THAN TRAINING ON A QUARTER OF THE INPUT.
        # When the config was moved to four cameras the DATA PIPELINE was still single-camera
        # (`build_targets.py` queries CAM_F0 only and stores one image per tuple). Silently
        # unsqueezing a 1-camera batch into a 4-camera model trains on 25 % of the intended input
        # while every log line says "n_cameras=4" -- the exact silent-mismatch class this package
        # has now found a dozen times. An explicit `--n-cameras 1` is a legitimate ablation; an
        # accident is not, so the two must look different.
        if N != self.cfg.n_cameras and not self.allow_camera_mismatch:
            raise ValueError(
                f"REFe is configured for {self.cfg.n_cameras} cameras ({', '.join(self.cfg.cameras)})"
                f" but received {N}. Either feed all {self.cfg.n_cameras}, or run with "
                f"--n-cameras {N} so the run RECORDS what it actually trained on. "
                f"(This used to say 'set REFeConfig.n_cameras', which no CLI could do -- a guard "
                f"whose remedy required editing source made a legitimate single-camera ablation "
                f"look unsupported.)")
        # ⛔ ALL CAMERAS THROUGH THE SAME FROZEN TRUNK IN ONE BATCH. Folding the camera axis into
        # the batch is what makes the trunk shared rather than replicated -- the parameter count
        # must NOT grow with camera count, only the activation memory and the token count.
        flat = img.reshape(B * N, *img.shape[2:])
        tok = self.backbone(flat)                                    # [B*N, P, width]
        visual = tok.reshape(B, N * tok.shape[1], tok.shape[2])      # [B, N*P, width]
        visual = visual + self._pos3d(img, N, calib)                 # per-camera 3D position
        # ⛔ THE REGISTERS NOW COMPRESS. They used to be CONCATENATED
        # (`cat([tok, registers])`), which made them 16 free bias parameters rather than a
        # compression basis: MEASURED by forward hooks, every decoder layer received a context of
        # `visual + 16` tokens with the IDENTICAL storage pointer, and no decoder ever saw a
        # 16-token context. At 512x960 / patch 16 that is 1,936 tokens instead of 16 -- a 121x
        # larger cross-attention context than the paper describes.
        # Paper §2.3: "learnable registers then compress these tokens into a small set of scene
        # tokens". One cross-attention pass, registers as queries and the visual tokens as K/V, is
        # that compression.
        scene = self.reg_compress(self.registers.expand(B, -1, -1), visual)   # [B, R, width]
        scene_ctx = self.scene_proj(scene)                           # [B, R, dec_width]
        visual_ctx = self.scene_proj(visual)                         # [B, P, dec_width]
        # ⭐ AND THE TWO DECODERS NOW READ DIFFERENT TENSORS, which is the paper's asymmetry:
        # the trajectory decoder cross-attends to the SCENE tokens, the scoring decoder attends to
        # the VISUAL tokens. Feeding one tensor to both erased that distinction entirely.
        g = goal.unsqueeze(-1) * self.goal_freqs                     # [B, G, F]
        goal_feat = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        ego_tok = self.ego_enc(torch.cat([ego, goal_feat], -1)).unsqueeze(1)
        q = self.queries.expand(B, -1, -1) + ego_tok                 # ego token added to queries
        for blk in self.dec:
            q = blk(q, scene_ctx)
        traj = self.traj_head(q).view(
            B, self.cfg.n_proposals, self.cfg.horizon_steps, self.cfg.traj_dim)
        # the candidate TRAJECTORY becomes the score query, detached so the score loss cannot
        # backpropagate into trajectory generation (the paper requires both properties)
        s = self.score_q_mlp((traj.detach() if detach_scorer else traj).flatten(2))
        # ⚠️ THE SECOND DETACH IS OURS, NOT THE PAPER'S -- see `REFeConfig.detach_scorer_context`.
        # The paper detaches the trajectories (the line above). Detaching the visual context as
        # well stops the PDM loss from adapting the encoder at all, which is a choice about the
        # very variable REFe is built to study, so it is a config flag rather than a constant.
        sctx = visual_ctx.detach() if self.cfg.detach_scorer_context else visual_ctx
        for blk in self.score_dec:
            s = blk(s, sctx)
        return traj, self.score_head(s)


def wta_loss(traj, target, yaw_w: float = 0.1):
    """Winner-takes-all L1 over POINTS, with yaw as its own weighted term.

    ⛔ THE OLD VERSION MIXED RADIANS INTO A METRE MEAN. It computed
    `(traj - target).abs().mean(dim=(2, 3))`, averaging over the point axis AND the three channels,
    so 1 radian cost exactly what 1 metre cost. The paper says "the L1 distance averaged over
    POINTS".
    ⭐ MEASURED counterexample, which is why this is a correctness fix and not a style one: against
    a zero target, a proposal with 0.00 m position error and 0.60 rad (34.4°) yaw error scored
    0.2000, while one with 0.30 m position error and zero yaw error scored 0.1000 -- the old metric
    picked the SECOND. A points-only distance scores the first 0.0000 and picks it. THE WINNER
    FLIPS.
    ⚠ On today's bank the practical effect is small: per-point mean |x| 13.566 m, |y| 0.847 m,
    |yaw| 0.0846 rad, so the yaw channel carried ~0.6 % of the pooled cost. The effect on NEAR-TIED
    proposals is unverified and needs a trained checkpoint.
    ⚠ Mean-vs-sum over the coordinate axis is a constant factor and cannot change the winner; only
    the yaw channel could. Selection therefore uses position only, and yaw is trained through a
    separate term on the SELECTED proposal so it is still learned without distorting the choice.
    """
    pos = (traj[..., :2] - target[..., :2].unsqueeze(1)).abs().sum(-1).mean(-1)       # [B, M]
    idx = pos.argmin(dim=1)
    ar = torch.arange(pos.shape[0], device=pos.device)
    l_pos = pos[ar, idx].mean()
    if traj.shape[-1] > 2:
        dy = traj[ar, idx, :, 2] - target[..., 2]
        dy = (dy + math.pi) % (2 * math.pi) - math.pi        # wrap; a 359-degree error is 1 degree
        l_yaw = dy.abs().mean()
    else:
        l_yaw = torch.zeros((), device=pos.device)
    return l_pos + yaw_w * l_yaw, idx


def param_report(model: nn.Module) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        if name.startswith("backbone") and (".A" in name or ".B" in name):
            key = "backbone.lora"
        elif name.startswith("backbone"):
            key = "backbone.frozen"
        else:
            key = name.split(".")[0]
        g = groups.setdefault(key, [0, 0])
        g[0] += p.numel()
        g[1] += p.numel() if p.requires_grad else 0
    return {"total": tot, "trainable": tr, "pct": 100.0 * tr / tot, "groups": groups}
