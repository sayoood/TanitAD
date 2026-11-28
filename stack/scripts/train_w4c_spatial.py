"""W4c — SPATIAL cross-attention scoring port (REF-C conf-pass style) on the
FROZEN W4 unicycle fan (v5.8f).

WHY (PREREG_W4C_SPATIAL_SCORING.md, registered 2026-08-10 ~19:25Z BEFORE any
run; ACTIVATED because BOTH W4b variants failed G1 — feat held-out selected ADE
**0.5600**, kin **0.5637**, gate <= 0.45 — while their TRAIN monitors sat at
0.21–0.33: the pooled offset-query feature lets a rescorer MEMORISE
train-window selection instead of learning a generalising rule): REF-C
concentrates selection mass on ~4–5 clean candidates (entropy 0.97) via a conf
pass whose queries cross-attend the SPATIAL conv map; v5f's final sel_score
(refined + factorised grafts off a flat rank≈16 state + vt gating) smears mass
over ~12 (entropy 2.22). W4c ports the REF-C mechanism: score each emitted
unicycle candidate by letting a per-candidate query ATTEND SPACE. The spatial
grounding — not extra priors — is the hypothesis under test, so there are NO
factorised grafts and NO vt gating on the output.

⭐ THE SPATIAL SURFACE (resolved from code, cited):
  * REF-C base: ``_decode`` builds per-anchor queries (refc.py:1188-1199) and
    every ``CrossAttnLayer`` (refc.py:1019-1036) cross-attends
    ``kv = self.feat_proj(fmap.flatten(2).transpose(1, 2))`` (refc.py:1307) —
    the trunk's spatial conv map as tokens, under FiLM. That is the surface
    whose selection entropy is 0.97.
  * v5f: ``WorldModel.encode_window`` (fourbrain.py:474-478) calls ``encode``
    (fourbrain.py:470-472) which POOLS the ViT token grid through
    ``SpatialGridReadout`` — only the pooled state ``[B, W, S]`` is exposed to
    the head. The pre-pooling token grid exists at ``ViTEncoder.forward``'s
    return (encoder.py:161-172, the post-``self.norm(t)`` tensor
    ``[B*W, N_tok, D]``). ``encode_tokens`` (fourbrain.py:466-468) would
    recompute it at full encoder cost, so W4c instead registers a FORWARD HOOK
    on ``world.encoder`` (:class:`SpatialTokenTap`) and captures the SAME
    tensor ``encode_window`` computes — identical numerics, zero extra
    compute. The LAST frame of the window is taken, matching REF-C's
    "cross-attends the LAST frame's feature map only" convention
    (refc.py:1165-1167).
  * EXACT TAP + SHAPE for the flagship v5f run (256x640 cyl, subframe
    176x624, patch 16, d_model 768 — config.py:367, encoder.py:120-122):
    ``[B, N_tok=11*39=429, D=768]`` float32, detached (frozen trunk).
    ``frozen_forward`` (train_v58f_unicycle_head.py:341-364) invokes the
    encoder EXACTLY once (``_goal_inputs``/``_imagination_inputs`` touch only
    the predictor — train_flagship_v4.py:220-265), and the tap ENFORCES
    n_calls == 1 so any future second encoder pass fails loudly instead of
    silently scoring the wrong tokens.

QUERIES: per-candidate embeddings of the EMITTED unicycle candidates —
:func:`candidate_query_features` = (a/A_MAX flattened K, kappa/KAPPA_MAX
flattened K, endpoint xy / ENDPOINT_SCALE, max|kappa|/KAPPA_MAX, mean
a/A_MAX) -> small MLP with ``--dropout`` (default 0.1) on the embedding (the
regularisation lever against the MEASURED W4b memorisation). The W4b input
surface (the pooled offset-head query q) is deliberately NOT an input — q
still feeds the FROZEN W4 emission that produces the fan (imported machinery,
not duplicated), but the SCORER reads only candidate geometry + space.

HEAD (:class:`W4cSpatialScorer`): ONE cross-attention block — queries
``[B, 256, d]`` attend keys/values = projected spatial tokens — mirroring
``CrossAttnLayer`` (refc.py:1019-1036) minus the FiLM condition (prereg: the
spatial grounding ALONE is under test), then linear -> logits ``[B, 256]``.
~1-2 M params, ASSERTED in band at construction (W4C_PARAM_BAND).

LOSS: the IDENTICAL margin rank loss — ``tanitad.models.tactical.ranking_loss``
(tactical.py:334-358, imported not re-derived) at the unicycle fan's GT-nearest
winner. 2000 steps default, same optimizer/save discipline as W4b.

ANTI-MEMORISATION REPORTING (the prereg's explicit demand): every save window
runs a cheap 64-window HELD-OUT probe (:func:`heldout_probe`) and logs the
TRAIN-vs-HELDOUT selected-ADE gap; the full history + final gap land in
``w4c_gate.json`` (W4b's failure signature was train 0.21–0.33 vs held-out
0.56 — the gap IS the diagnostic).

⛔ PRE-REGISTERED GATES (PREREG_W4C_SPATIAL_SCORING.md, verbatim — written to
``<out>/w4c_gate.json``):
  * **G1-c (port works):** held-out selected ADE <= 0.45 on the 881 grid
    (same as W4b's G1).
  * **G-mode (mechanism check, secondary):** selection entropy on held-out
    windows <= 1.5 (toward REF-C's 0.97 from v5f's 2.22) — passes only WITH
    G1-c; entropy alone proves nothing.
  * **G-null:** selected ADE > 0.45 => per-candidate scoring on this trunk's
    features does not generalise regardless of input surface; selection moves
    ENTIRELY to W7 WM-roll re-rank (already primary per W4b's G2), and the
    fast selector is retired to a W7-distillation target (L4) — no third
    scoring attempt without new evidence.

MEASUREMENT CONTRACT: same 881-window grid; selected / oracle / top-{4,8,16}
oracle / sel_gap (taniteval.selgap cluster CI, best-effort in-process,
pod-side rescore from the banked ``w4c_eval_windows.pt`` otherwise);
entropy/mode stats; train-vs-held-out gap explicit; four-families adjuncts on
the selected trajectory; tier **T0** (diagnostic, never driving performance);
references feat 0.5600 / kin 0.5637 / frozen 0.7933 / oracle 0.1077.

⛔ FROZEN MEANS PROVED FROZEN (the W4/W4b contract): trunk + head + grounding
``requires_grad_(False)`` inside ``load_v4_from_ck``; the W4 emission loaded
frozen via ``train_w4b_selector.load_w4_emission`` (imported); optimiser over
the scorer's params ONLY; world+head+emission md5-checksummed before/after
(``module_md5``, imported).

⚠️ POD-SIDE ONLY for the full path: this box has no GPU, no v5f checkpoint, no
W4 checkpoint, no v2 corpus. Runnable (and run) here: ``python -m py_compile``
+ the pure-part CPU tests ``stack/tests/test_w4c.py`` (scorer shapes/params
band, query-feature handling, rank-loss winner-rises, gate-JSON all three
branches incl. the G-mode only-with-G1c logic, spatial-tap shape contract).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or trainers die with
ModuleNotFound: tanitad):

  python3 train_w4c_spatial.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --w4-ckpt /workspace/experiments/w4-unicycle-head-c/unicycle_emission.pt \
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
      --v2-cache  /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --out /workspace/experiments/w4c-spatial
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stack root too, so `import tanitad` resolves even without the pod-side
# PYTHONPATH (harmless when PYTHONPATH is set — same directory).
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from train_v58f_unicycle_head import (A_MAX, DT, KAPPA_MAX,  # noqa: E402
                                      OffsetFeatureTap, module_md5)
from train_w4b_selector import (EXPECTED_GRID_WINDOWS, TOPK,  # noqa: E402
                                fan_ade, load_w4_emission,
                                selected_family_sums, topk_oracle_per_window)
from tanitad.models.tactical import ranking_loss  # noqa: E402  (torch-only)

# ---- the pre-registered constants (PREREG_W4C_SPATIAL_SCORING.md, verbatim) --
GATE_SELECTED_ADE = 0.45     # G1-c: held-out selected ADE (m), same as W4b G1
GATE_ENTROPY = 1.5           # G-mode: held-out selection entropy (nats)
REF_W4B_FEAT = 0.5600        # W4b feat held-out selected ADE (failed G1)
REF_W4B_KIN = 0.5637         # W4b kin held-out selected ADE (failed G1)
REF_FROZEN_SELECTOR_NEW_FAN = 0.7933   # frozen v5f selector on the W4 fan
REF_W4_ORACLE = 0.1077                 # W4 fan oracle ADE (registry §1.13)
REF_ENTROPY_REFC = 0.97      # REF-C conf-pass selection entropy (prereg)
REF_ENTROPY_V5F = 2.22       # v5f sel_score selection entropy (prereg)
ENDPOINT_SCALE = 50.0        # O(1) normalisation of endpoint xy (m) — a scale,
#                              not a bound (2 s at motorway speed ~ tens of m)
W4C_PARAM_BAND = (1.0e6, 2.0e6)   # the "~1-2M params" head budget, asserted


# ============================================================================
# the spatial tap — the encoder token grid BEFORE the state pooling
# ============================================================================
class SpatialTokenTap:
    """Capture the ViT token grid ``[B*W, N_tok, D]`` at ``ViTEncoder.forward``'s
    return (encoder.py:161-172, the post-``norm`` tensor) via a forward hook on
    ``world.encoder`` — the pre-pooling surface ``encode_window``
    (fourbrain.py:474-478) pools away through ``SpatialGridReadout``.

    STRICT contract: :meth:`last_frame` requires EXACTLY ONE captured call
    since :meth:`clear` — ``frozen_forward`` runs the encoder once per batch
    (train_v58f_unicycle_head.py:357; goal/imagination inputs are
    predictor-only, train_flagship_v4.py:220-265). A second pass appearing
    later must fail loudly, not silently score the wrong tokens.
    """

    def __init__(self, encoder: nn.Module):
        self._buf: list[Tensor] = []
        self._h = encoder.register_forward_hook(
            lambda _m, _args, output: self._buf.append(output))

    def clear(self) -> None:
        self._buf.clear()

    def n_calls(self) -> int:
        return len(self._buf)

    def last_frame(self, b: int, w: int) -> Tensor:
        """``(B, W) -> [B, N_tok, D]`` — the LAST frame's token grid (REF-C's
        last-frame convention, refc.py:1165-1167), detached float32 (frozen
        trunk: the scorer trains on it as a constant input)."""
        if len(self._buf) != 1:
            raise RuntimeError(
                f"SpatialTokenTap: {len(self._buf)} encoder passes captured, "
                "expected exactly 1 — was clear() called before "
                "frozen_forward, and does something now run the encoder "
                "twice?")
        t = self._buf[0]
        if t.ndim != 3 or t.shape[0] != b * w:
            raise ValueError(
                f"SpatialTokenTap: captured {tuple(t.shape)}, expected "
                f"[{b}*{w}={b * w}, N_tok, D] — not the encode_window path?")
        return t.reshape(b, w, *t.shape[1:])[:, -1].detach().float()

    def remove(self) -> None:
        self._h.remove()


# ============================================================================
# pure helpers (CPU-testable — tests/test_w4c.py)
# ============================================================================
def candidate_query_features(a_ctl: Tensor, kappa: Tensor,
                             fan: Tensor) -> Tensor:
    """Per-candidate query features of the EMITTED unicycle candidates:
    ``(a_ctl [B, N, K], kappa [B, N, K], fan [B, N, K, 2]) ->
    [B, N, 2K + 4]`` = (a/A_MAX flat, kappa/KAPPA_MAX flat,
    endpoint xy / ENDPOINT_SCALE, max|kappa|/KAPPA_MAX, mean a/A_MAX).

    All channels O(1) by the emission's own bounds (train_v58f_unicycle_head
    A_MAX/KAPPA_MAX tanh activations); endpoint by ENDPOINT_SCALE. This is
    candidate GEOMETRY only — the pooled offset-query q (W4b's memorised
    surface) is deliberately absent."""
    if a_ctl.ndim != 3 or a_ctl.shape != kappa.shape:
        raise ValueError(f"a_ctl/kappa must be matching [B, N, K], got "
                         f"{tuple(a_ctl.shape)} vs {tuple(kappa.shape)}")
    if fan.shape != a_ctl.shape + (2,):
        raise ValueError(f"fan must be [B, N, K, 2] matching controls, got "
                         f"{tuple(fan.shape)} for controls "
                         f"{tuple(a_ctl.shape)}")
    an = a_ctl / A_MAX
    kn = kappa / KAPPA_MAX
    end = fan[..., -1, :] / ENDPOINT_SCALE                   # [B, N, 2]
    kmax = kn.abs().amax(dim=-1, keepdim=True)               # [B, N, 1]
    amean = an.mean(dim=-1, keepdim=True)                    # [B, N, 1]
    return torch.cat([an, kn, end, kmax, amean], dim=-1)


def query_feat_dim(k: int) -> int:
    """Input width of :func:`candidate_query_features` for K horizons."""
    return 2 * int(k) + 4


def selection_entropy(logits: Tensor) -> Tensor:
    """Per-window selection entropy in nats: ``[B, N] -> [B]`` —
    H(softmax(logits)). Uniform over N -> ln N (~5.545 for N=256); one-hot ->
    0. The G-mode instrument (REF-C 0.97 vs v5f 2.22, prereg)."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, N], got {tuple(logits.shape)}")
    logp = torch.log_softmax(logits.float(), dim=1)
    return -(logp.exp() * logp).sum(dim=1)


# ============================================================================
# the ONLY trainable module of W4c
# ============================================================================
class W4cSpatialScorer(nn.Module):
    """REF-C conf-pass-style spatial scorer: per-candidate query embeddings
    cross-attend the trunk's spatial token grid, ONE block, linear to a logit
    per candidate. ``forward(qfeat [B, N, 2K+4], tokens [B, P, D]) ->
    logits [B, N]``.

    Structure mirrors ``CrossAttnLayer`` (refc.py:1019-1036) MINUS the FiLM
    condition — the prereg tests the SPATIAL grounding alone: no factorised
    grafts, no vt gating, no condition pathway. ``--dropout`` sits on the
    query embedding (the anti-memorisation lever; W4b memorised through its
    query surface). The FINAL logit layer is ZERO-INIT: at step 0 every
    candidate scores 0 — a defined, uninformed uniform warm start (the
    W4/W4b discipline), regardless of dropout.
    """

    def __init__(self, spatial_dim: int, k: int = 20, d: int = 256,
                 n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_dim, self.k, self.d = int(spatial_dim), int(k), int(d)
        self.in_dim = query_feat_dim(k)
        self.q_embed = nn.Sequential(
            nn.Linear(self.in_dim, d), nn.GELU(), nn.Linear(d, d),
            nn.Dropout(float(dropout)))
        self.kv_proj = nn.Linear(spatial_dim, d)
        self.kv_norm = nn.LayerNorm(d)
        self.norm_q = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))
        self.logit = nn.Linear(d, 1)
        nn.init.zeros_(self.logit.weight)        # uniform warm start
        nn.init.zeros_(self.logit.bias)

    def forward(self, qfeat: Tensor, tokens: Tensor) -> Tensor:
        if qfeat.ndim != 3 or qfeat.shape[-1] != self.in_dim:
            raise ValueError(f"qfeat must be [B, N, {self.in_dim}], got "
                             f"{tuple(qfeat.shape)}")
        if tokens.ndim != 3 or tokens.shape[-1] != self.spatial_dim:
            raise ValueError(f"tokens must be [B, P, {self.spatial_dim}], "
                             f"got {tuple(tokens.shape)}")
        if tokens.shape[0] != qfeat.shape[0]:
            raise ValueError(f"batch mismatch: qfeat {qfeat.shape[0]} vs "
                             f"tokens {tokens.shape[0]}")
        q = self.q_embed(qfeat)                              # [B, N, d]
        kv = self.kv_norm(self.kv_proj(tokens))              # [B, P, d]
        h = self.norm_q(q)
        q = q + self.cross(h, kv, kv, need_weights=False)[0]  # attend SPACE
        q = q + self.mlp(self.norm_f(q))
        return self.logit(q).squeeze(-1)                     # [B, N]


# ============================================================================
# gate JSON (pure: dict-in, dict-out; all three branches pinned by tests)
# ============================================================================
def build_w4c_gate(mini: dict, *, train_vs_heldout: dict) -> dict:
    """The pre-registered W4c gate record (PREREG_W4C_SPATIAL_SCORING.md,
    gates verbatim, all three branches + the only-with-G1c coupling of
    G-mode). Every number is a POINT ESTIMATE over the eval grid; the registry
    row carries the episode-cluster bootstrap CI (pod-side rescore from the
    banked per-window arrays)."""
    sel = mini["selected_ade"]
    ent = mini["entropy"]["mean"]
    g1c_pass = bool(sel <= GATE_SELECTED_ADE)
    ent_ok = bool(ent <= GATE_ENTROPY)
    return {
        "item": ("W4c — spatial cross-attention scoring port, REF-C "
                 "conf-pass style (PREREG_W4C_SPATIAL_SCORING.md, registered "
                 "2026-08-10 pre-launch; activated by W4b feat+kin G1 "
                 "failures)"),
        "gate_G1c_port_works": {
            "rule": (f"held-out selected ADE <= {GATE_SELECTED_ADE} on the "
                     "881 grid (same as W4b's G1)"),
            "selected_ade": sel,
            "threshold_m": GATE_SELECTED_ADE,
            "pass": g1c_pass,
        },
        "gate_Gmode_mechanism_check": {
            "rule": (f"selection entropy on held-out windows <= {GATE_ENTROPY}"
                     f" (toward REF-C's {REF_ENTROPY_REFC} from v5f's "
                     f"{REF_ENTROPY_V5F}) — passes only WITH G1-c; entropy "
                     "alone proves nothing"),
            "entropy_mean": ent,
            "threshold_nats": GATE_ENTROPY,
            "entropy_le_threshold": ent_ok,
            "pass": bool(ent_ok and g1c_pass),
            "note": ("secondary mechanism check — coupled to G1-c by the "
                     "prereg; a low entropy without G1-c is concentration on "
                     "the WRONG candidates, not the REF-C mechanism"),
        },
        "gate_Gnull": {
            "rule": (f"selected ADE > {GATE_SELECTED_ADE} => per-candidate "
                     "scoring on this trunk's features does not generalise "
                     "regardless of input surface; selection moves ENTIRELY "
                     "to W7 WM-roll re-rank (already primary per W4b's G2), "
                     "and the fast selector is retired to a W7-distillation "
                     "target (L4) — no third scoring attempt without new "
                     "evidence"),
            "engaged": not g1c_pass,
        },
        "reference": {
            "w4b_feat_selected_ade": REF_W4B_FEAT,
            "w4b_kin_selected_ade": REF_W4B_KIN,
            "frozen_selector_selected_ade_new_fan": REF_FROZEN_SELECTOR_NEW_FAN,
            "w4_oracle_ade_new_fan": REF_W4_ORACLE,
            "refc_conf_pass_entropy": REF_ENTROPY_REFC,
            "v5f_sel_score_entropy": REF_ENTROPY_V5F,
        },
        "train_vs_heldout": train_vs_heldout,
        "mini_eval": mini,
        "tier": "T0",
        "_tier_note": ("T0 teacher-forced diagnostic — conditioned on logged "
                       "frames; NEVER quotable as driving performance "
                       "(EVAL_DOCTRINE.md)"),
        "_estimator_note": ("POINT ESTIMATES over the eval grid (episodes<40, "
                            "stride 8). The decision-grade interval for any "
                            "registry claim is the EPISODE-CLUSTER BOOTSTRAP "
                            "(taniteval.selgap / taniteval/ci.py) on the "
                            "banked per-window arrays (w4c_eval_windows.pt) — "
                            "run it before publishing; never "
                            "overlapping_holdout_se."),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }


# ============================================================================
# the shared scored forward (train / probe / eval use the same path)
# ============================================================================
def scored_forward(world, head, tap, sp_tap, emission, batch, device, *,
                   probes, amp_on, w4_cond: str):
    """One frozen pass -> (out, logits-ready pieces). Returns
    ``(out, a_ctl, kappa, fan, qfeat, tokens, tgt)`` — everything detached
    float32 except nothing trainable is here at all; the caller applies the
    scorer. The sp_tap clear/capture discipline lives HERE so every consumer
    inherits the strict one-pass contract."""
    from train_v58f_unicycle_head import frozen_forward
    b_, w_ = batch["frames"].shape[:2]
    sp_tap.clear()
    out, emis_feat, v0, tgt = frozen_forward(
        world, head, tap, batch, device, probes=probes, amp_on=amp_on,
        cond=w4_cond)
    tokens = sp_tap.last_frame(b_, w_)                       # [B, P, D] f32
    with torch.no_grad():
        a_ctl, kappa, fan = emission(emis_feat, v0)
        qfeat = candidate_query_features(a_ctl.float(), kappa.float(),
                                         fan.float())
    return out, a_ctl, kappa, fan, qfeat, tokens, tgt


@torch.no_grad()
def heldout_probe(world, head, tap, sp_tap, emission, scorer, ds_val,
                  probe_idx, device, *, probes, amp_on, w4_cond: str,
                  batch: int = 16) -> float:
    """Cheap mid-training held-out monitor: selected ADE of the CURRENT scorer
    over a fixed ~64-window val subset (dropout off — .eval()). The
    train-vs-heldout gap this feeds is the prereg's explicit memorisation
    diagnostic (W4b: train 0.21–0.33 vs held-out 0.56)."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    scorer.eval()
    tot, n = 0.0, 0
    for b0 in range(0, len(probe_idx), batch):
        idx = probe_idx[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        _out, _a, _k, fan, qfeat, tokens, tgt = scored_forward(
            world, head, tap, sp_tap, emission, b, device, probes=probes,
            amp_on=amp_on, w4_cond=w4_cond)
        logits = scorer(qfeat, tokens)
        err = fan_ade(fan.float(), tgt)
        sel = logits.argmax(dim=1)
        tot += float(err[torch.arange(err.shape[0], device=err.device),
                         sel].sum())
        n += err.shape[0]
    scorer.train()
    return tot / max(n, 1)


def probe_indices(grid: list[int], n: int) -> list[int]:
    """Evenly-spaced ~n-window subset of the eval grid (deterministic; spans
    episodes rather than clustering in the first one)."""
    if n <= 0 or not grid:
        return []
    step = max(1, len(grid) // n)
    return grid[::step][:n]


# ============================================================================
# end-of-run mini-eval — SAME grid rule as W4b (episodes<40, stride 8 -> 881)
# ============================================================================
@torch.no_grad()
def mini_eval(world, head, tap, sp_tap, emission, scorer, ds_val, device, *,
              probes, amp_on, w4_cond: str, episodes: int = 40,
              stride: int = 8, batch: int = 16,
              out_dir: str | None = None) -> dict:
    """selected / oracle / top-{4,8,16} oracle / sel_gap / entropy+mode stats
    of the SPATIAL logits on the W4 unicycle fan over the eval-default grid —
    grid-comparable with the banked W4b numbers. Per-window arrays banked to
    ``w4c_eval_windows.pt`` for the pod-side episode-cluster bootstrap."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    scorer.eval()
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < episodes and t % stride == 0]
    if not grid:
        raise SystemExit("[w4c] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    errs, logits_all, eids = [], [], []
    fam = {k: 0.0 for k in ("speed_mae", "accel_mae", "heading_mae_rad",
                            "yaw_rate_mae_rads", "curvature_mae_1pm")}
    sums = {"selected": 0.0, "oracle": 0.0, "frozen_selected": 0.0,
            "winner_hit": 0.0, "rank_pct": 0.0}
    sums.update({f"top{k}": 0.0 for k in TOPK})
    n = 0
    t0 = time.time()
    for b0 in range(0, len(grid), batch):
        idx = grid[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        out, _a, _k, fan, qfeat, tokens, tgt = scored_forward(
            world, head, tap, sp_tap, emission, b, device, probes=probes,
            amp_on=amp_on, w4_cond=w4_cond)
        logits = scorer(qfeat, tokens)                       # [B, N]
        err = fan_ade(fan.float(), tgt)                      # [B, N]
        bs = err.shape[0]
        ar = torch.arange(bs, device=err.device)
        sel = logits.argmax(dim=1)
        win = err.argmin(dim=1)
        e_sel = err[ar, sel]
        sums["selected"] += float(e_sel.sum())
        sums["oracle"] += float(err.min(dim=1).values.sum())
        sums["frozen_selected"] += float(err[ar, out["sel_idx"]].sum())
        sums["winner_hit"] += float((sel == win).float().sum())
        sums["rank_pct"] += float(
            ((err < e_sel[:, None]).sum(dim=1).float()
             / max(err.shape[1] - 1, 1)).sum())
        for k in TOPK:
            sums[f"top{k}"] += float(
                topk_oracle_per_window(err, logits, k).sum())
        for key, v in selected_family_sums(fan[ar, sel], tgt).items():
            fam[key] += v
        errs.append(err.cpu())
        logits_all.append(logits.float().cpu())
        eids.extend(int(ds_val.index[i][0]) for i in idx)
        n += bs
    scorer.train()
    err_all = torch.cat(errs)                                # [Nw, N]
    log_all = torch.cat(logits_all)
    sel_all = log_all.argmax(dim=1)
    ent_all = selection_entropy(log_all)                     # [Nw]
    p = torch.softmax(log_all, dim=1)
    if out_dir is not None:
        torch.save({"fan_err_ade": err_all, "logits": log_all,
                    "sel_idx": sel_all, "entropy": ent_all,
                    "eid": torch.tensor(eids),
                    "_read": ("per-window per-candidate dense ADE of the W4 "
                              "unicycle fan + W4c spatial-scorer logits + "
                              "selection entropies over the eval grid — the "
                              "input to the pod-side episode-cluster "
                              "bootstrap (taniteval.selgap)")},
                   os.path.join(out_dir, "w4c_eval_windows.pt"))
    # best-effort selgap CI — stack does not DEPEND on taniteval (its rule).
    selgap_ci: dict | str
    try:
        root = Path(__file__).resolve().parents[2] / "taniteval"
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from taniteval.selgap import selgap as _selgap
        selgap_ci = _selgap(err_all.numpy(), sel_all.numpy(), eids,
                            scores=log_all.numpy(),
                            level="operative_w4c_spatial_scorer")
    except Exception as ex:                                  # noqa: BLE001
        selgap_ci = (f"taniteval unavailable here ({type(ex).__name__}: {ex})"
                     " — rescore pod-side from w4c_eval_windows.pt")
    res = {
        "n_windows": n,
        "n_candidates": int(err_all.shape[1]),
        "grid": {"episodes": episodes, "stride": stride, "batch": batch,
                 "expected_n": EXPECTED_GRID_WINDOWS,
                 "matches_banked_grid": n == EXPECTED_GRID_WINDOWS},
        "selected_ade": round(sums["selected"] / n, 6),
        "oracle_ade": round(sums["oracle"] / n, 6),
        "sel_gap": round((sums["selected"] - sums["oracle"]) / n, 6),
        "oracle_topk": {str(k): round(sums[f"top{k}"] / n, 6) for k in TOPK},
        "frozen_selected_ade": round(sums["frozen_selected"] / n, 6),
        "winner_hit_frac": round(sums["winner_hit"] / n, 6),
        "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
        "entropy": {
            "mean": round(float(ent_all.mean()), 6),
            "median": round(float(ent_all.median()), 6),
            "p90": round(float(ent_all.quantile(0.9)), 6),
            "n_eff_mean": round(float(ent_all.exp().mean()), 6),
            "top1_prob_mean": round(float(p.max(dim=1).values.mean()), 6),
            "note": ("softmax over the 256 spatial logits, nats; REF-C "
                     f"conf pass {REF_ENTROPY_REFC}, v5f sel_score "
                     f"{REF_ENTROPY_V5F} (prereg reference points)"),
        },
        "families": {
            "LONGITUDINAL": {
                "speed_mae_ms": round(fam["speed_mae"] / n, 6),
                "accel_mae_ms2": round(fam["accel_mae"] / n, 6),
                "headway_ttc": ("not computable here: no lead-agent channel "
                                "in this instrument — pod-side taniteval "
                                "harness item, stated per the 2026-08-02 "
                                "rule"),
            },
            "LATERAL": {
                "heading_mae_rad": round(fam["heading_mae_rad"] / n, 6),
                "yaw_rate_mae_rads": round(fam["yaw_rate_mae_rads"] / n, 6),
                "curvature_mae_1pm": round(fam["curvature_mae_1pm"] / n, 6),
                "note": "waypoint-derived adjuncts (atan2 of finite "
                        "differences; noisy near standstill)",
            },
            "TACTICAL": {
                "note": "selector rank quality IS this instrument's family",
                "winner_hit_frac": round(sums["winner_hit"] / n, 6),
                "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
            },
            "STRATEGIC": ("n/a: no route/goal label exists on PhysicalAI-AV "
                          "(settled, five probes — CLAUDE.md rule 2); stated "
                          "per the 2026-08-02 rule"),
        },
        "selgap_ci": selgap_ci,
        "wallclock_s": round(time.time() - t0, 1),
    }
    return res


# ============================================================================
# main (POD-SIDE: needs GPU + the v5f checkpoint + the W4 emission + corpora)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_w4c_spatial", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v5f checkpoint "
                    "(keys: model, grounding, head[, goal_head])")
    ap.add_argument("--w4-ckpt", required=True,
                    help="trained W4 unicycle_emission.pt (the W4 trainer's "
                         "save format) — loaded FROZEN")
    ap.add_argument("--head-config", default=None,
                    help="run config.json (default: sibling of --ckpt)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer (pass explicitly)")
    ap.add_argument("--probe-vocab", default=None,
                    help="probe_vocab.pt for cond_imagination heads "
                         "(default: sibling of --ckpt)")
    # corpus (v2 compressed only — the ONLY format v5 has); same seams as W4b
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000,
                    help="prereg budget: ~2000 steps, <= 2 h pod5")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4,
                    help="episode-grouped sampling (the MooseFS I/O shape)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint + metrics.json + HELD-OUT PROBE cadence")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--margin", type=float, default=0.1,
                    help="ranking margin (tanitad.models.tactical default)")
    # W4c head knobs
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout on the query embedding — the "
                         "anti-memorisation lever the prereg motivates")
    ap.add_argument("--d-model", type=int, default=256,
                    help="scorer width d (queries + projected kv)")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--heldout-probe-n", type=int, default=64,
                    help="windows in the mid-training held-out probe (the "
                         "train-vs-heldout gap instrument)")
    # mini-eval grid (eval defaults — the 881 grid)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[w4c] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v4_from_ck, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import build_train_episodes, make_sampler

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(
        a, cfg, label="train_w4c_spatial")
    plan = _plan(cfg)
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[w4c] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs W4c train frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[w4c] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    # ---- frozen v5f: EXACTLY the W4/W4b loader ------------------------------
    print(f"[w4c] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "head" in ck):
        raise SystemExit("[w4c] --ckpt has no 'head' key — W4c rescores the "
                         "v4 planner head's fan; a plain trunk has no fan.")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=a.anchors_dense, frame=model_frame)
    del ck
    horizons = head.cfg.horizons
    if tuple(horizons) != tuple(range(1, len(horizons) + 1)):
        raise SystemExit(f"[w4c] head horizons {horizons} are not contiguous "
                         f"1..K @10 Hz — the W4 fan is defined on the dense "
                         f"tick only.")
    K = len(horizons)

    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        pv = Path(a.probe_vocab or (Path(a.ckpt).parent / "probe_vocab.pt"))
        if not pv.exists():
            raise SystemExit(f"[w4c] cond_imagination head but no {pv} — a "
                             "silent skip would score a head minus 32 inputs")
        probes = torch.load(pv, map_location=device)
        print(f"[w4c] imagination probes: {tuple(probes.shape)}", flush=True)

    # ---- the frozen W4 emission (the fan under rescoring) -------------------
    feat_dim_q = int(head.decoder.offset_head.in_features)
    emission, w4_cond, w4_meta = load_w4_emission(
        a.w4_ckpt, device, k_expected=K, offset_in_features=feat_dim_q)
    print(f"[w4c] W4 emission loaded: cond={w4_cond} "
          f"feat_dim={w4_meta['feat_dim']} K={K} (step {w4_meta['w4_step']}, "
          f"base {w4_meta['w4_base_step']}) — FROZEN", flush=True)

    # ---- frozen means PROVED frozen (three locks, W4 contract) --------------
    assert not any(p.requires_grad
                   for m in (world, grounding, head, emission)
                   for p in m.parameters())
    md5_before = module_md5(world, head, emission)
    print(f"[w4c] trunk+head+emission frozen · base step {base_step} · "
          f"md5 {md5_before[:12]}", flush=True)

    # ---- data (same grid as W4b/eval) ---------------------------------------
    train_eps, train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_train = FlagshipV4Dataset(train_eps, window=cfg.predictor.window,
                                 max_horizon=plan.max_horizon,
                                 maneuver_h=plan.maneuver_h,
                                 channels=cfg.encoder.in_channels)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[w4c] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)
    sample = make_sampler(ds_train, a.eps_per_batch, rng)
    # the fixed held-out probe subset (same rule as the mini-eval grid)
    val_grid = [i for i, (e, t) in enumerate(ds_val.index)
                if e < a.episodes and t % a.stride == 0]
    probe_idx = probe_indices(val_grid, a.heldout_probe_n)
    print(f"[w4c] held-out probe: {len(probe_idx)} windows every "
          f"{a.save_every} steps (train-vs-heldout gap instrument)",
          flush=True)

    # ---- the taps + the ONLY trainable module -------------------------------
    tap = OffsetFeatureTap(head.decoder.offset_head)   # feeds the W4 emission
    sp_tap = SpatialTokenTap(world.encoder)            # the SPATIAL surface
    spatial_dim = int(world.encoder.cfg.d_model)
    n_tok = int(world.encoder.n_tokens)
    print(f"[w4c] spatial tap: world.encoder post-norm token grid "
          f"(encoder.py:161-172) — last frame [B, {n_tok}, {spatial_dim}] "
          f"(grid {world.encoder.grid_h}x{world.encoder.grid_w})", flush=True)
    scorer = W4cSpatialScorer(spatial_dim=spatial_dim, k=K, d=a.d_model,
                              n_heads=a.n_heads, ff_mult=a.ff_mult,
                              dropout=a.dropout).to(device)
    n_par = sum(p.numel() for p in scorer.parameters())
    lo, hi = W4C_PARAM_BAND
    if not (lo <= n_par <= hi):
        raise SystemExit(f"[w4c] scorer has {n_par} params, outside the "
                         f"prereg ~1-2M band [{int(lo)}, {int(hi)}] — check "
                         f"--d-model/--ff-mult")
    print(f"[w4c] W4cSpatialScorer d={a.d_model} heads={a.n_heads} "
          f"ff={a.ff_mult} dropout={a.dropout} in_dim={scorer.in_dim} "
          f"K={K} ({n_par / 1e6:.3f} M trainable, in band; frozen everything "
          f"else)", flush=True)
    opt = torch.optim.AdamW(scorer.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "w4c-spatial-scorer", "args": vars(a),
        "base_ckpt": a.ckpt, "base_step": base_step,
        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta,
        "trunk_head_emission_md5": md5_before, "n_trainable": n_par,
        "spatial_tap": {"module": "world.encoder (ViTEncoder post-norm, "
                                   "encoder.py:161-172; hook, not recompute)",
                        "n_tokens": n_tok, "d": spatial_dim,
                        "frame_rule": "last frame of the window "
                                      "(refc.py:1165-1167 convention)"},
        "in_dim": scorer.in_dim, "horizons_K": K,
        "train_parity": {"n_dirs": len(a.v2_cache)},
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    gap_history: list[dict] = []
    acc = {"loss": 0.0, "selected_ade": 0.0, "oracle_ade": 0.0,
           "winner_hit": 0.0, "n": 0}
    t0 = time.time()
    last_row: dict = {}
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        out, a_ctl, kappa, fan, qfeat, tokens, tgt = scored_forward(
            world, head, tap, sp_tap, emission, b, device, probes=probes,
            amp_on=amp_on, w4_cond=w4_cond)
        with torch.no_grad():
            err = fan_ade(fan.float(), tgt)                  # [B, N] targets
        # scorer in float32 OUTSIDE autocast — same numerics the gate is
        # evaluated at (the W4/W4b discipline).
        logits = scorer(qfeat, tokens)
        loss = ranking_loss(logits, err, a.margin)           # THE tactical loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        opt.step()
        sched.step()

        bs = err.shape[0]
        ar = torch.arange(bs, device=err.device)
        with torch.no_grad():
            sel = logits.argmax(dim=1)
            acc["loss"] += float(loss.detach()) * bs
            acc["selected_ade"] += float(err[ar, sel].sum())
            acc["oracle_ade"] += float(err.min(dim=1).values.sum())
            acc["winner_hit"] += float((sel == err.argmin(dim=1)).float()
                                       .sum())
            acc["n"] += bs

        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(float(loss.detach()), 5),
                   "selected_ade": round(float(err[ar, sel].mean()), 4),
                   "oracle_ade": round(float(err.min(dim=1).values.mean()),
                                       4),
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.save_every == 0:
            n_ = max(acc["n"], 1)
            train_monitor = acc["selected_ade"] / n_
            ho = heldout_probe(world, head, tap, sp_tap, emission, scorer,
                               ds_val, probe_idx, device, probes=probes,
                               amp_on=amp_on, w4_cond=w4_cond,
                               batch=a.eval_batch)
            gap_rec = {"step": step,
                       "train_selected_ade": round(train_monitor, 5),
                       "heldout_selected_ade": round(ho, 5),
                       "gap": round(ho - train_monitor, 5)}
            gap_history.append(gap_rec)
            row = {"step": step,
                   **{k: round(v / n_, 5) for k, v in acc.items()
                      if k != "n"},
                   "heldout_selected_ade": round(ho, 5),
                   "train_heldout_gap": gap_rec["gap"],
                   "elapsed_s": round(time.time() - t0, 1)}
            last_row = row
            history.append(row)
            acc = {k: 0.0 for k in acc} | {"n": 0}
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "gap_history": gap_history,
                           "args": vars(a),
                           "base_step": base_step, "w4_meta": w4_meta,
                           "_read": "rows are TRAIN-batch running means over "
                                    "the last save window PLUS the held-out "
                                    "probe (the train-vs-heldout gap is the "
                                    "memorisation diagnostic); the gate "
                                    "numbers are the held-out mini-eval in "
                                    "w4c_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf,
                          indent=1)
            torch.save({"scorer": scorer.state_dict(),
                        "spatial_dim": spatial_dim, "d_model": a.d_model,
                        "n_heads": a.n_heads, "ff_mult": a.ff_mult,
                        "dropout": a.dropout, "in_dim": scorer.in_dim,
                        "k": K, "step": step, "args": vars(a),
                        "base_ckpt": a.ckpt, "base_step": base_step,
                        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta},
                       os.path.join(a.out, "w4c_scorer.pt"))
            fh.write(json.dumps({"per500": row}) + "\n")
            fh.flush()
            print(f"[w4c @{step}] {row}", flush=True)

    # ---- frozen proof + the pre-registered gates ----------------------------
    md5_after = module_md5(world, head, emission)
    ev = mini_eval(world, head, tap, sp_tap, emission, scorer, ds_val, device,
                   probes=probes, amp_on=amp_on, w4_cond=w4_cond,
                   episodes=a.episodes, stride=a.stride, batch=a.eval_batch,
                   out_dir=a.out)
    if not ev["grid"]["matches_banked_grid"]:
        print(f"[w4c] ⚠ grid has {ev['n_windows']} windows, banked references "
              f"are on {EXPECTED_GRID_WINDOWS} — comparisons to "
              f"{REF_W4B_FEAT}/{REF_W4B_KIN}/{REF_FROZEN_SELECTOR_NEW_FAN}/"
              f"{REF_W4_ORACLE} are cross-grid; say so wherever quoted",
              flush=True)
    tvh = {
        "final_train_selected_ade": (last_row.get("selected_ade")
                                     if last_row else None),
        "final_heldout_probe_selected_ade": (
            last_row.get("heldout_selected_ade") if last_row else None),
        "final_gap": (last_row.get("train_heldout_gap")
                      if last_row else None),
        "history": gap_history,
        "probe_n_windows": len(probe_idx),
        "note": ("the memorisation diagnostic the prereg demands explicitly — "
                 "W4b failed with train monitor 0.21–0.33 vs held-out 0.56"),
    }
    gate = build_w4c_gate(ev, train_vs_heldout=tvh)
    gate.update({
        "steps": a.steps, "base_ckpt": a.ckpt, "base_step": base_step,
        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta,
        "n_trainable": n_par, "param_band": [int(lo), int(hi)],
        "in_dim": scorer.in_dim, "margin": a.margin, "dropout": a.dropout,
        "scorer_cfg": {"d_model": a.d_model, "n_heads": a.n_heads,
                       "ff_mult": a.ff_mult},
        "spatial_tap": {"module": "world.encoder (ViTEncoder post-norm, "
                                   "encoder.py:161-172; hook, not recompute)",
                        "n_tokens": n_tok, "d": spatial_dim,
                        "frame_rule": "last frame of the window "
                                      "(refc.py:1165-1167 convention)"},
        "frozen_proof": {"md5_before": md5_before, "md5_after": md5_after,
                         "identical": md5_before == md5_after,
                         "modules": "world + head + W4 emission"},
        "wall_s": round(time.time() - t0, 1),
    })
    with open(os.path.join(a.out, "w4c_gate.json"), "w") as gf:
        json.dump(gate, gf, indent=1)
    fh.write(json.dumps({"summary": gate}) + "\n")
    fh.close()
    tap.remove()
    sp_tap.remove()
    print(f"\n[W4C SUMMARY] {json.dumps(gate, indent=1)}", flush=True)
    if not gate["frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK/HEAD/EMISSION CHANGED DURING TRAINING — "
                         "run invalid")
    g1 = gate["gate_G1c_port_works"]
    gm = gate["gate_Gmode_mechanism_check"]
    gn = gate["gate_Gnull"]
    print(f"[W4C GATE G1-c] {'PASS' if g1['pass'] else 'FAIL'} "
          f"(selected {ev['selected_ade']:.4f} vs {GATE_SELECTED_ADE}; "
          f"W4b refs feat {REF_W4B_FEAT} / kin {REF_W4B_KIN}, frozen "
          f"{REF_FROZEN_SELECTOR_NEW_FAN}, oracle {ev['oracle_ade']:.4f} vs "
          f"ref {REF_W4_ORACLE})", flush=True)
    print(f"[W4C GATE G-mode] {'PASS' if gm['pass'] else 'FAIL'} "
          f"(entropy {ev['entropy']['mean']:.3f} vs {GATE_ENTROPY}; "
          f"only-with-G1c; refs REF-C {REF_ENTROPY_REFC} / v5f "
          f"{REF_ENTROPY_V5F})", flush=True)
    print(f"[W4C GATE G-null] engaged={gn['engaged']}"
          + (" — selection moves ENTIRELY to W7 WM-roll re-rank; fast "
             "selector retired to a W7-distillation target (L4)"
             if gn["engaged"] else ""), flush=True)
    print(f"[W4C train-vs-heldout] final gap {tvh['final_gap']} "
          f"(train {tvh['final_train_selected_ade']} vs heldout "
          f"{tvh['final_heldout_probe_selected_ade']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
