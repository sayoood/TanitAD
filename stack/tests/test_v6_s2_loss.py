"""S2 — strategic goal supervision in `train_v6_staged.py`: default-off,
head-only by MEASURED gradient reach, label loader with the stable-id join,
and every guard proven able to fail.

⛔ WHAT THIS PROTECTS. v6F is resuming from checkpoints of the incumbent
architecture, and the incumbent LOSS: the default (``w_s2_goal = 0``) must be
bit-identical — proved against a CONTENT-anchored pre-change revision of the
trainer (C75: never HEAD — a sibling's whole-index commit can sweep an
in-progress file into HEAD, after which a HEAD comparison is a module compared
with itself). The MODEL is untouched by this change (no v6.py edit); its own
byte-identity battery (test_v6_gstr_port.py / test_v6_factored_goal.py: default
87,893,449/405, config E 336,542,025/573, per-tensor vs content anchors) runs
in this same suite and keeps carrying that half.

⛔ THE BINDING RULE UNDER TEST (HIERARCHY_VOCABULARY §2, the diagram): labels
supervise GOAL/INTERPRETATION HEADS only, NEVER any WM trunk loss. Proved the
measured way (`test_v6_gstr_port.test_t1_gradient_reaches_the_port_...`
pattern): autograd from the S2 term alone reaches EXACTLY
``goal_head_str.* + act_head_str.*`` (16 tensors) — zero encoder/readout/
predictor/adapters, and the shared vocabulary tables are NOT touched ("+ vocab
embeddings if applicable" resolves to NOT APPLICABLE, measured: the heads'
logits/args come from their own trunk/type_head/arg_head; ``vocab.encode``
sits only on the downstream conditioning path this loss never reads). The
reach detector itself carries a negative control that shows it CAN see vocab
parameters when a path exists.

Labels: `s2-strategic-v1` (S2_STRATEGIC_GAP.md §1.2; built 2026-08-16, 797
records). The join is ``stable_episode_id`` ONLY — the legacy 16-bit id
collides for 69/2400 train + 7/600 val clips and is REFUSED.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.v6 import (  # noqa: E402
    GOAL_ARG_SLOTS, STAGES, STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS,
    V6Config, V6Stack, apply_stage_freeze)
import s2_labels as SL  # noqa: E402
from s2_labels import (  # noqa: E402
    IGNORE_ID, S2LabelError, load_s2_labels, stable_episode_id)
from train_v6_staged import (  # noqa: E402
    V6LossWeights, build_parser, dry_run, preflight, s2_goal_loss,
    synthetic_s2_batch, synthetic_train_batch, v6_loss_step)

#: the S2 term's whole reachable set — 2 heads x (trunk.0, trunk.2, type_head,
#: arg_head) x (weight, bias). MEASURED 2026-08-16 on the tiny stack below;
#: the test asserts the LIVE set EQUALS this, so a widened head or a new leak
#: both fail by name.
S2_REACH = {
    f"{h}.{m}.{p}"
    for h in ("goal_head_str", "act_head_str")
    for m in ("trunk.0", "trunk.2", "type_head", "arg_head")
    for p in ("weight", "bias")
}
ROUTE_TO_ID = STRATEGIC_GOAL_TOKENS.index("ROUTE_TO")


def tiny_cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def _stack(**kw) -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(tiny_cfg(**kw))


def _loss_batch(stack, *, batch=4, seed=1, s2_seed=2, with_s2=True,
                k_wp=10):
    b = synthetic_train_batch(stack, batch=batch, k=12, seed=seed)
    b["gt_wp"] = torch.randn(batch, k_wp, 2,
                             generator=torch.Generator().manual_seed(seed))
    if with_s2:
        b |= synthetic_s2_batch(batch, seed=s2_seed)
    return b


# =========================================================================== #
# 1. ⛔ DEFAULT OFF — the incumbent loss is bit-identical (content anchor)
# =========================================================================== #

def test_the_weight_defaults_off_and_for_stage_gates_it():
    """0.0 everywhere by default; in force ONLY where layer_str trains."""
    assert V6LossWeights().w_s2_goal == 0.0
    on = V6LossWeights(w_s2_goal=1.0)
    assert on.for_stage("S-W").w_s2_goal == 0.0
    assert on.for_stage("S-T").w_s2_goal == 0.0
    assert on.for_stage("S-S").w_s2_goal == 1.0
    assert on.for_stage("S-J").w_s2_goal == 1.0


def test_default_loss_emits_no_s2_key_and_no_s2_term():
    s = _stack()
    L = v6_loss_step(s, _loss_batch(s, with_s2=True), stage="S-S",
                     weights=V6LossWeights(), o1_k=10, o5_k=12)
    assert "s2" not in L and "s2" not in L["log"]["terms"]
    assert not any(k.startswith("s2_") for k in L["log"])


def _side_by_side(rel: str, marker: str, modname: str):
    """Newest committed revision of ``rel`` WITHOUT ``marker`` — the
    `test_loss_determinism._side_by_side` C75 pattern, self-contained."""
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", rel],
                             cwd=_ROOT, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=_ROOT,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if marker.encode() in r.stdout:
                continue
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / f"{modname}.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location(modname, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


@pytest.mark.parametrize("stage", STAGES)
def test_default_loss_is_bit_identical_to_the_PRE_S2_trainer(stage):
    """⛔ THE ONE THAT PROTECTS THE INCUMBENT LOSS. Old and new trainer over
    the SAME current model, default weights, same seeds: every term, every
    log key and the global RNG stream must match — a default path that drew
    one extra number would desynchronise everything after it."""
    old = _side_by_side("stack/scripts/train_v6_staged.py", "w_s2_goal",
                        "train_v6_staged_pre_s2")
    if old is None:
        pytest.skip("git could not produce a pre-S2 trainer revision")
    s = _stack()
    s.eval()
    b = _loss_batch(s, with_s2=False, k_wp=2)
    kw = dict(stage=stage, o1_k=2, o5_k=2)
    torch.manual_seed(3)
    lo = old.v6_loss_step(s, b, weights=old.V6LossWeights(),
                          generator=torch.Generator().manual_seed(11), **kw)
    rng_old = torch.random.get_rng_state().clone()
    torch.manual_seed(3)
    ln = v6_loss_step(s, b, weights=V6LossWeights(),
                      generator=torch.Generator().manual_seed(11), **kw)
    rng_new = torch.random.get_rng_state().clone()
    # ⭐ ADDITIVE diagnostic keys are allowed; REMOVALS and VALUE changes are not.
    # MEASURED 2026-08-22: --o5-form / --sigreg-accum added `o5_form`, `o6_rows`,
    # `o6_row_renorm` while the loss stayed BIT-IDENTICAL (loss, terms and RNG
    # draw-count all verified unchanged for S-W and S-J).
    # ⚠️ ORDER MATTERS: the key check used to assert FIRST and abort, so a real
    # numerical regression would have surfaced as "a log key changed" and the
    # value assertions would never have run. Value/terms/RNG now go first.
    ADDITIVE_OK = {"o5_form", "o6_rows", "o6_row_renorm",
                   # H-RANK-22: additive only; the loss is bit-identical.
                   "o1_detach_encoder",
                   # E-DEC-15: additive only -- records whether O1's separation term
                   # stop-gradded the FACTUAL branch (LIT-3 / PhyLatent CASC:
                   # "the factual prediction is treated as a STOP-GRADIENT
                   # REFERENCE"). Two arms differing only in
                   # --o1-stopgrad-factual must be distinguishable from their
                   # logs. Loss, terms and RNG draw-count all verified
                   # BIT-IDENTICAL on the default path for S-W and S-J.
                   "o1_stopgrad_factual"}
    assert torch.equal(lo["loss"], ln["loss"]),         f"{stage}: the DEFAULT loss MOVED against {old._ref}"
    assert lo["log"]["terms"] == ln["log"]["terms"]
    assert torch.equal(rng_old, rng_new),         "the default path consumed a different number of global draws"
    removed = set(lo["log"]) - set(ln["log"])
    assert not removed, f"{stage}: log keys REMOVED against {old._ref}: {removed}"
    unexpected = (set(ln["log"]) - set(lo["log"])) - ADDITIVE_OK
    assert not unexpected,         f"{stage}: unexpected new log keys against {old._ref}: {unexpected}"


def test_NEGATIVE_CONTROL_the_identity_guard_can_fail():
    """Prove the comparison bites: with the term ON the new trainer must
    DIFFER from itself-with-it-off — same batch, same seeds."""
    s = _stack()
    s.eval()
    b = _loss_batch(s, with_s2=True)
    torch.manual_seed(3)
    off = v6_loss_step(s, b, stage="S-S", weights=V6LossWeights(),
                       o1_k=2, o5_k=2)
    torch.manual_seed(3)
    on = v6_loss_step(s, b, stage="S-S",
                      weights=V6LossWeights(w_s2_goal=1.0), o1_k=2, o5_k=2)
    assert not torch.equal(off["loss"], on["loss"])
    assert "s2" in on["log"]["terms"] and "s2" not in off["log"]["terms"]


def test_the_s2_term_consumes_NO_global_rng():
    """C74 discipline: the S2 block draws nothing — switching it on must not
    move the global stream (which would silently re-couple every term after
    it, the exact confound `sigreg_generator` exists to remove)."""
    s = _stack()
    s.eval()
    b = _loss_batch(s, with_s2=True)
    torch.manual_seed(7)
    v6_loss_step(s, b, stage="S-S", weights=V6LossWeights(), o1_k=2, o5_k=2)
    st_off = torch.random.get_rng_state().clone()
    torch.manual_seed(7)
    v6_loss_step(s, b, stage="S-S", weights=V6LossWeights(w_s2_goal=1.0),
                 o1_k=2, o5_k=2)
    assert torch.equal(st_off, torch.random.get_rng_state())


# =========================================================================== #
# 2. ⛔ GOAL HEADS ONLY — gradient reach, MEASURED
# =========================================================================== #

def test_s2_gradient_reaches_EXACTLY_the_two_heads_and_nothing_else():
    """The binding rule, on a real autograd graph: the S2 term's live set is
    the 16 head tensors — no encoder, no readout, no predictor, no adapter,
    no planner, and NOT the shared vocabulary tables."""
    s = _stack()
    for p in s.parameters():
        p.requires_grad_(True)
    L = v6_loss_step(s, _loss_batch(s), stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    names = [n for n, _ in s.named_parameters()]
    grads = torch.autograd.grad(L["s2"], [p for _, p in s.named_parameters()],
                                allow_unused=True)
    live = {n for n, g in zip(names, grads)
            if g is not None and float(g.abs().max()) > 0}
    assert live == S2_REACH, (
        f"S2 reach moved: extra={sorted(live - S2_REACH)}, "
        f"lost={sorted(S2_REACH - live)}")
    assert {s.group_of(n) for n in live} == {"layer_str"}
    assert not any("vocab" in n for n in live), \
        "the vocab tables must NOT be trained by S2 (not applicable, measured)"


def test_NEGATIVE_CONTROL_the_reach_detector_can_see_vocab_params():
    """The 'vocab untouched' claim is only evidence if the detector COULD see
    a vocab parameter when a path exists — e_g_str goes through
    vocab_str.encode, so a loss on it must light the table up."""
    s = _stack()
    for p in s.parameters():
        p.requires_grad_(True)
    out = s.forward(**s.synthetic_batch(2))
    loss = out["e_g_str"].float().pow(2).mean()
    names = [n for n, _ in s.named_parameters()]
    grads = torch.autograd.grad(loss, [p for _, p in s.named_parameters()],
                                allow_unused=True)
    live = {n for n, g in zip(names, grads)
            if g is not None and float(g.abs().max()) > 0}
    assert any(n.startswith("vocab_str.") for n in live), \
        "detector blind: a path through vocab_str.encode showed no gradient"


def test_s2_trains_under_the_S_S_freeze_and_moves_nothing_frozen():
    """End to end through the real freeze map: backward of the FULL S-S loss
    (s1 + s2) populates .grad only inside layer_str."""
    s = _stack()
    apply_stage_freeze(s, "S-S")
    L = v6_loss_step(s, _loss_batch(s), stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    L["loss"].backward()
    got = [n for n, p in s.named_parameters()
           if p.requires_grad and p.grad is not None
           and float(p.grad.abs().sum()) > 0]
    assert any(n.startswith("goal_head_str.") for n in got)
    assert any(n.startswith("act_head_str.") for n in got)
    assert {s.group_of(n) for n in got} == {"layer_str"}
    s.zero_grad(set_to_none=True)


def _live_of_head_ce(s: V6Stack) -> set[str]:
    for p in s.parameters():
        p.requires_grad_(True)
    out = s.forward(**s.synthetic_batch(2))
    raw_ce = torch.nn.functional.cross_entropy(
        out["g_str"]["logits"].float(), torch.zeros(2, dtype=torch.long))
    names = [n for n, _ in s.named_parameters()]
    grads = torch.autograd.grad(raw_ce, [p for _, p in s.named_parameters()],
                                allow_unused=True)
    return {n for n, g in zip(names, grads)
            if g is not None and float(g.abs().max()) > 0}


def test_s2_REFUSES_to_run_with_the_planner_cut_off_and_the_leak_is_real():
    """⛔ The trunk-loss guard + the MEASURED leak it guards against, which is
    two-layered (the two cuts are independent levers, v6.py's own comment):

      * planner cut OFF, uplink ON  — a head CE escapes the goal heads into
        ``adapter_str.*``: the strategic layer's WM-side uplink adapter, the
        very tensors the s1 latent loss trains. Not the encoder — the uplink
        cut still holds — but already a label loss on the REPRESENTATION
        path, which is what "goal heads only, never a trunk loss" forbids.
      * BOTH cuts off — the same CE reaches the ENCODER and READOUT: the
        full trunk loss.

    The refusal keys on the planner cut alone, because that is the cut that
    detaches ``z_str_p`` — and there is no control-arm escape."""
    shallow = _stack(isolate_planner_from_encoder=False)
    live_shallow = _live_of_head_ce(shallow)
    assert any(n.startswith("adapter_str.") for n in live_shallow), \
        "expected the un-cut build to leak into adapter_str — detector broken?"
    assert not {n for n in live_shallow
                if shallow.group_of(n) in ("encoder", "readout")}, \
        "the uplink cut should still hold on this build"
    deep = _stack(isolate_planner_from_encoder=False, isolate_uplink=False)
    groups = {deep.group_of(n) for n in _live_of_head_ce(deep)}
    assert "encoder" in groups and "readout" in groups, \
        "expected the both-cuts-off build to reach the trunk"
    # (b) and the term refuses to be computed on the un-cut build:
    s = _stack(isolate_planner_from_encoder=False)
    with pytest.raises(ValueError, match="TRUNK loss|BINDING"):
        v6_loss_step(s, _loss_batch(s), stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)


# =========================================================================== #
# 3. the term's semantics — masks, bands, refusals
# =========================================================================== #

def test_all_invalid_windows_contribute_zero_but_stay_in_the_graph():
    s = _stack()
    b = _loss_batch(s, with_s2=True)
    for k in ("g_str_id", "a_str_id"):
        b[k] = torch.full_like(b[k], IGNORE_ID)
    b["s2_valid"] = torch.zeros_like(b["s2_valid"])
    for k in ("g_str_args", "g_str_arg_mask", "a_str_args", "a_str_arg_mask"):
        b[k] = torch.zeros_like(b[k])
    L = v6_loss_step(s, b, stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    assert torch.isfinite(L["loss"])
    assert float(L["s2"].detach()) == 0.0
    assert L["log"]["s2_n_valid"] == 0
    assert L["log"]["s2_g_ce"] is None and L["log"]["s2_g_tok_counts"] == {}
    L["loss"].backward()          # the zero term must not break the graph


def test_an_UNSET_arg_slot_sends_exactly_zero_gradient():
    """The §1.2 IGNORE discipline, measured: perturbing a label value in an
    UNSET slot changes nothing; the same perturbation in a SET slot does."""
    s = _stack()
    b = _loss_batch(s, with_s2=True)
    b["s2_valid"] = torch.ones_like(b["s2_valid"])
    b["g_str_id"] = torch.zeros_like(b["g_str_id"])       # KEEP_CORRIDOR
    b["a_str_id"] = torch.zeros_like(b["a_str_id"])
    b["g_str_arg_mask"] = torch.zeros(b["g_str_arg_mask"].shape)
    b["g_str_arg_mask"][:, 0] = 1.0                       # slot 0 SET only
    w = V6LossWeights(w_s2_goal=1.0)
    base = v6_loss_step(s, b, stage="S-S", weights=w, o1_k=10, o5_k=12)
    b2 = dict(b)
    b2["g_str_args"] = b["g_str_args"].clone()
    b2["g_str_args"][:, 3] += 100.0                       # UNSET slot moves
    pert_unset = v6_loss_step(s, b2, stage="S-S", weights=w, o1_k=10, o5_k=12)
    assert torch.equal(base["s2"], pert_unset["s2"])
    b3 = dict(b)
    b3["g_str_args"] = b["g_str_args"].clone()
    b3["g_str_args"][:, 0] += 1.0                         # SET slot moves
    pert_set = v6_loss_step(s, b3, stage="S-S", weights=w, o1_k=10, o5_k=12)
    assert not torch.equal(base["s2"], pert_set["s2"])


def test_invalid_rows_do_not_leak_into_the_CE():
    """Appending out-of-band rows (id = IGNORE) must leave the per-window CE
    untouched — the ignore_index path, verified against a hand computation."""
    s = _stack()
    torch.manual_seed(0)
    out = s.forward(**s.synthetic_batch(4))
    ids = torch.tensor([1, 2, 4, 5])
    valid = torch.tensor([True, True, False, False])
    batch = {
        "g_str_id": torch.where(valid, ids, torch.full_like(ids, IGNORE_ID)),
        "g_str_args": torch.zeros(4, GOAL_ARG_SLOTS),
        "g_str_arg_mask": torch.zeros(4, GOAL_ARG_SLOTS),
        "a_str_id": torch.where(valid, torch.tensor([0, 1, 2, 3]),
                                torch.full((4,), IGNORE_ID,
                                           dtype=torch.long)),
        "a_str_args": torch.zeros(4, GOAL_ARG_SLOTS),
        "a_str_arg_mask": torch.zeros(4, GOAL_ARG_SLOTS),
        "s2_valid": valid,
    }
    loss, log = s2_goal_loss(out["g_str"], out["a_str"], batch)
    hand = (torch.nn.functional.cross_entropy(
        out["g_str"]["logits"].float()[:2], ids[:2])
        + torch.nn.functional.cross_entropy(
            out["a_str"]["logits"].float()[:2], torch.tensor([0, 1])))
    assert torch.allclose(loss, hand, atol=1e-6)
    assert log["s2_n_valid"] == 2
    assert log["s2_g_tok_counts"] == {
        STRATEGIC_GOAL_TOKENS[1]: 1, STRATEGIC_GOAL_TOKENS[2]: 1}


def test_the_loss_REFUSES_route_to_and_missing_keys():
    s = _stack()
    b = _loss_batch(s, with_s2=True)
    b["s2_valid"] = torch.ones_like(b["s2_valid"])
    b["g_str_id"] = torch.full_like(b["g_str_id"], ROUTE_TO_ID)
    with pytest.raises(ValueError, match="ROUTE_TO"):
        v6_loss_step(s, b, stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    b2 = _loss_batch(s, with_s2=False)
    with pytest.raises(ValueError, match="missing.*s2_valid|s2"):
        v6_loss_step(s, b2, stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)


def test_the_log_is_per_family_and_per_token_never_pooled():
    s = _stack()
    L = v6_loss_step(s, _loss_batch(s), stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    for k in ("s2_g_ce", "s2_a_ce", "s2_g_arg_l1", "s2_a_arg_l1",
              "s2_g_acc", "s2_a_acc", "s2_g_tok_counts", "s2_a_tok_counts",
              "s2_n_valid", "s2_n_windows"):
        assert k in L["log"], k
    assert sum(L["log"]["s2_g_tok_counts"].values()) == L["log"]["s2_n_valid"]
    assert sum(L["log"]["s2_a_tok_counts"].values()) == L["log"]["s2_n_valid"]


# =========================================================================== #
# 4. the loader — every refusal proven able to fail, plus its passing twin
# =========================================================================== #

_CID_A = "0089a096-68be-40df-8097-780bf1ae1c19"
_CID_B = "00d05901-ed0a-4a43-adca-bdab70d30bfa"


def _legacy(cid: str) -> int:
    return int.from_bytes(cid.encode()[:4], "big")


def _index(clips, t0=8.0, band=(-2.0, 2.0), **hdr):
    out = {"_t0_s": t0, "_valid_window_s": list(band), "clips": {}}
    out.update(hdr)
    for cid, ent in clips.items():
        out["clips"][cid] = {
            "label_split": "aug120", "corpus": "test",
            "v2ep_file": f"{cid}.v2ep.pt",
            "episode_id_legacy": _legacy(cid),
            "episode_id_stable": stable_episode_id(cid),
            "excluded": False, **ent}
    return out


def _block(tok, tokens, slots=(), args=None, provenance="path",
           sources=("engine_a.route_v3",)):
    a = [0.0] * GOAL_ARG_SLOTS
    m = [0] * GOAL_ARG_SLOTS
    for i, s in enumerate(slots):
        m[s] = 1
        a[s] = (args or [27.3] * len(slots))[i]
    return {"token": tok, "token_id": tokens.index(tok), "args": a,
            "arg_mask": m, "provenance": provenance,
            "sources": list(sources)}


def _rec(cid, g_tok="TURN_LEFT", a_tok="HOLD_CORRIDOR", **over):
    rec = {
        "schema_version": "s2-strategic-v1", "clip_id": cid, "t0_s": 8.0,
        "g_str": _block(g_tok, STRATEGIC_GOAL_TOKENS,
                        slots=(0,) if g_tok not in ("FOLLOW_MAIN_ROAD",
                                                    "NONE_ABSTAIN",
                                                    "ROUTE_TO") else ()),
        "a_str": _block(a_tok, STRATEGIC_ACTION_TOKENS,
                        slots=(6,) if a_tok == "HOLD_CORRIDOR" else ()),
        "valid_window_s": [-2.0, 2.0],
        "disjointness": {"situation_classifier_output_used": False},
    }
    rec.update(over)
    return rec


def _write(tmp_path, records, index=None, name="s2_labels_test.jsonl"):
    d = tmp_path / "labels"
    d.mkdir(parents=True, exist_ok=True)
    if index is not None:
        (d / "clip_index.json").write_text(json.dumps(index))
    (d / name).write_text("\n".join(json.dumps(r) for r in records))
    return d


def test_a_wellformed_artifact_loads_and_reports_per_token(tmp_path):
    d = _write(tmp_path, [_rec(_CID_A), _rec(_CID_B, g_tok="STOP_AT")],
               _index({_CID_A: {}, _CID_B: {}}))
    ls = load_s2_labels(d)
    assert len(ls) == 2
    assert ls.token_census()["g_str"] == {"STOP_AT": 1, "TURN_LEFT": 1}
    assert ls.token_census()["a_str"] == {"HOLD_CORRIDOR": 2}
    assert ls.rows_by_stable[stable_episode_id(_CID_A)].g_id == \
        STRATEGIC_GOAL_TOKENS.index("TURN_LEFT")


def test_missing_clip_index_REFUSES(tmp_path):
    d = _write(tmp_path, [_rec(_CID_A)], index=None)
    with pytest.raises(S2LabelError, match="UNJOINABLE|MISSING"):
        load_s2_labels(d)


def test_ROUTE_TO_record_REFUSES_and_the_remap_loads(tmp_path):
    bad = _rec(_CID_A)
    bad["g_str"] = _block("ROUTE_TO", STRATEGIC_GOAL_TOKENS)
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="ROUTE_TO.*GATED|GATED"):
        load_s2_labels(d)
    ok = _rec(_CID_A, g_tok="TURN_RIGHT")            # the geometry remap
    _write(tmp_path, [ok], _index({_CID_A: {}}))
    assert len(load_s2_labels(tmp_path / "labels")) == 1


def test_token_id_drift_REFUSES(tmp_path):
    bad = _rec(_CID_A)
    bad["g_str"]["token_id"] = 3                     # TURN_LEFT is 4
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="token_id"):
        load_s2_labels(d)


def test_unknown_clip_duplicate_and_excluded_all_REFUSE(tmp_path):
    d = _write(tmp_path, [_rec(_CID_B)], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="not in"):
        load_s2_labels(d)
    d = _write(tmp_path, [_rec(_CID_A), _rec(_CID_A)], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="duplicate"):
        load_s2_labels(d)
    d = _write(tmp_path, [_rec(_CID_A)],
               _index({_CID_A: {"excluded": True}}))
    with pytest.raises(S2LabelError, match="EXCLUDED"):
        load_s2_labels(d)


def test_disjointness_scan_fires_on_the_payload_ONLY(tmp_path):
    """A situation-classifier spelling in the GOAL PAYLOAD refuses; the same
    word in a META field outside it does not — the polling-monitor trap: the
    scan must not match the record's own stamp/notes."""
    bad = _rec(_CID_A)
    bad["g_str"]["sources"] = ["situation_classifier.argmax"]
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="disjointness"):
        load_s2_labels(d)
    meta_only = _rec(_CID_A,
                     _provenance={"note": "situation classifier NOT used"})
    _write(tmp_path, [meta_only], _index({_CID_A: {}}))
    assert len(load_s2_labels(tmp_path / "labels")) == 1


def test_missing_disjointness_stamp_REFUSES(tmp_path):
    bad = _rec(_CID_A)
    del bad["disjointness"]
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="disjointness stamp"):
        load_s2_labels(d)


def test_a_drifted_stable_id_in_the_index_REFUSES(tmp_path):
    idx = _index({_CID_A: {}})
    idx["clips"][_CID_A]["episode_id_stable"] += 1
    d = _write(tmp_path, [_rec(_CID_A)], idx)
    with pytest.raises(S2LabelError, match="disagree on the hash"):
        load_s2_labels(d)


def test_mask_and_arg_discipline_violations_REFUSE(tmp_path):
    bad = _rec(_CID_A)
    bad["g_str"]["args"] = [1.0] * (GOAL_ARG_SLOTS - 1)          # wrong len
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="args must be"):
        load_s2_labels(d)
    bad = _rec(_CID_A)
    bad["g_str"]["arg_mask"][3] = 2                              # non-binary
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="arg_mask must be"):
        load_s2_labels(d)
    bad = _rec(_CID_A)
    bad["g_str"]["args"][3] = 5.0                    # value in an UNSET slot
    d = _write(tmp_path, [bad], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="UNSET"):
        load_s2_labels(d)


# =========================================================================== #
# 4b. ABSTENTION — a per-family mask, default-off and provably inert
# =========================================================================== #

def _abstain_block(provenance="review"):
    """An abstaining family block: NO token, nothing constrained."""
    return {"abstain": True, "args": [0.0] * GOAL_ARG_SLOTS,
            "arg_mask": [0] * GOAL_ARG_SLOTS, "provenance": provenance,
            "sources": ["review.pi_verdicts"]}


def test_an_abstaining_family_loads_unsupervised_and_is_COUNTED(tmp_path):
    """⛔ THE POINT OF THE CHANNEL. `a_str` has no abstain TOKEN — asserted
    here against the real vocabulary, because the whole design follows from
    it — so "we do not know the action" can only be said with a mask."""
    assert "NONE_ABSTAIN" not in STRATEGIC_ACTION_TOKENS
    assert "NONE_ABSTAIN" in STRATEGIC_GOAL_TOKENS      # g_str CAN say it
    r = _rec(_CID_A)
    r["a_str"] = _abstain_block()
    d = _write(tmp_path, [r, _rec(_CID_B)], _index({_CID_A: {}, _CID_B: {}}))
    ls = load_s2_labels(d)
    assert ls.has_abstain is True
    assert ls.abstain_census() == {"g_str": 0, "a_str": 1}
    # the declined family is COUNTED, not dropped: the census still sums to n
    cens = ls.token_census()
    assert cens["a_str"][SL.NO_LABEL] == 1
    assert sum(cens["a_str"].values()) == len(ls) == 2
    assert SL.NO_LABEL not in cens["g_str"]
    row = ls.rows_by_stable[stable_episode_id(_CID_A)]
    assert row.a_sup is False and row.g_sup is True
    assert row.a_id == IGNORE_ID


def test_an_abstaining_block_carrying_a_token_or_an_arg_REFUSES(tmp_path):
    """A declined family that still names a manoeuvre is the exact ambiguity
    the channel exists to remove, so it is refused rather than laundered."""
    for mutate, pat in (
            (lambda b: b.update(token="HOLD_CORRIDOR"), "abstain:true AND"),
            (lambda b: b.update(token_id=1), "abstain:true AND"),
            (lambda b: b["arg_mask"].__setitem__(6, 1), "abstains but"),
            (lambda b: b["args"].__setitem__(6, 3.0), "abstains but"),
            (lambda b: b.update(abstain="yes"), "must be true or absent")):
        r = _rec(_CID_A)
        r["a_str"] = _abstain_block()
        mutate(r["a_str"])
        d = _write(tmp_path, [r], _index({_CID_A: {}}))
        with pytest.raises(S2LabelError, match=pat):
            load_s2_labels(d)


def test_abstaining_on_BOTH_families_REFUSES(tmp_path):
    """A record that supervises nothing is an ABSENT label wearing an
    abstention's clothes — omit the clip instead."""
    r = _rec(_CID_A)
    r["g_str"], r["a_str"] = _abstain_block(), _abstain_block()
    d = _write(tmp_path, [r], _index({_CID_A: {}}))
    with pytest.raises(S2LabelError, match="BOTH families"):
        load_s2_labels(d)


def _sup_for(ls, eids=(_CID_A, _CID_B)):
    eps = [_Ep(stable_episode_id(c)) for c in eids]
    return ls.supervision(eps, window=6, dt=0.1,
                          index=[(i, 80) for i in range(len(eps))])


def test_the_batch_omits_the_family_keys_UNLESS_a_record_abstains(tmp_path):
    """⛔ THE INERTNESS CONTRACT, both branches. No abstention anywhere =>
    the incumbent SEVEN-key batch, byte-for-byte. One abstention => exactly
    two extra bool keys and nothing else moves."""
    plain = _loaded(tmp_path / "plain")
    b0 = _sup_for(plain).batch([0, 1])
    assert set(b0) == {"g_str_id", "g_str_args", "g_str_arg_mask", "a_str_id",
                       "a_str_args", "a_str_arg_mask", "s2_valid"}
    r = _rec(_CID_A)
    r["a_str"] = _abstain_block()
    d = _write(tmp_path / "abst", [r, _rec(_CID_B)],
               _index({_CID_A: {}, _CID_B: {}}))
    sup = _sup_for(load_s2_labels(d))
    b1 = sup.batch([0, 1])
    assert set(b1) - set(b0) == {"g_str_valid", "a_str_valid"}
    assert b1["s2_valid"].tolist() == [True, True]      # BOTH still in band
    assert b1["a_str_valid"].tolist() == [False, True]
    assert b1["g_str_valid"].tolist() == [True, True]
    assert sup.n_windows_in_band == 2
    assert sup.n_windows_supervised == {"g_str": 2, "a_str": 1}
    assert sup.emits_family_masks is True


def test_an_abstained_family_sends_EXACTLY_ZERO_gradient_to_its_head():
    """The measured half: with `a_str_valid` all-False, the S2 TERM reaches
    the goal head and NOT ONE action-head tensor. Isolated on ``L["s2"]``, the
    same way `test_s2_gradient_reaches_EXACTLY_the_two_heads` is — the total
    S-S loss has other terms that legitimately reach `act_head_str`, so
    backprop through `L["loss"]` would measure those and prove nothing about
    the mask (a probe answering the wrong question looks exactly like an
    answer)."""
    s = _stack()
    for p in s.parameters():
        p.requires_grad_(True)
    b = _loss_batch(s)
    b["s2_valid"] = torch.ones_like(b["s2_valid"])
    b["a_str_valid"] = torch.zeros_like(b["s2_valid"])
    b["g_str_valid"] = torch.ones_like(b["s2_valid"])
    L = v6_loss_step(s, b, stage="S-S",
                     weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    names = [n for n, _ in s.named_parameters()]
    grads = torch.autograd.grad(L["s2"], [p for _, p in s.named_parameters()],
                                allow_unused=True)
    live = {n for n, g in zip(names, grads)
            if g is not None and float(g.abs().max()) > 0}
    assert not any(n.startswith("act_head_str.") for n in live), \
        f"an abstained family still trained: {sorted(live)}"
    assert any(n.startswith("goal_head_str.") for n in live)
    assert L["log"]["s2_a_n_abstained"] == int(b["s2_valid"].sum())
    assert L["log"]["s2_g_n_abstained"] == 0
    assert L["log"]["s2_a_n_valid"] == 0


def test_the_family_masks_are_ABSENT_by_default_and_only_ever_REMOVE():
    """Absent => identical to the incumbent, bit-for-bit. Present-and-all-True
    => also identical (the AND is a no-op). And the mask cannot ADD a window
    the band excluded: all-True over an all-False `s2_valid` stays empty."""
    s = _stack()
    base = v6_loss_step(s, _loss_batch(s), stage="S-S",
                        weights=V6LossWeights(w_s2_goal=1.0),
                        o1_k=10, o5_k=12)
    b = _loss_batch(s)
    b["g_str_valid"] = torch.ones_like(b["s2_valid"])
    b["a_str_valid"] = torch.ones_like(b["s2_valid"])
    same = v6_loss_step(s, b, stage="S-S",
                        weights=V6LossWeights(w_s2_goal=1.0),
                        o1_k=10, o5_k=12)
    assert float(same["loss"]) == float(base["loss"])
    assert same["log"]["s2_g_n_abstained"] == 0
    b2 = _loss_batch(s)
    b2["s2_valid"] = torch.zeros_like(b2["s2_valid"])
    b2["g_str_valid"] = torch.ones_like(b2["s2_valid"])
    b2["a_str_valid"] = torch.ones_like(b2["s2_valid"])
    off = v6_loss_step(s, b2, stage="S-S",
                       weights=V6LossWeights(w_s2_goal=1.0), o1_k=10, o5_k=12)
    assert off["log"]["s2_g_n_valid"] == 0 and off["log"]["s2_a_n_valid"] == 0


def test_a_misshaped_family_mask_REFUSES():
    s = _stack()
    b = _loss_batch(s)
    b["a_str_valid"] = torch.ones(len(b["s2_valid"]) + 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="a_str_valid must be"):
        v6_loss_step(s, b, stage="S-S", weights=V6LossWeights(w_s2_goal=1.0),
                     o1_k=10, o5_k=12)


def test_stable_episode_id_fallback_matches_the_canonical_one():
    """The p8 pattern's pin, applied to this loader: whenever the canonical
    v2_dataset implementation is importable, the loader's function IS it (or
    computes identically through the fallback)."""
    try:
        from tanitad.data.v2_dataset import stable_episode_id as canon
    except ImportError:
        pytest.skip("v2_dataset not importable here (torchvision)")
    for cid in (_CID_A, _CID_B, "x"):
        assert stable_episode_id(cid) == canon(cid)


#: The SUPERSEDED v1 delivery. Kept as a NAME in this test file for exactly one
#: reason: to prove the loader refuses it. Never as a load target.
_SUPERSEDED = _ROOT / "TanitAD Research Lab" / "Data Engineering" / \
    "Implementation" / "incoming" / "2026-08-16-s2-v1-labels" / "labels"
#: The CANONICAL set, resolved through the code constant rather than retyped —
#: if the constant moves, this test follows it instead of silently testing a
#: path nothing uses.
_REAL = SL.s2_canonical_labels_dir(_ROOT)


def test_the_CANONICAL_label_set_is_the_CORRECTED_one_not_the_v1_delivery():
    """⛔ WHICH LABELS `--s2-labels` MUST POINT AT.

    The v1 delivery's lane-change rows were adjudicated ~78 % WRONG by the PI
    and the geometric derivation was removed (06b8782). Pinned here because
    the failure mode is INVISIBLE: the superseded set has the same 797
    records, the same index, the same band, and passes every other guard in
    the loader identically — it is simply wrong in 80 rows. So the test
    asserts the corrected census, and that the two tokens the review deleted
    are ABSENT rather than merely rarer."""
    if not _REAL.is_dir():
        pytest.skip("the S2 label delivery is not present in this checkout")
    ls = load_s2_labels(_REAL)
    assert len(ls) == 797
    g, a = ls.token_census()["g_str"], ls.token_census()["a_str"]
    assert g == {"FOLLOW_MAIN_ROAD": 474, "NONE_ABSTAIN": 14, "STOP_AT": 59,
                 "TURN_LEFT": 137, "TURN_RIGHT": 113}
    assert a == {"HOLD_CORRIDOR": 597, "PREPARE_STOP": 88, "REDUCE_TO": 94,
                 "RESUME_CRUISE": 18}
    assert "LANE_TARGET" not in g and "PREPARE_LANE_CHANGE" not in a
    assert sum(g.values()) == sum(a.values()) == 797
    assert ls.provenance_census() == {"g_str": {"path": 797},
                                      "a_str": {"path": 797}}
    assert ls.t0_s == 8.0 and ls.band == (-2.0, 2.0)
    assert ls.source["n_index_clips"] == 801
    assert ls.source["n_index_excluded"] == 4


def test_the_SUPERSEDED_v1_labels_are_REFUSED_and_name_their_replacement():
    """Pointing `--s2-labels` at the old delivery must FAIL, not train.

    ⛔ The negative half of the test above, and the load-bearing one: the
    superseded directory is still on disk (it is the review's own evidence),
    so nothing but this refusal stops a launch line copied from an older doc
    from supervising the strategic head with the 80 wrong targets."""
    if not _SUPERSEDED.is_dir():
        pytest.skip("the S2 v1 label delivery is not present in this checkout")
    assert (_SUPERSEDED / SL.SUPERSEDED_NAME).is_file()
    with pytest.raises(S2LabelError) as e:
        load_s2_labels(_SUPERSEDED)
    msg = str(e.value)
    assert "SUPERSEDED" in msg and "labels_v2" in msg
    # a single .jsonl inside a superseded dir is refused too — the marker
    # guards the DIRECTORY, not just the directory-shaped invocation.
    one = next(_SUPERSEDED.glob("s2_labels_*.jsonl"))
    with pytest.raises(S2LabelError, match="SUPERSEDED"):
        load_s2_labels(one)


def test_the_trainers_help_copy_of_the_canonical_path_CANNOT_drift():
    """`train_v6_staged` keeps its OWN literal of the canonical label path
    rather than importing it.

    ⛔ That is deliberate and was MEASURED: the module-level import added
    `s2_labels` to the trainer's IMPORT-TIME CLOSURE, which
    `test_runbook_commands.py` pins because the closure is the set of files
    that must be FILE-SHIPPED to a pod — importing a module to print a help
    string would have made it mandatory for every launch. The cost of the
    literal is a second copy, so the copies are audited against each other
    here (C81), and the help text is checked to actually carry it."""
    import train_v6_staged as T
    assert T.S2_CANONICAL_LABELS_REL == SL.S2_CANONICAL_LABELS_REL
    helps = " ".join(
        a.help or "" for a in build_parser()._actions
        if "--s2-labels" in (a.option_strings or []))
    assert SL.S2_CANONICAL_LABELS_REL in helps, \
        "--s2-labels' help must name the canonical set, not just imply it"
    assert "SUPERSEDED" in helps


def test_the_marker_and_the_code_constant_CANNOT_silently_disagree():
    """The canonical path is written in TWO places — `SUPERSEDED.json`'s
    `superseded_by` and `s2_labels.S2_CANONICAL_LABELS_REL`. C81's rule for a
    fact written twice is to audit the copies AGAINST each other, because the
    stale one is the one that gets read. They must resolve to one directory."""
    if not _SUPERSEDED.is_dir():
        pytest.skip("the S2 v1 label delivery is not present in this checkout")
    m = json.loads((_SUPERSEDED / SL.SUPERSEDED_NAME)
                   .read_text(encoding="utf-8"))
    assert (_SUPERSEDED / m["superseded_by"]).resolve() == _REAL.resolve()
    assert _REAL.is_dir(), "the constant names a directory that must exist"


def test_the_80_ex_lane_change_rows_are_RESOLVED_not_declined():
    """⛔ THE PI'S QUESTION, ANSWERED ON THE SHIPPED ARTIFACT (2026-08-16):
    *"investigate what these labels are … and give them one by adjusting the
    approach or confirm their cases for follow route"*.

    They are ROUTE-FOLLOWING cases, and the answer is a derivation, not a
    reassignment: `engine_a.route.token == "follow"` with `token_valid: true`
    for **80/80** (MEASURED, `labels/engine_a_*.jsonl`), so the route engine
    had already called them follow and the removed geometric gate was
    OVERRIDING its own route token. The `a_str` split is likewise derived —
    from the LONGITUDINAL engine the lane-change branch had been shadowing
    (median `net_dv` +0.27 m/s for the 71 HOLD_CORRIDOR vs −3.35 m/s for the
    9 REDUCE_TO).

    ⇒ **the abstain residual is ZERO for `a_str`.** The machinery exists
    (`abstain: true`) for a future source that can decide the goal but not the
    action; this corpus never needs it, and that is asserted rather than
    assumed. `has_abstain False` is what keeps the batch schema byte-identical
    for the live run."""
    if not _REAL.is_dir():
        pytest.skip("the S2 label delivery is not present in this checkout")
    ls = load_s2_labels(_REAL)
    a = ls.token_census()["a_str"]
    assert a["HOLD_CORRIDOR"] == 526 + 71 and a["REDUCE_TO"] == 85 + 9, (
        "the ex-lane-change rows moved: re-derive them from engine_a, do not "
        "adjust this number to match")
    assert ls.abstain_census() == {"g_str": 0, "a_str": 0}, (
        "an abstaining record appeared in the shipped labels. The PI's "
        "ruling is that these clips are RESOLVED as route-following, so an "
        "abstention here needs a stated reason: record it in "
        "INTEGRATION_CLOSE.md and update this test — do not delete it")
    assert ls.has_abstain is False
    assert SL.NO_LABEL not in a and SL.NO_LABEL not in ls.token_census()[
        "g_str"]


def test_the_ex_lane_change_clips_are_route_FOLLOW_in_the_route_engine():
    """The evidence behind the test above, read from the PRIMARY artifact
    (`engine_a_*.jsonl`) rather than inherited from the label census.

    ⛔ WHY IT IS A SEPARATE TEST. "these 80 are route-following" is the claim
    the whole relabel rests on; asserting it only through the labels would be
    circular — the labels are the thing being justified. This reads the route
    engine's own token."""
    d = _SUPERSEDED           # engine_a_*.jsonl ship beside the v1 labels
    if not d.is_dir():
        pytest.skip("the S2 label delivery is not present in this checkout")
    ea = {}
    for f in sorted(d.glob("engine_a_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                ea[r["clip_id"]] = r["engine_a"]
    assert len(ea) == 801
    changed = json.loads((_SUPERSEDED.parent / "review" / "raw" /
                          "lc_emit.json").read_text(encoding="utf-8"))
    ex = [row[0] for row in changed["changed"]]
    assert len(ex) == 80
    bad = [c for c in ex if not (ea[c]["route"]["token"] == "follow"
                                 and ea[c]["route"]["token_valid"])]
    assert not bad, (
        f"{len(bad)} of the ex-LANE_TARGET clips whose route engine does NOT "
        f"say follow: {bad[:5]} — the relabel's justification fails for them")
    # ⚠️ SCOPE, stated because the wider claim is FALSE and I measured it:
    # FOLLOW_MAIN_ROAD corpus-wide is NOT co-extensive with route==follow. Of
    # its 474 records, 98 carry route ('unknown', token_valid=False) and 3
    # carry a valid 'merge' — FOLLOW_MAIN_ROAD is the DECLARED DEFAULT when no
    # route is set (v6.py:148, HIERARCHY_VOCABULARY §3), so those are correct
    # and not evidence for the relabel. The 80 are.
    ls = load_s2_labels(_REAL)
    assert sum(1 for r in ls.rows_by_stable.values()
               if r.g_token == "FOLLOW_MAIN_ROAD") == 474


# =========================================================================== #
# 5. the join — stable ids in, legacy ids refused, the band applied
# =========================================================================== #

class _Ep:
    """Quacks exactly as much of the provider as the join reads."""

    def __init__(self, eid, T=118, ch=9):
        self.episode_id = int(eid)
        self.frames = torch.zeros(T, ch, 2, 2, dtype=torch.uint8)


def _loaded(tmp_path):
    d = _write(tmp_path, [_rec(_CID_A), _rec(_CID_B, g_tok="STOP_AT")],
               _index({_CID_A: {}, _CID_B: {}}))
    return load_s2_labels(d)


def test_the_join_supervises_exactly_the_band_windows(tmp_path):
    """window=6, 3-frame stack (9ch): raw now = t + 5 + 2, so the ±2 s band
    around t0=8 s (raw 60..100) is provider t ∈ [53, 93] — checked at both
    edges, both sides."""
    ls = _loaded(tmp_path)
    eps = [_Ep(stable_episode_id(_CID_A)), _Ep(12345)]     # ep 1 unlabeled
    index = [(0, t) for t in (52, 53, 93, 94)] + [(1, 60)]
    sup = ls.supervision(eps, window=6, dt=0.1, index=index)
    assert sup.n_matched_episodes == 1
    assert sup.n_windows_in_band == 2
    b = sup.batch([0, 1, 2, 3, 4])
    assert b["s2_valid"].tolist() == [False, True, True, False, False]
    tl = STRATEGIC_GOAL_TOKENS.index("TURN_LEFT")
    assert b["g_str_id"].tolist() == [IGNORE_ID, tl, tl, IGNORE_ID, IGNORE_ID]
    assert float(b["g_str_args"][1, 0]) == pytest.approx(27.3)
    assert b["g_str_arg_mask"][1].tolist() == [1.0] + [0.0] * 7
    assert sup.window_token_census["g_str"] == {"TURN_LEFT": 2}


def test_the_stack_offset_is_read_off_the_episode_channels(tmp_path):
    """A 3-channel (no-stack) episode has offset 0: the same t is now 0.2 s
    later on the raw clock than under a 9-channel stack — the band edge
    moves by exactly the two dropped frames."""
    ls = _loaded(tmp_path)
    eps = [_Ep(stable_episode_id(_CID_A), ch=3)]
    sup = ls.supervision(eps, window=6, dt=0.1,
                         index=[(0, 54), (0, 55), (0, 95), (0, 96)])
    b = sup.batch([0, 1, 2, 3])
    assert b["s2_valid"].tolist() == [False, True, True, False]


def test_LEGACY_16bit_ids_are_REFUSED_with_the_collision_diagnosis(tmp_path):
    """The collision trap: an episode carrying the 16-bit payload id (the
    stale-manifest / stable_ids=False shape) must refuse the whole join —
    naming the legacy id — never silently join or silently not-join."""
    ls = _loaded(tmp_path)
    eps = [_Ep(_legacy(_CID_A))]
    with pytest.raises(S2LabelError, match="LEGACY 16-bit"):
        ls.supervision(eps, window=6, dt=0.1, index=[(0, 60)])
    # negative control: the SAME clip under its stable id joins cleanly.
    sup = ls.supervision([_Ep(stable_episode_id(_CID_A))], window=6, dt=0.1,
                         index=[(0, 60)])
    assert sup.n_matched_episodes == 1


def test_an_unlabeled_corpus_joins_zero_and_says_so(tmp_path):
    ls = _loaded(tmp_path)
    sup = ls.supervision([_Ep(999), _Ep(1000)], window=6, dt=0.1,
                         index=[(0, 60), (1, 60)])
    assert sup.n_matched_episodes == 0 and sup.n_windows_in_band == 0
    assert sup.report()["n_matched_episodes"] == 0
    b = sup.batch([0, 1])
    assert not bool(b["s2_valid"].any())


# =========================================================================== #
# 6. the launch surface — preflight refusals + dry-run integration
# =========================================================================== #

def _args(*argv):
    ap = build_parser()
    import argparse
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    return ap.parse_args(list(argv))


def _tiny_argv(tmp_path, stage="S-S", *extra):
    return ["--stage", stage, "--out", str(tmp_path / "out"), "--dry-run",
            "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
            "--patch", "16", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "8",
            "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "2", "--d-tac", "32",
            "--d-str", "16", "--d-goal-embed", "16", "--adapter-hidden", "32",
            "--n-candidates", "3", "--sigreg-slices", "8",
            "--dry-steps", "1", "--dry-batch", "2", "--dry-k", "12",
            *extra]


def test_preflight_refuses_s2_in_stages_that_freeze_layer_str(tmp_path):
    for stage in ("S-W", "S-T"):
        probs = preflight(_args(*_tiny_argv(tmp_path, stage,
                                            "--w-s2-goal", "1")))
        assert any("--w-s2-goal" in p and stage in p for p in probs), stage


def test_preflight_refuses_weight_without_labels_on_a_REAL_run(tmp_path):
    a = _args("--stage", "S-S", "--out", str(tmp_path), "--w-s2-goal", "1",
              "--init-from", "x", "--v2-cache", "y")
    assert any("without --s2-labels" in p for p in preflight(a))
    # ...but a dry-run may smoke the loss on synthetic keys without labels.
    a = _args(*_tiny_argv(tmp_path, "S-S", "--w-s2-goal", "1"))
    assert not any("s2" in p.lower() for p in preflight(a))


def test_preflight_refuses_labels_with_the_term_off_unless_acked(tmp_path):
    lbl = _write(tmp_path, [_rec(_CID_A)], _index({_CID_A: {}}))
    a = _args(*_tiny_argv(tmp_path, "S-S", "--s2-labels", str(lbl)))
    assert any("not in force" in p and "--s2-labels" in p
               for p in preflight(a))
    a = _args(*_tiny_argv(tmp_path, "S-S", "--s2-labels", str(lbl),
                          "--i-know-this-is-the-control-arm"))
    assert not any("--s2-labels" in p for p in preflight(a))


def test_preflight_refuses_a_missing_labels_path_in_milliseconds(tmp_path):
    a = _args(*_tiny_argv(tmp_path, "S-S", "--w-s2-goal", "1",
                          "--s2-labels", str(tmp_path / "nope")))
    assert any("does not exist" in p for p in preflight(a))


def test_preflight_refuses_s2_with_no_isolate_planner_UNCONDITIONALLY(
        tmp_path):
    lbl = _write(tmp_path, [_rec(_CID_A)], _index({_CID_A: {}}))
    base = _tiny_argv(tmp_path, "S-S", "--w-s2-goal", "1", "--s2-labels",
                      str(lbl), "--no-isolate-planner")
    assert any("TRUNK loss" in p for p in preflight(_args(*base)))
    # the ack flag that unlocks other control arms does NOT unlock this one:
    probs = preflight(_args(*base, "--i-know-this-is-the-control-arm"))
    assert any("TRUNK loss" in p for p in probs)


def test_preflight_accepts_the_production_shape(tmp_path):
    lbl = _write(tmp_path, [_rec(_CID_A)], _index({_CID_A: {}}))
    for stage in ("S-S", "S-J"):
        a = _args(*_tiny_argv(tmp_path, stage, "--w-s2-goal", "1",
                              "--s2-labels", str(lbl)))
        assert not any("s2" in p.lower() for p in preflight(a)), stage


def test_the_weight_flag_reaches_the_loss_weights():
    from train_v6_staged import _weights_from_args
    a = _args("--stage", "S-S", "--out", "unused", "--w-s2-goal", "0.5")
    assert _weights_from_args(a).w_s2_goal == 0.5


def test_dry_run_exercises_the_s2_loss_and_the_real_loader(tmp_path):
    """An S-S dry-run with the term on runs the REAL loss path on synthetic
    keys (s2 log keys present), and when --s2-labels is supplied the LOADER
    is really exercised and its report lands in dry_run.json."""
    lbl = _write(tmp_path, [_rec(_CID_A)], _index({_CID_A: {}}))
    a = _args(*_tiny_argv(tmp_path, "S-S", "--w-s2-goal", "1",
                          "--s2-labels", str(lbl)))
    r = dry_run(a)
    assert r["s2_labels"]["exercised"] is True
    assert r["s2_labels"]["n_records"] == 1
    assert any(k.startswith("s2_") for k in r["steps"][0])
    assert "s2" in r["steps"][0]["terms"]
    out = json.loads((tmp_path / "out" / "dry_run.json").read_text())
    assert out["s2_labels"]["exercised"] is True
    # and WITHOUT the flag, the report says the loader was NOT exercised.
    a2 = _args(*_tiny_argv(tmp_path / "b", "S-S", "--w-s2-goal", "1"))
    r2 = dry_run(a2)
    assert r2["s2_labels"]["exercised"] is False
    assert "s2" in r2["steps"][0]["terms"]
