"""refcv7 diagnostics F4 (2026-10-04): the per-head detection gate (``tanitad.eval.detection_metrics``).

Source of the fix: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-map-box-diagnostics``
(RESULT.md sec. 2.3, fix F4; the TRAIN P = R gates in ``raw/B_box.json`` ``fit_train_diag.T3_gate``).

What is pinned, each with a LITERAL expected value and a deliberate-regression arm that must go RED:

A. ⛔ DEFAULT UNCHANGED. ``summarise(packs, head)`` -- the declared 0.5-gate eval keys -- on 118 real EVAL windows equals
   goldens computed from the TIP's ORIGINAL ``detection_metrics.py`` before this change (key set, a hash over every
   value, and the headline numbers); ``gated_census_keys(.., gates=None)`` emits NOTHING.
B. ACCEPTANCE on real per-window packs (every 9th EVAL-DIAG window of refcv7-r101-s0 at step 50,400, 118 windows, the
   diagnostics' own banked packs): the census AT the TRAIN P = R gate equals the LITERALS computed by the diagnostics'
   own procedure (``analyze_box.py``: shift the logit so the gate maps to 0.5, threshold at 0.5), for both heads, and the
   A10 alarm flips exactly as the diagnostics report (declared gate: alarm; TRAIN P = R gate: in band).
C. The shipped gates file (values copied from ``raw/B_box.json``, provenance) and the loader's refusals.
D. ⛔ THE PLANNER IS UNTOUCHED: ``AgentTokenEmbed`` (``refc_agents``: the raw ``sigmoid(presence_logit)`` as a feature
   and a soft scale; its OWN ``presence_gate`` in hard mode) is bit-identical before / after every gate API runs, never
   calls into ``detection_metrics``, and re-gating it by the detection gate goes RED.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.eval import detection_metrics as det
from tanitad.models import slot_presence as sp
from tanitad.refs import refc_agents as ra

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "refcv7_infer_fixes" / "refcv7_box_packs_eval_every9th.npz"
GATES = Path(det.__file__).resolve().parent.parent / "configs" / "refcv7_det_presence_gates_train.json"
#: raw/B_box.json fit_train_diag.T3_gate, copied
GATE = {"box3d": 0.2567453384399414, "agent": 0.23066936433315277}


def _packs(head):
    d = np.load(FIX)
    g = lambda k: d[head + "__" + k]                                      # noqa: E731
    out = []
    for i in range(g("logit").shape[0]):
        ng, npair = int(g("n_gt")[i]), int(g("n_pair")[i])
        out.append({"ep": None, "logit": g("logit")[i], "xy": g("xy")[i], "cls": g("cls")[i],
                    "cls_corr": g("cls_corr")[i], "matched": g("matched")[i], "exempt": g("exempt")[i],
                    "pair_err": g("pair_err")[i][:npair], "pair_size_err": g("pair_size_err")[i][:npair],
                    "pair_z_err": g("pair_z_err")[i][:npair],
                    "gt_xy": g("gt_xy")[i][:ng], "gt_cls": g("gt_cls")[i][:ng], "pos": g("pos")[i][:ng],
                    "ign": g("ign")[i][:ng], "hidden": g("hidden")[i][:ng]})
    return out


def _gates():
    return det.load_head_gates(GATES)[0]


_SUMM: dict = {}


def _summ(head, gate=None):
    """`summarise` (the full A9 grid -- seconds per call) cached per (head, gate) within this module."""
    if (head, gate) not in _SUMM:
        _SUMM[(head, gate)] = det.summarise(_packs(head), head) if gate is None             else det.summarise(_packs(head), head, gate=gate)
    return _SUMM[(head, gate)]


# =========================================================================== #
# A. the DEFAULT path is unchanged                                             #
# =========================================================================== #
#: computed from the TIP's ORIGINAL detection_metrics.py (ae58018) on the same 118 windows, BEFORE this change
GOLD = {
    "box3d": {'n_keys': 274, 'keys_sha256': 'da71665ba4bb4b16e634bfb5d56ae51c83449f891f41a5e21ba19cbe1f203770', 'metric_keys_sha256': '409644bc94de0459e2780047a9e547f482061818c6d71ebf8c4cbea24b909345', 'values': {'eval_box3d_n_pos': 367.0, 'eval_box3d_n_conf': 2.0, 'eval_box3d_tp@gate': 2.0, 'eval_box3d_conf_ratio': 0.005449591280653951, 'eval_box3d_prec@gate': 1.0, 'eval_box3d_rec@gate': 0.005449591280653951, 'eval_box3d_f1@gate': 0.010840108401084009, 'eval_box3d_conf_ratio_alarm': 1.0, 'eval_box3d_n_windows': 118.0, 'eval_box3d_cls_acc_tp': 1.0, 'eval_box3d_centre_err_p50': 0.7270494103431702, 'eval_box3d_det_ap4_all_all': 0.3343630944873319, 'eval_box3d_ap2m': 0.2581788772158131, 'eval_box3d_auroc_matched': 0.9090462220779086}, 'values_sha256': '80116a8acb3ee5754055d5cd205f4994e9a7992847de31cd1b05678c15d84d9c'},
    "agent": {'n_keys': 274, 'keys_sha256': 'e929d4bb6bdaac2cc9e33f4c88a850a37961e452c05ee34ff9547ab3a6a3f61c', 'metric_keys_sha256': '78858ebbc078e466b27a1f9dc61bc15d1e1f81ec9eea37cf0e6eeb76bacf1e53', 'values': {'eval_agent_n_pos': 367.0, 'eval_agent_n_conf': 0.0, 'eval_agent_tp@gate': 0.0, 'eval_agent_conf_ratio': 0.0, 'eval_agent_prec@gate': None, 'eval_agent_rec@gate': 0.0, 'eval_agent_f1@gate': None, 'eval_agent_conf_ratio_alarm': 1.0, 'eval_agent_n_windows': 118.0, 'eval_agent_cls_acc_tp': None, 'eval_agent_centre_err_p50': 0.571647047996521, 'eval_agent_det_ap4_all_all': 0.1983079485085372, 'eval_agent_ap2m': 0.15353515335820672, 'eval_agent_auroc_matched': 0.8744355551803722}, 'values_sha256': '611eda13c0d926a3fcb27a6c9c1fee7c11f3c41ab30559c6d6c3d83cfe246d0f'},
}


def _assert_default_summarise_is_the_tips(head, summary_fn) -> None:
    s = summary_fn(head)
    g = GOLD[head]
    keys = sorted(s)
    assert len(keys) == g["n_keys"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == g["keys_sha256"]
    assert hashlib.sha256(json.dumps(
        [[k, None if (isinstance(s[k], float) and s[k] != s[k]) else repr(s[k])] for k in keys]).encode()
    ).hexdigest() == g["values_sha256"]
    for k, v in g["values"].items():
        got = s[k]
        assert (got != got) if v is None else got == pytest.approx(v, rel=1e-12), k
    assert hashlib.sha256("\n".join(det.metric_keys(head)).encode()).hexdigest() == g["metric_keys_sha256"]


@pytest.mark.parametrize("head", ["box3d", "agent"])
def test_A_the_declared_gate_eval_keys_are_the_tips_value_for_value(head):
    _assert_default_summarise_is_the_tips(head, _summ)
    assert det.gated_census_keys(_packs(head), head, None) == {}              # a default run emits no new key
    assert det.resolve_gate(head, None) == 0.5 == sp.DETECTION_GATE


def test_A_DELIBERATE_REGRESSION_a_changed_default_gate_goes_RED():
    """The default silently moved to the TRAIN gate: the tip's golden must fail."""
    def moved(head):
        return _summ(head, GATE[head])
    with pytest.raises(AssertionError):
        _assert_default_summarise_is_the_tips("box3d", moved)


# =========================================================================== #
# B. ACCEPTANCE on real per-window packs vs the diagnostics' own literals      #
# =========================================================================== #
#: (n_conf, tp, n_pos, conf_ratio, prec, rec) computed by the DIAGNOSTICS' procedure on exactly these 118 windows
LIT = {'box3d': {'n_conf': 338, 'tp': 109, 'n_pos': 367, 'conf_ratio': 0.9209809264305178, 'prec': 0.3224852071005917, 'rec': 0.2970027247956403}, 'agent': {'n_conf': 395, 'tp': 84, 'n_pos': 367, 'conf_ratio': 1.0762942779291553, 'prec': 0.21265822784810126, 'rec': 0.22888283378746593}}


def _assert_gated_census_matches_the_diagnostics(census_fn) -> None:
    gates = _gates()
    for head in ("box3d", "agent"):
        k = census_fn(_packs(head), head, gates)
        lit = LIT[head]
        assert k["eval_%s_gated_gate" % head] == GATE[head]
        assert k["eval_%s_gated_n_conf" % head] == lit["n_conf"], head
        assert k["eval_%s_gated_tp" % head] == lit["tp"], head
        assert k["eval_%s_gated_n_pos" % head] == lit["n_pos"], head
        assert k["eval_%s_gated_conf_ratio" % head] == pytest.approx(lit["conf_ratio"], rel=1e-12), head
        assert k["eval_%s_gated_prec" % head] == pytest.approx(lit["prec"], rel=1e-12), head
        assert k["eval_%s_gated_rec" % head] == pytest.approx(lit["rec"], rel=1e-12), head
        p, r = lit["prec"], lit["rec"]
        assert k["eval_%s_gated_f1" % head] == pytest.approx(2 * p * r / (p + r), rel=1e-12), head
        assert k["eval_%s_gated_conf_ratio_alarm" % head] == 0.0, head          # in [0.5, 1.5] at the gate


def test_B_the_census_at_the_TRAIN_PR_gate_equals_the_diagnostics_on_real_windows():
    _assert_gated_census_matches_the_diagnostics(det.gated_census_keys)
    assert sorted(det.gated_census_keys(_packs("box3d"), "box3d", _gates())) == sorted(det.gated_key_names("box3d"))


def test_B_the_declared_alarm_fires_where_the_gated_one_is_clear_and_the_scalar_gate_agrees():
    """The diagnostics' headline: at the declared 0.5 gate the A10 alarm fires on BOTH heads (box3d 2 of 367, agent 0);
    at the TRAIN P = R gate conf_ratio is in band (0.92 / 1.08 on this subset)."""
    for head, declared_cr in (("box3d", 2 / 367), ("agent", 0.0)):
        s = _summ(head)
        assert s["eval_%s_conf_ratio" % head] == pytest.approx(declared_cr, rel=1e-12)
        assert s["eval_%s_conf_ratio_alarm" % head] == 1.0
        k = det.gated_census_keys(_packs(head), head, _gates())
        assert k["eval_%s_gated_conf_ratio_alarm" % head] == 0.0
        # the pre-existing scalar `gate=` API is the SAME census (two spellings, one number)
        s2 = _summ(head, GATE[head])
        assert s2["eval_%s_conf_ratio" % head] == k["eval_%s_gated_conf_ratio" % head]
        assert s2["eval_%s_n_conf" % head] == k["eval_%s_gated_n_conf" % head]


def test_B_DELIBERATE_REGRESSION_a_gate_that_is_not_applied_or_the_wrong_head_goes_RED():
    def ignores_the_gate(packs, head, gates):
        return det.gated_census_keys(packs, head, {h: 0.5 for h in det.HEADS})          # the old 0.5 behaviour
    with pytest.raises(AssertionError):
        _assert_gated_census_matches_the_diagnostics(ignores_the_gate)

    def swapped(packs, head, gates):
        other = "agent" if head == "box3d" else "box3d"
        out = det.gated_census_keys(packs, head, {head: gates[other], other: gates[head]})
        return out
    with pytest.raises(AssertionError):
        _assert_gated_census_matches_the_diagnostics(swapped)


# =========================================================================== #
# C. the shipped gates file and the loader                                     #
# =========================================================================== #
def test_C_the_shipped_gates_are_the_diagnostics_TRAIN_PR_gates_with_provenance():
    gates, stamp = det.load_head_gates(GATES)
    assert gates == GATE
    d = json.loads(GATES.read_text(encoding="utf-8"))
    pv = d["provenance"]
    assert d["schema"] == det.GATES_SCHEMA == "tanitad.det_presence_gates/1"
    assert pv["fit_split"] == "TRAIN" and pv["n_windows"] == 1112 and pv["checkpoint_step"] == 50400
    assert pv["source_file"].endswith("2026-10-04-refcv7-map-box-diagnostics/raw/B_box.json")
    assert pv["source_md5"] == "9cde687bcf52897936011565db79cce1"
    assert pv["sensitivity_gate_from_train_calib256"] == {"box3d": 0.2597883343696594, "agent": 0.23583702743053436}
    assert stamp["sha256"] == hashlib.sha256(GATES.read_bytes()).hexdigest() and stamp["gates"] == GATE


def _gates_doc(tmp_path, **edits):
    d = json.loads(GATES.read_text(encoding="utf-8"))
    for k, v in edits.items():
        d[k] = v
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_C_the_loader_and_resolve_gate_refuse_what_cannot_be_read_at_a_gate(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        det.load_head_gates(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="schema"):
        det.load_head_gates(_gates_doc(tmp_path, schema="x/1"))
    with pytest.raises(ValueError, match="one gate per head"):
        det.load_head_gates(_gates_doc(tmp_path, gates={"box3d": 0.25}))          # a half-configured pair
    with pytest.raises(ValueError, match="one gate per head"):
        det.load_head_gates(_gates_doc(tmp_path, gates={"box3d": 0.25, "agent": 0.2, "bev": 0.1}))
    for bad in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="probability in"):
            det.load_head_gates(_gates_doc(tmp_path, gates={"box3d": bad, "agent": 0.2}))
    with pytest.raises(ValueError, match="not in"):
        det.resolve_gate("bev", None)
    with pytest.raises(ValueError, match="no gate for head"):
        det.resolve_gate("agent", {"box3d": 0.25})
    assert det.resolve_gate("box3d", {"box3d": 0.25, "agent": 0.2}) == 0.25


# =========================================================================== #
# D. the PLANNER-side path is untouched                                        #
# =========================================================================== #
def _slots(logits):
    n = len(logits)
    g = torch.Generator().manual_seed(1)
    return {"presence_logit": torch.tensor([logits], dtype=torch.float32),
            "cls_logits": torch.randn(1, n, ra.N_AGENT_CLASSES, generator=g),
            "box": torch.tensor([[[10.0 + 5 * i, 0.5 * i, 4.5, 1.9] for i in range(n)]]),
            "yaw_vec": torch.tensor([[[0.0, 1.0]] * n]),
            "rates": torch.zeros(1, n, 3)}


def _embed(hard):
    torch.manual_seed(0)
    return ra.AgentTokenEmbed(d_out=24, cfg=ra.AgentSeamConfig(d_model=32, presence_hard=hard, presence_gate=0.5))


def _bytes(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


#: logits -2 / -0.5 / +0.5 -> presence 0.119 / 0.378 / 0.622. The planner's OWN hard gate (0.5) pads the first two; the
#: box3d detection gate (0.2567) would pad only the first.
LOGITS = [-2.0, -0.5, 0.5]


def test_D_the_planner_embed_is_bit_identical_after_every_gate_api_and_never_calls_into_them(monkeypatch):
    slots = _slots(LOGITS)
    soft, hard = _embed(False), _embed(True)
    with torch.no_grad():
        t0, p0 = soft(slots)
        h0, hp0 = hard(slots)
    assert p0.tolist() == [[False, False, False]]                     # soft: only structurally absent slots are padding
    assert hp0.tolist() == [[True, True, False]]                      # hard: the planner's OWN presence_gate = 0.5
    gates = _gates()
    det.gated_census_keys(_packs("agent"), "agent", gates)
    det.train_row_keys(_packs("agent"), "agent")
    with torch.no_grad():
        t1, p1 = soft(slots)
        h1, hp1 = hard(slots)
    assert _bytes(t0) == _bytes(t1) and _bytes(h0) == _bytes(h1)
    assert p0.tolist() == p1.tolist() and hp0.tolist() == hp1.tolist() == [[True, True, False]]
    assert sp.DETECTION_GATE == 0.5 and det.resolve_gate("agent", None) == 0.5
    # the planner never reaches the gate machinery: with all of it removed the embed still runs, identically
    for name in ("resolve_gate", "gated_census_keys", "load_head_gates", "_gate"):
        monkeypatch.setattr(det, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("planner called " + name)))
    with torch.no_grad():
        t2, _ = soft(slots)
        h2, hp2 = hard(slots)
    assert _bytes(t0) == _bytes(t2) and _bytes(h0) == _bytes(h2) and hp2.tolist() == [[True, True, False]]


def test_D_refc_agents_does_not_reference_the_detection_gate_machinery():
    src = Path(ra.__file__).read_text(encoding="utf-8")
    for needle in ("detection_metrics", "det_presence_gates", "gated_census", "load_head_gates", "resolve_gate"):
        assert needle not in src, needle


def test_D_DELIBERATE_REGRESSION_re_gating_the_planner_by_the_detection_gate_goes_RED(monkeypatch):
    """Reintroduce the forbidden coupling: the planner's hard mask reads the DETECTION gate. The pinned mask
    ([True, True, False] at the planner's own 0.5) must change -- and so this test must fail."""
    slots = _slots(LOGITS)
    hard = _embed(True)
    orig = ra.AgentTokenEmbed.forward

    def regated(self, s):
        tok, pad = orig(self, s)
        return tok, torch.sigmoid(s["presence_logit"]) < det.resolve_gate("box3d", _gates())
    monkeypatch.setattr(ra.AgentTokenEmbed, "forward", regated)
    with torch.no_grad():
        _, pad = hard(slots)
    with pytest.raises(AssertionError):
        assert pad.tolist() == [[True, True, False]]
    assert pad.tolist() == [[True, False, False]]                      # what the coupling would have done
