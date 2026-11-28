"""Tests of stopperc_check.py: argv pins, every arm-A gate clause with a deliberate-regression arm (the checker must FAIL), one pass on a well-formed record, the NULL reuse command, no CR bytes.
Run (dev box, CPU):  CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_stopperc_check.py -q -p no:cacheprovider --basetemp <existing parent>/b"""
import copy
import json
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
import stopperc_check as C  # noqa: E402

B = C.B
DAG = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
ARGVS = C.frozen_argvs()
A0 = list(ARGVS["A_s0"])
W = "3.4567"


def a_toks(tag="A_s0", w=W):
    t = list(ARGVS[tag])
    t[t.index(C.WPH)] = w
    return t


def good_a():
    """a well-formed arm-A selfcheck fragment: only the keys the A clauses read"""
    return {"levers": {"stop_aux": "on", "w_stop_aux": float(W), "stop_vision_only_source": True, "params": {"STOPAUX": 132098}},
            "info": {"init_ckpt": {"fresh_keys": list(C.STOP_KEYS), "declared_new_keys": list(C.STOP_KEYS)}},
            "stop_aux_extrinsics": {"path": C.TABLE_B2, "sha256": C.TABLE_SHA256, "coverage_train": 1.0, "n_train_clips": 4512, "n_train_clips_with_extrinsics": 4512},
            "stop_aux_probe": {"vision_only": {"source_problems": [], "keys_read": []}, "probe_n_has": 40, "probe_n_pos": 3, "step0_stop_cell": 0.69, "step0_stop_pres": 0.7, "term": 1.39,
                               "contribution": 4.8, "w_x_term": 4.8, "head_grad_diff": 0.5, "pres_head_grad_diff": 0.2, "adapter_grad_diff": 0.3, "head_grad_at_w0_zero_control": 0.0,
                               "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_allfalse_term": 0.0, "control_allfalse_total_diff": 0.0,
                               "control_allfalse_head_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "w_stop_aux_for_goal_equal": 3.45671},
            "live": {"terms": {"stop_aux": 1.39, "stop_cell": 0.69, "stop_pres": 0.7}, "unexpected_dead": []}}


def mut(path, value):
    sc = good_a()
    d = sc
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return sc


# ---- argvs (priority 1) -------------------------------------------------------------------------------------------------------------------------------
def test_frozen_argv_file_is_pinned():
    assert B.sha_file(C.ARGVS_FILE) == C.ARGVS_SHA256


def test_argv_diff_from_template_is_exactly_the_declared_flags():
    bal = {l.split(": ", 1)[0]: l.rstrip("\n").split(": ", 1)[1].split(" ") for l in open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/box2/argvs_stopbal.txt", encoding="utf-8") if l.strip()}
    dag = {k: json.load(open(f"{DAG}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))["argv"] for k in (0, 1)}

    def diff(a, b):
        return [n for n in range(max(len(a), len(b))) if n >= len(a) or n >= len(b) or a[n] != b[n]]
    out = lambda t: t.index("--out") + 1                                     # noqa: E731
    n0, rep2 = ARGVS["NULL_s0"], ARGVS["NULL_rep2"]
    assert diff(n0, bal["BAL-NULL_s0"]) == [out(n0)]                         # NULL_s0 = BAL-NULL_s0 verbatim except --out
    assert diff(rep2, n0) == [n0.index("--seed") + 1, out(n0)] and rep2[rep2.index("--seed") + 1] == "3"
    assert diff(bal["BAL-NULL_rep"], bal["BAL-NULL_s0"]) == [n0.index("--seed") + 1, out(n0)] and bal["BAL-NULL_rep"][n0.index("--seed") + 1] == "2"   # how BAL-NULL_rep differs: --seed 2 (+ --out)
    for k, tag in ((0, "A_s0"), (1, "A_s1")):
        a = ARGVS[tag]
        assert diff(a[:len(dag[k])], dag[k]) == [out(a)]
        assert a[len(dag[k]):] == ["--stop-aux", "on", "--w-stop-aux", C.WPH, "--stop-aux-extr", C.TABLE_B2]
    for tag, k in (("NULL_s1", 1), ("NULL_rep", 0)):
        assert diff(ARGVS[tag], bal["BAL-" + tag]) == [out(ARGVS[tag])]


def test_placeholder_is_refused_and_w_is_filled_from_env(monkeypatch):
    assert C.argv_problems("A_s0", ARGVS["A_s0"])                            # W_PLACEHOLDER is not a filled W
    monkeypatch.setenv("STOPPERC_W", W)
    assert C.frozen_argvs()["A_s0"][-3] == W and C.argv_problems("A_s0", C.frozen_argvs()["A_s0"]) == []
    monkeypatch.setenv("STOPPERC_W", "3.4")
    with pytest.raises(SystemExit):
        C.frozen_argvs()


def test_null_arms_carry_no_stop_flag_and_a_extra_flag_is_refused():
    assert C.argv_problems("NULL_s0", ARGVS["NULL_s0"]) == [] and C.argv_problems("NULL_rep2", ARGVS["NULL_rep2"]) == []
    assert C.argv_problems("NULL_s0", ARGVS["NULL_s0"] + ["--stop-aux", "on"])
    assert C.argv_problems("A_s0", a_toks(w=W)[:-6] + ["--stop-aux", "on", "--w-stop-aux", W, "--stop-aux-extr", "D:\\elsewhere.json"])
    assert C.argv_problems("A_s0", a_toks() + ["--stop-aux-extr", C.TABLE_B2])


def test_run_order_and_conditional_arms():
    assert C.RUN_ORDER == ["NULL_s0", "NULL_rep2", "A_s0", "A_s1"]
    assert C.TABLE["NULL_s1"]["blocked"] and C.TABLE["NULL_rep"]["blocked"]


# ---- arm A clauses: one pass, then one deliberate regression per clause ------------------------------------------------------------------------------------
def test_pass_on_a_well_formed_record():
    assert C.a_problems("A_s0", a_toks(), good_a()) == []


REGRESSIONS = [
    ("DVB1 stop_aux not built", ["levers", "stop_aux"], "off", "G-DVB1"),
    ("DVB1 W lever != argv", ["levers", "w_stop_aux"], 3.0, "G-DVB1"),
    ("DVB2 params", ["levers", "params", "STOPAUX"], 0, "G-DVB2"),
    ("DVB3 vision source", ["levers", "stop_vision_only_source"], False, "G-DVB3"),
    ("DVB3 keys_read", ["stop_aux_probe", "vision_only", "keys_read"], ["nav"], "G-DVB3"),
    ("DVB4 wrong table sha", ["stop_aux_extrinsics", "sha256"], "0" * 64, "G-DVB4"),
    ("DVB4 coverage 0.98", ["stop_aux_extrinsics", "coverage_train"], 0.98, "G-DVB4"),
    ("DVB4 n_train_clips", ["stop_aux_extrinsics", "n_train_clips"], 4000, "G-DVB4"),
    ("DVB4 path", ["stop_aux_extrinsics", "path"], "D:\\tanitad\\other.json", "G-DVB4"),
    ("DVB5 fresh keys", ["info", "init_ckpt", "fresh_keys"], list(C.STOP_KEYS)[:6], "G-DVB5"),
    ("LIVE1 term nan", ["stop_aux_probe", "term"], float("nan"), "G-LIVE1"),
    ("LIVE1 term zero", ["stop_aux_probe", "term"], 0.0, "G-LIVE1"),
    ("LIVE2 head grad 0", ["stop_aux_probe", "head_grad_diff"], 0.0, "G-LIVE2"),
    ("LIVE2 adapter grad 0", ["stop_aux_probe", "adapter_grad_diff"], 0.0, "G-LIVE2"),
    ("LIVE3 w0 control", ["stop_aux_probe", "head_grad_at_w0_zero_control"], 1e-9, "G-LIVE3"),
    ("LIVE4 mutation: total line deleted", ["stop_aux_probe", "contribution"], 0.0, "G-LIVE4"),
    ("LIVE5 control repeat", ["stop_aux_probe", "control_repeat_grad_diff"], 1e-7, "G-LIVE5"),
    ("LIVE5 control allfalse", ["stop_aux_probe", "control_allfalse_term"], 0.1, "G-LIVE5"),
    ("LIVE5 probe_n_pos 0", ["stop_aux_probe", "probe_n_pos"], 0, "G-LIVE5"),
    ("LIVE5 n_has < n_pos", ["stop_aux_probe", "probe_n_has"], 1, "G-LIVE5"),
    ("LIVE6 W mismatch", ["stop_aux_probe", "w_stop_aux_for_goal_equal"], 3.9, "G-LIVE6"),
    ("LIVE7 term missing", ["live", "terms"], {"stop_aux": 1.0}, "G-LIVE7"),
    ("LIVE7 unexpected dead", ["live", "unexpected_dead"], ["stop_head.0.bias"], "G-LIVE7"),
]


@pytest.mark.parametrize("name,path,value,clause", REGRESSIONS, ids=[r[0] for r in REGRESSIONS])
def test_each_clause_turns_the_checker_red(name, path, value, clause, monkeypatch):
    monkeypatch.setattr(C, "TABLE_SHA256", C.TABLE_SHA256)
    got = C.a_problems("A_s0", a_toks(), mut(path, value))
    assert any(clause in x for x in got), (name, got)


def test_table_sha_hook_and_pass_after_override(monkeypatch):
    monkeypatch.setenv("STOPPERC_TABLE_SHA", "1" * 64)
    assert any("G-DVB4" in x for x in C.a_problems("A_s0", a_toks(), good_a()))


def test_stop_key_in_an_off_arm_selfcheck_is_refused():
    sc = {"levers": {"params": {}}, "info": {}}
    assert C.null_problems("NULL_s0", ARGVS["NULL_s0"], sc) == []
    for bad in ({"stop_aux_probe": {}}, {"levers": {"stop_aux": "on"}}, {"levers": {"params": {"STOPAUX": 1}}}):
        sc2 = copy.deepcopy(sc)
        for k, v in bad.items():
            sc2.setdefault(k, {}).update(v) if isinstance(v, dict) else sc2.__setitem__(k, v)
        assert any("G-DVB5" in x for x in C.null_problems("NULL_s0", ARGVS["NULL_s0"], sc2)), bad
    sc3 = {"info": {"init_ckpt": {"fresh_keys": ["stop_head.0.weight"]}}}
    assert any("fresh stop keys" in x for x in C.null_problems("NULL_s0", ARGVS["NULL_s0"], sc3))


def test_real_off_record_has_no_stop_keys():
    p = "C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/selfcheck.json"
    if os.path.exists(p):
        assert C.stop_keys_anywhere(json.load(open(p, encoding="utf-8"))) == []


def test_gate_shows_the_base_the_architecture_without_the_head(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "_base_gate", lambda tag, toks, sc: seen.update(sc=sc) or [])
    sc = good_a()
    sc["levers"]["n_params_trainable"] = 7214495 + 132098
    assert C.gate_problems("A_s0", a_toks(), sc) == []
    assert "STOPAUX" not in seen["sc"]["levers"]["params"] and seen["sc"]["levers"]["n_params_trainable"] == 7214495 and seen["sc"]["info"]["init_ckpt"]["declared_new_keys"] == []
    assert sc["levers"]["params"]["STOPAUX"] == 132098                          # the original record is untouched


def test_base_checker_is_pinned(monkeypatch):
    C.base_ok()
    monkeypatch.setattr(C, "BASE_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        C.base_ok()


# ---- NULL reuse comparison (dev box) -----------------------------------------------------------------------------------------------------------------------------
def _ck(path, val):
    import torch
    torch.save({"model": {"w": torch.full((3,), val)}, "argv_sha": str(val)}, path)


def test_reuse_equal_and_differs(tmp_path, monkeypatch):
    a, bal, c = str(tmp_path / "a.pt"), str(tmp_path / "bal.pt"), str(tmp_path / "c.pt")
    _ck(a, 1.0), _ck(bal, 1.0), _ck(c, 2.0)
    monkeypatch.setattr(C, "BAL_NULL_S0_CKPT_SHA256", B.sha_file(bal))
    assert C.cmd_reuse({"ckpt": a, "bal-ckpt": bal, "out": str(tmp_path / "r1.json")}) == 0
    r1 = json.load(open(tmp_path / "r1.json"))
    assert r1["null_reuse"] == "EQUAL" and r1["file_sha_equal"] is False       # tensors equal although the files differ (argv_sha embedded): the tensor basis is the test
    assert C.cmd_reuse({"ckpt": c, "bal-ckpt": bal, "out": str(tmp_path / "r2.json")}) == 0 and json.load(open(tmp_path / "r2.json"))["null_reuse"] == "DIFFERS"
    monkeypatch.setattr(C, "BAL_NULL_S0_CKPT_SHA256", "0" * 64)               # a bal checkpoint that is not the pinned one is not-computed, never a pass
    assert C.cmd_reuse({"ckpt": a, "bal-ckpt": bal, "out": str(tmp_path / "r3.json")}) == 2 and json.load(open(tmp_path / "r3.json"))["null_reuse"] == "UNKNOWN"


def test_no_cr_bytes_in_shipped_files():
    for root, _, fs in os.walk(PKG):
        for f in fs:
            if f.endswith((".py", ".sh", ".txt", ".md")) and "__pycache__" not in root:
                assert b"\r" not in open(os.path.join(root, f), "rb").read(), os.path.join(root, f)


# ---- G-LIVE8 eval build (MM 2026-11-13: A_s0 failed here on the real box; the dry run skips this check) -------------------------------------------------------
_STUB = '''import torch
K30 = 3
class P10Prior:
    def __init__(self, mu, sd, flat):
        self.mu, self.sd, self.flat = mu, sd, flat
class V8FMiniConfig:
    def __init__(self, stop_aux="off", w_stop_aux=0.0):
        self.stop_aux, self.w_stop_aux = stop_aux, w_stop_aux
class V8FMini(torch.nn.Module):
    def __init__(self, cfg, prior, cons_mu=None, cons_sd=None):   # the signature of 3e264bbf v8f_mini.py:707: the prior is required
        super().__init__()
        assert isinstance(prior, P10Prior) and prior.flat.shape == (2 * K30,)
        self.trunk = torch.nn.Linear(2, 2)
        if cfg.stop_aux == "on":
            self.stop_head = torch.nn.Linear(2, 1)
'''


def _stub_tree(tmp_path, monkeypatch):
    import torch
    d = tmp_path / "repo" / "stack" / "tanitad" / "models"
    d.mkdir(parents=True)
    (d.parent / "__init__.py").write_text("")
    (d / "__init__.py").write_text("")
    (d / "v8f_mini.py").write_text(_STUB)
    for m in ("tanitad", "tanitad.models", "tanitad.models.v8f_mini"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(B, "repo", lambda: str(tmp_path / "repo"))
    ns = {}
    exec(_STUB, ns)
    on = ns["V8FMini"](ns["V8FMiniConfig"]("on", 0.0533), ns["P10Prior"](torch.zeros(8), torch.ones(8), torch.zeros(6)))
    return {"cfg": {"stop_aux": "on", "w_stop_aux": 0.0533}, "model": dict(on.state_dict())}


def test_eval_build_passes_a_well_formed_on_checkpoint_with_the_loaders_prior(tmp_path, monkeypatch):
    ck = _stub_tree(tmp_path, monkeypatch)
    assert C.eval_build_problems(ck) == []


def test_eval_build_red_arm_a_missing_head_tensor_is_refused(tmp_path, monkeypatch):
    ck = _stub_tree(tmp_path, monkeypatch)
    ck["model"].pop("stop_head.bias")
    p = C.eval_build_problems(ck)
    assert len(p) == 1 and p[0].startswith("G-LIVE8") and "stop_head.bias" in p[0]


# ---- G-LIVE6 scope (MM 2026-11-13: the real A_s1 gate refused W 0.0533 because its own step-0 ratio read 0.06279; the card fixes ONE W at the smoke) ----------
def test_live6_compares_w_on_the_smoke_arm_only():
    assert C.WSMOKE_TAG == "A_s0"
    s1 = mut(["stop_aux_probe", "w_stop_aux_for_goal_equal"], 0.06278651564882848)        # the real A_s1 reading
    assert not any("G-LIVE6" in x for x in C.a_problems("A_s1", a_toks("A_s1"), s1))
    assert any("G-LIVE6" in x for x in C.a_problems("A_s0", a_toks("A_s0"), s1))           # on the smoke's arm a different ratio is still refused


def test_live6_other_seed_still_needs_a_finite_positive_reading():
    for bad in (float("nan"), 0.0, None):
        got = C.a_problems("A_s1", a_toks("A_s1"), mut(["stop_aux_probe", "w_stop_aux_for_goal_equal"], bad))
        assert any("G-LIVE6" in x for x in got), (bad, got)
    assert C.a_problems("A_s1", a_toks("A_s1"), good_a()) == []
