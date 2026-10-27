"""Mutation proof for the T-EGOCUT lever (PREREG_V8F_TEGOCUT, claim row H-V8F-TEGOCUT-1): one mutant per rule, each must turn ``stack/tests/test_v8f_tegocut.py`` RED.

The three the PI's brief names are M01 (delete the single line that swaps the buffer in), M02 (default the flag ON) and M04 (make c trainable); the draft's M1-M7 are M01 (M1), M05 (M2: c from the live model, caught by the resume test),
M04 (M3), M06 (M4: the constant applied in run_s instead of run_t), W02 (M5: ``t_ego`` dropped from the saved config), M07 (M6: one extra ``torch.rand`` in the const branch) and G04 (M7: ``bad_t_ego`` accepts const
without ``--init-ckpt``); the rest delete one rule of the model, the gate or the probe at a time.

R77: the mutants run on a COPY. This script builds its own minimal copy of the tree (``--work``) from the tree it is given, mutates ONLY the copy, restores the copy's file after every mutant, and at the end checks that every
delivered file is byte-identical to its sha256 at start (it refuses to start if ``--work`` is inside the delivery tree).

usage:  python mutate_tegocut.py <delivery tree root> <python> [--work DIR] [--check] [--list] [--only NAME[,NAME]] [--group G] [--log FILE]
  --check   build the copy, verify that every anchor is found EXACTLY ONCE in its region and every mutated text compiles; run nothing
  --list    print the mutants
Per group: the UNMUTATED copy is run green first on the group's whole selector; then each mutant of the group runs its own (narrower) selector with -x and must fail (rc 1 and pytest's own "failed"/"error" line; a collection
error or "no tests ran" is INCONCLUSIVE, not a kill). Only one pytest process runs at a time (CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=4, one basetemp under the work dir, reused). A mutant that survives is printed as SURVIVOR;
the exit code is 1 then.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import time

MODEL = "stack/tanitad/models/v8f_mini.py"
TRAIN = "stack/scripts/v8f_mini_train.py"
TEST = "stack/tests/test_v8f_tegocut.py"
PKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAWT = PKG + "/raw/2026-10-27-tegocut-train"
REF_M, REF_T = RAWT + "/pre_tegocut_v8f_mini.py.ref", RAWT + "/pre_tegocut_v8f_mini_train.py.ref"
FILES = {"model": MODEL, "train": TRAIN}
COPY_TREES = ["stack/tanitad", "taniteval/taniteval"]
COPY_FILES = ["stack/pyproject.toml", "stack/tests/conftest.py", "taniteval/conftest.py", TEST, MODEL, TRAIN, REF_M, REF_T, PKG + "/raw/eval_grid_windows.json",
              "stack/tests/test_v8f_pgt.py", "stack/tests/test_v8f_mini.py"]                       # test_v8f_pgt / test_v8f_mini: the synthetic world and batch helpers the suite loads
DELIVERED = [MODEL, TRAIN, TEST, REF_M, REF_T, RAWT + "/mutate_tegocut.py", "stack/tests/test_v8f_flat.py", "stack/tests/test_v8f_mixo.py", "stack/tests/test_v8f_oacc.py", "stack/tests/test_v8f_window_start.py"]
# regions (an anchor is searched ONLY inside its region)
R_RUN_S = ("    def run_s(", "    def run_t(")
R_BAD_T = ("def bad_t_ego(", "def bad_init_ckpt(")
R_PROBE = ("def t_ego_probe(", "def dagger_probe(")
R_BASE = ("def _t_ego_base(", "def t_ego_probe(")

# ---- selectors (pytest -k): GROUP = the union run green first; each mutant's own is a subset
K_MODEL = ("test_the_flag_exists or test_the_default_model_is_the_saved or test_const_gives or test_only_ts_ego_slot or test_e_ego_gets_no_gradient or test_the_constant_is_a_buffer or test_set_t_ego_const_validates or "
           "test_a_const_model_without_the_buffer or test_state_dict_round_trip or test_the_flag_adds_no_rng_draw or test_a_bf16 or test_the_probe_passes or test_the_loss_the_trainer or test_g_dvb_passes")
K_GDVB = ("test_the_flag_exists or test_a_flag_without_an_entry or test_g_dvb or test_the_gate_does_not_move or test_read_init_t_ego_c or test_a_refused_declaration or test_an_init_checkpoint_without or test_flag_to_config")
K_PROBE = "test_the_probe or test_the_gate_refuses_a_model_whose_t_slot or test_a_default_run"
K_E2E = ("test_a_default_run or test_flag_to_config or test_the_loss_the_trainer or test_the_gate_refuses_a_model or test_a_refused_declaration or test_an_init_checkpoint_without or test_a_resumed_const_run or "
         "test_the_eval_loaders or test_k_stream")
K_IDENT = "test_defaults_reproduce or test_the_saved_references or test_the_single_applying_lines or test_the_default_model_is_the_saved"
GROUPS = {"model": K_MODEL, "gdvb": K_GDVB, "probe": K_PROBE, "e2e": K_E2E, "ident": K_IDENT}

DISK_FULL = ("No space left", "badbit", "Errno 28", "unexpected pos", "ENOSPC")


def E(old, new, region=None):
    return dict(old=old, new=new, region=region)


def M(name, rule, file, edits, group, k):
    return dict(name=name, rule=rule, file=file, edits=edits if isinstance(edits, list) else [edits], group=group, k=k)


SWAP = 'ego_tok = self._t_ego_const(i.shape[0]) if cfg.t_ego == "const" else self._ego(self._at(b["p10f"], i))'
LIVE = 'ego_tok = self._ego(self._at(b["p10f"], i))'

MUTANTS = [
    # ---- the model: the single line, the defaults, the buffer
    M("M01_swap_line_deleted", "T's ego slot is the buffer (the ONE line in run_t); deleted, T reads the live token", "model", E(SWAP, LIVE), "model", "test_const_gives or test_the_probe_passes or test_the_loss_the_trainer or test_flag_to_config"),
    M("M02_flag_defaults_on", "--t-ego defaults to full (off = every earlier arm, bit for bit); defaulted ON", "train", E('choices=M.T_EGO_MODES, default="full",', 'choices=M.T_EGO_MODES, default="const",'), "ident", "test_the_flag_exists or test_defaults_reproduce"),
    M("M02b_config_defaults_on", "the config default is full too", "model", E('    t_ego: str = "full" ', '    t_ego: str = "const" '), "ident", "test_the_flag_exists or test_the_default_model_is_the_saved or test_defaults_reproduce"),
    M("M03_default_model_carries_a_buffer", "a default model has no t_ego_c (a None buffer: not in the state dict)", "model", E('self.register_buffer("t_ego_c", None) ', 'self.register_buffer("t_ego_c", torch.zeros(cfg.d)) '), "ident",
      "test_the_default_model_is_the_saved or test_the_constant_is_a_buffer or test_defaults_reproduce"),
    M("M04_c_is_trainable", "c is a frozen BUFFER, never a Parameter", "model", E('        self.register_buffer("t_ego_c", t)', '        self.t_ego_c = nn.Parameter(t)'), "model", "test_the_constant_is_a_buffer or test_g_dvb_passes or test_the_probe_passes"),
    M("M05_c_from_the_live_model", "c is read from the init FILE, not the live model (on a resume the live model is random)", "train", E("model.set_t_ego_const(read_init_t_ego_c(a.init_ckpt))", "model.set_t_ego_const(model.e_ego.bias.detach().clone())"), "e2e",
      "test_a_resumed_const_run"),
    M("M06_constant_in_run_s_not_run_t", "the constant replaces T's slot, not S's", "model",
      [E(SWAP, LIVE), E('self._ego(self._at(b["p10f"], i))', '(self._t_ego_const(i.shape[0]) if cfg.t_ego == "const" else self._ego(self._at(b["p10f"], i)))', R_RUN_S)], "model",
      "test_const_gives or test_only_ts_ego_slot or test_the_probe_passes"),
    M("M07_extra_rng_draw", "the const branch adds no RNG draw (K-STREAM)", "model", E("        return self.t_ego_c[None].expand(B, -1)", "        torch.rand(1)\n        return self.t_ego_c[None].expand(B, -1)"), "model",
      "test_the_flag_adds_no_rng_draw or test_k_stream"),
    M("M08_buffer_not_persistent", "t_ego_c is saved in the state dict", "model", E('        self.register_buffer("t_ego_c", t)', '        self.register_buffer("t_ego_c", t, persistent=False)'), "model",
      "test_the_constant_is_a_buffer or test_state_dict_round_trip or test_the_eval_loaders"),
    M("M09_full_model_accepts_the_constant", "a FULL model refuses a state dict that carries t_ego_c (a checkpoint whose config lost t_ego is never evaluated with the live token)", "model",
      E('        if tkey in state_dict and self.cfg.t_ego == "const":', "        if tkey in state_dict:"), "model", "test_state_dict_round_trip or test_the_eval_loaders"),
    M("M10_missing_buffer_not_refused", "a const model without the buffer REFUSES at the forward", "model", E("        if self.t_ego_c is None:\n            raise RuntimeError(\"t_ego='const' but", "        if False:\n            raise RuntimeError(\"t_ego='const' but"), "model",
      "test_a_const_model_without_the_buffer or test_state_dict_round_trip"),
    M("M11_nonfinite_constant_accepted", "the constant must be finite", "model", E('    if not bool(torch.isfinite(t).all()):\n        raise ValueError("t_ego_c must be finite")', '    if False:\n        raise ValueError("t_ego_c must be finite")'), "model",
      "test_set_t_ego_const_validates or test_state_dict_round_trip"),
    M("M12_wrong_shape_accepted", "the constant must be [d]", "model", E("    if tuple(t.shape) != (int(d),):", "    if False:"), "model", "test_set_t_ego_const_validates or test_state_dict_round_trip"),
    M("M13_set_on_a_full_model_allowed", "only a model built const takes a constant", "model", E('        if self.cfg.t_ego != "const":\n            raise ValueError(f"set_t_ego_const needs', '        if False:\n            raise ValueError(f"set_t_ego_const needs'), "model",
      "test_set_t_ego_const_validates"),
    M("M14_lever_does_not_report_t_ego", "built_levers reports what T's ego slot IS", "model", E('"t_ego": cfg.t_ego,', '"t_ego": "full",'), "model", "test_g_dvb_passes or test_flag_to_config or test_the_constant_is_a_buffer"),
    M("M15_lever_does_not_report_the_sha", "built_levers reports the constant's sha256", "model", E('"t_ego_c_sha256": None if self.t_ego_c is None else tensor_sha256(self.t_ego_c),', '"t_ego_c_sha256": None,'), "model",
      "test_the_constant_is_a_buffer or test_flag_to_config or test_g_dvb_passes"),
    M("M16_config_accepts_any_mode", "t_ego is validated in __post_init__", "model", E("        if self.t_ego not in T_EGO_MODES:", "        if False:"), "model", "test_the_flag_exists"),
    M("M17_sha_over_other_bytes", "the sha256 is over the float32 bytes", "model", E(".to(torch.float32).contiguous().numpy().tobytes()).hexdigest()", ".to(torch.float64).contiguous().numpy().tobytes()).hexdigest()"), "model",
      "test_the_constant_is_a_buffer"),
    # ---- G-DVB
    M("G01_flag_has_no_entry", "every flag has a FLAG_CHECKS entry (t_ego)", "train", E('    "t_ego": "lever:t_ego",\n', ""), "gdvb", "test_the_flag_exists or test_a_flag_without_an_entry"),
    M("G02_gate_not_called", "declared_vs_built calls bad_t_ego", "train", E("    bad += bad_t_ego(a, model, data_info) ", "    bad += [] "), "gdvb", "test_g_dvb_refuses or test_a_refused_declaration"),
    M("G03_declared_not_compared_to_built", "G-DVB: the declared --t-ego is the built one (config AND levers)", "train", E('    if lv.get("t_ego") != a.t_ego or model.cfg.t_ego != a.t_ego:', "    if False:", R_BAD_T), "gdvb",
      "test_g_dvb_refuses_what_was_not_built"),
    M("G04_const_without_init_accepted", "const needs --init-ckpt (M7)", "train", E('    if not a.init_ckpt:\n        bad.append("--t-ego const needs --init-ckpt', '    if False:\n        bad.append("--t-ego const needs --init-ckpt', R_BAD_T), "gdvb",
      "test_g_dvb_refuses_const_without or test_a_refused_declaration"),
    M("G05_exclusivity_dropped", "const is never combined with --ego-perturb-p / --dagger-p", "train", E("    if a.ego_perturb_p > 0 or a.dagger_p > 0:", "    if False:", R_BAD_T), "gdvb", "test_g_dvb_refuses_const_combined or test_a_refused_declaration"),
    M("G06_absent_buffer_accepted", "const with no buffer built is refused", "train", E('    if c is None:\n        bad.append("--t-ego const is declared but no t_ego_c', '    if False:\n        bad.append("--t-ego const is declared but no t_ego_c', R_BAD_T), "gdvb",
      "test_g_dvb_refuses_what_was_not_built"),
    M("G07_buffer_form_not_checked", "the buffer is a finite float32 [d]", "train", E("    if c.dtype != torch.float32 or tuple(c.shape) != (model.cfg.d,) or not bool(torch.isfinite(c).all()):", "    if False:", R_BAD_T), "gdvb",
      "test_g_dvb_refuses_a_buffer_that"),
    M("G08_trainable_not_checked", "c is not trainable and not in parameters()", "train", E('    if c.requires_grad or any(p is c for p in model.parameters()) or any(n.split(".")[-1] == "t_ego_c" for n, _ in model.named_parameters()):', "    if False:", R_BAD_T), "gdvb",
      "test_g_dvb_refuses_a_constant_on_a_full_arm"),
    M("G09_record_sha_not_compared", "the built constant's sha256 equals the init record's", "train", E('        elif rec["e_ego_bias_sha256"] != lv.get("t_ego_c_sha256"):', "        elif False:", R_BAD_T), "gdvb", "test_g_dvb_refuses_a_constant_that_is_not"),
    M("G10_record_missing_accepted", "the init record must carry the sha256", "train", E('        if not rec.get("e_ego_bias_sha256"):', "        if False:", R_BAD_T), "gdvb", "test_g_dvb_refuses_what_was_not_built"),
    M("G11_file_not_reread", "the gate re-reads the init FILE and compares bit for bit", "train", E("            if want.shape != c.shape or not torch.equal(want, c.detach().cpu()):", "            if False:", R_BAD_T), "gdvb",
      "test_g_dvb_refuses_a_constant_that_is_not"),
    M("G12_twin_check_dropped", "the flag adds no Parameter (twin built full)", "train", E("    if sig(model) != sig(twin):", "    if False:", R_BAD_T), "gdvb", "test_g_dvb_refuses_a_constant_on_a_full_arm"),
    M("G13_full_with_buffer_accepted", "a full arm carries no buffer", "train", E('        if c is not None:\n            bad.append("--t-ego full is declared but', '        if False:\n            bad.append("--t-ego full is declared but', R_BAD_T), "gdvb",
      "test_g_dvb_refuses_a_constant_on_a_full_arm"),
    M("G14_gate_moves_the_rng", "the twin is built inside a forked RNG", "train", E("    with torch.random.fork_rng(devices=[]):                          # the twin's", "    if True:                          # the twin's", R_BAD_T), "gdvb",
      "test_the_gate_does_not_move_the_rng_stream"),
    M("G15_init_record_has_no_sha", "load_init_ckpt records the file's e_ego.bias sha256 (const)", "train", E('        rec["e_ego_bias_sha256"] = M.tensor_sha256(sd["e_ego.bias"])', "        pass"), "gdvb", "test_flag_to_config"),
    M("G16_init_without_bias_accepted", "an init checkpoint without e_ego.bias is refused with const", "train",
      E('        if "e_ego.bias" not in sd:\n            raise ValueError("--t-ego const needs the init checkpoint\'s e_ego.bias, and this checkpoint has none")', '        if False:\n            raise ValueError("--t-ego const needs the init checkpoint\'s e_ego.bias, and this checkpoint has none")'),
      "gdvb", "test_an_init_checkpoint_without"),
    M("G17_config_drops_t_ego", "argv --t-ego becomes cfg.t_ego (make_config)", "train", E("o_interface=a.o_interface, w_acc=a.w_acc, t_ego=a.t_ego)", "o_interface=a.o_interface, w_acc=a.w_acc)"), "gdvb",
      "test_the_flag_exists or test_flag_to_config"),
    M("G18_constant_never_set", "main() sets c from the init file", "train", E("model.set_t_ego_const(read_init_t_ego_c(a.init_ckpt))", "pass"), "gdvb", "test_flag_to_config or test_the_loss_the_trainer"),
    # ---- G-LIVE: t_ego_probe
    M("P01_probe_not_run", "the in-process gate runs t_ego_probe", "train", E("        tegop = t_ego_probe(model, probe, cfg) ", '        tegop = {"pass": True} '), "probe", "test_the_gate_refuses_a_model_whose_t_slot or test_a_default_run"),
    M("P02_probe_not_recorded", "selfcheck.json carries the probe's record", "train", E('    sc["t_ego_probe"] = tegop', "    pass"), "probe", "test_a_default_run or test_flag_to_config or test_defaults_reproduce"),
    M("P03_token_clause_dropped", "(a) the ego slot read_t receives is c on every row (const)", "train", E('    if not (c_["a_token_is_c_every_row"] and not c_["a_token_is_the_live_token"]):', "    if False:", R_PROBE), "probe",
      "test_the_probe_fires_when_the_token_is_not or test_the_probe_fires_when_the_live_token"),
    M("P04_token_control_dropped", "(a) the full branch reads T's own token (the control)", "train", E('    if not (f_["a_token_is_the_live_token"] and not f_["a_token_is_c_every_row"]):', "    if False:", R_PROBE), "probe",
      "test_the_probe_fires_when_the_full_branch_is_not"),
    M("P05_invariance_clause_dropped", "(b) every T output moves by EXACTLY 0.0 under a donor p10f (const)", "train", E("        if v != 0.0:\n            raise AssertionError(f\"t_ego probe: const is declared but T's {k} moved by", "        if False:\n            raise AssertionError(f\"t_ego probe: const is declared but T's {k} moved by", R_PROBE), "probe",
      "test_the_probe_fires_when_t_depends"),
    M("P06_invariance_control_dropped", "(b) the full branch moves (the control)", "train", E('    if not any(v > 0.0 for v in f_["b_max_abs_dT_under_donor_p10f"].values()):', "    if False:", R_PROBE), "probe",
      "test_the_probe_fires_when_the_full_control_cannot_move"),
    M("P07_gradient_clause_dropped", "(c) the gradient of T's terms w.r.t. e_ego is EXACTLY 0.0 (const)", "train", E('    if c_["c_grad_T_terms_wrt_e_ego"] != 0.0:', "    if False:", R_PROBE), "probe", "test_the_probe_fires_when_the_gradient"),
    M("P08_gradient_control_dropped", "(c) the full branch has a gradient (the control)", "train", E('    if not f_["c_grad_T_terms_wrt_e_ego"] > 0.0:', "    if False:", R_PROBE), "probe", "test_the_probe_fires_when_the_full_control_cannot_move"),
    M("P09_dead_group_accepted", "(d) every T group and e_ego still trains", "train", E("        if dead:\n            raise AssertionError(f\"t_ego probe: {mode}: no gradient on", "        if False:\n            raise AssertionError(f\"t_ego probe: {mode}: no gradient on", R_PROBE), "probe",
      "test_the_probe_fires_when_a_group"),
    M("P10_s_o_token_clause_dropped", "(h) S's and O's ego tokens are their own", "train", E('        if not (r["h_s_token_is_its_own"] and r["h_o_token_is_its_own"]):', "        if False:", R_PROBE), "probe", "test_the_probe_fires_when_s_or_o"),
    M("P11_determinism_dropped", "(g) the probe is deterministic", "train", E('        if r["g_determinism_max_abs_diff"] != 0.0 or not r["g_record_identical"]:', "        if False:", R_PROBE), "probe", "test_the_probe_fires_when_it_is_not_deterministic"),
    M("P12_buffer_clause_dropped", "(e) c is a buffer, not a Parameter", "train", E('    if not c_["e_is_a_buffer_not_a_parameter"]:', "    if False:", R_PROBE), "probe", "test_the_probe_fires_when_the_constant_is_a_parameter"),
    M("P13_frozen_bytes_dropped", "(e) c's bytes do not move under three optimiser steps", "train", E('    if c_["e_max_abs_change_of_c_after_3_steps"] != 0.0:', "    if False:", R_PROBE), "probe", "test_the_probe_fires_when_the_constant_moves"),
    M("P14_keys_clause_dropped", "admissibility: the same keys under both branches, inside INFERENCE_INPUT_KEYS", "train", E('    if keys["const"] != keys["full"] or set(keys["const"]) - set(M.INFERENCE_INPUT_KEYS):', "    if False:", R_PROBE), "probe",
      "test_the_probe_fires_when_the_branches_read"),
    M("P15_blind_const_probe_accepted", "a const launch whose probe is blind is refused", "train", E("    if blind:\n        if strict:", "    if blind:\n        if False:", R_PROBE), "probe", "test_a_single_window"),
    M("P16_single_window_const_accepted", "a const launch with a single-window probe batch is refused", "train", E("    if B < 2:\n        if not strict:", "    if B < 2:\n        if True:", R_PROBE), "probe", "test_a_single_window"),
    M("P17_const_branch_not_flipped", "the probe's branches flip cfg.t_ego (both run in every gate)", "train", E("    m.cfg.t_ego = mode\n", "    pass\n", R_BASE), "probe", "test_the_probe_passes"),
    # ---- the caller / the checkpoint
    M("W01_saved_config_drops_t_ego", "the checkpoint carries cfg.t_ego (M5: a dropped key makes the strict load refuse the buffer)", "train", E('"cfg": cfg.__dict__,', '"cfg": {k_: v_ for k_, v_ in cfg.__dict__.items() if k_ != "t_ego"},'), "e2e",
      "test_the_eval_loaders or test_flag_to_config or test_defaults_reproduce"),
    M("W02_resume_loses_c_check", "a resumed run holds the file's constant (the gate re-reads the file)", "train", E("            if want.shape != c.shape or not torch.equal(want, c.detach().cpu()):", "            if False:", R_BAD_T), "e2e", "test_a_resumed_const_run or test_g_dvb_refuses_a_constant_that_is_not"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_copy(root, work):
    if os.path.commonpath([os.path.abspath(root), os.path.abspath(work)]) == os.path.abspath(root):
        raise SystemExit(f"REFUSED (R77): --work {work} is inside the delivery tree {root}")
    shutil.rmtree(work, ignore_errors=True)
    for t in COPY_TREES:
        shutil.copytree(os.path.join(root, t), os.path.join(work, t), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results", "tests"))
    for f in COPY_FILES + ["stack/scripts/" + n for n in os.listdir(os.path.join(root, "stack/scripts")) if n.endswith(".py")]:
        if os.path.exists(os.path.join(root, f)):
            os.makedirs(os.path.dirname(os.path.join(work, f)), exist_ok=True)
            shutil.copyfile(os.path.join(root, f), os.path.join(work, f))


def region_bounds(text, region):
    if region is None:
        return 0, len(text)
    a = text.index(region[0])
    b = text.index(region[1], a + 1)
    return a, b


def apply(text, m):
    for e in m["edits"]:
        a, b = region_bounds(text, e["region"])
        seg = text[a:b]
        n = seg.count(e["old"])
        if n != 1:
            raise SystemExit(f"ANCHOR for {m['name']} found {n} times (need exactly 1) in {'its region' if e['region'] else 'the file'}: {e['old'][:90]!r}")
        text = text[:a] + seg.replace(e["old"], e["new"]) + text[b:]
    return text


def run_pytest(py, work, k, x, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4"}
    for _ in range(720):                                                                         # one pytest process at a time: wait for a slot (< 2 runs active across the machine)
        r = subprocess.run([py, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True)
        if r.stdout.strip().isdigit() and int(r.stdout.strip()) < 2:
            break
        time.sleep(30)
    os.makedirs(os.path.join(work, "bt"), exist_ok=True)                                        # pytest --basetemp needs its parent to exist
    bt = os.path.join(work, "bt", "cur")                                                        # ONE basetemp, reused: pytest clears it at the start of every run
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", bt, TEST, "-k", k] + (["-x"] if x else [])
    t = time.time()
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=3 * 3600)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    log(f"    rc={p.returncode} {tail}  [{time.time() - t:.0f}s]")
    return p.returncode, p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("python")
    ap.add_argument("--work", default="E:/Projects/TanitAD-artifacts/egocut_build/mut")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--log", default="")
    a = ap.parse_args(argv)
    muts = [m for m in MUTANTS if (not a.only or m["name"] in a.only.split(",") or any(m["name"].startswith(x) for x in a.only.split(","))) and (not a.group or m["group"] == a.group)]
    if a.list:
        for m in muts:
            print(f"{m['name']:42s} [{m['group']}] {m['rule']}")
        print(len(muts), "mutants")
        return 0
    lf = open(a.log, "a", encoding="utf-8") if a.log else None

    def log(s):
        print(s, flush=True)
        if lf:
            lf.write(s + "\n")
            lf.flush()
    before = {f: sha(os.path.join(a.root, f)) for f in DELIVERED if os.path.exists(os.path.join(a.root, f))}
    for f, h in before.items():
        log(f"BEFORE {h} {f}")
    build_copy(a.root, a.work)
    pristine = {k: open(os.path.join(a.work, f), encoding="utf-8").read() for k, f in FILES.items()}
    names = [m["name"] for m in MUTANTS]
    assert len(names) == len(set(names)), "duplicate mutant names"
    for m in MUTANTS:                                                                            # every anchor exactly once in its region, every mutated text compiles
        ast.parse(apply(pristine[m["file"]], m))
    log(f"{len(MUTANTS)} mutants, each anchor found exactly once in its region, each mutated text compiles")
    if a.check:
        return 0
    killed, survivors = [], []
    py = a.python
    for g, gk in GROUPS.items():
        gm = [m for m in muts if m["group"] == g]
        if not gm:
            continue
        log(f"== group {g}: baseline (unmutated copy) on  -k \"{gk[:120]}...\"")
        rc, out = run_pytest(py, a.work, gk, False, log)
        if rc != 0:
            log(f"BASELINE RED for group {g}: no mutant of this group is judged")
            log(out[-3000:])
            survivors += [m["name"] + " (baseline red)" for m in gm]
            continue
        for m in gm:
            path = os.path.join(a.work, FILES[m["file"]])
            open(path, "w", encoding="utf-8", newline="\n").write(apply(pristine[m["file"]], m))
            log(f"  {m['name']}: {m['rule']}")
            try:
                rc, out = run_pytest(py, a.work, m["k"], True, log)
            finally:
                open(path, "w", encoding="utf-8", newline="\n").write(pristine[m["file"]])
            if rc == 1 and any(x in out for x in DISK_FULL):                                         # a test that died because the DISK was full proves nothing about the rule
                survivors.append(m["name"])
                log(f"    SURVIVOR {m['name']}  (rc=1 but the output shows a full disk: INCONCLUSIVE, not a test failure)")
            elif rc == 1 and (" failed" in out or " error" in out):                                  # pytest's own "tests failed" exit; a collection error (2) or "no tests ran" (5) is NOT a kill
                killed.append(m["name"])
                log(f"    KILLED {m['name']}")
            else:
                survivors.append(m["name"])
                log(f"    SURVIVOR {m['name']}  (rc={rc}: {'INCONCLUSIVE, not a test failure' if rc in (2, 5) else 'the suite stayed green'})")
    for k, f in FILES.items():
        assert open(os.path.join(a.work, f), encoding="utf-8").read() == pristine[k]
    after = {f: sha(os.path.join(a.root, f)) for f in before}
    changed = [f for f in before if before[f] != after[f]]
    log(f"delivered files changed during the run: {changed or 'none'}")
    log(f"{len(killed)} of {len(killed) + len(survivors)} mutants killed" + (f"; SURVIVORS: {survivors}" if survivors else ""))
    return 0 if not survivors and not changed else 1


if __name__ == "__main__":
    sys.exit(main())
