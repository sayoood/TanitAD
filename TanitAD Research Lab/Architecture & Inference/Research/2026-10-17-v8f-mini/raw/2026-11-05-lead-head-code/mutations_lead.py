"""Source-level mutation arms for the lead head: each mutation edits ONE rule in a COPY of the patched tree and requires test_v8f_lead_head.py to go RED (rc != 0). The unmutated copy must be GREEN first.
usage: python mutations_lead.py <patched tree> <work dir> <log>"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TREE, WORK, LOG = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MODEL, TRAIN, DATA = "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"
MUTS = [
    ("total_line_deleted", MODEL, '        total = total + cfg.w_lead * T["lead"] if cfg.lead_head == "on" else total', "        pass"),
    ("head_dropped_from_O_cond_add", MODEL, "            ego_tok = ego_tok + self.e_lead(self._lead(b, adapt, i)[2])", "            pass"),
    ("token_dropped_from_O_cond", MODEL, "            toks.append(self.e_aux(self._aux_in(b, adapt, i)))", "            pass"),
    ("detach_removed", MODEL, "torch.stack([torch.sigmoid(logit.detach()), gap_hat.detach()], -1)", "torch.stack([torch.sigmoid(logit), gap_hat], -1)"),
    ("head_fed_ego_p10f", MODEL, "        o = self.lead_head(tok.mean(1)).float()", '        o = self.lead_head(self._ego(self._at(b["p10f"], last))).float()'),
    ("e_lead_not_zero_init", MODEL, "                nn.init.zeros_(self.e_lead.weight)", "                pass"),
    ("off_builds_the_lead_head", MODEL, 'if cfg.lead_head == "on":                                          # PREREG_LEVELHEAD_A2: after the level head', "if True:                                          # PREREG_LEVELHEAD_A2: after the level head"),
    ("label_mask_ignored", MODEL, 'lhas = (batch["lead_has"] & fok).float()', "lhas = fok.float()"),
    ("near_mask_ignored", MODEL, 'lnear = (batch["lead_near"] & batch["lead_has"] & fok).float()', "lnear = lhas"),
    ("huber_part_dropped_from_the_term", MODEL, 'T["lead"] = T["lead_bce"] + T["lead_huber"]', 'T["lead"] = T["lead_bce"]'),
    ("bce_target_wrong", MODEL, 'out["lead_logit"], batch["lead_near"].float(), reduction="none")', 'out["lead_logit"], torch.ones_like(out["lead_logit"]), reduction="none")'),
    ("aux_token_never_built", MODEL, 'if cfg.aux_cond == "token" and (self.lvl_head is not None or self.lead_head is not None):', 'if False and (self.lvl_head is not None or self.lead_head is not None):'),
    ("aux_token_zero_init", MODEL, "self.e_aux = nn.Linear(AUX_TOKEN_DIM, d)", "self.e_aux = nn.Linear(AUX_TOKEN_DIM, d); nn.init.zeros_(self.e_aux.weight); nn.init.zeros_(self.e_aux.bias)"),
    ("aux_input_fed_ego", MODEL, "        return torch.cat([lv[:, None], ld], -1)", '        return torch.cat([lv[:, None] + self._at(b["p10f"], last)[:, :1], ld], -1)'),
    ("loop_label_not_attached", TRAIN, '                b["lead_y"], b["lead_has"], b["lead_near"] = lead_lab', "                pass"),
    ("probe_batch_without_labels", TRAIN, '            probe["lead_y"], probe["lead_has"], probe["lead_near"] = lead_stage(probe, LL)', "            pass"),
    ("make_config_drops_lead_flag", TRAIN, "lead_head=a.lead_head, w_lead=a.w_lead, aux_cond=a.aux_cond)", 'lead_head="off", w_lead=a.w_lead, aux_cond=a.aux_cond)'),
    ("make_config_drops_aux_cond", TRAIN, "lead_head=a.lead_head, w_lead=a.w_lead, aux_cond=a.aux_cond)", 'lead_head=a.lead_head, w_lead=a.w_lead, aux_cond="add")'),
    ("gate_allows_a_head_with_no_loss", TRAIN, '        bad.append(f"lead_head on declared but w_lead is {a.w_lead}: a head with no loss -- a lever that is declared and does nothing")', "        pass"),
    ("gate_allows_a_weight_with_no_head", TRAIN, '            bad.append(f"w_lead {a.w_lead} declared with --lead-head off: a loss weight with no head -- a lever that is declared and does nothing")', "            pass"),
    ("gate_allows_token_with_no_head", TRAIN, '        bad.append("--aux-cond token declared with --level-head and --lead-head both off: there is no head output for the token to carry -- a lever that is declared and does nothing")', "        pass"),
    ("gate_allows_additive_paths_under_token", TRAIN, '            bad.append("aux_cond token declared but an additive path (e_level / e_lead) was built: the token REPLACES both additive paths")', "            pass"),
    ("gate_allows_no_labels_coverage", TRAIN, '        if ll.get("n_clips_with_labels", 0) < LEAD_MIN_CLIP_COVERAGE * max(ll.get("n_clips", 0), 1):', "        if False:"),
    ("missing_label_file_allowed", TRAIN, "    if not os.path.exists(a.lead_train):\n        raise SystemExit(", "    if False:\n        raise SystemExit("),
    ("flag_check_entries_removed", TRAIN, '    "lead_head": "lever:lead_head", "w_lead": "cfg:w_lead", "aux_cond": "lever:aux_cond",\n', ""),
    ("init_ckpt_prefixes_not_extended", TRAIN, "if k.startswith(M.NEW_LEVER_KEY_PREFIXES))", "if k.startswith(M.LEVEL_KEY_PREFIXES))"),
    ("bad_init_ckpt_lead_only_not_declared", TRAIN, '(a.level_head == "on" or a.lead_head == "on")', '(a.level_head == "on")'),
    ("init_ckpt_leaves_any_key_fresh", TRAIN, "        if set(res.missing_keys) - set(fresh_ok):\n            raise ValueError(", "        if False:\n            raise ValueError("),
    ("probe_call_replaced_in_main", TRAIN, "        ldp = lead_head_probe(model, probe, cfg)", '        ldp = {"lead": "off", "pass": True}'),
    ("probe_record_dropped", TRAIN, '        sc["lead_head_probe"] = ldp', "        pass"),
    ("lead_stage_wrong_step", TRAIN, 'LL["off"][batch["ci"].cpu().numpy()] + batch["t"].cpu().numpy()', 'LL["off"][batch["ci"].cpu().numpy()] + 0'),
    ("labels_k_rule_broken", DATA, "kj = 2 * np.arange(int(n[ci]))", "kj = np.arange(int(n[ci]))"),
    ("labels_range_threshold_doubled", DATA, "(gap <= LEAD_RANGE_M)", "(gap <= 2 * LEAD_RANGE_M)"),
    ("level_probe_reads_the_wrong_aux_column", TRAIN, 'reach_grad = {"e_aux_level_col_grad_at_w0": float(g_0["e_aux.weight"][:, 0].abs().sum())}', 'reach_grad = {"e_aux_level_col_grad_at_w0": float(g_0["e_aux.weight"][:, 1].abs().sum())}'),
    ("lead_probe_reads_the_wrong_aux_columns", TRAIN, 'reach_grad = {"e_aux_lead_cols_grad_at_w0": float(g_0["e_aux.weight"][:, 1:].abs().sum())}', 'reach_grad = {"e_aux_lead_cols_grad_at_w0": float(g_0["e_aux.weight"][:, :1].abs().sum())}'),
]
t0 = time.time()
src = WORK / "src"
if src.exists():
    shutil.rmtree(src)
for sub in ("stack/tanitad", "stack/scripts"):
    shutil.copytree(TREE / sub, src / sub, ignore=shutil.ignore_patterns("__pycache__"))
def _small(d, names):                                       # taniteval carries large result files the tests never read: copy code and small data only
    return [n for n in names if n == "__pycache__" or ((Path(d) / n).is_file() and (Path(d) / n).stat().st_size > 2_000_000)]


shutil.copytree(TREE / "taniteval", src / "taniteval", ignore=_small)
(src / "stack/tests").mkdir(parents=True)
for f in ("test_v8f_lead_head.py", "v8f_mini_tip_snapshot_lead.txt"):
    shutil.copy2(TREE / "stack/tests" / f, src / "stack/tests" / f)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{src / 'stack'};{src / 'taniteval'}")
bt = WORK / "bt"
bt.mkdir(parents=True, exist_ok=True)


def run(tag):
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "tests/test_v8f_lead_head.py", "-p", "no:cacheprovider", "-x", "-o", "tmp_path_retention_policy=failed", f"--basetemp={bt / tag}"], cwd=src / "stack", env=env,
                       capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)
    out = (r.stdout + r.stderr).splitlines()
    summ = next((ln for ln in reversed(out) if " passed" in ln or " failed" in ln or " error" in ln), "")
    first = next((ln for ln in out if ln.startswith(("FAILED", "ERROR"))), "")
    return r.returncode, summ.strip(), first[:200]


lines = []
rc, summ, first = run("clean")
lines.append(f"CLEAN rc={rc} {summ}")
if rc != 0:                                                  # a RED mutation means nothing when the unmutated copy is not GREEN
    lines.append(f"ALL-DONE {time.time() - t0:.0f}s INVALID: the unmutated copy is not green ({first})")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(2)
bad = False
for name, rel, old, new in MUTS:
    p = src / rel
    orig = (TREE / rel).read_bytes().decode("utf-8")
    if orig.count(old) != 1:
        lines.append(f"MUTATION {name}: ANCHOR NOT UNIQUE ({orig.count(old)}) -- not run")
        bad = True
        continue
    p.write_bytes(orig.replace(old, new).encode("utf-8"))
    try:
        rc, summ, first = run(name)
    finally:
        p.write_bytes(orig.encode("utf-8"))
    red = rc != 0
    bad |= not red
    lines.append(f"MUTATION {name}: {'RED (good)' if red else 'GREEN -- NOT CAUGHT'} rc={rc} {summ} | {first}")
lines.append(f"ALL-DONE {time.time() - t0:.0f}s {'FAIL' if bad else 'every mutation RED, clean GREEN'}")
LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(1 if bad else 0)
