"""Source-level mutation arms for lever (a): each mutation edits ONE rule in a COPY of the patched tree and requires test_v8f_level_head.py to go RED (rc != 0). The unmutated copy must be GREEN first.
usage: python mutations_level.py <patched tree> <work dir> <log>"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TREE, WORK, LOG = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MODEL, TRAIN = "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py"
MUTS = [
    ("total_line_deleted", MODEL, '        total = total + cfg.w_level * T["level"] if cfg.level_head == "on" else total', "        pass"),
    ("token_dropped_from_O_cond", MODEL, "            ego_tok = ego_tok + self.e_level(self._level(b, adapt, i)[1][:, None])", "            pass"),
    ("detach_removed", MODEL, "v_in = torch.where(ok, v_hat.detach(), torch.zeros_like(v_hat)) / LEVEL_SCALE", "v_in = torch.where(ok, v_hat, torch.zeros_like(v_hat)) / LEVEL_SCALE"),
    ("head_fed_ego_p10f", MODEL, "v_hat = self.lvl_head(tok.mean(1)).squeeze(-1).float()", 'v_hat = self.lvl_head(self._ego(self._at(b["p10f"], last))).squeeze(-1).float()'),
    ("e_level_not_zero_init", MODEL, "            nn.init.zeros_(self.e_level.weight)", "            pass"),
    ("off_builds_the_head", MODEL, 'if cfg.level_head == "on":                                         # PREREG_V8F_LEVEL: built LAST', "if True:                                         # PREREG_V8F_LEVEL: built LAST"),
    ("label_mask_ignored", MODEL, 'lok = (batch["level_ok"] & batch["frame_ok"][:, -1]).float()', 'lok = batch["frame_ok"][:, -1].float()'),
    ("loop_label_not_attached", TRAIN, '                b["level_y"], b["level_ok"] = lvl_lab', "                pass"),
    ("make_config_drops_the_flag", TRAIN, "level_head=a.level_head, w_level=a.w_level)", 'level_head="off", w_level=a.w_level)'),
    ("gate_allows_a_head_with_no_loss", TRAIN, '        bad.append(f"level_head on declared but w_level is {a.w_level}: a head with no loss -- a lever that is declared and does nothing")', "        pass"),
    ("gate_allows_a_weight_with_no_head", TRAIN, '            bad.append(f"w_level {a.w_level} declared with --level-head off: a loss weight with no head -- a lever that is declared and does nothing")', "            pass"),
    ("flag_check_entry_removed", TRAIN, '    "level_head": "lever:level_head", "w_level": "cfg:w_level",\n', ""),
    ("init_ckpt_leaves_any_key_fresh", TRAIN, "        if set(res.missing_keys) - set(fresh_ok):\n            raise ValueError(", "        if False:\n            raise ValueError("),
    ("bad_init_ckpt_fresh_check_removed", TRAIN, '        if set(got.get("fresh_keys", [])) - set(got.get("declared_new_keys", [])):', "        if False:"),
    ("probe_not_run_in_main", TRAIN, "        sc[\"level_head_probe\"] = lvp", "        pass"),
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
for f in ("test_v8f_level_head.py", "v8f_mini_tip_snapshot_level.txt"):
    shutil.copy2(TREE / "stack/tests" / f, src / "stack/tests" / f)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{src / 'stack'};{src / 'taniteval'}")
bt = WORK / "bt"
bt.mkdir(parents=True, exist_ok=True)


def run(tag):
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "tests/test_v8f_level_head.py", "-p", "no:cacheprovider", "-x", "-o", "tmp_path_retention_policy=failed", f"--basetemp={bt / tag}"], cwd=src / "stack", env=env,
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
