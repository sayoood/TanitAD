"""Default-off identity evidence (card 3.4 check 7, second half): the SAME synthetic argv, run by the pristine cdbff971 trainer+model and by the changed trainer+model (no --tac-src-* flag), is bit-identical:
train_log.jsonl (every term, per step, `sec` aside), the final checkpoint tensors (model + optimiser), and the selfcheck terms. argv: python default_off_identity.py <base_tree> <new_tree> <workdir>"""
import json
import os
import subprocess
import sys

import torch

base, new, work = [os.path.abspath(x) for x in sys.argv[1:4]]
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--steps", "8"]


def run(tree, tag):
    out = os.path.join(work, tag)
    env = dict(os.environ)
    env["PYTHONPATH"] = ";".join([os.path.join(tree, "stack"), tree])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    py = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
    code = f"import os,sys,tanitad;assert os.path.abspath(tanitad.__file__).startswith({os.path.join(tree, 'stack')!r}), tanitad.__file__;sys.path.insert(0,{os.path.join(tree, 'stack', 'scripts')!r});import v8f_mini_train as T;assert os.path.abspath(T.__file__).startswith({tree!r}), T.__file__;sys.exit(T.main(sys.argv[1:]))"
    r = subprocess.run([py, "-c", code, "--out", out, *COMMON], env=env, capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


a, b = run(base, "base"), run(new, "new")


def log(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"), encoding="utf-8")]
    for r in rows:
        r.pop("sec", None)
    return rows


la, lb = log(a), log(b)
assert la == lb, "train_log differs"
ca = torch.load(os.path.join(a, "ckpt_last.pt"), map_location="cpu", weights_only=False)
cb = torch.load(os.path.join(b, "ckpt_last.pt"), map_location="cpu", weights_only=False)
assert ca["model"].keys() == cb["model"].keys() and all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"]), "model tensors differ"
oa, ob = ca["opt"]["state"], cb["opt"]["state"]
assert oa.keys() == ob.keys() and all(torch.equal(oa[k][s], ob[k][s]) for k in oa for s in oa[k] if torch.is_tensor(oa[k][s])), "optimiser state differs"
assert ca.keys() == cb.keys(), (sorted(ca.keys()), sorted(cb.keys()))
sa = json.load(open(os.path.join(a, "summary.json"), encoding="utf-8"))
sb = json.load(open(os.path.join(b, "summary.json"), encoding="utf-8"))
for d in (sa, sb):
    d.pop("seconds", None)
    d.pop("argv_sha", None)         # the argv holds --out, which differs by construction
sca = json.load(open(os.path.join(a, "selfcheck.json"), encoding="utf-8"))
scb = json.load(open(os.path.join(b, "selfcheck.json"), encoding="utf-8"))
assert sca.keys() == scb.keys(), (sorted(sca), sorted(scb))
def strip(x):
    s = json.dumps(x, sort_keys=True, default=str)
    for p, t in ((a, "OUT"), (b, "OUT"), (base, "TREE"), (new, "TREE"), (work, "W")):
        s = s.replace(p.replace("\\", "/"), t).replace(p.replace("\\", "\\\\"), t)
    return s
assert sa.keys() == sb.keys(), "summary key set differs"
sd = [k for k in sa if strip(sa[k]) != strip(sb[k])]
print("summary values differing:", sd)
assert not sd, sd
diff = [k for k in sca if strip(sca[k]) != strip(scb[k])]
print("steps", len(la), "log rows equal, model+opt tensors equal, summary key set and values equal, selfcheck keys equal; selfcheck sections differing:", diff)
