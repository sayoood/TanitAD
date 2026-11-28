"""rh_selfcheck.py = sb_selfcheck.py (DSCALE10K gate driver) with the tree switched to /root/rollhist/tree and the study path dropped.
Stage-B G-LIVE driver (pod). Runs the trainer's own ``--selfcheck-only`` on a stage-B argv (real bank, real --init-ckpt), optionally with the DAgger swap MUTATED.

usage: sb_selfcheck.py <argv.json> <out_dir> <result.json> [--mutate-apply] [--max-clips N]
  --mutate-apply : DG.DaggerSwap.apply still draws and records its selection but RETURNS THE LOGGED BATCH (the swapped tensors are discarded). This is "delete the line that applies the swap":
                   the trainer's dagger_probe must refuse (an AssertionError turned into a refusal), so no passing selfcheck record exists. RED is the expected outcome of the mutation.
  --max-clips N  : smoke only (caps the TRAIN clips; the bank must have been built from an argv with the same cap)
Writes <result.json> = {"rc", "refused", "message", "selfcheck": <path or null>, "mutated": bool}; the trainer's selfcheck.json lands in <out_dir>."""
import json
import os
import sys
import traceback

TREE = os.environ.get("RH_TREE", "/root/rollhist/tree_cdb")   # ROLLHIST: the new tree (the DSCALE10K original pins /root/dscale10k/tree)
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE):
    if p not in sys.path:
        sys.path.insert(0, p)
argv_p, out_dir, res_p = sys.argv[1], sys.argv[2], sys.argv[3]
mut = "--mutate-apply" in sys.argv
mc = int(sys.argv[sys.argv.index("--max-clips") + 1]) if "--max-clips" in sys.argv else 0
argv = [str(t) for t in json.load(open(argv_p))["argv"] if t != "--fit-join"]


def setf(t, f, vals):
    t = list(t)
    if f not in t:
        return t + [f] + list(vals)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


argv = setf(argv, "--out", [out_dir])
if mc:
    argv = setf(argv, "--max-clips", [str(mc)])
argv += ["--selfcheck-only"]
os.makedirs(out_dir, exist_ok=True)
import v8f_mini_train as T  # noqa: E402
from tanitad.data import v8f_dagger as DG  # noqa: E402

if mut:
    _orig = DG.DaggerSwap.apply

    def _apply_discarding(self, batch, step, *, p=None):
        _orig(self, batch, step, p=p)                                   # the swap is drawn and recorded (self.last) ...
        return batch                                                    # ... and the swapped tensors are thrown away
    DG.DaggerSwap.apply = _apply_discarding
res = {"mutated": mut, "argv": argv_p, "out": out_dir}
try:
    rc = T.main(argv)
    res["rc"] = 0 if rc in (None, 0) else int(rc)
    res["refused"] = False
except SystemExit as e:
    res["rc"] = e.code if isinstance(e.code, int) else 1
    res["refused"] = res["rc"] != 0
    res["message"] = str(e.code)[:600]
except BaseException as e:  # noqa: BLE001
    res["rc"] = 1
    res["refused"] = True
    res["message"] = (type(e).__name__ + ": " + str(e))[:600]
    res["traceback_tail"] = traceback.format_exc()[-600:]
sc = os.path.join(out_dir, "selfcheck.json")
res["selfcheck"] = sc if os.path.isfile(sc) else None
if res["selfcheck"]:
    s = json.load(open(sc))
    res["declared_vs_built"] = s.get("declared_vs_built")
    res["dagger_probe_pass"] = (s.get("dagger_probe") or {}).get("pass")
    res["acc_probe_pass"] = (s.get("acc_probe") or {}).get("pass")
json.dump(res, open(res_p, "w"), indent=1)
print("SB-SELFCHECK", json.dumps({k: v for k, v in res.items() if k != "traceback_tail"})[:600])
