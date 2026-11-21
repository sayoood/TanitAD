"""The bit-identity guard's BASELINE has rotted — measured, then RE-RUN against
the TRUE pre-refcv6 file.

`stack/tests/test_refcv6_diffusion.py:63` materialises its "pre-refcv6"
baseline with `git show HEAD:stack/tanitad/refs/refc.py`. That blob is now
byte-identical to the working tree and carries 83 occurrences of "refcv6", so
the test compares refc.py against ITSELF — advisory class F item 1, *"the
expected value is an expression over the code under test"*.

This re-runs the test's OWN comparison with the baseline pinned to `8c7d215^`,
the commit before `refcv6_diffusion` entered `refc.py`. Two outcomes, both
stated in advance:
  * IDENTICAL -> the CLAIM still holds and only the GUARD is inert;
  * DIFFERENT -> the guard has been hiding a real off-path drift.
"""
from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path
import torch
from tanitad.refs import refc

ROOT = Path("D:/Projects/TanitAD")
REL = "stack/tanitad/refs/refc.py"
BASE = "8c7d215^"

def load_at(rev):
    out = subprocess.run(["git", "show", f"{rev}:{REL}"], cwd=ROOT,
                         capture_output=True, check=True).stdout
    tmp = Path(__file__).resolve().parent / f"_baseline_refc.py"
    tmp.write_bytes(out)
    try:
        spec = importlib.util.spec_from_file_location("_baseline_refc", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_baseline_refc"] = mod
        spec.loader.exec_module(mod)
        return mod, len(out)
    finally:
        tmp.unlink(missing_ok=True)

def build(mod, seed=7):
    torch.manual_seed(seed)
    cfg = mod.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2, sampler="ddim")
    torch.manual_seed(seed)
    d = mod.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=8, d_ctx=4, tac_latent_dim=4,
        anchors=torch.zeros(5, 4, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=(5, 10, 15, 20),
        v0_conditioned=True, control_units="alat")
    return d.eval()

def compare(old, new):
    r = {}
    a, b = build(old), build(new)
    sa, sb = a.state_dict(), b.state_dict()
    r["state_dict_keys_equal"] = bool(sa.keys() == sb.keys())
    r["keys_only_in_new"] = sorted(set(sb) - set(sa))
    r["keys_only_in_old"] = sorted(set(sa) - set(sb))
    r["weight_mismatches"] = [k for k in sa if k in sb and not torch.equal(sa[k], sb[k])]
    B = 64
    fmap = torch.randn(B, 16, 3, 5); m = torch.randn(B, 8)
    v0 = torch.rand(B) * 20 + 2
    torch.manual_seed(99); oa = a(fmap, m, steps=2, v_ms=v0)
    torch.manual_seed(99); ob = b(fmap, m, steps=2, v_ms=v0)
    r["out_keys_symmetric_diff"] = sorted(set(oa) ^ set(ob))
    diffs = {}
    for k, va in oa.items():
        if isinstance(va, torch.Tensor) and k in ob and isinstance(ob[k], torch.Tensor):
            if not torch.equal(va, ob[k]):
                diffs[k] = round(float((va.float() - ob[k].float()).abs().max()), 10)
    r["tensor_diffs"] = diffs
    r["sel_tele_equal"] = bool(oa.get("sel_tele") == ob.get("sel_tele"))
    r["BIT_IDENTICAL"] = bool(r["state_dict_keys_equal"] and not r["weight_mismatches"]
                              and not r["out_keys_symmetric_diff"] and not diffs
                              and r["sel_tele_equal"])
    return r

def main():
    out = {}
    head_blob = subprocess.run(["git","rev-parse",f"HEAD:{REL}"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
    wt_blob = subprocess.run(["git","hash-object",REL], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    out["HEAD_blob"] = head_blob; out["worktree_blob"] = wt_blob
    out["baseline_used_by_the_test_IS_the_worktree"] = bool(
        len(head_blob) == 40 and head_blob == wt_blob)
    for rev, tag in ((BASE, "TRUE_pre_refcv6_8c7d215^"),):
        mod, nbytes = load_at(rev)
        out[f"{tag}_bytes"] = nbytes
        out[f"{tag}_refcv6_mentions"] = subprocess.run(
            ["git","show",f"{rev}:{REL}"], cwd=ROOT,
            capture_output=True).stdout.decode("utf-8","replace").count("refcv6")
        out[tag] = compare(mod, refc)
    txt = json.dumps(out, indent=2, default=str); print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "bitidentity_baseline.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
