"""AFTER-measurement: per-term reproducibility with `sigreg_generator`, plus
the content-anchor SHAs the guards actually resolved."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

_STACK = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "tests"))

import test_loss_determinism as D  # noqa: E402


def per_term(sig_seed_a, sig_seed_b, maximal=False):
    out = {}
    for stage in D.STAGES:
        s = D._maximal() if maximal else D._build()
        s.train()
        b, kw = D._batch(s), D._kw(stage, maximal=maximal)
        x = D._call(s, b, kw, sig_seed=sig_seed_a)
        y = D._call(s, b, kw, sig_seed=sig_seed_b)
        terms = {}
        for t in x["log"]["terms"]:
            va, vb = float(x[t]), float(y[t])
            terms[t] = {"a": va, "b": vb,
                        "eq": bool(torch.equal(x[t], y[t])),
                        "rel_pct": 100 * abs(va - vb) / max(abs(va), 1e-30)}
        la, lb = float(x["loss"]), float(y["loss"])
        out[stage] = {"total": {"a": la, "b": lb,
                                "eq": bool(torch.equal(x["loss"], y["loss"])),
                                "rel_pct": 100 * abs(la - lb)
                                / max(abs(la), 1e-30)},
                      "terms": terms}
    return out


def anchor_sha(rel, marker):
    log = subprocess.run(["git", "log", "--format=%H", "--", rel], cwd=_ROOT,
                         capture_output=True, timeout=180)
    for sha in log.stdout.decode().split():
        r = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=_ROOT,
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout or marker.encode() in r.stdout:
            continue
        return sha
    return None


def show(tag, d):
    print(f"\n########## {tag}")
    for s, v in d.items():
        t = v["total"]
        print(f"== {s}  TOTAL a={t['a']:.6f} b={t['b']:.6f} eq={t['eq']} "
              f"rel={t['rel_pct']:.4f}%")
        for k, x in v["terms"].items():
            print(f"     {k:<7} a={x['a']:.6f} b={x['b']:.6f} "
                  f"eq={str(x['eq']):<5} rel={x['rel_pct']:.4f}%")


if __name__ == "__main__":
    a = per_term(11, 11)
    show("AFTER — same sigreg_generator seed (minimal build)", a)
    b = per_term(11, 11, maximal=True)
    show("AFTER — same sigreg_generator seed (ALL LEVERS ON)", b)
    c = per_term(11, 12)
    show("NEGATIVE CONTROL — different sigreg seeds", c)
    d = per_term(None, None)
    show("NEGATIVE CONTROL — DEFAULT path (incumbent, unchanged)", d)
    print("\n########## CONTENT ANCHORS RESOLVED")
    for rel, marker in (("stack/tanitad/models/sigreg.py", "sample_directions"),
                        ("stack/scripts/train_v6_staged.py",
                         "sigreg_generator")):
        print(f"  {rel}  (marker {marker!r}) -> {anchor_sha(rel, marker)}")
    Path(__file__).with_name("after_fix.json").write_text(json.dumps(
        {"same_seed": a, "same_seed_maximal": b, "diff_seed": c,
         "default_path": d}, indent=1))
