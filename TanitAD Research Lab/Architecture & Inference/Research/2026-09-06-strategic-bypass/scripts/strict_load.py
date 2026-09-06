# -*- coding: utf-8 -*-
"""STRICT-LOAD PROOF: the real banked refcv4b checkpoint must load into BOTH
an OFF build and an ON build, with zero missing and zero unexpected keys.

This is the half of "bypass, never delete" that a comment cannot assert. The
documented precedent is a dead 1.57 M tensor that had to stay in `state_dict`
because deleting it broke strict loads on 11/11 banked checkpoints with exactly
6 unexpected keys. `hierarchy=False` would do the same here; `--no-strategic`
must not.

ASCII-only prints (cp1252 dev box).
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stack"))
sys.path.insert(0, os.path.join(HERE, "stack", "scripts"))

import refc_v3_train as T                # noqa: E402
from tanitad.refs import refc_v3 as v3   # noqa: E402

CKPT = r"C:\Users\Admin\navcomp\ckpt\ckpt_step9500.pt"
ANCH = r"C:\Users\Admin\navcomp\ckpt\anchors.pt"
CONF = r"C:\Users\Admin\navcomp\ckpt\config.json"

argv_real = json.load(open(CONF))["argv"]
# swap the pod paths this test cannot reach; NOTHING else is changed, and the
# swapped flags are data paths that do not touch the module topology.
argv = list(argv_real)
for i, a in enumerate(argv):
    if a == "--anchors":
        argv[i + 1] = ANCH


def build(no_strategic):
    av = list(argv)
    if no_strategic:
        av.append("--no-strategic")
    ap = T.build_parser() if hasattr(T, "build_parser") else None
    if ap is None:                        # the parser lives inside main()
        raise SystemExit("no build_parser()")
    args = ap.parse_args(av)
    cfg = T._pin_trainer_cfg(
        v3.refc_v3_sized_config(args.size, hier=(args.arm == "hier")), args)
    art = T._read_anchor_artifact(args)
    T._check_anchor_artifact_against_cfg(art, cfg, args)
    return v3.RefCV3Model(cfg), cfg, args


print("loading the banked checkpoint ...")
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = blob.get("model", blob.get("state_dict", blob))
print(f"  ckpt step={blob.get('step')} tensors={len(sd)}")

rows = {}
for tag, flag in (("OFF (today)", False), ("ON  (--no-strategic)", True)):
    model, cfg, args = build(flag)
    r = model.load_state_dict(sd, strict=True)
    missing = list(getattr(r, "missing_keys", []))
    unexpected = list(getattr(r, "unexpected_keys", []))
    keys = sorted(model.state_dict())
    rows[tag] = dict(no_strategic=bool(getattr(cfg.core, "no_strategic", False)),
                     n_keys=len(keys), missing=missing, unexpected=unexpected,
                     hierarchy=bool(cfg.core.hierarchy),
                     params=sum(p.numel() for p in model.parameters()))
    print(f"\n[{tag}]")
    print(f"  cfg.core.hierarchy    = {cfg.core.hierarchy}   (stays True: BYPASS, not delete)")
    print(f"  cfg.core.no_strategic = {getattr(cfg.core, 'no_strategic', 'ABSENT')}")
    print(f"  state_dict keys       = {len(keys)}")
    print(f"  parameters            = {sum(p.numel() for p in model.parameters()):,}")
    print(f"  STRICT LOAD           : missing={len(missing)} unexpected={len(unexpected)}"
          f"  -> {'PASS' if not missing and not unexpected else 'FAIL'}")

k_off = set(build(False)[0].state_dict())
k_on = set(build(True)[0].state_dict())
same = (k_off == k_on)
print(f"\nstate_dict KEY SETS identical ON vs OFF : {'PASS' if same else 'FAIL'}"
      f"  (|OFF|={len(k_off)} |ON|={len(k_on)} sym-diff={len(k_off ^ k_on)})")

# the counter-example that shows the requirement is real
neg = None
try:
    cfg_del = T._pin_trainer_cfg(v3.refc_v3_sized_config("base", hier=True),
                                 build(False)[2])
    cfg_del.core.hierarchy = False
    cfg_del.hier = False
    m_del = v3.RefCV3Model(cfg_del)
    r = m_del.load_state_dict(sd, strict=True)
    neg = "LOADED (unexpected)"
except Exception as e:                    # noqa: BLE001
    msg = str(e).replace("\n", " ")
    neg = msg[:220]
print(f"\nNEGATIVE CONTROL -- hierarchy=False (the DELETE route) strict-loads?")
print(f"  {neg}")

ok = all(not rows[t]["missing"] and not rows[t]["unexpected"] for t in rows) and same
with open(os.path.join(HERE, "STRICT_LOAD.json"), "w") as f:
    json.dump({"ckpt": CKPT, "step": blob.get("step"), "rows": rows,
               "key_sets_identical": same, "negative_control": neg}, f,
              indent=1, default=str)
print(f"\nP4 STRICT LOAD: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
