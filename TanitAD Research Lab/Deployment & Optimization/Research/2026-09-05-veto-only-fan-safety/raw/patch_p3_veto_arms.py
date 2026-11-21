#!/usr/bin/env python3
"""P3 - the VETO-ONLY arms, as PRODUCTS, plus a zero-weight arm that is an actual null.

Adds three arms to `stack/scripts/rl_refcv3_min.py` and wires the three explicit veto
fields through `make_cfg` so a run's `config.json` states whether the constraint was in
force:

  veto200 / veto2k   reward weights ALL ZERO, veto ON  -> DDv2's constraint channel with
                     no ranking term at all. `ctrl_const` was accidentally exactly this
                     and it is the only arm that improved refcv3's fan feasibility
                     (fan_peak_g_mean -0.0859 g, top32_infeasible -0.0143, 200 steps).
  ctrl_null          reward weights ALL ZERO, veto OFF -> the control `ctrl_const` was
                     supposed to be: `veto_rate` must read EXACTLY 0.0.

`reg_echo` is pinned to `veto_collision=False`, which is what the OLD key-membership
code gave it (weights={"gt_similarity": 1.0} carries no "collision" key), so the banked
arm still reproduces.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
CR = chr(13)
LF = chr(10)
CRLF = CR + LF
NE = chr(9940)
ST = chr(11088)
WARN = chr(9888)


def rw(path, old, new, *, count=1):
    p = os.path.join(REPO, path)
    raw = open(p, "rb").read()
    is_crlf = CRLF.encode() in raw
    s = raw.decode("utf-8").replace(CRLF, LF)
    if new in s and old not in s:
        print("  [skip] %s: already patched" % path)
        return
    n = s.count(old)
    if n != count:
        raise SystemExit("[patch] %s: expected %d occurrence(s), found %d" % (path, count, n))
    s = s.replace(old, new)
    open(p, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
    print("  [ok]   %s (%s)" % (path, "CRLF" if is_crlf else "LF"))


# 1. reg_echo keeps the veto it actually ran with
rw("stack/scripts/rl_refcv3_min.py",
   '''    "reg_echo":   dict(weights={"gt_similarity": 1.0}, w_anchor=0.0, lr=1e-5, steps=2000,
                       use_gt_bar=False, noise_mode="two_scalar"),''',
   '''    "reg_echo":   dict(weights={"gt_similarity": 1.0}, w_anchor=0.0, lr=1e-5, steps=2000,
                       use_gt_bar=False, noise_mode="two_scalar",
                       # {W} REPRODUCIBILITY PIN. Under the OLD key-membership veto
                       # (`"collision" in spec.weights`) this arm's weights carry no
                       # "collision" key, so it ran with the COLLISION channel OFF and
                       # the TTC channel ON. Stated explicitly so the banked arm still
                       # reproduces after the veto became a config field.
                       veto_collision=False, veto_ttc=True),'''.replace("{W}", WARN))

# 2. the new arms, appended inside the ARMS dict
rw("stack/scripts/rl_refcv3_min.py",
   '''    "ctrl_const": dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar"),
}''',
   '''    "ctrl_const": dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar"),
    # {ST}{ST} THE VETO-ONLY PRODUCT (2026-09-05, Arch+Inference FlyWheel).
    # `ctrl_const` above was an UNINTENTIONALLY EXACT veto-only arm - every reward
    # weight 0.0 while `posttrain.py` keyed the veto on the reward's KEY SET - and it
    # is the ONLY arm in the whole RL panel that moved refcv3's fan the RIGHT way
    # (fan_peak_g_mean -0.0859 g, top32_infeasible -0.0143, top8_kamm_over -0.0137,
    # in 200 steps / 2 min 07 s). The composed reward then overwhelmed that gain and
    # drove the planner below the constant-velocity floor.
    # => the veto is promoted from accident to product. The reward is still
    # identically 0.0 (MEASURED, not asserted: RewardSpec over these weights returns a
    # tensor whose unique value is {0.0}); the ONLY signal is DDv2's constraint
    # channel, now keyed EXPLICITLY by `veto_enabled` rather than by a key set.
    "veto200":    dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar",
                       veto_enabled=True),
    "veto2k":     dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=2000, use_gt_bar=False, noise_mode="two_scalar",
                       veto_enabled=True),
    # {NE} THE ACTUAL NULL that `ctrl_const` was supposed to be: zero reward AND no
    # veto, so the advantage is identically zero and `veto_rate` reads EXACTLY 0.0.
    # It is the empirical proof that P2's fix works - a zero-weight control that is
    # not a null is the false-green class wearing a control's clothes.
    "ctrl_null":  dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar",
                       veto_enabled=False),
}'''.replace("{ST}", ST).replace("{NE}", NE))

# 3. make_cfg carries the veto fields into the run record
rw("stack/scripts/rl_refcv3_min.py",
   '''        reward_weights=dict(spec["weights"]), w_anchor=float(spec["w_anchor"]),''',
   '''        reward_weights=dict(spec["weights"]), w_anchor=float(spec["w_anchor"]),
        # {ST} THE VETO, FROM THE ARM SPEC AND INTO `config.json`. Never from the
        # reward's key set again (RESULT.md 12.1 / RETRACTION #24).
        veto_enabled=bool(spec.get("veto_enabled", True)),
        veto_collision=bool(spec.get("veto_collision", True)),
        veto_ttc=bool(spec.get("veto_ttc", True)),'''.replace("{ST}", ST))

print("[patch] P3 arms applied")
