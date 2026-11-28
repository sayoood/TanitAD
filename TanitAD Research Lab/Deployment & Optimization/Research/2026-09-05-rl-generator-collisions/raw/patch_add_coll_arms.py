"""Add the COLLISION-ONLY arms to `ARMS` in stack/scripts/rl_refcv3_min.py.

Idempotent and content-asserted: refuses if the anchor is absent, refuses to double-apply,
and verifies by re-importing the ARMS table and checking the one-variable contrast HOLDS
(coll200 differs from ctrl_null in the collision weight ALONE).
"""
import io
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PATH = os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py")

ANCHOR = '''    "ctrl_null":  dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar",
                       veto_enabled=False),
}'''

NEW = '''    "ctrl_null":  dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar",
                       veto_enabled=False),
    # \u2b50\u2b50 THE COLLISION LEVER (2026-09-05, PI instruction: "our model is creating
    # trajectories with collision, so RL must improve this ... by punishing trajectories
    # with collisions the quality of output trajectories must improve").
    #
    # \u26d4 NO ARM IN THIS PANEL HAS EVER RUN A COLLISION-PUNISHING REWARD IN ISOLATION.
    # `rl` carries collision at 1.00 but CONFOUNDED with progress 0.30 / headway 0.30 /
    # feasibility 0.50 / comfort 0.20; every `ctrl_*`/`veto*` arm has EVERY weight 0.0.
    #
    # \u2b50 ONE-VARIABLE, and the contrast is already banked: `ctrl_null` above is this
    # arm with the collision weight at 0.0 and NOTHING else different -- same w_anchor,
    # same lr, same steps, same noise_mode, same use_gt_bar, same veto_enabled=False.
    # The ONLY moving part is `collision` 0.0 -> 1.0.
    #
    # \u26a0 PRE-REGISTERED EXPECTATION (SPEC \u00a72, MEASURED 0-GPU before this ran): the
    # advantage is GROUP-RELATIVE across one window's candidates, and only 7.92 % of
    # windows contain BOTH a colliding and a non-colliding candidate. 92.08 % therefore
    # contribute an identically ZERO collision advantage. A null here is DIAGNOSTIC of
    # signal density, not of the reward's direction -- the composed reward already ranks
    # colliders last perfectly (AUC 0.0000).
    "coll200":    dict(weights={**{k: 0.0 for k in DEFAULT_WEIGHTS}, "collision": 1.0},
                       w_anchor=1.0, lr=1e-5, steps=200, use_gt_bar=False,
                       noise_mode="two_scalar", veto_enabled=False),
    "coll2k":     dict(weights={**{k: 0.0 for k in DEFAULT_WEIGHTS}, "collision": 1.0},
                       w_anchor=1.0, lr=1e-5, steps=2000, use_gt_bar=False,
                       noise_mode="two_scalar", veto_enabled=False),
}'''

src = None
for _ in range(10):
    try:
        with io.open(PATH, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        import time
        time.sleep(3)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read %s (mount)" % PATH)

if '"coll200"' in src:
    print("ALREADY APPLIED -- coll200 present, not double-applying")
elif ANCHOR not in src:
    raise SystemExit("REFUSED: anchor (ctrl_null block + closing brace) not found in %s" % PATH)
else:
    src = src.replace(ANCHOR, NEW, 1)
    for _ in range(10):
        try:
            with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
                fh.write(src)
            break
        except OSError:
            import time
            time.sleep(3)
    print("APPLIED -- coll200 + coll2k inserted after ctrl_null")

# ---- content assertion: re-read and check the ONE-VARIABLE contrast actually holds ----
with io.open(PATH, encoding="utf-8") as fh:
    back = fh.read()
assert '"coll200"' in back and '"coll2k"' in back, "VERIFY FAILED: arms absent after write"
ns = {}
exec(compile("DEFAULT_WEIGHTS = {'progress':0.30,'collision':1.00,'headway':0.30,"
             "'feasibility':0.50,'comfort':0.20}\n"
             + back[back.index("ARMS = {"):back.index("\ndef _p(")], PATH, "exec"), ns)
A = ns["ARMS"]
a, b = A["coll200"], A["ctrl_null"]
diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
wdiff = {k for k in set(a["weights"]) | set(b["weights"])
         if a["weights"].get(k) != b["weights"].get(k)}
print("ONE-VARIABLE CHECK")
print("  non-weight keys differing coll200 vs ctrl_null : %s" % (sorted(diff - {"weights"}) or "NONE"))
print("  weight keys differing                          : %s" % sorted(wdiff))
print("  coll200 weights  = %s" % a["weights"])
print("  ctrl_null weights= %s" % b["weights"])
ok = (diff - {"weights"}) == set() and wdiff == {"collision"}
print("ONE_VARIABLE=%s" % ("PASS" if ok else "FAIL"))
if not ok:
    raise SystemExit("REFUSED: the arm is not one-variable against ctrl_null")
