"""LEADSUP edit 2 (run once on tree_patch after apply_edits_data/train; argv[1] = tree root): a synthetic lead track for the SYNTHETIC world, opt-in (only the trainer's --synthetic path with --lead-sup on).
The default synthetic join has lead_valid 0 and lead_gap NaN everywhere (no release, no lead file), so a lead head would have nothing to learn from and the gate (rightly) refuses it."""
import sys

root = sys.argv[1]
p = f"{root}/stack/tanitad/data/v8f_mini_data.py"
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''# ------------------------------------------------------------------------------------------------ kinematic floors
KIN_MODES = ("steer", "yaw")''', '''def synthetic_lead_labels(J: dict, seed: int = 0) -> dict:
    """LEADSUP, SYNTHETIC worlds only (tests and the trainer's --synthetic selfcheck with --lead-sup on): fills ``J["lead_valid"]`` / ``J["lead_gap"]`` IN PLACE with a lead that is present in two blocks per clip
    (rows 6..34 and 40..end, a jump between them = a lead change), a gap that random-walks around 25 m (steps N(0, 0.4 m)), NaN while absent. Deterministic in ``seed``. Returns J."""
    rng = np.random.default_rng(seed + 7)
    for ci in range(len(J["off"])):
        o, n = int(J["off"][ci]), int(J["n"][ci])
        pres = np.zeros(n, bool)
        pres[6:35] = True
        pres[40:] = True
        g = np.empty(n)
        g[:] = 25.0 + np.cumsum(rng.normal(0.0, 0.4, n))
        g[40:] += 9.0                                               # a second lead appears with a jump
        g = np.clip(g, 1.0, 59.0)
        J["lead_valid"][o:o + n] = pres.astype(J["lead_valid"].dtype)
        J["lead_gap"][o:o + n] = np.where(pres, g, np.nan).astype(J["lead_gap"].dtype)
    return J


# ------------------------------------------------------------------------------------------------ kinematic floors
KIN_MODES = ("steer", "yaw")''')
open(p, "w", encoding="utf-8", newline="").write(s)

p = f"{root}/stack/scripts/v8f_mini_train.py"
s = open(p, encoding="utf-8", newline="").read()
rep('''        J = D.load_join(os.path.join(root, "join_train.npz"))
        cache = root
        clips_all = w["names"]["train"]''', '''        J = D.load_join(os.path.join(root, "join_train.npz"))
        if a.lead_sup == "on":                                       # LEADSUP: the synthetic join carries no lead track; this opt-in fills one (selfcheck only; a real launch never passes --synthetic)
            D.synthetic_lead_labels(J, a.seed)
        cache = root
        clips_all = w["names"]["train"]''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("OK")
