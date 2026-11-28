"""bias_quick.py -- QUICK_OFFSET.md: the logit-offset ARITHMETIC (NOT the weight edit) from the BANKED STOPDEC npz files.  numpy only, seconds.
p -> sigmoid(logit(p) + b) on the parent's banked stop-goal probabilities (goal_prob[:, 6]); decision p >= 0.5; hit on the 117 stop-point windows, fpr on the other 'ok' windows
(window sets as stopbal_read.window_sets).  Previews the b_match scan and the hit/fpr of GOALEDIT(b1), GOALEDIT(b2) of PREREG_V8F_PROOF_ADD4.md; the weight-edit run (bias_read.py) is the deciding read."""
import sys
import numpy as np
sys.path.insert(0, "F:/TanitAD-scratch/v8proof/bias")
import bias_common as B  # noqa: E402

TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal"
BANK = {"NULL_s0": TH + "/read/out/stopdec/stopdec_BAL-NULL_s0.npz", "NULL_rep": TH + "/read/out/stopdec/stopdec_BAL-NULL_rep.npz", "G_s0": TH + "/read/out/stopdec/stopdec_BAL-G_s0.npz",
        "G2_s0": TH + "/dose2/read/out/stopdec/stopdec_BAL-G2_s0.npz", "NULL_s1": TH + "/read/out/stopdec/stopdec_BAL-NULL_s1.npz", "G_s1": TH + "/read/out/stopdec/stopdec_BAL-G_s1.npz",
        "G2_s1": TH + "/dose2/read/out/stopdec/stopdec_BAL-G2_s1.npz"}


def sets(z):
    has = np.asarray(z["has"], bool)
    ok = has & (np.isfinite(z["err_mps"]).sum(1) > 0)
    return ok, (np.asarray(z["strat_tlstop"]) == 1) & ok


def rates(p, ok, sp):
    d = p >= 0.5
    return float(d[sp].mean()), float(d[ok & ~sp].mean())


def shift(p, b):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def main():
    Z = {t: dict(np.load(f)) for t, f in BANK.items()}
    L = ["# QUICK_OFFSET: logit-offset arithmetic on BANKED STOPDEC files (not the weight edit)", "",
         "MEASURED, T1-kin first tick (diagnostic), eval-141 STOPDEC windows (117 stop-point, 2,234 other labelled); from `theory/stopbal/{read,dose2/read}/out/stopdec/*.npz` by `bias_quick.py`; no bootstrap.",
         "It previews the offset arithmetic p -> sigmoid(logit p + b) of addendum 4 (frozen) on the parent's banked stop-goal probabilities. ln W1 = %.4f (W1 4.5616, BAL-G), ln W2 = %.4f (W2 20.81, BAL-G2) from the banked argv. "
         "The deciding read is the weight-edit run (`bias_read.py`); this table previews its offset rows, its b_match and the replicate gap only." % (B.B1, B.B2), ""]
    gap = {}
    ok0, sp0 = sets(Z["NULL_s0"])
    r0, rr = rates(Z["NULL_s0"]["goal_prob"][:, 6], ok0, sp0), rates(Z["NULL_rep"]["goal_prob"][:, 6], *sets(Z["NULL_rep"]))
    gap = {"hit": abs(rr[0] - r0[0]), "fpr": abs(rr[1] - r0[1])}
    L += [f"Replicate gap |NULL_rep - NULL_s0|: hit {gap['hit']:.4f}, fpr {gap['fpr']:.4f} (NULL_s0 {r0[0]:.4f} / {r0[1]:.4f}, NULL_rep {rr[0]:.4f} / {rr[1]:.4f}).", ""]
    grid = np.arange(-12.0, 12.0001, 0.0005)
    res = {}
    for s, par in B.PARENTS.items():
        ok, sp = sets(Z[par])
        p = Z[par]["goal_prob"][:, 6]
        L += [f"## seed {s[1]} (parent {par})", "", "| arm | hit | fpr | note |", "|---|---|---|---|"]
        L.append(f"| parent {par} (b = 0) | {rates(p, ok, sp)[0]:.4f} | {rates(p, ok, sp)[1]:.4f} | |")
        for nm, b in (("parent + ln W1", B.B1), ("parent + ln W2", B.B2)):
            h, f = rates(shift(p, b), ok, sp)
            L.append(f"| {nm} (b = {b:.4f}) | {h:.4f} | {f:.4f} | offset arithmetic |")
        for tag, nm in ((B.TRAINED[s]["G"], "trained BAL-G"), (B.TRAINED[s]["G2"], "trained BAL-G2")):
            okt, spt = sets(Z[tag])
            ht, ft = rates(Z[tag]["goal_prob"][:, 6], okt, spt)
            fp = np.array([rates(shift(p, b), ok, sp)[1] for b in grid])
            dd = np.abs(fp - ft); nr = np.flatnonzero(dd <= dd.min() + 1e-15); bm = float(grid[nr[len(nr) // 2]])
            hm, fm = rates(shift(p, bm), ok, sp)
            res[(s, tag)] = (ht, ft, bm, hm, fm)
            L.append(f"| {nm} ({tag}) | {ht:.4f} | {ft:.4f} | banked; b_match = {bm:.3f} -> parent offset reads hit {hm:.4f}, fpr {fm:.4f}; hit gap to trained {abs(hm - ht):.4f} (replicate gap {gap['hit']:.4f}) |")
        L.append("")
    L += ["## Reading (offset arithmetic only)", ""]
    for (s, tag), (ht, ft, bm, hm, fm) in res.items():
        b = B.B1 if tag.startswith("G_") else B.B2
        h, f = rates(shift(Z[B.PARENTS[s]]["goal_prob"][:, 6], b), *sets(Z[B.PARENTS[s]]))
        L.append(f"- {tag}: at ln W the offset reads hit {h:.4f} / fpr {f:.4f} vs trained {ht:.4f} / {ft:.4f} (differences {h - ht:+.4f} / {f - ft:+.4f}; gaps {gap['hit']:.4f} / {gap['fpr']:.4f}); "
                 f"b_match {bm:.3f} vs ln W {b:.3f}, hit at b_match {hm:.4f} vs trained {ht:.4f} ({'within' if abs(hm - ht) <= gap['hit'] else 'outside'} the replicate hit gap).")
    L += ["", "Evidence class: MEASURED arithmetic on banked probabilities. It is not a model run and does not read the edited checkpoint; BIAS-EQUALS-DOSE / MATCHES-ROC / DIFFERS are decided by `bias_read.py` only."]
    open(B.BIAS + "/QUICK_OFFSET.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


main()
