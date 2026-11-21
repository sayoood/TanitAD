"""make_reader.py -- derive levelprobe_read.py from the HISTSWAP-LF reader (histswap_lf/code/histswap_lf_read.py, sha256 4e3a75db...) by exact-match replacements (each must match once).
The reader keeps ALL of HISTSWAP-LF's known answers (KY, KA1, KC1-KC3, KF, K3, KB1), its rows (the 19 rows of Read A, X - Y, Y - P10 by gapdecomp.decompose: paired episode-cluster bootstrap, B 2,000, seed 0) and its outcome rule
(smooth_t1_read.arm_outcome: DOMINANT > HARMS > MOVED > NO EFFECT); only the arms change (HLT0 = --histswap cv_t0, HPRED = --histswap cv_pred, run-tag prefixes HT0 / HPR), KC1 is re-spelled for a constant level, and the card's
SUPPORTED / NOT SUPPORTED / INCONCLUSIVE rule (PREREG_LEVELPROBE.md section A) replaces the A / B / C interpretation.  python -I code/make_reader.py"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SRC = "E:/Projects/TanitAD-artifacts/dagland/histswap_lf/code/histswap_lf_read.py"
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "4e3a75db6023d7eb07465bb826fe56455dac01aa1aba42fda0b197626cbe119f", "REFUSED: not the HISTSWAP-LF reader"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('KAP_PK = "E:/Projects/TanitAD-artifacts/dagland/kappa_t1"\n', 'KAP_PK = "E:/Projects/TanitAD-artifacts/dagland/kappa_t1"\nHLF_PK = "E:/Projects/TanitAD-artifacts/dagland/histswap_lf"                 # the Y pins and the HLVL / HCV context live here\n')
rep('OUT_X = "F:/TanitAD-scratch/histswap_lf/out"\nWORK = "F:/TanitAD-scratch/histswap_lf/work/rig"', 'OUT_X = "F:/TanitAD-scratch/levelprobe_scratch/out"\nWORK = "F:/TanitAD-scratch/levelprobe_scratch/work/rig"')
rep('MODES = {"HCV": ("cv_exec", "HCV"), "HLVL": ("cv_logged", "HLV")}\nKA1_TAG, KA1_REF = "SREP5_DAGOACC_s0", "SREP_DAGOACC_s0"',
    'MODES_ALL = {"HLT0": ("cv_t0", "HT0"), "HPRED": ("cv_pred", "HPR")}      # HPRED is read only when its runs exist (F1: not launched without an approved amendment)\nMODES = dict(MODES_ALL)\nKA1_TAG, KA1_REF = "LPKA1_DAGOACC_s0", "SREP_DAGOACC_s0"')
rep('SHARE_BAR = 0.5\n', 'SHARE_BAR = 0.5\nPRED_CELL_BAR, PRED_CELLS_NEEDED, PRED_VS_T0, NOTSUP_BAR = 0.30, 3, 0.5, 0.10          # PREREG_LEVELPROBE.md section A\n')
# y pins come from the HISTSWAP-LF freeze (the same Y runs)
rep('    for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8"):\n        p = ln.split()\n        if len(p) >= 5 and p[0] == "Y"', '    for ln in open(f"{HLF_PK}/FROZEN_SHA.txt", encoding="utf-8"):\n        p = ln.split()\n        if len(p) >= 5 and p[0] == "Y"')
# card hash
rep('    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_HISTSWAP_LF_FROZEN.md sha256")][0]\n    got = SR.sha256_file(f"{PK}/PREREG_HISTSWAP_LF_FROZEN.md")',
    '    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_LEVELPROBE.md sha256")][0]\n    got = SR.sha256_file(f"{PK}/PREREG_LEVELPROBE.md")')
rep('        print(f"REFUSED: PREREG_HISTSWAP_LF_FROZEN.md {got} != frozen {want}", file=sys.stderr)', '        print(f"REFUSED: PREREG_LEVELPROBE.md {got} != frozen {want}", file=sys.stderr)')
# active modes: only those whose four runs are done
rep('    arr_dir = f"{a.raw}/arrays"\n    res = {"hypothesis": "H-HISTSWAP-LF-1", "lever": "--histswap cv_exec (HCV, deployable) and cv_logged (HLVL, ORACLE level only): a constant-twist ego history; the plan is committed to the executed state", "tier": "T1-kin",',
    '    arr_dir = f"{a.raw}/arrays"\n    global MODES\n    MODES = {m: v for m, v in MODES_ALL.items() if all(os.path.exists(f"{OUT_X}/done_{v[1]}_{t}.txt") for t in ("DAGOACC_s0", "DAGOACC_s1", "SOUP_s0", "SOUP_s1"))}\n    if "HLT0" not in MODES:\n        print("REFUSED: the four HLVL-T0 runs are not all done", file=sys.stderr)\n        return 2\n    res = {"hypothesis": "H-LEVELPROBE-1", "modes_read": list(MODES), "lever": "--histswap cv_t0 (HLVL-T0, ORACLE) and cv_pred (HLVL-PRED, vision-derived level): a constant-twist ego history at a per-window constant level; the plan is committed to the executed state", "tier": "T1-kin",')
# KC1
rep('''            if mode == "HCV":
                ok1 = bool(d_in.max() <= TOL)
            else:
                ok1 = bool(d_in[x0].max() <= TOL and (d_in[later] > 0.05).mean() > 0.05)
''', '''            wn = tx["win"]
            dev_const = max(float(np.abs(tx["v0_in"][wn == w_].astype(np.float64) - float(tx["v0_in"][(wn == w_) & x0][0])).max()) for w_ in np.unique(wn))    # per window: v0_in constant over all ticks
            ok1 = bool(dev_const <= TOL and ((mode != "HLT0") or (d_in[x0].max() <= TOL and (d_in[later] > 0.05).mean() > 0.05)))   # HLT0: the constant is the t0 speed (= tick-0 executed) and later executed speeds drift away from it
''')
rep('''            kc1[kk] = {"str_histswap": str(tx["histswap"]),''', '''            kc1[kk] = {"v0_in_max_dev_from_its_tick0_value_over_ticks": dev_const, "str_histswap": str(tx["histswap"]),''')
# outcome / interpretation
rep('''    res["interpretation"] = interpret(res["m1_reading"]["HCV"], res["m1_reading"]["HLVL"])
    json.dump(res, open(f"{a.raw}/HISTSWAP_LF_T1.json", "w", encoding="utf-8", newline="\\n"), indent=1, default=float)''',
    '''    res["card_outcome"] = level_outcome(cells.get("HLT0"), cells.get("HPRED"))
    res["harms_flag"] = {m: {arm: o["outcome"] == "HARMS" for arm, o in res["outcome"][m].items()} for m in MODES}
    try:
        ctx = json.load(open(f"{HLF_PK}/raw/HISTSWAP_LF_T1.json", encoding="utf-8"))
        res["context_histswap_lf"] = {mm: {k_: {"share_closed": c_["m1_cell"]["share_closed"], "better": c_["m1_cell"]["better"]} for k_, c_ in ctx["arms"].items() if k_.endswith(":" + mm)} for mm in ("HCV", "HLVL")}
    except Exception as e:  # noqa: BLE001
        res["context_histswap_lf"] = {"unavailable": repr(e)}
    json.dump(res, open(f"{a.raw}/LEVELPROBE.json", "w", encoding="utf-8", newline="\\n"), indent=1, default=float)''')
rep('''    print("INTERPRETATION", res["interpretation"])''', '''    print("CARD OUTCOME", res["card_outcome"])''')
rep('''    json.dump(res, open(f"{a.raw}/HISTSWAP_LF_T1.json", "w", encoding="utf-8", newline="\\n"), indent=1, default=float)
        print("REFUSED:"''', '''    json.dump(res, open(f"{a.raw}/LEVELPROBE.json", "w", encoding="utf-8", newline="\\n"), indent=1, default=float)
        print("REFUSED:"''')
# the card's rule (pure) before the heavy layer
rep('''def exec_chain_dev(t, m: int = REPLAN_M):''', '''def level_outcome(t0_cells, pred_cells):
    """PREREG_LEVELPROBE.md section A: SUPPORTED iff HLVL-PRED's per-cell share closed is >= 0.30 at >= 3 of the 4 cells AND its mean >= 0.5 x HLVL-T0's mean; NOT SUPPORTED iff its mean < 0.10; else INCONCLUSIVE.
    An undefined share counts as 0.  Returns NOT DEFINED when HLVL-PRED was not run (F1)."""
    def sh(cells):
        return [0.0 if c["share_closed"] is None else float(c["share_closed"]) for c in cells.values()]
    if t0_cells is None or len(t0_cells) != 4:
        return {"outcome": "NOT DEFINED", "reason": "HLVL-T0 cells missing"}
    t0 = sh(t0_cells)
    out = {"t0_mean_share_closed": float(np.mean(t0)), "t0_shares": t0}
    if pred_cells is None or len(pred_cells) != 4:
        out.update({"outcome": "NOT DEFINED", "reason": "HLVL-PRED not run: Stage 1's features cover the v0 < 6 windows only (F1, PREREG_LEVELPROBE.md section C)"})
        return out
    pr = sh(pred_cells)
    n30, mean_pred = int(sum(x >= PRED_CELL_BAR for x in pr)), float(np.mean(pr))
    if n30 >= PRED_CELLS_NEEDED and mean_pred >= PRED_VS_T0 * float(np.mean(t0)):
        o = "SUPPORTED"
    elif mean_pred < NOTSUP_BAR:
        o = "NOT SUPPORTED"
    else:
        o = "INCONCLUSIVE"
    out.update({"outcome": o, "pred_shares": pr, "pred_mean_share_closed": mean_pred, "n_cells_ge_0p30": n30, "ratio_pred_over_t0_mean": (mean_pred / float(np.mean(t0))) if float(np.mean(t0)) > 0 else None})
    return out


def exec_chain_dev(t, m: int = REPLAN_M):''')
out = f"{HERE}/levelprobe_read.py"
open(out, "wb").write(s.encode("utf-8"))
print("wrote", out, hashlib.sha256(s.encode("utf-8")).hexdigest())
