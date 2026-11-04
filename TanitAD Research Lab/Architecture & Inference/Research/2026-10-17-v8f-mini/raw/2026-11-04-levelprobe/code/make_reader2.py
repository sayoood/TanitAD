"""make_reader2.py -- derive levelprobe2_read.py (PREREG_LEVELPROBE2.md) from levelprobe_read.py by exact-match replacements (each must match once).  Arms: MEAN6 (M6), FUT3 (F3), FUT1 (F1), all --histswap cv_pred.
Keeps every known answer of HISTSWAP-LF and the 19 rows; adds KT0 (the L_T0 level file reproduces HLVL-T0 bit for bit), KC1's level-file match, and the card's TARGET-SPEED SUPPORTED / PROFILE / INTERMEDIATE rule.  python -I code/make_reader2.py"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
s = open(f"{HERE}/levelprobe_read.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('MODES_ALL = {"HLT0": ("cv_t0", "HT0"), "HPRED": ("cv_pred", "HPR")}', 'MODES_ALL = {"MEAN6": ("cv_pred", "M6"), "FUT3": ("cv_pred", "F3"), "FUT1": ("cv_pred", "F1")}')
rep('MODES = dict(MODES_ALL)\n', 'MODES = dict(MODES_ALL)\nLV = "F:/TanitAD-scratch/levelprobe_scratch/levels"\nLEVEL_FILES = {"MEAN6": f"{LV}/L_MEAN6.npz", "FUT3": f"{LV}/L_FUT3.npz", "FUT1": f"{LV}/L_FUT1.npz"}\nT0_REF_TAG, KT0_TAG = "HT0_DAGOACC_s0", "LP2KA_DAGOACC_s0"\n')
rep('    if "HLT0" not in MODES:\n        print("REFUSED: the four HLVL-T0 runs are not all done", file=sys.stderr)\n        return 2\n', '    if len(MODES) != 3:\n        print("REFUSED: the twelve MEAN6 / FUT3 / FUT1 runs are not all done", file=sys.stderr)\n        return 2\n')
rep('open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_LEVELPROBE.md sha256")', 'open(f"{PK}/FROZEN_SHA2.txt", encoding="utf-8") if ln.startswith("PREREG_LEVELPROBE2.md sha256")')
rep('got = SR.sha256_file(f"{PK}/PREREG_LEVELPROBE.md")', 'got = SR.sha256_file(f"{PK}/PREREG_LEVELPROBE2.md")')
rep('REFUSED: PREREG_LEVELPROBE.md {got}', 'REFUSED: PREREG_LEVELPROBE2.md {got}')
rep('"hypothesis": "H-LEVELPROBE-1"', '"hypothesis": "H-LEVELPROBE-2"')
rep('{a.raw}/LEVELPROBE.json', '{a.raw}/LEVELPROBE2.json', count=2)
rep('    kc1, kc2, kc3, kf = {}, {}, {}, {}\n', '''    _, dka = SR.load_dump(f"{OUT_X}/{KT0_TAG}/dump_{KT0_TAG}")
    _, dt0r = SR.load_dump(f"{OUT_X}/{T0_REF_TAG}/dump_{T0_REF_TAG}")
    tka = {q: v for q, v in tickload(f"{OUT_X}/{KT0_TAG}/ticks/ticks_{KT0_TAG}.npz").items() if q != "histswap"}
    tt0 = {q: v for q, v in tickload(f"{OUT_X}/{T0_REF_TAG}/ticks/ticks_{T0_REF_TAG}.npz").items() if q != "histswap"}
    sd, dd, _, bd, od = SR.cmp_arrays(dka, dt0r)
    st_, dt_, _, bt_, ot_ = SR.cmp_arrays(tka, tt0)
    res["controls"]["KT0_level_file_of_t0_speeds_reproduces_HLVL_T0_bit_for_bit"] = {"run": KT0_TAG, "reference": T0_REF_TAG, "dump_bit_identical": sd, "dump_max_abs_dev": dd, "dump_differing_keys": bd, "ticks_bit_identical": st_, "ticks_max_abs_dev": dt_,
                                                                                    "ticks_differing_keys": bt_, "key_set_diff": od + ot_, "PASS": bool(sd and st_ and not od and not ot_)}
    kc1, kc2, kc3, kf = {}, {}, {}, {}
''')
rep('            ok1 = bool(dev_const <= TOL and ((mode != "HLT0") or (d_in[x0].max() <= TOL and (d_in[later] > 0.05).mean() > 0.05)))',
    '''            lz = np.load(LEVEL_FILES[mode])
            lmap = {(int(c_), int(t_)): float(v_) for c_, t_, v_ in zip(lz["clip_index"], lz["ws"], lz["level"])}
            lev_dev = float(np.abs(tx["v0_in"].astype(np.float64) - np.array([lmap[(int(c_), int(t_))] for c_, t_ in zip(tx["ci"], tx["t0"])])).max())     # the network's input speed IS the file's level at every tick
            ok1 = bool(dev_const <= TOL and lev_dev <= 2e-4)''')
rep('kc1[kk] = {"v0_in_max_dev_from_its_tick0_value_over_ticks": dev_const,', 'kc1[kk] = {"level_file_max_dev": lev_dev, "v0_in_max_dev_from_its_tick0_value_over_ticks": dev_const,')
rep('res["card_outcome"] = level_outcome(cells.get("HLT0"), cells.get("HPRED"))', 'res["card_outcome"] = level_outcome2(cells)')
rep('def level_outcome(t0_cells, pred_cells):', '''def level_outcome2(cells):
    """PREREG_LEVELPROBE2.md section A: TARGET-SPEED SUPPORTED iff MEAN6's per-cell share closed is >= 0.30 at >= 3 of the 4 cells; PROFILE iff MEAN6, FUT1 and FUT3 all have mean share closed < 0.10; INTERMEDIATE otherwise.
    An undefined share counts as 0.  If both of the first two hold, both are reported."""
    sh = {m: [0.0 if c["share_closed"] is None else float(c["share_closed"]) for c in cs.values()] for m, cs in cells.items()}
    mean = {m: float(np.mean(v)) for m, v in sh.items()}
    n30 = {m: int(sum(x >= 0.30 for x in v)) for m, v in sh.items()}
    sup, prof = n30["MEAN6"] >= 3, all(mean[m] < 0.10 for m in ("MEAN6", "FUT1", "FUT3"))
    out = "TARGET-SPEED SUPPORTED and PROFILE (both hold)" if (sup and prof) else ("TARGET-SPEED SUPPORTED" if sup else ("PROFILE" if prof else "INTERMEDIATE"))
    return {"outcome": out, "target_speed_supported": bool(sup), "profile": bool(prof), "shares": sh, "mean_share_closed": mean, "n_cells_ge_0p30": n30, "horizon_pattern_mean_share": {m: mean[m] for m in ("FUT1", "FUT3", "MEAN6")}}


def level_outcome(t0_cells, pred_cells):''')
rep('        res["context_histswap_lf"] = {mm:', '        res["context_hlvl_t0_from_levelprobe1"] = json.load(open(f"{PK}/raw/LEVELPROBE.json", encoding="utf-8"))["m1_reading"]["HLT0"]\n        res["context_histswap_lf"] = {mm:')
out = f"{HERE}/levelprobe2_read.py"
open(out, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", out, hashlib.sha256(s.encode("utf-8")).hexdigest())
