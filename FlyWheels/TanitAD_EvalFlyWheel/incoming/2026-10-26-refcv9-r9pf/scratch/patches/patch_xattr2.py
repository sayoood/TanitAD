from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/x_attrib_r9pf.py"); s = p.read_text(encoding="utf-8")
rep = [
("""        cH = comps(H["S6_pos"], H["S2_pos"])""",
 """        cH = comps(H["S6_pos"], H["S2_pos"])
        FXp = np.load(LANDED / "FACTORED_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
        D8p = np.load(LANDED / "RETIME_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
        fdF, mdF = L2.lon_indicators(FXp["S6_pos"].astype(np.float32), E, comp6)[:2]
        cD, cHP = comps(D8p["S6_pos"], D8p["S2_pos"]), comps(I["S6_HP"], I["S2_HP"])"""),
("""            hyb2 = np.where(""", """            taus = json.loads((RC.SCRATCH / "fits" / f"{tag}_tau.json").read_text(encoding="utf-8"))
            with torch.no_grad():
                lg = torch.from_numpy(o["regime_logits"])
                dec_tau = R.decide(lg, float(taus["tau"])).numpy()
                ds_soft = (lg.softmax(-1)[:, :, None] * prof).sum(1)
                soft = R.retime(torch.from_numpy(np.ascontiguousarray(plan2, np.float32)), R.monotone_progress(s_p + ds_soft)).to(torch.float32).numpy()
            tp = plan_for(dec_tau, all2)
            arms["TAU"] = {"p8_2": tp, "p8_6": tp[j6], "tau": float(taus["tau"])}
            arms["SOFT"] = {"p8_2": soft, "p8_6": soft[j6]}
            hyb2 = np.where("""),
("""                      "ADE_S6_vs_H8": cell(PR, cA["S6"]["ade_m"][comp6], cH["S6"]["ade_m"][comp6], em6)}""",
 """                      "ADE_S6_vs_H8": cell(PR, cA["S6"]["ade_m"][comp6], cH["S6"]["ade_m"][comp6], em6),
                      "MD_vs_FX": cell(PR, mdA, mdF, eid6), "FD_vs_FX": cell(PR, fdA, fdF, eid6),
                      "ADE_S6_vs_D8": cell(PR, cA["S6"]["ade_m"][comp6], cD["S6"]["ade_m"][comp6], em6),
                      "ADE_S6_vs_HP": cell(PR, cA["S6"]["ade_m"][comp6], cHP["S6"]["ade_m"][comp6], em6)}
                if "tau" in d:
                    r_["tau"] = d["tau"]"""),
("""                    r_["ADE_S2_vs_H8"] = cell(PR, cA["S2"]["ade_m"][m2], cH["S2"]["ade_m"][m2], em2)""",
 """                    r_["ADE_S2_vs_H8"] = cell(PR, cA["S2"]["ade_m"][m2], cH["S2"]["ade_m"][m2], em2)
                    r_["ADE_S2_vs_D8"] = cell(PR, cA["S2"]["ade_m"][m2], cD["S2"]["ade_m"][m2], em2)
                    r_["ADE_S2_vs_HP"] = cell(PR, cA["S2"]["ade_m"][m2], cHP["S2"]["ade_m"][m2], em2)"""),
("""                r_["rows_separated_worse_than_H8"] = worse""",
 """                r_["rows_separated_worse_than_H8"] = worse
                r_["barlike_EXPLORATORY"] = {
                    "PA-1a": bool(sep_below(r_["MD_vs_H8"]) and not sep_above(r_["FD_vs_H8"])),
                    "PA-1b": bool(sep_below(r_["FD_vs_FX"]) and not sep_above(r_["MD_vs_FX"])),
                    "PA-2": (bool(sep_below(r_["ADE_S2_vs_D8"]) and sep_below(r_["ADE_S6_vs_D8"]) and sep_below(r_["ADE_S2_vs_HP"])
                                  and sep_below(r_["ADE_S6_vs_HP"])) if "ADE_S2_vs_D8" in r_ else None),
                    "PA-G": (not worse) if "p8_2" in d else None}"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
s = s.replace('  HYB       H2CN_DEPLOY8', '  TAU       the prereg s1 EXPLORATORY calibration-tuned decel tau (x_tau_fit.py, OOF Youden J) on the registered head\n  SOFT      the posterior-weighted profile sum_r p(r) dS[r] (no hard decision)\n  HYB       H2CN_DEPLOY8')
p.write_text(s, encoding="utf-8"); print("ok")
