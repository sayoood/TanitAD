"""Control 1 of IMPL_CHOICES_LADDER_READ: the two M2b models on IMAG1B's own eval-141 windows through THIS code path (encode -> lr_lib.roll_cells, context 1 -> lr_lib.health_stats),
compared with imag1b/RESULT_IMAG1B.json ia0 (tolerances fixed in the IMPL file before this ran: MRR 1e-3 abs, chance 1e-9, RankMe 1e-3 rel). Usage: python control_m2b.py [--wrong-seed]  (the flag swaps the latents: must FAIL)."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import lr_lib as L

TOL_MRR, TOL_CHANCE, TOL_RANK = 1e-3, 1e-9, 1e-3
CK = {"M2b_s0": f"{L.ART}/v8f-mini-hacc/M2b_s0_hacc.pt", "M2b_s1": f"{L.ART}/v8f-mini-hacc/M2b_s1_hacc.pt"}
LAT = os.path.join(L.THEORY, "wmact", "latents.pt")
RES = os.path.join(L.THEORY, "imag1b", "RESULT_IMAG1B.json")


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = L.M.V8FMini(L.M.V8FMiniConfig(**ck["cfg"]), L.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * L.M.K30)))
    r = m.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}
    return m.eval()


def run(wrong_seed=False, R=19):
    lat = torch.load(LAT, weights_only=False)
    eid, ok = np.array(lat["eid"]), lat["ok"].numpy()
    sel = np.arange(len(eid))[ok.all(1)]
    qe = eid[sel]
    uniq = np.unique(qe)
    pos = np.searchsorted(uniq, qe)
    W = L.IL.boot_weights(uniq)
    exp = json.load(open(RES))["datasets"]["eval141"]["ia0"]
    rows, passed = [], True
    for tag in CK:
        wm = load_ckpt(CK[tag]).wm
        src = tag if not wrong_seed else ("M2b_s1" if tag == "M2b_s0" else "M2b_s0")
        Z1 = lat["Z1"][src][sel]
        A = wm.scale_actions(lat["A_raw"][sel])[:, :-1]
        with torch.no_grad():
            enc = wm.stack.encode(Z1, A)
        for lv in (1, 2, 3):
            acts = A if lv == 1 else enc.act[lv - 1]
            roll = L.roll_cells(wm.stack, lv, enc.z[lv - 1], acts, [0], 1, 1)[:, 0, 0].numpy()          # [N, D]: the first rollout step from state 0, context 1
            ora = enc.z[lv - 1][:, 1].numpy()
            h = L.health_stats(roll, ora, pos, W, lv, R=R)
            e = exp[f"{tag}|{lv}"]
            row = {"tag": tag, "level": lv, "mrr": h["mrr"], "mrr_expected": e["mrr"], "chance": h["chance"], "chance_expected": e["chance"],
                   "rank_pred": h["rankme"]["pred_k1"], "rank_pred_expected": e["rankme"]["pred_k1"], "rank_ora": h["rankme"]["oracle_k1"], "rank_ora_expected": e["rankme"]["oracle_k1"]}
            row["ok"] = bool(abs(row["mrr"] - row["mrr_expected"]) <= TOL_MRR and abs(row["chance"] - row["chance_expected"]) <= TOL_CHANCE
                             and abs(row["rank_pred"] / row["rank_pred_expected"] - 1) <= TOL_RANK and abs(row["rank_ora"] / row["rank_ora_expected"] - 1) <= TOL_RANK)
            passed &= row["ok"]
            rows.append(row)
    return {"tier": "T0", "wrong_seed_regression": wrong_seed, "tolerances": {"mrr_abs": TOL_MRR, "chance_abs": TOL_CHANCE, "rankme_rel": TOL_RANK}, "n_windows": int(len(sel)), "rows": rows, "pass": bool(passed)}


if __name__ == "__main__":
    ws = "--wrong-seed" in sys.argv
    r = run(ws)
    out = os.path.join(HERE, "CONTROL_M2B_WRONGSEED.json" if ws else "CONTROL_M2B.json")
    json.dump(r, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"pass": r["pass"], "wrong_seed": ws, "max_mrr_abs_diff": max(abs(x["mrr"] - x["mrr_expected"]) for x in r["rows"])}))
    sys.exit(0 if (r["pass"] != ws) else 1)
