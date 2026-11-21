"""nav trace: banked battery nav_cmd vs the v9 per-frame eval label at the same (sid, k = ws + raw_offset)."""
import json, sys, hashlib
import numpy as np
from pathlib import Path

V9 = sys.argv[1]
ROOT = Path(sys.argv[2])
OUT = sys.argv[3]
RAW_OFFSET = 2     # n_stack 3 -> provider row + 2 (refc_v3_train.py:4290-4294; v9 manifest row_key)

z = np.load(V9, allow_pickle=False)
sid = z["clip__sid"]; row0 = z["clip__row0"]; nr = z["clip__n_rows"]; k0 = z["clip__k0"]
k = z["row__k"].astype(np.int64); tok = z["row__nav_token"]
by_sid = {int(s): i for i, s in enumerate(sid)}

# --- known-value control: the label file itself, all rows -----------------------------------------
lab = {"n_clips": int(len(sid)), "n_rows": int(len(k)), "token_counts_all_rows": {}, "clips_with_nonfinite_token": 0}
vals, cnt = np.unique(tok[np.isfinite(tok)], return_counts=True)
lab["token_counts_all_rows"] = {str(int(v)): int(c) for v, c in zip(vals, cnt)}
n_chg = 0; n_lr_clip = 0
for i in range(len(sid)):
    t = tok[row0[i]:row0[i] + nr[i]]
    if not np.all(np.isfinite(t)):
        lab["clips_with_nonfinite_token"] += 1
    u = np.unique(t[np.isfinite(t)])
    n_chg += int(len(u) > 1)
    n_lr_clip += int(np.any(np.isin(u, [1, 2])))
lab["clips_token_changes_within_clip"] = n_chg
lab["clips_with_any_L_or_R_row"] = n_lr_clip

def v9_at(s, kk):
    i = by_sid[int(s)]
    j = int(kk) - int(k0[i])
    if j < 0 or j >= int(nr[i]):
        return None
    r = int(row0[i]) + j
    assert int(k[r]) == int(kk), (s, kk, int(k[r]))      # positive key assertion, not an offset assumption
    return int(tok[r])

res = {"v9_label_file": lab, "dumps": {}}
for dump in sorted(p for p in ROOT.glob("dump_s*") if p.is_dir()):
    man = json.load(open(dump / "manifest.json", encoding="utf-8"))
    d = {"n_episodes": 0, "n_windows": 0, "fed_const_episodes": 0, "v9_changing_episodes_on_banked_windows": 0,
         "windows_fed_eq_v9": 0, "windows_fed_ne_v9": 0, "lr_windows_v9": 0, "lr_windows_fed_wrong": 0,
         "fed_lr_windows_v9_follow": 0, "missing_v9_rows": 0, "confusion_v9_row_fed_col": [[0] * 3 for _ in range(3)],
         "examples_sha12": []}
    for e in man["episodes"]:
        fi = int(e["file_index"]); s = int(e["episode_id"])
        dec = np.load(dump / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        ws = dec["ws"].astype(np.int64); fed = dec["nav_cmd"].astype(np.int64)
        epz = np.load(dump / f"ep{fi:03d}.npz", allow_pickle=False)
        assert np.array_equal(epz["ws"].astype(np.int64), ws)
        assert int(epz["eid"][0]) == int(e["episode_index"]) and s in by_sid
        v9 = []
        for w_, f_ in zip(ws, fed):
            t9 = v9_at(s, int(w_) + RAW_OFFSET)
            if t9 is None:
                d["missing_v9_rows"] += 1; continue
            v9.append(t9)
            d["n_windows"] += 1
            d["confusion_v9_row_fed_col"][t9][int(f_)] += 1
            if t9 == int(f_): d["windows_fed_eq_v9"] += 1
            else: d["windows_fed_ne_v9"] += 1
            if t9 in (1, 2):
                d["lr_windows_v9"] += 1
                d["lr_windows_fed_wrong"] += int(t9 != int(f_))
            if int(f_) in (1, 2) and t9 == 0:
                d["fed_lr_windows_v9_follow"] += 1
        d["n_episodes"] += 1
        d["fed_const_episodes"] += int(len(np.unique(fed)) == 1)
        chg = len(set(v9)) > 1
        d["v9_changing_episodes_on_banked_windows"] += int(chg)
        if chg and len(d["examples_sha12"]) < 5:
            sha = str(z["clip__sha12"][by_sid[s]])
            d["examples_sha12"].append({"episode_index": int(e["episode_index"]), "sha12": sha,
                                        "fed_tokens": sorted(set(int(x) for x in fed)),
                                        "v9_tokens": sorted(set(v9))})
    nw = max(d["n_windows"], 1)
    d["frac_windows_fed_ne_v9"] = round(d["windows_fed_ne_v9"] / nw, 4)
    d["frac_lr_windows_fed_wrong"] = round(d["lr_windows_fed_wrong"] / max(d["lr_windows_v9"], 1), 4)
    res["dumps"][dump.name] = d
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
