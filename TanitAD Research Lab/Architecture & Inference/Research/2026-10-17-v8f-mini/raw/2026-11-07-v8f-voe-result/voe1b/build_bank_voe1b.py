"""VOE1B step 1 (VOE1's build_bank_voe1.py, copied; CPU, minutes; labels and latents only, NO surprise statistic): one pass over the 141 eval clips.
Writes the bank (default F:/TanitAD-scratch/voe1b_bank/bank_voe1b.pt, env VOE1B_BANK): per clip frame the level-1 latent of every tag (M2b s0/s1 and the 32 untrained
tags UT{chassis}_{100..115}), the logged actions, the native-track speed, the lead labels and the event onsets; the window table of latents.pt; the reproduction check
against latents.pt. Command: see HANDOFF_VOE1B.md. The bank is re-creatable and is deleted at the end of the package."""
import json
import os
import sys
import time

import numpy as np
import torch

import voe1b_core as C
import voe1b_real as R
from voe1b_real import D, W

t_start = time.time()
OUT = R.BANK_PATH
TAGS = R.tags_for()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ds, wins, eid = W.open_windows(16)
store = ds.store
L = torch.load(os.path.join(R.HERE, "..", "wmact", "latents.pt"), weights_only=False)
eid_l, t_l, ok_l = np.array(L["eid"]), np.asarray(L["t"]), L["ok"].numpy()
assert eid_l.tolist() == eid.tolist(), "latents.pt window order != open_windows order"
clips = sorted(set(eid.tolist()))

trained = R.load_trained()
wms = R.build_wms(trained)
rep = R.reinit_report(trained, wms)
chassis = {s: trained[s][0] for s in (0, 1)}
tracks = D.load_native_tracks(f"{W.SF}/inputs/native_tracks.pt")
J = D.load_join(f"{W.RAW}/join_eval141.npz")
lead = D.lead_label_arrays(J, np.load(f"{W.ART}/v9labels/inputs/lead_eval139.npz", allow_pickle=True))
ci_of = {str(c): i for i, c in enumerate(J["clip"])}

off, Tn = {}, {}
Z = {t: [] for t in TAGS}
ACT, SPD, VC, NEAR, GAP, EVB, EVT, EVL = [], [], [], [], [], [], [], []
pos = 0
for k, clip in enumerate(clips):
    d = store.get(clip)
    T = len(d["pose"])
    z = R.clip_z1(wms, chassis, d["feats"])
    for t in TAGS:
        Z[t].append(z[t].numpy().astype(np.float32))
    act = np.stack([np.nan_to_num(d["a"]), d["ch1"]], -1).astype(np.float32)           # exactly V8FMiniWindows.item's "act"
    tr = tracks[clip]
    spd = tr[0::2, 3].astype(np.float32)
    assert len(spd) == T, (clip, len(spd), T)
    c = ci_of[clip]
    assert int(J["n"][c]) == T, (clip, int(J["n"][c]), T)
    rows = int(J["off"][c]) + np.arange(T)
    near = (lead["has"][rows] & lead["near"][rows])
    gap = np.where(near, D.LEAD_RANGE_M * lead["y"][rows], np.nan).astype(np.float32)
    ACT.append(act); SPD.append(spd); VC.append(np.asarray(d["v"], np.float32)); NEAR.append(near); GAP.append(gap)
    EVB.append(C.onset(act[:, 0] < -2.0))
    EVT.append(C.onset(np.abs(act[:, 1]) > 0.1))
    EVL.append(C.onset(C.lead_event_cond(near, gap)))
    off[clip], Tn[clip] = pos, T
    pos += T
    if (k + 1) % 20 == 0:
        print(f"{k + 1}/{len(clips)} clips {time.time() - t_start:.0f}s", flush=True)

Zb = {t: torch.from_numpy(np.concatenate(v)) for t, v in Z.items()}
ACTb, SPDb, VCb = (np.concatenate(x) for x in (ACT, SPD, VC))
NEARb, GAPb, EVBb, EVTb, EVLb = (np.concatenate(x) for x in (NEAR, GAP, EVB, EVT, EVL))

# ---- the window table and the reproduction check against latents.pt (the unmodified path must reproduce it)
keep = ok_l.all(1)
base = np.array([off[e] + int(t) - 15 for e, t in zip(eid_l, t_l)])
g0 = np.array([int(t) - 15 for t in t_l])
idx = base[:, None] + np.arange(16)[None]
chk = {}
for t in R.DECIDING:
    zz = Zb[t][torch.from_numpy(idx[keep])]
    ref = L["Z1"][t][torch.from_numpy(np.flatnonzero(keep))]
    chk[t] = {"max_abs_diff_z1": float((zz - ref).abs().max()), "ref_abs_max": float(ref.abs().max()), "n_windows": int(keep.sum())}
a_ref = L["A_raw"].numpy()[keep]
chk["actions_max_abs_diff"] = float(np.abs(ACTb[idx[keep]] - a_ref).max())
chk["speed_native_vs_cache_v_max_abs_diff"] = float(np.nanmax(np.abs(SPDb - VCb)))
chk["g0_min"] = int(g0[keep].min())
chk["teleport_max_frame_vs_clip_end_margin_min"] = int(min(Tn[e] - 1 - (int(g) + 20) for e, g, k_ in zip(eid_l, g0, keep) if k_))
print(json.dumps(chk, indent=1))
torch.save({"clips": clips, "off": off, "T": Tn, "z1": Zb, "act": torch.from_numpy(ACTb), "speed": torch.from_numpy(SPDb), "v_cache": torch.from_numpy(VCb),
            "near": NEARb, "gap": GAPb, "ev_brake": EVBb, "ev_turn": EVTb, "ev_lead": EVLb,
            "win": {"eid": eid_l.tolist(), "t": t_l, "keep": keep, "base": base, "g0": g0},
            "reinit": rep, "check": chk, "tanitad_file": W.tanitad.__file__}, OUT)
json.dump({"reinit": rep, "check": chk, "events": {"brake": int(EVBb.sum()), "turn": int(EVTb.sum()), "lead": int(EVLb.sum())},
           "n_frames": int(pos), "n_clips": len(clips), "seconds": round(time.time() - t_start, 1)},
          open(os.path.join(R.HERE, "bank_voe1b.report.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print("saved", OUT, round(time.time() - t_start, 1), "s")
