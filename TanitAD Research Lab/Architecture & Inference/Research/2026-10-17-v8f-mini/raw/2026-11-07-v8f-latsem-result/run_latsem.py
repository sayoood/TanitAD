"""PREREG_V8F_LATSEM (frozen prefix 4,882 bytes, sha256 0b0c2f73...): what each WM level encodes (R1 probes), imagines (R2 retrieval)
and whether its imagined future follows the action (R3 counterfactuals), with the card's four controls and pictures. Tier T0, CPU only.
Run from theory/latsem with WMACT_CODE_ROOT set (as for theory/wmact). Writes RESULT_LATSEM.json and fig/*.png."""
import hashlib, io, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402

CARD, CARD_N, CARD_SHA = "../PREREG_V8F_LATSEM.md", 4882, "0b0c2f73c4e5db9c72f25ee9a1df1b2b883988879c410dadfd4b6b1d1e7b1823"
assert hashlib.sha256(open(CARD, "rb").read()[:CARD_N]).hexdigest() == CARD_SHA, "card prefix changed"
FR = "E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
TAGS = ("M2b_s0", "M2b_s1")
STRIDE, KMAX, HSEC = {1: 1, 2: 2, 3: 4}, {1: 10, 2: 3, 3: 3}, {1: 2.0, 2: 1.2, 3: 2.4}
LAM, TARGETS = 10.0, ("v", "a", "w", "dv2", "dpsi2")
ACTS = {"brake": (-3.0, 0.0), "hold": (0.0, 0.0), "accel": (1.5, 0.0), "w+": (0.0, 0.25), "w-": (0.0, -0.25)}
F = 16
t0 = time.time()
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

# ------------------------------------------------------------------------------------------------ data
L = torch.load("../wmact/latents.pt", weights_only=False)
eid, tw, ok, A_raw = np.array(L["eid"]), np.asarray(L["t"]), L["ok"].numpy(), L["A_raw"]
have = {f.split(".")[0] for f in os.listdir(FR) if f.endswith(".v2ep.pt")}
wkeep = np.flatnonzero(np.isin(eid, sorted(have)))
TR = {k: np.asarray(v, np.float64) for k, v in torch.load("E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt",
                                                            weights_only=False)["tracks"].items()}
G = tw[:, None] - 15 + np.arange(F)[None]                                       # cache step of each window frame

key2b, occ_w, occ_f, occ_b = {}, [], [], []
for i in wkeep:
    for f in range(F):
        if ok[i, f] and G[i, f] >= 0:
            k = (eid[i], int(G[i, f]))
            b = key2b.setdefault(k, len(key2b))
            occ_w.append(i); occ_f.append(f); occ_b.append(b)
occ_w, occ_f, occ_b = map(np.asarray, (occ_w, occ_f, occ_b))
first = np.full(len(key2b), -1)
for j in range(len(occ_b) - 1, -1, -1):
    first[occ_b[j]] = j
bw, bf = occ_w[first], occ_f[first]
beid = np.array([k[0] for k in key2b]); bg = np.array([k[1] for k in key2b])


def track_labels(e, g):
    tr = TR[e]; r = 2 * int(g)
    v, psi = tr[r, 3], tr[r, 2]
    if r + 20 < len(tr) and np.isfinite(tr[r + 20, 3]):
        return v, psi, tr[r + 20, 3] - v, wrap(tr[r + 20, 2] - psi)
    return v, psi, np.nan, np.nan


lab = np.array([track_labels(e, g) for e, g in zip(beid, bg)])
Y = {"v": lab[:, 0], "a": A_raw[bw, bf, 0].numpy().astype(np.float64), "w": A_raw[bw, bf, 1].numpy().astype(np.float64),
     "dv2": lab[:, 2], "dpsi2": lab[:, 3]}
fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in beid])
res = {"card": {"file": "PREREG_V8F_LATSEM.md", "prefix_bytes": CARD_N, "sha256": CARD_SHA}, "tier": "T0",
       "n_bank_frames": int(len(beid)), "n_episodes": int(len(set(beid))), "controls": {}, "R1": {}, "R2": {}, "R3": {}}
ctrl = res["controls"]

# control 1: every copy of one (episode, g) has identical z1 and raw actions
c1 = 0.0                                                                          # AMENDMENT_LATSEM_1: z1 <= 1e-5 (float32 batch rounding), actions exactly 0
for tag in TAGS:
    Z = L["Z1"][tag]
    c1 = max(c1, float((Z[occ_w, occ_f] - Z[bw[occ_b], bf[occ_b]]).abs().max()))
ctrl["C1_copy_maxdiff_z1"] = c1
ctrl["C1_copy_maxdiff_actions"] = float((A_raw[occ_w, occ_f] - A_raw[bw[occ_b], bf[occ_b]]).abs().max())

uniq, Wt = W.boot_weights(beid)
pos = np.searchsorted(uniq, beid)


def esum(x, p=pos, n=None):
    s = np.zeros(n or len(uniq)); np.add.at(s, p, np.nan_to_num(np.asarray(x, np.float64))); return s


def draws(num_pos, num_neg, den):
    return (Wt @ num_pos - Wt @ num_neg) / (Wt @ den)


def row(num_pos, num_neg, den, n):
    p = float((num_pos.sum() - num_neg.sum()) / den.sum()); b = draws(num_pos, num_neg, den)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"point": p, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n": int(n), "n_episodes": int(len(uniq))}, b


def ridge_fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd; ym = y.mean()
    beta = np.linalg.solve(Xs.T @ Xs + LAM * np.eye(X.shape[1]), Xs.T @ (y - ym))
    return mu, sd, beta, ym


def ridge_pred(P, X):
    mu, sd, beta, ym = P
    return ((X - mu) / sd) @ beta + ym


def perm_labels(y, ep, rng):
    """labels permuted across episodes: each episode takes ANOTHER episode's label sequence (cycled to its length)."""
    out = np.empty_like(y); order = np.unique(ep)[rng.permutation(len(np.unique(ep)))]
    for e, e2 in zip(order, np.roll(order, 1)):                                  # a cyclic shift: no episode keeps its own labels
        src = y[ep == e2]; n = int((ep == e).sum()); out[ep == e] = src[np.arange(n) % len(src)]
    return out


# ------------------------------------------------------------------------------------------------ per model
PROBE, BZ, models = {}, {}, {}
R2draws = {}
for tag in TAGS:
    m, cfg, ck = W.load_model(W.CKPT[tag]); models[tag] = m
    st = m.wm.stack
    with torch.no_grad():
        zs = st.initial_latents(L["Z1"][tag][bw, bf])
    BZ[tag] = {l + 1: zs[l].double().numpy() for l in range(3)}
    # control 2: per-frame levels equal encode's levels at matched positions
    sel_c = wkeep[ok[wkeep].all(1)]
    with torch.no_grad():
        enc = st.encode(L["Z1"][tag][sel_c], m.wm.scale_actions(A_raw[sel_c])[:, :-1])
    c2 = 0.0
    for l in (2, 3):
        zl = enc.z[l - 1].double().numpy()
        for i in range(zl.shape[1]):
            b = np.array([key2b[(eid[w], int(G[w, i * STRIDE[l]]))] for w in sel_c])
            c2 = max(c2, float(np.abs(zl[:, i] - BZ[tag][l][b]).max()))
    ctrl[f"C2_level_maxdiff/{tag}"] = c2
    # R1 probes, cross-fitted by episode fold
    rng = np.random.default_rng(0)
    for l in (1, 2, 3):
        X = BZ[tag][l]
        for q in TARGETS:
            y = Y[q]; val = np.isfinite(y); pred = np.full(len(y), np.nan); base = np.full(len(y), np.nan); pshuf = np.full(len(y), np.nan)
            for k in (0, 1):
                fit, ev = val & (fold != k), val & (fold == k)
                P = ridge_fit(X[fit], y[fit]); PROBE[(tag, l, q, k)] = P
                pred[ev] = ridge_pred(P, X[ev]); base[ev] = P[3]
                Ps = ridge_fit(X[fit], perm_labels(y[fit], beid[fit], rng)); pshuf[ev] = ridge_pred(Ps, X[ev])
            sst, sse, ssh = esum((y - base) ** 2), esum((pred - y) ** 2), esum((pshuf - y) ** 2)
            r, b = row(sst, sse, sst, val.sum()); R2draws[(tag, l, q)] = b
            rs, _ = row(sst, ssh, sst, val.sum())
            res["R1"][f"{tag}/L{l}/{q}"] = {"R2": r, "R2_shuffled": rs}
    dd = (R2draws[(tag, 3, "a")] - R2draws[(tag, 1, "a")]) - (R2draws[(tag, 3, "v")] - R2draws[(tag, 1, "v")])
    pt = lambda l, q: res["R1"][f"{tag}/L{l}/{q}"]["R2"]["point"]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    res["R1"][f"{tag}/DD"] = {"point": pt(3, "a") - pt(1, "a") - pt(3, "v") + pt(1, "v"), "lo": float(lo), "hi": float(hi),
                              "separated": bool(lo > 0 or hi < 0)}
ctrl["C4_shuffled_any_separated_above_0"] = any(v["R2_shuffled"]["lo"] > 0 for k, v in res["R1"].items() if "/DD" not in k)

# ------------------------------------------------------------------------------------------------ R2 retrieval and R3 counterfactuals
sel = wkeep[ok[wkeep].all(1)]
qe = eid[sel]; qpos = np.searchsorted(uniq, qe); qfold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in qe])
SHR = json.load(open("../wmact/RESULT_WMACT_SHRINK.json", encoding="utf-8"))["rows"]


def nn_loo(Q, B, qeid):
    """index of the nearest bank frame (squared L2) from another episode, per query row."""
    out = np.empty(len(Q), np.int64); b2 = (B * B).sum(1)
    for s in range(0, len(Q), 256):
        q = Q[s:s + 256]; d = (q * q).sum(1)[:, None] - 2 * q @ B.T + b2[None]
        d[beid[None, :] == qeid[s:s + 256, None]] = np.inf
        out[s:s + 256] = d.argmin(1)
    return out


def true_label(q, i_rows, frame):
    if q == "v":
        return np.array([TR[eid[i]][2 * int(G[i, frame]), 3] for i in i_rows])
    return A_raw[i_rows, frame, 1].numpy().astype(np.float64)


def decode(tag, l, q, Z):
    out = np.empty(len(Z))
    for k in (0, 1):
        mk = qfold == k
        out[mk] = ridge_pred(PROBE[(tag, l, q, k)], Z[mk])
    return out


NNSAVE, CFSAVE, CFV = {}, {}, {}
for tag in TAGS:
    m = models[tag]; wm = m.wm
    with torch.no_grad():
        pk = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(A_raw[sel])[:, :-1])
    for l, ks in W.KS.items():
        B = BZ[tag][l]; z0 = pk["true"][l][:, 0].double().numpy()
        nn0 = nn_loo(z0, B, qe)
        for k in ks:
            zt = pk["true"][l][:, k].double().numpy(); zh = pk["pred"][(l, k)].double().numpy()
            qs = {"raw": zh, "oracle": zt}
            if l == 1:
                c = SHR[f"{tag}/L1/k{k}"]["c_train"]; qs["shrunk"] = z0 + c * (zh - z0)
            qs["start"] = z0
            nns = {nm: (nn0 if nm == "start" else nn_loo(Z, B, qe)) for nm, Z in qs.items()}
            NNSAVE[(tag, l, k)] = nns
            for q in ("v", "w"):
                ystar = true_label(q, sel, k * STRIDE[l]); e0 = esum((Y[q][nn0] - ystar) ** 2, qpos)
                for nm, nb in nns.items():
                    e1 = esum((Y[q][nb] - ystar) ** 2, qpos)
                    r, _ = row(e0, e1, e0, len(sel))
                    res["R2"][f"{tag}/L{l}/k{k}/{q}/{nm}"] = r
    # R3: constant synthetic actions
    n = len(sel); vh, wh = {}, {}
    for nm, (a, w) in list(ACTS.items()) + [("brake_twice", ACTS["brake"])]:
        Araw = torch.tensor([a, w], dtype=torch.float32).expand(n, F, 2).contiguous()
        with torch.no_grad():
            pkc = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(Araw)[:, :-1])
        for l in (1, 2, 3):
            Z = pkc["pred"][(l, KMAX[l])].double().numpy()
            vh[(nm, l)], wh[(nm, l)] = decode(tag, l, "v", Z), decode(tag, l, "w", Z)
            CFV[(tag, nm, l)] = vh[(nm, l)]
            CFSAVE[(tag, nm, l)] = Z
    one = esum(np.ones(n), qpos)
    v0 = true_label("v", sel, 0)
    for l in (1, 2, 3):
        h = HSEC[l]
        r, _ = row(esum(vh[("accel", l)] - vh[("brake", l)], qpos), np.zeros(len(uniq)), one, n)
        res["R3"][f"{tag}/L{l}/CF_LON"] = r
        r, _ = row(esum(wh[("w+", l)] - wh[("w-", l)], qpos), np.zeros(len(uniq)), one, n)
        res["R3"][f"{tag}/L{l}/CF_LAT"] = r
        res["R3"][f"{tag}/L{l}/kinematic_ref_LON"] = float(np.mean(v0 + 1.5 * h - np.maximum(v0 - 3.0 * h, 0.0)))
        res["R3"][f"{tag}/L{l}/decoded_means"] = {nm: {"v": float(vh[(nm, l)].mean()), "w": float(wh[(nm, l)].mean())} for nm in ACTS}
        ctrl[f"C3_cf_identical/{tag}/L{l}"] = float(np.abs(vh[("brake_twice", l)] - vh[("brake", l)]).max())
ctrl["C3_start_query_skill"] = max(abs(v["point"]) for k, v in res["R2"].items() if k.endswith("/start"))
# AMENDMENT_LATSEM_2 (frozen before this statistic existed): SPEED-SKILL through the cross-fitted R1 speed probe; CF-LON split (exploratory)
res["A2"], res["A2_exploratory"] = {}, {}
for tag in TAGS:
    wm = models[tag].wm
    with torch.no_grad():
        pk = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(A_raw[sel])[:, :-1])
    for l, ks in W.KS.items():
        z0 = pk["true"][l][:, 0].double().numpy(); v0h = decode(tag, l, "v", z0)
        for k in ks:
            vstar = true_label("v", sel, k * STRIDE[l]); e0 = esum((v0h - vstar) ** 2, qpos)
            zh, zt = pk["pred"][(l, k)].double().numpy(), pk["true"][l][:, k].double().numpy()
            qs = {"raw": zh, "oracle": zt, "start": z0}
            if l == 1:
                qs["shrunk"] = z0 + SHR[f"{tag}/L1/k{k}"]["c_train"] * (zh - z0)
            for nm, Z in qs.items():
                r, _ = row(e0, esum((decode(tag, l, "v", Z) - vstar) ** 2, qpos), e0, len(sel))
                res["A2"][f"{tag}/L{l}/k{k}/{nm}"] = r
ctrl["A2_start_query_skill"] = max(abs(v["point"]) for k, v in res["A2"].items() if k.endswith("/start"))
_v0 = true_label("v", sel, 0)
for tag in TAGS:
    for l in (1, 2, 3):
        dm = res["R3"][f"{tag}/L{l}/decoded_means"]
        for lo_, hi_, nm in ((-1, 3, "v0<3"), (3, 8, "v0 3-8"), (8, 99, "v0>=8")):
            mk = (_v0 >= lo_) & (_v0 < hi_)
            res["A2_exploratory"][f"{tag}/L{l}/{nm}"] = {"n": int(mk.sum()),
                "accel_minus_hold": float((CFV[(tag, "accel", l)] - CFV[(tag, "hold", l)])[mk].mean()),
                "hold_minus_brake": float((CFV[(tag, "hold", l)] - CFV[(tag, "brake", l)])[mk].mean())}


ok_ctrl = (ctrl["C1_copy_maxdiff_z1"] <= 1e-5 and ctrl["C1_copy_maxdiff_actions"] == 0.0 and all(v <= 1e-5 for k, v in ctrl.items() if k.startswith("C2_"))
           and ctrl["C3_start_query_skill"] == 0.0 and ctrl["A2_start_query_skill"] == 0.0 and all(v == 0.0 for k, v in ctrl.items() if k.startswith("C3_cf_"))
           and not ctrl["C4_shuffled_any_separated_above_0"])
res["controls_pass"] = bool(ok_ctrl)


# ------------------------------------------------------------------------------------------------ outcomes
def both(key, test):
    return all(test(res[key[0]][key[1].format(tag=t)]) for t in TAGS)


pos_sep = lambda r: r["separated"] and r["point"] > 0
out = {}
if not ok_ctrl:
    out["all"] = "INVALID (a control failed)"
else:
    for l in (1, 2, 3):
        for q in TARGETS:
            out[f"R1/L{l}/{q}"] = "ENCODES" if both(("R1", "{tag}/L%d/%s" % (l, q)), lambda r: pos_sep(r["R2"])) else "NOT-ENCODES"
    dds = [res["R1"][f"{t}/DD"] for t in TAGS]
    out["R1/TD"] = ("TD-CONSISTENT" if all(d["separated"] and d["point"] < 0 for d in dds) else
                    "TD-INCONSISTENT" if all(d["separated"] and d["point"] > 0 for d in dds) else "TD-NOT-RESOLVED")
    for l, ks in W.KS.items():
        for k in ks:
            nm = "shrunk" if l == 1 else "raw"
            out[f"R2/L{l}/k{k}/v"] = "IMAGINES-SPEED" if both(("R2", "{tag}/L%d/k%d/v/%s" % (l, k, nm)), pos_sep) else "NOT"
    for l, ks in W.KS.items():
        for k in ks:
            nm = "shrunk" if l == 1 else "raw"
            out[f"A2/L{l}/k{k}"] = "DECODES-FUTURE-SPEED" if both(("A2", "{tag}/L%d/k%d/%s" % (l, k, nm)), pos_sep) else "NOT"
    for l in (1, 2, 3):
        for ax in ("LON", "LAT"):
            rs = [res["R3"][f"{t}/L{l}/CF_{ax}"] for t in TAGS]
            out[f"R3/L{l}/{ax}"] = ("ACTION-SEMANTIC" if all(pos_sep(r) for r in rs) else
                                    "ACTION-BLIND" if not any(r["separated"] for r in rs) else "MIXED")
res["outcomes"] = out
res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open("RESULT_LATSEM.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
print("controls_pass", ok_ctrl, json.dumps(ctrl, default=float))
print(json.dumps(out, indent=0))

# ------------------------------------------------------------------------------------------------ pictures (decide nothing; seed s0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

os.makedirs("fig", exist_ok=True)
tag = "M2b_s0"


def lab_at(i, f):
    tr = TR[eid[i]]; r = 2 * int(G[i, f]); return tr[r, 3], tr[r, 2]


v0s = np.array([lab_at(i, 0)[0] for i in sel]); v12 = np.array([lab_at(i, 12)[0] for i in sel])
dpsi = np.array([wrap(lab_at(i, 12)[1] - lab_at(i, 0)[1]) for i in sel]); dv = v12 - v0s
cru = np.flatnonzero((np.abs(dpsi) < 0.05) & (np.abs(dv) < 0.5))
pick = {"STOP": int(np.argmin(dv)), "LAUNCH": int(np.argmax(dv)), "TURN": int(np.argmax(np.abs(dpsi))),
        "CRUISE": int(cru[np.argmin(np.abs(v0s[cru] - np.median(v0s)))])}
res["pictures"] = {nm: {"window": int(sel[j]), "episode": str(eid[sel[j]]), "t": int(tw[sel[j]]), "dv_12": float(dv[j]), "dpsi_12": float(dpsi[j])}
                   for nm, j in pick.items()}
json.dump(res, open("RESULT_LATSEM.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)

need = {}                                                                         # episode -> set of native frame indices


def want(e, g):
    need.setdefault(e, set()).add(2 * int(g))


strips = {}
for nm, j in pick.items():
    i = sel[j]; cells = {"now": (eid[i], G[i, 0])}
    for l, ks in W.KS.items():
        for k in ks:
            f = k * STRIDE[l]; nb = NNSAVE[(tag, l, k)]["shrunk" if l == 1 else "raw"][j]
            cells[(l, k, "true")] = (eid[i], G[i, f]); cells[(l, k, "imag")] = (beid[nb], bg[nb])
    if nm in ("LAUNCH", "TURN"):
        for an in ACTS:
            for l in (1, 2, 3):
                Zc = CFSAVE[(tag, an, l)][j:j + 1]
                if l == 1:
                    c = SHR[f"{tag}/L1/k{KMAX[1]}"]["c_train"]
                    z0 = BZ[tag][1][key2b[(eid[i], int(G[i, 0]))]][None]; Zc = z0 + c * (Zc - z0)
                b = nn_loo(Zc, BZ[tag][l], eid[i:i + 1])[0]
                cells[("cf", an, l)] = (beid[b], bg[b])
    strips[nm] = cells
    for e, g in cells.values():
        want(e, g)
IMG = {}
for e, frs in need.items():
    d = torch.load(f"{FR}/{e}.v2ep.pt", map_location="cpu", weights_only=False)
    buf, ln = d["jpeg_buf"].numpy(), d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)[:-1]])
    for r in frs:
        im = Image.open(io.BytesIO(buf[off[r]:off[r] + ln[r]].tobytes())).convert("RGB").resize((384, 156))
        IMG[(e, r)] = np.asarray(im)
    del d


def show(ax, e, g, title):
    ax.imshow(IMG[(e, 2 * int(g))]); ax.set_xticks([]); ax.set_yticks([])
    v = TR[e][2 * int(g), 3]; ax.set_title(f"{title}\n{v * 3.6:.0f} km/h", fontsize=8)


for nm, cells in strips.items():
    fig, axs = plt.subplots(3, 7, figsize=(21, 7.2))
    for r, l in enumerate((1, 2, 3)):
        for c in range(7):
            axs[r, c].axis("off")
        axs[r, 0].axis("on"); show(axs[r, 0], *cells["now"], f"L{l}  now")
        for c, k in enumerate(W.KS[l]):
            sec = k * STRIDE[l] * 0.2
            for dc, kind in enumerate(("true", "imag")):
                ax = axs[r, 1 + 2 * c + dc]; ax.axis("on")
                show(ax, *cells[(l, k, kind)], f"L{l} +{sec:.1f}s {'true' if kind == 'true' else 'imagined (nearest real frame)'}")
    fig.suptitle(f"{nm}: what each level imagines (M2b s0, eval window {res['pictures'][nm]['window']}; imagined = nearest frame from ANOTHER episode)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"fig/strip_{nm}.png", dpi=70); plt.close(fig)
    if nm in ("LAUNCH", "TURN"):
        fig, axs = plt.subplots(3, 6, figsize=(18, 7.2))
        for r, l in enumerate((1, 2, 3)):
            show(axs[r, 0], *cells["now"], f"L{l}  now")
            for c, an in enumerate(ACTS):
                show(axs[r, 1 + c], *cells[("cf", an, l)], f"L{l} +{HSEC[l]:.1f}s if '{an}'")
        fig.suptitle(f"{nm}: imagined futures under five constant actions (nearest real frame from another episode, M2b s0)", fontsize=11)
        fig.tight_layout(); fig.savefig(f"fig/imagine_{nm}.png", dpi=70); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for c, l in enumerate((1, 2, 3)):
    X = BZ[tag][l]; Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P2 = Xc @ Vt[:2].T
    ev = (S[:2] ** 2).sum() / (S ** 2).sum()
    for r, (q, cm, lab_) in enumerate((("v", "viridis", "speed (m/s)"), ("w", "coolwarm", "yaw rate (rad/s)"))):
        y = Y[q]; lim = np.nanpercentile(np.abs(y), 98) if q == "w" else None
        sc = axs[r, c].scatter(P2[:, 0], P2[:, 1], c=y, s=2, cmap=cm, vmin=-lim if lim else None, vmax=lim)
        axs[r, c].set_title(f"level {l} ({0.2 * STRIDE[l]:.1f} s step): PCA 1-2 ({100 * ev:.0f} % var)", fontsize=9)
        plt.colorbar(sc, ax=axs[r, c], label=lab_)
fig.suptitle(f"Per-level latent maps of {len(beid)} eval frames (M2b s0)"); fig.tight_layout(); fig.savefig("fig/pca_levels.png", dpi=80); plt.close(fig)
print("done", round(time.time() - t0, 1), "s")
