"""PREREG_V8F_LATSEM2 pictures (decide nothing; seed M2b_s0; windows chosen by labels only; IMPL_CHOICES_LATSEM2.md section 10).
1. strip_CLOSE.png / strip_FAR.png : per level, now / true future / nearest real frame (from ANOTHER episode) of the imagined state under accelerate / under brake.
2. pca_levels_scene.png            : per-level PCA maps of the bank frames, coloured by lead gap and by agents within 30 m (no camera frame inside).
The strips embed downscaled camera frames: local only, never for the public repo. Run from theory/latsem2 with WMACT_CODE_ROOT set."""
import hashlib, io, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402
import labels_latsem2 as LB  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

IMPL, IMPL_SHA = "IMPL_CHOICES_LATSEM2.md", "b17688dc0fea5c639ab2d44c7088f88cded05ab99a1b75f89272cbb78492e71d"
assert hashlib.sha256(open(IMPL, "rb").read()).hexdigest() == IMPL_SHA
STRIDE, KMAX = {1: 1, 2: 2, 3: 4}, {1: 10, 2: 3, 3: 3}
HSEC = {1: 2.0, 2: 1.2, 3: 2.4}
F, tag = 16, "M2b_s0"
B = LB.build_bank()
L, eid, tw, ok, A_raw, wkeep, G, key2b = B["L"], B["eid"], B["tw"], B["ok"], B["A_raw"], B["wkeep"], B["G"], B["key2b"]
bw, bf, beid, bg = B["bw"], B["bf"], B["beid"], B["bg"]
Y, _ = LB.build_labels(beid, bg)
m, _, _ = W.load_model(W.CKPT[tag]); wm = m.wm
with torch.no_grad():
    zs = wm.stack.initial_latents(L["Z1"][tag][bw, bf])
BZ = {l + 1: zs[l].double().numpy() for l in range(3)}
SHR = json.load(open("../wmact/RESULT_WMACT_SHRINK.json", encoding="utf-8"))["rows"]
sel = wkeep[ok[wkeep].all(1)]
b0 = np.array([key2b[(eid[i], int(G[i, 0]))] for i in sel])
m3 = np.flatnonzero(Y["lead_present"][b0] == 1)
gap0 = Y["lead_gap"][b0[m3]]
pick = {"CLOSE": int(m3[np.argmin(gap0)]), "FAR": int(m3[np.argmin(np.abs(gap0 - np.median(gap0)))])}


def nn_loo(Q, Bk, qeid):
    out = np.empty(len(Q), np.int64); b2 = (Bk * Bk).sum(1)
    d = (Q * Q).sum(1)[:, None] - 2 * Q @ Bk.T + b2[None]
    d[beid[None, :] == qeid[:, None]] = np.inf
    return d.argmin(1)


def label_txt(b):
    lp, lg, ag = Y["lead_present"][b], Y["lead_gap"][b], Y["agents30"][b]
    a = "lead ?" if not np.isfinite(lp) else (f"lead {lg:.0f} m" if lp == 1 else "no lead <40 m")
    return a + (f", {int(ag)} ag<=30 m" if np.isfinite(ag) else "")


need, cells_all, info = {}, {}, {}
for nm, j in pick.items():
    i = sel[j]
    Z1 = L["Z1"][tag][[i]]
    cells = {}
    cells[("now",)] = key2b[(eid[i], int(G[i, 0]))]
    z0s = {}
    for an, a in (("accel", 1.5), ("brake", -3.0)):
        Araw = torch.tensor([a, 0.0], dtype=torch.float32).expand(1, F, 2).contiguous()
        with torch.no_grad():
            pk = W.level_pack(wm, Z1, wm.scale_actions(Araw)[:, :-1])
        for l in (1, 2, 3):
            z0 = pk["true"][l][:, 0].double().numpy(); zh = pk["pred"][(l, KMAX[l])].double().numpy()
            if l == 1:
                zh = z0 + SHR[f"{tag}/L1/k{KMAX[1]}"]["c_train"] * (zh - z0)
            cells[(l, an)] = int(nn_loo(zh, BZ[l], eid[i:i + 1])[0])
    for l in (1, 2, 3):
        cells[(l, "true")] = key2b[(eid[i], int(G[i, KMAX[l] * STRIDE[l]]))]
    cells_all[nm] = cells
    info[nm] = {"window": int(i), "episode": str(eid[i]), "t": int(tw[i]), "lead_gap_frame0_m": float(Y["lead_gap"][cells[("now",)]])}
    for b in cells.values():
        need.setdefault(str(beid[b]), set()).add(2 * int(bg[b]))
IMG = {}
for e, frs in need.items():
    d = torch.load(f"{LB.FR}/{e}.v2ep.pt", map_location="cpu", weights_only=False)
    buf, ln = d["jpeg_buf"].numpy(), d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)[:-1]])
    for r in frs:
        IMG[(e, r)] = np.asarray(Image.open(io.BytesIO(buf[off[r]:off[r] + ln[r]].tobytes())).convert("RGB").resize((384, 156)))
    del d


def show(ax, b, title):
    ax.imshow(IMG[(str(beid[b]), 2 * int(bg[b]))]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\n{label_txt(b)}", fontsize=8)


os.makedirs("fig", exist_ok=True)
for nm, cells in cells_all.items():
    fig, axs = plt.subplots(3, 4, figsize=(16, 7.4))
    for r, l in enumerate((1, 2, 3)):
        show(axs[r, 0], cells[("now",)], f"L{l} now")
        show(axs[r, 1], cells[(l, "true")], f"L{l} true +{HSEC[l]:.1f} s (real future frame)")
        show(axs[r, 2], cells[(l, "accel")], f"L{l} +{HSEC[l]:.1f} s if accelerate +1.5 (nearest real frame)")
        show(axs[r, 3], cells[(l, "brake")], f"L{l} +{HSEC[l]:.1f} s if brake -3.0 (nearest real frame)")
    fig.suptitle(f"{nm} (lead gap {info[nm]['lead_gap_frame0_m']:.1f} m at frame 0, window {info[nm]['window']}): imagined futures, nearest frame from ANOTHER episode, M2b s0. Camera frames: local only", fontsize=10)
    fig.tight_layout(); fig.savefig(f"fig/strip_{nm}.png", dpi=70); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
lg, ag = Y["lead_gap"], Y["agents30"]
vmax = float(np.nanpercentile(ag, 98))
for c, l in enumerate((1, 2, 3)):
    X = BZ[l]; Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P2 = Xc @ Vt[:2].T
    ev = (S[:2] ** 2).sum() / (S ** 2).sum()
    has = np.isfinite(lg)
    axs[0, c].scatter(P2[~has, 0], P2[~has, 1], c="0.85", s=2)
    sc = axs[0, c].scatter(P2[has, 0], P2[has, 1], c=lg[has], s=3, cmap="viridis_r", vmin=0, vmax=40)
    axs[0, c].set_title(f"level {l}: PCA 1-2 ({100 * ev:.0f} % var), lead gap (grey: no lead <40 m / unlabelled)", fontsize=8)
    plt.colorbar(sc, ax=axs[0, c], label="lead gap (m)")
    fa = np.isfinite(ag)
    axs[1, c].scatter(P2[~fa, 0], P2[~fa, 1], c="0.85", s=2)
    sc = axs[1, c].scatter(P2[fa, 0], P2[fa, 1], c=np.clip(ag[fa], 0, vmax), s=3, cmap="magma_r", vmin=0, vmax=vmax)
    axs[1, c].set_title(f"level {l}: agents within 30 m (clipped at p98 = {vmax:.0f})", fontsize=9)
    plt.colorbar(sc, ax=axs[1, c], label="agents <= 30 m")
fig.suptitle(f"Per-level latent maps of {len(beid)} eval frames (M2b s0); no camera frame in this picture"); fig.tight_layout(); fig.savefig("fig/pca_levels_scene.png", dpi=80); plt.close(fig)
out = {"pictures": info, "files": {f: hashlib.sha256(open(f"fig/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir("fig")) if f.endswith(".png")}}
json.dump(out, open("PICTURES_LATSEM2.json", "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
