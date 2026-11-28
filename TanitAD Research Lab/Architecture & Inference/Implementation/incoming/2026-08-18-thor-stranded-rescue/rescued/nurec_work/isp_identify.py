"""Stage 2 (0-GPU): identify the 26.04 PPISP CRF layout.

The scene stores crf_params [6,3,7]; the PUBLISHED nv-tlabs/ppisp v1.0.0 stores 4 per
channel (toe, shoulder, gamma, center) with published activations. Rather than GUESS an
ordering over 7 slots, we:

  1. MEASURE the empirical transfer curve T_c(x) = median(ref | render=x) on frame 0,
     using only well-covered pixels (alpha high) so "no gaussian -> black" cannot
     contaminate it. This is non-parametric and uses none of the stored numbers.
  2. FIT the published 4-parameter PPISP CRF form to T_c.
  3. INVERT the published activations on the fit -> raw (toe,shoulder,gamma,center).
  4. CHECK whether those raw values appear among the 7 stored numbers. A match is a
     non-circular identification: 4 numbers fit from pixels landing on 4 numbers read
     from the file with zero fitting.
"""
import numpy as np, msgpack, json, itertools

D = np.load("/home/nvidia/nurec_work/linear_dump.npz")
frames, render, alpha, ref = D["frames"], D["render"], D["alpha"], D["ref"]
print("frames", frames, "render", render.shape, "alpha", alpha.shape)

obj = msgpack.unpackb(open("/home/nvidia/nurec_work/x/volume.msgpack","rb").read(), raw=False, strict_map_key=False)
sd = obj["nre_data"]["state_dict"]
CRF = np.frombuffer(sd[".post_processings.0.ppisp.crf_params"], np.float16).astype(np.float64).reshape(6,3,7)
VIG = np.frombuffer(sd[".post_processings.0.ppisp.vignetting_params"], np.float16).astype(np.float64).reshape(6,3,5)
CAM_IDX = 3   # camera_front_wide_120fov, identified by principal point (958.4,751.2)
print("\nSTORED crf_params[cam=3] (7 per channel):")
for c,n in enumerate("RGB"): print(f"  {n}: {np.round(CRF[3,c],5)}")
print("STORED vignetting_params[cam=3]:")
for c,n in enumerate("RGB"): print(f"  {n}: {np.round(VIG[3,c],5)}")

softplus = lambda z: np.logaddexp(0.0, z)
inv_softplus = lambda v: np.log(np.expm1(np.clip(v, 1e-12, None)))
sigmoid = lambda z: 1.0/(1.0+np.exp(-z))
logit = lambda p: np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))

def crf_forward(x, toe, shoulder, gamma, center):
    """PUBLISHED nv-tlabs/ppisp toe-shoulder CRF (tests/torch_reference.py)."""
    x = np.clip(x, 0.0, 1.0)
    lerp = toe + center*(shoulder - toe)
    a = (shoulder*center)/lerp
    b = 1.0 - a
    eps = 1e-6
    lo = a*np.power(np.clip(x/center, eps, None), toe)
    hi = 1.0 - b*np.power(np.clip((1.0-x)/(1.0-center), eps, None), shoulder)
    y = np.where(x <= center, lo, hi)
    return np.power(np.clip(y, eps, None), gamma)

# ---------------- 1. empirical transfer curve -------------------------------------
NB = 48
EDGES = np.linspace(0.0, 1.0, NB+1)
CTR = 0.5*(EDGES[:-1]+EDGES[1:])

def empirical_curve(fi_idx, ch, amin=0.995):
    x = np.clip(render[fi_idx][...,ch], 0, 1).ravel()
    y = ref[fi_idx][...,ch].ravel()
    m = alpha[fi_idx].ravel() >= amin
    x, y = x[m], y[m]
    idx = np.clip(np.digitize(x, EDGES)-1, 0, NB-1)
    med = np.full(NB, np.nan); cnt = np.zeros(NB, np.int64)
    for b in range(NB):
        s = idx == b
        cnt[b] = s.sum()
        if cnt[b] >= 200: med[b] = np.median(y[s])
    return med, cnt

print(f"\n=== empirical transfer curves (frame 0, alpha>=0.995) ===")
for c,n in enumerate("RGB"):
    med, cnt = empirical_curve(0, c)
    ok = ~np.isnan(med)
    print(f"  {n}: {ok.sum()}/{NB} usable bins, n_px={cnt.sum()}")
    show = [(round(float(CTR[b]),3), round(float(med[b]),4)) for b in range(NB) if ok[b]][::4]
    print(f"     x->y sample: {show}")

# ---------------- 2. fit the published 4-param form -------------------------------
def fit_crf(med, cnt, wmin=200):
    ok = (~np.isnan(med)) & (cnt >= wmin)
    xs, ys, ws = CTR[ok], med[ok], np.sqrt(cnt[ok].astype(float))
    def loss(p):
        toe, shoulder, gamma, center = p
        if not (0.3 < toe < 12 and 0.3 < shoulder < 12 and 0.1 < gamma < 12 and 0.01 < center < 0.999):
            return 1e9
        yh = crf_forward(xs, toe, shoulder, gamma, center)
        return float(np.sum(ws*(yh-ys)**2)/np.sum(ws))
    best, bp = 1e18, None
    for toe in np.linspace(0.32, 4.0, 14):
        for sh in np.linspace(0.32, 4.0, 14):
            for gm in np.geomspace(0.11, 8.0, 16):
                for ce in np.linspace(0.05, 0.98, 14):
                    l = loss((toe,sh,gm,ce))
                    if l < best: best, bp = l, (toe,sh,gm,ce)
    p = np.array(bp, float)                      # Nelder-Mead-free refine
    step = np.array([0.2,0.2,0.4,0.05])
    for _ in range(400):
        improved = False
        for i in range(4):
            for sgn in (+1,-1):
                q = p.copy(); q[i] += sgn*step[i]
                l = loss(q)
                if l < best: best, p, improved = l, q, True
        if not improved: step *= 0.5
        if step.max() < 1e-5: break
    return p, best, ok.sum()

print("\n=== fit of the PUBLISHED 4-param CRF to the empirical curve (frame 0) ===")
fits = {}
for c,n in enumerate("RGB"):
    med, cnt = empirical_curve(0, c)
    p, l, nb = fit_crf(med, cnt)
    fits[c] = p
    toe, sh, gm, ce = p
    raw = np.array([inv_softplus(toe-0.3), inv_softplus(sh-0.3), inv_softplus(gm-0.1), logit(ce)])
    print(f"  {n}: toe={toe:.4f} shoulder={sh:.4f} gamma={gm:.4f} center={ce:.4f}   "
          f"wRMSE={np.sqrt(l):.5f} over {nb} bins")
    print(f"      -> INVERTED RAW (what the file should store): "
          f"toe={raw[0]:+.3f} shoulder={raw[1]:+.3f} gamma={raw[2]:+.3f} center={raw[3]:+.3f}")
    print(f"      -> STORED 7    : {np.round(CRF[3,c],3)}")

# ---------------- 3. score every 4-slot assignment of the stored 7 ----------------
print("\n=== PRE-REGISTERED: score EVERY ordered 4-of-7 assignment, no fitting ===")
print("(toe,shoulder,gamma,center) <- 4 of the 7 stored raws, published activations.")
res = []
for perm in itertools.permutations(range(7), 4):
    tot, per_ch = 0.0, []
    for c in range(3):
        p = CRF[3,c]
        toe = 0.3+softplus(p[perm[0]]); sh = 0.3+softplus(p[perm[1]])
        gm  = 0.1+softplus(p[perm[2]]); ce = sigmoid(p[perm[3]])
        med, cnt = empirical_curve(0, c)
        ok = (~np.isnan(med)) & (cnt >= 200)
        yh = crf_forward(CTR[ok], toe, sh, gm, ce)
        w = np.sqrt(cnt[ok].astype(float))
        e = float(np.sum(w*(yh-med[ok])**2)/np.sum(w))
        per_ch.append(e); tot += e
    res.append((tot/3.0, perm, per_ch))
res.sort()
for rank,(e, perm, pc) in enumerate(res[:8]):
    p = CRF[3,0]
    toe = 0.3+softplus(p[perm[0]]); sh = 0.3+softplus(p[perm[1]])
    gm  = 0.1+softplus(p[perm[2]]); ce = sigmoid(p[perm[3]])
    print(f"  #{rank+1} idx(toe,sh,gamma,center)={perm}  wRMSE={np.sqrt(e):.5f}  "
          f"[R: toe={toe:.3f} sh={sh:.3f} g={gm:.3f} c={ce:.3f}]")
print(f"  ... worst of {len(res)}: wRMSE={np.sqrt(res[-1][0]):.5f}")
np.save("/home/nvidia/nurec_work/crf_fit.npy", np.stack([fits[c] for c in range(3)]))
json.dump({"best_perm": list(res[0][1]), "best_wrmse": float(np.sqrt(res[0][0])),
           "free_fit": {c: list(map(float, fits[c])) for c in range(3)}},
          open("/home/nvidia/nurec_work/crf_ident.json","w"), indent=1)
