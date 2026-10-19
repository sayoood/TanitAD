"""r8_wrapper_diag2.py -- the second discriminator, with LIVE (non-zero) noise held PER-ROW FIXED.

r8_wrapper_diag.py zeroed every eval draw. This arm keeps the noise live but makes it a property of the ROW, not of the
row's position in the call: `rec_plain` (4 rows, seed 0) RECORDS every eval draw -- `torch.randn` without a generator
(refc.py:3036, the base DDIM eps), `R8Generator.randn_eval` (refc.py:3042, extras eps) and `R8Generator.rand_eval`
(refcv8_conditioning.py:446, source-anchor choice) -- and every other arm REPLAYS each draw row by row (a micro-part
covering rows [o, o+m) gets rows [o, o+m) of the recorded tensor; the permuted arm gets them permuted; a row alone gets
its own row). Prediction if the wrapper is exact and the model has no batch coupling: every replayed arm reproduces
rec_plain to float tolerance on every per-row output and every loss term.

Plus the noise-redraw distribution: plain at inference seeds 1..7 (no wrapper), to size the change a mere redraw makes
on THESE 4 rows (the G0 probe's `wrapper` and `floor` are both redraw deviations on CUDA).

Banks raw/diag2_arms.json after every arm; assert on it, not on the exit code.
"""
from __future__ import annotations

import json
import time
import traceback

import r8_wrapper_diag as D  # noqa: E402  (module-level: env, repoint, imports; main() not run)

torch, G, W, L, r8c = D.torch, D.G, D.W, D.L, D.r8c
from microbatch import MicroBatchForward  # noqa: E402

OUT = D.RAW / "diag2_arms.json"
REC: dict = {"tool": "r8_wrapper_diag2.py", "started": time.strftime("%FT%T"), "arms": {}, "errors": {}}


def bank():
    tmp = OUT.with_suffix(".part")
    json.dump(REC, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    tmp.replace(OUT)


class Noise:
    """mode 'record': draw normally and keep each draw; mode 'replay': serve the recorded draw for the requested rows."""

    def __init__(self):
        self.rec: dict = {}
        self.mode = None
        self.row_ids = None
        self.offset = 0
        self.counters: dict = {}
        self.served: dict = {}

    def _take(self, site, shape, draw):
        if self.mode == "record":
            t = draw()
            self.rec.setdefault(site, []).append(t.clone())
            return t
        k = self.counters.get(site, 0)
        self.counters[site] = k + 1
        full = self.rec[site][k]
        if full.shape[0] != len(self.row_ids_full):
            raise RuntimeError(f"replay: recorded {site}#{k} {tuple(full.shape)} is not batch-first")
        ids = self.row_ids[self.offset:self.offset + int(shape[0])]
        t = full[ids]
        if tuple(t.shape) != tuple(shape):
            raise RuntimeError(f"replay: {site}#{k} wants {tuple(shape)}, serving {tuple(t.shape)}")
        self.served[site] = self.served.get(site, 0) + 1
        return t.clone()

    def install(self):
        self.o = (torch.randn_like, torch.randn, r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval)
        o_rn, o_re, o_ue = self.o[1], self.o[2], self.o[3]
        me = self
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)        # the probe's own patch, unchanged

        def randn(*size, **kk):
            if kk.get("generator") is not None:
                return o_rn(*size, **kk)
            shp = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
            return me._take("randn", tuple(shp), lambda: o_rn(*size, **kk))
        torch.randn = randn
        r8c.R8Generator.randn_eval = lambda gen, shape, device, site, dtype=torch.float32: me._take(
            f"randn_eval:{site}", tuple(shape), lambda: o_re(gen, shape, device, site, dtype))
        r8c.R8Generator.rand_eval = lambda gen, shape, device, site: me._take(
            f"rand_eval:{site}", tuple(shape), lambda: o_ue(gen, shape, device, site))
        return self

    def remove(self):
        torch.randn_like, torch.randn, r8c.R8Generator.randn_eval, r8c.R8Generator.rand_eval = self.o


def main():
    t_all = time.time()
    tr = L.trainer()
    config = L.load_config(D.CONFIG)
    model, cfg, args, mrec = L.build_model(config, D.CKPT, "cpu")
    REC["model"] = {"step": mrec["state_dict"].get("step"), "training": bool(model.training)}
    tf = G.training_flags(model)
    G.set_flags(model, tf["flags"])
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)
    if not D.CACHE.exists():
        raise SystemExit("run r8_wrapper_diag.py first (it builds cache/batch4.pt)")
    eb4 = D.load_or_build_batch(tr, model, cfg, args, config)
    REC["batch_meta"] = D.REC.get("batch_meta")
    G.patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    N = D.N
    nz = Noise()
    nz.row_ids_full = list(range(N))

    def run(name, nmode, seed, rows, sizes=None):
        t0 = time.time()
        sb = G.sub_batch(eb4, len(rows), rows=rows)
        w = cap = None
        nz.mode, nz.row_ids, nz.offset, nz.counters, nz.served = nmode, rows, 0, {}, {}
        if nmode == "record":
            nz.rec = {}
        try:
            nz.install()
            torch.manual_seed(seed)
            if sizes:
                w = MicroBatchForward(model, sizes).install()
                inner = w._orig
                st = {"o": 0}

                def part(frames, **kw):
                    nz.offset, nz.counters = st["o"], {}
                    r = inner(frames, **kw)
                    st["o"] += int(frames.shape[0])
                    return r
                w._orig = part
            cap = D.Capture(model).install()
            with torch.no_grad():
                d = tr.compute_losses_v3(model, sb, "cpu", mode=mode, ablate_frames=abl)
        except Exception as exc:                                     # noqa: BLE001
            REC["errors"][name] = f"{type(exc).__name__}: {str(exc)[:400]}\n{traceback.format_exc()[-1500:]}"
            print(f"[diag2] {name} RAISED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            bank()
            return None
        finally:
            if cap is not None:
                cap.remove()
            if w is not None:
                w.remove()
            nz.remove()
        scal = W._scalars(d)
        del d
        out = cap.outs[-1]
        per_row = {p: t.detach().float().cpu().clone() for p, t in D.flatten(out)
                   if t.dim() >= 1 and t.shape[0] == len(rows)}
        REC["arms"][name] = {"noise": nmode, "seed": seed, "rows": rows, "sizes": sizes, "wall_s": round(time.time() - t0, 1),
                             "scalars": scal, "n_forward_calls": len(cap.outs),
                             "recorded_sites": ({k: [list(t.shape) for t in v] for k, v in nz.rec.items()}
                                                if nmode == "record" else None),
                             "served": dict(nz.served) if nmode == "replay" else None}
        cap.outs.clear()
        print(f"[diag2] {name}: {REC['arms'][name]['wall_s']}s cascade={scal.get('cascade')} "
              f"r8_subscore={scal.get('r8_subscore')} traj={scal.get('traj')} served={nz.served}", flush=True)
        return per_row

    def compare(name, cur, ref, row_map):
        res = {}
        for k, t in cur.items():
            if k not in ref:
                continue
            r = ref[k][row_map]
            if r.shape != t.shape:
                continue
            fin = torch.isfinite(r) & torch.isfinite(t)
            dd = (t - r).abs()[fin]
            mx = float(dd.max()) if dd.numel() else 0.0
            sc = float(r.abs()[fin].max()) if fin.any() else 0.0
            res[k] = {"max_abs": mx, "rel": mx / max(sc, 1e-12)}
        top = sorted(((v["rel"], k) for k, v in res.items()), reverse=True)[:8]
        REC["arms"][name]["vs_ref_top"] = [[k, r] for r, k in top]
        REC["arms"][name]["vs_ref_keys"] = {k: res[k] for k in D.SAVE_KEYS + ("out.layer_u0_hat[0]", "out.u0_hat",
                                                                               "out.offset") if k in res}
        print(f"[diag2]   {name} top rel: {[(k, f'{r:.2e}') for r, k in top[:4]]}", flush=True)

    ident, rev = list(range(N)), list(range(N))[::-1]
    ref = run("rec_plain", "record", 0, ident)
    bank()
    for nm, rows, sizes, rmap in (("rp_micro13", ident, [1, 3], ident), ("rp_micro31", ident, [3, 1], ident),
                                  ("rp_perm", rev, None, rev), ("rp_row0", [0], None, [0]), ("rp_row1", [1], None, [1]),
                                  ("rp_row2", [2], None, [2]), ("rp_row3", [3], None, [3])):
        cur = run(nm, "replay", 0, rows, sizes)
        if cur is not None and ref is not None:
            compare(nm, cur, ref, rmap)
        bank()
    # the redraw distribution on these 4 rows (plain, no wrapper, live noise via the ordinary RNG)
    for s in range(1, 8):
        nz_mode_live = "record"                       # record = draw normally; the recording is discarded
        run(f"live_plain_s{s}", nz_mode_live, s, ident)
        bank()
    REC["finished"] = time.strftime("%FT%T")
    REC["wall_s"] = round(time.time() - t_all, 1)
    bank()
    print(f"[diag2] DONE {REC['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
