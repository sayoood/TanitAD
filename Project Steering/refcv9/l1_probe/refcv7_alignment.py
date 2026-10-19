"""refcv7_alignment.py -- is refcv8's along-track mis-alignment inherited from refcv7?  Same statistics as along_track_localise.py on
the refcv7-50,400 plans banked inside refcv8's eval139 panels (arm b_refcv7_50400_s{seed}), same compliant S2 windows, same prior.
CPU only.  Paired with refcv8 EMA-30k os (same windows, same episodes, one resample matrix)."""
import json, os, sys, time, hashlib
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import along_track_localise as AL
RT = AL.RT

def main():
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    out = {"tool": "refcv7_alignment.py", "generated": time.strftime("%FT%T"), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "tree": "step30000ema_unc (refcv8) panels; refcv7 arm = b_refcv7_50400_s{seed}", "seeds": {}}
    first = {}
    for seed in AL.SEEDS:
        ctx = CR.prepare_seed("step30000ema_unc", seed, tools, tools["rc"])
        S = RT.surface_inputs(tools, ctx, "S2", first, "step30000ema_unc")
        P = ctx["panels"]["S2"]
        _m, files, _e = tools["NR"].panel_layout(P["dir"])
        r7, osp = [], []
        for f in files:
            with np.load(f) as z:
                r7.append(z[f"b_refcv7_50400_s{seed}"]); osp.append(z["os"])
        r7, osp = np.concatenate(r7), np.concatenate(osp)
        assert np.array_equal(osp, S["X"]["O"]), "row order"
        sel = np.asarray(S["mask"], bool)
        eid = np.asarray([str(e) for e in S["eid"]])[sel]
        sp, sh = AL.arc(S["X"]["HP"][sel]), AL.arc(S["X"]["G"][sel])
        h = sh - sp
        per = {}
        for a, X in (("refcv8_os", S["X"]["O"][sel]), ("refcv7_os", r7[sel])):
            u, dh, dd, hh = AL.ep_sums(AL.arc(X) - sp, h, eid)
            per[a] = (dh, dd, hh)
        c = AL.cell(per, AL.counts(len(u)))
        c["n_windows"] = int(sel.sum())
        out["seeds"][str(seed)] = c
        del ctx
    (HERE / "REFCV7_ALIGNMENT.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ZZR7ALIGNDONEZZ")

if __name__ == "__main__":
    main()
