import sys, numpy as np
sys.path.insert(0, r"C:/Users/Admin/refcv8_pf/refcv9_l1/code/probe_repro")
import along_track_localise as AL
RT = AL.RT
import compliant_reread_r8 as CR
tools = RT.tools_load()
PR, ra = tools["PR"], tools["ra"]
ctx = CR.prepare_seed("step30000ema_unc", 0, tools, tools["rc"])
print("panels", list(ctx["panels"].keys()))
S = RT.surface_inputs(tools, ctx, "S2", {}, "step30000ema_unc")
O, G = S["X"]["O"], S["X"]["G"]
print("O", O.shape, "dt", S["dt"], "hz", S["hz"], "mask", int(np.sum(S["mask"])), "clip", S["clip"][:3], "ws", S["ws"][:3])
c = PR.components(ra, O, G, S["dt"])
print(sorted(c.keys()))
print([m[1] for m in PR.METRICS])
print("dec keys", list(S["dec"].keys())[:20])
S6 = RT.surface_inputs(tools, ctx, "S6", {}, "step30000ema_unc")
print("S6 O", S6["X"]["O"].shape, "mask", int(np.sum(S6["mask"])), S6["hz"])
