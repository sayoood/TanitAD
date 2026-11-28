p = "blocks/top.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)

rep('''    cxy, sctx = cur["cxy"], cur["sctx"]
    hxy, htg, hm, hrow = (x.to(env.dev) for x in tail)
    B_, K_ = int(cxy.shape[0]), int(cxy.shape[1])
''', '''    cxy, sctx = cur["cxy"], cur["sctx"]
    cxy32, sctx32 = cxy.float(), sctx.float()          # the direct probes below run in fp32 with NO autocast (the cached context is bf16; their identities are exact or 1e-5 in fp32); the run's own term runs under the trainer's autocast
    hxy, htg, hm, hrow = (x.to(env.dev) for x in tail)
    B_, K_ = int(cxy.shape[0]), int(cxy.shape[1])
''')
rep("clean = HN.hn_term(armB, cxy, hxy, htg, hm, sctx).loss", "clean = HN.hn_term(armB, cxy32, hxy, htg, hm, sctx32).loss")
rep("junk = HN.hn_term(armB, cxy, junk_xy, junk_tg, hm, sctx).loss", "junk = HN.hn_term(armB, cxy32, junk_xy, junk_tg, hm, sctx32).loss")
rep('''        # the fan rows of the masked call vs the one-call fan logits (the term never touches the fan call)
        both = torch.cat([cxy, torch.where(hm.bool()[..., None, None], hxy, torch.zeros_like(hxy))], 1)
        blocked = HN.hn_attn_mask(K_, int(hxy.shape[1]), device=cxy.device)
        two = HN.score_set(armB, both, sctx, blocked)[:, :K_]
        one = armB(cxy, sctx)
        g["fan_call_max_abs_diff"] = float((two.float() - one.float()).abs().max())
''', '''        # (1) B's fan logits AS THE LOSS USES THEM (fwd's sxB) vs the one-call path, same shape and same autocast: bit-identical by construction (the term never touches the fan call)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            one_ac = armB(cxy, sctx)
        g["fan_call_max_abs_diff"] = float((sB.float() - one_ac.float()).abs().max())
        # (2) the masked call's fan rows must not see any hn slot: the SAME 72-row call with the hn slots zeroed gives bit-identical fan rows (same shapes, same kernels, so the identity is exact)
        both = torch.cat([cxy32, torch.where(hm.bool()[..., None, None], hxy, torch.zeros_like(hxy))], 1)
        both0 = torch.cat([cxy32, torch.zeros_like(hxy)], 1)
        blocked = HN.hn_attn_mask(K_, int(hxy.shape[1]), device=cxy.device)
        two = HN.score_set(armB, both, sctx32, blocked)[:, :K_]
        two0 = HN.score_set(armB, both0, sctx32, blocked)[:, :K_]
        g["fan_rows_hn_isolation_max_abs"] = float((two.float() - two0.float()).abs().max())
        # (3) INFORMATIONAL (not gated): the masked call's fan rows vs the 64-row one-call -- different GEMM shapes may differ in the last bits, so this is a scale, not an identity
        one32 = armB(cxy32, sctx32)
        g["fan_rows_vs_one_call_max_abs"] = float((two.float() - one32.float()).abs().max())
        g["fan_rows_scale"] = float(one32.float().abs().max())
''')
rep("full = HN.hn_scores(armB, cxy, hxy, hm.bool(), sctx)", "full = HN.hn_scores(armB, cxy32, hxy, hm.bool(), sctx32)")
rep("alone = HN.hn_scores(armB, cxy, hxy, one_m, sctx)", "alone = HN.hn_scores(armB, cxy32, hxy, one_m, sctx32)")
rep("dxy = torch.zeros(B_, 8, cxy.shape[2], cxy.shape[3], device=env.dev)", "dxy = torch.zeros(B_, 8, cxy32.shape[2], cxy32.shape[3], device=env.dev)")
rep("dxy[:, 0], dtg[:, 0], dm[:, 0] = cxy[:, j], tg[:, j, :6].float(), True", "dxy[:, 0], dtg[:, 0], dm[:, 0] = cxy32[:, j], tg[:, j, :6].float(), True")
rep("lhn = HN.hn_term(Bn, cxy, dxy, dtg, dm, sctx).loss.float()", "lhn = HN.hn_term(Bn, cxy32, dxy, dtg, dm, sctx32).loss.float()")
rep("fan_logits = Bn(cxy, sctx)[:, j, :6].float()", "fan_logits = Bn(cxy32, sctx32)[:, j, :6].float()")
rep('and g["fan_call_max_abs_diff"] == 0.0\n', 'and g["fan_call_max_abs_diff"] == 0.0 and g["fan_rows_hn_isolation_max_abs"] == 0.0\n')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
