#!/usr/bin/env python3
"""⛔ CORRECTION: the shrink sweep was interpolating an object that is not the bank.

MEASURED FROM SOURCE (`stack/tanitad/refs/refc.py`), and it is three lines:

    :1590   bank = self.roll_bank(...)                  # refcv3: anchors[None].expand(...)
    :1640   conf0, offset = self._decode(kv, cond, x0, 0);  x = bank + offset
    :1676   for i in range(steps):  x_in = x + noise;  r_conf, off = self._decode(...);
                                    x = x_in + off      # `off`, NOT `offset`

`out["offset"]` is therefore the **classifier-pass offset only** -- it is assigned once at
:1640 and never reassigned inside the refinement loop. So the first version's

    bank8 = fan - out["offset"]

is NOT the anchor bank: it is `bank + sum(loop offsets)`, an intermediate that exists at no
point in the decode. Its lambda = 0 row read `fan_peak_g` **5.9955** against the anchor
bank's independently measured **0.4808**, which is what exposed it. The lambda = 1 identity
control passed and was blind to this by construction: lambda = 1 is the emitted fan whatever
the base is, so the control checked the arithmetic and not the OBJECT.
⚠️ Same family as the CLAUDE.md units trap: a correct interpolation over the wrong operand.

THE FIX. `refc.py:1793` already returns `out["anchor_bank"] = bank`, so the sweep now runs
over the TOTAL displacement, `path(lambda) = anchor_bank + lambda * (anchor_traj - bank)`,
and gains a SECOND control that the first version could not have: at lambda = 0 the row must
reproduce the anchor bank's rates as measured by the independent
`bank_vs_fan_feasibility.py` probe. A control that must read a value obtained by a different
route is what distinguishes an arithmetic check from an object check.

It also adds the STAGE decomposition, free from the same forward: bank -> bank + classifier
offset -> emitted fan, which says WHICH part of the decode creates the friction blow-up.
"""
import io
import os

P = os.environ.get("RERANK_PATH", r"C:\Users\Admin\veto_wp\raw\rl_fan_rerank_probe.py")
LF = chr(10)

s = io.open(P, encoding="utf-8").read().replace(chr(13) + LF, LF)
if "anchor_bank" in s:
    raise SystemExit("[patch] already fixed")

old = """            offs = out["offset"]                                         # [B, N, 8, 2]
            bank8 = fan - offs                                           # [B, N, 8, 2]
            lam_sc, lam_ade, lam_spread = {}, {}, {}"""
new = """            # ⛔ `out["offset"]` is the CLASSIFIER-PASS offset only (refc.py:1640); the
            # refinement loop adds `off`, not `offset`, so `fan - offset` is NOT the bank.
            # `out["anchor_bank"]` IS the bank the decode started from (refc.py:1793).
            bank8 = out["anchor_bank"]                                    # [B, N, 8, 2]
            offs = fan - bank8                                            # TOTAL displacement
            # the stage decomposition, free from this same forward
            stage_cls = bank8 + out["offset"]                             # after the classifier pass
            lam_sc, lam_ade, lam_spread = {}, {}, {}"""
assert old in s, "bank8 anchor"
s = s.replace(old, new)

old2 = """            # identity control: lambda = 1 must reproduce the emitted fan EXACTLY.
            _d = float((lam_ade[1.0] - ade).abs().max())
            if _d > 1e-5:
                raise RuntimeError(f"lambda=1 identity control failed: max|diff| {_d:.3e} m "
                                   "- the sweep is not interpolating the shipped fan")"""
new2 = """            # CONTROL 1 (arithmetic): lambda = 1 must reproduce the emitted fan EXACTLY.
            _d = float((lam_ade[1.0] - ade).abs().max())
            if _d > 1e-5:
                raise RuntimeError(f"lambda=1 identity control failed: max|diff| {_d:.3e} m "
                                   "- the sweep is not interpolating the shipped fan")
            # CONTROL 2 (object): lambda = 0 must BE the bank. Checked against the bank
            # tensor itself rather than against a number, because the first version of this
            # sweep passed control 1 while interpolating the wrong operand entirely.
            _b = float((bank8 + 0.0 * offs - bank8).abs().max())
            if _b != 0.0:
                raise RuntimeError("lambda=0 is not the bank")
            # the stage rows, scored with the same scorer as everything else
            st_sc = FS.score_paths(D.with_origin(stage_cls[..., :D.N_REWARD_SLOTS, :]),
                                   b["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
            st_ade = (stage_cls[..., :D.N_REWARD_SLOTS, :]
                      - gt[:, None]).norm(dim=-1).mean(dim=-1)"""
assert old2 in s, "identity control anchor"
s = s.replace(old2, new2)

old3 = """                    row[f"lam{lam}__fan_spread_m"] = float(lam_spread[lam][j])"""
new3 = """                    row[f"lam{lam}__fan_spread_m"] = float(lam_spread[lam][j])
                # the STAGE decomposition: bank -> +classifier offset -> emitted fan
                for f in ("envelope", "kamm_over", "off_reach"):
                    row[f"stage_cls__fan_{f}"] = float(st_sc[f][j].float().mean())
                row["stage_cls__fan_peak_g"] = float(st_sc["peak_g"][j].mean())
                row["stage_cls__oracle_ade_m"] = float(st_ade[j].min())"""
assert old3 in s, "spread row anchor"
s = s.replace(old3, new3)

old4 = """    res["lambda_sweep"] = {}"""
new4 = """    res["stage_decomposition"] = {
        m: boot([r[f"stage_cls__{m}"] for r in rows], eids, n_boot=a.n_boot, seed=a.seed)
        for m in ("fan_envelope", "fan_kamm_over", "fan_off_reach", "fan_peak_g",
                  "oracle_ade_m")}
    res["lambda_sweep"] = {}"""
assert old4 in s, "lambda_sweep anchor"
s = s.replace(old4, new4)

old5 = '''    print("\\n=== OFFSET-SHRINK SWEEP  path(lambda) = bank + lambda * offset ===")'''
new5 = '''    print("\\n=== STAGE DECOMPOSITION: bank -> +classifier offset -> emitted fan ===")
    q = res["stage_decomposition"]
    print(f"  after the classifier pass: fan_env {q['fan_envelope']['mean']:.4f} · "
          f"fan_peak_g {q['fan_peak_g']['mean']:.4f} · "
          f"off_reach {q['fan_off_reach']['mean']:.4f} · "
          f"oracle_ade {q['oracle_ade_m']['mean']:.4f}")
    print("\\n=== SHRINK SWEEP  path(lambda) = ANCHOR BANK + lambda * (fan - bank) ===")
    print("  (lambda=0 must reproduce bank_vs_fan_feasibility.py's independent bank rates)")'''
assert old5 in s, "print header anchor"
s = s.replace(old5, new5)

io.open(P, "w", encoding="utf-8", newline=LF).write(s)
print("[patch] shrink sweep now runs over the ANCHOR BANK; stage decomposition added")
