#!/usr/bin/env python3
"""SCORER1 deliberate-regression arms THROUGH THE GATE (CPU, no data).  Prints ZZTEST_PASS.

The two arms of PREREG_SCORER1 'Launch gate' are run through the gate's own consumer (`launch_gate.consume_scorer_preflight`, which re-judges G16 from its RAW fields):
  NOBALANCE  REFE_PAIR_MUTATE_NOBALANCE=1 -> the REAL PairLoss keeps every candidate -> kept F != kept S on a real-shaped set -> the F/S ratio check goes RED.
  DROP_LOSS  REFE_PAIR_MUTATE_DROP_LOSS=1 -> L_pair is computed and dropped from `loss` -> grad(w=1) - grad(w=0) is EXACTLY 0 on every scorer module -> 'not live IN the loss' RED.
The control (no mutation) must come back PASS, so a RED is the mutation's and not the harness's.
SCOPE (stated, not hidden): the raw G16 fields here are built by the REAL PairLoss.select / chord_speed and, for DROP_LOSS, by the physics of the mutation (a term dropped from `loss` adds nothing to the
gradient); they are NOT written by scorer_finetune.py's own --preflight, which needs the pod / dev-box bank.  The trainer-side end to end run is HANDOFF step 'E2E mutation'.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402
import scorer_finetune as S  # noqa: E402

PROF = L.PROFILES["refe_scorer1"]
ARGV = L.scorer1_argv("S1", 0.37)
ns, err = C.parse_scorer_argv(S.build_parser(), ARGV)
assert not err, err
HEADS = ["NC", "DAC", "TTC"]


def counts(mutate):
    for e in S.PAIR_MUTATE_ENVS:
        os.environ.pop(e, None)
    if mutate:
        os.environ[mutate] = "1"
    try:
        pl = S.PairLoss(HEADS, 0.15)
        g = torch.Generator().manual_seed(5)
        mb, bank = {}, {}
        for h in HEADS:
            col = S.PAIR_HEADS[h]
            kf = ks = sets = withp = 0
            for n in range(40):
                speed = torch.rand(64, generator=g) * 14.0
                cxy = torch.zeros(1, 64, 20, 3)
                cxy[0, :, :, 0] = (speed[:, None] * 0.2) * torch.arange(1, 21)[None, :]
                y = (torch.rand(64, generator=g) > 0.5).float()
                ii, jj, cF, cS, kF, kS = pl.select(y, S.chord_speed(cxy)[0], 10.0)
                sets += 1
                withp += int(kF + kS > 0)
                kf, ks = kf + kF, ks + kS
            mb[h] = {"kept_F": kf, "kept_S": ks}
            bank[h] = {"kept_F": kf, "kept_S": ks, "share_with_pair": float(withp / sets)}
        return mb, bank, bool(pl.balance)
    finally:
        for e in S.PAIR_MUTATE_ENVS:
            os.environ.pop(e, None)


def g16_for(mutate=None):
    mb, bank, balanced = counts("REFE_PAIR_MUTATE_NOBALANCE" if mutate == "nobalance" else None)
    drop = mutate == "drop_loss"
    delta = {m: (0.0 if drop else 1.5e-3) for m in C.SCORER_GROUPS}
    return {"loss_grad_delta_w1_minus_w0": delta, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS}, "controls_exactly_zero": True,
            "micro_batch_pairs": mb, "bank_share": bank, "balanced_built": balanced, "mutate_nobalance": mutate == "nobalance", "mutate_drop_loss": drop,
            "ok": not mutate}


def consume(g16):
    req = C.scorer_required_gates(ns)
    assert "G16" in req
    gates = {}
    for g in req:
        gates[g + "_stub"] = {"ok": True}
    gates["G16_pair_live_balanced"] = g16
    gates.pop("G16_stub")
    for g, v in list(gates.items()):
        if g.startswith("G11") or g.startswith("G12") or g.startswith("G13"):
            gates.pop(g)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
        p = os.path.join(td, "preflight.jsonl")
        with open(p, "w") as fh:
            fh.write(json.dumps({"event": "data", "train_sets": 1}) + "\n")
            fh.write(json.dumps({"event": "preflight", "verdict": "PASS", "gates": {k: (dict(v, ok=True) if k.startswith("G16") else v) for k, v in gates.items()}}) + "\n")
        return L.consume_scorer_preflight(p, ARGV, PROF)


ctl = consume(g16_for(None))
assert ctl["status"] == "PASS", ctl["reasons"]
print("control PASS ok")

nb = consume(g16_for("nobalance"))
assert nb["status"] == "FAIL" and any("F/S ratio != 1" in r for r in nb["reasons"]) and any("regression arm" in r for r in nb["reasons"]), nb["reasons"]
mb, _, bal = counts("REFE_PAIR_MUTATE_NOBALANCE")
assert bal is False and any(v["kept_F"] != v["kept_S"] for v in mb.values()), "the REAL PairLoss under NOBALANCE must unbalance F/S"
print("NOBALANCE RED ok:", [r[:70] for r in nb["reasons"]])

dl = consume(g16_for("drop_loss"))
assert dl["status"] == "FAIL" and any("not live IN the loss" in r for r in dl["reasons"]), dl["reasons"]
print("DROP_LOSS RED ok:", [r[:70] for r in dl["reasons"]])

# the mutation branches really exist in the trainer (a grep of the guard only in the test file would mean the guard does not exist)
src = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf8").read()
assert src.count('os.environ.get("REFE_PAIR_MUTATE_DROP_LOSS") == "1"') >= 1 and src.count('os.environ.get("REFE_PAIR_MUTATE_NOBALANCE") != "1"') == 1
print("ZZTEST_PASS")
