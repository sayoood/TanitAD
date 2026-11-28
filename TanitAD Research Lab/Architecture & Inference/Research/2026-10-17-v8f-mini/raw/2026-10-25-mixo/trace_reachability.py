"""PREREG_V8F_MIXO proof 4: reachability of `--code-mix` FROM THE ACTUAL CALLER, hop by hop, with file:line at each hop.

    python trace_reachability.py <repo root>          (CPU only; CUDA_VISIBLE_DEVICES=-1; writes into a fresh temp dir)

PART 1 (static): the line of every hop, found by its exact text (an anchor that is not found exactly once aborts).
PART 2 (dynamic): a REAL `TR.main` run on the synthetic world with `--code-mix kind` and wrappers that record, from the live frames, where the mixing line was CALLED from and what `run_o`
received on every training step: the file:line in the output are the runtime call sites, not a grep. The tensor `run_o` received must be where(mask, GT code, T's own code) exactly.
PART 3 (the control): the mixing function replaced by a no-op that returns an all-False mask (the line "deleted"): the gate's own probe turns RED, and so does the run.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(sys.argv[1]).resolve()
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402

torch.set_num_threads(1)
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

MOD = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
src = {MOD: MOD.read_text(encoding="utf-8").split("\n"), TRN: TRN.read_text(encoding="utf-8").split("\n")}


def where(path, text, after=None):
    hits = [i for i, x in enumerate(src[path], 1) if text in x and (after is None or i > after)]
    if len(hits) != 1:
        if after is None or not hits:
            raise SystemExit(f"ANCHOR {text!r} found {len(hits)}x in {path.name}")
        return hits[0]
    return hits[0]


HOPS = [
    ("1  argv        --code-mix declared (parser; default whole)", TRN, 'ap.add_argument("--code-mix", dest="code_mix"'),
    ("2  argv->cfg   make_config passes it into the config", TRN, "code_mix=a.code_mix)"),
    ("3  G-DVB       FLAG_CHECKS has an entry (a flag without one is refused)", TRN, '"code_mix": "lever:code_mix"'),
    ("4  cfg         V8FMiniConfig.code_mix (default whole) + the value check", MOD, '    code_mix: str = "whole"'),
    ("4b cfg         __post_init__ refuses any other value", MOD, "if self.code_mix not in CODE_MIX_MODES:"),
    ("5  built       built_levers() records the BUILT code_mix", MOD, '"code_mix": cfg.code_mix'),
    ("6  G-DVB       declared == built (declared_vs_built)", TRN, 'if built.get("code_mix") != a.code_mix:'),
    ("6b G-DVB       kind on a constant p_gt is refused (declared and does nothing)", TRN, 'if a.code_mix == "kind" and built["p_gt_start"] == built["p_gt_end"]'),
    ("7  G-LIVE      the probe reads the code O receives (called in main, before training)", TRN, "mix = code_mix_probe(model, probe, cfg)"),
    ("7b token       selfcheck.json carries code_mix_probe", TRN, '"code_mix_probe": mix'),
    ("8  _forward    the branch on the BUILT config", MOD, 'if cfg.code_mix == "kind":'),
    ("9  _forward    the MIXING LINE: where(mask, GT code, T's own code)", MOD, 'code = torch.where(self._code_mix_mask(b["lab_t"]["has"], p_gt, gen), self._code_gt(b["lab_t"]), code)'),
    ("10 mask        one Bernoulli(p_gt) draw per sample AND per unit", MOD, "draw = torch.rand(has.shape[0], N_CODE_UNITS, generator=gen, device=has.device) < p_gt"),
    ("11 mask        broadcast through CODE_UNIT_OF; has gates", MOD, "return (draw & has[:, None])[:, unit]"),
    ("12 _forward    `code` is what O reads", MOD, "traj60, p30 = self.run_o(b, adapt, last_now, ck_for_o, ck_ok, code)"),
]
print("PART 1 (static): every hop of argv -> config -> _forward -> the code O reads")
for name, path, text in HOPS:
    print(f"  {name:100s} {path.relative_to(ROOT).as_posix()}:{where(path, text)}")

COMMON = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5",
          "--arm", "M2b", "--steps", "4"]


def run(out, extra, patch=None):
    cap = {"mix": [], "own": None, "gt": None, "o": []}
    mix0, pred0, gt0, run_o0 = M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o

    def mix(self, has, p, gen=None):
        m = (patch or mix0)(self, has, p, gen)
        fr = sys._getframe(1)
        cap["mix"].append({"p": float(p), "grad": torch.is_grad_enabled(), "m": m.detach().clone(), "site": f"{Path(fr.f_code.co_filename).name}:{fr.f_lineno}"})
        return m

    def pred(self, tout):
        c = pred0(self, tout)
        cap["own"] = c.detach().clone()
        return c

    def gt(self, lab):
        g = gt0(self, lab)
        cap["gt"] = g.detach().clone()
        return g

    def run_o(self, bb, ad, last, ck_o, ck_ok, code):
        fr = sys._getframe(1)
        cap["o"].append({"code": code.detach().clone(), "own": cap["own"], "gt": cap["gt"], "grad": torch.is_grad_enabled(), "n_mix": len(cap["mix"]),
                         "site": f"{Path(fr.f_code.co_filename).name}:{fr.f_lineno}"})
        return run_o0(self, bb, ad, last, ck_o, ck_ok, code)
    M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o = mix, pred, gt, run_o
    try:
        rc = TR.main(["--out", str(out), *COMMON, *extra])
    except SystemExit as e:
        rc = f"SystemExit: {e}"
    finally:
        M.V8FMini._code_mix_mask, M.V8FMini._code_pred, M.V8FMini._code_gt, M.V8FMini.run_o = mix0, pred0, gt0, run_o0
    return rc, cap


tmp = Path(tempfile.mkdtemp(prefix="mixo_trace_"))
print("\nPART 2 (dynamic): TR.main --code-mix kind on the synthetic world; runtime call sites from the live frames")
rc, cap = run(tmp / "kind", ["--code-mix", "kind"])
assert rc == 0, rc
grad = [m for m in cap["mix"] if m["grad"]]
steps = 4
train, gate = grad[-steps:], grad[:-steps]
o_train = [o for o in cap["o"] if o["grad"]][-steps:]
print(f"  the mixing line was called {len(cap['mix'])} times: {len(gate)} G-LIVE forwards (p_gt 0.5) + {len(train)} training steps; no_grad calls (the probe): {len(cap['mix']) - len(grad)}")
print(f"  call site of the mix (from the frame):    {sorted({m['site'] for m in cap['mix']})}")
print(f"  call site of run_o (from the frame):      {sorted({o['site'] for o in cap['o']})}")
for k, (m, o) in enumerate(zip(train, o_train)):
    exact = torch.equal(o["code"], torch.where(m["m"], o["gt"], o["own"]))
    units = torch.stack([m["m"][:, torch.as_tensor(idx)].all(1) for idx in M.CODE_MIX_UNITS.values()], 1)
    part = int((units.any(1) & ~units.all(1)).sum())
    print(f"  train step {k}: p_gt {m['p']:.3f}  run_o's code == where(mask, GT, own): {exact}  samples with a PARTLY-GT code: {part}/{units.shape[0]}")
    assert exact
assert any(int((torch.stack([m['m'][:, torch.as_tensor(idx)].all(1) for idx in M.CODE_MIX_UNITS.values()], 1).any(1)
                & ~torch.stack([m['m'][:, torch.as_tensor(idx)].all(1) for idx in M.CODE_MIX_UNITS.values()], 1).all(1)).sum()) > 0 for m in train)
import json  # noqa: E402

sc = json.loads((tmp / "kind" / "selfcheck.json").read_text(encoding="utf-8"))
print(f"  selfcheck.json: declared_vs_built {sc['declared_vs_built']}; levers.code_mix {sc['levers']['code_mix']}; code_mix_probe pass {sc['code_mix_probe']['pass']} n_partial {sc['code_mix_probe']['n_partial']} "
      f"of {sc['code_mix_probe']['n_draws_samples']}")
ck = torch.load(tmp / "kind" / "ckpt_last.pt", map_location="cpu", weights_only=False)
print(f"  checkpoint cfg.code_mix: {ck['cfg']['code_mix']}")
rc_w, cap_w = run(tmp / "whole", [])
assert rc_w == 0
print(f"  CONTROL, default run (no --code-mix): the mixing line was called {sum(1 for m in cap_w['mix'] if m['grad'])} times in training (must be 0); probe n_partial "
      f"{json.loads((tmp / 'whole' / 'selfcheck.json').read_text(encoding='utf-8'))['code_mix_probe']['n_partial']} (must be 0)")
assert sum(1 for m in cap_w["mix"] if m["grad"]) == 0

print("\nPART 3 (the control): the mixing function deleted (an all-False mask = O always reads T's own code), --code-mix kind declared")
rc_d, cap_d = run(tmp / "deleted", ["--code-mix", "kind"], patch=lambda self, has, p, gen=None: torch.zeros(has.shape[0], M.CODE_DIM, dtype=torch.bool))
print(f"  trainer result: {rc_d}")
assert rc_d != 0 and "ZZDVB-REFUSEDZZ" in str(rc_d) and "NO labelled sample read a partly-GT code" in str(rc_d), rc_d
print("  => RED: the in-process gate refused to train (the check turns RED when the mixing line is gone)")
print("\nTRACE OK")
