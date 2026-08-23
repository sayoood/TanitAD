#!/usr/bin/env bash
# CPU smoke for --goal-mode BEFORE any GPU run: does the switch import, does the
# oracle branch delegate to _goal_inputs verbatim, do produced/neutral build
# legal indices for the head's embeddings?
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
export TANITEVAL_STACK_OVERRIDE=/root/v4eval/stack
cd /root/v4eval/stack/scripts || exit 1

python3 - <<'PY'
import torch, inspect
import goal_modes as gm
from tanitad.models.flagship_v4 import v4_config, FlagshipV4Head
from tanitad.models.strategic_goal import GoalScalarConfig, GoalScalarHead
from train_flagship_v4 import _goal_inputs

print("GOAL_MODES:", gm.GOAL_MODES)

cfg = v4_config(); cfg.window = 8
cfg.state_dim = cfg.readout_grid ** 2 * cfg.d_cell     # the real geometry (2048)
cfg.cond_imagination = False          # as in the 15k run's own config.json
cfg.decoder.d = 32; cfg.decoder.layers = 1; cfg.decoder.n_heads = 2
cfg.n_anchors = 16
head = FlagshipV4Head(cfg).eval()
B = 6
states = torch.randn(B, cfg.window, cfg.state_dim)
v0 = torch.rand(B) * 20.0
batch = {"vt_band": torch.randint(0, 23, (B,)),
         "route": torch.randint(0, 4, (B,)),
         "route_graded": torch.rand(B) * 2 - 1,
         "strat_scalars": torch.randn(B, 4),
         "strat_scalar_mask": torch.ones(B, 4, dtype=torch.bool)}
gh = GoalScalarHead(GoalScalarConfig(in_dim=cfg.state_dim)).eval()

# --- 1. ORACLE must be byte-for-byte the same dict _goal_inputs returns -------
ref = _goal_inputs(cfg, batch, v0)
got, rec = gm.resolve_goal("oracle", head=head, batch=batch, v0=v0,
                           states=states, goal_head=gh)
assert set(ref) == set(got), (sorted(ref), sorted(got))
for k in ref:
    assert torch.equal(torch.as_tensor(ref[k]), torch.as_tensor(got[k])), k
print("OK oracle == _goal_inputs, keys:", sorted(got), "| rec:", rec)

# --- 2. produced / neutral: legal indices + a real forward --------------------
for mode in ("produced", "neutral"):
    kw, rec = gm.resolve_goal(mode, head=head, batch=batch, v0=v0,
                              states=states, goal_head=gh)
    kw.pop("scalars", None); rec.pop("scalars", None)
    r, vb = kw["route"], kw["vt_band"]
    assert r.dtype == torch.long and int(r.min()) >= 0 and int(r.max()) <= gm.ROUTE_DROPPED, r
    assert vb.dtype == torch.long and int(vb.min()) >= 0 and int(vb.max()) <= gm.VT_DROPPED, vb
    out = head(states, v0, lambda_plan=1.0, **kw)
    print(f"OK {mode}: route={r.tolist()} vt_band={vb.tolist()} "
          f"graded={[round(float(x),3) for x in kw['route_graded']]} "
          f"traj{tuple(out['traj'].shape)} | fallback={rec.get('fallback')}")

# --- 3. produced with NO goal_head must REFUSE, not silently substitute -------
try:
    gm.resolve_goal("produced", head=head, batch=batch, v0=v0, states=states,
                    goal_head=None)
    print("FAIL: produced with no goal_head did NOT refuse")
except SystemExit as e:
    print("OK refusal:", str(e)[:90], "...")
kw, rec = gm.resolve_goal("produced", head=head, batch=batch, v0=v0,
                          states=states, goal_head=None, allow_fallback=True)
print("OK explicit fallback ->", rec["fallback"], "|", rec["_read"][:60], "...")

# --- 4. provenance block ------------------------------------------------------
for mode in gm.GOAL_MODES:
    p = gm.provenance(mode, cfg=cfg)
    print(f"  prov[{mode}]: source={p['goal_source']} deployable={p['deployable']} "
          f"oracle_fields_fed={p['oracle_fields_fed']}")

# --- 5. eval_flagship_v4 still parses + carries the flag ----------------------
import eval_flagship_v4 as E
src = inspect.getsource(E.main)
assert "--goal-mode" in src and "--goal-fallback" in src
print("OK eval_flagship_v4 exposes --goal-mode/--goal-fallback; default =",
      [l for l in src.splitlines() if 'default="oracle"' in l])
print("SMOKE_OK")
PY
