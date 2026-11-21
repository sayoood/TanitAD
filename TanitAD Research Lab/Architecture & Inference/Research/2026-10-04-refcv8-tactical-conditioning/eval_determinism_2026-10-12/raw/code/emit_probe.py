"""trainer-vs-loader `out.r8_emit_keep`: a FRESH (never-scheduled) refcv8 build vs the loader's schedule at the
gate checkpoint's step 0 (refcv7_loader.py:517-519). CPU, rig, provisional A2 recipe (emit_start 1000)."""
import json, torch
import repro_lib as L
from tanitad.train import refcv8_train as RT
cfg, m_t = L.build("r8")                      # the 'trainer' model: built, never stepped
_c, m_l = L.build("r8")                       # the 'loader' model: same weights, schedule applied at step 0
m_l.load_state_dict(m_t.state_dict())
s8 = RT.checkpoint_r8_step(m_t.state_dict().keys(), 0)
live_l = RT.apply_emit_schedule(m_l, s8)
bt = L.R.batch(cfg, b=3)
a = L.digests(L.fwd_forked(cfg, m_t, bt)); b = L.digests(L.fwd_forked(cfg, m_l, bt))
print(json.dumps({"tanitad": __import__("tanitad").__file__, "emit_start": int(cfg.refcv8.emit_start),
                  "fresh_build_r8_emit_live": bool(m_t.core.decoder.r8_emit_live),
                  "loader_r8_step": s8, "loader_live": bool(live_l),
                  "trainer_vs_loader_differ": L.differ(a, b)}))
