"""One-off MEASUREMENT (not a test): the DEFAULT build at the PRODUCTION default geometry
(V6Config() -- tac_vocab_version v7.0) is byte-identical per tensor, key order and RNG stream
to the TIP module; and the ON build adds exactly the port keys and the literal parameter delta.
Usage: python prod_geometry_byte_identity.py <my_stack_root> <tip_v6_py_copy>"""
import importlib.util
import json
import sys
from pathlib import Path

root, pre_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
sys.path.insert(0, root + "/scripts")
import torch  # noqa: E402

from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402

name = "tanitad.models._v6_pre_r1r4"
spec = importlib.util.spec_from_file_location(name, pre_path)
pre = importlib.util.module_from_spec(spec)
sys.modules[name] = pre
spec.loader.exec_module(pre)

torch.manual_seed(0)
old = pre.V6Stack(pre.V6Config())
r_old = torch.random.get_rng_state()
torch.manual_seed(0)
new = V6Stack(V6Config())
r_new = torch.random.get_rng_state()
so, sn = old.state_dict(), new.state_dict()
n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
res = {"tac_vocab_version": new.cfg.tac_vocab_version,
       "tip_params": n(old), "new_default_params": n(new),
       "tip_keys": len(so), "new_default_keys": len(sn),
       "key_order_identical": list(so) == list(sn),
       "tensors_differing": [k for k in so if not torch.equal(so[k], sn[k])],
       "rng_stream_identical": bool(torch.equal(r_old, r_new))}
del old
torch.manual_seed(0)
on = V6Stack(V6Config(tac_op_cond="e2e", max_speed_input_v6=True, plan_vmax_cap=True))
son = on.state_dict()
res["on_params"] = n(on)
res["on_minus_default_params"] = n(on) - n(new)
res["on_new_keys"] = sorted(set(son) - set(sn))
res["on_preexisting_tensors_moved"] = [k for k in sn if not torch.equal(sn[k], son[k])]
res["_expected_literal_delta"] = "tac_op_port 256*128+128=32,896 + vmax_tac 4*512+512=2,560 = 35,456"
res["_evidence_class"] = "MEASURED (CPU, seed 0, production default V6Config)"
print(json.dumps(res, indent=1))
