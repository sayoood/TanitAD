import importlib, sys
mods = [
    "alpasim_grpc",
    "alpasim_grpc.v0.egodriver_pb2_grpc",
    "alpasim_utils.geometry",
    "alpasim_wizard",
    "alpasim_runtime",
    "alpasim_controller",
    "alpasim_physics",
    "grpc",
    "numpy",
]
ok, bad = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, repr(e)[:160]))
print("OK:", ", ".join(ok))
for m, e in bad:
    print("FAIL:", m, "->", e)
# driver geometry helpers
try:
    from alpasim_utils.geometry import quat_to_yaw, yaw_to_quat_components  # noqa
    print("GEOM_HELPERS_OK")
except Exception as e:
    print("GEOM_HELPERS_FAIL", repr(e)[:160])
print("VERIFY_DONE bad=%d" % len(bad))
