import importlib, sys
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
mods = ["torch", "grpc", "alpasim_grpc", "alpasim_grpc.v0.egodriver_pb2_grpc",
        "alpasim_grpc.v0.sensorsim_pb2", "alpasim_utils.geometry",
        "tanitad.refs.refc", "tanitad.data.calib", "tanitad.data.comma2k19",
        "refc_v12_cache", "numpy", "cv2"]
for m in mods:
    try:
        importlib.import_module(m)
        print("OK ", m)
    except Exception as e:
        print("FAIL", m, "->", repr(e)[:120])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
