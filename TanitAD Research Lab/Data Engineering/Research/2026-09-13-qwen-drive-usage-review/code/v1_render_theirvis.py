import sys, shutil, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "/home/nvidia/qwendrive/qwen-drive/src")
from qwen_drive_perception.dataset import PerceptionFrame
from qwen_drive_perception.visualize import render_frame
Q = Path("/home/nvidia/qwendrive"); T = Q / "v2" / "v1_render_frames"; O = Q / "v2" / "v1_theirvis"
O.mkdir(parents=True, exist_ok=True)
for tok in sys.argv[1:]:
    src, dst = Q / "frames" / tok, T / tok
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = dict(np.load(src / "gt.npz"))
    # v1 packed no occ/map GT; the upstream renderer requires both keys. Fill them EMPTY so
    # the GT panels are blank, never invented, and draw v1's OWN predictions exactly as upstream decodes them.
    np.savez(dst / "gt.npz", boxes=g["boxes"], labels=g["labels"], occ=np.full((200, 200, 16), 9, np.int8), map=np.zeros((200, 400), np.int8))
    r = dict(np.load(Q / "out" / f"{tok}.npz"))
    img = render_frame(PerceptionFrame(dst), r, score_threshold=0.25)
    cv2.imwrite(str(O / f"{tok}.png"), img); print("rendered", tok, img.shape)
