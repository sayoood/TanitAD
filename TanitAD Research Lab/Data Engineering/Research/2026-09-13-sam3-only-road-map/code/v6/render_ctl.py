"""CONTROL for the native/class-7 renderer patch: rendering an already-rendered v5 clip with the patched module must give
byte-identical PNGs (v5 maps have no class 7 and no native side cameras, so nothing may change for them)."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sam3map_render_v4_new as r4  # noqa: E402
r4.V2 = Path(os.environ["SAM3MAP_ROOT"])
sys.argv = ["sam3map_render_v4_new.py"] + sys.argv[1:]
r4.main()
