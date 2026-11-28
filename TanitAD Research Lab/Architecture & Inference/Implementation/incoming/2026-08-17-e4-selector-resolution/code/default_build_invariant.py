"""Default-build invariant: 87,893,449 params / 405 keys. Run before AND after."""
import pyarrow  # noqa: F401  -- MUST precede torch on the dev box (0xC0000005)
import json, sys, pathlib
REPO = pathlib.Path(__file__).resolve()
while REPO.name != "TanitAD":
    REPO = REPO.parent
    if REPO == REPO.parent:
        REPO = pathlib.Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
        break
REPO = pathlib.Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "stack"))
import torch  # noqa
import train_v6_staged as T

a = T.build_parser().parse_args(
    ["--stage", "S-W", "--out", "x", "--v2-cache", "c"])
st = T.build_stack_from_args(a)
sd = st.state_dict()
n = sum(p.numel() for p in st.parameters())
print(json.dumps({"params": n, "keys": len(sd),
                  "selector": a.selector, "tac_goal_cond": a.tac_goal_cond,
                  "torch": torch.__version__}, indent=1))
