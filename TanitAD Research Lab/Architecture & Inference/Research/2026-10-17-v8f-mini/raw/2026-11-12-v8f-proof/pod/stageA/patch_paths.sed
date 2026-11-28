s#^OPS = "/root/ops"#OPS = "/workspace/v8proof/stageA/ops"#
s#^DATA = "/root/data"#DATA = "/workspace/v8proof/data/podwork"#
s#^WORKERS_DEFAULT = 13 .*#WORKERS_DEFAULT = 5                       \# 7.65-core cgroup quota; never nproc#
s#"TANITAD_STACK", "/root/tad/stack"#"TANITAD_STACK", "/workspace/v8proof/tree/stack"#
s#"/root/.hf_token"#"/root/.hf_token"#
s#^PEP = os.environ.get("V8F_LORA_BUILDER", "/root/identity_pod/code/pull_encode_pooled.py")#PEP = os.environ.get("V8F_LORA_BUILDER", "/workspace/v8proof/stageA/code/pull_encode_pooled.py")#
s#^SHIP = os.environ.get("V8F_LORA_SHIP", "/root/tad/stack/scripts/dinov3_fp8_encode_ship.py")#SHIP = os.environ.get("V8F_LORA_SHIP", "/workspace/v8proof/tree/stack/scripts/dinov3_fp8_encode_ship.py")#
s#^LEDGER = Path("/root/data/v2ep_ledger.jsonl")#LEDGER = Path("/workspace/v8proof/data/podwork/v2ep_ledger.jsonl")#
s#^BUILD_LOG = Path("/root/logs/v2ep_run.log")#BUILD_LOG = Path("/workspace/v8proof/hierrep/v2ep_run.log")#
s#^V2EP = Path("/root/data/v2ep")#V2EP = Path("/workspace/v8proof/data/podwork/v2ep")#
s#^BOX2_PULLED = Path("/root/identity_pod/cache_c") .*#BOX2_PULLED = Path("/workspace/v8proof/data/podwork/none_box2_pulled")   \# no box-2 comparison files on this pod#
s#ap.add_argument("--out", default="/root/data/cache_pod")#ap.add_argument("--out", default="/workspace/v8proof/data/cache_pod")#
s#open("/root/models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors", "rb")#open(os.environ["DINO_SAFETENSORS"], "rb")#
