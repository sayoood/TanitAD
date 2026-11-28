import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
import time
t0=time.time()
p = hf_hub_download('nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams',
                    'cosmos_synthetic/single_view/generation.tar.gz.part-000',
                    repo_type='dataset', local_dir='/root/cosmos_dl')
print('DONE', p, round(time.time()-t0,1),'s', flush=True)
