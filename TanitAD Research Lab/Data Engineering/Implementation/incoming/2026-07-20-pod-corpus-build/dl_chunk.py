import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
import time
t0=time.time()
p = hf_hub_download('commaai/comma2k19', 'raw_data/Chunk_1.zip',
                    repo_type='dataset', local_dir='/root/comma_dl')
print('DONE', p, round(time.time()-t0,1),'s', flush=True)
