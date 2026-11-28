import os
from huggingface_hub import snapshot_download
for repo, sub in (('Sayood/tanitad-refc-base','refc-base'),
                  ('Sayood/tanitad-refc-xl','refc-xl'),
                  ('Sayood/tanitad-refc-base-e1b-clsft','refc-base-e1b-clsft'),
                  ('Sayood/flagship-v4.2b','flagship-v4.2b')):
    try:
        p = snapshot_download(repo, repo_type='model',
                              local_dir=f'/home/nvidia/models/{sub}')
        sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)
        print(f'PULLED {sub} {sz/1e6:.0f} MB', flush=True)
    except Exception as e:
        print(f'FAIL {sub} {type(e).__name__}: {str(e)[:120]}', flush=True)
print('ALL_DONE', flush=True)
