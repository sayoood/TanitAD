# PUBLISHED-CODE extract - E2E-AD/AD-MLP @ 4b93ba085ee47474152f282177865796ea577fc0
# path: README.md
# git blob sha1 of the fetched file: 8eefdc561bec7564eeec3ab919a15318e1326666 (7293 B); verified against the repo tree listing at the pinned commit: MATCH
# fetched 2026-09-19T13:46:02Z from raw.githubusercontent.com (read-only source text; excerpts only, attribution retained)

## lines 37-70: results table + 'applies the ST-P3 evaluation'
   37  ## Results
   38  - Open-loop planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit). 
   39  
   40  | Method | L2 (m) 1s $\downarrow$ | L2 (m) 2s $\downarrow$ | L2 (m) 3s $\downarrow$ | Avg L2 (m) | Col. (%) 1s $\downarrow$ | Col. (%) 2s $\downarrow$ | Col. (%) 3s $\downarrow$ | Avg Col. (%)
   41  | :---: | :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: |
   42  | ST-P3 | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
   43  | UniAD | 0.48 | 0.96 | 1.65 | 1.03 | **0.05** | 0.17 | 0.71 | 0.31 |
   44  | VAD-Tiny | 0.20 | 0.38 | 0.65 | 0.41 | 0.10 | 0.12 | 0.27 | 0.16 | 
   45  | VAD-Base | **0.17** | 0.34 | 0.60 | 0.37 | 0.07 | **0.10** | **0.24** | **0.14** |
   46  | Ours | 0.20 | **0.26** | **0.41** | **0.29** | 0.17 | 0.18 | **0.24** | 0.19 |
   47  
   48  ## Get Started
   49  
   50  * Environment
   51    Linux, Python==3.7.9, CUDA == 11.2, pytorch == 1.9.1 or paddlepaddle == 2.3.2. Besides, follow instruction in ST-P3 for running its evaluation process.
   52    ```
   53    cd deps/stp3
   54    conda env create -f environment.yml
   55    ```
   56  
   57  * Prepare Data   
   58  Download the [nuScenes](https://www.nuscenes.org/download) Dataset.
   59  
   60  * Pretrained weights   
   61  To verify the performance on the nuScenes Dataset, we provide the pretrained model weights ([Google Drive](https://drive.google.com/drive/folders/1CJa54-Ft8qakR4EyRtxvswQxT1dgPB_9) and [Baidu Netdisk](https://pan.baidu.com/s/1cEDETxG-HHwyC7ATBk_hyQ?pwd=9fbf)). Please download them (paddle checkpoint, token of validation set...) to the root directory of this project.
   62  
   63  * Paddle Evaluation   
   64    ```
   65    python paddle/model/AD-MLP.py
   66    python deps/stp3/evaluate_for_mlp.py
   67    ```
   68    The first line saves the predicted 6 frames' trajectories of the next 3s in output_data.pkl. And the second line applies the ST-P3 evaluation on it. The final evaluation output contains the L2 error and collision rate in the next 1, 2 and 3s.
   69    
   70    Two versions of evaluation metrics are provided: online and offline. The offline version uses pre-stored ground truth and is far faster than online one. The code defaults to offline.

## lines 99-100: collision-rate caveat
   99  * Collision rate evaluation:
  100  We have observed that the evaluation of model collision rates is sensitive to certain samples. One typical example is when the ego vehicle is in a stationary state, the model often predicts trajectories for the next 3 seconds that are very close to the origin (but not exactly 0). If obstacles exist in the range of [0, 0.5m) in the x,y dimensiosn around the origin, the model's predictions of coordinates with very small absolute values can introduce unstable systematic errors due to the resolution of the occupancy map. To mitigate this issue, we recommend processing the model outputs in the deps/stp3/evaluate_for_mlp.py or at the original model inference stage. For instance, you can set the model's predicted trajectory to zero if the distance from the origin is smaller than a little threshold, e.g. 1e-2m. This approach is similar to the filtering of ground truth trajectories that collide in the original evaluation code, as both methods aim to remove systematic errors. We also suggest reviewing cases where collisions occur.

