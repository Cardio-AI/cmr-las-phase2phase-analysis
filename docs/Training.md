Training
------------
To automatically detect LAS, you need to train two models:
- **Deformable image registration model** (self-supervised, no groundtruth annotations necessary)
- **Segmentation model** for at least the left ventricle (we used bi-ventricular segmentation from <a target="_blank" href="https://www.ub.edu/mnms-2/">M&Ms-2</a>

### Deformable image registration model
Our trainings script supports single- and multi-GPU training (local or  cluster). Trainings work-flow:
1. **Config setup**
   - Start from or modify an example config in <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/data/configs">data/configs</a>
2. **Run Training**
      ```
    python src/models/train_regression_model.py \
   - cfg_reg <path_to_config> \
   - data_json <path_to_dataset_json> \
   - data <data_root> \
   - inmemory <true/false>         
    ```
    -  ```cfg_reg ```: Path to an experiment config (examples in ```data/configs```)
    - ```data```: Root folder with 3d nrrd or nii.gz files (4CH single slice cine CMR; see <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">data/Data</a>)
    - ```data_json```: Path to dataset.json, which contains all necessary information about labels, suffix and post-processing  (examples in ```data/configs```).
    - ```inmemory```: Enables in-memory pre-processing for cluster-based trainings

3.  **Cross Validation**
    - Script can train on multiple folds sequentially (defined in config ```"FOLDS":[0, 1, 2, 3],```). For multi-fold splits, you must supply a **_df_kfold.csv_** file (see <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">data/Data</a>)
    - If no folds are provided, the model is trained on all available data
    - After training, predictions ar automatically saved into f0-f3 subfolders. 

4.  **Outputs per fold**
   ```
   ├── config (config used in this experiment fold)
   ├── Log_errors.log (logging.error logfile)
   ├── Log.log (console and trainings progress logs, if activated
   ├── model (model graph and weights for later usage
   ├── model.png (graph as model png)
   ├── model_summary.txt (layer input/output shapes as structured txt file
   └── tensorboard_logs (tensorboard logfiles: train-/test scalars and model predictions per epoch)
   ```
   
### Segmentation model
For segmentation, we used the publicly available <a target="_blank" href="https://github.com/MIC-DKFZ/nnUNet"> nnU-Net </a>  framework. 
Please use 2D 4CH cine CMR images for the training of the model. 

Following training, please also run the command to automatically determine the best configuration, as postprocessing will be applied.

Note: nnU-Net does not support time sequences. During LAS estimation, our pipeline processes each timestep independently and assembles cine masks.

Segmentation labels:
- LV bloodpool: 1
- LV MYO (with septum): 2
- RV bloodpool: 3