Keyframe detection
-------

Usually the inference/evaluation scripts are executed automatically per fold when running ```train_regression_model.py```. 
Nevertheless, you can also run predictions manually on new datasets or groundtruth or with modified experiment parameters.

### Run Prediction script

 ```
    python src/models/predict_phase_reg_model.py \
        -exp <experiment_root> \
        -data <data_root>
 ```

-  ```exp``` (str): Path to the root of one experiment
-  ```data``` (str): Path to the data-root folder (see <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">data/Data</a>)


If you want to evaluate cyclic frame differences between predicted keyframes and ground truth, you must provide a ```phases.csv``` file
(see <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">data/Data</a>).

By default, predictions are stored in the experiments' folder.
