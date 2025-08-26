Phase2Phase aligned long-axis strain
==============================

This repository contains code to perform the automatic calculation of the long-axis strain (LAS) from 
four-chamber long-axis (4CH) cardiac magnetic resonance (CMR) cine images.

This repository was used for the following paper:

**An automatic self-supervised phase-based approach to aligned long axis strain measurements in four chamber CMR**

For a more detailed description, a model definition and further results we refer to our 
<a target="_blank" href="https://link.springer.com/chapter/10.1007/978-3-031-94562-5_11">paper</a>.



Motivation
-


![Visual Abstract of Pipeline](/docs/img/Visual%20Abstract_V3.png)


Abstract
=
Cardiovascular magnetic resonance imaging (CMR) is the gold standard for quantifying ventricular function, from which 
several parameters are derived. Among these, long-axis strain (LAS) is valuable for diagnosis of cardiovascular diseases. 
Unlike global longitudinal strain (GLS), which needs multi-plane imaging, LAS can be effectively derived from a single-plane 
four-chamber long-axis (4CH) CMR. Conventional analysis focuses on end-diastolic (ED) to end-systolic (ES) LAS, 
overlooking intermediate dynamics that could help distinguish diseases.

In this study, we present a novel framework for estimating left ventricular LAS across five cardiac phases. 
The proposed method combines a self-supervised deformable image registration model for key frame detection with a 
supervised segmentation for landmark identification. LAS is calculated between ED and intermediate phases $K$ (ED2K) 
and between consecutive phases (K2K).

The methodology was developed and validated on the publicly available M&M2 dataset. The evaluation demonstrated 
significant differences between healthy individuals and patients with five of the seven cardiac diseases investigated 
not only in ED2ES, but also mid-systole to ES and ES to peak-flow. This emphasizes the diagnostic potential of 
phase-specific LAS analysis. The method is fully automated and fast, underscoring its potential for clinical application.

Paper:
--------
Please cite the following paper (accepted for the @ FIMH2025) if you use/modify or adapt parts of this repository:

**Bibtext**
```
@InProceedings{10.1007/978-3-031-94562-5_11,
  author="Mueller, Sarah Kaye
  and Jonathan Kiekenap
  and Koehler, Sven
  and Andre, Florian
  and Frey, Norbert
  and Greil, Gerald
  and Hussain, Tarique
  and Wolf, Ivo
  and Engelhardt, Sandy",

  editor="Chaniniok, Radek
  and Zou, Q.
  and Hussain, T.
  and Nguyen, H.H.,
  and Zaha, V.G.
  and Gusseva, M.",

  title="SAn Automatic Self-supervised Phase-Based Approach to Aligned Long-Axis Strain Measurements in Four Chamber Cardiovascular Magnetic Resonance Imaging",
  booktitle="Functional Imaging and Modeling of the Heart. FIMH 2025. Lecture Notes in Computer Science",
  year="2025",
  publisher="Springer",
  address="Cham",
  pages="113--125",
  abstract="Cardiovascular Magnetic Resonance Imaging (cardiac MRI) is the gold standard for quantifying ventricular function, from which several parameters are derived. Among these, long-axis strain (LAS) is valuable for diagnosis of cardiovascular diseases. Unlike global longitudinal strain (GLS), which needs multi-plane imaging, LAS can be effectively derived from a single-plane four-chamber long-axis (4CH) cardiac MRI. Conventional analysis focuses on end-diastolic (ED) to end-systolic (ES) LAS, overlooking intermediate dynamics that could help distinguish diseases.
In this study, we present a novel framework for estimating left ventricular LAS across five cardiac phases. The proposed method combines a self-supervised deformable image registration model for key frame detection with a supervised segmentation for landmark identification. LAS is calculated between ED and intermediate phases K (ED2K) and between consecutive phases (K2K).
The methodology was developed and validated on the publicly available M&M2 dataset. The evaluation demonstrated significant differences between healthy individuals and patients with four of the seven cardiac diseases investigated not only in ED2ES, but also mid-systole to ES and ES to peak-flow. This emphasizes the diagnostic potential of phase-specific LAS analysis. The method is fully automated and fast, underscoring its potential for clinical application. The code and reference annotations will be made publicly available. .",
  isbn="978-3-031-94561-8"
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, analyze the datasets
    │   ├── Evaluate       <- Evaluate the model performance, create plots
    │   ├── Predict        <- Use the models on new data
    │   ├── Train          <- Train a new model
    │   └── Test_IO        <- IO tests
    │   └── Test_Models    <- Tensorflow functional or subclassing tests
    │
    ├── exp            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── tensorboard_logs  <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Helper functions that will be used by the notebooks.
        ├── data           <- create, preprocess and extract the nrrd files
        ├── models         <- Modelzoo, Modelutils and Tensorflow layers
        ├── utils          <- Metrics, callbacks, io-utils, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations

Datasets
------------
For this project we used the 2D+t cine-SSFP 4CH CMR immages from the publicly available Multi-Disease, Multi-View & Multi-Center
Right Ventricular Segmentation in Cardiac MRI (<a target="_blank" href="https://www.ub.edu/mnms-2/">M&Ms-2</a>) dataset. The training of both models, registration and segmentation, was performed on the 200 datasets from the **Training set**. Evaluation was performed on the **Testing set.** The annotations for the 5 keyframes in the Testing set were made by an experienced physician. These manual labels including the end-diastole (ED), mid-systole (MS; maximum contraction resulting in a peak ejection between ED and ES), end-systole (ES), peak flow (PF; peak early diastolic relaxation) and mid-diastole (MD; phase before atrial contraction at the on-set of the p-wave). Please contact us if you are interested in these labels.

Training
------------
For the automatic detection of the LAS, you need to train two models:
- A **deformable image registration model** (self-supervised, no groundtruth annotations necessary)
- A **segmentation model** for at least the left ventricle (we used bi-ventricular segmentation as provided by <a target="_blank" href="https://www.ub.edu/mnms-2/">M&Ms-2</a>

### Deformable image registration model
Our trainings script for deformable image registration support single and multi-GPU training (data-parallelisms) and should run locally, and on clusters. The trainings-flow is as follows:
1. Re-use or modify one of the example configs provided in <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/data/configs">data/configs</a>
2. Run src/models/train_regression_model.py, which parse the following arguments:
  ```
  - cfg (str): Path to an experiment config, you can find examples in data/configs
  - data (str): Path to the data-root folder with 3d nrrd or nii.gz files (4CH single slice cine CMR)
  - inmemory (bool) in memory preprocessing for cluster-based trainings         
  ```
3. Our trainings script will sequentially train four models on the corresponding dataset splits. The experiment config, model-definition/-weights, trainings-progress and tensorboard logs etc. will be saved automatically. After each model convergence we call the prediction scripts on the corresponding fold and save the predicted files into the sub-folders f0-f3. You can also train the model with a single fold. The number of folds is defined in the config file as:
    ```
   "FOLDS":[0, 1, 2, 3],
    ```
    For a split in several folds you have to provide a **_df_kfold.csv_** file. Here you should have a row for each patient and fold and if it belons to "train" or "test" in the modality column.
    You can find an example for a df_kfold.csv in data/mnms-2.

4. Each fold (f0...,f3) contains the following model specific files:
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
For the segmentation model we used the publicly available <a target="_blank" href="https://github.com/MIC-DKFZ/nnUNet"> nnU-Net </a>  framework. 
Please use 2D 4CH cine CMR images for the training of the model. As nnU-Net does not support time sequences, our pipeline will automatically compute the segmentation per timestep and compute the cine masks.

Please use the following assignment for segmentation:
- LV bloodpool: 1
- LV MYO (with septum): 2
- RV bloodpool: 3

Setup
------------
### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
- Clone repository
```
git clone %reponame%
cd %reponame%
```
- Create a conda environment from enrionment.yaml (environment name will be cmr-las)
```
conda env create --file environment.yaml
```
- Activate environment
```
conda activate phase_detection
```
- Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
- Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab
```
python -m ipykernel install --user --name pdet --display-name "phase_det kernel"
```
- Enable interactive widgets in Jupyterlab
Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
- Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
  
Further infos on how to enable the jupyterlab-extensions:

<a target="_blank" href="https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension">JupyterLab</a>

Affiliation
--------
For more Information of our work, please visit our Webside:
<a target="_blank" href="https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/institute-for-artificial-intelligence-in-cardiovascular-medicine-aicm">Institute for Artificial Intelligence in Cardiovascular Medicine </a>

We are part of the Department of Cardiology, Angiology, Pneumology, Heidelberg University Hospital, Heidelberg, Germany
