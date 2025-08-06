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
@InProceedings{xx.xxxx/xxx-x-xxx-xxxx,
  author="Mueller, Sarah Kaye
  and Jonathan K
  and Hussain, Tarique
  and Hussain, Hamza
  and Young, Daniel
  and Sarikouch, Samir
  and Pickardt, Thomas
  and Greil, Gerald
  and Engelhardt, Sandy",
  editor="Camara, Oscar
  and Puyol-Ant{\'o}n, Esther
  and Qin, Chen
  and Sermesant, Maxime
  and Suinesiaputra, Avan
  and Wang, Shuo
  and Young, Alistair",
  title="Self-supervised Motion Descriptor for Cardiac Phase Detection in 4D CMR Based on Discrete Vector Field Estimations",
  booktitle="Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers",
  year="2022",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="65--78",
  abstract="Cardiac magnetic resonance (CMR) sequences visualise the cardiac function voxel-wise over time. Simultaneously, deep learning-based deformable image registration is able to estimate discrete vector fields which warp one time step of a CMR sequence to the following in a self-supervised manner. However, despite the rich source of information included in these 3D+t vector fields, a standardised interpretation is challenging and the clinical applications remain limited so far. In this work, we show how to efficiently use a deformable vector field to describe the underlying dynamic process of a cardiac cycle in form of a derived 1D motion descriptor. Additionally, based on the expected cardiovascular physiological properties of a contracting or relaxing ventricle, we define a set of rules that enables the identification of five cardiovascular phases including the end-systole (ES) and end-diastole (ED) without usage of labels. We evaluate the plausibility of the motion descriptor on two challenging multi-disease, -center, -scanner short-axis CMR datasets. First, by reporting quantitative measures such as the periodic frame difference for the extracted phases. Second, by comparing qualitatively the general pattern when we temporally resample and align the motion descriptors of all instances across both datasets. The average periodic frame difference for the ED, ES key phases of our approach is {\$}{\$}0.80{\backslash}pm {\{}0.85{\}}{\$}{\$}0.80{\textpm}0.85, {\$}{\$}0.69{\backslash}pm {\{}0.79{\}}{\$}{\$}0.69{\textpm}0.79which is slightly better than the inter-observer variability ({\$}{\$}1.07{\backslash}pm {\{}0.86{\}}{\$}{\$}1.07{\textpm}0.86, {\$}{\$}0.91{\backslash}pm {\{}1.6{\}}{\$}{\$}0.91{\textpm}1.6) and the supervised baseline method ({\$}{\$}1.18{\backslash}pm {\{}1.91{\}}{\$}{\$}1.18{\textpm}1.91, {\$}{\$}1.21{\backslash}pm {\{}1.78{\}}{\$}{\$}1.21{\textpm}1.78). Code and labels are available on our GitHub repository. https://github.com/Cardio-AI/cmr-phase-detection.",
  isbn="978-3-031-23443-9"
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



Setup native with OSX or Ubuntu
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
- Create a virtual environment either via virtualenv or conda
```
make environment
```
- Install dependencies via requirements.txt
```
make requirement
```
- Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Further infos on how to enable the jupyterlab-extensions:

<a target="_blank" href="https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension">JupyterLab</a>
