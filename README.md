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


How to get started?
------------
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Setup.md">Setup</a>
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">Data</a>

#### Training
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Training.md">Training</a>

#### Inference
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Keyframe-detection.md">Keyframe detection</a>
- <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Phase2Phase-LAS.md">Phase to phase LAS computing</a>

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


Affiliation
--------
For more Information of our work, please visit our website:
<a target="_blank" href="https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/institute-for-artificial-intelligence-in-cardiovascular-medicine-aicm">Institute for Artificial Intelligence in Cardiovascular Medicine </a>

We are part of the Department of Cardiology, Angiology, Pneumology, Heidelberg University Hospital, Heidelberg, Germany
