Phase2Phase LAS Calculation
------------

This section describes the workflow for running phase-to-phase (K2K) and ED-to-phase (ED2K) LAS calculation and the subsequent LAS anaylsis.

#### Prerequisites
Before running the analysis, you must have already:
1. <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Training.md">Trained</a>
both the registration and segmentation models

2. Performed inference for:
   - <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Keyframe-detection.md">Keyframe detection</a>
   - <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/CINE_CMR_Segmentation.md">Segmentation</a>

After the inference you will have to folder:
**Registration folder** (from keyframe detection inference):
- ```cfd.csv```: Calculated cyclic frame difference between prediction and ground truth (if available)
- ```gt_phases.csv```: Ground-truth cardiac keyframes (if available)
- ```pred_phases.csv```: Predicted cardiac keyframes (Necessary for K2K and ED2K calculation)

**Data folder with Segmentation**
- Subfolder ```lax``` containing:
  - CINE NIFTI files (```lax/*CINE.nii.gz```)
  - Corresponding NIFTI files (```lax/*_masks.nii.gz```)
  - Both must follow the same naming conventions
- ```dataset_information.csv```: Metadata as described in - <a target="_blank" href="https://github.com/Cardio-AI/cmr-las-phase2phase-analysis/tree/main/docs/Data.md">Data</a>
for disease split

#### Running LAS Analysis

Once the prerequisites are satisfied, you can run the Jupyter notebook:

```notebookds/Evaluate/las_analysis.ipynb```

This notebook allows you to calculate LAS either ED2K or K2K

##### Notebook features
The notebook provides multiple options for post-processing and visualisation:
- **Export results**
  - As Excel files (including anatomical landmark coordinates)
  - As overlay masks (nii.gz) with anatomical landmarks and long axis
- **Statistical analysis**
- **Visualisation**
  - Generate plots (mean, median, per patient, per pathology)
  - Create videos
