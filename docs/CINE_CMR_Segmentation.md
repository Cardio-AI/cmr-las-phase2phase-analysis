Cine CMR Segmentation inference
------------

To generate segmentation masks from CMR sequences, use the notebook:

```notebookds/Evaluate/segmentation_model_analysis.ipynb```

### Workflow
1. **Run Sections 0**

   These sections handle data loading. Select the folder containing your CMR sequences. 
   If you use a naming convention different from the default, update the parameter ```file_regex``` in Section 0.1 accordingly.

2. **Skip Section 1** 

    This Section provides some playaround to test different segmentations for individual subjects interactively.

3. **Run Section 2**

   In the first two cells, select the output folder where you want to store your CINE masks.
   In the next cells, specify the folder containing the segmentation model you wish to use.
   Skip Section 2.1, as it applies dataset-specific preprocessing not required for the default workflow.
4. **Configure Section 2.2**

    The notebook attempts to infer your nnU-Net settings from the selected path. 
    Update the list of f to choose which folds you want to use, and whether to apply postprocessing. 
    Check the output of the second cell in Section 2.2 to ensure it matches your nnU-Net configuration.
       
5. **Run Cell 3 in 2.2**

    This step performs the segmentation with your specified nnU-Net settings. 
    Depending on your dataset and model, this may take some time.
   
The notebook generates segmentation masks in a subfolder named `masks` inside your chosen output directory.

### Additional features
The notebook also includes visualisation and analysis tools:
 - Segmentation visualisation: overlay masks on CMR sequences
 - Plots: generate left ventricular blood-volume curves
 - key-frame detection from blood volume curves