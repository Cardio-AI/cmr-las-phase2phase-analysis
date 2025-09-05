Dataset
------------
For this project we used the 2D+t cine-SSFP 4CH CMR images from the publicly available Multi-Disease, Multi-View & Multi-Center
Right Ventricular Segmentation in Cardiac MRI (<a target="_blank" href="https://www.ub.edu/mnms-2/">M&Ms-2</a>) dataset.

The training of both models, **registration** and **segmentation**, was performed on the 200 datasets from the **Training set**.
Evaluation was performed on the **Testing set**, for which annotations of 5 keyframes were provided by an experienced physician.
These manual labels include:
- End-diastole (ED)
- mid-systole (MS): maximum contraction resulting in a peak ejection between ED and ES
- end-systole (ES)
- peak flow (PF): peak early diastolic relaxation
- mid-diastole (MD): phase before atrial contraction at the on-set of the p-wave

Please contact us if you are interested in these labels.

### Structure

The project uses one data-root folder, which should at least contain a folder  ```lax ``` with the cine CMR files.
Additionally, the folder can contain a csv file for splitting the data **df_kfold.csv** and one with the ground-truth keyframe annotations.

 ```
   ├── lax (folder with cine 4CH CMR) 
   ├── dataset.json (Json with all necessary information about labels, suffix and post processing)
   ├── df_kfold.csv (dataframe with split for k-fold split validation)
   └── phases.csv (dataframe with keyframes for calculation of cyclic frame difference)
 ```

#### Cine 4CH CMR

If you want to use your own data, make sure the images are provided in NRRD ( ```nrrd ```) or NIfTI ( ```.nii.gz ```)format,
and that the 4CH cine CMR images consist of a single slice.
The expected format is:
 ```
t × x × y × z
 ```
where z = 1.

In addition, you must set the correct temporal dimension in the configuration file to at least t + 1 ( ```"T_SHAPE" ```).
By default, it is set to 40. If your sequence has more than 39 time steps, you need to increase this number accordingly.

#### File naming conventions
File names are used to extract patient IDS, which must mathc those in phases.csv and df_kfold.csv.

Supported regex patterns (from source datasets):
- ```r'\d+-([a-zA-Z0-9]+)_\d{4}-\d{2}-\d{2}.*'```  -> GCN: 0000-0ae4r74l_1900-01-01_...
- ```r'(\d+)_LA_CINE.*'```  -> M&Ms and M&Ms-2: 039_LA_CINE.nii.gz
- ```r'patient(\d+)_.*'```  -> ACDC: patient001_4d.nii.gz
- 
If your dataset uses a different convention, add your regex pattern to:
```src/data/Dataset.py -> extract_id```.


#### Dataframe for k-fold split validation
For a split in several folds you have to provide a **_df_kfold.csv_** file. 
Here you should have a row for each patient and fold and if it belongs to "train" or "test" in the modality column.
You can find an example for a df_kfold.csv in ```data/mnms-2```.


