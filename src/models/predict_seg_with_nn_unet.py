import glob
import logging
import os

import subprocess
import numpy as np
import SimpleITK as sitk


def predict_segmentation(file_paths, save_dir, clean_memory=False, dataset="007", configuration="2d", fold="0", permute=None, trainer=None, suffix=None):
    """
    Function to predict use_segmentation of all patients given in file paths.
    Every single time steps is stacked in one nifti and saved in the given saving directory.

    """
    path = os.path.join(save_dir, "masks")
    create_directory(path)

    if suffix is None:
        suffix = ".nrrd"

    for file in file_paths:
        # Predict use_segmentation for one file
        print(f"Processing: {file}")
        patient_id, input_path, output_path = segmentation_with_nn_unet(file, root_path=save_dir, dataset=dataset,
                                                                        configuration=configuration, fold=fold, permute=permute, suffix=suffix)

        # Load and stack masks for each time step. Save
        masks = collect_files(os.path.join(output_path), "*_t*" + suffix)
        f_name = path + "/" + str(patient_id) + "_masks" + suffix

        if len(masks) == 0:
            masks = collect_files(input_path, "*_t*" + suffix)
            f_name = path + "/" + str(patient_id) + "_masks" + suffix
        stacked_masks = stack_nifti(masks)
        adjust_z_spacing(stacked_masks, z_spacing=1)

        sitk.WriteImage(stacked_masks, f_name)

        # Remove folders used for nnU-Net use_segmentation after loading files
        if clean_memory:
            clean_directory(input_path)
            clean_directory(output_path)


def segmentation_with_nn_unet(patient_file, root_path=None, dataset="003", configuration="100epochs__2d", fold="0", permute=None,trainer=None, suffix=".nrrd"):
    """
    Function to predict use_segmentation of one patient with NN unet.

    Parameters
    :param patient_file: either path (type==str) to a single patient CINE or a np.array, containing one patient CINE
    :param root_path: root directory where CINE time steps and masks are going to be saved
    :param clean_memory: whether to delete folders with single time steps and masks (folder are necessary for nnU-Net use_segmentation)

    Return
    :return: np. array with predicted use_segmentation of one patient with NN unet
    """
    if type(patient_file) == str:
        patient_id = patient_file.split("/")[-1].split('.')[0]
    else:
        patient_id = "xx"

    # Create input and output path for the nnU-Net use_segmentation
    input_path, output_path = create_folders_for_segmentation(patient_id, root_path)

    # Fill input folder with single timesteps of patient
    save_single_time_steps(patient_file, saving_path=[os.path.join(input_path, patient_id), suffix], permute=None)

    # Check z-dimension and fix, if necessary
    for patient in collect_files(input_path, "*" + suffix):
        check_z_spacing(patient)

    # Run nnU-Net use_segmentation in shell
    run_segmentation_jp(input_folder=input_path, output_folder=output_path, dataset=dataset,
                        configuration=configuration, fold=fold, trainer=trainer)

    return patient_id, input_path, output_path


def open_nifti_sitk(file):
    nifti = sitk.ReadImage(file)
    return nifti


def get_single_slice_sitk(file, timestep=0):
    nifti = open_nifti_sitk(file)

    # get the shape of NIfTI
    print(nifti.GetSize())

    single_timestep = nifti[..., timestep]

    print(single_timestep.GetSize())


def permute_image_with_direction(image, permutation):
    """
    Permute image axes and update direction and origin accordingly.
    """
    # Permute image data
    permuted_image = sitk.PermuteAxes(image, permutation)

    # Update direction matrix
    old_direction = image.GetDirection()
    dim = image.GetDimension()
    old_dir_matrix = [old_direction[i:i + dim] for i in range(0, len(old_direction), dim)]

    # Permute direction matrix rows and columns
    new_dir_matrix = [[old_dir_matrix[permutation[i]][permutation[j]] for j in range(dim)] for i in range(dim)]

    # Flatten the new direction matrix
    new_direction = [new_dir_matrix[i][j] for i in range(dim) for j in range(dim)]
    permuted_image.SetDirection(new_direction)

    # Update origin
    old_origin = image.GetOrigin()
    new_origin = tuple(old_origin[permutation[i]] for i in range(dim))
    permuted_image.SetOrigin(new_origin)

    return permuted_image


def save_single_time_steps(file, saving_path=None, permute = None):
    nifti = open_nifti_sitk(file)

    extracted_steps = [nifti[:, :, :, i] for i in range(nifti.GetSize()[-1])]
    for i, timestep in enumerate(extracted_steps):
        if i <= 9:
            file_path = saving_path[0] + '_t0' + str(i) + '_0000' + saving_path[1]
        else:
            file_path = saving_path[0] + '_t' + str(i) + '_0000' + saving_path[1]

        if permute is not None:
            timestep = permute_image_with_direction(timestep, permutation=permute)
        sitk.WriteImage(timestep, file_path)


def check_z_spacing(file_path):
    nifti = open_nifti_sitk(file_path)
    spacing = nifti.GetSpacing()
    if spacing[-1] < spacing[0] or spacing[-1] < spacing[1]:
        adjust_z_spacing(nifti, save_file=file_path)
        return True
    else:
        return False


def adjust_z_spacing(img, z_spacing=10, save_file=None):
    new_spacing = []
    orig_spacing = img.GetSpacing()
    for ele in orig_spacing:
        new_spacing.append(ele)
    new_spacing[-1] = z_spacing
    img.SetSpacing(new_spacing)

    if save_file is not None:
        sitk.WriteImage(img, save_file)

    return img


def stack_nifti_to_ndarray(nifti_paths):
    stacked_nifti = []
    for path in nifti_paths:
        nifti_data = open_nifti_sitk(path)
        stacked_nifti.append(nifti_data)

    stacked_data = np.stack(nifti_data, axis=-1)
    stacked_data = np.transpose(stacked_data, (3, 1, 0, 2))
    return stacked_data


def stack_nifti(nifti_paths):
    stacked_data = []
    for path in nifti_paths:
        nifti_data = open_nifti_sitk(path)
        stacked_data.append(nifti_data)

    stacked_nifti = sitk.JoinSeries(stacked_data)
    return stacked_nifti


def run_segmentation_jp(input_folder="Input", output_folder="Output", dataset="006",
                        configuration="100epochs__3d_fullres", fold="0", trainer = None,
                        post_processing = " - pp_pkl_file / mnt / ssd / sarah / data / nnUNET / nnUNET_results / Dataset008_mnm2_sax_all / nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres / crossval_results_folds_0 / postprocessing.pkl "
                                  " - np 8 - plans_json / mnt / ssd / sarah / data / nnUNET / nnUNET_results / Dataset008_mnm2_sax_all / nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres / crossval_results_folds_0 / plans.json"):

    if trainer is not None:
        nnUNet_command = ("nnUNetv2_predict -d " + dataset + " -i " + input_folder + " -o " + output_folder +
                          " -f " + fold + " -tr " + trainer +  " -c " + configuration )
    else:
        nnUNet_command = ("nnUNetv2_predict -d " + dataset + " -i " + input_folder + " -o " + output_folder +
                          " -f " + fold  + " -c " + configuration)

    subprocess.run(nnUNet_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # nnUNet_postprocess_command = ("nnUNetv2_apply_postprocessing -i " + output_folder + " -o " + output_folder
    #                               + " -pp_pkl_file /mnt/ssd/sarah/data/nnUNET/nnUNET_results/Dataset005_mnm2_fimh/nnUNetTrainer_100epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl "
    #                                 " -np 8 "
    #                                 "-plans_json /mnt/ssd/sarah/data/nnUNET/nnUNET_results/Dataset005_mnm2_fimh/nnUNetTrainer_100epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json")

    if post_processing is not None:
        nnUNet_postprocess_command = ("nnUNetv2_apply_postprocessing - i " + output_folder + " - o " + output_folder + post_processing)
    subprocess.run(nnUNet_postprocess_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_folders_for_segmentation(patient_id, root_path=None):
    if root_path is None:
        root_path = os.getcwd()
    input_subfolder_path = os.path.join(root_path, "Input_" + patient_id)
    output_subfolder_path = os.path.join(root_path, "Output_" + patient_id)

    create_directory(input_subfolder_path)
    create_directory(output_subfolder_path)

    return input_subfolder_path, output_subfolder_path


def collect_files(path, file_regex="Dataset*"):
    path_ = os.path.join(path, file_regex)
    return sorted(glob.glob(path_))


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_directory(path):
    files = collect_files(path, "*")
    for file in files:
        delete_file_or_dir(file)
    delete_file_or_dir(path)


def delete_file_or_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)


if __name__ == "__main__":
    import argparse, os, sys, glob

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a use_segmentation of patients with NN unet.')

    parser.add_argument('-data', action='store', default='')
    parser.add_argument('-saving_dir', action='store', default='/mnt/ssd/sarah/data/masks')
    parser.add_argument('-work_dir', action='store', default='/mnt/ssd/sarah/git')

    results = parser.parse_args()
    os.chdir(results.work_dir)
    sys.path.append(os.getcwd())

    origin_data = results.data
    saving_dir = results.saving_dir

    origin_data_paths = collect_files(origin_data, "*.nrrd")

    if len(origin_data) > 0:
        predict_segmentation(origin_data_paths, saving_dir)
