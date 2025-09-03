import logging
import shutil
from collections import OrderedDict
import json
import numpy as np

import matplotlib.pyplot as plt
import nibabel as nib


from nnunet.inference.predict import predict_from_folder
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.preprocessing.preprocessing import resample_and_normalize

from nnunetv2.paths import  nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def predict_segmentation_nnU_Net(model_path, nifti_list):
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, model_path),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    predicted_segmentations = predictor.predict_from_files(nifti_list, None, save_probabilities=False, overwrite=True,
                                                           num_processes_preprocessing=2,
                                                           num_processes_segmentation_export=2,
                                                           folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    return predicted_segmentations

def main(args):
    import os, datetime
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    from src.utils.Utils_io import ensure_dir

    if args.cfg:
        import json
        cfg = args.cfg_reg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))
        EXPERIMENTS_ROOT = 'exp/'
        if config.get('EXP_PATH', None) is None:
            config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestamp)
        EXP_PATH = config['EXP_PATH']
        logging.debug(config)
    else:
        logging.error('no config given, please select one from the templates in exp/examples')

    ensure_dir(EXP_PATH)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these parameters should encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/sds/exp/fimh_base')
    parser.add_argument('-data', action='store', default=None)  # path to any dataset that can be processed

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    main(results)
