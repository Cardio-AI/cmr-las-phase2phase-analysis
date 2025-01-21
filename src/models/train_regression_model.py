import logging

def train_fold(config, dataset_json, in_memory=False):
    # make sure necessary config params are given, otherwise replace with default
    import tensorflow as tf
    import numpy as np
    tf.get_logger().setLevel('FATAL')
    tf.random.set_seed(config.get('SEED', 42))
    np.random.seed(config.get('SEED', 42))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as _:
            print(_)

    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))
    # ------------------------------------------ import helpers
    # from tensorflow.python.client import device_lib
    # import external libs
    from time import time
    import logging, os

    # local imports
    from src.utils.Utils_io import ConsoleAndFileLogger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
    from src.models.PhaseRegModels import PhaseRegressionModel

    # make all config params known to the local namespace
    locals().update(config)

    # overwrite the experiment names and paths, so that each cv gets an own sub-folder
    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}/f{}'.format(EXPERIMENT, FOLD)

    EXPERIMENTS_ROOT = 'exp/'
    EXP_PATH = config.get('EXP_PATH')
    FOLD_PATH = os.path.join(EXP_PATH, 'f{}'.format(FOLD))
    MODEL_PATH = os.path.join(FOLD_PATH, 'model', )
    TENSORBOARD_PATH = os.path.join(FOLD_PATH, 'tensorboard_logs')
    CONFIG_PATH = os.path.join(FOLD_PATH, 'config')
    SEGMENTATION_EXP = config.get("SEGMENTATION_EXP", None)
    # PRETRAINED_SEG = SEGMENTATION_EXP is not None
    # if PRETRAINED_SEG:
    #     SEGMENTATION_MODEL = os.path.join(SEGMENTATION_EXP, 'model.json')
    #     SEGMENTATION_WEIGHTS = os.path.join(SEGMENTATION_EXP, f'f{FOLD}', 'model', 'model.h5')

    ensure_dir(MODEL_PATH)
    ensure_dir(TENSORBOARD_PATH)
    ensure_dir(CONFIG_PATH)

    DATA_PATH_LAX = config.get('DATA_PATH_LAX')
    DF_FOLDS = config.get('DF_FOLDS', None)
    DF_META = config.get('DF_META', None)
    EPOCHS = config.get('EPOCHS', 100)

    ConsoleAndFileLogger(path=FOLD_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))

    suffix = dataset_json["suffix"]
    file_ending = dataset_json["file_ending"]

    # get k-fold data from DATA_ROOT and subdirectories
    x_train_lax, y_train_lax, x_val_lax, y_val_lax = get_trainings_files(data_path=DATA_PATH_LAX,
                                                                         suffix=suffix,
                                                                         ftype=file_ending,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                         fold=FOLD)

    logging.info('LAX train CMR: {}, LAX train masks: {}'.format(len(x_train_lax), len(y_train_lax)))
    logging.info('LAX val CMR: {}, LAX val masks: {}'.format(len(x_val_lax), len(y_val_lax)))

    t0 = time()
    debug = 0  # make sure single threaded

    # Create the batchgenerators
    config['BATCHSIZE'] = 1
    if debug:
        config['SHUFFLE'] = False
        config['WORKERS'] = 1
        config['BATCHSIZE'] = 1
    batch_generator = PhaseRegressionGenerator_v2(x_train_lax, x_train_lax, config=config, dataset_json=dataset_json, in_memory=in_memory)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['HIST_MATCHING'] = False
    val_config['AUGMENT_TEMP'] = False
    validation_generator = PhaseRegressionGenerator_v2(x_val_lax, x_val_lax, config=val_config, dataset_json=dataset_json, in_memory=in_memory)

    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D

    if debug:
        path_ = 'data/interim/{}_focus_mse/'.format('tof_volume')
        ensure_dir(path_)
        i = 0
        for b in batch_generator:
            x, y = b
            x = x[0]
            for p in x:
                patient = os.path.basename(batch_generator.IMAGES[i]).split('_')[0]
                fig = show_2D_or_3D(p[0, ..., 0:1])
                plt.savefig('{}{}_{}.png'.format(path_, i, patient))
                plt.close()
                i = i + 1
        i = 0
        for b in validation_generator:
            x, y = b
            x = x[0]
            for p in x:
                patient = os.path.basename(validation_generator.IMAGES[i]).split('_')[0]
                fig = show_2D_or_3D(p[0, ..., 0:1])
                plt.savefig('{}v{}_{}.png'.format(path_, i, patient))
                plt.close()
                i = i + 1

    # get model
    model = PhaseRegressionModel(config=config).get_model()

    # write the model summary to a txt file
    with open(os.path.join(EXP_PATH, 'model_summary.txt'), 'w') as fh:
        model.summary(line_length=140, print_fn=lambda x: fh.write(x + '\n')) # Pass the file handle in as a lambda function to make it callable

    # plot the model structure as graph
    tf.keras.utils.plot_model(
        model,
        show_dtype=False,
        show_shapes=True,
        to_file=os.path.join(FOLD_PATH, 'model.png'),
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    # training
    initial_epoch = 0
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=get_callbacks(config, batch_generator, validation_generator),
        initial_epoch=initial_epoch,
        verbose=2 if in_memory else 1)  # 1 for local, 2 for cluster, assuming that cluster is always in_memory

    # free as much memory as possible
    import gc
    tf.keras.backend.clear_session()
    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    del batch_generator
    del validation_generator
    del model
    gc.collect()
    return config


def main(args=None, in_memory=False, seg_exp_path=None):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime, gc
    sys.path.append(os.getcwd())

    EXPERIMENTS_ROOT = 'exp/'

    if args.cfg:
        import json
        cfg = args.cfg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        # Define new paths, make sure that:
        # 1. we don't overwrite a previous config
        # 2. cluster based trainings are compatible with saving locally (cluster/local)
        # we don't need to initialise this config, as it should already have the correct format,
        # The fold configs will be saved within each fold run
        # add a timestamp and a slurm jobid to each project to make repeated experiments unique
        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))
        if args.jobid != None: timestamp += "_{}".format(args.jobid)
        if seg_exp_path is not None: config['SEGMENTATION_EXP'] = seg_exp_path
        if config.get('EXP_PATH', None) is None: config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestamp)

        if args.data:  # if we specified a different data path (training from workspace or node temporal disk)
            config['DATA_PATH_LAX'] = os.path.join(args.data, "lax/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DF_META'] = os.path.join(args.data, "gt_phases.csv")

        logging.debug(config)
    else:
        logging.error('no config given, please select one from the templates in exp/examples')

    if args.data_json:
        with open(args.data_json, encoding='utf-8') as data_file:
            data_json = json.load(data_file)

    from src.models.predict_phase_reg_model import predict
    for f in config.get('FOLDS', [0]):
        logging.info('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        data_json_ = data_json.copy()
        data_json_['FOLD'] = f
        cfg = train_fold(config_, data_json_, in_memory=in_memory)
        predict(cfg)
        gc.collect()
        logging.info('train fold: {} finished'.format(f))
        # evaluate dice with 2D slices but phase generator
    from src.models.evaluate_phase_reg import evaluate_supervised
    from src.models.predict_phase_reg_model import predict_phase_from_deformable
    try:
        evaluate_supervised(config.get('EXP_PATH'))
    except Exception as e:
        logging.error('{} evaluate failed with: {}'.format(config.get('EXPERIMENT'), e))
    try:
        predict_phase_from_deformable(config.get('EXP_PATH'),
                                      create_figures=True,
                                      norm_thresh=50,
                                      dir_axis=0,
                                      roll_by_gt=False,
                                      mask_channels=None,
                                      ct_calculation=None
                                      )
    except Exception as e:
        logging.error('{} predict phase failed with: {}'.format(config.get('EXPERIMENT'), e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these three parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg_reg', action='store',
                        default=None)  # path to a cfg file, such as the example cfgs in data/cfgs
    parser.add_argument('-data_json', action='store',
                        default=None)  # path to a cfg file, such as the example cfgs in data/cfgs
    parser.add_argument('-data', action='store', default=None)  # path to any dataset that can be processed
    parser.add_argument('-inmemory', action='store', default=False)  # enable in memory pre-processing on the cluster
    parser.add_argument('-jobid', action='store', default=None)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    try:
        import distutils.util

        in_memory = distutils.util.strtobool(results.inmemory)
        if in_memory:
            print('running in-memory={}, watch for memory overflow!'.format(in_memory))
    except Exception as e:
        print(e)
        in_memory = False  # fallback
    results.cfg = results.cfg_reg
    main(results, in_memory=in_memory)
    exit()
