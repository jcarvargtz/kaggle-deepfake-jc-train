"""Run a training job on Cloud ML Engine for a given use case.
Usage:
    trainer.task --download <download> --n_frames <n_frames>
                --d1 <d1> --d2 <d2> --channels <channels>

                [--download_dir <download_dir>] [--dest_dir <dest_dir>]
                [--output_dir <outdir>]
                [--batch_size <batch_size>] [--hidden_units <hidden_units>]
Options:
    -h --help     Show this screen.
    --download <download> True or False
    --n_frames <n_frames> number of frames to consider
    --d1 <d1> 244
    --d2 <d2> 244
    --channels <channels> 3
    --download_dir <download_dir> 
    --dest_dir <dest_dir> 
    --output_dir <outdir>
    --batch_size <batch_size> 
    --hidden_units <hidden_units>
"""
from docopt import docopt

import trainer.Model  # Your model.py file.
import trainer.download_and_pre_process_for_virtual_machine as dppvm
import trainer.pre_process_funcs as ppf
from pathlib import Path
import subprocess
import pandas as pd
from trainer.blazeface import BlazeFace
import logging
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, LSTM, Dropout, AveragePooling2D
from keras.layers import Flatten, BatchNormalization, Convolution2D,Input
from keras. layers import TimeDistributed, Reshape, concatenate
from keras.layers import Activation, MaxPool2D
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.optimizers import Adam
import tensorflow as tf
from keras.applications import ResNet152V2, ResNet50

if __name__ == '__main__':
    arguments = docopt(__doc__)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # Assign model variables to commandline arguments
    dppvm.zip_down_dir = arguments['<download_dir>']
    if arguments['<download>'] == True:
        subprocess.call(['curl_get.sh', dppvm.zip_down_dir])
    dppvm.DEST = Path(arguments['<dest_dir>'])
    dims = (int(arguments['<d1>']),int(arguments['<d12>']))
    channels = int(arguments['<channels>'])
    n_frames = int(arguments['<n_frames>'])
    ppf.frames_per_video = n_frames
    saved_model_path = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_model_path, monitor="val_accuracy",verbose=1,save_best_only=True)
    earlystop = EarlyStopping(monitor= "val_accuracy", min_delta = 0.01, patience = 5, restore_best_weights=True)
    callbacks_list = [checkpoint, earlystop]
    optimizer = Adam()
    binloss = BinaryCrossentropy()
    acc = Accuracy()

    # # # Run the training job
    zipfiles = dppvm.extract_zips(dppvm.zip_down_dir)
    DATA = Path()
    DEST = dppvm.DEST
    logging.basicConfig(filename='extract.log', level=logging.INFO)
    zipfiles = sorted(list(DATA.glob('dfdc_train_part_*.zip')), key=lambda x: x.stem)
    # Extract the zip files
    start = int(time.time())
    with multiprocessing.Pool() as pool: # use all cores available
        pool.map(dppvm.extract_zip, zipfiles)
    # logging.info(f"Extracted all zip files in {int(time.time()) - start} seconds!")
    path_video_files = dppvm.DEST/'videos'
    path_meta = dppvm.DEST/'metadata'/'all_meta.json'
    all_meta = pd.read_json(path_meta).T
    all_meta["path"] = path_video_files + r'/' + all_meta.index
    # Train the model
    val_msk = int(len(all_meta) * 0.9)
    gener = ppf.DataGenerator(all_meta[:val_msk].index,video_path=all_meta[:val_msk].path,meta=all_meta[:val_msk])
    val = ppf.DataGenerator(all_meta[val_msk:].index,video_path=all_meta[val_msk:],meta=all_meta[val_msk:])
    model = Model.make_model()
    model.compile(optimizer= optimizer, loss = binloss, metrics = [acc])
    Model.train_and_evaluate(gener,callbacks=callbacks_list,validation_data=val,use_multiprocessing=True,workers=-1,verbose=1,epochs=500)

    # Make_predicctions
