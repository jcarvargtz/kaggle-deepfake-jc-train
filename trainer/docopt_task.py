"""
Usage: trainer.task --download=<download> --n_frames=<n_frames> --d1=<d1> --d2=<d2> --channels=<channels>

Options:
    -h --help     Show this screen.
    --download <download>    True or False [default: True]
    --n_frames <n_frames>    Number of frames to consider [default: 30]
    --d1 <d1>                Height [default: 244]
    --d2 <d2>                Width [default: 244]
    --channels <channels>    N chanels [default: 3]
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


print("0.0")
if __name__ == '__main__':
    print("0.1")
    arguments = docopt(__doc__)
    print("1")
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # Assign model variables to commandline arguments
    print("2")
    if arguments['<download>'] == True:
        subprocess.call([dppvm.curl, dppvm.zip_down_dir],shell=True)
    print("3")
    dppvm.DEST = Path(arguments['<dest_dir>'])
    dims = (int(arguments['<d1>']),int(arguments['<d12>']))
    channels = int(arguments['<channels>'])
    n_frames = int(arguments['<n_frames>'])
    ppf.frames_per_video = n_frames
    print("4")
    saved_model_path = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_model_path, monitor="val_accuracy",verbose=1,save_best_only=True)
    earlystop = EarlyStopping(monitor= "val_accuracy", min_delta = 0.01, patience = 5, restore_best_weights=True)
    callbacks_list = [checkpoint, earlystop]
    optimizer = Adam()
    binloss = BinaryCrossentropy()
    acc = Accuracy()
    print("5")
    # # # Run the training job
    try:
        zipfiles = dppvm.extract_zips(dppvm.zip_down_dir)
        print("se descargo")
    except:
        print("no se descargo ni madres")
    DATA = Path("download")
    DEST = dppvm.DEST
    print("6")
    logging.basicConfig(filename='extract.log', level=logging.INFO)
    zipfiles = sorted(list(DATA.glob('dfdc_train_part_*.zip')), key=lambda x: x.stem)
    # Extract the zip files
    print("7")
    start = int(time.time())
    with multiprocessing.Pool() as pool: # use all cores available
        pool.map(dppvm.extract_zip, zipfiles)
    # logging.info(f"Extracted all zip files in {int(time.time()) - start} seconds!")       
    try:
        os.mkdir(DEST/"captures")
    except:
        pass
    failed = []
    for video in os.listdir(DEST /"videos"):
        try:
            dppvm.pre_process_video(video_file_path=DEST /"videos"/video, output_dir=DEST / "captures", dims=dims, channels=channels)
            os.remove(video)
        except:
            failed.append(video)
    path_video_files = dppvm.DEST/'videos'
    path_meta = dppvm.DEST/'metadata'/'all_meta.json'
    all_meta = pd.read_json(path_meta).T
    all_meta["path"] = path_video_files + r'/' + all_meta.index
    # Train the model
    val_msk = int(len(all_meta) * 0.9)
    gener = ppf.DataGenerator(all_meta[:val_msk].index,video_path=all_meta[:val_msk].path,meta=all_meta[:val_msk])
    val = ppf.DataGenerator(all_meta[val_msk:].index,video_path=all_meta[val_msk:],meta=all_meta[val_msk:])
    model = Model.make_model(n_frames,dims,channels)
    model.compile(optimizer= optimizer, loss = binloss, metrics = [acc])
    Model.train_and_evaluate(gener,callbacks=callbacks_list,validation_data=val,use_multiprocessing=True,workers=-1,verbose=1,epochs=500)

    # Make_predicctions
