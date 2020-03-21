"""
Usage: trainer.task 

Options:
    -h --help     Show this screen.

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
    try:
        arguments = docopt(__doc__)
    except:
        print("la neta no valedor")
    print("1")
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # Assign model variables to commandline arguments
    print("2")
    command = "curl 'https://storage.googleapis.com/kaggle-competitions-detached-data/16880/dfdc_train_all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1585092363&Signature=gq7%2BEEgLvt81v2Y9bLhODt5TAQh3hM4hu6kGFqKFk6lw2y%2B99trbTUhzfNu43ZfvnIYuwRSM%2FoIyIR5i32u76IOeXUvuwIsB8fFDVWEk1A%2B6rgjP2JfMgZXM2IumO26rqI%2BilI1M6hdyBndlnaFLIjYDqOXXngzfKFmOGQg6mVDu7iAgcQuvvQDwRNTxP%2Bitg9Uo2s3KGH9I%2BcbBdKCNSSgAHuUNrP5%2BFZxSqjU1Oe1wvZ1lIqUHo6xabZ4QlAFiSf1YKymD1IKaAYfNNsTjgiUfowBc8pga8qrLv6A0rWDgTQxs0qoITEStGnWohSOQ9XMvW9lHsjoM2c34l1PdqQ%3D%3D' -H 'authority: storage.googleapis.com' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36' -H 'sec-fetch-dest: document' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' -H 'sec-fetch-site: none' -H 'sec-fetch-mode: navigate' -H 'accept-language: en-US,en;q=0.9' -output download/dfdc_train_all.zip --progress-bar  --compressed"
    # command = "sudo curl 'https://www.kaggle.com/c/16880/datadownload/dfdc_train_all.zip' -H 'authority: www.kaggle.com' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36' -H 'sec-fetch-dest: document' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' -H 'sec-fetch-site: none' -H 'sec-fetch-mode: navigate' -H 'accept-language: en-US,en;q=0.9' -H 'cookie: CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL-60kvAT_P-S1DkvpjRRFQ4FkkSY0233zsS201Rn_Y2cOfKcRWHq48eN8fX_ngD9lIBMyL4gV5ArCAgoK_ufExlILwm6AKpNrK85kPkIpfZJx3mXG1dA5eR2U3Ex0ihNqM; GCLB=CIWm-NuI1aWQYA; gmNav=true; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzdWIiOiJqY2FydmFyZ3R6IiwibmJ0IjoiMjAyMC0wMi0yOFQxNTozMzoyMy4xMDU2MzYyWiIsImlhdCI6IjIwMjAtMDItMjhUMTU6MzM6MjMuMTA1NjM2MloiLCJqdGkiOiIxNWIzOTRlYS0xNTQ2LTRlNDktODEzNC01MWVjZDU1YThiZjUiLCJleHAiOiIyMDIwLTAzLTI4VDE1OjMzOjIzLjEwNTYzNjJaIiwidWlkIjoxNDIwMDUzLCJmZiI6WyJGbGV4aWJsZUdwdSIsIktlcm5lbHNJbnRlcm5ldCIsIkRhdGFFeHBsb3JlclYyIiwiRGF0YVNvdXJjZVNlbGVjdG9yVjIiLCJLZXJuZWxzVmlld2VySW5uZXJUYWJsZU9mQ29udGVudHMiLCJGb3J1bVdhdGNoRGVwcmVjYXRlZCIsIlV0aWxpdHlTY3JpcHRzIiwiTmV3S2VybmVsV2VsY29tZSIsIk1kZUltYWdlVXBsb2FkZXIiLCJLZXJuZWxzQmFja2VuZE5ld0FyY2hFbmFibGVkIiwiRGlzYWJsZUN1c3RvbVBhY2thZ2VzIiwiUGluT3JpZ2luYWxEb2NrZXJWZXJzaW9uIiwiUGhvbmVWZXJpZnlGb3JHcHUiLCJDbG91ZFNlcnZpY2VzS2VybmVsSW50ZWciLCJVc2VyU2VjcmV0c0tlcm5lbEludGVnIiwiTG9naW5NaWdyYXRlRGVwcmVjYXRlZFNzbyIsIk5hdmlnYXRpb25SZWRlc2lnbiIsIktlcm5lbHNTbmlwcGV0cyIsIktlcm5lbFdlbGNvbWVMb2FkRnJvbVVybCIsIlRwdUtlcm5lbEludGVnIl19.; ka_sessionid=fc6f3c30578b9914386d0525cef670e0; .ASPXAUTH=72B7A5EBD41090F9D39A984FC9A29F833D8E89040B9464C0F96DBB1676D09F9A82621CF46613F4907B66A8ECDEBB48F07182285822084AA46E8F53BCC57B79513C4230FF9596AEBF28D76B88D7396476E42140EC; XSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL_eb_vg8ZB9K7CcRPp7Hf7rXb8XvHG-DCsj-VG-Daf7yMxpq1ul_RwHGXML8wb8s_-Whh6laSu_nGB_V4MZNLtjJuLZJh2uLuiwI1OAtHOm3lcCFiispOdyYG_cJ2QF6xHe-ICe1CLzWCvdNJev-cA2yV2Z80ngYzfjTD2E4woPkw; intercom-session-koj6gxx6=R2FmUTIvNDlsU0FHbVFqQ2trMktXaHdER2EzL0FBSzJIcVZTSWV0R1ZxYis1NDA3Sjd2TGpXR3lXNnBpNnNGbS0tYS9pR1JLRDhkTTRsQUxFYUM2LzU4dz09--645c846c0b31f7b32eccd906029ee0b4db4c97a4; .AspNetCore.Mvc.CookieTempDataProvider=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9H8kjMjgL0VQ-ipcW2Qgu3wIn2sZHJl4ahPQGq7cVGvFppoAmlXXZZezmPwHdd686s8Svo56TGOzfg4kfztN1twMXCezVtIZyOC-6XIZ9gmN9iSQARwBuPRODHagiwMJ4' --compressed -output download/dfdc_train_all.zip --progress-bar"
    # subprocess.Popen('sudo -S' , shell=True,stdout=subprocess.PIPE)
    subprocess.call(command , shell=True)
    # subprocess.call([dppvm.curl, dppvm.zip_down_dir],shell=True)
    print("3")
    dppvm.DEST = Path("destination_directory")
    dims = (244,244)
    channels = 3
    n_frames = 30
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
