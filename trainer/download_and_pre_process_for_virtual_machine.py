import os
from tqdm import tqdm
from zipfile import ZipFile
import multiprocessing
from pathlib import Path
from time import time
import logging
from typing import Union
import pandas as pd
import numpy as np
import subprocess
from trainer.pre_process_funcs import *

# Place where the download will happen
zip_down_dir = "/home/User/zips/dfdc_train_all.zip"
# Download the data
subprocess.call(['curl_get.sh', zip_down_dir])

def get_zipfiles(directory):
    list_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".zip"): 
            list_files.append(os.path.join(directory, filename))
    return list_files

# Lists all zip files
def extract_zips(directory):
    zipfiles = get_zipfiles(directory)
    zipfiles.sort()

    # Extract the files
    for zipfile in zipfiles:
        print(f'Extracting {zipfile}...')
        with ZipFile(file=zipfile) as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
                zip_file.extract(member=file)

        os.remove(zipfile)
    return zipfiles

DEST = Path('destination_directory')

def extract_zip(zipfile: Union[str, Path])->None:
    zip_no = zipfile.stem[-2:]
    with ZipFile(zipfile) as zip_file:
        for file in [Path(fn) for fn in zip_file.namelist()]:
            try:
                zip_info = zip_file.getinfo(str(file))
                if file.suffix == '.json':
                    zip_info.filename = f'{file.stem}_{zip_no}{file.suffix}'
                    dest = DEST/'metadata'
                else:
                    zip_info.filename = file.name
                    dest = DEST/'videos'
                zip_file.extract(zip_info, path=dest)
            except:
                logging.error(f'error extracing {file.stem} from {zipfile.stem}')

    # add zip file number to metadata
    meta_fn = DEST/'metadata'/f'metadata_{zip_no}.json'
    if Path(DEST/'metadata'/'all_meta.json').is_file() == False:
        df_meta = pd.read_json(meta_fn).T
        df_meta['zip_no'] = zip_no
        all_meta_df = df_meta.copy()
        df_meta.to_json(meta_fn)
        all_meta_df.to_json(DEST/'metadata'/'all_meta.json')
    else:
        df_meta = pd.read_json(meta_fn).T
        df_meta['zip_no'] = zip_no
        all_meta_df.concat([all_meta_df,df_meta],axis=0)
        df_meta.to_json(meta_fn)

    # delete zip file
    zipfile.unlink()

    logging.info(f'Finished extracting and deleted {zipfile.stem}')

        
def pre_process_video(video_file_path,output_dir,n_frames,dims,channels):
    # def Process_video(video_file_path, n_frames, dims, channels):
    size = dims[0]
    # Generate data
    temp_1 = np.empty([n_frames, *dims, channels],dtype=int)
    temp_2 = np.empty([n_frames, *dims, channels],dtype=int)
    temp_3 = np.empty([n_frames, *dims, channels],dtype=int)
    os.mkdir(output_dir)
    video_pth = Path(video_file_path)
    # Store sample
    faces = face_extractor.process_multy_faces_video(video_file_path)
    count = 0
    for i in range(n_frames):
        if len(faces[i]["faces"]) > count:
            count = len(faces[i][faces])
    for j in range(count):
        os.mkdir(output_dir + "/"+ j)
    for frame in range(len(faces)):
        if len(faces[frame]["faces"]) > 0 :
            face_1 = faces[frame]["faces"][0]
            iso_1 = isotropically_resize_image(face_1,size)
            square_1 = make_square_image(iso_1)
            temp_1[frame,] = square_1
            if len(faces[frame]["faces"]) > 1:
                face_2 = faces[frame]["faces"][1]
                iso_2 = isotropically_resize_image(face_2,size)
                square_2 = make_square_image(iso_2)
                temp_2[frame,] = square_2
                if len(faces[frame]["faces"])>2:
                    face_3 = faces[frame]["faces"][2]
                    iso_3 = isotropically_resize_image(face_3,size)
                    square_3 = make_square_image(iso_3)
                    temp_3[frame,] = square_3
                else: 
                    temp_3[frame,] = np.zeros([*dims,channels])
            else:
                temp_2[frame,] = np.zeros([*dims,channels])
                temp_3[frame,] = np.zeros([*dims,channels])
        else:
            temp_1[frame,] = np.zeros([*dims,channels])
            temp_2[frame,] = np.zeros([*dims,channels])
            temp_3[frame,] = np.zeros([*dims,channels])
    
    X = [temp_1, temp_2, temp_3]
    for x in range(len(X)):
        out = cv2.VideoWriter(output_dir + "/" + video_pth.stem / str(x)+".mp4" ,cv2.VideoWriter_fourcc(*'DIVX'), 15, dims)
        for i in range(len(X[x])):
            out.write(X[x][i])

os.mkdir(DEST/"captures")
failed = []
for video in os.listdir(DEST /"videos"):
    try:
        pre_process_video(video_file_path=DEST /"videos"/video, output_dir=DEST / "captures", dims=(224,224), channels=3)
        os.remove(video)
    except:
        failed.append(video)





