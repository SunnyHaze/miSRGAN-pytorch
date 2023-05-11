import argparse
import SimpleITK as sitk
import os
import numpy as np
import cv2
import scipy
import scipy.misc
import json
import torch
from tqdm import tqdm

def get_vol_from_dicom_dir(dicom_dir, verbose=False):
    """
    goes through all dicom files in dicom_dir, finds the series (as it could 
    be multiple), and get the volume for each series
    """

    if verbose:
        print("Processing folder: ", dicom_dir)
    
    reader = sitk.ImageSeriesReader()    
    series_found = reader.GetGDCMSeriesIDs(dicom_dir)

    series_found_count = 0
    sitk_im_list = []
    for serie in series_found:
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, serie)
        if len(dicom_names):
            print("  Found series", serie)
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            sitk_im = reader.Execute()
            sitk_im_list.append(sitk_im)
            
    if verbose:
        print ("Found", len(sitk_im_list), "series")

    return sitk_im_list

def get_a_person_vol(person_dir):
    subdir = os.listdir(person_dir)[0]
    print(subdir)
    person_dir = os.path.join(person_dir, subdir)
    record_list = os.listdir(person_dir)
    print(record_list)
    path = []  # path, shape
    shape = []
    volumes = []
    
    for record in record_list:
        record_path = os.path.join(person_dir, record)
        series_list = get_vol_from_dicom_dir(record_path)
        assert len(series_list) == 1 , "series should only have one record in list."
        single_serie = series_list[0]
        single_serie = sitk.RescaleIntensity(single_serie, 0.0, 255.0)
        arr = sitk.GetArrayFromImage(single_serie)
        # rescale

        
        path.append(record_path)
        shape.append(arr.shape)
        volumes.append(arr)
    return path, shape, volumes

def parse_all_person_dicom(data_path_list, output_path):
    index = 0
    # max_vol = -10000
    # min_vol = 10000
    metadata = []
    for data_path in data_path_list:
        person_list = os.listdir(data_path)
        for person in tqdm(person_list):        
            full_person_path = os.path.join(data_path, person)
            if os.path.isdir(full_person_path):
                paths, shapes, volumes = get_a_person_vol(full_person_path)
                for path, shape, volume in zip(paths, shapes, volumes):
                    print(volume.shape)
                    # local_max, local_min = np.max(volume), np.min(volume)
                    # print(local_max, local_min)
                    # max_vol = local_max if max_vol < local_max else max_vol
                    # min_vol = local_min if min_vol > local_min else min_vol
                    full_output_path = os.path.join(output_path, "{:0>4d}.pt".format(index))
                    save_path = full_output_path
                    torch.save(volume, save_path)
                    metadata.append([index, shape, full_output_path, path])
                    index += 1
                break

    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    # print(f"Done! MAX:{max_vol}, MIN:{min_vol} .", )      # exit(0)
        


    # with open("meta_data.json", "w+") as f:
    
    
parse_all_person_dicom(
    data_path_list=[
        r"G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737\PROSTATE-DIAGNOSIS",
        r"G:\Datasets\Medical\datas\manifest-hjL8tlLc1556886850502670511\PROSTATEx",
        r"G:\Datasets\Medical\datas\manifest-MQ0R2nDM7840353659486226295\PROSTATEx"
    ], 
    output_path= r"G:\Datasets\Medical\outputs_"
)
    
    