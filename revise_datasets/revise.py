import pickle
import pandas as pd
import SimpleITK as sitk
import os
import json
from tqdm import tqdm
import random

"""
Hint : Change the paths below to your own path of Prostate datasets!!!
"""
meta_data_list = [
    [r'G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737\metadata.csv', "T2WTSEAX" ], # second elements is for filterate the targets in T2w Axial format.
    [r'G:\Datasets\Medical\datas\manifest-hjL8tlLc1556886850502670511\metadata.csv', "t2tsetra" ], # second elements is for filterate the targets in T2w Axial format.
    [r'G:\Datasets\Medical\datas\manifest-MQ0R2nDM7840353659486226295\metadata.csv', "t2tsetra"]  # second elements is for filterate the targets in T2w Axial format.
]
root_dir = r"G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737"
output_path = r"G:\Datasets\Medical\datas\revised_data"
train_test_rate = 0.8 # rate for split the dataset into training set and test sets.
"""
Change the paths above to your own path of Prostate datasets.
"""

if not os.path.exists(output_path):
    os.mkdir(output_path)

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
            # print("  Found series", serie)
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_names)
            sitk_im = reader.Execute()
            sitk_im_list.append(sitk_im)
            
    if verbose:
        print ("Found", len(sitk_im_list), "series")

    return sitk_im_list

meta_data = []
idx = 0

for meta_path, target_type in meta_data_list:
    df = pd.read_csv(meta_path)
    print(df.columns)
    df = df[df["Series Description"] == target_type]
    # df['File Location']
    print(df["File Location"])

    _tqdm = tqdm(df['File Location'])
    for path in _tqdm:
        full_path = os.path.join(os.path.dirname(meta_path), path)
        series_list = get_vol_from_dicom_dir(full_path)
        assert len(series_list) == 1
        single_serie = series_list[0]
        single_serie = sitk.RescaleIntensity(single_serie, 0.0, 255.0)
        arr = sitk.GetArrayFromImage(single_serie)
        shape = arr.shape
        _tqdm.set_postfix({"shape" : shape})
        # print(shape)
        file_name = "{:0>4}.pkl".format(idx)   
        full_output_path = os.path.join(output_path, file_name)
        
        meta_data.append([idx ,file_name, shape])
        with open(full_output_path, "wb") as f:
            pickle.dump(arr, f)
        
        idx += 1

# split train and test sets.
random.shuffle(meta_data)
train_length = int(len(meta_data) * train_test_rate)

with open("meta_data_train.json", "w") as f:
    json.dump(meta_data[:train_length], f)
    
with open("meta_data_test.json", "w") as f:
    json.dump(meta_data[train_length:], f)