import pickle
import pandas as pd
import SimpleITK as sitk
import os
import json
from tqdm import tqdm

"""
Change the paths below to your own path of Prostate datasets.
"""
root_dir = r"G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737"
output_path = r"G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737\revised_data"
"""
Change the paths above to your own path of Prostate datasets.
"""


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

df = pd.read_csv("./metadata.csv")
print(df.columns)
df = df[df["Series Description"] == "T2WTSEAX"]
# df['File Location']
print(df["File Location"])

meta_data = []
for idx, path in enumerate(tqdm(df['File Location'])):
    full_path = os.path.join(root_dir, path)
    series_list = get_vol_from_dicom_dir(full_path)
    assert len(series_list) == 1
    single_serie = series_list[0]
    single_serie = sitk.RescaleIntensity(single_serie, 0.0, 255.0)
    arr = sitk.GetArrayFromImage(single_serie)
    shape = arr.shape
    print(shape)
    
    file_name = "{:0>4}.pkl".format(idx)   
    full_output_path = os.path.join(output_path, file_name)
    
    meta_data.append([idx ,file_name, shape, full_output_path])
    
    with open(full_output_path, "wb") as f:
        pickle.dump(arr, f)

with open("meta_data.json", "w") as f:
    json.dump(meta_data, f)

    
    
    
    