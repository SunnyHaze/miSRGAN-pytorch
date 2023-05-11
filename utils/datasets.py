from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import albumentations as albu
import json
import numpy as np
import pickle
import cv2
import torch
class sr_dataset(Dataset):
    def __init__(self, meta_data_list, output_size = (224, 224), if_return_index=False ) -> None:
        self.data_lenth = []
        self.data_path = []
        
        prefix_temp = -1 # start from 0
        self.prefix_sum = []
        
        self.meta_data_list = meta_data_list
        for meta_data_dir in meta_data_list:
            with open(meta_data_dir, "r") as f:
                self.meta_data = json.load(f)   # list of [index:int, file_name:str ,shape(n, h, w) , file_path:str ]
                for index, file_name , (n, h, w), file_path in self.meta_data:      
                    self.data_lenth.append(n)
                    self.data_path.append(file_path)
                    
                    prefix_temp += (n-2)
                    self.prefix_sum.append(prefix_temp)
        
        self.length = prefix_temp
        self.prefix_sum = np.array(self.prefix_sum)
        
        self.if_return_index = if_return_index
        # self.transform = T.Compose([
        #     T.Normalize(mean=0, std = 1),
        #     T.ToTensor()
        # ])
        # print(self.data_path)
        # print(self.data_lenth)
        
        # print(self.prefix_sum)
        # print(prefix_temp)
    
    def _transform_image(self, img):
        img = cv2.resize(img, (224, 224))
        img = (img / 255 * 2) - 1 # Norm to -1~1
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        return img
    
    def _parse_index(self, index):
        check = np.sum((self.prefix_sum < index) * 1)
        pre_index = 0 if check == 0 else self.prefix_sum[check - 1] + 1
        in_img_index = index - pre_index # which slice in a serie
        pkl_index = check
        return pkl_index, in_img_index
        
    
    def __getitem__(self, index):
        pkl_index, in_img_index = self._parse_index(index)
        # load image
        with open(self.data_path[pkl_index], "rb" ) as f:
            image = pickle.load(f)
            prev_img = image[in_img_index]
            target_img = image[in_img_index + 1]
            next_img = image[in_img_index + 2]
            
            prev_img = self._transform_image(prev_img)
            target_img = self._transform_image(target_img)
            next_img = self._transform_image(next_img)
        if self.if_return_index:
            return prev_img, target_img, next_img, pkl_index, in_img_index
        
        return prev_img, target_img, next_img
    
    def __len__(self):
        return self.length
        


"""
Code below for testing
"""
if __name__ == "__main__":
    meta_data_dir = r"G:\Datasets\Medical\datas\manifest-gJIZVVFt6412408718812805737\meta_data.json"
    train_data = sr_dataset(meta_data_list= [meta_data_dir], if_return_index=True)
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    print(len(train_data))
    for i, (a, b, c, idx1, idx2) in enumerate(train_data):
        if i % 500 == 0:
            print(i)
        #     plt.subplot(1,3,1)
        #     plt.imshow(a[0], norm=colors.Normalize(0,255))
        #     plt.title(f"{idx1}-{idx2}")
        #     plt.subplot(1,3,2)
        #     plt.imshow(b[0], norm=colors.Normalize(0,255))
        #     plt.title(f"{idx1}-{idx2+1}")
        #     plt.subplot(1,3,3)
        #     plt.title(f"{idx1}-{idx2+2}")
        #     plt.imshow(c[0], norm=colors.Normalize(0,255))
        #     plt.show()
