import os

import PIL.Image
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Mydatasets_bbox(Dataset):
    def __init__(self,annotation_line):
        self.annotation_line = annotation_line
    def __len__(self):
        return len(self.annotation_line)

    def __getitem__(self, item):

        data_info = self.annotation_line[item]
        img_path  = data_info.split("[")[0].strip()
        img_org = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        #周围padding16
        img = cv2.copyMakeBorder(img_org,16,16,16,16,cv2.BORDER_CONSTANT)
        mask_info = data_info.split("[")[1].split("]")[0]
        mask_list = mask_info.split(",")


        mask_0    = mask_list[0].strip("'")
        mask_1 = mask_list[1].strip("'").strip().strip("'")
        mask_2 = mask_list[2].strip("'").strip().strip("'")
        mask_3 = mask_list[3].strip("'").strip().strip("'")
        mask_4 = mask_list[4].strip("'").strip().strip("'")
        mask_5 = mask_list[5].strip("'").strip().strip("'")
        mask_6 = mask_list[6].strip("'").strip().strip("'")
        mask_7 = mask_list[7].strip("'").strip().strip("'")
        mask_8 = mask_list[8].strip("'").strip().strip("'")
        mask_9 = mask_list[9].strip("'").strip().strip("'")
        mask_10 = mask_list[10].strip("'").strip().strip("'")
        mask_11 = mask_list[11].strip("'").strip().strip("'")
        mask_12 = mask_list[12].strip("'").strip().strip("'")
        mask_13 = mask_list[13].strip("'").strip().strip("'")
        mask_14 = mask_list[14].strip("'").strip().strip("'")
        mask_15 = mask_list[15].strip("'").strip().strip("'")
        mask_16 = mask_list[16].strip("'").strip().strip("'")
        mask_17 = mask_list[17].strip("'").strip().strip("'")
        mask_18 = mask_list[18].strip("'").strip().strip("'")
        mask_19 = mask_list[19].strip("'").strip().strip("'")
        mask_20 = mask_list[20].strip("'").strip().strip("'")
        mask_21 = mask_list[21].strip("'").strip().strip("'")
        mask_22 = mask_list[22].strip("'").strip().strip("'")
        mask_23 = mask_list[23].strip("'").strip().strip("'")
        mask_24 = mask_list[24].strip("'").strip().strip("'")


        mask_pro_0 = mask_0.split(";")[1] #0
        mask_pro_1 = mask_1.split(";")[1] #1
        mask_pro_2 = mask_2.split(";")[1] #10
        mask_pro_3 = mask_3.split(";")[1] #11
        mask_pro_4 = mask_4.split(";")[1] #12
        mask_pro_5 = mask_5.split(";")[1] #13
        mask_pro_6 = mask_6.split(";")[1] #14
        mask_pro_7 = mask_7.split(";")[1] #15
        mask_pro_8 = mask_8.split(";")[1] #16
        mask_pro_9 = mask_9.split(";")[1] #17
        mask_pro_10 = mask_10.split(";")[1] #18
        mask_pro_11 = mask_11.split(";")[1] #19
        mask_pro_12 = mask_12.split(";")[1] #2
        mask_pro_13 = mask_13.split(";")[1] #20
        mask_pro_14 = mask_14.split(";")[1] #21
        mask_pro_15 = mask_15.split(";")[1] #22
        mask_pro_16 = mask_16.split(";")[1] #23
        mask_pro_17 = mask_17.split(";")[1] #24
        mask_pro_18 = mask_18.split(";")[1] #3
        mask_pro_19 = mask_19.split(";")[1] #4
        mask_pro_20 = mask_20.split(";")[1] #5
        mask_pro_21 = mask_21.split(";")[1] #6
        mask_pro_22 = mask_22.split(";")[1] #7
        mask_pro_23 = mask_23.split(";")[1] #8
        mask_pro_24 = mask_24.split(";")[1] #9


        #label拼接成5x5的格式
        label_mask = np.zeros((9,9),dtype=np.float32)

        label_mask[2,2] = float(mask_pro_0)
        label_mask[3,2] = float(mask_pro_20)
        label_mask[4,2] = float(mask_pro_2)
        label_mask[5,2] = float(mask_pro_7)
        label_mask[6,2] = float(mask_pro_13)

        label_mask[2,3] = float(mask_pro_1)
        label_mask[3,3] = float(mask_pro_21)
        label_mask[4,3] = float(mask_pro_3)
        label_mask[5,3] = float(mask_pro_8)
        label_mask[6,3] = float(mask_pro_14)

        label_mask[2,4] = float(mask_pro_12)
        label_mask[3,4] = float(mask_pro_22)
        label_mask[4,4] = float(mask_pro_4)
        label_mask[5,4] = float(mask_pro_9)
        label_mask[6,4] = float(mask_pro_15)

        label_mask[2,5] = float(mask_pro_18)
        label_mask[3,5] = float(mask_pro_23)
        label_mask[4,5] = float(mask_pro_5)
        label_mask[5,5] = float(mask_pro_10)
        label_mask[6,5] = float(mask_pro_16)

        label_mask[2,6] = float(mask_pro_19)
        label_mask[3,6] = float(mask_pro_24)
        label_mask[4,6] = float(mask_pro_6)
        label_mask[5,6] = float(mask_pro_11)
        label_mask[6,6] = float(mask_pro_17)

        #读取mask
        mask0 = cv2.imread(mask_0,-1)
        mask0 = cv2.copyMakeBorder(mask0,16,16,16,16,cv2.BORDER_CONSTANT)

        mask1 = cv2.imread(mask_1,-1)
        mask1 = cv2.copyMakeBorder(mask1, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask2 = cv2.imread(mask_2,-1)
        mask2 = cv2.copyMakeBorder(mask2, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask3 = cv2.imread(mask_3,-1)
        mask3 = cv2.copyMakeBorder(mask3, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask4 = cv2.imread(mask_4,-1)
        mask4 = cv2.copyMakeBorder(mask4, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask5= cv2.imread(mask_5,-1)
        mask5 = cv2.copyMakeBorder(mask5, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask6 = cv2.imread(mask_6,-1)
        mask6 = cv2.copyMakeBorder(mask6, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask7 = cv2.imread(mask_7,-1)
        mask7 = cv2.copyMakeBorder(mask7, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask8 = cv2.imread(mask_8,-1)
        mask8 = cv2.copyMakeBorder(mask8, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask9 = cv2.imread(mask_9,-1)
        mask9 = cv2.copyMakeBorder(mask9, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask10 = cv2.imread(mask_10,-1)
        mask10 = cv2.copyMakeBorder(mask10, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask11 = cv2.imread(mask_11,-1)
        mask11 = cv2.copyMakeBorder(mask11, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask12 = cv2.imread(mask_12,-1)
        mask12 = cv2.copyMakeBorder(mask12, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask13 = cv2.imread(mask_13,-1)
        mask13 = cv2.copyMakeBorder(mask13, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask14 = cv2.imread(mask_14,-1)
        mask14= cv2.copyMakeBorder(mask14, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask15 = cv2.imread(mask_15,-1)
        mask15 = cv2.copyMakeBorder(mask15, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask16 = cv2.imread(mask_16,-1)
        mask16 = cv2.copyMakeBorder(mask16, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask17 = cv2.imread(mask_17,-1)
        mask17 = cv2.copyMakeBorder(mask17, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask18 = cv2.imread(mask_18,-1)
        mask18 = cv2.copyMakeBorder(mask18, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask19 = cv2.imread(mask_19,-1)
        mask19 = cv2.copyMakeBorder(mask19, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask20 = cv2.imread(mask_20,-1)
        mask20 = cv2.copyMakeBorder(mask20, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask21 = cv2.imread(mask_21,-1)
        mask21 = cv2.copyMakeBorder(mask21, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask22 = cv2.imread(mask_22,-1)
        mask22 = cv2.copyMakeBorder(mask22, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask23 = cv2.imread(mask_23,-1)
        mask23 = cv2.copyMakeBorder(mask23, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        mask24 = cv2.imread(mask_24,-1)
        mask24 = cv2.copyMakeBorder(mask24, 16, 16, 16, 16, cv2.BORDER_CONSTANT)

        #拼接img和mask
        # train_img = np.stack([img,mask0,mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,mask9,mask10,
        #                       mask11,mask12,mask13,mask14,mask15,mask16,mask17,mask18,mask19,mask20,mask21,mask22,
        #                       mask23,mask24],axis=0)
        #归一化
        train_img = np.stack([img/255,mask0/255,mask1/255,mask2/255,mask3/255,mask4/255,mask5/255,mask6/255,mask7/255,mask8/255,mask9/255,mask10/255,
                              mask11/255,mask12/255,mask13/255,mask14/255,mask15/255,mask16/255,mask17/255,mask18/255,mask19/255,mask20/255,mask21/255,mask22/255,
                              mask23/255,mask24/255],axis=0)
        label_mask_out = np.expand_dims(label_mask,axis=0)


        return img_path,train_img,label_mask_out

'''
最初5x5代码
'''
# class Mydatasets_bbox(Dataset):
#     def __init__(self,annotation_line):
#         self.annotation_line = annotation_line
#     def __len__(self):
#         return len(self.annotation_line)
#
#     def __getitem__(self, item):
#
#         data_info = self.annotation_line[item]
#         img_path  = data_info.split("[")[0].strip()
#         img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
#         mask_info = data_info.split("[")[1].split("]")[0]
#         mask_list = mask_info.split(",")
#
#         mask_0    = mask_list[0].strip("'")
#         mask_1 = mask_list[1].strip("'").strip().strip("'")
#         mask_2 = mask_list[2].strip("'").strip().strip("'")
#         mask_3 = mask_list[3].strip("'").strip().strip("'")
#         mask_4 = mask_list[4].strip("'").strip().strip("'")
#         mask_5 = mask_list[5].strip("'").strip().strip("'")
#         mask_6 = mask_list[6].strip("'").strip().strip("'")
#         mask_7 = mask_list[7].strip("'").strip().strip("'")
#         mask_8 = mask_list[8].strip("'").strip().strip("'")
#         mask_9 = mask_list[9].strip("'").strip().strip("'")
#         mask_10 = mask_list[10].strip("'").strip().strip("'")
#         mask_11 = mask_list[11].strip("'").strip().strip("'")
#         mask_12 = mask_list[12].strip("'").strip().strip("'")
#         mask_13 = mask_list[13].strip("'").strip().strip("'")
#         mask_14 = mask_list[14].strip("'").strip().strip("'")
#         mask_15 = mask_list[15].strip("'").strip().strip("'")
#         mask_16 = mask_list[16].strip("'").strip().strip("'")
#         mask_17 = mask_list[17].strip("'").strip().strip("'")
#         mask_18 = mask_list[18].strip("'").strip().strip("'")
#         mask_19 = mask_list[19].strip("'").strip().strip("'")
#         mask_20 = mask_list[20].strip("'").strip().strip("'")
#         mask_21 = mask_list[21].strip("'").strip().strip("'")
#         mask_22 = mask_list[22].strip("'").strip().strip("'")
#         mask_23 = mask_list[23].strip("'").strip().strip("'")
#         mask_24 = mask_list[24].strip("'").strip().strip("'")
#
#
#         mask_pro_0 = mask_0.split(";")[1] #0
#         mask_pro_1 = mask_1.split(";")[1] #1
#         mask_pro_2 = mask_2.split(";")[1] #10
#         mask_pro_3 = mask_3.split(";")[1] #11
#         mask_pro_4 = mask_4.split(";")[1] #12
#         mask_pro_5 = mask_5.split(";")[1] #13
#         mask_pro_6 = mask_6.split(";")[1] #14
#         mask_pro_7 = mask_7.split(";")[1] #15
#         mask_pro_8 = mask_8.split(";")[1] #16
#         mask_pro_9 = mask_9.split(";")[1] #17
#         mask_pro_10 = mask_10.split(";")[1] #18
#         mask_pro_11 = mask_11.split(";")[1] #19
#         mask_pro_12 = mask_12.split(";")[1] #2
#         mask_pro_13 = mask_13.split(";")[1] #20
#         mask_pro_14 = mask_14.split(";")[1] #21
#         mask_pro_15 = mask_15.split(";")[1] #22
#         mask_pro_16 = mask_16.split(";")[1] #23
#         mask_pro_17 = mask_17.split(";")[1] #24
#         mask_pro_18 = mask_18.split(";")[1] #3
#         mask_pro_19 = mask_19.split(";")[1] #4
#         mask_pro_20 = mask_20.split(";")[1] #5
#         mask_pro_21 = mask_21.split(";")[1] #6
#         mask_pro_22 = mask_22.split(";")[1] #7
#         mask_pro_23 = mask_23.split(";")[1] #8
#         mask_pro_24 = mask_24.split(";")[1] #9
#
#
#         #label拼接成5x5的格式
#         label_mask = np.zeros((5,5),dtype=np.float32)
#
#         label_mask[0,0] = float(mask_pro_0)
#         label_mask[1,0] = float(mask_pro_20)
#         label_mask[2,0] = float(mask_pro_2)
#         label_mask[3,0] = float(mask_pro_7)
#         label_mask[4,0] = float(mask_pro_13)
#
#         label_mask[0,1] = float(mask_pro_1)
#         label_mask[1,1] = float(mask_pro_21)
#         label_mask[2,1] = float(mask_pro_3)
#         label_mask[3,1] = float(mask_pro_8)
#         label_mask[4,1] = float(mask_pro_14)
#
#         label_mask[0,2] = float(mask_pro_12)
#         label_mask[1,2] = float(mask_pro_22)
#         label_mask[2,2] = float(mask_pro_4)
#         label_mask[3,2] = float(mask_pro_9)
#         label_mask[4,2] = float(mask_pro_15)
#
#         label_mask[0,3] = float(mask_pro_18)
#         label_mask[1,3] = float(mask_pro_23)
#         label_mask[2,3] = float(mask_pro_5)
#         label_mask[3,3] = float(mask_pro_10)
#         label_mask[4,3] = float(mask_pro_16)
#
#         label_mask[0,4] = float(mask_pro_19)
#         label_mask[1,4] = float(mask_pro_24)
#         label_mask[2,4] = float(mask_pro_6)
#         label_mask[3,4] = float(mask_pro_11)
#         label_mask[4,4] = float(mask_pro_17)
#
#         #读取mask
#         mask0 = cv2.imread(mask_0,-1)
#         mask1 = cv2.imread(mask_1,-1)
#         mask2 = cv2.imread(mask_2,-1)
#         mask3 = cv2.imread(mask_3,-1)
#         mask4 = cv2.imread(mask_4,-1)
#         mask5= cv2.imread(mask_5,-1)
#         mask6 = cv2.imread(mask_6,-1)
#         mask7 = cv2.imread(mask_7,-1)
#         mask8 = cv2.imread(mask_8,-1)
#         mask9 = cv2.imread(mask_9,-1)
#         mask10 = cv2.imread(mask_10,-1)
#         mask11 = cv2.imread(mask_11,-1)
#         mask12 = cv2.imread(mask_12,-1)
#         mask13 = cv2.imread(mask_13,-1)
#         mask14 = cv2.imread(mask_14,-1)
#         mask15 = cv2.imread(mask_15,-1)
#         mask16 = cv2.imread(mask_16,-1)
#         mask17 = cv2.imread(mask_17,-1)
#         mask18 = cv2.imread(mask_18,-1)
#         mask19 = cv2.imread(mask_19,-1)
#         mask20 = cv2.imread(mask_20,-1)
#         mask21 = cv2.imread(mask_21,-1)
#         mask22 = cv2.imread(mask_22,-1)
#         mask23 = cv2.imread(mask_23,-1)
#         mask24 = cv2.imread(mask_24,-1)
#
#         #拼接img和mask
#         # train_img = np.stack([img,mask0,mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,mask9,mask10,
#         #                       mask11,mask12,mask13,mask14,mask15,mask16,mask17,mask18,mask19,mask20,mask21,mask22,
#         #                       mask23,mask24],axis=0)
#         #归一化
#         train_img = np.stack([img/255,mask0/255,mask1/255,mask2/255,mask3/255,mask4/255,mask5/255,mask6/255,mask7/255,mask8/255,mask9/255,mask10/255,
#                               mask11/255,mask12/255,mask13/255,mask14/255,mask15/255,mask16/255,mask17/255,mask18/255,mask19/255,mask20/255,mask21/255,mask22/255,
#                               mask23/255,mask24/255],axis=0)
#         label_mask_out = np.expand_dims(label_mask,axis=0)
#
#
#         return img_path,train_img,label_mask_out



