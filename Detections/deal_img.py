import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--mask_IOU_file_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Repair\data\test_data\mask')
parser.add_argument("--rep_img_file_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Repair\result\test_data')
parser.add_argument("--save_out_mask_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Detections\repair_mask')
#生成list
parser.add_argument("--mask_img_file_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Detections\repair_mask')
parser.add_argument("--img_file_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Locations\save_out')
parser.add_argument("--data_list",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Detections\data_list')


args = parser.parse_args()

def get_repair_mask():
    mask_list = os.listdir(args.mask_IOU_file_path)
    if not os.path.exists(args.save_out_mask_path):
        os.makedirs(args.save_out_mask_path)
    for mask_name in tqdm(mask_list):
        mask_path = args.mask_IOU_file_path + "/" + mask_name
        save_name = mask_name.split("b")[0]
        rep_img_name = mask_name.split(".")[0]
        rep_img_path = args.rep_img_file_path + "/" + rep_img_name + ".bmp"
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        mask[mask>10]=1
        rep_img = cv2.imread(rep_img_path,cv2.IMREAD_GRAYSCALE)
        out_mask = rep_img*mask
        cv2.imwrite(args.save_out_mask_path + "/" + "{}".format(save_name) + ";0.0;.bmp",out_mask)

def get_test_list():
    mask_list = os.listdir(args.mask_img_file_path)
    img_num   = os.listdir(args.img_file_path)
    for j in range(int(len(img_num))):
        img_name = "{}".format(str(j).zfill(6))
        img_path = args.img_file_path + "/" + img_name+".bmp"
        name_list = []
        for mask_name in mask_list:
            mask_path = args.mask_img_file_path + "/" + mask_name
            name = mask_name.split("_")[0]
            if name == img_name:
                name_list.append(mask_path)

        with open(args.data_list + "/"+"test.txt","a") as t:
            t.write("{} {}\n".format(img_path,name_list))

#
# if __name__ == "__main__":
#     get_repair_mask()