import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--img_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Locations\save_out')
parser.add_argument("--edge_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Get_edge\out_edge')
parser.add_argument("--save_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Repair\data\test_data')

args = parser.parse_args()

def get_mask_str():
    img_file_path = args.img_path
    edge_file_path = args.edge_path
    img_new_file_path = args.save_path
    if not os.path.exists(r"{}\images".format(img_new_file_path)):
        os.makedirs(r"{}\images".format(img_new_file_path))
    if not os.path.exists(r"{}\structure".format(img_new_file_path)):
        os.makedirs(r"{}\structure".format(img_new_file_path))
    if not os.path.exists(r"{}\mask".format(img_new_file_path)):
        os.makedirs(r"{}\mask".format(img_new_file_path))
    if not os.path.exists(r"{}\edge".format(img_new_file_path)):
        os.makedirs(r"{}\edge".format(img_new_file_path))

    img_file = os.listdir(img_file_path)
    x_list = []
    y_list = []
    for m in range(5):
        for n in range(5):
            x_list.append(48+32*m)
            y_list.append((48+32*n))
    #遍历图像
    for img_name in tqdm(img_file):
        num      = img_name.split(".")[0]
        img_path = img_file_path +"/" + img_name
        img      = cv2.imread(img_path)
        edge_path = edge_file_path +"/"+num+".bmp"
        edge      = cv2.imread(edge_path)

        for i in range(25):
            img_save3 = np.copy(img)
            edge_save = np.copy(edge)
            img_mask = np.zeros([256,256,3])
            np_mask = np.ones((32, 32))*255
            np_zeros=np.zeros((32, 32))
            #在原图上拼接mask
            img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] +  32,0] = np_mask
            img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 1] = np_zeros
            img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 2] = np_zeros
            cv2.imwrite(r"{}\structure/{}_{}.bmp".format(img_new_file_path,num, i),img_save3)
            cv2.imwrite(r"{}\images/{}_{}.bmp".format(img_new_file_path,num,i),img)
            cv2.imwrite(r"{}\edge/{}_{}.bmp".format(img_new_file_path,num,i),edge_save)

            #在黑图上拼接mask
            img_mask[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32,0] = np_mask
            cv2.imwrite(r"{}\mask/{}_{}.{}.{}.bmp".
                        format(img_new_file_path,num,i,int(x_list[i]),int(y_list[i])),img_mask)