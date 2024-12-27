import os

import cv2
from tqdm import tqdm

img_path_file = r'E:\WZL\LunWen\MEDFE-master\data\fracture_datasets\images'
img_file = os.listdir(img_path_file)
n= 0
for img_name in tqdm(img_file):
    img_path = img_path_file +"/" +img_name
    img      = cv2.imread(img_path)
    img_new  = cv2.resize(img,(256,256))
    cv2.imwrite(r"E:\WZL\LunWen\MEDFE-master\data\fracture_datasets\images2\{}.jpg".format(n),img_new)
    n+=1