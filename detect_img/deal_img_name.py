import os
import shutil

from tqdm import tqdm

img_file_path = r'E:\WZL\AI_Medicine\Fracture_Detections\detect_img\img'
img_list = os.listdir(img_file_path)
img_save = r'E:\WZL\AI_Medicine\Fracture_Detections\detect_img\img_test'
if not os.path.exists(img_save):
    os.makedirs(img_save)
n=0
for img_name in tqdm(img_list):
    img_path = img_file_path + "/" + img_name
    new_name = str(n).zfill(6)
    new_path = img_save + "/" +"{}.bmp".format(new_name)
    shutil.move(img_path,new_path)
    n+=1