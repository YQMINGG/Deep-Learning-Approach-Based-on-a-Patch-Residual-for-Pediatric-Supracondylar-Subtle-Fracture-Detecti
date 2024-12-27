import os

import cv2
from tqdm import tqdm


def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


# 处理输入图片，后缀改为000000格式
def deal_img(img_path,save_path):
    img_lists = os.listdir(img_path)
    n = 0
    for name in tqdm(img_lists):
        img_one_path = img_path + "/" +name
        img = cv2.imread(img_one_path)
        cv2.imwrite(save_path + "/" + "{}.jpg".format(str(n).zfill(6)),img)
        n+=1

if __name__ == "__main__":
    img_path = r'D:\wzl\nijie\1'
    save_path =r'E:\WZL\AI_Medicine\Fracture_Detections\input_img'
    deal_img(img_path,save_path)