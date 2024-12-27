import argparse
import shutil

import cv2
from skimage.color import rgb2gray
from skimage.feature import canny

from Detections.deal_img import get_repair_mask, get_test_list

from options.test_options import TestOptions
from models.models import create_model
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms

#canny算子
def to_canny(img):
    return canny(img, sigma=2, mask=None)
#傅里叶算子
def gaussian_filter_low_f( fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = 1 - np.exp(- dis_square / (2 * D ** 2))  # 高斯过滤器

    return template * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg


def get_high_f(img, D):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = gaussian_filter_low_f(fshift.copy(), D=D)

    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
            np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_high

parser = argparse.ArgumentParser()
parser.add_argument("--weight_path",default=r'E:\WZL\AI_Medicine\Fracture_Detections\Weights_file')
parser.add_argument("--test_out",default="test_data")
parser.add_argument("--edge_txt",default=r"E:\WZL\AI_Medicine\Fracture_Detections\Get_edge\test.txt")
args = parser.parse_args()


def repair():
    # os.remove(args.edge_txt)

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    opt = TestOptions().parse()
    model = create_model(opt)
    weight_path = args.weight_path
    weight_num = r'5'
    model.netEN.module.load_state_dict(torch.load(os.path.join(weight_path +"/"+ "{}_net1_EN.pth".format(weight_num))))
    model.netDE.module.load_state_dict(torch.load(os.path.join(weight_path +"/"+  "{}_net1_DE.pth".format(weight_num))))
    model.netMEDFE.module.load_state_dict(torch.load(os.path.join(weight_path +"/"+  "{}_net1_MEDFE.pth".format(weight_num))))
    model.netmaskEN.module.load_state_dict(torch.load(os.path.join(weight_path + "/"+ "{}_net1_maskEN.pth".format(weight_num))))
    out_show_file = args.test_out
    results_dir = r'./result/{}'.format(out_show_file)
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)

    mask_paths = glob('{:s}/*'.format(opt.mask_root))
    de_paths = glob('{:s}/*'.format(opt.de_root))
    st_path = glob('{:s}/*'.format(opt.st_root))
    edge_path = glob('{:s}/*'.format(opt.edge_root))
    image_len = len(de_paths )
    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_m = mask_paths[i]
        path_d = de_paths[i]
        img_name = path_d.split(".")[1].split("/")[-1][7:]


        path_s = st_path[i]
        x      = int(path_m.split(".")[-3])
        y      = int(path_m.split(".")[-2])

        #HED
        edge_img = cv2.imread(edge_path[i])
        edge_gray_img = rgb2gray(edge_img)


        img    = cv2.imread(path_d,cv2.IMREAD_GRAYSCALE)
        FLY_img2 = get_high_f(img,D=10)
        FLY_img  = edge_gray_img*255 + FLY_img2

        FLY_img = np.stack((FLY_img,)*3, axis=-1)
        FLY_img[int(x):int(x) + 32,int(y):int(y) + 32,0] = 255
        FLY_img[int(x):int(x) + 32, int(y):int(y) + 32, 1] = 0
        FLY_img[int(x):int(x) + 32, int(y):int(y) + 32, 2] = 0

        FLY_img[FLY_img>255]=255

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")

        fly_img = mask_transform(Image.fromarray(np.uint8(FLY_img))).unsqueeze(0)
        mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)


        with torch.no_grad():
            model.set_input(detail, structure, mask,x,y,None,FLY_img=fly_img)
            model.forward()
            fake_out = model.fake_out

        output = fake_out.detach().cpu().numpy()[0].transpose((1, 2, 0))
        output = cv2.normalize(output,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        cv2.imwrite(results_dir + "/" + "{}.bmp".format(img_name),output.astype(np.uint8))


