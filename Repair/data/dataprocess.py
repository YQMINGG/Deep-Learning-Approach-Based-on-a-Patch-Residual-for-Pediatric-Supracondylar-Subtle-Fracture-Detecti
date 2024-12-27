import random

import cv2
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from skimage.feature import canny



class DataProcess(torch.utils.data.Dataset):
    def __init__(self, de_root, st_root, mask_root,edge_root, opt, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.Train = False
        self.opt = opt

        if train:
            self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
            self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
            self.mask_paths = sorted(glob('{:s}/*'.format(mask_root), recursive=True))
            self.edge_paths = sorted(glob('{:s}/*'.format(edge_root),recursive=True))
            self.Train=True
        self.N_mask = len(self.mask_paths)
        print(self.N_mask)
    def __getitem__(self, index):

        de_img = Image.open(self.de_paths[index])
        st_img = Image.open(self.st_paths[index])
        edge_img = cv2.imread(self.edge_paths[index])

        mask_path = self.mask_paths[index]
        mask_x  = mask_path.split(".")[-3]
        mask_y  = mask_path.split(".")[-2]
        mask_img  = Image.open(mask_path)
        de_img = self.img_transform(de_img.convert("RGB"))
        st_img = self.img_transform(st_img.convert("RGB"))
        mask_img = self.mask_transform(mask_img.convert("RGB"))

        # #canny
        # canny_img = cv2.imread(self.de_paths[index])
        # canny_img_gray = rgb2gray(canny_img)
        # canny          = self.to_canny(canny_img_gray)*255

        #HED
        hed_img = rgb2gray(edge_img)

        self.img    = cv2.imread(self.de_paths[index],cv2.IMREAD_GRAYSCALE)
        FLY_img2 = self.get_high_f(self.img,D=10)
        FLY_img  = hed_img*255 + FLY_img2

        FLY_img = np.stack((FLY_img,)*3, axis=-1)
        FLY_img[int(mask_x):int(mask_x) + 32,int(mask_y):int(mask_y) + 32,0] = 255
        FLY_img[int(mask_x):int(mask_x) + 32, int(mask_y):int(mask_y) + 32, 1] = 0
        FLY_img[int(mask_x):int(mask_x) + 32, int(mask_y):int(mask_y) + 32, 2] = 0

        FLY_img[FLY_img>255]=255

        FLY_img = self.mask_transform(Image.fromarray(np.uint8(FLY_img)).convert('RGB'))

        return de_img, st_img, mask_img,int(mask_x),int(mask_y),FLY_img

    def __len__(self):
        return len(self.de_paths)

    def to_canny(self,img):
        return canny(img,sigma=1.5,mask=None)

    def gaussian_filter_low_f(self,fshift, D):
        # 获取索引矩阵及中心点坐标
        h, w = fshift.shape
        x, y = np.mgrid[0:h, 0:w]
        center = (int((h - 1) / 2), int((w - 1) / 2))

        # 计算中心距离矩阵
        dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

        # 计算变换矩阵
        template = 1 - np.exp(- dis_square / (2 * D ** 2))  # 高斯过滤器

        return template * fshift

    def ifft(self,fshift):
        """
        傅里叶逆变换
        """
        ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
        iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
        iimg = np.abs(iimg)  # 返回复数的模
        return iimg

    def get_high_f(self,img,D):
        """
        获取低频和高频部分图像
        """
        # 傅里叶变换
        # np.fft.fftn
        f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
        fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

        # 获取低频和高频部分
        hight_parts_fshift = self.gaussian_filter_low_f(fshift.copy(), D=D)

        high_parts_img = self.ifft(hight_parts_fshift)

        # 显示原始图像和高通滤波处理图像
        img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

        # uint8
        img_new_high = np.array(img_new_high * 255, np.uint8)
        return img_new_high


# class DataProcess(torch.utils.data.Dataset):
#     def __init__(self, de_root, opt, train=True):
#         super(DataProcess, self).__init__()
#         self.img_transform = transforms.Compose([
#             transforms.Resize(opt.fineSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#         ])
#         # mask should not normalize, is just have 0 or 1
#         self.mask_transform = transforms.Compose([
#             transforms.Resize(opt.fineSize),
#             transforms.ToTensor()
#         ])
#         self.Train = False
#         self.opt = opt
#         # self.x  = random.randint(40,215)
#         # self.y  = random.randint(40,215)
#
#         if train:
#             self.de_paths = sorted(glob('{:s}/*'.format(de_root), recursive=True))
#             self.Train=True
#
#     def __getitem__(self, index):
#
#         de_img = Image.open(self.de_paths[index])
#         str_img = de_img.copy()
#         mask_img = Image.new("RGB",(255,255),(0,0,0))
#         mask = Image.new("RGB",(32,32),(90,60,150))
#         x = random.randint(32,191)
#         y = random.randint(32,191)
#
#         mask_img.paste(mask,(x,y))
#         str_img.paste(mask,(x,y))
#         mask_img = self.img_transform(mask_img.convert("RGB"))
#         str_img = self.img_transform(str_img.convert("RGB"))
#         de_img = self.img_transform(de_img.convert('RGB'))
#
#         return de_img,str_img,mask_img,x,y
#
#     def __len__(self):
#         return len(self.de_paths)
