import os
import shutil
from datetime import datetime
from glob import glob
import cv2
import numpy as np
import torch
from PIL import Image
from imgviz import rgb2gray
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from Detections.dataloader_test import Mydatasets_bbox
from Detections.model.model_ConvNext_large import convnext_large
from Get_edge.dataloader import Dataset
from Get_edge.model2 import HED
from Locations.yolo_cut import YOLO
from Repair.models.MEDFE import MEDFE
from Repair.options.test_options import TestOptions
from args import args
from utils import del_file
import warnings



class Detect():
    def __init__(self,args):
        super(Detect,self).__init__()
        #HED提取边缘
        self.hed = HED(args.device)
        #修复骨折
        self.opt = TestOptions()

    def forward(self):
        # #yolo裁剪
        self.get_location()
        #生成hed—txt
        self.get_list()
        #hed边缘检测
        self.test()
        #生成修复数据集
        self.get_mask_str()
        #骨折修复
        self.repair()
        #生成检测数据集
        self.get_repair_mask()
        #生成检测txt
        self.get_test_list()
        #检测骨折
        self.detections()

    #yolo裁剪肘部
    def get_location(self):

        print("------开始骨折检测------")
        yolo = YOLO()
        save_file_path = args.save_img_path
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)

        img_detect_path = args.input_img_path
        img_list = os.listdir(img_detect_path)
        for img_name in img_list:
            img_path = img_detect_path + "/" + img_name

            img = Image.open(img_path)
            _, _, outbbox, image_shape = yolo.detect_image(image=img)
            if outbbox is not None:
                # 获得h,w,bbox
                left = int(outbbox[1])  # xmin
                up = int(outbbox[0])  # ymin
                right = int(outbbox[3])  # xmax
                below = int(outbbox[2])  # ymax
                # 中心点
                x = int((right - left) / 2 + left)
                y = int((below - up) / 2 + up)
                # 最短边
                lower_line = min(int((right - left)), int((below - up)))
                long_line = max(int((right - left)), int((below - up)))

                need_line = int((long_line + lower_line) / 2)
                xmin = int(x - need_line / 2)
                ymin = int(y - need_line / 2)
                xmax = int(x + need_line / 2)
                ymax = int(y + need_line / 2)

                img_cut = img.copy()
                crop_img = img_cut.crop([xmin, ymin, xmax, ymax])
                save_img = crop_img.resize((256, 256))
                save_img.save(save_file_path + "/" + "{}".format(img_name))

    #HED提取边缘
    def test(self):
        # 读取数据
        test_dataset = Dataset(file_txt=args.edge_test_list)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=4, drop_last=True, shuffle=True)

        # 加载预训练权重
        net = torch.nn.DataParallel(HED(args.device))
        net.to(args.device)
        net.load_state_dict(torch.load(args.weight_path))

        # 设置保存文件
        if not os.path.exists(args.test_out):
            os.makedirs(args.test_out)
        for batch_index, (images, img_name) in enumerate(test_loader):
            images = images.to(args.device)
            out_list = net(images)
            result_img = out_list[-1].detach().cpu().numpy()[0, 0]
            cv2.imwrite(args.test_out + "/{}.bmp".format(str(img_name[0])), result_img * 255)

    def get_list(self):
        img_file_path = args.img_path
        img_file = os.listdir(img_file_path)
        for img_name in img_file:
            img_path = img_file_path + "/" + img_name
            with open(args.edge_test_list, "a") as t:
                t.write("{}\n".format(img_path))
        t.close()

    #生成修复数据集
    def get_mask_str(self):
        img_file_path = args.img_data_path
        edge_file_path = args.edge_data_path
        img_new_file_path = args.save_data_path
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
                x_list.append(48 + 32 * m)
                y_list.append((48 + 32 * n))
        # 遍历图像
        for img_name in (img_file):
            num = img_name.split(".")[0]
            img_path = img_file_path + "/" + img_name
            img = cv2.imread(img_path)
            edge_path = edge_file_path + "/" + num + ".bmp"
            edge = cv2.imread(edge_path)

            for i in range(25):
                img_save3 = np.copy(img)
                edge_save = np.copy(edge)
                img_mask = np.zeros([256, 256, 3])
                np_mask = np.ones((32, 32)) * 255
                np_zeros = np.zeros((32, 32))
                # 在原图上拼接mask
                img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 0] = np_mask
                img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 1] = np_zeros
                img_save3[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 2] = np_zeros
                cv2.imwrite(r"{}\structure/{}_{}.bmp".format(img_new_file_path, num, i), img_save3)
                cv2.imwrite(r"{}\images/{}_{}.bmp".format(img_new_file_path, num, i), img)
                cv2.imwrite(r"{}\edge/{}_{}.bmp".format(img_new_file_path, num, i), edge_save)

                # 在黑图上拼接mask
                img_mask[x_list[i]:x_list[i] + 32, y_list[i]:y_list[i] + 32, 0] = np_mask
                cv2.imwrite(r"{}\mask/{}_{}.{}.{}.bmp".
                            format(img_new_file_path, num, i, int(x_list[i]), int(y_list[i])), img_mask)

    def repair(self):
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

        opt = self.opt.parse()
        model = MEDFE(opt)
        weight_path = args.weight_repair_path
        weight_num = r'5'
        model.netEN.module.load_state_dict(
            torch.load(os.path.join(weight_path + "/" + "{}_net1_EN.pth".format(weight_num))))
        model.netDE.module.load_state_dict(
            torch.load(os.path.join(weight_path + "/" + "{}_net1_DE.pth".format(weight_num))))
        model.netMEDFE.module.load_state_dict(
            torch.load(os.path.join(weight_path + "/" + "{}_net1_MEDFE.pth".format(weight_num))))
        model.netmaskEN.module.load_state_dict(
            torch.load(os.path.join(weight_path + "/" + "{}_net1_maskEN.pth".format(weight_num))))
        out_show_file = args.test_data
        results_dir = r'./Repair/result/{}'.format(out_show_file)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        mask_paths = glob('{:s}/*'.format(opt.mask_root))
        de_paths = glob('{:s}/*'.format(opt.de_root))
        st_path = glob('{:s}/*'.format(opt.st_root))
        edge_path = glob('{:s}/*'.format(opt.edge_root))
        image_len = len(de_paths)
        for i in (range(image_len)):

            path_m = mask_paths[i]
            path_d = de_paths[i]
            img_name = path_d.split(".")[1].split("/")[-1][7:]

            path_s = st_path[i]
            x = int(path_m.split(".")[-3])
            y = int(path_m.split(".")[-2])

            # HED
            edge_img = cv2.imread(edge_path[i])
            edge_gray_img = rgb2gray(edge_img)

            img = cv2.imread(path_d, cv2.IMREAD_GRAYSCALE)
            FLY_img2 = self.get_high_f(img,D=10)
            FLY_img = edge_gray_img * 255 + FLY_img2

            FLY_img = np.stack((FLY_img,) * 3, axis=-1)
            FLY_img[int(x):int(x) + 32, int(y):int(y) + 32, 0] = 255
            FLY_img[int(x):int(x) + 32, int(y):int(y) + 32, 1] = 0
            FLY_img[int(x):int(x) + 32, int(y):int(y) + 32, 2] = 0

            FLY_img[FLY_img > 255] = 255

            mask = Image.open(path_m).convert("RGB")
            detail = Image.open(path_d).convert("RGB")
            structure = Image.open(path_s).convert("RGB")

            fly_img = mask_transform(Image.fromarray(np.uint8(FLY_img))).unsqueeze(0)
            mask = mask_transform(mask)
            detail = img_transform(detail)
            structure = img_transform(structure)
            mask = torch.unsqueeze(mask, 0)
            detail = torch.unsqueeze(detail, 0)
            structure = torch.unsqueeze(structure, 0)

            with torch.no_grad():
                model.set_input(detail, structure, mask, x, y, None, FLY_img=fly_img)
                model.forward()
                fake_out = model.fake_out

            output = fake_out.detach().cpu().numpy()[0].transpose((1, 2, 0))
            output = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(results_dir + "/" + "{}.bmp".format(img_name), output.astype(np.uint8))

    # 傅里叶算子
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
        hight_parts_fshift = self.gaussian_filter_low_f(fshift.copy(),D=D)

        high_parts_img = self.ifft(hight_parts_fshift)

        # 显示原始图像和高通滤波处理图像
        img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

        # uint8
        img_new_high = np.array(img_new_high * 255, np.uint8)
        return img_new_high

    #生成检测数据集
    def get_repair_mask(self):
        mask_list = os.listdir(args.mask_IOU_file_path)
        if not os.path.exists(args.save_out_mask_path):
            os.makedirs(args.save_out_mask_path)
        for mask_name in mask_list:
            mask_path = args.mask_IOU_file_path + "/" + mask_name
            save_name = mask_name.split("b")[0]
            rep_img_name = mask_name.split(".")[0]
            rep_img_path = args.rep_img_file_path + "/" + rep_img_name + ".bmp"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask > 10] = 1
            rep_img = cv2.imread(rep_img_path, cv2.IMREAD_GRAYSCALE)
            out_mask = rep_img * mask
            cv2.imwrite(args.save_out_mask_path + "/" + "{}".format(save_name) + ";0.0;.bmp", out_mask)
    #生成检测txt
    def get_test_list(self):
        mask_list = os.listdir(args.mask_img_file_path)
        img_num = os.listdir(args.img_file_path)
        for name in img_num:
            img_name = name.split(".")[0]
            img_path = args.img_file_path + "/" + name
            name_list = []
            for mask_name in mask_list:
                mask_path = args.mask_img_file_path + "/" + mask_name
                name = mask_name.split("_")[0]
                if name == img_name:
                    name_list.append(mask_path)
            if len(name_list) == 0:
                continue
            with open(args.data_list + "/" + "test.txt", "a") as t:
                t.write("{} {}\n".format(img_path, name_list))

    #检测骨折
    def detections(self):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print("use device:{}".format(device))

        # cal the picture
        test_path = args.test_file_path
        with open(test_path, 'r') as t:
            test_annotation = t.readlines()

        test_datasets = Mydatasets_bbox(test_annotation)

        batch_size = args.bacth_size

        test_dataloader = DataLoader(test_datasets,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1
                                     )
        # cal the model
        weight_path = args.weight_detect_path
        model = convnext_large().to(args.device)
        dict = torch.load(weight_path)
        model.load_state_dict(dict)
        if not os.path.exists(args.out_pred_path):
            os.makedirs(args.out_pred_path)

        for batch_idx, (img_path, img_test, label_mask) in (enumerate(test_dataloader)):
            # cal img,label
            show_img_path = img_path[0]
            img_name      = show_img_path.split("/")[-1].split(".")[0]
            img_show_org = cv2.imread(show_img_path)
            img_show = cv2.copyMakeBorder(img_show_org, 16, 16, 16, 16, cv2.BORDER_CONSTANT)
            img_show = img_show[:, :, ::-1]
            img = img_test.type(torch.FloatTensor)
            input_img = img
            out_bbox = torch.sigmoid(model(input_img.to(device)))
            # 在原图上用热图显示_如果输出的有大于0.3的区域，则认定为骨折
            out_mask = out_bbox.detach().cpu().numpy().squeeze()
            if np.max(out_mask) >= 0.001:
                # label = label.detach().cpu().numpy().squeeze()
                # out_mask[out_mask>= 0.01] = 1
                heatmap = to_pil_image(out_mask, mode='F')
                overlay = heatmap.resize((288, 288), resample=Image.Resampling.BICUBIC)
                cmap = cm.get_cmap('jet')
                overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
                alpha = .7
                result = (alpha * np.asarray(img_show) + (1 - alpha) * overlay).astype(np.uint8)
                save_result = result[:, :, ::-1]
                cv2.imwrite(args.out_pred_path + "/{}.bmp".format(img_name), result)
            else:
                print("-----图像 {} 无骨折-----".format(img_name))
        print("------检测完成------")

if __name__ == "__main__":
    print("------开始检测------- ")
    warnings.filterwarnings("ignore")
    start_time = datetime.now()
    begin = Detect(args)
    begin.forward()
    end_time = datetime.now()
    print("------检测用时：{}s".format(end_time-start_time))
    #释放内存
    print("开始释放内存")
    os.remove(args.edge_txt)
    os.remove(args.test_file_path)
    shutil.rmtree(args.test_out)
    shutil.rmtree(args.rep_img_file_path)
    shutil.rmtree(args.save_data_path)
    shutil.rmtree(args.save_out_mask_path)
    shutil.rmtree(args.save_img_path)
    print("-----内存释放完毕------")
    #
    # print("清除输入和输出")
    # del_