import argparse
import os

import cv2
import numpy as np
import torch

from args import args
from dataloader import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model2 import HED


device = torch.device("cpu" if args.cpu else "cuda")

def test():
    #读取数据
    test_dataset = Dataset(file_txt=args.test_list)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              num_workers=4, drop_last=True, shuffle=True)

    #加载预训练权重
    net = torch.nn.DataParallel(HED(device))
    net.to(device)
    net.load_state_dict(torch.load(args.weight_path))

    #设置保存文件
    if not os.path.exists(args.test_out):
        os.makedirs(args.test_out)
    for batch_index, (images,img_name) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        out_list = net(images)
        result_img = out_list[-1].detach().cpu().numpy()[0,0]
        cv2.imwrite(args.test_out + "/{}.bmp".format(str(img_name[0])),result_img*255)

def get_list():
    img_file_path =args.img_path
    img_file = os.listdir(img_file_path)
    for img_name in img_file:
        img_path = img_file_path + "/" + img_name
        with open("test.txt", "a") as t:
            t.write("{}\n".format(img_path))
    t.close()


