import argparse
import os.path

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from Detections.model.model_ConvNext import convnext_tiny
from dataloader_test import Mydatasets_bbox


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='device id(i.e. 0 or 0,1 or cpu')
parser.add_argument('--test_file_path', type=str, default=r'./data_list\test.txt')
parser.add_argument('--weight_path', type=str, default= r'E:\WZL\AI_Medicine\Fracture_Detections\Weights_file\detection_weights.pth')
parser.add_argument('--out_pred_path', type=str, default= r'E:\WZL\AI_Medicine\Fracture_Detections\out_result')
parser.add_argument('--bacth_size', type=int, default=1)

opt = parser.parse_args()
def detections():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("use device:{}".format(device))


    #cal the picture
    test_path = args.test_file_path
    with open(test_path,'r') as t:
        test_annotation = t.readlines()

    test_datasets   = Mydatasets_bbox(test_annotation)

    batch_size     = args.bacth_size



    test_dataloader = DataLoader(test_datasets,
                                  batch_size= batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=1
                                  )

    # cal the model
    weight_path = args.weight_path
    model = convnext_tiny().to(device)
    dict  = torch.load(weight_path)
    model.load_state_dict(dict)

    if not os.path.exists(args.out_pred_path):
        os.makedirs(args.out_pred_path)



    for batch_idx, (img_path,img_test,label_mask) in tqdm(enumerate(test_dataloader)):

        # cal img,label
        show_img_path = img_path[0]
        img_show_org = cv2.imread(show_img_path)
        img_show     = cv2.copyMakeBorder(img_show_org,16,16,16,16,cv2.BORDER_CONSTANT)
        img_show     = img_show[:,:,::-1]
        img = img_test.type(torch.FloatTensor)
        input_img = img
        label = label_mask
        out_bbox = torch.sigmoid(model(input_img.to(device)))

        #在原图上用热图显示_如果输出的有大于0.5的区域，则认定为骨折
        out_mask = out_bbox.detach().cpu().numpy().squeeze()
        if np.max(out_mask) >= 0.3:
            # label = label.detach().cpu().numpy().squeeze()
            heatmap = to_pil_image(out_mask, mode='F')
            overlay = heatmap.resize((288,288), resample=Image.Resampling.BICUBIC)
            cmap = cm.get_cmap('jet')
            overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
            alpha = .7
            result = (alpha * np.asarray(img_show) + (1 - alpha) * overlay).astype(np.uint8)
            save_result = result[:,:,::-1]
            cv2.imwrite(args.out_pred_path + "/{}.bmp".format(batch_idx),result)


if __name__=="__main__":
    main(opt)

    #
    # if label == 0:
    #     rep_img = rep_mask.detach().cpu().numpy().squeeze()
    #     rep_img = np.array(np.transpose(cv2.normalize(rep_img,None,0,255,cv2.NORM_MINMAX),(2,1,0)),dtype=np.uint8)
    #     org_img = str_mask.detach().cpu().numpy().squeeze()
    #     org_img = np.array(np.transpose(cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX), (2,1,0)),dtype=np.uint8)
    #     cv2.imwrite("out/0/show_rep/{}.bmp".format(batch_idx),rep_img)
    #     cv2.imwrite("out/0/show_org/{}.bmp".format(batch_idx), org_img)
    # else:
    #     rep_img = rep_mask.detach().cpu().numpy().squeeze()
    #     rep_img = np.array(np.transpose(cv2.normalize(rep_img,None,0,255,cv2.NORM_MINMAX),(2,1,0)),dtype=np.uint8)
    #     org_img = str_mask.detach().cpu().numpy().squeeze()
    #     org_img = np.array(np.transpose(cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX), (2,1,0)),dtype=np.uint8)
    #     cv2.imwrite("out/1/show_rep/{}.bmp".format(batch_idx),rep_img)
    #     cv2.imwrite("out/1/show_org/{}.bmp".format(batch_idx), org_img)
