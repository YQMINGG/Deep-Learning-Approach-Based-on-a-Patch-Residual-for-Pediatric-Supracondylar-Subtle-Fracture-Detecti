import argparse

parser = argparse.ArgumentParser(description="test")
#yolo裁剪参数
parser.add_argument("--input_img_path",default=r"E:\WZL\AI_Medicine\Fracture_Detections\input_img")
parser.add_argument("--save_img_path",default=r"./Locations/save_out")

#HED提取边缘参数
parser.add_argument("--img_path",default=r"./Locations/save_out")
parser.add_argument("--edge_test_list",default="./Get_edge/test.txt",)
parser.add_argument("--test_out",default=r"./Get_edge/out_edge")
parser.add_argument("--weight_path",default=r"./Weights_file\hed_weight.pth")
parser.add_argument("--batch_size",default=1,type=int)

#生成修复数据集参数
parser.add_argument("--img_data_path",default=r"./Locations/save_out")
parser.add_argument("--edge_data_path",default=r'./Get_edge\out_edge')
parser.add_argument("--save_data_path",default=r'./Repair\data\test_data')

#修复骨折参数
parser.add_argument("--weight_repair_path",default=r'./Weights_file')
parser.add_argument("--test_data",default="test_data")
parser.add_argument("--edge_txt",default=r"./Get_edge\test.txt")

#生成检测数据集
parser.add_argument("--mask_IOU_file_path",default=r'./Repair\data\test_data\mask')
parser.add_argument("--rep_img_file_path",default=r'./Repair\result\test_data')
parser.add_argument("--save_out_mask_path",default=r'./Detections\repair_mask')
#生成检测list
parser.add_argument("--mask_img_file_path",default=r'./Detections\repair_mask')
parser.add_argument("--img_file_path",default=r"./Locations/save_out")
parser.add_argument("--data_list",default=r'./Detections\data_list')

#检测骨折参数
parser.add_argument('--device', default='cuda:0', help='device id(i.e. 0 or 0,1 or cpu')
parser.add_argument('--test_file_path', type=str, default=r'./Detections/data_list\test.txt')
parser.add_argument('--weight_detect_path', type=str, default=r'./Weights_file\detection_weights.pth')
parser.add_argument('--out_pred_path', type=str, default= r'./out_result')
parser.add_argument('--bacth_size', type=int, default=1)


args = parser.parse_args()