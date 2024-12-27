import cv2
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self,file_txt="train.txt"):
        self.file_txt = file_txt
        if self.file_txt == "train.txt":
            with open(self.file_txt,"r") as t:
                self.infor_train_line = t.readlines()
        else:
            with open(self.file_txt,"r") as t:
                self.infor_test_line = t.readlines()
    def __len__(self):
        if self.file_txt =="train.txt":
            return len(self.infor_train_line)
        else:
            return len(self.infor_test_line)
    def __getitem__(self, item):
        if self.file_txt == "train.txt":
            img_lines = self.infor_train_line[item]
            edge = cv2.imread(img_lines.split(";")[1].split()[0], cv2.IMREAD_GRAYSCALE)
            edge = edge[np.newaxis, :, :]  # Add one channel at first (CHW).
            edge[edge < 127.5] = 0.0
            edge[edge >= 127.5] = 1.0
            edge = edge.astype(np.float32)

            image = np.array(cv2.imread(img_lines.split(";")[0]),dtype=np.float32)

            image = image - np.array((104.00698793,
                                      116.66876762,
                                      122.67891434))
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
            image = image.astype(np.float32)

            return image,edge
        else:
            image = np.array(cv2.imread(self.infor_test_line[item].split()[0]),dtype=np.float32)
            img_name = self.infor_test_line[item].split("/")[-1].split(".")[0]

            image = image - np.array((104.00698793,
                                      116.66876762,
                                      122.67891434))
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
            image = image.astype(np.float32)

            return image,img_name






