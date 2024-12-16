import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


# Example usage
img_dir = 'images'
label_dir = 'labels'

# todo 自定义dataset
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(416, 416)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Initialize targets tensor
        targets = torch.zeros((len(lines), 6))
        '''
            每个样本的标签是一个大小为 [N, 6] 的张量，其中 N 是图像中目标的数量，6 表示每个目标的类别索引、中心坐标 x 和 y、宽度和高度，
            以及一个置信度（在这个示例中，我们将置信度设置为 1.0，因为 YOLO 格式中的置信度通常是 1）。
        '''

        # Parse label data
        for i, line in enumerate(lines):
            data = line.strip().split()
            class_idx = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])

            # Convert YOLO format to normalized coordinates
            x_center *= self.img_size[0]
            y_center *= self.img_size[1]
            width *= self.img_size[0]
            height *= self.img_size[1]

            # Store target data
            targets[i] = torch.tensor([class_idx, x_center, y_center, width, height, 1.0])

        return img, targets

# todo 实例化dataset
dataset = YOLODataset(img_dir, label_dir)

# todo 创建dataloader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, targets in data_loader:

    '''
        开始训练数据
    '''
    print("Batch of images:", images.shape)
    print("Batch of targets:", targets.shape)
    break
