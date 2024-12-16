import os
from pathlib import Path

import cv2
import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms

def img_paths2label_paths(img_dir,label_foldername,label_suffix='.txt'):
    '''
        假设图像文件夹和对应的标签文件夹位于同一个父目录下，并且它们的文件名，只是路径不同。
        --father_folder
            img_floder:
                1.jpg
                2.jpg
            label_foldername:
                1.txt
                2.txt

        return imgs_path,labels_path
    '''
    # 定义图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    # 设置你的图像目录
    img_dir = Path(img_dir)
    label_dir=img_dir.parent/label_foldername

    imgs_path=[]
    labels_path=[]
    # 遍历目录及其子目录
    for image_path in img_dir.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            label_path=label_dir/f'{image_path.stem}{label_suffix}'
            if label_path.exists():
                imgs_path.append(str(image_path))
                labels_path.append(str(label_path))
    return imgs_path,labels_path



class unalignedCacheDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images path from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images path from '/path/to/data/trainB'


        self.isA_label=opt.isA_label
        self.isB_label=opt.isB_label

        if self.isA_label:
            # self.A_label_paths=img_paths2label_paths(self.A_paths,opt.phase + 'A',opt.phase + 'A_label')
            self.A_paths,self.A_mask_paths=img_paths2label_paths(self.dir_A,opt.phase + 'A_masks',label_suffix='.png')
            # from build_model.cache_labels import CacheLabels,GetSigMaskFromCache_label
            # self.GetSigMaskFromCache_label=GetSigMaskFromCache_label
            # self.A_paths,self.A_labels, self.A_shapes, self.A_segments=CacheLabels(self.A_paths,self.A_label_paths,retun_cache=False)

            # self.dir_A_label = os.path.join(opt.dataroot, opt.phase + 'A_label')  # create a path '/path/to/data/trainA'
            #
            # if os.path.exists(self.dir_A_label):
            #     self.A_label_paths= sorted(make_dataset(self.dir_A_label, opt.max_dataset_size))


        if self.isB_label:
            self.B_paths,self.B_mask_paths=img_paths2label_paths(self.dir_B,opt.phase + 'B_masks',label_suffix='_mask.png')

            # self.B_label_paths =img_paths2label_paths(self.B_paths, opt.phase + 'B',opt.phase + 'B_label')
            # from build_model.cache_labels import CacheLabels,GetSigMaskFromCache_label
            # self.GetSigMaskFromCache_label=GetSigMaskFromCache_label
            # self.B_paths,self.B_labels, self.B_shapes, self.B_segments=CacheLabels(self.B_paths,self.B_label_paths,retun_cache=False)
            # self.dir_B_label = os.path.join(opt.dataroot, opt.phase + 'B_label')  # create a path '/path/to/data/trainB'
            #
            # if os.path.exists(self.dir_B_label):
            #     self.B_label_paths= sorted(make_dataset(self.dir_B_label, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1),colojet=False)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1),colojet=False)

        self.mask_transform=get_transform(self.opt, grayscale=True)
        # self.mask_transform=transforms.Compose([transforms.Normalize((0.5,),(0.5,)),])



        # self.transform_A =transforms.Compose([transforms.Resize((self.opt.load_size,self.opt.load_size)),transforms.ToTensor()])
        # self.transform_B =transforms.Compose([transforms.Resize((self.opt.load_size,self.opt.load_size)),transforms.ToTensor()])
        # # self.transform_B =transforms.Resize(self.opt.load_size)


        # if self.isA_label or self.isB_label:
        #     from get_mask_label import load_label_fromfile
        #     self.load_label=load_label_fromfile()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        gen_data=True
        try_times=0
        while gen_data and try_times<20:
            try:
                index_A=index % self.A_size
                A_path = self.A_paths[index_A]  # make sure index is within then range
                if self.opt.serial_batches:   # make sure index is within then range
                    index_B = index % self.B_size
                else:   # randomize the index for domain B to avoid fixed pairs.
                    index_B = random.randint(0, self.B_size - 1)
                B_path = self.B_paths[index_B]
                A_img = Image.open(A_path).convert('RGB')
                B_img = Image.open(B_path).convert('RGB')


                # apply image transformation
                A = self.transform_A(A_img)
                B = self.transform_B(B_img)



                if self.isA_label:

                    currt_A_mask=Image.open(self.A_mask_paths[index_A])
                    currt_A_mask=self.mask_transform(currt_A_mask)

                    A_label =[currt_A_mask,currt_A_mask]

                if self.isB_label:

                    currt_B_mask=Image.open(self.B_mask_paths[index_B],)
                    currt_B_mask=self.mask_transform(currt_B_mask)

                    B_label =[currt_B_mask,currt_B_mask]


                gen_data=False
                try_times=0


                if self.isA_label:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label }

                elif  self.isB_label:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'B_label': B_label }

                elif self.isA_label and self.isB_label:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label,'B_label': B_label }


                else:
                    return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

            except Exception as e:
                print(f'error：{e}')
                print(f'wrong file A：{A_path}')
                print(f'wrong file B：{B_path}')

                index = random.randint(0, max(self.A_size, self.B_size))
                try_times+=1

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
