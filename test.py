

import os
import random
import shutil
import time

import cv2

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.nn import functional as F
from pathlib import Path

from tqdm import tqdm

import torch

import torch
import numpy as np



class GenCrack(nn.Module):

    def __init__(self,G_pt,img_hw=[256,256]):

        super().__init__()
        from torchvision import transforms as transforms

        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.netG = torch.load(G_pt).to(self.device)
        transform_list=[transforms.ToTensor(),
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize(img_hw),
                                           transforms.Normalize([0.415],[0.25])]

        if img_hw[0] > 0  and img_hw[1]>0:
            transform_list.insert(2, transforms.Resize(img_hw))

        self.transform_list = transforms.Compose(transform_list)

        cv_transform_list=[transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize(img_hw),
                                           transforms.Normalize([0.5],[0.5])]

        if img_hw[0] > 0  and img_hw[1]>0:
            cv_transform_list.insert(2, transforms.Resize(img_hw))

        self.cv_transform = transforms.Compose(cv_transform_list)



    def forward(self,img0s):
        imgs_org_size = []
        if isinstance(img0s[0],str):
            imgs=[]
            for img_path in img0s:
                img=Image.open(img_path)
                imgs_org_size.append([img.size[1],img.size[0]])
                imgs.append(self.transform_list(img))
            img0s=torch.stack(imgs)
        elif isinstance(img0s,list):
            imgs=[]
            for img in img0s:
                imgs_org_size.append(img.shape[:2])
                imgs.append(self.cv_transform(np.array(img)))
            img0s=torch.stack(imgs)
        elif isinstance(img0s,torch.Tensor):
            for img in img0s:
                imgs_org_size.append(img.shape[-2:])


        with torch.no_grad():
            G_masks,G_imgs=self.netG(img0s.to(self.device))

        resized_g_imgs=[]
        resized_g_masks=[]
        org_imgs=[]
        for img,mask,org_hw in zip(G_imgs,G_masks,imgs_org_size):
            original_height, original_width=org_hw
            resized_img = F.interpolate(img.unsqueeze(0),size=(original_height, original_width), mode='bilinear')
            resized_mask = F.interpolate(mask.unsqueeze(0),size=(original_height, original_width), mode='bilinear')

            resized_img=((resized_img.cpu().squeeze().numpy()+1)/2*255).astype(np.uint8)
            resized_mask=((resized_mask.cpu().squeeze().numpy()+1)/2*255).astype(np.uint8)

            resized_g_imgs.append(resized_img)
            resized_g_masks.append(resized_mask)

            org_imgs.append(((img0s.cpu().squeeze().numpy()+1)/2*255).astype(np.uint8))


        return resized_g_imgs,resized_g_masks,org_imgs


def gen_crack_mask(img_dir,g_pt,save_dir,rand_rename=False,repeat_times=1,bs=1,gen_suffix='_gen'):

    save_img_dir=os.path.join(save_dir,'img')
    save_mask_dir=os.path.join(save_dir,'mask')
    os.makedirs(save_img_dir,exist_ok=True)
    os.makedirs(save_mask_dir,exist_ok=True)

    gen_crack_mask=GenCrack(g_pt)

    if isinstance(img_dir,str):
        img_suffix = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(img_suffix)]
    elif isinstance(img_dir,list):
        img_suffix = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_files = [f for f in img_dir if f.lower().endswith(img_suffix)]
    elif isinstance(img_dir, torch.Tensor):
        image_files = img_dir
    else:
        print(f'error')

    for _ in range(repeat_times):
        for i in tqdm(range(0, len(image_files), bs)):
            batch_paths = image_files[i:i + bs]
            batch_datas=[cv2.imread(f,0) for f in batch_paths]


            cracked_imgs, masks,img_orgs = gen_crack_mask.forward(batch_datas)
            for j in range(len(cracked_imgs)):
                f1=batch_datas[j]
                cracked_img=cracked_imgs[j]
                mask=masks[j]
                if rand_rename:
                    time_str=time.strftime('%H%H%S',time.localtime())
                    new_name_stem=str(random.randint(0,999999)).zfill(6)+time_str
                else:
                    new_name_stem=os.path.splitext(os.path.basename(batch_paths[j]))[0]+gen_suffix

                save_img_path=os.path.join(save_img_dir,f'{new_name_stem}.png')
                save_mask_path=os.path.join(save_mask_dir,f'{new_name_stem}_mask.png')
                cv2.imwrite(save_img_path, cracked_img)
                cv2.imwrite(save_mask_path, mask)


                cv2.imwrite(os.path.join(save_img_dir,f'{new_name_stem}_org.png'), img_orgs[j])


if __name__ == '__main__':

    g_pt='gan_pt/latest_net_G.pth'
    img_dir='test_imgs_dir'
    save_dir='gen_imgs'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


    gen_crack_mask(img_dir,g_pt,save_dir,rand_rename=False,)




