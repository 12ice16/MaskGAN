

import os
import random

import numpy as np
import torch
import itertools

import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import functional  as F
from torch.cuda.amp import autocast,GradScaler

from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from thop import profile

from torchinfo import summary


class ColorBrightnessConsistencyLoss(nn.Module):
    def __init__(self, lambda_color_loss=1.0, lambda_brightness_loss=1.0):
        super(ColorBrightnessConsistencyLoss, self).__init__()
        self.lambda_color_loss = lambda_color_loss
        self.lambda_brightness_loss = lambda_brightness_loss

        self.color_loss = nn.L1Loss()
        self.brightness_loss = nn.L1Loss()

    def forward(self, input_img, output_img):
        is_grayscale_input = self.is_grayscale(input_img)
        is_grayscale_output = self.is_grayscale(output_img)

        total_loss = 0.0
        if self.lambda_color_loss:
            if is_grayscale_input != is_grayscale_output:
                raise ValueError("img must be gray")

            if not is_grayscale_input:
                color_loss = self.color_loss(input_img, output_img)
                total_loss += color_loss*self.lambda_color_loss

        if self.lambda_brightness_loss:
            input_brightness = self.calculate_brightness(input_img)
            output_brightness = self.calculate_brightness(output_img)
            brightness_loss = self.brightness_loss(input_brightness, output_brightness)
            total_loss += brightness_loss*self.lambda_brightness_loss

        return total_loss

    def is_grayscale(self, img):
        return img.size(1) == 1

    def calculate_brightness(self, img):
        if self.is_grayscale(img):
            return torch.mean(img)
        else:
            grayscale_img = 0.2989 * img[:, 0, ...] + 0.5870 * img[:, 1, ...] + 0.1140 * img[:, 2, ...]
            return torch.mean(grayscale_img)



class SSIMLoss(nn.Module):
    def __init__(self, C1=4e-4, C2=3.6e-3, alpha=1.0, beta=1.0, gamma=1.0, use_gaussian=True,window_size=11,
                 channels=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.C3 = C2 / 2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_gaussian = use_gaussian
        self.channels = channels

        self.window = self.create_window(window_size, channels)

    def create_window(self, size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        sigma = 1.5
        _1D_window = gaussian(size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, size, size).contiguous())
        return window

    def gaussian_filter(self, img, window):
        padding = self.window_size // 2
        window = window.to(img.device)
        return F.conv2d(img, window, padding=padding, groups=self.channels)

    def ssim(self, x, y):
        if self.use_gaussian:
            mu_x = self.gaussian_filter(x, self.window)
            mu_y = self.gaussian_filter(y, self.window)
        else:
            mu_x = x.mean(dim=[2, 3], keepdim=True)
            mu_y = y.mean(dim=[2, 3], keepdim=True)

        sigma_x = ((x - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_y = ((y - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=[2, 3], keepdim=True)

        L = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
        C = (2 * sigma_x.sqrt() * sigma_y.sqrt() + self.C2) / (sigma_x + sigma_y + self.C2)
        S = (sigma_xy + self.C3) / (sigma_x.sqrt() * sigma_y.sqrt() + self.C3)

        L = L ** self.alpha
        C = C ** self.beta
        S = S ** self.gamma

        return L * C * S

    def forward(self, x, y):
        assert x.size() == y.size(), "must be same size"
        ssim_value = self.ssim(x, y)
        ssim_loss = 1 - ssim_value
        return ssim_loss.mean()


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        import torchvision.models as models

        vgg16 = models.vgg16(pretrained=True).features

        self.blocks = nn.ModuleList([
            vgg16[:4],
            vgg16[4:9],
            vgg16[9:16],
            vgg16[16:23]
        ])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False  # 冻结这些层的参数
        # # RGB
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        if self.resize:
            input = nn.functional.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = nn.functional.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        input = (input + 1) / 2
        target = (target + 1) / 2

        # 归一化
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        loss = 0.0
        for block in self.blocks:
            input = block(input)
            target = block(target)
            loss += nn.functional.mse_loss(input, target)

        return loss



class process_subimg_and_compute_loss(nn.Module):
    def __init__ (self, num_subimages=10, subimage_size= (32, 32), threshold=0.8,img_mean_percent=0.5,patch_loss_model='ssim',ssim_alpha=1.0, ssim_beta=1.0, ssim_gamma=1.0):
        super().__init__()
        self.num_subimages=num_subimages
        self.subimage_size=subimage_size
        self.threshold=threshold

        if 'pro' in patch_loss_model.lower():
            self.patch_loss=VGGPerceptualLoss().to('cuda')
        elif 'ssim' in patch_loss_model.lower():
            self.patch_loss=SSIMLoss(alpha=ssim_alpha, beta=ssim_beta, gamma=ssim_gamma)
        self.total_loss=SSIMLoss(alpha=1.0, beta=1.0, gamma=1.0)
        self.img_mean_percent=img_mean_percent

    def forward(self,images_a, images_b, masks,):

        batch_size, channels, height, width = images_a.shape
        subimage_height, subimage_width = self.subimage_size

        binary_masks = (masks <= self.threshold).float()

        processed_images_a = images_a * binary_masks
        processed_images_b = images_b * binary_masks

        total_loss=self.total_loss(processed_images_a,processed_images_b)

        if self.num_subimages>0:
            subimages_a = []
            subimages_b = []

            for i in range(batch_size):
                for _ in range(self.num_subimages):
                    top = random.randint(0, height - subimage_height)
                    left = random.randint(0, width - subimage_width)
                    subimage_a = processed_images_a[i, :, top:top + subimage_height, left:left + subimage_width]
                    subimage_b = processed_images_b[i, :, top:top + subimage_height, left:left + subimage_width]
                    subimages_a.append(subimage_a)
                    subimages_b.append(subimage_b)

            subimages_a_batch = torch.stack(subimages_a)
            subimages_b_batch = torch.stack(subimages_b)

            l1_loss = self.patch_loss(subimages_a_batch, subimages_b_batch)
        elif self.num_subimages==0:
            l1_loss = 0
        else:
            l1_loss=self.patch_loss(processed_images_a, processed_images_b)
        return l1_loss*(1-self.img_mean_percent)+total_loss*self.img_mean_percent



class MaskGanModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.binary_thr=self.opt.binary_thr
        self.isA_label=self.opt.isA_label
        self.isB_label=self.opt.isB_label

        self.output_nc=self.opt.output_nc


        self.lambda_ColorConsistency=max(0,self.opt.lambda_ColorConsistency)
        self.lambda_BrightnessConsistency=max(0,self.opt.lambda_BrightnessConsistency)

        self.trained_times = 0


        self.loss_names = ['G','GA', 'D_G','diversity']
        if self.lambda_ColorConsistency or self.lambda_BrightnessConsistency:
            self.loss_names.append('colorBright')

        self.visual_names  = ['fake_A','fake_A_mask', 'real_B','B_masks','real_A', ]

        if self.isTrain:
            self.model_names = ['G','D_G']
        else:
            self.model_names = ['G',]


        from build_model.creat_model_from_cfg import CreatModelFromCfg
        self.netG = CreatModelFromCfg(self.opt.G_cfg,ch=self.opt.input_nc).to(self.device)

        if self.isTrain:  # define discriminators
            if self.opt.DG_cfg:
                self.netD_G = CreatModelFromCfg(self.opt.DG_cfg,ch=self.opt.input_nc+1).to(self.device)
            else:
                self.netD_G = networks.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.loss_cont=process_subimg_and_compute_loss(self.opt.num_subimages,self.opt.subimage_size,
                                                           self.opt.binary_thr,self.opt.img_mean_percent,
                                                           patch_loss_model=self.opt.patch_loss_model,
                                                           ssim_alpha=self.opt.ssim_alpha,ssim_beta=self.opt.ssim_beta,ssim_gamma=self.opt.ssim_gamma)


            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), ),  lr=opt.G_lr, betas=(opt.G_beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_G.parameters(), ), lr=opt.D_lr, betas=(opt.D_beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.diversity_cretic=SSIMLoss()

            self.G_loss_mask=torch.nn.MSELoss().to(self.device)

            self.sclar_G=GradScaler()
            self.sclar_D=GradScaler()

            self.Color_Brightness_loss=ColorBrightnessConsistencyLoss(self.lambda_ColorConsistency,self.lambda_BrightnessConsistency)

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        brightness_factor = torch.FloatTensor(1).uniform_(0.5, 1.1).to(self.device)
        self.real_B=(self.real_B+1)*brightness_factor-1
        brightness_factor = torch.FloatTensor(1).uniform_(0.8, 1.5).to(self.device)
        self.real_A=(self.real_A+1)*brightness_factor-1

        if self.isA_label:
            self.A_labels=input['A_label' if AtoB else 'B_label']
            self.A_lbs=self.A_labels[0].to(self.device)
            self.A_masks=self.A_labels[1].to(self.device)     # [b,h,w]
            self.A_masks=self.A_masks.float()     #
        if self.isB_label:
            self.B_labels=input['B_label' if AtoB else 'A_label']
            self.B_lbs=self.B_labels[0].to(self.device)     # [b,cls,x,y,h,w]
            self.B_masks=self.B_labels[1].to(self.device)     # [b,h,w]
            self.B_masks=self.B_masks.float()     # [b,h,w]


    def forward(self):
        if self.opt.amp:
            with autocast():
                self.fake_A_label, self.fake_A= self.netG(self.real_A)  # G_A(A)
                self.fake_A_mask= self.fake_A_label

        else:
            self.fake_A_label, self.fake_A = self.netG(self.real_A)  # G_A(A)
            self.fake_A_mask = self.fake_A_label

        self.fakeA_fakeAmask = torch.cat((self.fake_A,self.fake_A_mask),dim=1)


    def backward_D_basic(self, netD, real, fake):

        if self.opt.amp:
            with autocast():
                # Real
                pred_real = netD(real)
                loss_D_real = self.criterionGAN(pred_real, True)

                # Fake
                pred_fake = netD(fake.detach())
                loss_D_fake = self.criterionGAN(pred_fake, False)
                # Combined loss and calculate gradients
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            self.sclar_D.scale(loss_D).backward()
        else:
            # Real
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)

            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
        return loss_D



    def backward_D_G(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        real_crack_mask=torch.cat((self.real_B,self.B_masks),dim=1)
        self.loss_D_G = self.backward_D_basic(self.netD_G, real_crack_mask, self.fakeA_fakeAmask)

    def optimize_G(self):
        b=len(self.fake_A)
        if b % 2 != 0:
            b = b - 1
        half_batch=int(b/2)

        if self.opt.amp:
            with autocast():
                # GAN loss D_A(G_A(A))
                self.loss_GA = self.criterionGAN(self.netD_G(self.fakeA_fakeAmask), True)
                self.loss_GA*=self.opt.lambda_GA


                self.loss_diversity=1-self.diversity_cretic.forward(self.fake_A[:half_batch,:,:,:],self.fake_A[half_batch:2*half_batch,:,:,:])

                if self.epoch > self.opt.crack_star_epoch:
                    loss_colorBright_G = self.loss_cont(self.real_A, self.fake_A, self.fake_A_mask)
                    self.loss_colorBright=loss_colorBright_G*self.lambda_BrightnessConsistency

                else:
                    self.loss_colorBright =0

                self.loss_G = self.loss_GA+self.loss_colorBright+self.loss_diversity

        else:
            # GAN loss D_A(G_A(A))
            self.loss_GA = self.criterionGAN(self.netD_G(self.fakeA_fakeAmask), True)
            self.loss_GA *= self.opt.lambda_GA

            self.loss_diversity = 1 - self.diversity_cretic.forward(self.fake_A[:half_batch, :, :, :],
                                                                    self.fake_A[half_batch:2 * half_batch, :, :, :])

            if self.epoch > self.opt.crack_star_epoch:
                loss_colorBright_G = self.loss_cont(self.real_A, self.fake_A, self.fake_A_mask)
                self.loss_colorBright = loss_colorBright_G * self.lambda_BrightnessConsistency

            else:
                self.loss_colorBright = 0

            self.loss_G = self.loss_GA + self.loss_colorBright + self.loss_diversity

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        if self.opt.amp:
            self.sclar_G.scale(self.loss_G).backward()
        else:
            self.loss_G.backward()

        if self.opt.amp:
            self.sclar_G.step(self.optimizer_G)
            self.sclar_G.update()
        else:
            self.optimizer_G.step()  # update G_A and G_B's weights


    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.epoch=epoch
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_G, ], False)
        self.optimize_G()  # calculate gradients

        self.set_requires_grad([self.netD_G,], True)
        self.optimizer_D.zero_grad()

        self.backward_D_G()  # calculate gradients

        if self.opt.amp:
            self.sclar_D.step(self.optimizer_D)
            self.sclar_D.update()
        else:
            self.optimizer_D.step()  # update weights



