
import argparse
import copy
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

import models
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util import util
import data



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser.add_argument('--dataroot', default=r'H:\datatemp\crack_Gan', help='path to images (should have subfolders trainA, trainB,trainB_masks)')
    parser.add_argument('--name', type=str, default='', help='')
    parser.add_argument('--display_port', type=int, default=8901, help='visdom port of the web display')

    # model cfg
    parser.add_argument('--model', type=str, default='Mask_gan', help='')
    parser.add_argument('--G_cfg', type=str, default='build_model/model_cfg/G_net.yaml',help='G model config yaml')

    parser.add_argument('--lambda_GA', type=float, default=0.80, help='weight for GA loss (A -> B )')
    parser.add_argument('--lambda_GB', type=float, default=1.0, help='weight for GB loss (B -> A)')

    parser.add_argument('--lambda_ColorConsistency', type=float, default=0.0, help='use for RGB')
    parser.add_argument('--lambda_BrightnessConsistency', type=float, default=0.5, help='use for gray img')
    parser.add_argument('--patch_loss_model', type=str, default='pro', help='patch loss model,   "ssim"   "pro" ')
    parser.add_argument('--ssim_alpha', type=float, default=1.0, help='')
    parser.add_argument('--ssim_beta', type=float, default=1.0, help='')
    parser.add_argument('--ssim_gamma', type=float, default=1.0, help='')

    parser.add_argument('--num_subimages', type=int, default=-1, help='patch number for local loss')
    parser.add_argument('--subimage_size', type=tuple, default=(32,32), help='patch size for local loss')
    parser.add_argument('--binary_thr', type=float, default=0.8, help='')
    parser.add_argument('--img_mean_percent', type=float, default=0.5, help='')

    parser.add_argument('--DG_cfg', type=str, default='build_model/model_cfg/D_net.yaml',help='D_net model config yaml')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

    # model parameters
    parser.add_argument('--crack_star_epoch', type=int,default= 70,  help="number of initialized epochs")

    # chose model
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=200,   help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--G_beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--D_beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--G_lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.00001, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan',      help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--lr_policy', type=str, default='linear',       help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=10,   help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--isTrain', type=bool, default=True, help='whether train ? ')
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=128, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

    parser.add_argument('--pool_size', type=int, default=0,   help='the size of image buffer that stores previously generated images')



    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='unalignedCache', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization|unalignedWnet]')
    parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
    parser.add_argument('--isA_label', type=bool, default=False, help='wether use data A label')
    parser.add_argument('--isB_label', type=bool, default=True, help='wether use data B label')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    # parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', default=True, help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    parser.add_argument('--wandb_project_name', type=str, default='', help='specify wandb project name')

    parser.add_argument('--amp', type=bool, default=False, help='weather use amp to training')

    # visdom and HTML visualization parameters
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost",      help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main',  help='visdom display environment name (default is "main")')
    parser.add_argument('--update_html_freq', type=int, default=1000,   help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--no_html', action='store_true',  help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    # network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10,   help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,    help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')


    # get the basic options
    opt, _ = parser.parse_known_args()


    isTrain=opt.isTrain

    # modify dataset-related parser options
    dataset_name = opt.dataset_mode
    dataset_option_setter = data.get_option_setter(dataset_name)
    parser = dataset_option_setter(parser, isTrain)


    return parser.parse_known_args()[0] if known else parser.parse_args() ,parser


def print_options( opt,parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

    save_G_cfg_path=os.path.join(expr_dir,os.path.basename(opt.G_cfg))
    if os.path.exists(save_G_cfg_path):
        os.remove(save_G_cfg_path)
    shutil.copy(opt.G_cfg,save_G_cfg_path)

    save_DG_cfg_path=os.path.join(expr_dir,os.path.basename(opt.DG_cfg))
    if os.path.exists(save_DG_cfg_path):
        os.remove(save_DG_cfg_path)
    shutil.copy(opt.DG_cfg,save_DG_cfg_path)


    model_filename = opt.model + "_model.py"
    save_model_file_path=os.path.join(expr_dir,model_filename)
    if os.path.exists(save_model_file_path):
        os.remove(save_model_file_path)
    shutil.copy(os.path.join('models',model_filename),save_model_file_path)




    current_script = __file__
    trainpy_save_path = os.path.join(expr_dir, os.path.basename(current_script))
    shutil.copyfile(current_script, trainpy_save_path)


if __name__ == '__main__':
    # opt = TrainOptions().parse()   # get training options

    opt,parser = parse_opt()


    opt.name = f'{time.strftime("%Y%m%d_%H-%M-%S")}_{opt.name}'

    # process opt.suffix
    if opt.suffix:
        suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        opt.name = opt.name + suffix

    print_options(opt,parser)

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

