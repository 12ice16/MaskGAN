# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license


import argparse
import os
import platform
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import cv2
from PIL import Image
import torch
import torchvision.utils
from torchvision import transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from build_model.common import *
from build_model.common2 import *
from build_model.general import LOGGER, check_version, check_yaml, make_divisible, print_args,colorstr
from build_model.plots import feature_visualization
from build_model.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device, time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None



class CreatModelFromCfg(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=1, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # todo 直接或者从.yaml文件获取模型dict
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value，对锚框的坐标点四舍五入为整数。

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        self.inplace = self.yaml.get('inplace', True)

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs

        layer_i=0
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            layer_i+=1

        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                if 'Batch' in str(m.bn).replace('__main__.', ''):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)



class infer():
    def __init__(self,weight,cfg='',img_size=None):
        self.device=('cuda:0' if torch.cuda.is_available() else 'cpu')

        if cfg:
            self.model=CreatModelFromCfg(cfg)

            self.model_state_dict=torch.load(weight,map_location='cpu')
            self.model.load_state_dict(self.model_state_dict)
        else:
            self.model=torch.load(weight)
        self.model=self.model.to(self.device).float()

        self.img_size=img_size


    def __call__(self, img_dir,save_dir='infer_out'):
        os.makedirs(save_dir,exist_ok=True)
        img_data,img_paths=self.getImgFromPath(img_dir,self.img_size)
        with torch.no_grad():
            out=self.model(img_data)

        for i,img_path in enumerate(img_paths):
            img_name=os.path.splitext(os.path.basename(img_path))[0]    # 不带后缀
            shutil.copy(img_path,os.path.join(save_dir,os.path.basename(img_path)))

            gen_mask_name=f'{img_name}_mask.png'
            org_img_shape=cv2.imread(img_path).shape
            tran2org=transforms.Resize(org_img_shape[:2])
            out_org=tran2org(out[i])

            torchvision.utils.save_image(out_org,os.path.join(save_dir,gen_mask_name))

        return out

    def getImgFromPath(self,img_dir,img_size=None):
        img_tensors=[]
        img_paths=[]
        img0s=[]
        # if not isinstance(img_paths,list):
        #     img_paths=[img_paths]
        data_tans=transforms.Compose(
            [transforms.Resize(img_size),
            transforms.ToTensor(),]
        )
        for img_name in os.listdir(img_dir):
            img_path=os.path.join(img_dir,img_name)
            if os.path.exists(img_path) and os.path.isfile(img_path):
                img_paths.append(img_path)
                img=Image.open(img_path)
                img0s.append(np.array(img))
                img_tensors.append(data_tans(img))
                # img=cv2.imread(img_path)
        #         img_tensor=torch.from_numpy(img)
        #         img_tensor=img_tensor.to(self.device)
        #         img_tensors.append(img_tensor/255.0)
        img_tensors=torch.stack(img_tensors)
        # img_tensors=img_tensors.permute(0,3,1,2)
        return img_tensors.to(self.device),img_paths




def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")    # 打打印日志信息，  >3表示靠右3的宽度
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors

    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out，参数初始化
    for i, (f, n, m, args) in enumerate(d['encoder'] + d['decoder']):  # from, number, module, args
    # for i, (f, n, m, args) in enumerate(d['encoder'] + d['decoder']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):

                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]

            # if c2 != no:  # if not output
            #     c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

        elif m in {FP}:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m in {mobilenet_v3_small,}:
            c1, c2 = ch[f], args[0]
            args=args[1:]    # mobilenet_v3_small只需要传入slice参数，不需要c2

        elif m in {UpConv,DepthDown,DepthUp,FSFDownUP,FSFDilation,FSFDilationF}:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]

        elif m in {SpatialAttention}:
            c1, c2 = ch[f], ch[f]

        elif m in {SelfAttention}:
            c1, c2 = ch[f], ch[f]
            args=[c1]

        elif m in {ResnetBlock}:
            c1, c2 = ch[f], args[0]

        elif m in {IdentOut,SplitLabelImg,Feature2Confidence,Last_Act}:
            pass
        elif m in {UnetInerConUpC3}:
            c1=ch[f[0]]
            c2=args[0]
            c_iner1=ch[f[1]]
            c_iner2=args[1]
            args=[c1,c2,c_iner1,c_iner2,*args[2:]]
        elif m in {ConcatTransformerBlock,}:
            c1=ch[f[-1]]
            c2 = ch[f[0]]+args[0]  # 注意，这个c2是拼接后的c2，不是transformer的输出通道C2
            args = [c1,  *args]
        elif m in [CBAM,ChannelAttention,ResCBAM]:
            c1=ch[f]
            c2=ch[f]
            args = [c1, *args]
        elif m in [AddFeature,]:
            c2=ch[f[-1]]

        elif m in [GetPartFeatures,]:
            c1 = ch[f]
            if args[-1]==-1:
                c2=c1-args[0]
            else:
                c2 = args[1]-args[0]
            args = [*args]

        elif m in [GetNoise,]:
            c1=  ch[f]
            c2= args[0]


        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type

        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def print_model():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='det_cycle_gan.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='mobilev3-seg-4x-DownSample.yaml', help='model.yaml')
    parser.add_argument('--cfg', type=str, default='model_cfg/wr_resnet_net.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='model_cfg/w_C3_net2.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='model_cfg/w_C3_net.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='model_cfg/w_net.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='model_cfg/yolov5s-seg-4x-DownSample.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    # im = torch.rand(opt.batch_size, 3, 1024, 1024).to(device)
    im = torch.rand(opt.batch_size, 3, 256, 256).to(device)
    # im = torch.rand(opt.batch_size, 3, 1024, 3200).to(device)

    model = CreatModelFromCfg(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = CreatModelFromCfg(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()

    out=model(im)

    # f1=model.model
    # f2=f1[2]
    f3=im.shape

    summary(model,input_size=im.shape)




if __name__ == '__main__':
    print_model()




