'''
    20240326
    主要用于理解cnn的流程，暂时还没有验证是否能完整运行。

    cnn主要流程：
        自定义dataset
            class custom_dataset(torch.utils.data.Dataset):
                def __init__(self,path_dir):
                    ...
                def __len__(self):
                    return(len(imgs))
                def __getitem__(self,idx):
                    ...
                    return sig_img, targets
        创建dataloader
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        构建模型
            backbone + head

            backbone: 利用各种卷积获取一系列的特征图
            head： 将backbone的特征图卷积输出为想要的内容，包括 分类、检测、分割。
                分类：
                    self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
                    self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
                    self.drop = nn.Dropout(p=dropout_p, inplace=True)
                    self.linear = nn.Linear(c_, c_out)  # to x(b,c_out)   c_out 为类别数

                    self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


                检测：
                    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
                    # no:每个anchor/方格对应的参数个数     no=nc+5     (5：box+conf)
                    # na: 每个特征层/方格子上的anchor个数

                分割：
                    box部分：
                    self.no = 5 + nc + self.nm    # nm 输出的mask个数，经验值取32
                    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

                    self.cv1 = Conv(c1, c_, k=3)    # c_ 经验值取256
                    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
                    self.cv2 = Conv(c_, c_, k=3)
                    self.cv3 = Conv(c_, nm)    # nm 输出的mask个数



        计算损失
        反向传播
        保存模型
'''

import os

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common import *



# todo 定义dataset
class clas_dataset(Dataset):
    '''
        dataset/
            class_1/
                image1.jpg
                image2.jpg
                ...
            class_2/
                image1.jpg
                image2.jpg
                ...
            ...


        from torchvision import transforms

        transform = transforms.Compose([
                                    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
                                    transforms.ToTensor(),           # 将图像转换为 PyTorch 张量
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
                                      ])

    '''

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        if  transform:
            self.transform = transform
        else:
            self.transform=transforms.Compose([transforms.ToTensor(),])
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        '''
            将 图片路径和图片类别 组层一个2维list：
            [[img_path1, label1],
             [img_path2, label2],
             ...
             ]
        '''
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                images.append((image_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image =cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class det_dataset(Dataset):
    def __init__(self,img_dir,label_dir,img_size=(640,640)):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.img_size=img_size
        self.img_files=os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name=self.img_files[idx]
        img_path=os.path.join(self.img_dir,img_name)

        img0=cv2.imread(img_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img0,self.img_size)

        label_path=os.path.join(self.label_dir,img_name.replace('.jpg','.txt'))
        with open(label_path,'r') as lf:
            lines=lf.readlines()

        targets = torch.zeros((len(lines), 6))
        '''
            每个样本的标签是一个大小为 [N, 6] 的张量，其中 N 是图像中目标的数量，6 表示每个目标的类别索引、中心坐标 x 和 y、宽度和高度，
            以及一个置信度（在这个示例中，我们将置信度设置为 1.0，因为 YOLO 格式中的置信度通常是 1）。
        '''
        for i,line in enumerate(lines):
            data=line.strip().split()
            class_idx=int(data[0])
            x_center, y_center, width, height = map(float, data[1:])

            # Convert YOLO format to normalized coordinates
            x_center *= self.img_size[0]
            y_center *= self.img_size[1]
            width *= self.img_size[0]
            height *= self.img_size[1]

            # Store target data
            targets[i] = torch.tensor([class_idx, x_center, y_center, width, height, 1.0])

        return img,targets


class seg_dataset(Dataset):
    def __init__(self,img_dir,label_dir,img_size=(640,640)):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.img_size=img_size
        self.img_files=os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name=self.img_files[idx]
        img_path=os.path.join(self.img_dir,img_name)

        # 获取图片数据
        img0=cv2.imread(img_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img0,self.img_size)

        h,w,_=img.shape

        label_path=os.path.join(self.label_dir,img_name.replace('.jpg','.txt'))
        with open(label_path,'r') as lf:
            labels = [x.split() for x in lf.read().strip().splitlines() if len(x)]

            if any(len(x) > 4 for x in labels):
                classes = np.array([x[0] for x in labels], dtype=np.float32)

                # todo 获取分割坐标点
                segments=[]
                if len(labels):
                    for label in labels:
                        org_lab=np.array(label[1:],dtype=np.float32).reshape(-1,2)   # 原始的归一化后的坐标点

                        # 将归一化后的坐标点映射到新的图片尺寸上
                        new_lab=np.copy(org_lab)
                        new_lab[...,0]=org_lab[...,0]*self.img_size[0]
                        new_lab[...,1]=org_lab[...,1]*self.img_size[1]

                        segments.append(new_lab)

                    # 根据segment获取box
                    box=[]
                    for sig_seg in segments:
                        xs,ys=sig_seg.T
                        box.append([xs.min(),ys.min(),xs.max(),ys.max()])

                    # todo 获取类别和box信息
                    targets = np.concatenate((classes.reshape(-1, 1), box, 1))  # (cls, xywh)

                    # 制作masks
                    masks=[]
                    for mi in range(len(segments)):
                        mask = np.zeros(self.img_size, dtype=np.uint8)
                        polygons = np.asarray(segments[mi])
                        polygons = polygons.astype(np.int32)
                        cv2.fillPoly(mask, polygons, color=1)

                        masks.append(mask)

        return torch.from_numpy(img), torch.from_numpy(targets), torch.from_numpy(masks)



# todo 定义检测头
class Classify_head(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    '''
        先卷积，然后pool，然后flatten，然后drop，最后linear
    '''
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

class Detect_head(nn.Module):
    # YOLOv5 Detect head for detection models
    '''
        检测头的本质就是使用k=1的卷积进行全连接：  conv2D(c1,c2=na*(nc+5),k=1)
    '''
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment_head):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    '''
                        .split((2, 2, self.nc + 1), 4)：这是对 sigmoid 处理后的张量进行切分操作。
                        (2, 2, self.nc + 1) 表示切分的位置，具体地，前两个元素表示切分后第一个部分的大小为 2（即 xy 坐标），
                            中间两个元素表示切分后第二个部分的大小为 2（即 wh，宽度和高度），
                            最后一个元素表示切分后第三个部分的大小为 self.nc + 1（即 conf 和类别概率）。
                        4 表示沿着第四个维度（通常是通道维度）进行切分。
                    '''
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Segment_head(Detect_head):
    # YOLOv5 Segment head for segmentation models
    '''
       nm 是指分割任务中的掩模（mask）数量。这个数量通常是根据具体的分割任务和模型设计进行选择的。
       npr 表示原型（prototype）的数量，用于分割任务。这个数量的选择通常是根据任务的复杂性、模型设计以及性能需求进行权衡的。
            原型通常用于分割任务中的实例分割或语义分割，它们可以看作是一种特征表示，用于指示图像中不同区域的特征和属性。
            选择适当数量的原型对于模型的分割性能和泛化能力非常重要。
       通常情况下，nm 和 npr的选择取决于以下因素：
            任务复杂度： 分割任务的复杂度会影响到掩模的数量。如果任务涉及到多个类别、多个目标实例以及复杂的场景，可能需要更多的掩模来捕捉目标的细节和变化。
            分辨率： 分割任务中的掩模需要足够高的分辨率来准确地捕捉目标的边界和细节。如果输入图像的分辨率较高，可能需要更多的掩模来保证分割的精度。
            模型设计： YOLOv5 中的掩模数量可能是根据模型架构和设计选择的。掩模的数量可能会影响模型的参数数量、计算量以及分割性能。
            在 YOLOv5 中选择 nm=32 的原因可能是通过实验和调优确定的。具体的选择可能是为了平衡模型的性能和效率，以及保证在一定程度上能够满足分割任务的要求。
    '''
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect_head.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


# todo 解析模型
def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")  # 打打印日志信息，  >3表示靠右3的宽度
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    '''
        前面几项直接从字典d中取值,最后一项act使用了d.get()的原因是:
        activation是可选配置的参数,模型配置字典d可能包含,也可能不包含该键。
        直接使用d['activation']如果键不存在,会抛出KeyError异常。
        而d.get('activation')在键不存在时,会直接返回None,避免了错误。
        也就是说,对可选参数使用d.get()更加安全,可以处理参数未指定的情况。
        这在配置解析中是很常见的做法,用d.get()获取一个可选配置键的值,否则返回默认值(这里是None)。
    '''
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    '''
        上式用于计算第一层的anchors的数量，除以2是因为，如果是列表，则需要列出anchor的长宽。如yaml配置文件中：
        anchors:
          - [10,13, 16,30, 33,23]  # P3/8
          - [30,300, 62,45, 200,500]  # P4/16
          - [30,61, 62,45, 59,119]  # P4/16
          - [500,500, 800,800, 1000,1000]  # P5/32

        则na=3，不是4.
        所以为了避免出错，多尺度输出的时候需要每层的anchor数量一致，只是尺寸不一致而已。
    '''
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out，参数初始化
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        '''
            eval()被用来解析字符串表示的函数或类名。
            具体逻辑是:
                检查m是否是一个字符串， 如果是字符串,使用eval()执行这个字符串表达式,解析成一个可调用对象。如果不是字符串,直接返回原始值。
            如：
                m = 'nn.ReLU' # 字符串形式
                m = eval(m)   # m现在是nn.ReLU类

            对于配置文件中的模块名缩写,如 Conv、C3等,直接使用eval()是无法转化为可调用对象的。
            这里的解决方案是:
            在解析配置之前,先定义这些缩写对应的模块类,例如:
                Conv = torch.nn.Conv2d
                C3 = C3Module # 自定义的C3模块类
                m = eval(m)
            因为这里的eval()搜索命名空间,会找到事先定义的Conv和C3对应类,从而成功转化。

            YOLOv5中解决模块字符串到对象解析的方法是:
                在解析配置之前,先定义了一系列别名,对应从配置中可能出现的缩写字符串:
                from utils.general import *
                from utils.plots import *
                from models.common import *
                from models.experimental import *
                from models.yolo import *
                from data import *
            这些语句预先导入了utils、models、data等模块中定义的类。例如 'C3' 可以找到 models.common 中定义的 C3类。
            这样在解析配置时,直接可以使用eval()转换字符串:
        '''
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                '''
                    suppress允许你指定在某个代码块中忽略某些类型的异常。
                    这样即使块内代码出现异常,也可以继续执行而不是终止。
                    在YOLOv5中,suppress被用来忽略eval()解析字符串时可能出现的NameError:
                '''
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            '''
                *args[1:] 表示解包args列表,排除第一个元素。
                具体解释:
                    args是一个包含模块参数的列表
                    args[1:] 取得从索引1开始至结尾的所有元素
                    *args[1:] 对args[1:]进行解包(unpacking)操作
                    解包后的结果是将args[1:]展开为个别元素
                    举例如下:
                        args = [64, 64, 3, 2, 1]  
                        *args[1:] = 64, 3, 2, 1
                    所以在这里,*args[1:] 作用是将args中除第一个元素外的全部元素解包加入到新的列表中。
                    这种解包语法可以避免手动遍历args拼接列表,非常简洁实用。
            '''
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect_head, Segment_head}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment_head:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

        # todo ----   start        self add 20240321  -------
        elif m in {FP}:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m in {mobilenet_v3_small}:
            c1, c2 = ch[f], args[0]
            args = args[1:]  # mobilenet_v3_small只需要传入slice参数，不需要c2
        # todo ----   end        self add 20240321  -------

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        t = str(m)[8:-2].replace('__main__.', '')  # module type

        # todo  计算模型的参数量
        np = sum(x.numel() for x in m_.parameters())

        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        '''
                这段代码的作用是：对于给定的变量 f，如果它是整数，则将其加入一个列表中，并将该列表中不为 -1 的元素与变量 i 取模后的结果添加到 save 
            列表中。如果 f 不是整数，则直接将 f 中不为 -1 的元素与变量 i 取模后的结果添加到 save 列表中。
        '''
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

# 自定义模型
class custom_model():
    def __init__(self,model_cfg,ch):
        # 获取模型字典
        with open(model_cfg, encoding='ascii', errors='ignore') as cfg_f:
            self.model_dict = yaml.safe_load(cfg_f)
        self.model,self.save = parse_model(self.model_dict, ch)

    def forward(self,x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


if __name__ == '__main__':
    # Example usage
    img_dir = 'images'
    label_dir = 'labels'
    model_cfg='model.yaml'
    ch=[3,]      # 图片通道数

    # todo 实例化dataset
    dataset = det_dataset(img_dir, label_dir)

    # todo 创建dataloader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 实例化模型
    cus_model=custom_model(model_cfg,ch).to('cuda:0')



    for images, targets in data_loader:
        p=cus_model(images)


