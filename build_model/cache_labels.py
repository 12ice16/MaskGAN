import contextlib
import hashlib
import os
from pathlib import Path
from multiprocessing.pool import Pool, ThreadPool
from itertools import repeat

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm




IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'


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


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy     #对当前分割信息 s 进行转置（Transpose），将其转换为两个一维数组 x 和 y，分别表示分割信息中的 x 坐标和 y 坐标。
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh

def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)

                    # todo 获取分割坐标点
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)

                    # todo 获取类别和box信息
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]




def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask

def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)

def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index



def CacheLabels(img_paths,
                label_paths,
                cache_path='',
                prefix='',
                NUM_THREADS=-1,
                cache_version=0.6,
                retun_cache=True
                ):
    '''
        Cache dataset labels, check images and read shapes

        返回将标签数据放在cache字典，并保存cache到本地,并返回cache。
        cache={'img_path1'=[lb,img_shape,sege_poins],
                'img_path1'=[lb,img_shape,sege_poins],
                'img_path1'=[lb,img_shape,sege_poins],
                ...
                'hash' = ***
                'results' = nf, nm, ne, nc, len(img_paths)
                'msgs' = msgs  # warnings
                'version' = cache_version  # cache version
                }

        注意，img_paths和label_paths中的路径必须是字符串，不然hash会报错

        return cache     or    img_paths,list(labels), np.array(shapes), list(segments)
    '''
    if not cache_path:
        cache_path = Path(label_paths[0]).parent.with_suffix('.cache')

    # Check cache
    try:
        cache = np.load(cache_path, allow_pickle=True).item()  # load dict
        print(f'cache is already exist in {cache_path}')
        assert cache['version'] == cache_version  # matches current version
        assert cache['hash'] == get_hash(label_paths + img_paths)  # identical hash
    except Exception:
        cache = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {cache_path.parent / cache_path.stem}...'

        if NUM_THREADS<0:
            NUM_THREADS = min(8, max(1, os.cpu_count() - 1))   

        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(img_paths, label_paths, repeat(prefix))),
                        desc=desc,
                        total=len(img_paths),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    cache[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)

                if ne_f:
                    print(f'ne_f         {im_file}')
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

        pbar.close()
        if msgs:
            print('\n'.join(msgs))
        if nf == 0:
            print(f'{prefix}WARNING ⚠️ No labels found in {cache_path}. {HELP_URL}')
        cache['hash'] = get_hash(label_paths + img_paths)
        cache['results'] = nf, nm, ne, nc, len(img_paths)
        cache['msgs'] = msgs  # warnings
        cache['version'] = cache_version  # cache version
        try:
            np.save(cache_path, cache)  # save cache for next time
            cache_path.with_suffix('.cache.npy').rename(cache_path)  # remove .npy suffix
            print(f'{prefix}New cache created: {cache_path}')
        except Exception as e:
            print(f'{prefix}WARNING ⚠️ Cache directory {cache_path.parent} is not writeable: {e}')  # not writeable


    # return cache or labes
    if retun_cache:
        return cache
    else:
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        d = f'Scanning cache... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
        tqdm(None, desc=d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
        if cache['msgs']:
            print('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

        labels, shapes, segments = zip(*cache.values())
        img_paths = list(cache.keys())

        return img_paths,list(labels), np.array(shapes), list(segments)



def GetSigMaskFromCache_label(img_path,img_label,shape,img_segment,overlap=True,downsample_ratio=1,save_mask=False,normal255=False):
    '''
        利用来自cache的img_path,img_label,shape,img_segment获取对应mask图像
        img_path,img_label,shape,img_segment 来自cache第i个索引
        其中shape 是（w,h）
    '''
    img_shape=(shape[1],shape[0])      # h,w
    nl = len(img_label)  # number of labels
    sig_img_masks=[]
    if nl:
        for s_i in range(len(img_segment)):
            img_segment[s_i][..., 0] = img_segment[s_i][..., 0] * shape[0]
            img_segment[s_i][..., 1] = img_segment[s_i][..., 1] * shape[1]

        if overlap:
            sig_img_masks, sorted_idx = polygons2masks_overlap(img_shape, img_segment,downsample_ratio=downsample_ratio)
            if save_mask:
                if normal255:
                    # 归一化和转换为uint8,可以避免保存的mask掩码是全黑的，但是对于多类别的掩码，这样归一化会导致分类混乱。
                    max_val = sig_img_masks.max()
                    if max_val > 0:
                        sig_img_masks = (sig_img_masks / max_val) * 255
                    sig_img_masks = sig_img_masks.astype(np.uint8)

                img_dir=Path(img_path).parent
                mask_dir=os.path.join(img_dir.parent,f'{img_dir.name}_masks')
                Path(mask_dir).mkdir(parents=True, exist_ok=True)
                mask_path=os.path.join(mask_dir,f'{Path(img_path).stem}.png')
                cv2.imwrite(mask_path,sig_img_masks)
            sig_img_masks = sig_img_masks[None]  # (640, 640) -> (1, 640, 640)

            # 对结果进行阈值处理，大于0的像素设为1，其余为0
            sig_img_masks = (sig_img_masks > 0).astype(np.float32)



        else:
            sig_img_masks = polygons2masks(img_shape, img_segment, color=1, downsample_ratio=downsample_ratio)

    sig_img_masks = (torch.from_numpy(sig_img_masks) if len(sig_img_masks) else torch.zeros(1 if overlap else nl, img_shape[0]//downsample_ratio, img_shape[1] //downsample_ratio))

    return sig_img_masks




def GetAllLabelFromCache(cache,return_mask=False,overlap=True,downsample_ratio=1,save_mask=False,normal255=False):
    # Display cache
    nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
    d = f'Scanning cache... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
    tqdm(None, desc=d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
    if cache['msgs']:
        print('\n'.join(cache['msgs']))  # display warnings

    # Read cache
    [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

    labels, shapes, segments = zip(*cache.values())
    img_paths=list(cache.keys())

    masks=[]
    if return_mask:
        for i_seg in range(len(segments)):
            nl = len(labels[i_seg])  # number of labels
            sig_img_masks=[]
            if nl:
                img_shape=(shapes[i_seg][1],shapes[i_seg][0])

                for s_i in range(len(segments[i_seg])):
                    segments[i_seg][s_i][..., 0] = segments[i_seg][s_i][..., 0] * shapes[i_seg][0]
                    segments[i_seg][s_i][..., 1] = segments[i_seg][s_i][..., 1] * shapes[i_seg][1]

                if overlap:
                    sig_img_masks, sorted_idx = polygons2masks_overlap(img_shape,
                                                               segments[i_seg],
                                                               downsample_ratio=downsample_ratio)
                    if save_mask:
                        if normal255:
                            # 归一化和转换为uint8,可以避免保存的mask掩码是全黑的，但是对于多类别的掩码，这样归一化会导致分类混乱。
                            max_val = sig_img_masks.max()
                            if max_val > 0:
                                sig_img_masks = (sig_img_masks / max_val) * 255
                            sig_img_masks = sig_img_masks.astype(np.uint8)


                        img_dir=Path(img_paths[i_seg]).parent
                        mask_dir=os.path.join(img_dir.parent,f'{img_dir.name}_masks')
                        Path(mask_dir).mkdir(parents=True, exist_ok=True)
                        # os.makedirs(mask_dir,exist_ok=True)
                        mask_path=os.path.join(mask_dir,f'{Path(img_paths[i_seg]).stem}.png')
                        cv2.imwrite(mask_path,sig_img_masks)
                    sig_img_masks = sig_img_masks[None]  # (640, 640) -> (1, 640, 640)
                else:
                    sig_img_masks = polygons2masks(img_shape, segments[i_seg], color=1, downsample_ratio=downsample_ratio)

            sig_img_masks = (torch.from_numpy(sig_img_masks) if len(sig_img_masks) else torch.zeros(1 if overlap else nl, img_shape[0]//downsample_ratio, img_shape[1] //downsample_ratio))

            masks.append(sig_img_masks)


    return img_paths,list(labels), np.array(shapes), list(segments),masks



def save_mask_from_txt_dir(img_dir,label_folde_name):
    '''
        假设图像文件夹和对应的标签文件夹位于同一个父目录下，并且它们的文件名，只是路径不同。
        --father_folder
            img_floder:
                1.jpg
                2.jpg
            label_foldername:
                1.txt
                2.txt

    '''
    img_path,label_path=img_paths2label_paths(img_dir,label_folde_name)
    cache=CacheLabels(img_path,label_path,)
    img_paths,labels, shapes, segments,masks=GetAllLabelFromCache(cache,return_mask=True,save_mask=True,normal255=True)





if __name__ == '__main__':
    img_dir=r'crack_Gan_patch_label3/trainB'
    label_dir=r'crack_Gan_patch_label3/trainB_label'

    img_path=[]
    for img_p in Path(img_dir).rglob('*.jpg'):
        img_path.append(str(img_p))

    img_path,label_path=img_paths2label_paths(img_dir,'trainB_label')

    cache=CacheLabels(img_path,label_path,)
    img_paths,labels, shapes, segments,masks=GetAllLabelFromCache(cache,return_mask=True,save_mask=True,normal255=True)

    # img_paths,labels, shapes, segments=CacheLabels(img_path,label_path,retun_cache=False)
    # idx=1
    # sig_img_mask=GetSigMaskFromCache_label(img_paths[idx],labels[idx],shapes[idx],segments[idx],save_mask=True,normal255=True)

    print()
