import numpy as np
import os
from args import args
from PIL import Image
from os import listdir
from os.path import join
import torch
from torchvision import transforms

"""
增强版工具函数（兼容新版 scipy）
使用 Pillow 替代已废弃的 scipy.misc.imread/imsave/imresize
与原始 utils.py 接口完全一致，可直接替换使用
"""


def load_dataset(ir_imgs_path, vi_imgs_path, BATCH_SIZE, num_imgs=None):
    """将红外和可见光图像路径列表按 batch 大小对齐，返回路径列表和 batch 数量。"""
    if num_imgs is None:
        num_imgs = len(ir_imgs_path)
    ir_imgs_path = ir_imgs_path[:num_imgs]
    vi_imgs_path = vi_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        ir_imgs_path = ir_imgs_path[:-mod]
        vi_imgs_path = vi_imgs_path[:-mod]
    batches = int(len(ir_imgs_path) // BATCH_SIZE)
    return ir_imgs_path, vi_imgs_path, batches


def make_floor(path1, path2):
    """拼接路径并自动创建目录（如果不存在）"""
    path = os.path.join(path1, path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def _pil_resize(image, size, mode=Image.NEAREST):
    """使用 Pillow 的 resize 替代 scipy.misc.imresize"""
    return image.resize((size[1], size[0]), mode)


def get_image(path, height=args.hight, width=args.width, mode='L'):
    """
    读取单张图像，缩放到指定尺寸。
    返回 numpy 数组，灰度图值域 [0, 255]。
    """
    if mode == 'L':
        image = Image.open(path).convert('L')
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    else:
        image = Image.open(path)

    if height is not None and width is not None:
        image = image.resize((width, height), Image.NEAREST)

    image = np.array(image, dtype=np.float32)
    return image


def get_train_images_auto(paths, height=args.hight, width=args.width, mode='RGB'):
    """
    批量读取训练图像，归一化到 [-1, 1] 并返回 torch.Tensor。
    注意：原始代码在 get_train_images_auto 和 get_train_images 中做了双重归一化，
    这里保持与原始行为一致。
    """
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = (images - 127.5) / 127.5
    return images


def prepare_data(directory):
    """
    读取指定目录下所有图像文件的路径（支持 png, jpg, jpeg, bmp, tif）。
    """
    directory = os.path.join(os.getcwd(), directory)
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def save_feat(index, C, ir_atten_feat, vi_atten_feat, result_path):
    """
    将红外和可见光的注意力特征图（每个通道）保存为图像文件。
    """
    ir_atten_feat = (ir_atten_feat / 2 + 0.5) * 255
    vi_atten_feat = (vi_atten_feat / 2 + 0.5) * 255

    ir_feat_path = make_floor(result_path, "ir_feat")
    index_irfeat_path = make_floor(ir_feat_path, str(index))
    vi_feat_path = make_floor(result_path, "vi_feat")
    index_vifeat_path = make_floor(vi_feat_path, str(index))

    for c in range(C):
        ir_temp = ir_atten_feat[:, c, :, :].squeeze()
        vi_temp = vi_atten_feat[:, c, :, :].squeeze()
        feat_ir = ir_temp.cpu().clamp(0, 255).data.numpy()
        feat_vi = vi_temp.cpu().clamp(0, 255).data.numpy()

        ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
        ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
        Image.fromarray(feat_ir.astype(np.uint8)).save(ir_atten_path)

        vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
        vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
        Image.fromarray(feat_vi.astype(np.uint8)).save(vi_atten_path)


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = (image - 127.5) / 127.5
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(Image.fromarray(image.astype(np.uint8))).float().numpy() * 255
            image = (image - 127.5) / 127.5
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def save_images(path, data):
    """保存单张图像"""
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    Image.fromarray(data.astype(np.uint8)).save(path)


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images
