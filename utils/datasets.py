import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
对数据集进行操作的py文件，包含图像的填充、图像大小的调整、测试数据集的加载类、评估数据集的加载类。整个文件包含3个函数和2个类
"""

'''图片填充函数：
将图片用pad_value填充成一个正方形，返回填充后的图片以及填充的位置信息'''
def pad_to_square(img, pad_value):  # 图片填充为正方形，pad_value：补全部分所填充的数值
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 填充方式，如果高小于宽则上下填充，如果高大于宽，左右填充
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # 图片填充，参数img是原图，pad是填充方式（0,0,pad1,pad2）或（pad1，pad2,0，0），value是填充的值
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


'''图片调整大小：将正方形图片使用插值方法，改变到固定size大小'''
def resize(image, size):
    #torch.nn.functional.interpolate实现插值和上采样，size输出大小，scale_factor指定输出为输入的多少倍数，
    #mode可使用的上采样算法，有’nearest’, ‘linear’, ‘bilinear’, ‘bicubic’ , ‘trilinear’和’area’. 默认使用’nearest’
    #将原始图片解压后用“nearest”方法进行填充，然后再压缩
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


"""
   随机裁剪函数：将图片随机裁剪到某个尺寸（使用插值法）
   min_size,max_size 随机数所在的范围
"""
def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images



'''
   用来定义数据集的标准格式
   从文件夹中读取图片，将图片padding成正方形，所有的输入图片大小调整为416*416，返回图片的数量
'''
#用于预测：在detect.py中加载数据集时使用
class ImageFolder(Dataset): # 这是定义数据集的标准格式
    def __init__(self, folder_path, img_size=416):#初始化的参数为：测试图片所在的文件夹的路径、图片的尺寸（用于输入到网络的图片的大小）
        #获取文件夹下图片的路径，files是图片路径组成的列表
        self.files = sorted(glob.glob("%s/*.*" % folder_path))#例在detect.py中folder_path=data/samples
        self.img_size = img_size #初始化图片的尺寸

    def __getitem__(self, index): #根据索引获取列表里的图片的路径
        img_path = self.files[index % len(self.files)]
        # 将图片转换为tensor的格式
        img = transforms.ToTensor()(Image.open(img_path))
        # 用0将图片填充为正方形
        img, _ = pad_to_square(img, 0)
        # 将图片大小调整为指定大小
        img = resize(img, self.img_size)
        return img_path, img  # 返回 index 对应的图片的 路径和 图片

    def __len__(self):
        return len(self.files) # 所有图片的数量


"""
Dataset类：
    pytorch读取图片，主要通过Dataset类。Dataset类作为所有datasets的基类，所有的datasets都要继承它
    init： 用来初始化一些有关操作数据集的参数
    getitem:定义数据获取的方式（包括读取数据，对数据进行变换等），该方法支持从 0 到 len(self)-1的索引。obj[index]等价于obj.getitem
    len:获取数据集的大小。len(obj)等价于obj.len()

    数据集加载类2：加载并处理图片和图片标签，返回的是图片路径，经过处理后的图片，经过处理后的标签
"""


# 用于评估：在test.py中加载数据集时候使用
class ListDataset(Dataset):
    # 数据的载入
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        # 初始化参数：list_path为验证集图片的路径组成的txt文件，的路径、img_size为图片大小（输入到网络中的图片的大小）、augment是否数据增强、multiscale是否使用多尺度，normalized_labels标签是否归一化
        # 获取验证集图片路径img_files,是一个列表
        with open(list_path, "r") as file:  # 打开valid.txt文件，内容为data/custom/images/train.jpg，指明了验证集对应的图片路径
            self.img_files = file.readlines()
        # 获取验证集标签路径label_files：是一个列表，根据验证集图片的路径获取标签路径，两者之间是文件夹及后缀名不同，
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        # 其他设置
        self.img_size = img_size
        self.max_objects = 100  # 最多目标个数
        self.augment = augment  # bool. 是否使用增强
        self.multiscale = multiscale  # bool. 是否多尺度输入，每次喂到网络中的batch中图片大小不固定。
        self.normalized_labels = normalized_labels  # bool. 默认label.txt文件中的bbox是归一化到0-1之间的
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32  # self.min_size和self.max_size的作用主要是经过数据处理后生成三种不同size的图像，目的是让网络对小物体和大物体都有较好的检测结果。
        self.batch_count = 0  # 当前网络训练的是第几个batch

        # 根据下标 index 找到对应的图片,并对图片、标签进行填充，适应于正方形，对标签进行归一化。返回图片路径，图片，标签

    def __getitem__(self, index):  # 读取数据和标签

        # ---------
        #  Image
        # ---------
        # 根据索引获取图片的路径
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img_path = 'data/coco' + img_path
        # print (img_path)
        # 把图片变为tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        #  把图片变为三个通道，获取图像的宽和高
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)  # 如果标注bbox不是归一化的，则标注里面的保存的就是真实位置
        # 把图片填充为正方形，返回填充后的图片，以及填充的信息 pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        img, pad = pad_to_square(img, 0)
        # 填充后的高和宽
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        # 根据索引，获取标签路径
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        label_path = 'data/coco' + label_path
        # print (label_path)

        targets = None
        if os.path.exists(label_path):  # 读取某张图片的标签信息
            # 读取一张图片内的边界框：txt文件包含的边界框的坐标信息是归一化后的坐标
            boxes = torch.from_numpy(
                np.loadtxt(label_path).reshape(-1, 5))  # [0class_id, 1x_c, 2y_c, 3w, 4h] 归一化的, 归一化是为了加速模型的收敛
            # np.loadtxt()函数主要将标签里的值转化为araray
            #  将归一化后的坐标变为适应于原图片的坐标
            # 使用(x_c, y_c, w, h)获取真实坐标（左上，右下）
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # 将坐标变为适应于填充为正方形后图片的坐标
            # 标注要和原图做相同的调整 pad（0左，1右，2上，3下）
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # 将边界框的信息转变为（x,y,w,h）形式,并归一化
            # (padded_w, padded_h)是当前padding之后图片的宽度
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            # (w_factor, h_factor)是原始图的宽高
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            # #长度为6：（0，类别索引，x,y,w,h）
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)  # 数据增强

        return img_path, img, targets  # 返回index对应的图片路径，填充和调整大小之后的图片，图片标签归一化后的格式 (img_id, class_id, x_c, y_c, w, h)

    # collate_fn：实现自定义的batch输出。如何取样本的，定义自己的函数来准确地实现想要的功能，并给target赋予索引
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))  # #获取批量的图片路径、图片、标签
        # target的每个元素为每张图片的所有边界框的信息
        targets = [boxes for boxes in targets if boxes is not None]
        # 读取target的每个元素，每个元素为一张图片的所有边界框信息，并微每张图片的边界框标相同的序号
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i  # 为每个边界框增加索引，序号
        targets = torch.cat(targets, 0)  # 直接将一个batch中所有的bbox合并在一起，计算loss时是按batch计算
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        # 每10个样本随机调整图像大小
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])  # 调整图像大小放入栈中
        self.batch_count += 1
        return paths, imgs, targets  # 返回归一化后的[img_id, class_id, x_c, y_c, h, w]

    def __len__(self):
        return len(self.img_files)

