from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    ##########################################################################################################################
    '''(1)参数解析'''
    parser = argparse.ArgumentParser()
    # 测试文件夹路径
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    # yolov3的模型信息（网络层，每层的卷积核数量，尺寸，步长。。。）
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # 预训练模型路径
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    # 类名字
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    # 目标置信度阈值
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    # NMS的IoU阈值
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # 批量大小
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # CPU线程
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # 图片维度
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # checkpoint_model
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)  # 创建预测图片的输出位置
    ##########################################################################################################################
    '''(2)模型构建'''
    # 加载模型：这条语句加载darkent模型结构，即YOLOv3模型。Darknet模型在model.py中进行定义。
    # 将模型设置为评估模式
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)  # 根据模型的配置文件，搭建模型的结构
    # 为模型结构加载训练的权重（模型参数）
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # 设置模型为评估模式，不然只要输入数据就会进行参数更新、优化
    ##########################################################################################################################
    '''(3)数据集加载、类别加载'''
    # 加载测试的图片：
    # dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
    # 也可以使用`for inputs, labels in dataloaders`进行可迭代对象的访问
    # 一般我们实现一个datasets对象，传入到dataloader中；然后内部使用yeild返回每一次batch的数据
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        # 评估数据集，ImageFolder在datasets.py中定义，返回的是图片路径，和经过处理（填充、调整大小）的图片
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # 加载类别名，classes是一个列表
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    # 创建保存图片路径和图片检测信息的列表
    imgs = []
    img_detections = []
    ##########################################################################################################################
    """(3)模型预测：将图片路径、图片预测结果存入imgs和img_detections列表中"""

    print("\nPerforming object detection:")
    prev_time = time.time()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # 测试图片的检测：并将图片路径和检测结果信息保存
    # 算出batch中图片的地址img_paths和检测结果detections
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):  # 使用dataloader加载数据，加载的数据为一批量的数据
        # 把输入图像转换为tensor并变为变量
        input_imgs = Variable(input_imgs.type(Tensor))
        # 目标检测：使用模型检测图像，检测结果为一个张量，
        # 对检测结果进行非极大值抑制，得到最终结果
        with torch.no_grad():
            detections = model(input_imgs)
            # print(detections.shape)#[：, 10647, 85]
            ##非极大值抑制：将边界框信息，转变为左上右下坐标，并且去除置信度低的坐标. (x1, y1, x2, y2, object_conf, class_score, class_pred)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)  # 非极大值抑制[:,:,7]

        # 打印:检测时间，检测的批次
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # 保存图片路径，图片的检测信息(经过NMS处理后)
        imgs.extend(img_paths)
        img_detections.extend(detections)  # 长度为7

    ##########################################################################################################################
    """（4）将检测结果绘制到图片，并保存"""

    # 边界框颜色
    cmap = plt.get_cmap("tab20b")  # Bounding-box colors
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # 遍历图片
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))

        # 读取图片并将图片绘制在plt.figure
        img = np.array(Image.open(path))  # 读取图片
        plt.figure()  # 创建图片画布
        fig, ax = plt.subplots(1)
        ax.imshow(img)  # 将读取的图片绘制到画布

        # 将图片对应的检测的边界框和标签绘制到图片上
        if detections is not None:
            # 将检测的边界框（对填充、调整大小的原图的预测），重新设置尺寸，使其与原图目标能匹配
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

            # 获取检测结果的类标签，并为每一个类指定一种颜色
            unique_labels = detections[:, -1].cpu().unique()  # 返回参数数组中所有不同的值,并按照从小到大排序可选参数
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)  # 为每一类分配一个边界框颜色

            # 遍历图片对应检测结果的每一个边界框
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:  # 检测结果为左上和右下坐标
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                # 边界框宽和高
                box_w = x2 - x1
                box_h = y2 - y1
                # 将边界框写入图片中，并设置颜色
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # 创建一个矩形边界框
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # 吧矩形边界框写入画布
                ax.add_patch(bbox)
                # 为检测边界框添加类别信息
                plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top",
                         bbox={"color": color, "pad": 0})

        # 将绘制好边界框的图片保存
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
