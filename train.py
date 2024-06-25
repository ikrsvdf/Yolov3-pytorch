from __future__ import division
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from terminaltables import AsciiTable
import os
from test import evaluate
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

"""
--data_config config/coco.data  
--pretrained_weights weights/darknet53.conv.74
"""
if __name__ == "__main__":
    '''（1）参数解析'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    # 梯度累加数
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    # parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74",
                        help="if specified starts from checkpoint model") # 模型迁移，使用预训练模型参数darknet53.conv.74
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_9.pth",
                        help="path to weights file")
    opt = parser.parse_args()
    print(opt)
    '''(2)实例化日志类'''
    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''(3)文件夹创建'''
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    """(4)初始化模型：模型构建，模型参数装载"""
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    """(5)数据集加载"""
    data_config = parse_data_config(opt.data_config)  # 调用parse_config.py文件的数据配置解析函数，获取data_config为一个字典
    train_path = data_config["train"]  # 训练集路径
    valid_path = data_config["valid"]  # 验证集路径
    class_names = load_classes(data_config["names"])  # 调用utils.py内的load_classes函数用于获取数据集包含的类别名称

    # dataset是数据集中，图片的路径和、图片、标签（归一化的格式x,y,w,h）的集合
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    # dataloader是dataset装载成批量形式
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    """(7)优化器"""
    optimizer = torch.optim.Adam(model.parameters())

    """(8)模型训练"""
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    for epoch in range(opt.epochs):  # 迭代epoch次训练

        model.train()  # 设置模型为训练模式
        start_time = time.time()
        print('start_time', start_time)

        for batch_i, (_, imgs, targets) in enumerate(dataloader):  # 每一epoch的批量迭代

            # 批量的累计迭代数
            batches_done = len(dataloader) * epoch + batch_i

            # 图片、标签的变量化处理
            imgs = Variable(imgs.to(device))  # 把图像变为变量，可以记录梯度
            targets = Variable(targets.to(device), requires_grad=False)  # 把标签变为变量，不记录梯度

            # 获取模型的输出与损失，损失反向传播
            loss, outputs = model(imgs, targets)  # 将图片和标签输入模型，获取输出
            loss.backward()

            # 计算梯度
            if batches_done % opt.gradient_accumulations:
                # 在每一步之前计算梯度Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # 训练的epoch及batch信息
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch + 1, opt.epochs, batch_i + 1, len(dataloader))
            # print('log_str',log_str)#例---- [Epoch 1/10, Batch 1/10] ----

            # 创建行索引
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]  # 创建训练过程中的表格，行索引
            # print(metric_table)# [['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2']]

            # 在每一个 YOLO layer的各项指标信息
            for i, metric in enumerate(metrics):  # metrics为各项指标名称组成的列表，上面已经定义
                # 获取metrics各个项的数值类型
                formats = {m: "%.6f" for m in metrics}  # 将所有的metrics中的输出数值类型定义，这一步把全部的输出类型全部定义保留6位小数
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                # print(' formats', formats)#{'grid_size': '%2d', 'loss': '%.6f', 'x': '%.6f', 'y': '%.6f', 'w': '%.6f', 'h': '%.6f', 'conf': '%.6f', 'cls': '%.6f', 'cls_acc': '%.2f%%', 'recall50': '%.6f', 'recall75': '%.6f', 'precision': '%.6f', 'conf_obj': '%.6f', 'conf_noobj': '%.6f'}

                # 表格赋值
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in
                               model.yolo_layers]  # ？？？？？？？？？？？？？
                # print('row_metrics',row_metrics)
                metric_table += [[metric, *row_metrics]]

                # Tensorboard 日志信息
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]  # 把除grid_size的其余信息，添加到日志中
                tensorboard_log += [("loss", loss.item())]  # 把损失也添加到日志信息中
                # 把日志信息列表写入创建的日志对象
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str打印各项指标参数：
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # 计算该epoch剩余需要的大概时间
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)
            model.seen += imgs.size(0)
        '''(9)训练时评估'''
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # 在评估数据集上对当前模型进行评估，具体评估细节可以看test.py
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        '''(10)模型保存'''
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
