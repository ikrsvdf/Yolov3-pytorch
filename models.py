from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.parse_config import *
from utils.utils import build_targets, to_cpu

'''构建网络函数：通过获取的模型定义module_defs来构建YOLOv3模型结构,根据module_defs中的模块配置构造层块的模块列表'''
def create_modules(module_defs):

    '''构建模型结构'''
    '''（1）解析模型超参数，获取模型的输入通道数'''
    #从model_def获取net的配置信息组成的字典hyperparams。model_def是由parse_config函数解析出来的列表，每个元素为一个字典，每一个字典包含了某层、模块的参数信息
    hyperparams = module_defs.pop(0)#hyperparams为module_defs的第一个字典元素，是模型的超参数信息{'type': 'net',...}
    output_filters = [int(hyperparams["channels"])]

    '''(2)构建nn.ModuleList()，用来存放创建的网络层、模块'''
    module_list = nn.ModuleList()

    '''(3)遍历模型定义列表的每个字典元素，创建相应的层、模块，添加到nn.ModuleList()中'''
    #遍历 module_defs的每个字典，根据字典内容，创建相应的层或模块。其中字典的type的值有一下几种："convolutional"，"maxpool"
    #"upsample"， "route"，"shortcut"， "yolo"
    for module_i, module_def in enumerate(module_defs):
        #创建一个 nn.Sequential()
        modules = nn.Sequential()

        #卷积层构建，并添加到nn.Sequential()
        if module_def["type"] == "convolutional":
            #获取convolutional层的参数信息
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            #创建convolution层：根据convolutional层的参数信息，创建convolutional层，并将改层加入到nn.Sequential()中
            modules.add_module(f"conv_{module_i}",#层在模型中的名字
                nn.Conv2d(#层
                    in_channels=output_filters[-1],#输入的通道数
                    out_channels=filters,#输出的通道数
                    kernel_size=kernel_size,#卷结核大小
                    stride=int(module_def["stride"]),#步长
                    padding=pad,#填充
                    bias=not bn,
                ),
            )
            if bn:
                #添加BatchNorm2d层
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                #添加激活层LeakyReLU
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        #池化层构建，并添加到nn.Sequential()
        elif module_def["type"] == "maxpool":
            # 获取maxpool层的参数信息
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            # 根据maxpool层的参数信息，创建maxpool层，并将改层加入到 nn.Sequential()中
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            #创建maxpool层
            modules.add_module(f"maxpool_{module_i}",
                               nn.MaxPool2d(
                                   kernel_size=kernel_size, #卷积核大小
                                   stride=stride, #步长
                                   padding=int((kernel_size - 1) // 2))#填充
                               )

        #上采样层构建，并添加到nn.Sequential()
        #上采样层是自定义的层，需要实例化Upsample为一个对象，将对象层添加到模型列表中
        elif module_def["type"] == "upsample":
            #上采样的配置例，如下
            # [upsample]
            # stride = 2

            # 构建upsample层，上采样层类，重写了forward函数
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            #层添加到模型
            modules.add_module(f"upsample_{module_i}", upsample)


        elif module_def["type"] == "route":
            #youte信息，例
            # [route]
            # layers = -1, 36

            # 获取route层的参数信息
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())#EmptyLayer()为“路线”和“快捷方式”层的占位符

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())#EmptyLayer()为“路线”和“快捷方式”层的占位符

        elif module_def["type"] == "yolo":
            #例：假设yolo的配置信息如下
            # [yolo]
            # mask = 3,4,5
            # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            # classes=80
            # num=9
            # jitter=.3
            # ignore_thresh = .7
            # truth_thresh = 1
            # random=1

            #获取anchor的索引，上例为3,4,5
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]

            #提取anchor尺寸信息,放入列表
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            #print('anchors1:', anchors)#上例为anchors1: [(30, 61), (62, 45), (59, 119)]

            #获取图片的输入尺寸
            img_size = int(hyperparams["height"])

            #定义yolo检测层：实例化yolo类，创建yolo层，传入的参数为三个anchor的尺寸，类别的数量，图像的大小
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)

            #将YOLO层加入到模型列表
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules) #将创建的nn.Sequential()即创建的层，添加到 nn.ModuleList()中
        output_filters.append(filters)#将创建的层的输出通道数添加到filters列表中，作为下次创建层的输入通道数

    return hyperparams, module_list#返回网络的参数、网络结构即层组成的列表

'''上采样层'''
class Upsample(nn.Module):
    """ nn.Upsample 被重写 """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor#上采样步长
        self.mode = mode
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)#上采样方法，插值
        return x#返回上采样结果

'''emptylayer定义'''
class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

'''yolo层定义：检测层'''
class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim=416):#参数为三个anchor的尺寸，类别的数量，图像的大小
        super(YOLOLayer, self).__init__()
        #基础设置
        self.anchors = anchors#anchor的尺寸信息，例某一层yolo尺寸为[(30, 61), (62, 45), (59, 119)]
        self.num_anchors = len(anchors)#anchor的数量
        self.num_classes = num_classes#类别的数量

        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
    #计算网格单元偏移
    def compute_grid_offsets(self, grid_size, cuda=True):

        #获取网格尺寸（几×几）
        self.grid_size = grid_size
        g = self.grid_size
        # print('g',g)  g可能的取值为13/26/52，对应不同yolo层的特征图的尺寸
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        #获取网格单元大小
        self.stride = self.img_dim / self.grid_size#网格单元的尺寸

        # Calculate offsets for each grid，假设g取13,
        #torch.arange(g)  为tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])
        #torch.arange(g).repeat(g, 1)  为由tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])组成的13行一列的张量
        #torch.arange(g).repeat(g, 1).view([1, 1, g, g])  改变视图为【1,1,13,13】
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)#
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)


        #把anchor的宽和高转变为相对于网格单元大小的度量
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])#例某一层yolo尺寸为[(30, 61), (62, 45), (59, 119)]
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))#获取anchor的宽
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))#获取anchor的高

    def forward(self, x, targets=None, img_dim=None):
        #yolo层的前向传播，参数为yolo层来自上层的输出作为输入x
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        #图片的大小
        self.img_dim = img_dim

        #获取x的形状
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)#（num_samples,3,85,gride_size,grid_size）
            .permute(0, 1, 3, 4, 2)#permute是用来做维度换位置的，（num_samples,3,gride_size,grid_size,85）
            .contiguous()#调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样。而不是与原数据公用一份内存。
        )
        # 得到outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

"""Darknet类：YOLOv3模型"""
class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        # parse_model_config（）模型配置的解析器:用来解析yolo-v3层配置文件(yolov3.cfg)并返回模块定义
        #（模型定义module_defs是一个列表，每一个元素是一个字典，该字典描绘了网络每一个模块/层的信息）
        self.module_defs = parse_model_config(config_path)

        #通过获取的模型定义module_defs，来构建YOLOv3模型
        self.hyperparams,self.module_list = create_modules(self.module_defs)#模型参数和模型结构
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
