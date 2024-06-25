from __future__ import division
import tqdm
import torch
import numpy as np
def to_cpu(tensor):
    return tensor.detach().cpu()
'''加载数据集类别信息：返回类别组成的列表'''
def load_classes(path):#参数为类别名称文件的路径。例coco.names或classes.names的路径
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]#将文件的每一行数据存入列表，这使得数据集的每个类别的名称存入到一个列表
    return names#返回类别名称构成的列表
'''权重初始化函数'''
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:#卷积层权重初始化设置
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:#批量归一化层权重初始化设置
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
'''改变预测边界框的尺寸函数：参数为，边界框、当前的图片尺寸（标量）、原始图片尺寸。因为网络预测的边界框信息是，
对图像填充、调整大小后的图片进行预测的结果，因此需要对预测的边界框进行调整使其适应于原图的目标'''
def rescale_boxes(boxes, current_dim, original_shape):
    #原始图片的高和宽
    orig_h, orig_w = original_shape

    #原始图片的填充信息：根据原图的宽高的差值来计算。
    #pad_x为宽天长的像素数量， pad_y为高填充的像素数量
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))# 原图的高大于宽。改变后图片的大小/原图的最长边的尺寸=缩放比率
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    #将预测的边界框信息，调整为适应于原图
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # 改变预测边界框的尺寸，使其是适用于原图片
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w#左上x的坐标
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h#左上y的坐标
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes#返回调整后的预测边界框的信息/
'''将边界框信息转换为左上右下坐标表示函数'''
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
"""度量计算：参数为true_positive（值为0或1,list）、预测置信度(list)，预测类别(list)，真实类别(list)
返回：p, r, ap, f1, unique_classes.astype("int32")"""
def ap_per_class(tp, conf, pred_cls, target_cls):#参数：true_positives, pred_scores, pred_labels 、图片真实标签信息

    # 按照置信度排序，后的tp, conf, pred_cls
    i = np.argsort(-conf)
    #print('所有预测框的个数为',len(i))
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]#按照置信度排序后的tp(值为0,1), conf, pred_cls
    #print('tp[i]',tp[i])

    # 获取图片中真实框所包含的类别（类别不重复）
    unique_classes = np.unique(target_cls)
    #print('unique_classes',unique_classes)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):#为每一个类别计算AP

        # i:对于所有预测边界框的类pred_cls，判断与当前c类是否相同，相同则该位置为true否则为false,得到与pred_class形状相同的布尔列表
        i = pred_cls == c

        # ground truth 中类别为c的数量
        n_gt = (target_cls == c).sum()

        #预测边界框中类别为c的数量
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # 计算FP和TP
            fpc = (1 - tp[i]).cumsum()#i列表记录着索引对应位置是否是c类别的边界框，tp记录着索引对应位置是否是正例框
            tpc = (tp[i]).cumsum()
            # print('tp[i]',tp[i],len(tp[i]))#tp[i]是所有框中类别为c的预测框的true_positive信息（值为0或1，1代表与真值框iou大于阈值）
            # print('fpc',fpc,len(fpc))#fpc为类别为c的预测框中为正例的预测框
            # print('tpc', tpc,len(tpc))#tpc为类别为c的预测框中为负例的预测框

            #计算召回率
            recall_curve = tpc / (n_gt + 1e-16)
            #print('recall_curve',recall_curve)
            r.append(recall_curve[-1])
            #print('r',r)

            #计算精度
            precision_curve = tpc / (tpc + fpc)
            #print('precision_curve',precision_curve)
            p.append(precision_curve[-1])
            #print('p',p)

            # 计算AP：AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype("int32")
"""计算AP"""
def compute_ap(recall, precision):#参数精度和召回率
    # correct AP calculation
    # 给Precision-Recall曲线添加头尾
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    # 简单的应用了一下动态规划，实现在recall=x时，precision的数值是recall=[x, 1]范围内的最大precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # 寻找recall[i]!=recall[i+1]的所有位置，即recall发生改变的位置，方便计算PR曲线下的面积，即AP
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    # 用积分法求PR曲线下的面积，即AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
'''统计信息计算：参数，模型预测输出（NMS处理后的结果），真实标签（适应于原图的x,y,x,y）,iou阈值。
返回，true_positive（值为0/1，如果预测边界框与真实边界框重叠度大则值为1，否则为0）,预测置信度，预测类别'''
def get_batch_statistics(outputs, targets, iou_threshold):
    # outputs为非极大值抑制后的结果(x,y,x,y,object_confs,class_confs,class_preds)长度为7
    batch_metrics = []
    for sample_i in range(len(outputs)):#遍历每个output的边界框，因为是批量操作的，每个批量有很多图片，每个图片对应一个output,所以遍历每个output
        if outputs[sample_i] is None:
            continue
        '''图片的预测信息：'''
        output = outputs[sample_i]#取第sample_i个output信息，每个output里面包含很多边界框
        pred_boxes = output[:, :4]#预测边界框的坐标信息
        pred_scores = output[:, 4]#预测边界框的置信度
        pred_labels = output[:, -1]#预测边界框的类别

        true_positives = np.zeros(pred_boxes.shape[0])#true_positive的长度为pre_boxes的个数

        '''图片的标注信息（groundtruth）：'''
        #坐标信息，格式为（xyxy）
        annotations = targets[targets[:, 0] == sample_i][:, 1:]#这句把对应ID下的target和图像进行匹配，dataset.py里的ListDataset类里的collate_fn函数给target赋予ID
        #类别信息
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []#创建空列表
            target_boxes = annotations[:, 1:]#真实边界框（groundtruth）坐标
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):#遍历预测框：坐标和类别
                if len(detected_boxes) == len(annotations):
                    break
                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # 计算预测框和真实框的IOU
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                #如果预测框和真实框的IOU大于阈值，那么可以认为该预测边界框预测’正确‘，并把该边界框的true_positives值设置为1
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics#true_positive,预测置信度，预测类别
"""未用到"""
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area
"""计算两个边界框的IOU值"""
def bbox_iou(box1, box2, x1y1x2y2=True):

    #获取边界框的左上右下坐标值
    if not x1y1x2y2:
        #如果边界框的表示方式为（center_x,center_y,width,height）则转换表示格式为（x，y，x，y）
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]#box1的左上右下坐标
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]#box1的左上右下坐标

    #相交矩形的左上右下坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 相交矩形的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    #并集的面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou#返回重叠度IOU的值
'''非极大值抑制函数：返回边界框【x1,y1,x2,y2,conf,class_conf,class_pred】，参数为，模型预测，置信度阈值，nms阈值'''
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    """（1）模型预测坐标格式转变： (center x, center y, width, height) to (x1, y1, x2, y2)"""
    #三个yolo层,有三个尺寸的输出分别为13,26,52，所以对于一张图片，
    # 模型输出的shape是(10647,85),(13*13+26*26+52*52)*3=10647,后面的85是(x,y,w,h, conf, cls) xywh加一个置信度加80个分类。
    #prediction的形状为[1, 10647, 85]，85的前4个信息为坐标信息（center x, center y, width, height）
    # 第5个信息为目标置信度，第6-85的信息为80个类的置信度
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])# 将模型预测的坐标信息由(center x, center y, width, height) 格式转变为 (x1, y1, x2, y2)格式
    output = [None for _ in range(len(prediction))]

    #遍历每个图片，每张图片的预测image_pred：
    for image_i, image_pred in enumerate(prediction):#遍历预测边界框
        """（2）边界框筛选：去除目标置信度低于阈值的边界框"""
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]#筛选每幅图片预测边界框中目标置信度大于阈值的边界框
        # If none are remaining => process next image
        if not image_pred.size(0):#判断本图片经过目标置信度阈值的赛选是否还存在边界框，如果没有边界框则执行下一个图片的NMS
            continue

        """(3)非极大值抑制：根据score进行排序得到最大值，找到和这个score最大的预测类别相同的计算iou值，通过加权计算，得到最终的预测框(xyxy),最后从prediction中去掉iou大于设置的iou阈值的边界框。"""
        # 分数=目标置信度*80个类别得分的最大值。
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # 根据score为图片中的预测边界框进行排序
        image_pred = image_pred[(-score).argsort()]#形状【经过置信度阈值筛选后的边界框数量，85】
        #类别置信度最大值和类别置信度最大值所在位置（索引，也就是预测的类别）
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)#
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)#(x,y,x,y,object_confs,class_confs,class_preds)长度为7

        keep_boxes = []
        while detections.size(0):
            # 将当前第一个边界框（当前分数最高的边界框）与剩余边界框计算IoU，并且大于NMS阈值的边界框
            #第一个bbx与其余bbx的iou大于nms_thres的判别(0, 1)， 1为大于，0为小于
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres

            # 判断他们的类别是否相同，只有相同时才进行nms， 相同时为1， 不同时为0
            label_match = detections[0, -1] == detections[:, -1]

            # invalid 为Indices of boxes with lower confidence scores, large IOUs and matching labels
            # 只有在两个bbx的iou大于thres，且类别相同时，invalid为True，其余为False
            invalid = large_overlap & label_match
            # weights为对应的权值, 其格式为：将True bbx中的confidence连成一个Tensor
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            # 这里得到最后的bbx它是跟他满足IOU大于threshold，并且相同label的一些bbx，根据confidence重新加权得到
            # 并不是原始bbx的保留。
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            ## 去掉这些invalid，即iou大于阈值且预测同一类
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output#返回NMS后的边界框(x,y,x,y,object_confs,class_confs,class_preds)长度为7、

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
