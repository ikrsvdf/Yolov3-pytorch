from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

"""模型评估函数：参数为模型、valid数据集路径、iou阈值。nms阈值、网络输入大小、批量大小"""
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    #加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值
    model.eval()

    '''(1)获取评估数据集：变为batch组成的数据集'''
    # dataset（验证集图片路径集、验证集图片集，验证集标签集）
    # dataloader获取批量batch,验证集图片路径batch、验证集图片batch，验证集标签batch）
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=1,
                                             collate_fn=dataset.collate_fn)#collate_fn参数，实现自定义的batch输出
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):#tqdm进度条
        '''(2) batch标签处理'''
        labels += targets[:, 1].tolist()#将targets的类别信息转变为list存到label列表中
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])#将targets的坐标变为（xyxy）形式，此时的坐标也是归一化的形式
        targets[:, 2:] *= img_size#适应于原图的比target形式

        '''(3)batch图片预测，并进行NMS处理'''
        # 图片输入模型，并对模型输出进行非极大值抑制
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        '''（4）预测信息统计：得到经过NMS处理后，预测边界框的true_positive（值为或1）、预测置信度，预测类别信息'''
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)#参数：模型输出，真实标签（适应于原图的x,y,x,y）,iou阈值

    # 这里需要注意,github上面的代码有错误,需要添加if条件语句，训练才能正常运行
    if len(sample_metrics) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # sample_metrics信息解析，获取独立的 true_positive（值为或1）、预测置信度，预测类别  信息
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    #计算 precision, recall, AP, f1, ap_class，这里调用了utils.py中的函数进行计算
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)#pred_labels, labels的长度是不同的
    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    '''(1)参数解析'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_9.pth", help="path to weights file")#"weights/yolov3.weights"
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    #print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """（2）数据解析"""
    # 调用parse_config。py中的数据解析桉树，返回值 data_config 为字典{class:80,train:路径，valid:路径。。。}
    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]#验证集路径valid=data/custom/valid.txt
    class_names = load_classes(data_config["names"])#类别路径

    """（3）模型构建：构建模型，加载模型参数"""
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)#
    else:
        model.load_state_dict(torch.load(opt.weights_path))#自定义的函数

    print("Compute mAP...")

    """(4)模型评估"""
    precision, recall, AP, f1, ap_class = evaluate(
        model,#模型
        path=valid_path,#验证集路径
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,#置信度阈值
        nms_thres=opt.nms_thres,#nms阈值
        img_size=opt.img_size,#网路输入尺寸
        batch_size=8,#批量
    )
    print(precision, recall, AP, f1, ap_class)
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
