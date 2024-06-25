import torch
import torch.nn.functional as F
import numpy as np

"""
进行数据增强的文件，本项目只是进行水平翻转的数据增强，图像进行翻转的时候，
对应标注信息也进行了修改，最终返回的是翻转后的图片和翻转后的图片对应的标签。

horisontal_flip(images, targets)
输入：image,targets 是原始图像和标签；
返回：images，targets是翻转后的图像和标签。
功能：horisontal_flip() 函数是对图像进行数据增强，使得数据集得到扩充。在此处只采用了对图片进行水平方向上的镜像翻转。

torch.flip(input,dims) ->tensor
功能：对数组进行反转
参数: input 反转的tensor ; dim 反转的维度
返回： 反转后的tensor
由于image 是用数组存储起来的（c,h,w）,三个维度分别代表颜色通道，垂直方向，水平方向。python 中[-1] 代表最后一个数，即水平方向。
targets是对应的标签[置信度，中心点高度，中心点宽度，框高度，框宽度], 其中高度宽度都是用相对位置表示的，范围是[0,1]。

"""
def horisontal_flip(images, targets): #对图像和标签进行镜像翻转
    '''
    torch.flip(input,dims) ->tensor
    功能：对数组进行反转
    参数: input 反转的tensor ; dim 反转的维度
    返回： 反转后的tensor
    由于image 是用数组存储起来的（c,h,w）,三个维度分别代表颜色通道，垂直方向，水平方向。python 中[-1] 代表最后一个数，即水平方向。
    targets是对应的标签[置信度，中心点高度，中心点宽度，框高度，框宽度], 其中高度宽度都是用相对位置表示的，范围是[0,1]。
    '''

    images = torch.flip(images, [-1]) #镜像翻转
    targets[:, 2] = 1 - targets[:, 2]
    # targets是对应的标签[置信度，中心点高度，中心点宽度，框高度，框宽度]
    # 镜像翻转时，受影响的只有targets[:, 2],
    return images, targets
