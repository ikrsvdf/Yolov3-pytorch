""" """
'''包含两个解析器：
  1.模型配置解析器：返回一个列表model_defs，列表的每一个元素为一个字典，字典代表模型某一个层（模块）的信息 。
  2.数据配置解析器：返回一个字典，每一个键值对描述了，数据的名称路径，或其他信息。

   模型配置解析器：解析yolo-v3层配置文件函数，并返回模块定义module_defs，path就是yolov3.cfg路径
'''

'''
   模型配置解析器：解析yolo-v3层配置文件函数，并返回模块定义module_defs，path就是yolov3.cfg路径
'''
def parse_model_config(path):
    '''
    看此函数，一定要先看config文件夹下的yolov3.cfg文件，如下是yolov3。cfg的一部分内容展示：
    [convolutional]
    batch_normalize=1
    filters=32
    size=3
    stride=1
    pad=1
    activation=leaky
    # Downsample
    [convolutional]
    batch_normalize=1
    filters=64
    size=3
    stride=2
    pad=1
    activation=leaky
    。。。
    :param path: 模型配置文件路径，yolov3.cfg的路径
    :return: 模型定义，列表类型，列表中的元素是字典，字典包含了每一个模块的定义参数
    '''

    # 打开yolov3.cfg文件,并将文件内容存入列表，列表的每一个元素为文件的一行数据。
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')  ]  # 不读取注释
    lines = [x.rstrip().lstrip() for x in lines]  # 去除边缘空白

    # 定义一个列表modle_defs
    module_defs = []
    # 读取cfg的每一行内容：
    #   1.如果该行内容以[开头:代表是模型的一个新块的开始，给module_defs列表新增一个字典
    #   字典的‘type’=[]内的内容，如果[]内的内容是convolution,则字典添加'batch_normalize'：0
    #   2.如果该行内容不以[开头，代表是块的具体内容
    #   等号前的值为字典的key，等号后的值为字典的value
    for line in lines  :  # 读取yolov3.cfg文件的每一行

        # 如果一行内容以[开头说明是一个模型的开始,[]里的内容是模块的名称，如[convolutional][convolutional][shortcut]。。。。
        if line.startswith('['): # This marks the start of a new block
            # 将一个空字典添加到模型定义module_defs列表中
            module_defs.append({})
            # 给该字典内容赋值：例{’type‘：’convolutional‘}
            module_defs[-1]['type'] = line[1:-1].rstrip()
            # 如果当前的模块是convolutional模块，给字典的内容赋值：{’type‘：’convolutional‘，'batch_normalize'：0}
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0

        # 如果一行内容不以[开头说明是模块里的具体内容
        else:
            key, value = line.split("=")
            value = value.strip(  )  # strip()删除头尾空格，rstrip()删除结尾空格
            # 将该行内容添加到字典中，key为等式左边的内容，value为等式右边的内容
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs  # 模型定义，是一个列表，列表每一个元素为一个字典，字典包含一个模块的具体信息

'''数据配置解析器：参数path为配置文件的路径'''
def parse_data_config(path):
    """
    数据配置包含的信息：
    classes= 80
    train=data/coco/trainvalno5k.txt
    valid=data/coco/5k.txt
    names=data/coco.names
    backup=backup/
    eval=coco
    """

    # 创建一个字典
    options = dict()

    # 为字典添加元素
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'

    # 读取数据配置文件的每一行，并将每一行的信息以键值对的形式存入字典中
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options  # 返回一个字典，字典的key为名称（train，valid,names..），value为路径或其他信息
