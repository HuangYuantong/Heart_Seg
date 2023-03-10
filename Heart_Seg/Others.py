import pickle
import math
import matplotlib.pyplot as plt
import numpy
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def Show_Image(image_list):
    """显示多张图片"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文
    length = len(image_list)
    # column = 3  # 列数
    column = math.ceil(pow(length, 0.5))  # 列数，向上取整
    row = length // column if length % column == 0 else length // column + 1  # 行数
    for i in range(length):
        plt.subplot(row, column, i + 1)  # 布局（行、列、序号）
        plt.imshow(image_list[i])  # imshow()对图像进行处理，画出图像，show()进行图像显示
        plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()


def LoaderTest(dataset):
    dataloader = DataLoader(dataset, 3, shuffle=False)
    for images, labels, classes in dataloader:
        # print(train_labels[0].numpy().transpose(1, 2, 0)[100])
        print(images.shape, labels.shape, classes.shape)
        mix = torch.vstack([(images[i] * labels[i] / 3).unsqueeze(0) for i in range(len(images))])
        image_list = []
        [image_list.extend([images[i].numpy().transpose(1, 2, 0), mix[i].numpy().transpose(1, 2, 0),
                            labels[i].unsqueeze(0).numpy().transpose(1, 2, 0)]) for i in range(len(images))]
        Show_Image(image_list)
        pass


def DrawLoss(loss):
    train_loss_list = numpy.array([i.to('cpu') for i in loss])
    plt.plot(train_loss_list, label="train_data loss")
    plt.title("Change of loss")
    plt.savefig(fname="data/outputs/损失变化.png")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # loss_list = pickle.load(open('data/outputs/loss_list.pickle', 'rb'))
    # DrawLoss(loss_list)
    image = Image.open('data/Image/DCM01image10.png')
    # image = Image.open('data/Image/HCM12image7.png')  # 亮度最低的
    # 图像预处理
    transform = transforms.Compose([
        # 数据增强
        # transforms.RandomEqualize(1),  # 有待商榷
        # transforms.ColorJitter((2, 2), contrast=(2, 2)),  # 未实验
        # 共同
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomPerspective(0.5),  # 随机倾斜固定角度
        transforms.RandomAffine((0, 135), (0.125, 0.125), (0.6, 1.2)),  # 旋转(0,135) 平移(32/256,32/256) 缩放(0.6,1.2)
        # image
        transforms.ColorJitter(brightness=(0.15, 1.2), contrast=(0.2, 1.8)),  # 原数据集范围:亮度(0.15,1.2) 对比度(0.2,1.8) 饱和度-色调-
        # transforms.RandomAdjustSharpness(0),
    ])
    # 使int64 float等图像可以正常在matplotlib显示黑白图像
    transform_show = transforms.Compose([transforms.PILToTensor(),
                                         transforms.Lambda(lambda y: y.to(torch.int)),
                                         transforms.ToPILImage(),
                                         ])
    image_list = list(transform_show(transform(image)) for _ in range(15))
    image_list.insert(0, transform_show(transforms.RandomEqualize(1)(image)))
    Show_Image(image_list)
