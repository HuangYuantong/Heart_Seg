import os

import numpy
from PIL import Image, ImageFilter
from random import randint
import cv2

from Others import LoaderTest, Show_Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

image_size = 256
CLASSES_LIST = ['DCM', 'HCM', 'NOR']


class Train_Dataset(Dataset):
    """
    单张图片为单位时，返回(图片，分割目标，分类)\n
    一组图片为单位时，返回(该组随机一张图片, 相应分割目标, 分类)
    """

    def __init__(self, image_dir: str, mode='single'):
        # 从image_dir构建所有image完整路径、对应label完整路径、对应疾病分类
        self.images, self.labels, self.classes = Locate_All(image_dir, mode)
        self.mode = mode  # single、group
        # X、Y：形变，同时作用在图像、标签上，需要使二者的随机同步
        self.transform_share = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   # transforms.CenterCrop(image_size),
                                                   # 旋转(0,135) 平移(32/256,32/256) 缩放(0.6,1.2),
                                                   transforms.RandomAffine(degrees=(0, 15), scale=(0.5, 1.5), ),
                                                   # NEAREST to make sure label stays only 4 values
                                                   transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
                                                   ])
        # image：单通道黑白图像
        self.image_transform = transforms.Compose([transforms.RandomAdjustSharpness(0, p=0.2),
                                                   transforms.GaussianBlur(kernel_size=3),
                                                   transforms.ToTensor(), ])
        # label：单通道黑白图像，0、85、170、255
        self.label_transform = transforms.Compose([transforms.PILToTensor(),
                                                   transforms.Lambda(lambda y: y.to(dtype=torch.int64).squeeze()), ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = None, None
        if self.mode == 'single':
            # 打开相应image、label
            image = Image.open(self.images[idx])
            label = Image.open(self.labels[idx])
        elif self.mode == 'group':
            # 每个idx对应一组图片
            _images, _labels = self.images[idx], self.labels[idx]
            # 每次随机打开其中一张
            temp_idx = randint(0, len(_images) - 1)
            image = Image.open(_images[temp_idx])
            label = Image.open(_labels[temp_idx])
        else:
            exit('Train_Dataset中mode拼写错误')
        '''
        elif self.mode == 'group':
            # 每个idx对应一组图片
            _images, _labels = self.images[idx], self.labels[idx]
            # 图像变化
            images, labels = [], []
            for i in range(len(_images)):
                _image, _label = Image.open(_images[i]), Image.open(_labels[i])
                # 图像转化
                # X、Y共同
                # 在执行transform_share(x)、(y)前将seed置为相同
                seed = randint(0, 2147483647)
                torch.manual_seed(seed)
                _image = self.transform_share(_image)
                torch.manual_seed(seed)
                _label = self.transform_share(_label)
                # X/Y单独
                _image = self.image_transform(_image)
                _label = self.label_transform(labels)
                # 将颜色0、85、170、255分别转化为类0、1、2、3
                _label[_label == 85], _label[_label == 170], _label[_label == 255] = 1, 2, 3
                images.append(_image.unsqueeze(0))
                labels.append(_label.unsqueeze(0))
            images, labels = torch.vstack(images), torch.vstack(labels)
            return images, labels, self.classes[idx]
            '''
        # 图像增强
        image = Enhancement(image)
        # X、Y共同
        # 在执行transform_share(x)、(y)前将seed置为相同
        seed = randint(0, 2147483647)
        torch.manual_seed(seed)
        image = self.transform_share(image)
        torch.manual_seed(seed)
        label = self.transform_share(label)
        # X/Y单独
        image = self.image_transform(image)
        label = self.label_transform(label)
        # 将颜色0、85、170、255分别转化为类0、1、2、3
        label[label == 85], label[label == 170], label[label == 255] = 1, 2, 3
        return image, label, self.classes[idx]


class Test_Dataset(Dataset):
    """测试用Test_DataSet\n
    返回（图片，分割目标，分类, 宽高, 完整原路径）"""

    def __init__(self, image_dir: str):
        # 从image_dir构建所有image完整路径、对应label完整路径、对应疾病分类
        self.images, self.labels, self.classes = Locate_All(image_dir, 'single')
        # image：单通道黑白图像
        self.image_transform = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
                                                   # transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(), ])
        # label：单通道黑白图像，0、85、170、255
        self.label_transform = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
                                                   # transforms.CenterCrop(image_size),
                                                   transforms.PILToTensor(),
                                                   transforms.Lambda(lambda y: y.to(dtype=torch.int64).squeeze())])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 打开相应image、label
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        # 图像增强
        image = Enhancement(image)
        size = torch.tensor([image.height, image.width])  # 图片原尺寸（Image下width、height相反）
        path = self.images[idx]  # 图片完整原路径（…相对路径/文件名.后缀）
        # 图像变化
        image = self.image_transform(image)
        label = self.label_transform(label)
        # 将颜色0、85、170、255分别转化为类0、1、2、3
        label[label == 85], label[label == 170], label[label == 255] = 1, 2, 3
        return image, label, self.classes[idx], size, path


def Locate_All(image_dir: str, mode: str):
    """从image_dir构建所有图片的完整路径"""
    images, labels, classes = [], [], []
    # 对三类分别遍历
    for _classes in CLASSES_LIST:
        # 在windows中可_types in ['Image', 'Label']，判断_types添加到相应的images/labels即可
        # CG的Linux中，读取Image、Label文件中01、02…等子文件夹时的顺序不同
        # 因此只好使用字符串替换由image构建相应label的位置
        # <Linux就是garbage>
        for root, dirs, files in os.walk(os.path.join(image_dir, 'Image_' + _classes, 'png', 'Image')):
            if mode == 'single':
                for file in files:
                    name_in_Image = os.path.join(root, file)
                    name_in_Label = os.path.join(root[:-10] + root[-10:].replace('Image', 'Label'),
                                                 file.replace('image', 'label'))
                    images.append(name_in_Image)
                    labels.append(name_in_Label)
                    classes.append(CLASSES_LIST.index(_classes))  # 每张返回一个分类标签
            elif mode == 'group':
                # 每个子文件夹下图片视为同一组
                if not files: continue
                files = list(os.path.join(root, file) for file in files)
                images.append(files)
                # Linux只能使用字符串替换由image构建相应label的位置
                files = list(file[:-20] + file[-20:].replace('Image', 'Label').replace('image', 'label') for file in files)
                labels.append(files)
                classes.append(CLASSES_LIST.index(_classes))  # 每组返回一个分类标签
            else:
                exit('Locate_All函数中mode拼写错误')
    return images, labels, classes


def Prepare(image_dir: str, save_image_dir='data/Image', save_label_dir='data/Label'):
    """从image_dir读取所有图片，并分别保存到Image、Label文件夹中"""
    if not os.path.exists(save_image_dir): os.makedirs(save_image_dir)
    if not os.path.exists(save_label_dir): os.makedirs(save_label_dir)
    images, labels = [], []
    transform = transforms.CenterCrop(256)
    for classes in ['DCM', 'HCM', 'NOR']:
        for types in ['Image', 'Label']:
            for root, dirs, files in os.walk(os.path.join(image_dir, 'Image_' + classes, 'png', types)):
                for file in files:
                    # 对图像进行保存
                    image = Image.open(os.path.join(root, file))
                    image = transform(image) if transform else image
                    new_name = os.path.join(save_image_dir if types == 'Image' else save_label_dir,  # 路径
                                            classes + root[-2:] + file)  # 文件名
                    image.save(new_name)
                    # 返回新的路径
                    images.append(new_name) if types == 'Image' else labels.append(new_name)
    return images, labels


def Enhancement(image: Image):
    """图像增强"""
    # 直方图均衡化
    temp = numpy.array(image)
    temp = ((10 / temp.mean()) * temp).astype(numpy.uint8)
    temp = cv2.equalizeHist(temp)
    # cv2.imshow('1', temp)
    image = Image.fromarray(temp)
    # 边缘增强
    # image = image.filter(ImageFilter.EDGE_ENHANCE)
    # 锐化
    # image.filter(ImageFilter.SHARPEN)
    # 细节
    # image.filter(ImageFilter.DETAIL)
    # cv2.imshow('2', numpy.array(image))
    # cv2.waitKey()
    # exit()
    return image


if __name__ == '__main__':
    # Prepare('Heart Data')
    LoaderTest(Train_Dataset('train', mode='group'))
    # image = Image.open('data/Image/DCM01image10.png')
    # image = Image.open('data/Image/HCM12image7.png')  # 亮度最低的
    # Enhancement(image)
