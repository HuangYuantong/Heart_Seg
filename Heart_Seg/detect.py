import os
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset

from Load import Enhancement


image_size = 256
CLASSES_LIST = ['DCM', 'HCM', 'NOR']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Locate_All(image_dir: str):
    images = []
    for root, dirs, files in os.walk(image_dir):
        images.extend(list(os.path.join(root, file) for file in files))
    return images


class Test_Dataset(Dataset):
    """返回（image原图像, size宽高, path完整原路径）"""

    def __init__(self, image_dir: str):
        # 从image_dir构建所有image完整路径、对应label完整路径、对应疾病分类
        self.images = Locate_All(image_dir)
        # image：单通道黑白图像
        self.image_transform = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
                                                   transforms.ToTensor(), ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        # 图像增强
        image = Enhancement(image)
        size = torch.tensor([image.height, image.width])  # 图片原尺寸（Image下width、height相反）
        path = self.images[idx]  # 图片完整原路径（…相对路径/文件名.后缀）
        # 图像变化
        image = self.image_transform(image)
        return image, size, path


if __name__ == '__main__':
    # 加载
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    net_dir = 'data/outputs/Model_epoch40_loss0.02180.ckpt'  # 0.85295 / 0.9184, 0.7825
    model = torch.load(net_dir).to(device)
    # dataloader
    image_dir = input('数据集所在路径输入要求：\n'
                      '   1、路径不能以分隔符"/"结尾。\n'
                      '正确示例：\n'
                      '   (绝对路径) /headless/Desktop/cgshare/PracticalTraining2_submit/Image\n'
                      '   (相对路径) ../PracticalTraining2_submit/Image\n'
                      '==================================\n'
                      ' 请输入数据集所在路径：')
    if image_dir[-1] == os.sep: exit('路径输入错误，请重新输入！')
    save_dir = image_dir[:-len(image_dir.split(os.sep)[-1])] + 'Label-06'
    if not os.path.exists(image_dir):
        exit('路径输入错误，请重新输入！')
    dataset = Test_Dataset(image_dir)
    dataloader = DataLoader(dataset, 1, False)

    with torch.set_grad_enabled(False):
        model.eval()
        for images, sizes, paths in tqdm(dataloader, '标签预测中'):
            images = images.to(device)
            outputs_segment, outputs_classify = model(images)
            _predicts_classify = torch.argmax(outputs_classify, dim=1).to(device='cpu', dtype=torch.int).numpy()
            _predicts_segment = torch.argmax(outputs_segment[0], dim=1).to(device='cpu', dtype=torch.int)
            for idx, predict in enumerate(_predicts_segment):
                # 恢复原尺寸
                temp_transform = transforms.Resize(tuple(sizes[idx].numpy()), InterpolationMode.NEAREST)
                predict = temp_transform(predict.unsqueeze(0)).squeeze(0)
                # 将类别标签转回像素值，转为Image并保存
                predict[predict == 1], predict[predict == 2], predict[predict == 3] = 85, 170, 255
                predict = transforms.ToPILImage()(predict.to(dtype=torch.uint8))  # 指定为uint8才存储8位黑白图像
                # 路径
                # all dir after '/Image/'
                _temp = paths[idx].replace(image_dir, '')[1:]
                temp = _temp.split(os.sep)
                temp_dir = os.path.join(save_dir, _temp[:-len(temp[-1]) - 1])
                if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                # 命名:类别+原文件名
                temp_1 = temp[-1].split('.')
                temp_name = temp_1[0].replace('image', 'label') + f'-{CLASSES_LIST[_predicts_classify[idx]]}.{temp_1[1]}'
                predict.save(os.path.join(temp_dir, temp_name))
    print(f'结果已生成在{save_dir}文件下')
