import os
import pickle
import numpy
from tqdm import tqdm
from sklearn.metrics import f1_score
from Model.loss_dice import dice_loss

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from Load import Test_Dataset

score_method = 'f1'
image_dir = 'test'
save_dir = 'data/predicts'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

if __name__ == '__main__':
    for i in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, i))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # model
    net_dir = pickle.load(open('data/outputs/loss_min.pickle', 'rb')) if os.path.exists('data/outputs/loss_min.pickle') \
        else 'data/outputs/UNet3_epoch60_loss19.83897.ckpt'  # 0.90438, 0.859
    # mode: 'group'
    # 300 epoch CenterCrop, RandomSharpness
    # net_dir = 'data/outputs/UNet3_epoch150_loss8.13529.ckpt'  # 0.8512, 0.7718
    # net_dir = 'data/outputs/UNet3_epoch200_loss8.05117.ckpt'  # 0.8712, 0.7923
    # net_dir = 'data/outputs/UNet3_epoch300_loss7.95909.ckpt'  # 0.8965, 0.8272
    # 600 epoch Resize, RandomSharpness, RandomAffine(scale=(0.5, 1.5)), GaussianBlur(kernel_size=3)
    # net_dir = 'data/outputs/UNet3_epoch300_loss8.20730.ckpt'  # 0.822, 0.7682
    # net_dir = 'data/outputs/UNet3_epoch400_loss8.10881.ckpt'  # 0.8642, 0.8106
    # net_dir = 'data/outputs/UNet3_epoch500_loss8.02973.ckpt'  # 0.87674, 0.836386
    # net_dir = 'data/outputs/UNet3_epoch600_loss8.01949.ckpt'  # 0.89015, 0.836176

    # mode: 'single'
    # 70 epoch Resize, RandomSharpness, RandomAffine(scale=(0.5, 1.5)), GaussianBlur(kernel_size=3)
    # net_dir = 'data/outputs/UNet3_epoch10_loss81.66241.ckpt'  # 0.7187, 0.67776
    # net_dir = 'data/outputs/UNet3_epoch70_loss71.94922.ckpt'  # 0.890, 0.8369
    # 80 epoch Resize, trans, deep
    # net_dir = 'data/outputs/UNet3_epoch50_loss72.48322.ckpt'  # 0.883, 0.8251
    # net_dir = 'data/outputs/UNet3_epoch60_loss72.35831.ckpt'  # 0.89273, 0.8315
    # net_dir = 'data/outputs/UNet3_epoch70_loss71.71933.ckpt'  # 0.8994, 0.843955 <--classify best
    # net_dir = 'data/outputs/UNet3_epoch80_loss71.65264.ckpt'  # 0.8926, 0.838766
    # 80 epoch Resize, trans, deep, CE weight=[0.6, 1, 1, 1]
    # net_dir = 'data/outputs/UNet3_epoch40_loss21.45197.ckpt'  # 0.8796, 0.81069
    # net_dir = 'data/outputs/UNet3_epoch50_loss20.94285.ckpt'  # 0.8918, 0.851679
    # net_dir = 'data/outputs/UNet3_epoch60_loss19.83897.ckpt'  # 0.90438, 0.859   <--best
    # net_dir = 'data/outputs/UNet3_epoch80_loss19.56409.ckpt'  # 0.9035, 0.8518468

    print(net_dir)
    net = torch.load(net_dir).to(device)
    # dataloader
    dataset = Test_Dataset(image_dir)
    dataloader = DataLoader(dataset, 15, False)

    with torch.set_grad_enabled(False):
        net.eval()
        score_list = []
        for images, labels, classes, sizes, paths in tqdm(dataloader, '标签预测中'):
            images = images.to(device)
            predicts = net(images)
            _predicts = torch.argmax(predicts[0], dim=1).to(device='cpu', dtype=torch.int)
            for idx, predict in enumerate(_predicts):
                # 如果最后一层输出为全黑，那么尝试使用深监督的输出
                # 不过这没有用，倒数第二层的输出与最后一层输出差不多
                # temp = transforms.PILToTensor()(Image.open('data/predicts/Image_HCM15image6.png'))  # 62
                # print(torch.count_nonzero(temp))
                # if torch.count_nonzero(predict) < 50:
                #     # 中心裁剪部分，放大后重新输入计算
                #     temp = transforms.Resize((256, 256))((transforms.CenterCrop((160, 160))(image[idx])))
                #     predict = net(temp.unsqueeze(0))[0]
                #     predict = torch.argmax(predict, dim=1).to(device='cpu', dtype=torch.int)
                #     # 缩小后还原到原尺寸
                #     predict = transforms.CenterCrop((256, 256))(transforms.Resize((160, 160))(predict))

                # 恢复原尺寸
                # 如果Train_Dataset使用的是Resize
                temp_transform = transforms.Resize(tuple(sizes[idx].numpy()), InterpolationMode.NEAREST)
                predict = temp_transform(predict.unsqueeze(0)).squeeze(0)
                label = temp_transform(labels[idx].unsqueeze(0)).squeeze(0).to(dtype=torch.int)
                # 如果Train_Dataset使用的是CenterCrop
                # predict = transforms.CenterCrop(size[idx].numpy())(predict)
                # label = transforms.CenterCrop(size[idx].numpy())(labels[idx]).to(dtype=torch.int)
                # 计算score（f1_score多分类只能接入一维）
                score = dice_loss(predict, label, True) if score_method == 'dice' else \
                    f1_score(label.reshape(-1, 1).numpy(), predict.reshape(-1, 1).numpy(), average='macro')
                score_list.append(score)

                # 将类别标签转回像素值
                predict[predict == 1], predict[predict == 2], predict[predict == 3] = 85, 170, 255
                # 转为Image并保存
                temp = paths[idx].split(os.path.sep)
                name = temp[1] + temp[-2] + temp[-1]  # 命名
                predict = transforms.ToPILImage()(predict.to(dtype=torch.uint8))  # 指定为uint8才存储8位黑白图像
                predict.save(os.path.join(save_dir, name))

        # 得分与画图
        from matplotlib import pyplot as plt

        print(f'max {score_method}：{max(score_list)}，min {score_method}:{min(score_list)}\n'
              f'平均{score_method}：{numpy.mean(score_list)}\n')
        plt.scatter(list(range(len(score_list))), score_list, s=5)
        plt.title(f'{score_method} Score for each image, mean: {numpy.mean(score_list):.4f}')
        plt.savefig(fname=f'data/outputs/{score_method}得分分布-{image_dir}.png')
        plt.legend()
        plt.show()
