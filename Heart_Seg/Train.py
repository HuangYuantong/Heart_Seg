import os
from tqdm import tqdm
import pickle
import shutil

import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader

from Load import Train_Dataset
from Model.UNet_3Plus import UNet_3Plus, weight_init
from Model.loss_dice import dice_loss
from Model.loss_msssim import msssim

deep_supervise = True
mode = 'single'  # single: (图片，分割目标，分类)，group: (每组随机一张图片, 相应分割目标, 分类)

is_preTrain = False
preTrain = 'data/outputs/UNet3_epoch30_loss73.70703.ckpt'
image_dir = 'train'
loss_min_init = float('inf')

epochs = 80
batch_size = 5
learning_rate = 1e-5

# 模型等定义
############################################################################
if __name__ == '__main__':
    # 模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    net = torch.load(preTrain).to(device) if is_preTrain else UNet_3Plus(1, 4, deep_supervise).to(device)
    if not is_preTrain: net.apply(weight_init)
    # 优化器、调度器
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)  # 1e-5, 1e-8, 0.9
    # optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)  # 2
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
    # 损失
    crossEntropy_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1, 1, 1], device=device))
    # dataloader
    dataset = Train_Dataset(image_dir, mode=mode)
    dataloader = DataLoader(dataset, batch_size, True)


def Total_Loss(y, outputs_list, weight):
    if len(weight) != len(outputs_list): exit("Total_Loss: weight and outputs_list don't match!")
    total_loss = None
    _y = functional.one_hot(y, net.n_classes).permute(0, 3, 1, 2).float()
    for i in range(len(outputs_list)):
        _output = functional.softmax(outputs_list[i], dim=1)
        # loss = 分割交叉熵 + 分割dice_loss + 分割ms_ssim_loss
        loss = weight[i] * (
                crossEntropy_loss_fn(outputs_list[i], _y)  # CrossEntropyLoss的input需要为原始输出概率
                + dice_loss(_output, _y, multiclass=True)
                + 0.2 * msssim(_output, _y, normalize=True)
        )
        total_loss = loss if total_loss is None else total_loss + loss
    return total_loss


# 训练
############################################################################
def Train_Model(total_epoch):
    loss_min = loss_min_init  # 保存目前为止损失最小的epoch的loss
    loss_list = []  # 保存损失变化用于画图
    if not os.path.exists('data/outputs'): os.makedirs('data/outputs')

    for epoch in tqdm(range(total_epoch)):
        loss_epoch = 0  # 当前epoch的总损失
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            # loss = 分割交叉熵 + 分割dice_loss + 分割ms_ssim_loss
            loss = Total_Loss(y, outputs, [0.9, 0.1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.data
        # 保存loss用于画图
        tqdm.write(f'epoch总损失={loss_epoch}')
        loss_list.append(loss_epoch)
        # 保存损失最小的epoch的网络
        if loss_epoch < loss_min:
            if loss_min != loss_min_init:
                os.remove(f'data/outputs/UNet3_epoch{epochs}_loss{loss_min:0.5f}.ckpt')  # 如果放这一行，则只保存本轮训练loss最低的1个
                pass
            loss_min = loss_epoch
            torch.save(net, f'data/outputs/UNet3_epoch{epochs}_loss{loss_epoch:0.5f}.ckpt')
        # 保存epoch%100=0时的最佳网络
        if ((epoch + 1) % (100 if mode == 'group' else 10) == 0) \
                and ((epoch + 1) > (200 if mode == 'group' else 20)):
            try:
                shutil.copyfile(src=f'data/outputs/UNet3_epoch{epochs}_loss{loss_min:0.5f}.ckpt',
                                dst=f'data/outputs/UNet3_epoch{epoch + 1}_loss{loss_min:0.5f}.ckpt')
            except BaseException:
                pass

    pickle.dump(f'data/outputs/UNet3_epoch{epochs}_loss{loss_min:0.5f}.ckpt', open(f'data/outputs/loss_min.pickle', 'wb'))
    pickle.dump(loss_list, open(f'data/outputs/loss_list.pickle', 'wb'))


if __name__ == '__main__':
    # 训练模型
    Train_Model(epochs)

    # 画图
    import numpy
    import matplotlib.pyplot as plt

    train_loss_list = pickle.load(open('data/outputs/loss_list.pickle', 'rb'))
    train_loss_list = numpy.array([i.to('cpu') for i in train_loss_list])
    plt.plot(train_loss_list, label="train_data loss")
    plt.title("Change of loss")
    plt.savefig(fname="data/outputs/损失变化.png")
    plt.legend()
    plt.show()
