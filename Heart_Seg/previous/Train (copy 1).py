import os

import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader

from Load import Train_Dataset
from Model.loss_dice import dice_loss

from tqdm import tqdm
import pickle

mode = 'single'
epochs = 100
batch_size = 10
learning_rate = 1e-4

loss_min_init = float('inf')
flag_preTrain = False

image_dir = 'Heart Data'
preTrain = None

# 模型等定义
############################################################################
if __name__ == '__main__':
    # 模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = torch.load(preTrain).to(device) if flag_preTrain else UNet_3Plus(1, 4).to(device)
    # 优化器、调度器、交叉熵
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
    crossEntropy_loss = nn.CrossEntropyLoss()
    # dataloader
    dataset = Train_Dataset(image_dir, mode=mode)
    dataloader = DataLoader(dataset, 1 if mode == 'group' else batch_size, True)


# 训练
############################################################################
def Train_Model(total_epoch):
    loss_min = loss_min_init  # 保存目前为止损失最小的epoch的loss
    loss_list = []  # 保存损失变化用于画图
    if not os.path.exists('data/outputs'): os.makedirs('data/outputs')

    for epoch in tqdm(range(total_epoch)):
        loss_epoch = 0  # 当前epoch的总损失
        for x, y, _class in dataloader:
            # print(x.shape, y.shape, _class.shape)
            if mode == 'group':
                x, y = x.squeeze(0), y.squeeze(0)  # [1,10,1,256,256]->[10,1,256,256]
            x, y, _class = x.to(device), y.to(device), _class.to(device)
            # print(x.shape, y.shape, _class.shape)

            outputs, prediction = net(x)
            # outputs, prediction = functional.softmax(outputs, 1), functional.softmax(prediction, 1)
            print(prediction)
            y = functional.one_hot(y, net.n_classes).permute(0, 3, 1, 2).float()
            _class = functional.one_hot(_class, net.n_classification).float()
            # print(_class)
            # exit()

            # loss = 分割交叉熵 + 分割dice_loss + 预测交叉熵
            with torch.no_grad():
                print(crossEntropy_loss(outputs, y).detach().cpu().numpy(),
                      dice_loss(functional.softmax(outputs, 1), y, multiclass=True).detach().cpu().numpy(),
                      crossEntropy_loss(prediction, _class).detach().cpu().numpy())
                if epoch > 3: exit()
            loss = crossEntropy_loss(outputs, y) \
                   + dice_loss(functional.softmax(outputs, 1), y, multiclass=True) \
                   + crossEntropy_loss(functional.softmax(prediction, 1), _class)
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
                os.remove(f'data/outputs/UNet_s2c_epoch{epochs}_loss{loss_min:0.5f}.ckpt')  # 如果放这一行，则只保存本轮训练loss最低的1个
                pass
            loss_min = loss_epoch
            torch.save(net, f'data/outputs/UNet_s2c_epoch{epochs}_loss{loss_epoch:0.5f}.ckpt')
        # 保存epoch%100=0的网络
        if (epoch + 1) % 100 == 0:
            # torch.save(net, f'data/outputs/UNet_s2c_epoch{epochs}_{epoch + 1}.ckpt')
            pass
    pickle.dump(f'data/outputs/UNet_s2c_epoch{epochs}_loss{loss_min:0.5f}.ckpt', open(f'data/outputs/loss_min.pickle', 'wb'))
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
