import os
from tqdm import tqdm
import pickle
import shutil

import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader

from Model.Model import Model
from Load import Test_Dataset

image_dir = 'train10'
loss_min_init = float('inf')

epochs = 50
batch_size = 15
learning_rate = 1e-5

dropout_p = 0.1
dropout_size = 5
close_drop = True
# unet_3plus_dir = 'data/outputs/UNet3_epoch60_loss19.83897.ckpt'  # 0.90438, 0.859
unet_3plus_dir = 'data/outputs/UNet3_epoch80_loss19.56409.ckpt'  # 0.9035, 0.8518468

# 模型等定义
############################################################################
if __name__ == '__main__':
    # 模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    unet_3plus = torch.load(unet_3plus_dir)
    net = Model(3, unet_3plus, dropout_p=dropout_p, dropout_size=dropout_size, close_drop=close_drop).to(device)
    # 优化器(网络有冻结)
    # 优化器、调度器
    optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, net.parameters()),
                              lr=learning_rate, weight_decay=1e-8, momentum=0.9)  # 1e-5, 1e-8, 0.9
    # optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)  # 2
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, 2)
    # 损失
    crossEntropy_loss_fn = nn.CrossEntropyLoss()
    # dataloader
    dataset = Test_Dataset('train')
    dataloader = DataLoader(dataset, batch_size, True)


# 训练
############################################################################
def Train_Model(total_epoch):
    loss_min = loss_min_init  # 保存目前为止损失最小的epoch的loss
    loss_list = []  # 保存损失变化用于画图

    for epoch in tqdm(range(total_epoch)):
        loss_epoch = 0  # 当前epoch的总损失
        for image, label, classes, size, path in dataloader:
            # print(classes, end=',  ')
            image, classes = image.to(device), classes.to(device)
            _, outputs_classify = net(image)
            # print(outputs_classify.detach())
            # 损失：三分类的交叉熵
            loss = crossEntropy_loss_fn(outputs_classify,
                                        functional.one_hot(classes, net.n_classification).float())

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
                os.remove(f'data/outputs/Model_epoch{epochs}_loss{loss_min:0.5f}.ckpt')  # 如果放这一行，则只保存本轮训练loss最低的1个
                pass
            loss_min = loss_epoch
            torch.save(net, f'data/outputs/Model_epoch{epochs}_loss{loss_epoch:0.5f}.ckpt')
        # 保存epoch%10=0时的最佳网络
        if (epoch + 1) % 10 == 0:
            try:
                shutil.copyfile(src=f'data/outputs/Model_epoch{epochs}_loss{loss_min:0.5f}.ckpt',
                                dst=f'data/outputs/Model_epoch{epoch + 1}_loss{loss_min:0.5f}.ckpt')
            except BaseException:
                pass

    pickle.dump(f'data/outputs/Model_epoch{epochs}_loss{loss_min:0.5f}.ckpt', open(f'data/outputs/loss_min.pickle', 'wb'))
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
