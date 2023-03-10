import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
from torchvision.models import densenet169, inception_v3, resnet101, resnet50, densenet121, resnet18
from tqdm import tqdm
import pickle
import os
from Load import Train_Dataset, DrawLoss, Test_Dataset

learning_rate = 1e-4
epochs = 50
batch_size = 64
val_size = 0.2
do_validation = True
loss_min_init = float('inf')
accurate_high_init = float(10)

train_label = 'data/train_im.csv'
test_label = 'data/valid_im.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

def inceptionoutputs_v3(out_channel: int = 3):
    model = inception_v3(False)
    model.aux_logits = False
    in_channel = model.fc.in_features
    model.fc = torch.nn.Linear(in_channel, out_channel)
    return model


def Train(epochs):
    loss_list = []  # 保存损失变化用于画图
    accurate_list = []  # 保存val正确率变化用于画图
    loss_min = loss_min_init
    accurate_high = accurate_high_init
    temp_accurate = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 40, 60, 80], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)  # goal: maximize Dice score
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), '模型训练', position=1):
        loss_epoch = 0  # 当前epoch的总损失
        for data, target in tqdm(train_dataloader, desc='during one epoch...', position=0):
            data = data.to(device)
            target = target.to(device)
            # model.eval()
            score = model(data)
            optimizer.zero_grad()
            loss = loss_criterion(score, target)
            loss.backward()
            optimizer.step()
            # loss_epoch += loss.data
            loss_epoch = loss.data
        scheduler.step()

        print(f'epoch:{epoch}')
        print("train:")
        accurate = Val(model, device, train_dataloader, len(train_dataset))
        loss_list.append(loss_epoch.to('cpu'))

        # 保存val正确率最high的epoch的网络
        if do_validation:
            print("valid:")
            accurate = Val(model, device, val_dataloader, len(val_dataset))
            accurate_list.append(accurate)
            model.train()
            if accurate > 80:
                torch.save(model, f'data/outputs/DenseNet_epoch{epoch}_accurate{accurate}.ckpt')

    if do_validation:
        pickle.dump(f'data/outputs/DenseNet_epoch{epochs}_accurate{accurate_high}.ckpt',
                    open(f'data/outputs/accurate_high.pickle', 'wb'))
        pickle.dump(accurate_list, open(f'data/outputs/accurate_list.pickle', 'wb'))

    pickle.dump(f'data/outputs/DenseNet_epoch{epochs}_loss{loss_min:0.5f}_accurate{temp_accurate}.ckpt'
                if do_validation else f'data/outputs/DenseNet_epoch{epochs}_loss{loss_min:0.5f}.ckpt',
                open(f'data/outputs/loss_min.pickle', 'wb'))
    pickle.dump(loss_list, open(f'data/outputs/loss_list.pickle', 'wb'))

    torch.save(model, 'data/outputs/model.ckpt')


def Val(model, device, dataloader, dataset_size, show_process=False):
    correct_count = 0
    num_0 = 0
    pre_0 = 0
    with torch.no_grad():
        # model.eval()
        for image, label in tqdm(dataloader) if show_process else dataloader:
            image = image.to(device)
            score = model(image)
            prediction = torch.argmax(score, dim=1).to('cpu').numpy()
            correct_count += (label.numpy() == prediction).sum()
            for i, j in zip(np.array(label), np.array(prediction)):
                if i == 1:
                    num_0 = num_0 + 1
                    pre_0 += (i == j).sum()

            label = label.to(device)
            loss = loss_criterion(score, label)
    print(f'loss={loss}')
    print(f"正确个数：{correct_count}，正确率{(100 * correct_count) / dataset_size}%")
    print(f"HCM正确个数：{pre_0}，正确率{(100 * pre_0) / num_0}%")
    # return (100 * correct_count) / dataset_size
    # exit(0)
    return (100 * pre_0) / num_0


if __name__ == '__main__':
    # model = inceptionoutputs_v3(3)
    model = densenet121(pretrained=False)

    # for layer1 in model.features.children():
    #     if layer1._get_name() == '_DenseBlock' or layer1._get_name() == '_DenseLayer':
    #         for layer2 in layer1.children():
    #             for layer3 in layer2.children():
    #                 if layer3._get_name() == 'Conv2d':
    #                     layer3.kernel_size = (10, 10)

    model = torch.nn.Sequential(model, torch.nn.Linear(1000, 256),
                                torch.nn.ReLU(inplace=True),
                                # torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, 3))

    model = model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()

    # 数据集
    train_dataset = Train_Dataset(train_label, None, True)
    # 划分训练集验证集
    if do_validation:
        val_dataset = Test_Dataset(test_label)
        val_dataloader = DataLoader(val_dataset, batch_size, False)

    # 设置训练集
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f'length of train_dataset: {len(train_dataset)}')
    Train(epochs)
