import os
import shutil
from PIL import Image
import pickle
import numpy
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader

from Load import Train_Dataset, Test_Dataset

image_dir = 'test'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


if __name__ == '__main__':
    # model
    net_dir = pickle.load(open('data/outputs/loss_min.pickle', 'rb')) if os.path.exists('data/outputs/loss_min.pickle') \
        else 'data/outputs/Model_epoch40_loss0.11021.ckpt'
    # no dropout
    # 50 epoch 13/2
    # net_dir = 'data/outputs/Model_epoch10_loss0.42695.ckpt'  # 1.0, 0.9186, 0.7817
    # net_dir = 'data/outputs/Model_epoch40_loss0.07062.ckpt'  # 1.0, 0.9374, 0.8284 <-- best
    net_dir = 'data/outputs/Model_epoch30_loss0.08952.ckpt'

    # 50 epoch 10/5
    # 1
    # net_dir = 'data/outputs/Model_epoch20_loss0.21259.ckpt'  # 1.0, 0.9093, 0.7521
    # net_dir = 'data/outputs/Model_epoch30_loss0.13899.ckpt'  # 1.0, 0.8669, 0.6299
    # net_dir = 'data/outputs/Model_epoch40_loss0.11021.ckpt'  # 1.0, 0.9172, 0.77657
    # 2
    # net_dir = 'data/outputs/Model_epoch20_loss0.43645.ckpt'  # 1.0, 0.923, 0.791 <-- best
    # 3
    # net_dir = 'data/outputs/Model_epoch20_loss0.19779.ckpt'  # 1.0, 0.884, 0.683
    # net_dir = 'data/outputs/Model_epoch30_loss0.14658.ckpt'  # 1.0, 0.8959, 0.717
    # net_dir = 'data/outputs/Model_epoch40_loss0.02180.ckpt'  # 1.0, 0.9184, 0.7825  <-- second
    # dropout 0.1
    # 60 epoch 10/5
    # net_dir = 'data/outputs/Model_epoch40_loss0.08468.ckpt'  # 1.0, 0.9087, 0.7486
    # net_dir = 'data/outputs/Model_epoch50_loss0.07447.ckpt'  # 1.0, 0.917, 0.774977
    # net_dir = 'data/outputs/Model_epoch60_loss0.00503.ckpt'  # 1.0, 0.91549, 0.7677

    # DropBlock(0.3, 5)
    # 50 epoch DropBlock(0.3, 5)
    # net_dir = 'data/outputs/Model_epoch20_loss0.49477.ckpt'  # 1.0, 0.51957
    # net_dir = 'data/outputs/Model_epoch30_loss0.33995.ckpt'  # 1.0, 0.566
    # net_dir = 'data/outputs/Model_epoch40_loss0.18964.ckpt'  # 1.0, 0.56919
    # 80 epoch DropBlock(0.3, 5)
    # net_dir = 'data/outputs/Model_epoch20_loss1.95299.ckpt'  # 0.9914, 0.72457
    # net_dir = 'data/outputs/Model_epoch30_loss0.51741.ckpt'  # 1.0, 0.75148
    # net_dir = 'data/outputs/Model_epoch50_loss0.04845.ckpt'  # 1.0, 0.7373

    print(net_dir)
    model = torch.load(net_dir).to(device)
    # dataloader
    dataset = Test_Dataset(image_dir)
    dataloader = DataLoader(dataset, 15, False)

    with torch.set_grad_enabled(False):
        model.eval()
        predict_list = []
        classes_list = []
        for image, _, classes, _, path in tqdm(dataloader):
            # print(classes, end=',  ')
            image = image.to(device)
            _, outputs_classify = model(image)
            # 转为标签并存储进行打分
            _predicts = torch.argmax(outputs_classify, dim=1).to(device='cpu', dtype=torch.int).numpy()
            # print(_predicts.shape,classes.shape)
            # exit()
            predict_list.extend(_predicts)
            classes_list.extend(classes.numpy())
            # list(print(f'({_predicts[i]}, {classes[i]})') for i in range(len(classes)))

        # 计算score（f1_score多分类只能接入一维）
        score = f1_score(predict_list, classes_list, average='macro')
        print(f'平均F1得分: {score}')
