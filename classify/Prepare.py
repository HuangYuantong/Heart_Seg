import pandas as pd
import numpy as np
import os
import cv2
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
from Load import DrawLoss


#DCM:2
#HCM:1
#NOR:0
label_path = 'data/total.csv'



def heng():
    df = []
    source_path = os.path.abspath('Heart Data/Image_HCM/png/Label')  # 源文件夹
    target_path = os.path.abspath('data/1')  # 目标文件夹
    lab = 'HCM'
    label = 1

    if not os.path.exists(target_path):  # 目标文件夹不存在就新建
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                name = lab + '_' + root[-2:] + '_' + file
                # print(name)
                df.append([name, label])
                im = cv2.imread(src_file)
                cv2.imwrite(target_path + '/' + name, im)

    df = pd.DataFrame(df, columns=['seg', 'label'])
    df.to_csv('data/'+lab +'.csv', index=None)


def pinjie():
    df0 = pd.read_csv('data/NOR.csv')
    df1 = pd.read_csv('data/HCM.csv')
    df2 = pd.read_csv('data/DCM.csv')

    df = pd.concat([df0, df1])
    df = pd.concat([df, df2])
    df.to_csv('data/total.csv', index=None)


def split():
    df = pd.read_csv(label_path)

    l = np.array(df)

    train_data, val_data = train_test_split(l, test_size=0.2, random_state=0)
    df_train = []
    df_val = []

    for index in train_data:
        df_train.append([index[0], index[1]])

    for index in val_data:
        df_val.append([index[0], index[1]])

    df1 = pd.DataFrame(df_train, columns=['seg', 'label'])
    df2 = pd.DataFrame(df_val, columns=['seg', 'label'])

    df1.to_csv('data/train.csv', index=None)
    df2.to_csv('data/val.csv', index=None)


def heng1():
    df = []
    source_path = os.path.abspath('data/predicts')  # 源文件夹
    target_path = os.path.abspath('data/1')  # 目标文件夹
    lab = 'HCM'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                # src_file = os.path.join(root, file)
                print(file)
                label = file[6:7]
                print(label)
                if label =='N':
                    label_n = 0
                if label =='H':
                    label_n = 1
                if label =='D':
                    label_n = 2
                print(label_n)
                df.append([file, label_n])

    df = pd.DataFrame(df, columns=['seg', 'label'])
    df.to_csv('test.csv', index=None)

def look_loss():
    with open("data/outputs/loss_list.pickle", 'rb') as f:
        loss_list = pickle.load(f, encoding='iso-8859-1')
    DrawLoss(loss_list)



if __name__ == '__main__':
    # heng()
    # pinjie()
    split()
    # heng1()
    print('666')












