from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from torchvision.models import densenet169, inception_v3
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset


test_label = 'data/test_im.csv'
save_path = 'data/pre.csv'  # 结果保存
image_size = 256
test_image_path = 'data/rgb_pre_all/'
net_dir_des = 'data/model/one_to_three_good.ckpt'


def one_to_three(name):
    source_path = 'data/image/'
    label_path = 'data/predicts/'
    ini_path = source_path + name

    label_name = 'Image_' + name.replace('_', '')
    la_path = label_path + label_name

    image1 = Image.open(ini_path).convert('L')
    image2 = Image.open(la_path).convert('L')


    lab = np.array(image2)
    lab = lab.astype(str)

    merge = np.zeros((3, lab.shape[0], lab.shape[1]))
    idx1 = {'0': 0, '170': 170, '255': 0, '85': 0}
    idx2 = {'0': 0, '170': 0, '255': 255, '85': 85}

    merge[0] = np.array(image1)
    ix, jx = lab.shape
    for i in range(ix):
        for j in range(jx):
            pixel = lab[i][j]
            if pixel != '0':
                merge[0][i][j] = 0
            pixelid1 = idx1[pixel]
            merge[1][i][j] = pixelid1
            pixelid2 = idx2[pixel]
            merge[2][i][j] = pixelid2

    merge = merge.astype("int32")
    merge = Image.fromarray(np.uint8(merge.transpose((1, 2, 0))))

    return merge




class Test_Dataset(Dataset):
    def __init__(self, annotations_file):
        self.file = pd.read_csv(annotations_file)

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化的正态分布
        ])

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        img_path = test_image_path + self.file['seg'][index]
        name = self.file['seg'][index]
        # image = one_to_three(name)

        image = Image.open(img_path).convert('RGB')
        # image = cv2.imread(img_path)
        # image = crop_image_from_gray(image, tol=7)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (image_size, image_size))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), image_size / 10), -4, 128)

        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.image_transform(image)
        label = self.file['label'][index]
        return image, label, name


def Test(dataloader, size):
    pre_data = []
    correct_count = 0
    with torch.no_grad():
        model_des.eval()  # 关闭网络中batchNorm、drop层
        for image, label, name in tqdm(dataloader, '标签预测中'):
            image = image.to(device)
            score_des = model_des(image)
            # print(f'name:{name}, score:{score_des}')
            prediction_des = torch.argmax(score_des, dim=1).to('cpu').numpy()

            # ss = score_des.tolist()
            # if((ss[0][prediction_des[0]]-ss[0][1]<=4)and(((ss[0][1]<ss[0][2])and(ss[0][1]>ss[0][0]))or((ss[0][1]<ss[0][0])and(ss[0][1]>ss[0][2])))):
            #
            # # if (ss[0][prediction_des[0]] - ss[0][1] <= 1):
            #     prediction = [1]
            # else:
            #     prediction = prediction_des

            prediction = prediction_des
            correct_count += (label.numpy() == prediction).sum()
            pre_data.append([name, prediction])
            # exit(0)

    print(f"正确个数：{correct_count}，正确率{(100 * correct_count) / size}%")
    return pre_data


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(net_dir_des)
    model_des = torch.load(net_dir_des).to(device)
    test_dataset = Test_Dataset(test_label)
    test_dataloader = DataLoader(test_dataset, 1, False)

    result = Test(test_dataloader, len(test_dataset))
    df = pd.DataFrame(result, columns=['seg', 'label'])
    df.to_csv(save_path, index=None)
