import os
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageEnhance

#'seg', 'label'

image_size = 256

train_image_path = 'data/rgb/'
valid_image_path = 'data/rgb_pre_all/'


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img



class Train_Dataset(Dataset):
    def __init__(self, annotations_file, total, is_train):
        self.file = pd.read_csv(annotations_file)

        self.image_transform = transforms.Compose([#transforms.CenterCrop(image_size),
                                                   # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                                   # transforms.RandomAdjustSharpness(0, 0.6),
                                                   # transforms.RandomHorizontalFlip(),
                                                   # transforms.GaussianBlur(3),
                                                   transforms.Resize((image_size, image_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                   ]) \
        if is_train else transforms.Compose([ #transforms.CenterCrop(image_size),
                                             transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        img_path = train_image_path + self.file['seg'][index]
        image = Image.open(img_path).convert('RGB')
        # name = self.file['seg'][index]
        # image = one_to_three(name)


        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = crop_image_from_gray(image, tol=7)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (image_size, image_size))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), image_size / 10), -4, 128)

        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = self.image_transform(image)
        label = self.file['label'][index]
        return image, label


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
        img_path = valid_image_path + self.file['seg'][index]
        image = Image.open(img_path).convert('RGB')
        name = self.file['seg'][index]
        # image = one_to_three(name)

        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = crop_image_from_gray(image, tol=7)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (image_size, image_size))
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), image_size / 10), -4, 128)

        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = self.image_transform(image)
        label = self.file['label'][index]
        return image, label


# 损失变化画图
def DrawLoss(loss_list, accurate_list=None):
    plt.plot(loss_list, label="train_data loss")
    # plt.plot(accurate_list, label="val_data accurate")
    plt.title("Change of loss")
    plt.savefig(fname="data/loss.png")
    plt.legend()
    plt.show()


def enhance():
    image = ''
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 5
    image_brightened = enh_bri.enhance(brightness)

    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)
    ###################################################
    nh_bri = ImageEnhance.Brightness(image)
    brightness = 5
    image_brightened = enh_bri.enhance(brightness)

    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)


def one_to_three(name):
    source_path = 'data/total/'
    label_path = 'data/total_label/'
    ini_path = source_path + name

    label_name = name.replace('image', 'label')
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


# if __name__ == '__main__':
#     # chuli()
#     img = cv2.imread('train/20051019_38557_0100_PP.png')
#     a = crop_image_from_gray(img, tol=7)
#     plt.imshow(a)
#     plt.show()
#     print("666")



