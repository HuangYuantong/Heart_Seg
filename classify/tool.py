import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageEnhance


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



def enhance():
    image_size = 256
    img_path = 'data/pic_im/DCM_01_image10.png'
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    # image = cv2.imread(img_path)
    image = Image.open(img_path)
    plt.subplot(231)
    plt.title("原图")
    plt.imshow(image)

    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 4
    image_brightened = enh_bri.enhance(brightness)
    plt.subplot(232)
    plt.title("亮度增强")
    plt.imshow(image_brightened)

    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = 50
    image_colored = enh_col.enhance(color)
    plt.subplot(233)
    plt.title("色度增强")
    plt.imshow(image_colored)

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)
    plt.subplot(234)
    plt.title("对比度增强")
    plt.imshow(image_contrasted)

    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image_brightened)
    sharpness = 100
    image_sharped = enh_sha.enhance(sharpness)
    plt.subplot(235)
    plt.title("亮度+锐度增强")
    plt.imshow(image_sharped)


    enh_sha0 = ImageEnhance.Sharpness(image)
    sharpness0 = 100
    image_sharped0 = enh_sha0.enhance(sharpness0)
    plt.subplot(236)
    plt.title("原锐度增强")
    plt.imshow(image_sharped0)

    plt.show()


def value_add():
    source_path = os.path.abspath('data/valid_im')  # 原图片文件夹
    label_path = os.path.abspath('label/valid_label')  # 标签文件夹
    target_path = os.path.abspath('pinjie_val')  # 目标文件夹

    if not os.path.exists(target_path):  # 目标文件夹不存在就新建
        os.makedirs(target_path)

    for root, dirs, files in os.walk(source_path):
        for file in files:
            label_name = file.replace('image', 'label')
            la_path = label_path + '/' + label_name
            ini_path = root + '/' + file

            image1 = cv2.imread(ini_path)
            image2 = cv2.imread(la_path)
            image = cv2.addWeighted(image1, 5, image2, -4, 128)

            cv2.imwrite(target_path + '/' + file, image)

    print('ok')



def one_to_three_1():
    source_path = os.path.abspath('data/train_im')  # 原图片文件夹
    label_path = os.path.abspath('label/train_label')  # 标签文件夹
    target_path = os.path.abspath('one_to_three')  # 目标文件夹

    if not os.path.exists(target_path):  # 目标文件夹不存在就新建
        os.makedirs(target_path)

    for root, dirs, files in os.walk(source_path):
        for file in files:
            label_name = file.replace('image', 'label')
            # print(label_name)
            la_path = label_path + '/' + label_name
            ini_path = root + '/' + file

            image1 = cv2.imread(ini_path, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(la_path, cv2.IMREAD_GRAYSCALE)

            h, w = image1.shape
            blank = np.zeros([h, w], image1.dtype)

            image = np.stack([image1, image2, blank], axis=-1)
            # print(image.shape)
            # plt.imshow(image)
            # plt.show()

            cv2.imwrite(target_path + '/' + file, image)
    print('ok')

def one_to_three(name):
    source_path = 'data/total/'
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

if __name__ == '__main__':
    one_to_three('HCM_14_image6.png')
    print(666)





