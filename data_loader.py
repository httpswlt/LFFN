import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from itertools import product
from ChannelAug import ChannelAdapGray, ChannelRandomColorErasingNormal


class ColorExchange(object):
    def __init__(self, probability):
        super(ColorExchange, self).__init__()
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        index = [0, 1, 2]
        c1 = random.choice(index)
        index.remove(c1)
        c2 = random.choice(index)
        index.remove(c2)
        c3 = index[0]
        img_ex = torch.cat([img[c1:c1 + 1, ...], img[c2:c2 + 1, ...], img[c3:c3 + 1, ...]], dim=0)
        return img_ex


class ChannelExchange(object):
    """
        gray
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


def brightness(img, mean):
    factor = mean / np.mean(img)
    # factor = random.uniform(factor - 0.2, factor + 0.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)

    return img


def brightness_torch(img, mean):
    factor = mean / (img.mean().item() * 255)
    enhancer = transforms.functional.adjust_brightness(img, factor)
    return enhancer


class ImagePatch(object):
    def __init__(self, probability=0.5, w=144, h=288, sw=2, sh=4):
        self.probability = probability
        self.w = w
        self.h = h
        self.sw = sw
        self.sh = sh
        self.pw = self.w // self.sw
        self.ph = self.h // self.sh
        self.coordinate = self.generate_coordinate()
        self.gray = ChannelExchange(3)

    def generate_coordinate(self):
        x = list(range(0, self.w, self.pw))
        y = list(range(0, self.h, self.ph))
        coordinate = list(product(x, y))
        return coordinate

    def __call__(self, img, sp=1):
        if random.uniform(0, 1) > self.probability:
            return img
        img_gray = self.gray(img.clone())
        number = random.randint(1, sp)
        coordinates = random.sample(self.coordinate, number)

        for x, y in coordinates:
            img[:, y:y + self.ph, x:x + self.pw] = img_gray[:, y:y + self.ph, x:x + self.pw]
        return img


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        # data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.ToTensor(),
            ColorExchange(probability=0.5),
            ChannelRandomColorErasingNormal(probability=0.5),
            ChannelAdapGray(probability=0.5)
        ])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.ToTensor(),
            ImagePatch(0.5),
            ColorExchange(probability=0.5),
            ChannelRandomColorErasingNormal(probability=0.5),
        ])

        self.transform_color1_0 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2),
            transforms.ToTensor(),
            ChannelExchange(gray=3)
        ])
        self.transform_color1_1 = transforms.Compose([
            ChannelRandomColorErasingNormal(probability=0.5)
        ])

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform_color(img1)

        img2_mean = np.mean(img2)
        img2_0 = self.transform_thermal(img2)

        img1_1 = self.transform_color1_0(img1)
        img1_1 = brightness_torch(img1_1, img2_mean)
        img1_1 = self.transform_color1_1(img1_1)

        # cv2.imwrite('./imgs/ori_img.jpg', img1)
        # cv2.imwrite('./imgs/ori_img1.jpg', img2)
        # torchvision.utils.save_image(img1_0, "./imgs/output01.jpg")
        # torchvision.utils.save_image(img1_1, "./imgs/output02.jpg")
        # torchvision.utils.save_image(img2_0, "./imgs/output03.jpg")

        return img1_0, img1_1, img2_0, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelRandomColorErasingNormal(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelRandomColorErasingNormal(probability=0.5)])

        self.transform_color1_0 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelExchange(gray=3)
        ])
        self.transform_color1_1 = transforms.Compose([
            ChannelRandomColorErasingNormal(probability=0.5)
        ])

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform_color(img1)

        img2_mean = np.mean(img2)
        img2_0 = self.transform_thermal(img2)

        img1_1 = self.transform_color1_0(img1)
        img1_1 = brightness_torch(img1_1, img2_mean)
        img1_1 = self.transform_color1_1(img1_1)

        return img1_0, img1_1, img2_0, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
