from __future__ import absolute_import

from torchvision.transforms import *

# from PIL import Image
import random
import math

# import numpy as np
import torch


class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

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
            img = img

        return img


class ChannelAdapGray(object):
    """
        Adaptive selects Gray.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        idx = random.randint(0, 3)

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


class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ChannelRandomColorErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.33, r1=0.3):

        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if random.randint(0, 1):
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, ...].mean()
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, ...].mean()
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, ...].mean()
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                return img

        return img


class ChannelRandomColorErasingNormal(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.33, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
        self.normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def color_erase(self, img):
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = min(int(round(math.sqrt(target_area * aspect_ratio))), img.size()[1] // 3)
            w = min(int(round(math.sqrt(target_area / aspect_ratio))), img.size()[2] // 3)

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if random.randint(0, 1):
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, ...].mean()
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, ...].mean()
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, ...].mean()
                else:
                    img[:, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                return img
        return img

    def normal_erase(self, img):
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    if random.randint(0, 1):
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        value = random.choice(self.mean)
                        img[:, x1:x1 + h, y1:y1 + w] = value
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

    def __call__(self, img):
        img = self.normal(img)
        # if random.uniform(0, 1) > self.probability:
        #     img = self.normal(img)
        #     return img
        #
        # if random.randint(0, 1):
        #     img = self.normal(img)
        #     img = self.normal_erase(img)
        # else:
        #     img = self.color_erase(img)
        #     img = self.normal(img)

        return img
