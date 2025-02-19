import random
from PIL import Image
import numpy as np
import cv2
import albumentations as A


class MyTransform:
    def __init__(self):
        self.alb = A.Compose([
            A.OneOf([
                A.RandomGamma(p=1),
                # 只做 hue 增强
                A.HueSaturationValue(p=1),
                A.RandomBrightnessContrast(p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.GaussNoise(),
                # A.IAAAdditiveGaussianNoise(loc=10, scale=(0, 15), p=1)
            ], p=0.5),
            A.OneOf([
                A.Equalize(p=1),
                A.Solarize(p=1),
                A.Posterize(p=1),
            ], p=0.5),
        ])

    def __call__(self, image: Image.Image):
        image = np.asarray(image)
        # 空间增强
        code = random.randint(0, 7)
        # 二分之一概率翻转
        if code % 2:
            image = cv2.flip(image, 0)
        # 四分之一概率不转
        code = code // 2
        if code != 3:
            image = cv2.rotate(image, code)
        # 颜色增强
        image = self.alb(image=image)['image']
        return image
