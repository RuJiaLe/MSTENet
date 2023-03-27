import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import PIL.ImageOps
import numpy as np
from PIL import ImageEnhance


def get_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([Resize(input_size),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])


def get_train_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([Resize(input_size),
                               RandomFlip(),
                               Random_crop(10),
                               RandomRotation(),
                               ColorEnhance(),
                               Not_GT(),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])


class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """

    def __call__(self, samples):
        rand_flip_index = random.randint(-1, 2)

        if rand_flip_index == 0:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']
                image = F.hflip(image)
                gt = F.hflip(gt)
                not_gt = F.hflip(not_gt)
                sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
                samples[i] = sample

        elif rand_flip_index == 1:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']
                image = F.vflip(image)
                gt = F.vflip(gt)
                not_gt = F.vflip(not_gt)
                sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
                samples[i] = sample

        else:
            for i in range(len(samples)):
                sample = samples[i]
                image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']
                image = F.vflip(F.hflip(image))
                gt = F.vflip(F.hflip(gt))
                not_gt = F.vflip(F.hflip(not_gt))
                sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
                samples[i] = sample

        return samples


class Resize(object):
    """ Resize PIL image use both for training and inference"""

    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']

            image = F.resize(image, self.size, InterpolationMode.BILINEAR)
            gt = F.resize(gt, self.size, InterpolationMode.BILINEAR)
            not_gt = F.resize(not_gt, self.size, InterpolationMode.BILINEAR)

            sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
            samples[i] = sample

        return samples


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']

            image = F.to_tensor(image)

            gt = F.to_tensor(gt)
            not_gt = F.to_tensor(not_gt)

            sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
            samples[i] = sample

        return samples


class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) ? Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """

    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]

            image = sample['image']
            image = F.normalize(image, self.mean, self.std)

            sample["image"] = image

            samples[i] = sample

        return samples


class Random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, samples):
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        width, height = samples[0]["image"].size
        assert samples[0]["image"].size == samples[0]["gt"].size
        region = [x, y, width - x, height - y]

        for i in range(len(samples)):
            sample = samples[i]
            image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']

            image = image.crop(region)
            gt = gt.crop(region)
            not_gt = not_gt.crop(region)

            image = image.resize((width, height), Image.BILINEAR)
            gt = gt.resize((width, height), Image.BILINEAR)
            not_gt = not_gt.resize((width, height), Image.BILINEAR)

            sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
            samples[i] = sample

        return samples


class RandomRotation(object):

    def __call__(self, samples):

        mode = Image.BILINEAR
        random_angle = np.random.randint(-15, 15)
        
        if random.random() > 0.8:

            for i in range(len(samples)):
                sample = samples[i]
                image, gt, not_gt = sample['image'], sample['gt'], sample['not_gt']

                image = image.rotate(random_angle, mode)

                gt = gt.rotate(random_angle, mode)
                not_gt = not_gt.rotate(random_angle, mode)

                sample['image'], sample['gt'], sample['not_gt'] = image, gt, not_gt
                samples[i] = sample

        return samples


class ColorEnhance(object):

    def __call__(self, samples):
        bright_intensity = random.randint(5,15) / 10.0
        contrast_intensity = random.randint(5,15) / 10.0
        color_intensity = random.randint(0,20) / 10.0
        sharp_intensity = random.randint(0,30) / 10.0
     

        for i in range(len(samples)):
            sample = samples[i]
            image = sample['image']

            image = ImageEnhance.Brightness(image).enhance(bright_intensity)
            image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
            image = ImageEnhance.Color(image).enhance(color_intensity)
            image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)

            sample['image'] = image
            samples[i] = sample

        return samples


class Not_GT(object):

    def __call__(self, samples):

        for i in range(len(samples)):
            sample = samples[i]
            not_gt = sample['not_gt']

            not_gt = PIL.ImageOps.invert(not_gt)

            sample['not_gt'] = not_gt
            samples[i] = sample

        return samples

