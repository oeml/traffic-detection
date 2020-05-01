import glob
import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    diff = np.abs(h - w)
    start_padding, end_padding = diff // 2, diff - diff // 2
    padding = (0, 0, start_padding, end_padding) if h <= w else (start_padding, end_padding, 0, 0)
    img = F.pad(img, padding, 'constant', value=pad_value)
    return img, padding


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = h, w
        img, padding = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None

        if os.path.exists(label_path):
            bboxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1,5))
            x1 = w_factor * (bboxes[:, 1] - bboxes[:, 3] / 2) + padding[0]
            y1 = h_factor * (bboxes[:, 2] - bboxes[:, 4] / 2) + padding[2]
            x2 = w_factor * (bboxes[:, 1] + bboxes[:, 3] / 2) + padding[1]
            y2 = h_factor * (bboxes[:, 2] + bboxes[:, 4] / 2) + padding[3]

            bboxes[:, 1] = ((x1 + x2) / 2) / padded_w
            bboxes[:, 2] = ((y1 + y2) / 2) / padded_h
            bboxes[:, 3] *= w_factor / padded_w
            bboxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(bboxes), 6))
            targets[:, 1:] = bboxes

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)  # concat tensors in list to one tensor
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
