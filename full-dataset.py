import os
import numpy as np
import torch
import json
from PIL import Image


class LemonDataset(object):
    def __init__(self, data_path, transforms=None, return_filenames=False, return_raw_classes=False):
        self.imgs = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        with open('data/lemon-dataset/annotations/instances_default.json') as file:
            self.categories = {}
            annotations = json.load(file)
            id_to_filename = {}
            for image in annotations['images']:
                filename = image['file_name'].split('/')[1]
                id_to_filename[image['id']] = filename
                self.categories[filename] = []

            for annotation in annotations['annotations']:
                category_id = annotation['category_id']
                if category_id not in [7, 9, 1]:
                    filename = id_to_filename[annotation['image_id']]
                    self.categories[filename].append(category_id)

        self.transforms = transforms
        self.return_filenames = return_filenames
        self.return_raw_classes = return_raw_classes

    def __getitem__(self, idx):
        image_filename = self.imgs[idx]
        img_path = os.path.join('data/lemon-dataset/images', image_filename)
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if len(self.categories[image_filename]) == 0:
            y = 0  # healthy
        elif 4 in self.categories[image_filename]:
            y = 1  # mould
        else:
            y = 2  # other

        if self.return_filenames:
            return np.array(img), y, image_filename
        else:
            return np.array(img), y

    def __len__(self):
        return len(self.imgs)
