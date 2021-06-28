#%%
import os
from shutil import copyfile
from dataset import LemonDataset
from torch.utils.data import DataLoader

#%%
TRAIN_DATA_PATH = 'data/train'
VAL_DATA_PATH = 'data/val'
TEST_DATA_PATH = 'data/test'

#%%
data_path = TEST_DATA_PATH

#%%
test_dataset = LemonDataset(data_path, return_filenames=True, return_raw_classes=True)

#%%
os.mkdir(os.path.join(data_path, 'healthy'))
os.mkdir(os.path.join(data_path, 'mould'))

#%%
data_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
images, labels, file_names = next(iter(data_loader))

for image_file_name, label in zip(file_names, labels):
    if label == 0:
        copyfile(os.path.join(data_path, image_file_name), os.path.join(data_path, 'healthy', image_file_name))
    elif label == 1:
        copyfile(os.path.join(data_path, image_file_name), os.path.join(data_path, 'mould', image_file_name))
#%%
from pycocotools.coco import COCO

coco = COCO('data/lemon-dataset/annotations/instances_default.json')
