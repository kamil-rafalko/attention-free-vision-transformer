import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import LemonDataset

DATA_PATH = 'data/lemon-dataset/images'


def split_dataset():
    dataset = LemonDataset(DATA_PATH)
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels, file_names = next(iter(data_loader))
    X_train, X_test, y_train, y_test = train_test_split(file_names, labels, test_size=0.2, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

    os.mkdir('data/train')
    os.mkdir('data/test')
    os.mkdir('data/val')

    for image_file_name in X_train:
        copyfile(os.path.join(DATA_PATH, image_file_name), os.path.join('data/train', image_file_name))
            
    for image_file_name in X_test:
        copyfile(os.path.join(DATA_PATH, image_file_name), os.path.join('data/test', image_file_name))
        
    for image_file_name in X_val:
        copyfile(os.path.join(DATA_PATH, image_file_name), os.path.join('data/val', image_file_name))


if __name__ == '__main__':
    split_dataset()