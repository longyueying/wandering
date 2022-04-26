import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


def plot_single_image(filepath):
    with open(filepath, "rb") as f:
        dict_cifar = pickle.load(f, encoding='bytes')
        print(dict_cifar.keys())
        img = dict_cifar[b"data"]
        print(type(img))
        label = dict_cifar[b"labels"]
        single_img = np.array(img[5])
        print(img.shape)
        single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
        plt.figure(figsize=(0.32, 0.32))
        plt.imshow(single_img_reshaped)
        plt.show()


class Cifar10Dataset(Dataset):
    def __init__(self, dir_cifar10, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if train:
            file_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        else:
            file_list = ["test_batch"]
        self.data = []
        self.labels = []
        for file_name in file_list:
            file_path = os.path.join(dir_cifar10, file_name)
            with open(file_path, "rb") as f:
                dict_cifar = pickle.load(f, encoding='bytes')
                imgs = dict_cifar[b"data"]
                labels = dict_cifar[b"labels"]
                self.data.extend(imgs)
                self.labels.extend(labels)
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    # filepath = "E:/Data/cifar/cifar-10-batches-py/data_batch_1"
    # plot_single_image(filepath)
    cifar10_dir = "E:/Data/cifar/cifar-10-batches-py"
    data_set = Cifar10Dataset(cifar10_dir)
    img, label = data_set.__getitem__(1)
    # plt.figure(figsize=(0.32, 0.32))
    # plt.imshow(img)
    # plt.show()