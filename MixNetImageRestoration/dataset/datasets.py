import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from dataset.transformation import train_transformation, test_transformation
import glob
import matplotlib.pyplot as plt

class ImageRestorationDataset(Dataset):
    def __init__(self, img_paths, transform):
        """
        Initializes the object with a list of image paths and a transformation
        to be applied on them.

        :param img_paths: A list of strings representing the file paths of the images.
        :type img_paths: list[str]
        :param transform: A transformation to be applied on the images, e.g. resizing, cropping, etc.
        :type transform: torchvision.transforms
        """
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_target = test_transformation(img)
        img_input = self.transform(img)
        return img_input, img_target

def my_collate_fn(batch):
    img_input, img_target = [], []
    for item in batch:
        img_input.append(item[0])
        img_target.append(item[1])
    img_input = padding_batch(img_input)
    img_target = padding_batch(img_target)
    return img_input, img_target

def padding_batch(batch):
    """Pad a batch of images. each img in batch (list) is tensor with shape (C, H, W)"""
    batch_size = len(batch)
    max_h, max_w = 0, 0
    for img in batch:
        # print(type(img))
        h, w = img.shape[1], img.shape[2]
        if max_h < h:
            max_h = h
        if max_w < w:
            max_w = w
    padded_batch = torch.ones((batch_size, 3, max_h, max_w))
    for i, img in enumerate(batch):
        padded_batch[i, :, :img.shape[1], :img.shape[2]] = img
    return padded_batch

def preprocess_train(img_paths, transform, batch_size):
    dataset = ImageRestorationDataset(img_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    return dataloader

def preprocess_test(img_paths, transform, batch_size):
    dataset = ImageRestorationDataset(img_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
    return dataloader

if __name__ == "__main__":
    train_transformation = preprocess_train(glob.glob("/home/tran/ImageRestoration/real_1204/*.png"), train_transformation, batch_size=4)
    iter = iter(train_transformation)
    image_input, image_target = next(iter)  
    # print(image_input)
    print(image_input.shape)
    print(image_target.shape)
    for i in range(1,9):
        plt.subplot(4, 2, i)
        if i%2==0:
            plt.imshow(image_input[(i-1)//2].permute(1, 2, 0))
        else:
            plt.imshow(image_target[(i-1)//2].permute(1, 2, 0))
        plt.xticks([]), plt.yticks([])
    
    
    plt.show()