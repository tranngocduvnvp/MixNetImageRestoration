import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import RandomErasing
import torch
# from torchvision.transforms.v2 import RandomErasing



def blur_motion(image, p=1.0):
    return alb.Compose([
        alb.MotionBlur(p=p),
    ])(image=image)["image"]

def gausian_blur(image, p=1.0):
    return alb.Compose([
        alb.GaussianBlur(p=p),
    ])(image=image)["image"]

def resize_blur(image, scale=0.5):
    h, w = image.shape[:2]
    img = cv2.resize(image, (int(w * scale), int(h * scale)))
    img = cv2.resize(img, (w, h))
    return img

def peper_noise(image, density=0.01):
    noise_density = density # Mật độ của nhiễu
    noisy_image = np.copy(image)
    salt_and_pepper = np.random.rand(*image.shape[:2])
    noisy_image[salt_and_pepper < noise_density] = 0
    noisy_image[salt_and_pepper > 1 - noise_density] = 255
    return noisy_image


def erase_noise(image, p=1.0):
    image = torch.from_numpy(image).permute(2, 0, 1)
    return RandomErasing(p=p, value=255, scale=(0.01, 0.03), ratio=(0.3, 3.3))(image).permute(1, 2, 0).numpy()


def mix_noise(img):
    img = gausian_blur(img, np.random.uniform(0.0, 1.0))
    img = resize_blur(img)
    img = erase_noise(img, np.random.uniform(0.0, 1.0))
    return img

def train_transformation(image):
    function_list = [blur_motion, gausian_blur, resize_blur, peper_noise, erase_noise, mix_noise]
    function = np.random.choice(function_list)
    # print(function)
    image_aug = function(image)
    # image_aug = alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image = image_aug)["image"]
    image_aug = ToTensorV2()(image = image_aug)["image"]/255.0
    # print(image_aug)
    return image_aug

def test_transformation(image):
    # image = alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image = image)["image"]
    image = ToTensorV2()(image = image)["image"]/255.0
    return image


if __name__ == '__main__':
    img = cv2.imread("/home/tran/ImageRestoration/train_10002_0_mathit.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transform = train_transformation(img)
    print(img_transform.shape)
    plt.imshow(img_transform.permute(1,2,0))
    plt.show()

    # plt.subplot(3,2,1)
    # plt.imshow(img)
    # plt.title("Original")
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(3,2,2)
    # plt.imshow(gausian_blur(img))
    # plt.title("Gaussian Blur")
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(3,2,3)
    # plt.imshow(blur_motion(img))
    # plt.title("Motion Blur")
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(3,2,4)
    # plt.imshow(resize_blur(img))
    # plt.title("Resize Blur")
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(3,2,5)
    # plt.imshow(peper_noise(img))
    # plt.title("Pepper Noise")
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(3,2,6)
    # plt.imshow(mix_noise(img))
    # plt.title("Erase Noise")
    # plt.xticks([]), plt.yticks([])
    # plt.show()



