from train import Args, build
import torch
import cv2
import numpy as np
import matplotlib
from dataset.transformation import train_transformation
import matplotlib.pyplot as plt

args = Args(
    root="/home/tran/ImageRestoration/real_1204/*.png", 
    epochs=100, 
    batch_size=4, 
    dataset="real_1204",
    mgpu="false",
    lrs="true",
    lrs_min=1e-6,
    lr = 1e-3,
    type_lr = "StepLR",
    checkpoint_path = "/home/tran/ImageRestoration/MixNetImageRestoration/Trained models/MixNet_real_1204.pt",
    encoder_block="MSK",
    optim="NAdam",
    img_size = 256,
    choice_gpu = 1
)

if __name__ == '__main__':
    (device, train_dataloader, val_dataloader, \
            model, optimizer, psnr_metric, ssim_metric, \
            loss_fn, checkpoint, scheduler) = build(args)
    
    model.eval()
    img = cv2.imread("/home/tran/ImageRestoration/train_10002_0_mathit.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transformation = train_transformation(img).unsqueeze(0)
    print("transformation.shape:",img_transformation.shape)
    result = model(img_transformation)
    
    print("result.shape:", result.shape)
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.title("Original")
  

    plt.subplot(3,1,2)
    plt.title("Transformation")
    plt.imshow(img_transformation[0].permute(1,2,0).numpy())
  

    plt.subplot(3,1,3)
    plt.title("Deblurred")
    plt.imshow(result[0].detach().permute(1,2,0).numpy())


    plt.show()