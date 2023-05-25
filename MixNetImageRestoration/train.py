import torch
from dataset.datasets import preprocess_test, preprocess_train
from dataset.transformation import test_transformation, train_transformation
from model.mixNet import MixNet
import torch.nn as nn
import glob
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from losses.losses import PSNRLoss
import time
import sys
import numpy as np
import os
from utils.img_utils import tensor2img


class Args:
    def __init__(self,root, epochs, batch_size, dataset, mgpu, lrs_min,\
                 lrs, lr, type_lr, checkpoint_path, encoder_block, optim, img_size):
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.mgpu = mgpu
        self.lrs_min = lrs_min
        self.lrs = lrs
        self.lr = lr
        self.type_lr = type_lr
        self.checkpoint_path = checkpoint_path
        self.encoder_block = encoder_block
        self.optim = optim
        self.img_size = img_size
        


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

 
    img_path = glob.glob(args.root) 
    train_dataloader = preprocess_train(img_path, train_transformation, args.batch_size)
    val_dataloader = preprocess_test(img_path, test_transformation, 1)
    

    model = MixNet()
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    #===================== Optimizer ===================================================
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optim == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    elif args.optim == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "ASGD":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
    elif args.optim == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    #===================================================================================

    psnr_metric = calculate_psnr
    ssim_metric = calculate_ssim
    loss_fn = PSNRLoss()

    if args.lrs == "true":
        if args.type_lr == "LROnP":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, mode="max", patience=5, factor=0.75, min_lr=args.lrs_min, verbose=True)
        elif args.type_lr == "StepLR":
            print("Using StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(
                  optimizer, step_size=20, gamma=0.5, verbose=False)
        elif args.type_lr == "MultiStepLR":
            print("Using MultiStepLR")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                  optimizer, milestones=[10, 20, 30, 60], gamma=0.5, verbose=False)

        
    if args.checkpoint_path == None:
        checkpoint = {"test_measure_psnr":None, "epoch":0}
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (device, train_dataloader, val_dataloader, \
            model, optimizer, psnr_metric, ssim_metric, \
            loss_fn, checkpoint, scheduler)



def train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for k in range(0, data.shape[0], 4):
            data_input = data[k:k + 4]
            target_input = target[k:k+4]
            output = model(data_input)
            loss = loss_fn(output, target_input)
            loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(), time.time() - t, ), end="", )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator), time.time() - t, ) )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, psnr_metric, ssim_metric, phase):
    t = time.time()
    model.eval()
    PSNR_metrics = []
    SSIM_metrics = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        # print(output.shape)
        output_change_format = tensor2img(output, rgb2bgr=True)
        target_change_formate = tensor2img(target, rgb2bgr=True)
        psnr = psnr_metric(output_change_format, target_change_formate, crop_border=0)
        ssim = ssim_metric(output_change_format, target_change_formate, crop_border=0)
        PSNR_metrics.append(psnr)
        SSIM_metrics.append(ssim)

        
        if batch_idx + 1 < len(test_loader):
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tPSNR: {:.6f}\tSSIM: {:.6f}\tTime: {:.6f}".format(
                    phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(PSNR_metrics), np.mean(SSIM_metrics), time.time() - t, ), end="", )
        else:
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tPSNR: {:.6f}\tSSIM: {:.6f}\tTime: {:.6f}".format(
                    phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(PSNR_metrics), np.mean(SSIM_metrics), time.time() - t, ))

    return np.mean(PSNR_metrics), np.mean(SSIM_metrics)



def train(args):
    (device, train_dataloader, val_dataloader, \
            model, optimizer, psnr_metric, ssim_metric, \
            loss_fn, checkpoint, scheduler) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = checkpoint["test_measure_psnr"]
    print("best test:", prev_best_test, "epoch:", checkpoint["epoch"])
    
    for epoch in range(1, args.epochs + 1):
        try:
            # loss = train_epoch(
            #     model, device, train_dataloader, optimizer, epoch, loss_fn
            # )
            test_measure_psnr, test_measure_ssim = test(
                model, device, val_dataloader, epoch, psnr_metric, ssim_metric, "val"
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            if args.type_lr == "LROnP":
                scheduler.step(test_measure_psnr)
            else:
                scheduler.step()
        if prev_best_test == None or test_measure_psnr > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "loss": loss,
                    "test_measure_psnr": test_measure_psnr,
                    "test_measure_ssim": test_measure_ssim,
                },
                f"./Trained models/MixNet_" + args.dataset + ".pt",
            )
            prev_best_test = test_measure_psnr


def main(args):
    train(args)


if __name__ == "__main__":
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
    checkpoint_path = None,
    encoder_block="MSK",
    optim="NAdam",
    img_size = 256
)
    
    main(args)
    

