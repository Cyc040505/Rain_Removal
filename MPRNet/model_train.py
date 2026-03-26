import os
from Config import Config
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import Utils
from Dataset import get_training_data, get_validation_data
from MPRNet import MPRNet
import Loss
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm


def train():
    start_epoch = 1
    best_psnr = 0
    best_epoch = 0

    save_model_dir = opt.TRAINING.SAVE_MODEL_DIR

    Utils.mkdir(save_model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    # Model
    model_restoration = MPRNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler
    warmup_epochs = 5
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    # scheduler.step()

    # Resume
    if opt.TRAINING.RESUME:
        path_chk_rest = Utils.get_last_path(save_model_dir, '_latest.pth')
        Utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = Utils.load_start_epoch(path_chk_rest) + 1
        Utils.load_optim(optimizer, path_chk_rest)

        checkpoint = torch.load(path_chk_rest)
        best_psnr = checkpoint['best_psnr']
        best_epoch = checkpoint['best_epoch']

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print("==> Resuming Training with learning rate:", new_lr)

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    # Loss
    criterion = Loss.TotalLoss(char_weight=1.0, edge_weight=0.05, freq_weight=0.1)

    # DataLoaders
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS))
    print('===> Loading datasets')

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        print("[Epoch {}]".format(epoch))
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            restored = model_restoration(input_)

            # Compute loss at each stage
            loss, loss_details = criterion(restored, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(Utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'best_psnr': best_psnr,
                            'best_epoch': best_epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(save_model_dir, "model_best.pth"))

            print(
                "Val PSNR: {:.4f} \t Best Val: {:.4f}(epoch {})".format(psnr_val_rgb, best_psnr, best_epoch))

        scheduler.step()

        print("Time: {:.2f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'best_psnr': best_psnr,
                    'best_epoch': best_epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(save_model_dir, "model_latest.pth"))


if __name__ == '__main__':
    # Set Seeds
    random.seed(1002)
    np.random.seed(1002)
    torch.manual_seed(1002)
    torch.cuda.manual_seed_all(1002)

    train()
