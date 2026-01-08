from Utils import *
from Dataset import get_training_data, get_validation_data
from Loss import *
from MPRNet import MPRNet
from tqdm import tqdm
from pdb import set_trace as stx

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import random
import time
import numpy as np


def train_model():
    """模型初始化"""
    model_restoration = MPRNet()
    model_restoration.to(device)

    """优化器"""
    optimizer = optim.Adam(model_restoration.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)

    """学习率调度器"""
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=1e-6 / LR_INITIAL,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        eta_min=LR_MIN
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[WARMUP_EPOCHS]
    )

    """恢复训练"""
    start_epoch = 1
    if RESUME:
        path_chk_rest = get_last_path(checkpoint_dir, '_latest.pth')
        load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = load_start_epoch(path_chk_rest) + 1
        load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('-' * 70)
        print(f"==> Resuming Training with learning rate: {current_lr}")
        print('-' * 70)

    """损失函数"""
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()

    """数据加载器"""
    print('===> Loading datasets')
    train_dataset = get_training_data(TRAIN_DIR, {'patch_size': TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(VAL_DIR, {'patch_size': VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False,
                            num_workers=2, drop_last=False, pin_memory=True)

    """训练循环"""
    print(f'===> Start Epoch: {start_epoch}\tEnd Epoch: {NUM_EPOCHS}\n')

    best_psnr = 0
    best_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"[Epoch {epoch}]")

        epoch_start_time = time.time()
        epoch_loss = 0
        train_psnr = 0
        train_batches = 0

        # 训练阶段
        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # 清空梯度
            optimizer.zero_grad()

            target = data[0].to(device)
            input_ = data[1].to(device)

            # 前向传播
            restored = model_restoration(input_)

            # 计算损失
            loss_char = torch.sum(torch.stack([criterion_char(restored[j], target) for j in range(len(restored))]))
            loss_edge = torch.sum(torch.stack([criterion_edge(restored[j], target) for j in range(len(restored))]))
            loss = loss_char + (0.05 * loss_edge)

            # 反向传播
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 计算训练PSNR（使用最终输出）
            with torch.no_grad():
                restored_final = restored[0]  # 取最终输出
                for res, tar in zip(restored_final, target):
                    train_psnr += torchPSNR(res, tar).item()
                train_batches += len(target)

        # 计算平均训练PSNR
        avg_train_psnr = train_psnr / train_batches if train_batches > 0 else 0

        # 更新学习率
        scheduler.step()

        # 验证阶段（每轮都进行验证）
        model_restoration.eval()
        psnr_val_rgb = []
        val_batches = 0

        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]  # 取最终输出

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(torchPSNR(res, tar))
            val_batches += len(target)

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        # 保存最佳模型
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "model_best.pth"))

        # 每10个epoch保存一次模型
        if epoch % SAVE_AFTER_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))
            print(f"Model saved at epoch {epoch}")

        # 打印训练和验证结果
        print(f"Train PSNR: {avg_train_psnr:.2f}\tVal PSNR: {psnr_val_rgb:.2f} \tLoss: {epoch_loss:.4f}")
        print(f"Time: {time.time() - epoch_start_time}\tBest Val PSNR: {best_psnr:.2f}(Epoch {best_epoch})")
        columns = os.get_terminal_size().columns
        print("=" * columns)

        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr
        }, os.path.join(model_dir, "model_latest.pth"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining completed! Best PSNR: {best_psnr:.2f} at epoch {best_epoch}")


if __name__ == '__main__':
    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nGPU Name:", torch.cuda.get_device_name(0))
    else:
        print("\nGPU is not available. Run CPU.")
        device = torch.device("cpu")

    # 优化器
    BATCH_SIZE = 4
    NUM_EPOCHS = 250
    LR_INITIAL = 2e-4
    LR_MIN = 1e-6

    # 训练
    VAL_AFTER_EVERY = 1
    SAVE_AFTER_EVERY = 10
    RESUME = False
    WARMUP_EPOCHS = 3
    TRAIN_PS = 128
    VAL_PS = 128
    TRAIN_DIR = '../dataset/train'
    VAL_DIR = '../dataset/val'
    TEST_DIR = '../dataset/test'

    # 结果保存
    checkpoint_dir = '../checkpoints'
    mkdir(checkpoint_dir)
    model_dir = '../models'
    mkdir(model_dir)

    # 随机种子
    random.seed(1002)
    np.random.seed(1002)
    torch.manual_seed(1002)
    torch.cuda.manual_seed_all(1002)

    # 模型训练
    train_model()
