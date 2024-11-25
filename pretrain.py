import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# import wandb
from network.kgnet import KneeGraphNetwork, KneeLoss
from network.dataloader import KGNetDataloader as Dataloader
from utils.Config import Config
from utils.Result_pre import Result
from utils.utils_net import get_lr, init_train, save_model
from utils.parser import args


def train():
    st = time.time()
    running_loss = 0.0
    lr = scheduler.get_last_lr()[0]
    net.train()
    for data in tqdm(dataloader["train"], ncols=60, desc="train", unit="b", leave=None):
        data.to(cfg.device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            preds = net(data)
            loss = lossfunc.pre(data, preds)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
    scheduler.step()
    ft = time.time()
    e_loss = running_loss / len(dataloader["train"])
    logging.info(
        f"\n\nEPOCH: {epoch}, TRAIN_LOSS : {e_loss:.3f}, TIME: {ft - st:.1f}s, LR: {lr:.2e}"
    )
    # if not args.test_mode:
    #     wandb.log({"train_loss": e_loss, "learning_rate": lr}, step=epoch)
    return e_loss


@torch.no_grad()
def eval(dataset_type):
    net.eval()
    result.init()
    for data in dataloader[dataset_type]:
        data.to(cfg.device)
        preds = net(data)
        loss = lossfunc.pre(data, preds)
        result.eval(data, preds, loss)
    result.stastic(epoch)
    result.show()
    # if not args.test_mode:
    #     wandb.log({"test_loss": loss})
    return


if __name__ == "__main__":

    scaler = GradScaler()
    cfg = Config(args)
    init_train(cfg)
    net = KneeGraphNetwork()
    net.load_pretrain(cfg.ckpt)
    net = net.to(cfg.device)
    lossfunc = KneeLoss(cfg.device, args.dataset)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, cfg.num_epoch, 1e-8)
    result = Result(num_seg=8, save_path=cfg.log_dir)

    net.forward = net.pretrain
    dataloader = Dataloader(cfg)

    for epoch in range(cfg.num_epoch):
        train()
        eval('valid')
        save_model(result, net, cfg)
        eval('test')
        
    # if not args.test_mode:
    #     wandb.finish()
