import os
import time
from os.path import join

import torch
import yaml

# import wandb


class Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
        # Training Settings
        self.num_workers = cfg["num_workers"]
        self.bs = cfg["bs"]
        self.fold = args.fold
        self.test = args.test_mode
        self.ckpt = args.ckpt
        self.device = torch.device("cuda")

        # Data Settings
        self.path = cfg["path"]
        self.index_folder = cfg["index_folder"]
        self.result = cfg["result"]
        self.num_cls = cfg["num_cls"]
        self.views = [_ for _ in cfg["views"].split(",")]

        # Optimizer settings
        self.lr = cfg["lr"]
        # self.momentum = cfg["momentum"]
        self.wd = cfg["weight_decay"]
        # self.lr_freq = cfg["lr_freq"]
        # self.lr_decay = cfg["lr_decay"]

        # Model Settings
        self.task = cfg["task"]
        self.net = cfg["net"]
        self.input_size = cfg["input_size"]
        self.num_epoch = cfg["num_epoch"]

        self.TIME = time.strftime("%Y-%m-%d-%H-%M-%S")  # time of we run the script

        if self.test:
            self.path_log = join("results", "temp")
            self.path_ckpt = self.path_log
            self.log_dir = join(self.path_log, "test.log")
            self.best_ckpt = join(self.path_ckpt, "best.pth")
            self.last_ckpt = join(self.path_ckpt, "last.pth")
        else:
            self.path_log = join(self.result, "logs", self.task, self.net)
            self.path_ckpt = join(self.result, "checkpoints", self.task, self.net)
            self.log_dir = join(self.path_log, f"{self.fold}-{self.TIME}.log")
            self.best_ckpt = join(self.path_ckpt, f"{self.fold}-{self.TIME}-best.pth")
            self.last_ckpt = join(self.path_ckpt, f"{self.fold}-{self.TIME}-last.pth")
        # if not self.test:
        #     wandb.init(
        #         project=f"Knee-MAE-{self.task}",
        #         name=f"{self.fold}-{self.TIME}",
        #         config={
        #             "task": self.task,
        #             "net": self.net,
        #             "fold": self.fold,
        #             "bs": self.bs,
        #             "data": self.path,
        #             "lr": self.lr,
        #             "wd": self.wd,
        #             "num_epoch": self.num_epoch,
        #             "input_size": self.input_size,
        #         },
        #     )
        #     wandb.config.update(cfg)
        # else:
        #     pass
