import os
import time
from os.path import join

import torch
import yaml


class Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = yaml.load(open(args.c), Loader=yaml.FullLoader)
        # Training Settings
        self.num_workers = cfg["num_workers"]
        self.bs = cfg["bs"]
        self.fold = args.f
        self.pre = args.pre
        self.device = torch.device("cuda")

        # Data Settings
        self.path = cfg["path"]
        self.index_folder = cfg["index_folder"]
        self.result = cfg["result"]
        self.num_cls = cfg["num_cls"]

        # Optimizer settings
        self.lr = cfg["lr"]
        self.wd = cfg["weight_decay"]

        # Model Settings
        self.task = cfg["task"]
        self.net = cfg["net"]
        self.input_size = cfg["input_size"]
        self.num_epoch = cfg["num_epoch"]

        self.TIME = time.strftime("%Y-%m-%d-%H-%M-%S")  # time of we run the script
