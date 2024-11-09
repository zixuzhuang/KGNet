import logging
import os

import torch


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


def get_lr(epoch, cfg):
    lr = cfg.lr * (cfg.lr_decay ** (epoch // cfg.lr_freq + 1))
    return lr


def init_train(cfg):
    if not os.path.exists(cfg.path_log):
        os.makedirs(cfg.path_log)
    initLogging(cfg.log_dir)
    logging.info("Input: " + cfg.path)
    logging.info("Log: " + cfg.log_dir)

    forma = ("\n" + "|{:^9}" * 4 + "|") * 2
    title = ["NET", "BS", "FOLD", "LR"]
    items = [cfg.net, cfg.bs, cfg.fold, cfg.lr]
    logging.info(forma.format(*title, *items))


def save_model(result, net, cfg):
    if not os.path.exists(cfg.path_ckpt):
        os.makedirs(cfg.path_ckpt)
    if result.best_epoch == result.epoch:
        torch.save(net, cfg.best_ckpt)
    logging.info(f"BEST EPOCH: {result.best_epoch}, RESULT: {result.best_result:.3f}")
    return
