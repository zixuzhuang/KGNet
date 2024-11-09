import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--config_file", type=str, default="CSNet/csnet_subject.yaml")
parser.add_argument("--test_mode", action="store_true", help="Test Mode")
parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--dataset", type=str, default='kgnet')
args = parser.parse_args()
