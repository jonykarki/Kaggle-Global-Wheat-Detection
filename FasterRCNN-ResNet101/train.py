import os, sys
import argparse
import random
import numpy as np
import pandas as pd
from config import cfg
from utils.logger import init_logger

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(cfg, logger):
    seed_everything(cfg.SEED)
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE 
    checkpoint = cfg.MODEL.CHECKPOINT_PATH


def main():
    parser = argparse.ArgumentParser(description="Train FasterRCNN ResNet-101 Backbone")
    parser.add_argument(
        "--experiment_file", default="", help="path to the experiment file", type=str
    )
    parser.add_argument(
        "opts", default=None, help="Modify config options using command line", nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.experiment_file != "":
        cfg.merge_from_file(args.experiment_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR 
    os.makedirs(output_dir, exist_ok=True)

    logger = init_logger("wheat-detection", output_dir)
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)

    if args.experiment_file != "":
        logger.info(f"Loaded experiment file {args.experiment_file}")
        with open(args.experiment_file, 'r') as cf:
            logger.info("\n" + cf.read())

    logger.info(f"Running experiment from {args.experiment_file}")

    train(cfg, logger)

if __name__ == "__main__":
    main()