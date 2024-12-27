import os
import torch
import numpy as np
import time
from engine import trainer
from Params import args
from tqdm import tqdm
import logging
from utils import seed_torch

# 记录所有参数
def log_params(args):
    logging.info("Model Parameters:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

def main():
    # 1. config and init
    seed_torch()  # set random seed
    device = args.device
    print(device)
    engine = trainer()
    print("start training...", flush=True)
    # log config
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=args.log_url,
                    filemode='a')
    # log_params(args)

    # 2. train and evaluate
    for i in tqdm(range(1, args.epoch+1)):
        t1 = time.time()
        engine.train(i)
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')      
        print()

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))