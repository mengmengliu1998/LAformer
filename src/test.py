# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
import argparse
import logging
import os, sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from functools import partial
import random
import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from utils_files import utils, config
from model_main import ModelMain
from argoverse.evaluation.competition_util import generate_forecasting_h5

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)

def eval_instance_argoverse(batch_size, args, pred, pred_score, mapping, file2pred, file2labels, file2pred_score, DEs, iter_bar=None):
    for i in range(batch_size):
        a_pred = pred[i]
        a_pred_score = pred_score[i]
        assert a_pred.shape == (6, args.future_frame_num, 2)
        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2pred_score[file_name_int] = a_pred_score
        if not args.do_test:
            file2labels[file_name_int] = mapping[i]['origin_labels']

def do_test(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading Evalute Dataset", args.data_dir)
    from datascripts.dataset_argoverse import Dataset
    eval_dataset = Dataset(args, args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                  sampler=eval_sampler,
                                                  collate_fn=utils.batch_list_to_batch_tensors,
                                                  pin_memory=False)
    model = ModelMain(args)
    print('torch.cuda.device_count', torch.cuda.device_count())
    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")

    model_recover = torch.load(args.model_recover_path)
    model.load_state_dict(model_recover, strict=False)

    model.to(device)
    model.eval()
    file2pred = {}
    file2labels = {}
    file2pred_score = {}
    iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)')
    DEs = []

    for step, batch in enumerate(iter_bar):
        pred_trajectory, pred_score, _ = model(batch, device)

        mapping = batch
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
            assert pred_score[i].shape == (6,)

        eval_instance_argoverse(batch_size, args, pred_trajectory, pred_score, mapping, file2pred, file2labels, file2pred_score, DEs, iter_bar)
    output_path = 'competition_files/'
    generate_forecasting_h5(file2pred, output_path, probabilities=file2pred_score, filename="LAformer")

def main():
    parser = argparse.ArgumentParser()
    config.add_argument(parser)
    args: config.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_test(args)

if __name__ == "__main__":
    main()
