# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
import argparse
import logging
import os
from functools import partial
import random
import numpy as np
import torch
import json
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from utils_files import utils, config
from model_main import ModelMain

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def eval_instance_argoverse(batch_size, args, pred, pred_score, mapping, file2pred, file2labels, file2pred_score, DEs,\
                             submission_nuscenes, iter_bar=None, eval_instance_num=0):
    for i in range(batch_size):
        a_pred = pred[i]
        a_pred_score = pred_score[i]
        assert a_pred.shape == (args.mode_num, args.future_frame_num, 2)
        if "nuscenes" in args.other_params:
            file_name_int = eval_instance_num
            eval_instance_num += 1
            instance_nus={"instance":mapping[i]["file_name"],
                          "sample": mapping[i]["sample_token"],
                          "prediction":a_pred.tolist(),
                          "probabilities":a_pred_score.tolist()}
            submission_nuscenes.append(instance_nus)
        else:
          file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2pred_score[file_name_int] = a_pred_score
        if not args.do_test:
            file2labels[file_name_int] = mapping[i]['origin_labels']

    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)
        miss_rate = 0.0
        if 0 in utils.method2FDEs:
            FDEs = utils.method2FDEs[0]
            miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)

        if iter_bar: iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))
    return eval_instance_num, submission_nuscenes


def do_eval(args):
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
    submission_nuscenes = []
    iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)')
    DEs = []
    global eval_instance_num
    eval_instance_num = 0
    for step, batch in enumerate(iter_bar):
        with torch.no_grad():
            pred_trajectory, pred_score, _ = model(batch, device)

        mapping = batch
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (args.mode_num, args.future_frame_num, 2)
            assert pred_score[i].shape == (args.mode_num,)

        eval_instance_num, submission_nuscenes =  eval_instance_argoverse(batch_size, args, pred_trajectory, pred_score, mapping, file2pred, file2labels,\
                                                     file2pred_score,  DEs, submission_nuscenes, iter_bar, eval_instance_num)
    if "nuscenes" in args.other_params and "submission" in args.other_params:
        output_path = 'competition_files_nuscenes/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        json.dump(submission_nuscenes, open(os.path.join(output_path, "evalai_submission.json"), "w"), cls=NpEncoder)   
    from datascripts.dataset_argoverse import post_eval
    post_eval(args, file2pred, file2labels, file2pred_score, DEs)


def main():
    parser = argparse.ArgumentParser()
    config.add_argument(parser)
    args: config.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_eval(args)


if __name__ == "__main__":
    main()
