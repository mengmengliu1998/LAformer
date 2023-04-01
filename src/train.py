# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
import argparse
import logging
import os
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model_main import ModelMain
from utils_files import utils, config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm_, dynamic_ncols=True)


def is_main_device(device):
    return isinstance(device, torch.device) or device == 0



def learning_rate_decay_nuscenes_10(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch

    if i_epoch > 0 and i_epoch % 10 == 0:
        for p in optimizer.param_groups: p['lr'] *= 0.5

def learning_rate_decay_nuscenes_5(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch
    if i_epoch > 0 and i_epoch % 16 == 0:
        for p in optimizer.param_groups: p['lr'] *= 0.3

def learning_rate_decay_nuscenes(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch
    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups: p['lr'] *= 0.3


def learning_rate_decay(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch

    if 'stage_two' in args.other_params:
        if i_epoch > 0 and i_epoch % 4 == 0:
            for p in optimizer.param_groups: p['lr'] *= 0.3

    else:
        if i_epoch > 0 and i_epoch % 3 == 0:
            for p in optimizer.param_groups: p['lr'] *= 0.3

def learning_rate_decay_16(args, i_epoch, optimizer, optimizer_2=None):
    utils.i_epoch = i_epoch

    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups: p['lr'] *= 0.3


def train_one_epoch(model, iter_bar, optimizer, device, args: config.Args, i_epoch, queue=None, optimizer_2=None):
    model.train()
    li_FDE = []
    utils.other_errors_dict.clear()
    if 'data_ratio_per_epoch' in args.other_params:
        max_iter_num = int(float(args.other_params['data_ratio_per_epoch']) * len(iter_bar))
        if is_main_device(device):
            print('data_ratio_per_epoch', float(args.other_params['data_ratio_per_epoch']))

    if args.distributed_training:
        assert dist.get_world_size() == args.distributed_training

    for step, batch in enumerate(iter_bar):
        if 'data_ratio_per_epoch' in args.other_params:
            max_iter_num -= 1
            if max_iter_num == 0:
                break
        loss, DE, _ = model(batch, device)
        loss.backward()
        if step%100 == 0:
        # if is_main_device(device):
            utils.logging(f'loss={loss.item():.3f}',f'step: {step}')
            print(f'loss={loss.item():.3f}',f'step: {step}')

        final_idx = batch[0].get('final_idx', -1)
        li_FDE.extend([each for each in DE[:, final_idx]])

        if optimizer_2 is not None:
            optimizer_2.step()
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

    if not args.debug and is_main_device(device) :
        if "stage_two" in args.other_params:
            train_epoch = args.other_params['stage-two-epoch']
        else:
            train_epoch = args.num_train_epochs
        if train_epoch > 16 and (i_epoch+1)%10==0:
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.model_save_dir, "model.{0}.bin".format(i_epoch + 1))
            torch.save(model_to_save.state_dict(), output_model_file)
        elif train_epoch==16:
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.model_save_dir, "model.{0}.bin".format(i_epoch + 1))
            torch.save(model_to_save.state_dict(), output_model_file)

    if is_main_device(device):
        for i in range(args.distributed_training - 1):
            other_errors_dict_ = queue.get()
            for key in utils.other_errors_dict:
                utils.other_errors_dict[key].extend(other_errors_dict_[key])
    else:
        queue.put(utils.other_errors_dict)

    if is_main_device(device):
        print()
        miss_rates = (utils.get_miss_rate(li_FDE, dis=2.0), utils.get_miss_rate(li_FDE, dis=4.0),
                      utils.get_miss_rate(li_FDE, dis=6.0))

        utils.logging(f'FDE: {np.mean(li_FDE) if len(li_FDE) > 0 else None}',
                      f'MR(2m,4m,6m): {miss_rates}',
                      type='train_loss', to_screen=True)


def run_training(rank, world_size, kwargs, queue):
    args = kwargs['args']
    if world_size > 0:
        print(f"Running DDP on rank {rank}.")

        def setup(rank, world_size):
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = args.master_port
            # initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        setup(rank, world_size)

        utils.args = args
        model = ModelMain(args).to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = ModelMain(args).to(rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if rank == 0 and world_size > 0:
        receive = queue.get()
        assert receive == True

    if args.distributed_training:
        dist.barrier()
    args.reuse_temp_file = True

    from datascripts.dataset_argoverse import Dataset
    train_dataset = Dataset(args, args.train_batch_size, to_screen=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=args.do_train)
    assert args.train_batch_size % world_size == 0
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size // world_size,
        collate_fn=utils.batch_list_to_batch_tensors)

    continue_training_epoch_cnt = 0
    if args.model_recover_path is not None:
        continue_training_epoch_cnt = int(args.model_recover_path[-6:-4])
        print("Continue training with model: " + args.model_recover_path)
        model_recover = torch.load(args.model_recover_path)
        model_recover_rename_key = {}
        for key in model_recover.keys():
            model_recover_rename_key['module.'+key] = model_recover[key]
        model.load_state_dict(model_recover_rename_key, strict=True)
        model.to(rank)


    for i_epoch in range(continue_training_epoch_cnt):
        if "nuscenes" in args.other_params:
            if int(train_epoch)  == 50:
                learning_rate_decay_nuscenes_5(args, i_epoch, optimizer)
            elif int(train_epoch)  == 100:
                learning_rate_decay_nuscenes_10(args, i_epoch, optimizer)
            else:
                learning_rate_decay_nuscenes(args, i_epoch, optimizer)
        else:
            if int(train_epoch) == 16:
                learning_rate_decay_16(args, i_epoch, optimizer)
            else:
                learning_rate_decay(args, i_epoch, optimizer)
    if "stage_two" in args.other_params:
        train_epoch = args.other_params['stage-two-epoch']
    else:
        train_epoch = args.num_train_epochs
    for i_epoch in range(continue_training_epoch_cnt, continue_training_epoch_cnt+int(train_epoch)):
        if "nuscenes" in args.other_params:
            if int(train_epoch)  == 50:
                learning_rate_decay_nuscenes_5(args, i_epoch, optimizer)
            elif int(train_epoch)  == 100:
                learning_rate_decay_nuscenes_10(args, i_epoch, optimizer)
            else:
                learning_rate_decay_nuscenes(args, i_epoch, optimizer)
        else:
            if int(train_epoch) == 16:
                learning_rate_decay_16(args, i_epoch, optimizer)
            else:
                learning_rate_decay(args, i_epoch, optimizer)
        utils.logging(optimizer.state_dict()['param_groups'])
        if rank == 0:
            print('Epoch: {}/{}'.format(i_epoch, int(train_epoch)), end='  ')
            print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
        train_sampler.set_epoch(i_epoch)
        # if rank == 0:
        #     iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        # else:
        iter_bar = train_dataloader

        train_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue)
        

        if args.distributed_training:
            dist.barrier()
    if args.distributed_training:
        dist.destroy_process_group()


def distributed_run(args):
    from datascripts.dataset_argoverse import Dataset
    if args.distributed_training:
        queue = mp.Manager().Queue()
        kwargs = {'args': args}
        spawn_context = mp.spawn(run_training,
                                args=(args.distributed_training, kwargs, queue),
                                nprocs=args.distributed_training,
                                join=False)
        _ = Dataset(args, args.train_batch_size)  # to generate tmp file for distributed_training
        queue.put(True)
        while not spawn_context.join():
            pass
    else:
        assert False, 'Please set "--distributed_training 1" to use single gpu'


def main():
    parser = argparse.ArgumentParser()
    config.add_argument(parser)
    args: config.Args = parser.parse_args()
    utils.init(args, logger)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    distributed_run(args)

    logger.info('Finish.')


if __name__ == "__main__":
    main()
