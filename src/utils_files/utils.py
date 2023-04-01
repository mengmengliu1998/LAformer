import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
from statistics import mode
import subprocess
import sys
import imageio
import time
import torch.nn as nn
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path
from matplotlib.pyplot import MultipleLocator
from torch import Tensor
from utils_files import  config
from utils_files.config import *

args: config.Args = None
logger = None

def init(args_: config.Args, logger_):
    global args, logger
    args = args_
    logger = logger_

    if not args.do_eval and not args.debug and os.path.exists(args.output_dir):
        print('{} {} exists'.format(get_color_text('Warning!'), args.output_dir))
        # input() # skip input as we may train on cluster using task

    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.temp_file_dir is None:
        args.temp_file_dir = os.path.join(args.output_dir, 'temp_file')
    else:
        args.reuse_temp_file = True
        args.temp_file_dir = os.path.join(args.temp_file_dir, 'temp_file')

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.temp_file_dir, exist_ok=True)
    if not args.do_eval and not args.debug:
        src_dir = os.path.join(args.output_dir, 'src')
        if os.path.exists(src_dir):
            subprocess.check_output('rm -r {}'.format(src_dir), shell=True, encoding='utf-8')
        os.makedirs(src_dir, exist_ok=False)
        for each in os.listdir('src'):
            is_dir = '-r' if os.path.isdir(os.path.join('src', each)) else ''
            subprocess.check_output(f'cp {is_dir} {os.path.join("src", each)} {src_dir}', shell=True, encoding='utf-8')
        with open(os.path.join(src_dir, 'cmd'), 'w') as file:
            file.write(' '.join(sys.argv))
    if "hard_mining_train" in args.other_params:
        args.model_save_dir = os.path.join(args.output_dir, 'model_save_hard_mining')
        os.makedirs(args.model_save_dir, exist_ok=True)
    else:
        args.model_save_dir = os.path.join(args.output_dir, 'model_save')
        os.makedirs(args.model_save_dir, exist_ok=True)

    if "nuscenes" in args.other_params:
        args.mode_num = args.other_params['nuscenes_mode_num']
        args.future_frame_num = 12

    def init_args_do_eval():
        args.data_dir = args.data_dir_for_val if not args.do_test else 'test_obs/data/'
        if args.model_recover_path is None:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.16.bin')
        elif len(args.model_recover_path) <= 3:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save',
                                                   'model.{}.bin'.format(args.model_recover_path))
        args.do_train = False

        if 'set_predict' in args.other_params:
            # TODO
            pass

        if len(args.method_span) != 2:
            args.method_span = [args.method_span[0], args.method_span[0] + 1]

        if args.mode_num != 6:
            add_eval_param(f'mode_num={args.mode_num}')

    def init_args_do_train():
        if args.model_recover_path is not None:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save',
                                                   'model.{}.bin'.format(args.model_recover_path))
        # if 'interactive' in args.other_params:
        #     args.data_dir = 'tf_example/validation_interactive/'
        pass

    if args.do_eval:
        init_args_do_eval()
    else:
        init_args_do_train()

    print(dict(sorted(vars(args_).items())))
    # print(json.dumps(vars(args_), indent=4))
    args_dict = vars(args)
    print()
    logger.info("***** args *****")
    for each in ['output_dir', 'other_params']:
        if each in args_dict:
            temp = args_dict[each]
            if each == 'other_params':
                temp = [param if args.other_params[param] is True else (param, args.other_params[param]) for param in
                        args.other_params]
            print("\033[31m" + each + "\033[0m", temp)
    logging(vars(args_), type='args', is_json=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.temp_file_dir, time_begin), exist_ok=True)

    if isinstance(args.data_dir, str):
        args.data_dir = [args.data_dir]

    assert args.do_train or args.do_eval


def add_eval_param(param):
    if param not in args.eval_params:
        args.eval_params.append(param)


def get_name(name='', append_time=False):
    if name.endswith(time_begin):
        return name
    prefix = 'test.' if args.do_test else 'eval.' if args.do_eval and not args.do_train else ''
    prefix = 'debug.' + prefix if args.debug else prefix
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


eps = 1e-5

origin_point = None
origin_angle = None


def get_pad_vector_nus(li, vector_size):
    """
    Pad vector to length of args.hidden_size
    """
    assert len(li) <= vector_size
    li.extend([0] * (vector_size - len(li)))
    return li


def get_pad_vector(li):
    """
    Pad vector to length of args.hidden_size
    """
    # assert len(li) <= args.hidden_size
    # li.extend([0] * (args.hidden_size - len(li)))
    assert len(li) <= args.vector_size
    li.extend([0] * (args.vector_size - len(li)))
    return li


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def round_value(v):
    return round(v / 100)


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def get_sub_matrix(traj, object_type, x=0, y=0, angle=None):
    res = []
    for i in range(0, len(traj), 2):
        if i > 0:
            vector = [traj[i - 2] - x, traj[i - 1] - y, traj[i] - x, traj[i + 1] - y]
            if angle is not None:
                vector[0], vector[1] = rotate(vector[0], vector[1], angle)
                vector[2], vector[3] = rotate(vector[2], vector[3], angle)
            res.append(vector)
    return res


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_(x, y, cos, sin):
    res_x = x * cos - y * sin
    res_y = x * sin + y * cos
    return res_x, res_y


index_file = 0

file2pred = {}


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


files_written = {}


def logging(*inputs, prob=1.0, type='1', is_json=False, affi=True, sep=' ', to_screen=False, append_time=False, as_pickle=False):
    """
    Print args into log file in a convenient style.
    """
    if to_screen:
        print(*inputs, sep=sep)
    if not random.random() <= prob or not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, get_name(type, append_time))
    if as_pickle:
        with open(file, 'wb') as pickle_file:
            assert len(inputs) == 1
            pickle.dump(*inputs, pickle_file)
        return
    if file not in files_written:
        with open(file, "w", encoding='utf-8') as fout:
            files_written[file] = 1
    inputs = list(inputs)
    the_tensor = None
    for i, each in enumerate(inputs):
        if isinstance(each, torch.Tensor):
            # torch.Tensor(a), a must be Float tensor
            if each.is_cuda:
                each = each.cpu()
            inputs[i] = each.data.numpy()
            the_tensor = inputs[i]
    np.set_printoptions(threshold=np.inf)

    with open(file, "a", encoding='utf-8') as fout:
        if is_json:
            for each in inputs:
                print(json.dumps(each, indent=4), file=fout)
        elif affi:
            print(*tuple(inputs), file=fout, sep=sep)
            if the_tensor is not None:
                print(json.dumps(the_tensor.tolist()), file=fout)
            print(file=fout)
        else:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)


mpl.use('Agg')


def larger(a, b):
    return a > b + eps


def equal(a, b):
    return True if abs(a - b) < eps else False


def get_valid_lens(matrix: np.ndarray):
    valid_lens = []
    for i in range(matrix.shape[0]):
        ok = False
        for j in range(2, matrix.shape[1], 2):
            if equal(matrix[i][j], 0) and equal(matrix[i][j + 1], 0):
                ok = True
                valid_lens.append(j)
                break

        assert ok
    return valid_lens


visualize_num = 0
def visualize_goals_2D(mapping, future_frame_num, labels: np.ndarray = None,
                        predict: np.ndarray = None, mode_num = 6):
    print("mapping['file_name']",mapping['file_name'])
    print('in visualize_goals_2D', mapping['file_name'])
    print('speed', mapping.get('seep', None))
    plot_last_obs = True
    assert predict is not None
    predict = predict.reshape([mode_num, future_frame_num, 2])
    assert labels.shape == (future_frame_num, 2)

    assert labels is not None
    labels = labels.reshape([-1])

    fig_scale = 1.0
    marker_size_scale = 2
    target_agent_color, target_agent_edge_color = '#4bad34', '#c5dfb3'

    def get_scaled_int(a):
        return round(a * fig_scale)

    plt.cla()
    fig = plt.figure(0, figsize=(get_scaled_int(45), get_scaled_int(38)))

    if True:
        plt.xlim(-100, 100)
        plt.ylim(-30, 100)

    # plt.figure(0, dpi=300)
    cmap = plt.cm.get_cmap('Reds')
    vmin = np.log(0.0001)
    vmin = np.log(0.00001)

    name = os.path.split(mapping['file_name'])[1].split('.')[0] # argoverse only

    add_end = True

    linewidth = 5

    for lane in mapping['vis_lanes']:
        lane = lane[:, :2]
        assert lane.shape == (len(lane), 2), lane.shape
        plt.plot(lane[:, 0], lane[:, 1], linestyle="-", color="black", marker=None,
                 markersize=0,
                 alpha=0.5,
                 linewidth=2,
                 zorder=0)

    yaw_0 = None

    def draw_his_trajs():
        # trajs = mapping['trajs']
        trajs = mapping["past_traj"]
        # print('trajs', trajs)
        trajs = [trajs]
        for i, traj in enumerate(trajs):
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 2 and traj.shape[1] == 2, traj.shape
            if i == 0:
                traj = np.array(traj).reshape([-1])
                t = np.zeros(len(traj) + 2)
                t[:len((traj))] = traj
                t[-2] = labels[0]
                t[-1] = labels[1]

                plt.plot(t[0::2], t[1::2], linestyle="-", color=target_agent_color, marker=None,
                         alpha=1,
                         linewidth=linewidth,
                         zorder=0)
                if plot_last_obs:
                    plt.plot(traj[-2], traj[-1], markersize=10 * marker_size_scale, color=target_agent_color, marker='o',
                            markeredgecolor='black')

    draw_his_trajs()

    if True:
        for each in predict:
            #add last observed point
            new_each = np.zeros([len(each) + 1, 2])
            new_each[1:] = each
            new_each[0:1] = np.array([0, 0])
            function2 = plt.plot(new_each[:, 0], new_each[:, 1], linestyle="-", color="darkorange", marker=None,
                                 linewidth=linewidth)

            if add_end:
                plt.plot(each[-1, 0], each[-1, 1], markersize=15 * marker_size_scale, color="darkorange", marker="*",
                         markeredgecolor='black')

        if add_end and "visualize_test" not in args.other_params:
            plt.plot(labels[-2], labels[-1], markersize=15 * marker_size_scale, color=target_agent_color, marker="*",
                     markeredgecolor='black')
        if "visualize_test" not in args.other_params:
            function1 = plt.plot(labels[0::2], labels[1::2], linestyle="-", color=target_agent_color, linewidth=linewidth)
    if "visualize_test" not in args.other_params:
      functions = function1 + function2
    else:
      functions = function2
    fun_labels = [f.get_label() for f in functions]
    plt.legend(functions, fun_labels, loc=0)

    ax = plt.gca()
    ax.set_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.label.set_color('black')        #setting up X-axis label color to black
    ax.yaxis.label.set_color('black')          #setting up Y-axis label color to black
    ax.set_xlim(-40, 40) # now ax is defined
    ax.set_ylim(-20, 60) 
    global visualize_num
    visualize_num += 1
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin), exist_ok=True)
    plt.savefig(os.path.join(args.log_dir, 'visualize_' + time_begin,
                             get_name("visualize" + ("" if name == "" else "_" + name) + ".png")), bbox_inches='tight', dpi=80)
    plt.close()


_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}
color_dict = {"AGENT": '#4bad34', "OTHERS": "#d3e8ef", "AV": "#007672"}
def visualize_gifs(mapping, future_frame_num, labels: np.ndarray = None,
                        predict: np.ndarray = None, mode_num = 6):

    print("mapping['file_name']",mapping['file_name'])
    plot_last_obs = True
    assert predict is not None
    predict = predict.reshape([mode_num, future_frame_num, 2])
    assert labels.shape == (future_frame_num, 2)

    assert labels is not None
    labels = labels.reshape([-1])

    fig_scale = 1.0
    marker_size_scale = 2
    target_agent_color, target_agent_edge_color = '#4bad34', '#c5dfb3'

    def get_scaled_int(a):
        return round(a * fig_scale)
    name = os.path.split(mapping['file_name'])[1].split('.')[0] # argoverse only
    imgs = []
    for frame in range(1, 50):

        plt.cla()
        fig = plt.figure(0, figsize=(get_scaled_int(45), get_scaled_int(38)))

        if True:
            plt.xlim(-100, 100)
            plt.ylim(-30, 100)

        # plt.figure(0, dpi=300)
        cmap = plt.cm.get_cmap('Reds')
        vmin = np.log(0.0001)
        vmin = np.log(0.00001)

        name = os.path.split(mapping['file_name'])[1].split('.')[0] # argoverse only


        add_end = True

        linewidth = 5

        for lane in mapping['vis_lanes']:
            lane = lane[:, :2]
            assert lane.shape == (len(lane), 2), lane.shape
            plt.plot(lane[:, 0], lane[:, 1], linestyle="-", color="black", marker=None,
                    markersize=0,
                    alpha=0.5,
                    linewidth=2,
                    zorder=0)

        yaw_0 = None

        def draw_his_trajs():
            trajs = mapping['trajs']
            # trajs = mapping["past_traj"]
            for i, traj in enumerate(trajs):
                assert isinstance(traj, np.ndarray)
                assert traj.ndim == 2 and traj.shape[1] == 2, traj.shape
                if i == 0:
                    if frame < 50:
                        traj = np.array(traj)[:frame+1].reshape([-1])
                    else:
                        traj = np.array(traj).reshape([-1])
                    if frame < 50:
                        t = traj

                    plt.plot(t[0::2], t[1::2], linestyle="-", color=target_agent_color, marker=None,
                            alpha=1,
                            linewidth=linewidth,
                            zorder=_ZORDER["AGENT"])
                    # current_pos 
                    if frame < 49:
                        plt.plot(t[-2], t[-1], markersize=10 * marker_size_scale, color=target_agent_color, marker='^',
                                markeredgecolor='black', zorder=_ZORDER["AGENT"])
                    if frame > 20:
                        plt.plot(t[38], t[39], markersize=10 * marker_size_scale, color=target_agent_color, marker='o',
                                markeredgecolor='black', zorder=_ZORDER["AGENT"])
                else:
                    if len(traj) >= 2:
                        color = "darkblue"
                        if frame < 50:
                            traj = np.array(traj)[:frame+1]
                        else:
                            traj = np.array(traj)
                        plt.plot(traj[:, 0], traj[:, 1], linestyle="-", color=color_dict['OTHERS'], marker=None,
                                    alpha=1,
                                    linewidth=linewidth,
                                    zorder=_ZORDER["OTHERS"])
                    if plot_last_obs:
                        plt.plot(traj[-1, 0], traj[-1, 1], markersize=10 * marker_size_scale, color=color_dict['OTHERS'], marker='o',
                                zorder=_ZORDER["OTHERS"])

        draw_his_trajs()

        if frame > 19:
            for each in predict:
                #add last observed point
                new_each = np.zeros([len(each) + 1, 2])
                new_each[1:] = each
                new_each[0:1] = np.array([0, 0])
                new_each = new_each[:frame-18]
                function2 = plt.plot(new_each[:, 0], new_each[:, 1], linestyle="-", color="darkorange", marker=None,
                                    linewidth=linewidth, alpha=1, zorder=_ZORDER["AGENT"])

                if add_end and frame==49:
                    plt.plot(each[-1, 0], each[-1, 1], markersize=15 * marker_size_scale, color="darkorange", marker="*",
                            markeredgecolor='black', zorder=_ZORDER["AGENT"])
                else:
                    plt.plot(new_each[-1, 0], new_each[-1, 1], markersize=10 * marker_size_scale, color="darkorange", marker="^",
                            markeredgecolor='black', zorder=_ZORDER["AGENT"])

            if add_end and "visualize_test" not in args.other_params and frame==49:
                plt.plot(labels[-2], labels[-1], markersize=15 * marker_size_scale, color=target_agent_color, marker="*",
                        markeredgecolor='black', zorder=_ZORDER["AGENT"])
            if "visualize_test" not in args.other_params:
                label_draw = labels[:(frame-19)*2]
                function1 = plt.plot(label_draw[0::2], label_draw[1::2], linestyle="-", color=target_agent_color, linewidth=linewidth, zorder=_ZORDER["AGENT"])
            if "visualize_test" not in args.other_params:
                functions = function1 + function2
            else:
                functions = function2
            fun_labels = [f.get_label() for f in functions]
            plt.legend(functions, fun_labels, loc=0)
        ax = plt.gca()
        ax.set_aspect(1)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.label.set_color('black')        #setting up X-axis label color to black
        ax.yaxis.label.set_color('black')          #setting up Y-axis label color to black
        ax.set_xlim(-40, 40) # now ax is defined
        ax.set_ylim(-20, 60) 
        global visualize_num
        visualize_num += 1
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin, 
                        get_name("visualize" + ("" if name == "" else "_" + name))), exist_ok=True)
        plt.savefig(os.path.join(args.log_dir, 'visualize_' + time_begin, get_name("visualize" + ("" if name == "" else "_" + name)),
                                str(frame) + ".png"), bbox_inches='tight', dpi=50)
        plt.close()
    os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin), exist_ok=True)
    filename = os.path.join(args.log_dir, 'visualize_' + time_begin,
                            get_name("visualize" + ("" if name == "" else "_" + name) + ".gif"))
    imgs = []
    for file in os.listdir(os.path.join(args.log_dir, 'visualize_' + time_begin, get_name("visualize" + ("" if name == "" else "_" + name)))):
        img = imageio.imread(os.path.join(args.log_dir, 'visualize_' + time_begin, get_name("visualize" + ("" if name == "" else "_" + name)), file))
        imgs.append(img)
    for i in range(10):
        imgs.append(imgs[-1])
    imageio.mimsave(filename, imgs, format='GIF', fps=5)


def load_model(model, state_dict, prefix=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix)

    if logger is None:
        return

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, json.dumps(missing_keys, indent=4)))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, json.dumps(unexpected_keys, indent=4)))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def batch_origin_init(mapping):
    global origin_point, origin_angle
    batch_size = len(mapping)
    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']


def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    hidden_size = args.hidden_size if hidden_size is None else hidden_size
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    # print("merge_tensor_lengths:" + str(lengths))
    return res, lengths


def merge_tensors_lane(tensors: List[torch.Tensor],
                       device,
                       hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
        temp version function to:
        merge a list of tensors into a tensor, will merge to function : merge_tensors
        so please see annotation of function : merge_tensors
    """
    lengths = []
    hidden_size = args.hidden_size if hidden_size is None else hidden_size
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    # print("merge_tensor_lengths:" + str(lengths))
    return res, lengths

def merge_tensors_loss(tensors: List[torch.Tensor], device, pred_length) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[1] if tensor is not None else 0)
    
    res = torch.ones([len(tensors), pred_length, max(lengths)], device=device)*10000.0
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i,:,:tensor.shape[1]] = tensor
    return res, lengths

def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]



def get_subdivide_points(polygon, include_self=False, threshold=1.0, include_beside=False, return_unit_vectors=False):
    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    average_dis = 0
    for i, point in enumerate(polygon):
        if i > 0:
            average_dis += get_dis(point, point_pre)
        point_pre = point
    average_dis /= len(polygon) - 1

    points = []
    if return_unit_vectors:
        assert not include_self and not include_beside
        unit_vectors = []
    divide_num = 1
    while average_dis / divide_num > threshold:
        divide_num += 1
    for i, point in enumerate(polygon):
        if i > 0:
            for k in range(1, divide_num):
                def get_kth_point(point_a, point_b, ratio):
                    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio)

                points.append(get_kth_point(point_pre, point, k / divide_num))
                if return_unit_vectors:
                    unit_vectors.append(get_unit_vector(point_pre, point))
        if include_self or include_beside:
            points.append(point)
        point_pre = point
    if include_beside:
        points_ = []
        for i, point in enumerate(points):
            if i > 0:
                der_x = point[0] - point_pre[0]
                der_y = point[1] - point_pre[1]
                scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
                der_x *= scale
                der_y *= scale
                der_x, der_y = rotate(der_x, der_y, math.pi / 2)
                for k in range(-2, 3):
                    if k != 0:
                        points_.append((point[0] + k * der_x, point[1] + k * der_y))
                        if i == 1:
                            points_.append((point_pre[0] + k * der_x, point_pre[1] + k * der_y))
            point_pre = point
        points.extend(points_)
    if return_unit_vectors:
        return points, unit_vectors
    return points
    # return points if not return_unit_vectors else points, unit_vectors


def get_one_subdivide_polygon(polygon):
    new_polygon = []
    for i, point in enumerate(polygon):
        if i > 0:
            new_polygon.append((polygon[i - 1] + polygon[i]) / 2)
        new_polygon.append(point)
    return new_polygon


def get_subdivide_polygons(polygon, threshold=2.0):
    if len(polygon) == 1:
        polygon = [polygon[0], polygon[0]]
    elif len(polygon) % 2 == 1:
        polygon = list(polygon)
        polygon = polygon[:len(polygon) // 2] + polygon[-(len(polygon) // 2):]
    assert_(len(polygon) >= 2)

    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def get_average_dis(polygon):
        average_dis = 0
        for i, point in enumerate(polygon):
            if i > 0:
                average_dis += get_dis(point, point_pre)
            point_pre = point
        average_dis /= len(polygon) - 1
        return average_dis

    average_dis = get_average_dis(polygon)

    if average_dis > threshold:
        length = len(polygon)
        point_a = polygon[length // 2 - 1]
        point_b = polygon[length // 2]
        point_mid = (point_a + point_b) / 2
        polygon_a = polygon[:length // 2]
        polygon_a = get_one_subdivide_polygon(polygon_a)
        polygon_a = polygon_a + [point_mid]
        polygon_b = polygon[length // 2:]
        polygon_b = get_one_subdivide_polygon(polygon_b)
        polygon_b = [point_mid] + polygon_b
        assert_(len(polygon) == len(polygon_a))
        # print('polygon', np.array(polygon), 'polygon_a',np.array(polygon_a), average_dis, get_average_dis(polygon_a))
        return get_subdivide_polygons(polygon_a) + get_subdivide_polygons(polygon_b)
    else:
        return [polygon]


i_epoch = None
method2FDEs = defaultdict(list)



def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)




def get_dis_list(points: np.ndarray, point_label) -> np.ndarray:
    points = points.astype(float)
    point_label = np.array(point_label).astype(float)
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def to_origin_coordinate(points, idx_in_batch, scale=None):
    """
    :param points: [F, H, 2]"""
    device = points.device
    ori_angle = torch.tensor(origin_angle[idx_in_batch], device=device)
    rotate_mat = torch.tensor([[torch.cos(ori_angle), -torch.sin(ori_angle)],
        [torch.sin(ori_angle), torch.cos(ori_angle)]],
        device=device,dtype=torch.float32)
    diff2 = points - torch.tensor([origin_point[idx_in_batch][0], origin_point[idx_in_batch][1]], device=device, dtype=torch.float32)
    diff2 = diff2.permute(0, 2, 1)
    rotate_mat2 = rotate_mat.repeat(args.mode_num, 1, 1)
    points = torch.matmul(rotate_mat2, diff2)
    points = points.permute(0, 2, 1)
    return points



def to_relative_coordinate(points, x, y, angle):
    for point in points:
        point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


time_begin = get_time()


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


def get_miss_rate(li_FDE, dis=2.0):
    return np.sum(np.array(li_FDE) > dis) / len(li_FDE) if len(li_FDE) > 0 else None


def get_color_text(text, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False


other_errors_dict = defaultdict(list)


def other_errors_put(error_type, error):
    other_errors_dict[error_type].append(error)


def other_errors_to_string():
    res = {}
    for each, value in other_errors_dict.items():
        res[each] = np.mean(value)
    return str(res)





def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    li = []
    for each in mapping:
        if each is not None:
            li.append(each[key])
    return li


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


