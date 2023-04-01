import argparse
from typing import Dict


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--z_size",
                        default=2,
                        type=int,
                        help="Gaussian dimension.")
    parser.add_argument("--topk",
                        default=2,
                        type=int,
                        help="Gaussian dimension.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_all",
                        action='store_true',
                        help="Whether to train all valid trajectories.")
    parser.add_argument("-e", "--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true')
    parser.add_argument("--data_dir",
                        default='train/data/',
                        type=str)
    parser.add_argument("--data_dir_for_val",
                        default='val/data/',
                        type=str)
    parser.add_argument("--output_dir", default="tmp/", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--temp_file_dir", default=None, type=str)
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--subdivide_length",
                        default=5,
                        type=int,
                        help="lane subdivide length")
    parser.add_argument("--lane_loss_weight",
                        default=10,
                        type=float,
                        help="lane loss weight")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=16.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=777,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int)
    parser.add_argument("--vector_size",
                        default=32,
                        type=int)
    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--sub_graph_depth",
                        default=3,
                        type=int)
    parser.add_argument("--global_graph_depth",
                        default=1,
                        type=int)
    parser.add_argument("--debug",
                        action='store_true')
    parser.add_argument("--initializer_range",
                        default=0.02,
                        type=float)
    parser.add_argument("-d", "--distributed_training",
                        nargs='?',
                        default=8,
                        const=4,
                        type=int)
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)
    parser.add_argument("--use_map",
                        action='store_true')
    parser.add_argument("--reuse_temp_file",
                        action='store_true')
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("--core_num",
                        default=1,
                        type=int,
                        help="core_num for pre-processing")
    parser.add_argument("--visualize",
                        action='store_true')
    parser.add_argument("--train_extra",
                        action='store_true')
    parser.add_argument("--use_centerline",
                        action='store_true')
    parser.add_argument("--add_prefix",
                        default=None)
    parser.add_argument("--method_span",
                        nargs='*',
                        default=[0, 1],
                        type=int)
    parser.add_argument("--master_port", default='12355')
    parser.add_argument("--future_frame_num",
                        default=30,
                        type=int)
    parser.add_argument("--historical_steps",
                        default=20,
                        type=int)
    parser.add_argument("--future_test_frame_num",
                        default=16,
                        type=int)
    parser.add_argument("--mode_num",
                        default=6,
                        type=int)


class Args:
    data_dir = None
    data_kind = None
    debug = None
    train_batch_size = None
    seed = None
    eval_batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = None
    sub_graph_depth = None
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    model_recover_path = None
    do_train = None
    max_distance = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    core_num = None
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    add_prefix = None
    do_test = None
    method_span = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None



