
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from nuscenes import NuScenes
from nuscenes.prediction.input_representation.static_layers import correct_yaw, quaternion_yaw
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
import os
import time

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


time_begin = get_time()
nuscenes = NuScenes('v1.0-trainval', dataroot="nuscenes", verbose=True)
helper = PredictHelper(nuscenes)
# Raster maps for visualization.
map_extent = [-50, 50, -20, 80 ]
resolution = 0.1
static_layer_rasterizer = StaticLayerRasterizer(helper,
                                                resolution=resolution,
                                                meters_ahead=map_extent[3],
                                                meters_behind=-map_extent[2],
                                                meters_left=-map_extent[0],
                                                meters_right=map_extent[1])

agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2,
                                                resolution=resolution,
                                                meters_ahead=map_extent[3],
                                                meters_behind=-map_extent[2],
                                                meters_left=-map_extent[0],
                                                meters_right=map_extent[1])

raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())


def global_to_local(origin, global_pose):
    """
    Converts pose in global co-ordinates to local co-ordinates.
    :param origin: (x, y, yaw) of origin in global co-ordinates
    :param global_pose: (x, y, yaw) in global co-ordinates
    :return local_pose: (x, y, yaw) in local co-ordinates
    """
    # Unpack
    global_x, global_y, global_yaw = global_pose
    origin_x, origin_y, origin_yaw = origin

    # Translate
    local_x = global_x - origin_x
    local_y = global_y - origin_y

    # Rotate
    global_yaw = correct_yaw(global_yaw)
    theta = np.arctan2(-np.sin(global_yaw-origin_yaw), np.cos(global_yaw-origin_yaw))

    r = np.asarray([[np.cos(np.pi/2 - origin_yaw), np.sin(np.pi/2 - origin_yaw)],
                    [-np.sin(np.pi/2 - origin_yaw), np.cos(np.pi/2 - origin_yaw)]])
    local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

    local_pose = (local_x, local_y)

    return local_pose

def get_target_agent_global_pose(i_t, s_t):
    """
    Returns global pose of target agent
    :param idx: data index
    :return global_pose: (x, y, yaw) or target agent in global co-ordinates
    """
    sample_annotation = helper.get_sample_annotation(i_t, s_t)
    x, y = sample_annotation['translation'][:2]
    yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
    yaw = correct_yaw(yaw)
    global_pose = (x, y, yaw)

    return global_pose



def generate_nuscenes_gif( mapping,  future_frame_num, predict, mode_num = 6):
    """
    Generates gif of predictions for the given set of indices.
    :param idcs: val set indices corresponding to a particular instance token.
    """

    imgs = []
    i_t = mapping['file_name']
    s_t = mapping['sample_token']

    # Get raster map
    hd_map = raster_maps.make_input_representation(i_t, s_t)
    r, g, b = hd_map[:, :, 0] / 255, hd_map[:, :, 1] / 255, hd_map[:, :, 2] / 255
    hd_map_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    last_obs = [0, 0]
    # Convert to target agent's frame of reference
    origin = get_target_agent_global_pose(i_t, s_t)
    for n, mode in enumerate(predict):
        for m, pose in enumerate(mode):
            local_pose = global_to_local(origin, (pose[0], pose[1], 0))
            mode[m] = np.asarray([local_pose[0], local_pose[1]])

    # Predict
    predictions = predict

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(hd_map, extent=map_extent)
    ax[1].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)
    # ax[2].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)

    for n, traj in enumerate(predictions):
        traj = np.concatenate([np.asarray(last_obs).reshape(1, 2), traj], axis=0)
        ax[1].plot(traj[:, 0] , traj[:, 1] , lw=4,
                    color='r', alpha=0.8)
        ax[1].scatter(traj[-1, 0] , traj[-1, 1] , 60,
                        color='r', alpha=0.8)

    traj_gt = helper.get_future_for_agent(i_t, s_t, seconds=6, in_agent_frame=True)
    traj_gt = np.concatenate([np.asarray(last_obs).reshape(1, 2), traj_gt], axis=0)
    ax[1].plot(traj_gt[:, 0] , traj_gt[:, 1] , lw=4, color='g')
    ax[1].scatter(traj_gt[-1, 0] , traj_gt[-1, 1] , 60, color='g')

    ax[0].axis('off')
    ax[1].axis('off')
    # ax[2].axis('off')
    fig.tight_layout(pad=0)
    ax[0].margins(0)
    ax[1].margins(0)
    # ax[2].margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    os.makedirs('visualize_' + time_begin, exist_ok=True)
    plt.savefig(os.path.join('visualize_' + time_begin, i_t + s_t + ".png"), bbox_inches='tight', dpi=50)
    imgs.append(image_from_plot)
    plt.close(fig)

    return imgs
