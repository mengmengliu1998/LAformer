"""
    the nuscenes dataloader and visualizer
    author: yangsixun
    
    variables to be modified according to need:
        - DATAROOT: the root of nuscenes dataset
        - STOREDIR: the directory where pickle file will be stored
        - VERSION: the version of nuscenes dataset, e.g. 'v1.0-trainval'
        - SPLIT: which part of dataset should this script process, e.g. 'train' & 'train_val'
        - cores: num of cores
"""
import abc
import math
import multiprocessing
import zlib
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
import numpy as np
import pickle
import argparse
from typing import Union 
import torch.utils.data as torch_data
from multiprocessing import Process
from utils_files.utils import get_dis_list, rotate, get_pad_vector_nus
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap, discretize_lane
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer, get_lanes_in_radius, correct_yaw, quaternion_yaw
from nuscenes.prediction.input_representation.agents import *
from shapely.geometry import Point

map_extent = [-50, 50, -20, 80]



def discard_poses_outside_extent(pose_set, map_extent, ids=None):
    """
    Discards lane or agent poses outside predefined extent in target agent's frame of reference.
    :param pose_set: agent or lane polyline poses
    :param ids: annotation record tokens for pose_set. Only applies to lanes.
    :return: Updated pose set
    """
    updated_pose_set = []
    updated_ids = []

    for m, poses in enumerate(pose_set):
        flag = False
        for n, pose in enumerate(poses):
            if map_extent[0] <= pose[0] <= map_extent[1] and \
                    map_extent[2] <= pose[1] <= map_extent[3]:
                flag = True

        if flag:
            updated_pose_set.append(poses)
            if ids is not None:
                updated_ids.append(ids[m])
        else:
            pass
            # print("discard_poses_outside_extent", poses)

    if ids is not None:
        return updated_pose_set, updated_ids
    else:
        return updated_pose_set





class TrajectoryDataset(torch_data.Dataset):
    """
    Base class for trajectory datasets.
    """

    def __init__(self, mode: str, data_dir: str):
        """
        Initialize trajectory dataset.
        :param mode: Mode of operation of dataset
        :param data_dir: Directory to store extracted pre-processed data
        """
        if mode not in ['extract_data', 'load_data']:
            raise Exception('Dataset mode needs to be one of {extract_data or load_data}')
        self.mode = mode
        self.data_dir = data_dir
        if mode != 'load_data' and not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns size of dataset
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Union[Dict, int]:
        """
        Get data point, based on mode of operation of dataset.
        param idx: data index
        """
        if self.mode == 'extract_data':
            return self.extract_data(idx)
        else:
            return self.load_data(idx)

    def extract_data(self, idx: int):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        param idx: data index
        """
        data = self.get_mapping(idx)
        return data

    @abc.abstractmethod
    def get_mapping(self, idx: int) -> Dict:
        """
        Extracts model inputs.
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        param idx: data index
        return data: Dictionary with pre-processed data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_data(self, idx: int, data: Dict):
        """
        Function to save extracted pre-processed data.
        :param idx: data index
        :param data: Dictionary with pre-processed data
        """
        raise NotImplementedError()


class SingleAgentDataset(TrajectoryDataset):
    """
    Base class for single agent dataset. While we implicitly model all surrounding agents in the scene, predictions
    are made for a single target agent at a time.
    """

    @abc.abstractmethod
    def encode_lanes(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        encode surrounding lanes
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def encode_agents(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        encode all the agents in this dataset
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def encode_future_traj(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target agent representation
        :param idx: data index
        """
        raise NotImplementedError()


class NuScenesData(SingleAgentDataset):
    """
    NuScenes dataset class for single agent prediction
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, nuscenes: NuScenes):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param args: Dataset arguments
        :param helper: NuScenes PredictHelper
        """
        super().__init__(mode, data_dir)
        self.nuscenes = nuscenes
        self.helper = PredictHelper(nuscenes)
        self.invalid_ego_traj_length = False
        self.rf_is_observed = True
        self.subdivide = True
        self.semantic_lane = True
        self.subdivide_len = 5
        self.scale = 1.0
        self.vector_size = 32
        self.args = args
        if self.args['img_only'] and self.args['mapping_only']:
            raise Exception("conflicting args: \"img_only\" and \"mapping_only\"")

        # nuScenes sample and instance tokens for prediction challenge
        # {instance_token}_{sample_token}
        self.token_list = get_prediction_challenge_split(self.args['split'], dataroot=self.helper.data.dataroot)
        print(f"length of data: {len(self.token_list)}")

        # Past and prediction horizons

        self.label_is_valid = np.ones(2 * self.args['t_f'])
        self.eval_frames = 2 * self.args['t_f']
        self.stepwise_label_index = [i for i in range(self.eval_frames)]

        self.data_num = len(self.token_list)
        self.data_per_core = int(self.data_num / self.args['cores']) + 1
        self.buffer_size = 50

    def __len__(self):
        """
        Size of dataset
        """
        return len(self.token_list)

    def initialize(self):
        """
        reset some important variables
        """
        self.agents_past_traj_abs = []
        self.agents_past_traj_rel = []
        self.lanes_midlines_abs = []
        self.valid_lanes_midlines_abs = []
        self.lanes_midlines_rel = []
        self.polygons = []
        self.stepwise_label = np.zeros((self.eval_frames))
        self.vectors = []
        self.polyline_spans = []
        self.matrix = []
        self.mapping = {}

    def get_mapping(self, idx: int) -> Dict:
        """
        Gets model inputs for nuScenes single agent prediction
        :param idx: data index
        :return self.mapping: Dictionary with necessary info
        """
        self.idx = idx
        self.initialize()
        rt = self.extract_ego_info(idx)
        if rt is None:
            return self.mapping
        self.encode_agents(idx)
        self.encode_lanes(idx)
        if self.mapping is None:  # if no valid lanes
            print("no valid lanes")
            return None
        self.encode_future_traj(idx)
        self.update()
        return self.mapping

    def save_data(self, idx: int, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        filename = os.path.join(self.data_dir, str(idx) + '_' + self.token_list[idx] + '.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with batched tensors
        """
        filename = os.path.join(self.data_dir, self.token_list[idx] + '.pickle')

        if not os.path.isfile(filename):
            raise Exception('Could not find data. Please run the dataset in extract_data mode')

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def get_past_motion_states(self, i_t, s_t):
        """
        Returns past motion states: v, a, yaw_rate for a given instance and sample token over self.t_h seconds
        """
        motion_states = np.zeros((2 * self.args['t_h'] + 1, 3))
        motion_states[-1, 0] = self.helper.get_velocity_for_agent(i_t, s_t)
        motion_states[-1, 1] = self.helper.get_acceleration_for_agent(i_t, s_t)
        motion_states[-1, 2] = self.helper.get_heading_change_rate_for_agent(i_t, s_t)
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.args['t_h'], in_agent_frame=True, just_xy=False)

        for k in range(len(hist)):
            motion_states[-(k + 2), 0] = self.helper.get_velocity_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 1] = self.helper.get_acceleration_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 2] = self.helper.get_heading_change_rate_for_agent(i_t, hist[k]['sample_token'])

        motion_states = np.nan_to_num(motion_states)
        return motion_states

    def extract_ego_info(self, idx: int) -> Union[int, None]:
        """
        extract information of ego cars, such as history/ future poses, turning angles
        Args:
            idx: data index
        """
        try:
            token = self.token_list[idx].split('_')
        except IndexError:
            return None

        self.instance_token, self.sample_token = token[0], token[1]  # token
        self.sample = self.nuscenes.get('sample', self.sample_token)
        self.scene_token = self.sample['scene_token']
        self.scene = self.nuscenes.get('scene', self.scene_token)
        self.log_token = self.scene['log_token']
        self.log = self.nuscenes.get('log', self.log_token)
        self.location = self.log['location']
        self.map = NuScenesMap(dataroot=self.args['dataroot'], map_name=self.location)

        ego_car_info = self.helper.get_sample_annotation(sample_token=self.sample_token, instance_token=self.instance_token)
        self.cent_x, self.cent_y = ego_car_info['translation'][0], ego_car_info['translation'][1]  # global
        self.ego_past_traj_abs = self.helper.get_past_for_agent(self.instance_token, self.sample_token,
                                                       seconds=self.args['t_h'], in_agent_frame=False)[::-1]
        self.ego_future_traj_abs = self.helper.get_future_for_agent(self.instance_token, self.sample_token,
                                                       seconds=self.args['t_f'], in_agent_frame=False)
        self.ego_future_traj_abs_api = self.helper.get_future_for_agent(self.instance_token, self.sample_token,
                                                                    seconds=self.args['t_f'], in_agent_frame=True)
        past_len = len(self.ego_past_traj_abs)
        if past_len == 0:
            pass
        elif 0 < past_len < 2 * self.args['t_h']:
            self.ego_past_traj_abs = [self.ego_past_traj_abs[0]] * (2 * self.args['t_h'] - past_len) + list(self.ego_past_traj_abs)
            self.ego_past_traj_abs = np.array(self.ego_past_traj_abs)
        # self.angle = 0
        next_pose = self.ego_future_traj_abs[0]
        self.cal_angle = math.atan2(next_pose[1] - self.cent_y, next_pose[0] - self.cent_x)

        tmp_past = self.helper.get_past_for_agent(
            self.instance_token, self.sample_token,
            seconds=self.args['t_h'], in_agent_frame=False, just_xy=False
        )[::-1]
        tmp_future = self.helper.get_future_for_agent(
            self.instance_token, self.sample_token,
            seconds=self.args['t_f'], in_agent_frame=False, just_xy=False
        )
        yaws = [quaternion_yaw(Quaternion(d['rotation'])) for d in tmp_past]
        yaw = quaternion_yaw(Quaternion(ego_car_info['rotation']))
        yaws.extend([yaw])
        # yaws = [correct_yaw(yaw) for yaw in yaws]
        self.angle = - np.array(yaws).mean() + math.pi / 2
        assert len(self.ego_future_traj_abs) == 2 * self.args['t_f'] and len(self.ego_past_traj_abs) == 2 * self.args['t_h']

        self.ego_past_traj_rel =np.array([rotate(pos[0] - self.cent_x, pos[1] - self.cent_y, self.angle)
                                  for pos in self.ego_past_traj_abs])
        self.ego_future_traj_rel = np.array([rotate(pos[0] - self.cent_x, pos[1] - self.cent_y, self.angle)
                                  for pos in self.ego_future_traj_abs])
        self.sample_tokens_past = [d['sample_token'] for d in tmp_past]
        self.sample_tokens_future = [d['sample_token'] for d in tmp_future]

        past_timestamps = [int(self.nuscenes.get('sample', token)['timestamp'] / 1000.0)
                               for token in self.sample_tokens_past]  # unix timestamp /ms
        if past_len < 2 * self.args['t_h']:
            extend_li = [0] * (2 * self.args['t_h'] - past_len)
            last_timestamp = past_timestamps[0]
            for i in range(1, len(extend_li)+1):
                extend_li[-i] = last_timestamp - i * 500
            past_timestamps = extend_li + past_timestamps
        future_timestamps = [int(self.nuscenes.get('sample', token)['timestamp'] / 1000.0)
                               for token in self.sample_tokens_future]
        self.timestamps = past_timestamps + \
                          [int(self.nuscenes.get('sample', self.sample_token)['timestamp'] / 1000.0)] + \
                          future_timestamps
        self.start_time = self.timestamps[0]

        motion_states = self.get_past_motion_states(self.instance_token, self.sample_token)
        self.ego_past_traj_abs_, self.ego_past_traj_rel_ = np.zeros((2 * self.args['t_h']+1, 5)), np.zeros((2 * self.args['t_h']+1, 5))
        if self.rf_is_observed:
            self.ego_past_traj_abs = list(self.ego_past_traj_abs)
            self.ego_past_traj_abs.append(np.array([self.cent_x, self.cent_y]))
            self.ego_past_traj_abs_[:,:2] = np.array(self.ego_past_traj_abs)
            self.ego_past_traj_abs_[:, 2:] = motion_states
            self.ego_past_traj_rel = list(self.ego_past_traj_rel)
            self.ego_past_traj_rel.append(np.array([0, 0]))
            self.ego_past_traj_rel_[:,:2] = np.array(self.ego_past_traj_rel)
            self.ego_past_traj_rel_[:, 2:] = motion_states
            self.ego_past_traj_abs, self.ego_past_traj_rel =  self.ego_past_traj_abs_, self.ego_past_traj_rel_ 
        return 0

    def encode_lanes(self, idx: int) -> int:
        """
        encode surrounding lanes
        :param idx: data index
        """
        self.get_lane_midlines()
        self.subdivide_lanes()

        # encode subdivided polygons

        for i_polygon, polygon in enumerate(self.subdivided_lane_traj_rel):
            start = len(self.vectors)
            for i_point, point in enumerate(polygon):
                if i_point == 0:
                    continue
                vector = [0] * self.vector_size
                vector[-1] = polygon[i_point-1][0]
                vector[-2] = polygon[i_point-1][1]
                vector[-3] = point[0]
                vector[-4] = point[1]
                vector[-5] = 1
                vector[-6] = i_point
                vector[-7] = len(self.polyline_spans)
                if self.semantic_lane:
                    flags = self.lanes_attrs['has_traffic_control'][self.subdivided_lane_2_origin_lane_labels[i_polygon]]
                    flag = flags[self.rel_ind_2_abs_ind_offset[i_polygon] + i_point]

                    if flag.any():
                        vector[-8] = 1  # has traffic control
                        vector[-11] = 1 if bool(flag[0]) else -1
                        vector[-12] = 1 if bool(flag[1]) else -1
                    else:
                        vector[-8] = -1  # has traffic control
                        vector[-11] = -1
                        vector[-12] = -1

                    direction = self.lanes_attrs['turn_direction'][self.subdivided_lane_2_origin_lane_labels[i_polygon]]
                    vector[-9] = 1 if direction == 'R' else -1 if direction == 'L' else 0

                    vector[-10] = 1 if self.lanes_attrs['is_intersection'][self.subdivided_lane_2_origin_lane_labels[i_polygon]] else -1

                point_pre_pre = (2 * polygon[i_point-1][0] - point[0],
                                 2 * polygon[i_point-1][1] - point[1])
                if i_point >= 2:
                    point_pre_pre = polygon[i_point - 2]
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]
                self.vectors.append(vector)
            end = len(self.vectors)
            if start < end:
                self.polyline_spans.append([start, end])
        if len(self.polyline_spans) == self.map_start_polyline_idx:
            self.mapping = None
        self.matrix = np.array(self.vectors)
        return 0

    def encode_agents(self, idx: int) -> int:
        """
        encode all the agents
        :param idx: data index
        """
        self.get_agents_traj()

        # encode ego traj
        start = len(self.vectors)
        for i_ego, ego_point in enumerate(self.ego_past_traj_rel):
            vector = [
                      ego_point[0], 
                      ego_point[1],
                      ego_point[2], # v
                      ego_point[3], # a
                      ego_point[4], # yaw rate
                        1,
                      False,  # vehicle
                      True,  # ego
                      False,  # pedestrian
                      0,
                      i_ego]
            self.vectors.append(get_pad_vector_nus(vector, self.vector_size))
        end = len(self.vectors)
        self.polyline_spans.append([start, end])

        # encode other traj
        for i_traj, past_traj in enumerate(self.agents_past_traj_rel):
            start = len(self.vectors)
            for i_point, point in enumerate(past_traj):
                vector = [
                          point[0],
                          point[1],
                          point[2], # v
                          point[3], # a
                          point[4], # yaw rate
                            1,
                          True if i_traj < self.vehicle_index else False,   # vehicle
                          False,  # ego
                          True if i_traj >= self.vehicle_index else False,  # pedestrian
                          i_traj + 1,  # ego is 0
                          i_point]
                self.vectors.append(get_pad_vector_nus(vector, self.vector_size))
            end = len(self.vectors)
            if end -start == 0:
                assert False
            else:
                self.polyline_spans.append([start, end])
        assert len(self.agents_past_traj_rel) == len(self.polyline_spans) - 1
        self.map_start_polyline_idx = len(self.polyline_spans)
        return 0

    def encode_future_traj(self, idx: int) -> int:
        """
        encode future traj
        :param idx: data index
        """
        for index in self.stepwise_label_index:
            assert index < 2 * self.args['t_f']
            pose_at_index = self.ego_future_traj_rel[index]
            if len(self.subdivided_lane_traj_rel) == 0:
                label = 0
            else:
                label = np.argmin([min(get_dis_list(lane, pose_at_index)) for lane in self.subdivided_lane_traj_rel])
            self.stepwise_label[index] = label
        agent_goal_poses = self.ego_future_traj_rel[2*self.args['t_f'] - 1]
        if len(self.agents_past_traj_rel) == 0:
            label = 0
        else:
            label = np.argmin([min(get_dis_list(agent, agent_goal_poses)) for agent in self.agents_past_traj_rel])
        self.goal_agent_label = label
        return 0

    def update(self):
        """
        update self.mapping dict
        """
        self.mapping.update(
            file_name = self.instance_token,
            sample_token = self.sample_token,
            start_time=self.start_time,
            city_name=self.location,
            scale=self.scale,
            eval_time=self.eval_frames,

            angle=self.angle,
            cent_x=self.cent_x,
            cent_y=self.cent_y,
            origin_labels=self.ego_future_traj_abs,
            labels=self.ego_future_traj_rel,
            labels_is_valid=self.label_is_valid,
            past_traj=self.ego_past_traj_rel,
            trajs=self.agents_past_traj_rel,
            polygons=self.subdivided_lane_traj_rel,
            polyline_spans=[slice(each[0], each[1]) for each in self.polyline_spans],
            map_start_polyline_idx=self.map_start_polyline_idx,
            matrix=np.array(self.vectors),

            dense_lane_labels=self.stepwise_label,
            goal_agent_label=self.goal_agent_label,
        )

    def get_agents_traj(self):
        """
        get the history trajectories of all agents exsisting in this scene
        """
        human_tokens = []  
        vehicle_tokens = []
        sample_annotation_tokens = self.sample['anns']
        for token in sample_annotation_tokens:
            sample_ann = self.nuscenes.get('sample_annotation', token)
            category = sample_ann['category_name']
            if category.startswith('human'):
                human_tokens.append(sample_ann['instance_token'])
            elif category.startswith('vehicle'):
                vehicle_tokens.append(sample_ann['instance_token'])

        for v_token in vehicle_tokens:
            if v_token == self.instance_token:
                continue
            past_traj_abs = self.helper.get_past_for_agent(
                sample_token=self.sample_token,
                instance_token=v_token,
                seconds=self.args['t_h'],
                in_agent_frame=False)[::-1]
            anno = self.helper.get_sample_annotation(
                            sample_token=self.sample_token,
                            instance_token=v_token
                        )
            # print("anno", anno)
            pose_at_rf = self.helper.get_sample_annotation(
                sample_token=self.sample_token,
                instance_token=v_token
            )['translation'][:2]
            if 0 < len(past_traj_abs) < 2 * self.args['t_h']:
                past_traj_abs = [past_traj_abs[0]] * (2 * self.args['t_h'] - len(past_traj_abs)) + list(past_traj_abs)
            elif len(past_traj_abs) == 0:
                past_traj_abs = np.array([pose_at_rf] * (2 * self.args['t_h']))
            past_traj_abs = list(past_traj_abs)
            past_traj_abs.append(np.array(pose_at_rf))
            past_traj_abs = np.array(past_traj_abs)
            # add motion state
            motion_states = self.get_past_motion_states(v_token, self.sample_token)
            self.agents_past_traj_abs_ = np.zeros((2 * self.args['t_h']+1, 5))
            self.agents_past_traj_abs_[:, :2] = past_traj_abs
            self.agents_past_traj_abs_[:, 2:] = motion_states
            past_traj_abs = self.agents_past_traj_abs_
            self.agents_past_traj_abs.append(past_traj_abs)


        self.vehicle_index = len(self.agents_past_traj_abs)

        for v_token in human_tokens:
            if v_token == self.instance_token:
                continue
            past_traj_abs = self.helper.get_past_for_agent(
                sample_token=self.sample_token,
                instance_token=v_token,
                seconds=self.args['t_h'],
                in_agent_frame=False)[::-1]
            pose_at_rf = self.helper.get_sample_annotation(
                sample_token=self.sample_token,
                instance_token=v_token
            )['translation'][:2]
            if 0 < len(past_traj_abs) < 2 * self.args['t_h']:
                past_traj_abs = [past_traj_abs[0]] * (2 * self.args['t_h'] - len(past_traj_abs)) + list(past_traj_abs)
            elif len(past_traj_abs) == 0:
                past_traj_abs = np.array([pose_at_rf] * (2 * self.args['t_h']))
            past_traj_abs = list(past_traj_abs)
            past_traj_abs.append(np.array(pose_at_rf))
            past_traj_abs = np.array(past_traj_abs)
            motion_states = self.get_past_motion_states(v_token, self.sample_token)
            self.agents_past_traj_abs_ = np.zeros((2 * self.args['t_h']+1, 5))
            self.agents_past_traj_abs_[:, :2] = past_traj_abs
            self.agents_past_traj_abs_[:, 2:] = motion_states
            past_traj_abs = self.agents_past_traj_abs_
            self.agents_past_traj_abs.append(past_traj_abs)
        

        for abs_traj in self.agents_past_traj_abs:
            rel_traj = np.array([rotate(pose[0] - self.cent_x, pose[1] - self.cent_y, self.angle) for pose in abs_traj])
            rel_traj = np.concatenate([rel_traj, abs_traj[:, 2:]], axis=1)
            # rel_traj *= rel_traj * self.scale
            self.agents_past_traj_rel.append(rel_traj)
        self.agents_past_traj_rel = discard_poses_outside_extent(self.agents_past_traj_rel, map_extent)

        return 0

    def get_lane_midlines(self):
        """
        get the midline points of all lanes in surrounding area
        """
        def get_arc_curve(pts)->float:
            """
            get the arc of a curve defined bypts
            Args:
                pts: points of curve
            Returns:
                the arc
            """
            # calculate the length of the chord
            start = np.array(pts[0])
            end = np.array(pts[len(pts) - 1])
            l_arc = np.sqrt(np.sum(np.power(end - start, 2)))

            # cal the max dis between points on chord and the line
            # function: \frac{1}{2a}\sqrt{(a+b+c)(a+b-c)(a+c-b)(b+c-a)}
            a = l_arc
            b = np.sqrt(np.sum(np.power(pts - start, 2), axis=1))
            c = np.sqrt(np.sum(np.power(pts - end, 2), axis=1))
            tmp = (a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)

            if (abs(tmp) < 1e-6).all():
                return 10000

            dist = np.sqrt(tmp) / (2 * a)

            h = dist.max()

            r = ((a * a) / 4 + h * h) / (2 * h)

            return r

        # leftdown, rightup
        scene_x_min = - 50
        scene_x_max = 50
        scene_y_min = - 20
        scene_y_max = 80

        lane_id = get_lanes_in_radius(x=self.cent_x, y=self.cent_y, radius=100,
                                      discretization_meters=1.0, map_api=self.map)
        if len(lane_id) == 0:
            self.mapping = None
        self.lanes_attrs = {
            "has_traffic_control": [],
            "turn_direction": [],
            "is_intersection": []
        }

        lane_traj_tokens = []
        valid_lane_traj_tokens = []
        for token, li in lane_id.items():
            li = [np.array([coor[0], coor[1]]) for coor in li]
            if len(li) > 1:
                self.lanes_midlines_abs.append(li)
                lane_traj_tokens.append(token)
        # polygons = self.get_polygons_around_agent()
        # flags = self.get_lane_flags(self.lanes_midlines_abs, polygons)
        # assert len(flags) == len(self.lanes_midlines_abs) == len(valid_lane_traj_tokens)
        self.valid_lanes_midlines_abs = []

        for line_idx, li in enumerate(self.lanes_midlines_abs):
            rel_li = np.array([rotate(point[0] - self.cent_x, point[1] - self.cent_y, self.angle) for point in li])
            tmp_rel_li = []
            tmp_abs_li = []
            for i_point, coor in enumerate(rel_li):
                if scene_x_min <= coor[0] <= scene_x_max and scene_y_min <= coor[1] <= scene_y_max:
                    tmp_rel_li.append(coor)
                    tmp_abs_li.append(li[i_point])
            assert len(tmp_abs_li) == len(tmp_rel_li)
            if len(tmp_rel_li) > 1:
                self.lanes_midlines_rel.append(np.array(tmp_rel_li))
                self.valid_lanes_midlines_abs.append(np.array(tmp_abs_li))
                valid_lane_traj_tokens.append(lane_traj_tokens[line_idx])

        polygons = self.get_polygons_around_agent()
        flags = self.get_lane_flags(self.valid_lanes_midlines_abs, polygons)
        assert len(flags) == len(self.valid_lanes_midlines_abs) == len(valid_lane_traj_tokens) == len(self.lanes_midlines_rel)

        for i_flag, flag in enumerate(flags):
            assert len(flag) == len(self.lanes_midlines_rel[i_flag])

        for rel_li_idx, rel_li in enumerate(self.lanes_midlines_rel):
            if len(rel_li) > 1:
                # assert len(rel_li) == len(self.valid_lanes_midlines_abs[rel_li_idx])

                # get 'is_intersection' attribute
                idx = int(len(rel_li) / 2)
                # idx = 0
                attr = self.map.layers_on_point(
                    self.valid_lanes_midlines_abs[rel_li_idx][idx][0],
                    self.valid_lanes_midlines_abs[rel_li_idx][idx][1], layer_names=['road_segment', ])
                road_seg_attr = attr['road_segment']
                # stop_line_attr = attr['stop_line']
                if len(road_seg_attr) > 0:
                    self.lanes_attrs['is_intersection'].append(
                        self.map.get(
                            'road_segment',
                            road_seg_attr)['is_intersection']
                    )
                else:
                    self.lanes_attrs['is_intersection'].append(False)

                # get 'turn_direction' attribute
                token = valid_lane_traj_tokens[rel_li_idx]
                arcline = self.map.arcline_path_3.get(token)
                traj = discretize_lane(arcline, 1.0)
                traj = np.array([np.array((point[0], point[1]), dtype=float) for point in traj])
                curvature = get_arc_curve(traj)
                if curvature < 100:
                    li = self.valid_lanes_midlines_abs[rel_li_idx]
                    lane_angle = - math.atan2(li[1][1] - li[0][1], li[1][0] - li[0][0]) + math.pi / 2
                    origin_point = li[0]
                    end_point = li[-1]
                    end_point = np.array(rotate(end_point[0] - origin_point[0], end_point[1] - origin_point[1], lane_angle))
                    if end_point[0] > 0:
                        direction = 'R'
                    else:
                        direction = 'L'
                    self.lanes_attrs['turn_direction'].append(direction)
                else:
                    self.lanes_attrs['turn_direction'].append("S")
                # self.lanes_midlines_rel.append(rel_li)

                # get 'has_traffic_control' attribute
                self.lanes_attrs['has_traffic_control'].append(flags[rel_li_idx])

        self.map_extent = [-50, 50, -50, 150]

        return 0

    def subdivide_lanes(self):
        """
        subdivide all the lanes acquired by function "get_lane_midlines"
        """
        # self.subdivided_lane_traj_abs = []
        self.subdivided_lane_traj_rel = []
        self.subdivided_lane_2_origin_lane_labels = []
        self.rel_ind_2_abs_ind_offset = []

        for lane_id, lane_traj in enumerate(self.lanes_midlines_rel):
            if len(lane_traj) <= 1:
                continue
            # print(lane_id)
            self.divide_lane(lane_traj, lane_id)

        self.subdivided_lane_traj_rel = np.array(self.subdivided_lane_traj_rel, dtype=object)

        assert len(self.subdivided_lane_2_origin_lane_labels) == len(self.subdivided_lane_traj_rel)

    def divide_lane(self, traj, l_id):
        """
        divide a lane
        Args:
            traj: the trajectory of a lane
            l_id: the lane index of the origin lane
        """
        left_index = 0
        length = len(traj)
        bound = self.subdivide_len 
        # print(length)
        while True:
            if length - left_index >= bound:
                self.subdivided_lane_traj_rel.append(traj[left_index:left_index+bound])
                self.subdivided_lane_2_origin_lane_labels.append(l_id)

                if len(self.subdivided_lane_2_origin_lane_labels) == 1 or \
                        self.subdivided_lane_2_origin_lane_labels[-1] != self.subdivided_lane_2_origin_lane_labels[-2]:
                    self.rel_ind_2_abs_ind_offset.append(0)
                else:
                    self.rel_ind_2_abs_ind_offset.append(self.rel_ind_2_abs_ind_offset[-1] + len(self.subdivided_lane_traj_rel[-2]))

                left_index += bound
            elif 1 < length - left_index < bound:
                self.subdivided_lane_traj_rel.append(traj[left_index:])
                self.subdivided_lane_2_origin_lane_labels.append(l_id)

                if len(self.subdivided_lane_2_origin_lane_labels) == 1 or \
                        self.subdivided_lane_2_origin_lane_labels[-1] != self.subdivided_lane_2_origin_lane_labels[-2]:
                    self.rel_ind_2_abs_ind_offset.append(0)
                else:
                    self.rel_ind_2_abs_ind_offset.append(self.rel_ind_2_abs_ind_offset[-1] + len(self.subdivided_lane_traj_rel[-2]))
                break
            else:
                break
        return 0

    def get_polygons_around_agent(self) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        Returns:
            polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, = self.cent_x, self.cent_y
        radius = 80
        record_tokens = self.map.get_records_in_radius(x, y, radius, ['stop_line', 'ped_crossing'])
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = self.map.get(k, record_token)['polygon_token']
                polygons[k].append(self.map.extract_polygon(polygon_token))
        return polygons

    @ staticmethod
    def get_lane_flags(lanes, polygons) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        Args:
            lanes: list of lane poses
            polygons: dictionary of polygon layers
        Returns:
            lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags

    def extract_multiprocess(self):
        """
        the parallel process of extracting data
        """
        ex_list = []

        def run(queue: multiprocessing.Queue, queue_res: multiprocessing.Queue):
            process_id = queue.get()
            if process_id == self.args['cores'] - 1:
                li = list(range(process_id * self.data_per_core, self.data_num))
            elif process_id is not None:
                li = list(range(process_id * self.data_per_core, (process_id + 1) * self.data_per_core))

            for idx in tqdm(li):
                a = self.__getitem__(idx)
                if not self.args['img_only']:
                    if a is None:
                        pass
                    else:
                        data_compress = zlib.compress(pickle.dumps(a))
                        queue_res.put(data_compress)



        queue = multiprocessing.Queue(self.args['cores'])
        queue_res = multiprocessing.Queue()

        processes = [Process(target=run, args=(queue, queue_res)) for _ in range(self.args['cores'])]

        for each in processes:
            each.start()

        for i in range(self.args['cores']):
            queue.put(i)
            # pbar.update(1)

        while not queue.empty():
            pass

        save_pbar = tqdm(range(self.data_num))
        if not self.args['img_only']:
            for _ in save_pbar:
                try:
                    a = queue_res.get(block=True, timeout=20)
                    ex_list.append(a)
                    save_pbar.update(1)
                except Exception as e:
                    break

        for each in processes:
            each.join()

        print("all thread end")

        print("length of ex list: ", len(ex_list))
        if not self.args['img_only']:
            os.makedirs(self.data_dir, exist_ok=True)
            if 'train' in self.args['split']:
                with open(os.path.join(self.data_dir, 'ex_list'), 'wb') as f:
                    pickle.dump(ex_list, f)
            elif 'val' in self.args['split']:
                with open(os.path.join(self.data_dir, 'eval.ex_list'), 'wb') as f:
                    pickle.dump(ex_list, f)
        print("dump finished!")

def get_parser():
    parser = argparse.ArgumentParser(description='LAformer')
    parser.add_argument('--DATAROOT', type=str, default='nuscenes', help='DATAROOT')
    parser.add_argument('--STOREDIR', type=str, default='temp_file_nuscenes', help='dir to store processed data')
    parser.add_argument('--VERSION', type=str, default='v1.0-trainval', help='dataset version, v1.0-mini, v1.0-trainval')
    parser.add_argument('--SPLIT', type=str, default='train', help='dataset split, mini_train, mini_val, train, val')
    parser.add_argument('--CORES', type=int, default=16, help='number of cores to use')
    return parser
if __name__ == '__main__':
    from tqdm import tqdm
    parser = get_parser()
    p = parser.parse_args()
    args = {
        't_h': 2,
        't_f': 6,
        'split': p.SPLIT,
        'dataroot':p.DATAROOT,
        'cores': p.CORES,
        'vis_mode': 'rel',  # only work when using 'visualize' for vis
        'show_fig': False,  # True for displaying only False for saving only
        'fig_dir': './debug',  # only work when 'show_fig' is False
        'vis_func': 'visualize',  # ways of vis
        'img_only': False,
        'debug': False,  # visualize encoded info, make sure matrix is totally correct
        'mapping_only': True  # these 2 variable can't both be True,
                                # when these 2 are both False, loader will generate mapping and pics
    }
    nuscenes = NuScenes(p.VERSION, dataroot=p.DATAROOT)
    dataset = NuScenesData(mode='extract_data', data_dir=p.STOREDIR, args=args, nuscenes=nuscenes)
    dataset.extract_multiprocess()
    # for i in range(405, 425):
    #     _ = dataset[i]
