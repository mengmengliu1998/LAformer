import math
import multiprocessing
import os, sys
import pickle
import zlib
import random
from multiprocessing import Process
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from utils_files import utils, config
from utils_files.utils import get_name, get_angle, logging, rotate, round_value, get_pad_vector, get_dis, larger, assert_, get_subdivide_points

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3
map_extent = [-50, 50, -20, 60]

def discard_poses_outside_extent(pose_set, map_extent, ids= None):
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

def discard_poses_outside_extent_agent(id2info, map_extent):
    """
    Discards agent poses outside predefined extent in target agent's frame of reference.
    :param id2info: agent dictionary
    :param map_extent: extent of map
    :return: Updated pose set
    """
    id2delete = []
    for id in id2info:
        if id == "AGENT" or id == "AV":
            continue
        flag = False
        info = id2info[id]
        for line in info:
            if map_extent[0] <= line[X] <= map_extent[1] and \
                    map_extent[2] <= line[Y] <= map_extent[3]:
                flag = True
        if flag:
            pass
        else:
            id2delete.append(id)
    print("discard_poses_outside_extent_agent", len(id2delete))
    print("id3info", len(id2info))
    for id in id2delete:
        del id2info[id]
    return id2info


def get_sub_map(args: config.Args, x, y, city_name, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.

    Only take lanes which are no more than args.max_distance away from (x, y).

    """
    assert isinstance(am, ArgoverseMap)
    if 'semantic_lane' in args.other_params:
        lane_ids = am.get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
        local_lane_centerlines = [am.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
        polygons = local_lane_centerlines
        if args.do_eval and not args.do_test:   # visualization
            angle = mapping['angle']
            vis_lanes = [am.get_lane_segment_polygon(lane_id, city_name)[:, :2] for lane_id in lane_ids]
            t = []
            for each in vis_lanes:
                for point in each:
                    point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                num = len(each) // 2
                t.append(each[:num].copy())
                t.append(each[num:num * 2].copy())
            vis_lanes = t
            mapping['vis_lanes'] = vis_lanes
    else:
        polygons = am.find_local_lane_centerlines(x, y, city_name,
                                                    query_search_range_manhattan=args.max_distance)

    polygons = [polygon[:, :2].copy() for polygon in polygons]
    angle = mapping['angle']
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
            if 'scale' in mapping:
                assert 'enhance_rep_4' in args.other_params
                scale = mapping['scale']
                point[0] *= scale
                point[1] *= scale

    # polygons, lane_ids = discard_poses_outside_extent(polygons, map_extent, lane_ids)
    lane_idx_2_polygon_idx = {}
    for polygon_idx, lane_idx in enumerate(lane_ids):
        lane_idx_2_polygon_idx[lane_idx] = polygon_idx


    polygon_new_list = []

    for index_polygon, polygon in enumerate(polygons):
        assert_(2 <= len(polygon) <= 10, info=len(polygon))
        if 'semantic_lane' in args.other_params:
            assert len(lane_ids) == len(polygons)
            lane_id = lane_ids[index_polygon]
            lane_segment = am.city_lane_centerlines_dict[city_name][lane_id]
        assert_(len(polygon) >= 2)
        subdivide_points = get_subdivide_points(polygon)
        polygon_list = []
        if len(subdivide_points) <= args.subdivide_length:
            polygon_list.append(polygon)
        else:
            polygon_list.append(subdivide_points[0:args.subdivide_length])
        
        if len(polygons[0]) < 2:
            assert False

        for j, pt in enumerate(subdivide_points):
            if ((j % (args.subdivide_length) == 0)|(j==len(subdivide_points)-1))and(j > args.subdivide_length):
                polygon_new = []
                if j-args.subdivide_length < 0:
                    print("j-10<0")
                    assert False
                polygon_new = subdivide_points[j-args.subdivide_length:j]
                if len(polygon_new) >= 2:
                    polygon_list.append(np.array(polygon_new))
        for polygon in polygon_list:
            start = len(vectors)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * args.vector_size
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                    vector[-5] = 1
                    vector[-6] = i

                    vector[-7] = len(polyline_spans)

                    if 'semantic_lane' in args.other_params:
                        vector[-8] = 1 if lane_segment.has_traffic_control else -1
                        vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                            -1 if lane_segment.turn_direction == 'LEFT' else 0
                        vector[-10] = 1 if lane_segment.is_intersection else -1
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    if i >= 2:
                        point_pre_pre = polygon[i - 2]
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
                point_pre = point

            end = len(vectors)
            if start < end:
                polyline_spans.append([start, end])
                polygon_new_list.append(np.array(polygon))
    mapping['polygons'] = polygon_new_list
    return (vectors, polyline_spans)



def preprocess(args, id2info, mapping):
    """
    This function calculates matrix based on information from get_instance.
    """
    polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    vectors = []
    two_seconds = mapping['two_seconds']
    mapping['trajs'] = []
    mapping['agents'] = []
    for id in keys:

        info = id2info[id]
        start = len(vectors)

        if args.do_eval and not args.do_test:  # noly for visualization
            traj = np.zeros([args.hidden_size])
            for i, line in enumerate(info):
                if larger(line[TIMESTAMP], two_seconds):
                    traj = traj[:i * 2].copy()
                    break
                traj[i * 2], traj[i * 2 + 1] = line[X], line[Y]
                if i == len(info) - 1:
                    traj = traj[:(i + 1) * 2].copy()
            traj = traj.reshape((-1, 2))
            mapping['trajs'].append(traj)

        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            x, y = line[X], line[Y]
            if i > 0:
                # print(x-line_pre[X], y-line_pre[Y])
                vector = [line_pre[X], line_pre[Y], x, y, line[TIMESTAMP], line[OBJECT_TYPE] == 'AV',
                          line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS', len(polyline_spans), i]
                vectors.append(get_pad_vector(vector))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != 'AGENT'
        else:

            polyline_spans.append([start, end])


    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping['map_start_polyline_idx'] = len(polyline_spans)
    if args.use_map:
        vectors, polyline_spans = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'],
                                              vectors=vectors,
                                              polyline_spans=polyline_spans, mapping=mapping)


    matrix = np.array(vectors)

    labels = []
    past_traj = []
    info = id2info['AGENT']
    observed_traj = info[:mapping['agent_pred_index']]
    future_traj = info[mapping['agent_pred_index']:]
    if not args.do_test:
        assert len(future_traj) == 30
    for line in future_traj:
        labels.append(line[X])
        labels.append(line[Y])
    for line in observed_traj:
        past_traj.append(line[X])
        past_traj.append(line[Y])


    if 'test' in args.data_dir[0]:
        labels = [0.0 for _ in range(60)]
    polygons = mapping['polygons']
    dense_lane_labels = [0] * args.future_frame_num
    min_di = [10000.0] * args.future_frame_num
    for i, polygon in enumerate(polygons):
        polygon = np.array(polygon)
        for j in range(args.future_frame_num):
            temp_dist = np.min(get_dis(polygon, np.array(labels[j * 2:j * 2 + 2])))
            if temp_dist < min_di[j]:
                min_di[j] = temp_dist
                dense_lane_labels[j] = i
    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([30, 2]),
        past_traj=np.array(past_traj).reshape([mapping['agent_pred_index'], 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
        dense_lane_labels=dense_lane_labels,
    ))
    del mapping['polygons']
    return mapping


def argoverse_get_instance(actor_id, lines, file_name, args):
    """
    Extract polylines from one example file content.
    """

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping['file_name'] = file_name
    agent_tobe_av = False
    is_agent = False
    for i, line in enumerate(lines):
        line = line.strip().split(',')
        if i == 0:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]
        if args.do_train and args.train_all:
            if line[TRACK_ID] == actor_id and line[OBJECT_TYPE] == 'AGENT':
                is_agent = True
            if line[OBJECT_TYPE] == 'AV' and line[TRACK_ID] != actor_id:
                line[TRACK_ID] = line[OBJECT_TYPE]
            elif line[OBJECT_TYPE] == 'AV' and line[TRACK_ID] == actor_id:
                line[TRACK_ID] = 'AGENT'
                line[OBJECT_TYPE] = 'AGENT'
                agent_tobe_av = True
            elif line[TRACK_ID] == actor_id:
                line[TRACK_ID] = 'AGENT'
                line[OBJECT_TYPE] = 'AGENT'
            elif line[OBJECT_TYPE] == 'AGENT' and agent_tobe_av:
                line[OBJECT_TYPE] = 'AV'
                line[TRACK_ID] = 'AV'
            elif line[OBJECT_TYPE] == 'AGENT':
                line[OBJECT_TYPE] = 'OTHERS'

        else:
            is_agent = True
            if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
                line[TRACK_ID] = line[OBJECT_TYPE]
        

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]
            if 'direction' in args.other_params:
                span = agent_lines[-6:]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][X] - span[j][X], span[j + interval][Y] - span[j][Y]
                            angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]
    if not args.do_test:
        if is_agent:
            assert len(id2info['AGENT']) == 50
        if len(id2info['AGENT']) != 50:
            return None


    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    if args.do_eval:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info['AGENT'][20:]):
            origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
        mapping['origin_labels'] = origin_labels
    if args.do_train and args.train_all:
        min_distance_in_history = math.sqrt((id2info['AGENT'][19][X] - id2info['AGENT'][-1][X])**2 + \
            (id2info['AGENT'][19][Y] - id2info['AGENT'][-1][Y])**2) < 3
        if not is_agent and min_distance_in_history:
            return None
    angle = -get_angle(der_x, der_y) + math.radians(90)
    if 'direction' in args.other_params:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale
    # id2info = discard_poses_outside_extent_agent(id2info, map_extent)
    return preprocess(args, id2info, mapping)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        data_dir = args.data_dir
        self.ex_list = []
        self.args = args

        if args.reuse_temp_file:
            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'rb')
            self.ex_list = pickle.load(pickle_file)
            pickle_file.close()
        else:
            global am
            from argoverse.map_representation.map_api import ArgoverseMap
            am = ArgoverseMap()
            if args.core_num >= 1:
                # TODO
                files = []
                for each_dir in data_dir:
                    dirs = os.listdir(each_dir)
                    files.extend([os.path.join(each_dir, file) for file in dirs if
                                  file.endswith("csv") and not file.startswith('.')])
                print(files[:5], files[-5:])

                pbar = tqdm(total=len(files))

                queue = multiprocessing.Queue(args.core_num)
                queue_res = multiprocessing.Queue()
                def calc_ex_list(queue, queue_res, args):

                    dis_list = []
                    while True:
                        file = queue.get()
                        if file is None:
                            break
                        if file.endswith("csv"):
                            
                            with open(file, "r", encoding='utf-8') as fin:
                                lines = fin.readlines()[1:]
                            if args.do_train and args.train_all:
                                df = pd.read_csv(file)
                                actor_ids = list(df['TRACK_ID'].unique())
                                num_nodes = len(actor_ids)
                                for actor_id in actor_ids:
                                    instance = argoverse_get_instance(actor_id, lines, file, args)
                                    if instance is not None:
                                        data_compress = zlib.compress(pickle.dumps(instance))
                                        queue_res.put(data_compress)
                                    else:
                                        pass
                                        # queue_res.put(None)
                            else:
                                instance = argoverse_get_instance(0, lines, file, args)
                                if instance is not None:
                                    data_compress = zlib.compress(pickle.dumps(instance))
                                    queue_res.put(data_compress)
                                else:
                                    pass
                                    # queue_res.put(None)

                processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) for _ in range(args.core_num)]
                for each in processes:
                    each.start()
                for file in files:
                    assert file is not None
                    queue.put(file)
                    pbar.update(1)
                # necessary because queue is out-of-order
                while not queue.empty():
                    pass
                pbar.close()
                self.ex_list = []
                pbar = tqdm(total=len(files)*20)
                print("qsize", queue_res.qsize())
                for i in range(len(files)*20):
                    try:
                        t = queue_res.get(timeout=10)
                        if t is not None:
                            self.ex_list.append(t)
                        pbar.update(1)
                    except:
                        break
                pbar.close()
                print("len(self.ex_list)", len(self.ex_list))

                for i in range(args.core_num):
                    queue.put(None)
                for each in processes:
                    each.join(5)

            else:
                assert False

            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'wb')
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()
        assert len(self.ex_list) > 0
        if "hard_mining_train" in args.other_params:
            self.corner_data_index = []
            for i in range(len(self.ex_list)):
                data_compress = self.ex_list[i]
                instance = pickle.loads(zlib.decompress(data_compress))
                if self.corner_data(instance):
                    self.corner_data_index.append(i)
            # self.ex_list = [self.ex_list[i] for i in self.corner_data_index]
            ex_list_corner = [self.ex_list[i] for i in self.corner_data_index]
            ex_list_corner = ex_list_corner + ex_list_corner
            self.ex_list = self.ex_list + ex_list_corner
            print("total corner data", len(self.ex_list))
            utils.logging(f'total corner data: {len(self.ex_list)}')
        if to_screen:
            print("valid data size is", len(self.ex_list))
            logging('max_vector_num', max_vector_num)
        self.batch_size = batch_size
    
    def corner_data(self, data):
        """On average, a lane is 3.8 meters in width. 
        Thus, we consider a longitudinal offset of more than 3m to be a turning scene."""

        if abs(data['labels'][0, 0]-data['labels'][-1, 0]) > 3:
            return True
        else:
            return False

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        if self.args.do_train and instance is not None:
            if torch.randint(2, (1, 1)).squeeze().bool().item():
                instance = self.flip_horizontal(instance)
        return instance

    def flip_horizontal(self, data):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """

        # Flip agent and lanes
        hist = data['matrix']
        hist[:, 0] = -hist[:, 0]  # x-coord
        if "nuscenes" not in self.args.other_params:
            hist[:, 2] = -hist[:, 2]  # prev_x
        hist[:, -1] = -hist[:, -1]  # prev_y
        hist[:, -3] = -hist[:, -3]  # next_x
        hist[:, -17] = -hist[:, -17]  # prepre_x
        data['matrix'] = hist

        # Flip groud truth trajectory
        fut = data['labels']
        fut[:, 0] = -fut[:, 0]  # x-coord
        data['labels'] = fut

        #Flip past trajectory
        past = data['past_traj']
        past[:, 0] = -past[:, 0]  # x-coord
        data['past_traj'] = past

        return data

def post_eval(args, file2pred, file2labels, file2pred_score, DEs):
    score_file = args.model_recover_path.split('/')[-1]
    if "nuscenes" in args.other_params:
        from utils_files import eval_metrics
        metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, args.mode_num, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)    
        print("brier-minFDE",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE",('%.4f' % metric_results["minADE"]),",minFDE",('%.4f' % metric_results["minFDE"]),",MR",('%.4f' % metric_results["MR"]))
        if args.mode_num == 10:
            metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, 5, args.future_frame_num, 2.0, file2pred_score)
            utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)    
            print("brier-minFDE1",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE1",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE5",('%.4f' % metric_results["minADE"]),",minFDE5",('%.4f' % metric_results["minFDE"]),",MR1",('%.4f' % metric_results["MR"]))
        metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, 1, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)    
        print("brier-minFDE1",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE1",('%.4f' % metric_results["brier-minADE"]),\
        ",minADE1",('%.4f' % metric_results["minADE"]),",minFDE1",('%.4f' % metric_results["minFDE"]),",MR1",('%.4f' % metric_results["MR"]))
    else:
        from argoverse.evaluation import eval_forecasting
        metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, args.mode_num, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)    
        print("brier-minFDE",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE",('%.4f' % metric_results["minADE"]),",minFDE",('%.4f' % metric_results["minFDE"]),",MR",('%.4f' % metric_results["MR"]))
        metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 1, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)    
        print("brier-minFDE1",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE1",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE1",('%.4f' % metric_results["minADE"]),",minFDE1",('%.4f' % metric_results["minFDE"]),",MR1",('%.4f' % metric_results["MR"]))

