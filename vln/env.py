import os
import pickle
import random

import numpy as np
import networkx as nx
import torch

from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis

from base_navigator import BaseNavigator
from utils import load_datasets, load_nav_graph

_SUCCESS_THRESHOLD = 2


def load_features(features_dir, features_name):
    print("=================================")
    print("=====Loading image features======")
    assert type(features_name) == str
    feature_file = os.path.join(features_dir, features_name + '.pickle')
    print('feature_file', feature_file)
    if os.path.isfile(feature_file):
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
    else:
        raise ValueError('could not read image features')
    assert len(features) > 0
    any_pano = list(features.keys())[0]
    any_features = list(features[any_pano].values())[0]
    feature_shape = any_features.shape
    assert 'feature_shape' not in features
    features['feature_shape'] = feature_shape
    return features


class EnvBatch:
    def __init__(self, opts, image_features, batch_size=64, name=None, tokenizer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.name = name
        self.image_features = image_features
        self.tokenizer = tokenizer

        self.navs = []
        print("=====Initializing %s navigators=====" % self.name)
        for i in range(batch_size):  # tqdm(range(batch_size)):
            nav = BaseNavigator(self.opts.dataset_dir)
            self.navs.append(nav)
        print("=====================================")

    def new_episodes(self, pano_ids, headings):
        """ Iteratively initialize the simulators for # of batchsize"""
        for i, (panoId, heading) in enumerate(zip(pano_ids, headings)):
            self.navs[i].graph_state = (panoId, heading)
            self.navs[i].initial_pano_id = panoId
            
    def _get_imgs(self, batch_size):
        imgs = []
        for i in range(batch_size):
            nav = self.navs[i]
            pano, heading = nav.graph_state
            image_feature = self.image_features[pano][heading]
            imgs.append(image_feature)
        imgs = np.array(imgs, dtype=np.float32)
        if self.opts.config.use_image_features == 'resnet_fourth_layer':
            assert imgs.shape[-1] == 100  # (batch_size, 100, 100)
        elif self.opts.config.use_image_features == 'resnet_last_layer':
            assert imgs.shape[-1] == 2048  # (batch_size, 5, 2048)
        elif self.opts.config.use_image_features == 'segmentation':
            assert imgs.shape[-1] == 25  # (batch_size, 5, 25)
        else:
            ValueError('image features not processed')
        return torch.from_numpy(imgs).to(self.device)

    def _get_junction_type(self, batch_size):
        junction_types = []
        for i in range(batch_size):
            nav = self.navs[i]
            pano, _ = nav.graph_state

            num_neighbors = len(nav.graph.nodes[pano].neighbors)
            if num_neighbors == 3:
                junction_type = 1
            elif num_neighbors == 4:
                junction_type = 2
            elif num_neighbors > 4:
                junction_type = 3
            else:
                junction_type = 0

            junction_types.append(junction_type)
        junction_types = np.array(junction_types, dtype=np.int64)  # (batch_size)
        return torch.from_numpy(junction_types).to(self.device)

    def _get_gt_action(self, batch):
        gt_action = []
        for i, item in enumerate(batch):
            gt_action.append(self._get_gt_action_i(batch, i))

        gt_action = np.array(gt_action, dtype=np.int64)
        return torch.from_numpy(gt_action).to(self.device)

    def _get_gt_action_i(self, batch, i):
        nav = self.navs[i]
        gt_path = batch[i]['route_panoids']
        panoid, heading = nav.graph_state
        if panoid not in gt_path:
            return None
        pano_index = gt_path.index(panoid)
        if pano_index < len(gt_path) - 1:
            gt_next_panoid = gt_path[pano_index + 1]
        else:
            gt_action = 3  # STOP
            return gt_action
        pano_neighbors = nav.graph.nodes[panoid].neighbors
        neighbors_id = [neighbor.panoid for neighbor in pano_neighbors.values()]
        gt_next_heading = list(pano_neighbors.keys())[neighbors_id.index(gt_next_panoid)]
        delta_heading = (gt_next_heading - heading) % 360
        if delta_heading == 0:
            gt_action = 0  # FORWARD
        elif delta_heading < 180:
            gt_action = 2  # RIGHT
        else:
            gt_action = 1  # LEFT
        return gt_action

    def _action_select(self, a_prob, ended, num_act_nav, trajs, total_steps, batch):
        """Called during testing."""
        a = []
        heading_changes = []
        action_list = ["forward", "left", "right", "stop"]
        for i in range(len(batch)):
            nav = self.navs[i]
            if ended[i].item():
                a.append([3])
                heading_changes.append([[0]])
                continue

            action_index = a_prob[i].argmax()
            action = action_list[action_index]

            if self.opts.config.oracle_initial_rotation:
                if len(trajs[i]) == 1:
                    gt_action_index = self._get_gt_action_i(batch, i)
                    action = action_list[gt_action_index]
            if self.opts.config.oracle_directions:
                pano, _ = nav.graph_state
                if len(nav.graph.nodes[pano].neighbors) > 2:
                    gt_action_index = self._get_gt_action_i(batch, i)
                    if gt_action_index is not None: # agent is still on the gold path
                        if gt_action_index != 3: # if intersection is also stopping; don't tell the model
                            action = action_list[gt_action_index]
            if self.opts.config.oracle_stopping:
                gt_action_index = self._get_gt_action_i(batch, i)
                if gt_action_index is not None:  # agent is still on the gold path
                    if gt_action_index == 3 or action == 'stop':  # oracle if gold is stop or agent incorrectly stops
                        action = action_list[gt_action_index]

            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1

            nav.step(action)
            a.append([action_list.index(action)])
            heading_change = nav.get_heading_change()
            heading_changes.append([[heading_change]])

            new_pano, new_heading = nav.graph_state
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                trajs[i].append(new_pano)

            total_steps[0] += 1
        a = np.asarray(a, dtype=np.int64)
        heading_changes = np.asarray(heading_changes, dtype=np.float32)
        return torch.from_numpy(a).to(self.device), torch.from_numpy(heading_changes).to(self.device)

    def cal_cls(self, graph, traj, gt_traj):
        PC = np.mean(np.exp([-np.min(
                [nx.dijkstra_path_length(graph, traj_point, gt_traj_point)
                for traj_point in traj])
                for gt_traj_point in gt_traj]))
        EPL = PC * len(gt_traj)
        LS = EPL / (EPL + np.abs(EPL - len(traj)))
        return LS * PC

    def cal_dtw(self, graph, prediction, reference, success):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = nx.dijkstra_path_length(graph, prediction[i-1], reference[j-1])
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        dtw_group = [dtw]
        ndtw = np.exp(-dtw/(_SUCCESS_THRESHOLD * np.sqrt(len(reference))))
        dtw_group += [ndtw, success * ndtw]
        return dtw_group

    def _eva_metrics(self, trajs, batch, graph, metrics):
        for i, item in enumerate(batch):
            success = 0
            traj = trajs[i]
            gt_traj = item["route_panoids"]
            ed = edit_dis(traj, gt_traj)
            ed = 1 - ed / max(len(traj), len(gt_traj))
            target_list = list(nx.all_neighbors(graph, gt_traj[-1])) + [gt_traj[-1]]
            if traj[-1] in target_list:
                success = 1
                metrics[0] += 1
                metrics[2] += ed

            metrics[1] += nx.dijkstra_path_length(graph, traj[-1], gt_traj[-1])
            if self.opts.CLS:
                metrics[3] += self.cal_cls(graph, traj, gt_traj)
            if self.opts.DTW:
                dtw_group = self.cal_dtw(graph, traj, gt_traj, success)
                for j in range(-3, 0):
                    metrics[j] += dtw_group[j]

    def action_step(self, target, ended, num_act_nav, trajs, total_steps):
        action_list = ["forward", "left", "right", "stop"]
        heading_changes = list()
        for i in range(len(ended)):
            nav = self.navs[i]
            if ended[i].item():
                heading_changes.append([[0.0]])
                continue
            action = action_list[target[i]]
            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1
            nav.step(action)
            heading_change = nav.get_heading_change()
            if self.opts.config.heading_change_noise > 0:
                noise = np.random.normal(0.0, abs(heading_change * self.opts.config.heading_change_noise))
                if action == 'left' and (heading_change + noise) >= 0:
                    noise = 0
                if action == 'right' and (heading_change + noise) <= 0:
                    noise = 0
                heading_change += noise
                heading_change = max(-1.0, heading_change)
                heading_change = min(1.0, heading_change)
            heading_changes.append([[heading_change]])

            new_pano, new_heading = nav.graph_state
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                trajs[i].append(new_pano)

            total_steps[0] += 1
        heading_changes = np.asarray(heading_changes, dtype=np.float32)
        return torch.from_numpy(heading_changes).to(self.device)


class OutdoorVlnBatch:
    def __init__(self, opts, image_features, batch_size=64, splits=["train"], tokenizer=None, name=None, sample_bpe=False):
        self.env = EnvBatch(opts, image_features, batch_size, name, tokenizer)
        self.opts = opts

        self.batch_size = batch_size
        self.splits = splits

        self.tokenizer = tokenizer
        self.json_data = load_datasets(splits, opts)
        self.sample_bpe = sample_bpe

        self.data = None
        self.reset_epoch()

        self._load_nav_graph()

    def _get_data(self):
        data = []
        tokenizer = self.tokenizer
        for i, item in enumerate(self.json_data):
            instr = item["navigation_text"]
            if self.sample_bpe:
                _encoder_input = tokenizer.encode(instr, enable_sampling=True, alpha=0.3, nbest_size=-1)
            else:
                _encoder_input = tokenizer.encode(instr)
            _encoder_input.append(tokenizer.eos_id())
            _encoder_input.insert(0, tokenizer.bos_id())
            item["instr_encoding"] = _encoder_input
            data.append(item)
        return data

    def _load_nav_graph(self):
        self.graph = load_nav_graph(self.opts)
        print("Loading navigation graph done.")
        
    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        self.ix += self.batch_size
        self.batch = batch
        
    def get_imgs(self):
        if self.opts.config.use_image_features:
            return self.env._get_imgs(len(self.batch))
        else:
            return None

    def get_junction_type(self):
        return self.env._get_junction_type(len(self.batch))

    def reset(self):
        self._next_minibatch()
        pano_ids = []
        headings = []
        trajs = []
        for item in self.batch:
            pano_ids.append(item["route_panoids"][0])
            headings.append(int(item["start_heading"]))
            trajs.append([pano_ids[-1]])
            
        self.env.new_episodes(pano_ids, headings)
        
        return trajs  # returned a batch of the first panoid for each route_panoids
        
    def get_gt_action(self):
        return self.env._get_gt_action(self.batch)

    def action_select(self, a_t, ended, num_act_nav, trajs, total_steps):
        return self.env._action_select(a_t, ended, num_act_nav, trajs, total_steps, self.batch)
            
    def reset_epoch(self):
        self.ix = 0
        if self.sample_bpe or self.data is None:
            self.data = self._get_data()
        random.shuffle(self.data)
            
    def eva_metrics(self, trajs, metrics):
        self.env._eva_metrics(trajs, self.batch, self.graph, metrics)
