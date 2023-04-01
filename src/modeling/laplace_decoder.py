# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import numpy as np
from utils_files import utils, config
from utils_files.utils import init_weights
from modeling.vectornet import *
from modeling.motion_refinement import trajectory_refinement
from utils_files.loss import *

class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states

class GRUDecoder(nn.Module):
    def __init__(self, args, vectornet) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.future_steps = args.future_frame_num
        self.num_modes = args.mode_num
        self.min_scale = min_scale
        self.args = args
        self.dense = args.future_frame_num
        self.z_size = args.z_size
        self.smothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1))
        self.aggregate_global_z = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.reg_loss = LaplaceNLLLoss(reduction='none')
        # self.reg_loss = GaussianNLLLoss(reduction='none')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='none')
        if "step_lane_score" in args.other_params:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size*2, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))  
            decoder_layer_dense_label = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.dense_label_cross_attention = nn.TransformerDecoder(decoder_layer_dense_label, num_layers=1)
            self.dense_lane_decoder = DecoderResCat(self.hidden_size, self.hidden_size * 3, out_features=self.dense)
            self.proj_topk = MLP(self.hidden_size+1, self.hidden_size)
            decoder_layer_aggregation = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.aggregation_cross_att= nn.TransformerDecoder(decoder_layer_aggregation, num_layers=1)
        else:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))
        self.apply(init_weights)   
        if "stage_two" in args.other_params:
            if args.do_train:
                model_recover = torch.load(args.other_params['stage-two-train_recover'])
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)
                # # self must be vectornet
                # for p in vectornet.parameters():
                #     p.requires_grad = False
            self.trajectory_refinement = trajectory_refinement(args)

    def dense_lane_aware(self, i, mapping, lane_states_batch, lane_states_length, element_hidden_states, \
                            element_hidden_states_lengths, global_hidden_states, device, loss):
        """dense lane aware
        Args:
            mapping (list): data mapping
            lane_states_batch (tensor): [max_len, N]
            lane_states_length (tensor): [N]
            element_hidden_states (tensor): [N]
            global_hidden_states (tensor): [N]
            device (device): device"""
        def dense_lane_scores():
            lane_states_batch_attention = lane_states_batch + self.dense_label_cross_attention(
                lane_states_batch, element_hidden_states.unsqueeze(0), tgt_key_padding_mask=src_attention_mask_lane)
            dense_lane_scores = self.dense_lane_decoder(torch.cat([global_hidden_states.unsqueeze(0).expand(
                lane_states_batch.shape), lane_states_batch, lane_states_batch_attention], dim=-1)) # [max_len, N, H]
            dense_lane_scores = F.log_softmax(dense_lane_scores, dim=0)
            return dense_lane_scores
        max_vector_num = lane_states_batch.shape[1]
        batch_size = len(mapping)
        src_attention_mask_lane = torch.zeros([batch_size, lane_states_batch.shape[1]], device=device) # [N, max_len]
        for i in range(batch_size):
            assert lane_states_length[i] > 0
            src_attention_mask_lane[i, :lane_states_length[i]] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        lane_states_batch = lane_states_batch.permute(1, 0, 2) # [max_len, N, H]
        dense_lane_pred = dense_lane_scores() # [max_len, N, H]
        dense_lane_pred = dense_lane_pred.permute(1, 0, 2) # [N, max_len, H]
        lane_states_batch = lane_states_batch.permute(1, 0, 2) # [N, max_len, H]
        dense  = self.dense
        dense_lane_pred =  dense_lane_pred.permute(0, 2, 1) # [N, H, max_len]
        dense_lane_pred = dense_lane_pred.contiguous().view(-1, max_vector_num)  # [N*H, max_len]
        if self.args.do_train:
            dense_lane_targets = torch.zeros([batch_size, dense], device=device, dtype=torch.long)
            for i in range(batch_size):
                dense_lane_targets[i, :] = torch.tensor(np.array(mapping[i]['dense_lane_labels']), dtype=torch.long, device=device)
            loss_weight = self.args.lane_loss_weight
            dense_lane_targets = dense_lane_targets.view(-1) # [N*H]
            loss += loss_weight*F.nll_loss(dense_lane_pred, dense_lane_targets, reduction='none').\
                    view(batch_size, dense).sum(dim=1)
        mink = self.args.topk
        dense_lane_topk = torch.zeros((dense_lane_pred.shape[0], mink, self.hidden_size), device=device) # [N*dense, mink, hidden_size]
        dense_lane_topk_scores = torch.zeros((dense_lane_pred.shape[0], mink), device=device)   # [N*dense, mink]
        # dense_lane_pred = dense_lane_pred.exp()
        for i in range(dense_lane_topk_scores.shape[0]):
            idxs_lane = i // dense
            k = min(mink, lane_states_length[idxs_lane])
            _, idxs_topk = torch.topk(dense_lane_pred[i], k)
            dense_lane_topk[i][:k] = lane_states_batch[idxs_lane, idxs_topk] # [N*dense, mink, hidden_size]
            dense_lane_topk_scores[i][:k] = dense_lane_pred[i][idxs_topk] # [N*dense, mink]
        dense_lane_topk = torch.cat([dense_lane_topk, dense_lane_topk_scores.unsqueeze(-1)], dim=-1) # [N*dense, mink, hidden_size + 1]
        dense_lane_topk = dense_lane_topk.view(batch_size, dense*mink, self.hidden_size + 1) # [N, sense*mink, hidden_size + 1]
        return dense_lane_topk # [N, dense*mink, hidden_size + 1]

    def forward(self, mapping: List[Dict], batch_size, lane_states_batch, lane_states_length, inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param global_embed: hidden states of agents after encoding by global graph (shape [batch_size, hidden_size])
        :param local_embed: hidden states of agents before encoding by global graph (shape [batch_size, hidden_size])
        :param lane_states_batch: hidden states of lanes (shape [batch_size, max_num_lanes, hidden_size])
        """
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_steps])
        local_embed = inputs[:, 0, :]  # [batch_size, hidden_size]
        global_embed = hidden_states[:, 0, :] # [batch_size, hidden_size]
        if "step_lane_score" in self.args.other_params:
            dense_lane_topk = self.dense_lane_aware\
            (0, mapping, lane_states_batch, lane_states_length, local_embed, inputs_lengths, global_embed, device, loss) # [N, dense*mink, hidden_size + 1]
            dense_lane_topk = dense_lane_topk.permute(1, 0, 2)  # [dense*mink, N, hidden_size + 1]
            dense_lane_topk = self.proj_topk(dense_lane_topk) # [dense*mink, N, hidden_size]
            global_embed_att = global_embed + self.aggregation_cross_att(global_embed.unsqueeze(0), dense_lane_topk).squeeze(0) # [N, D]
            global_embed = torch.cat([global_embed, global_embed_att], dim=-1) # [N, 2*D]
        local_embed = local_embed.repeat(self.num_modes, 1, 1)  # [F, N, D]
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
        batch_size = global_embed.shape[0]
        global_embed = global_embed.transpose(0, 1)  # [F, N, D] 
        # if "stage_two" in self.args.other_params:
        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()  # [N, F]
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]

        z_size = self.z_size
        z = torch.randn(self.num_modes*batch_size,  z_size, device=device) # [F*N, 5]
        global_embed = torch.cat([global_embed, z], dim=-1)  # [F x N, D+z_size]
        global_embed = self.aggregate_global_z(global_embed)  # [F x N, D]
        
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0)+ 1.0 + self.min_scale  # [F x N, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if "stage_two" in self.args.other_params:
            past_traj = utils.get_from_mapping(mapping, 'past_traj') #[N, T, 2]]
            past_traj = torch.tensor(np.array(past_traj), dtype=torch.float32, device=device)
            past_traj = past_traj[:,:,:2]
            past_traj = past_traj.expand(self.num_modes, *past_traj.shape)  # [F, N, T, 2]
            full_traj = torch.cat((past_traj, loc), dim=2) # [F, N, H+T, 2]
            loc_delta, _ = self.trajectory_refinement(out, full_traj, global_embed, local_embed) #  [N, F, H], [F, N, H, 2]
        if "stage_two" in self.args.other_params:
            return self.laplace_decoder_loss((loc, loc_delta, past_traj), scale, pi, labels_is_valid, loss, DE, device, labels, mapping)
        else:
            return self.laplace_decoder_loss(loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping)


    def laplace_decoder_loss(self, loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping=None):
        if "stage_two" in self.args.other_params:
            original_loc, loc_delta, past_traj = loc
            loc = original_loc + loc_delta
        y_hat = torch.cat((loc, scale), dim=-1)
        batch_size = y_hat.shape[1]
        labels = torch.tensor(np.array(labels), device = device)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - labels, p=2, dim=-1) ).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
        if "stage_two" in self.args.other_params and self.args.do_train:
            loc_delta_best = loc_delta[best_mode, torch.arange(y_hat.shape[1])]
            delta_label = labels-original_loc[best_mode, torch.arange(y_hat.shape[1])]
            reg_delta_loss = torch.norm(loc_delta_best-delta_label, p=2, dim=-1) # [N, H]
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1) + 5*reg_delta_loss  # [N, H]
            loss += get_angle_diff(labels, y_hat_best[:, :, :2], past_traj)*2
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach() #[N, F]
            cls_loss = self.cls_loss(pi, soft_target)
        else:
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1)
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach() #[N, F]
            cls_loss = self.cls_loss(pi, soft_target)
        if self.args.do_train:
            for i in range(batch_size):
                if self.args.do_train:
                    assert labels_is_valid[i][-1]
                loss_ = reg_loss[i]
                loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_steps, 1)
                if labels_is_valid[i].sum() > utils.eps:
                    loss[i] += loss_.sum() / labels_is_valid[i].sum()
                loss[i] += cls_loss[i]
        if self.args.do_eval:
            outputs = loc.permute(1, 0, 2, 3).detach()
            pred_probs = F.softmax(pi, dim=-1).cpu().detach().numpy()
            for i in range(batch_size):
                if self.args.visualize:
                    labels = utils.get_from_mapping(mapping, 'labels')
                    labels = np.array(labels)
                    utils.visualize_gifs(
                        mapping[i], self.args.future_frame_num,
                        labels[i], outputs[i].cpu().numpy())
                outputs[i] = utils.to_origin_coordinate(outputs[i], i)
                if "vis_nuscenes" in self.args.other_params:
                    from utils_files import vis_nuscenes
                    vis_nuscenes.generate_nuscenes_gif(mapping[i], self.args.future_frame_num, outputs[i].cpu().numpy())
            outputs = outputs.cpu().numpy()
            return outputs, pred_probs, None

        return loss.mean(), DE, None






