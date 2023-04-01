# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
import torch
import torch.nn as nn
from modeling.vectornet import *

class trajectory_refinement(nn.Module):
    def __init__(self,args):    
        super(trajectory_refinement, self).__init__()       
        self.args = args    
        self.hidden_size = args.hidden_size
        self.num_modes = args.mode_num
        self.MLP = MLP(2, self.hidden_size)
        self.MLP_2 = MLP(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc_delta = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        
    def forward(self, stage_one_out, full_traj, global_embed, local_embed):
        """two stage motion refinement module like faster rcnn and cascade rcnn
            parameters:stage_one_out: out feature embedding from GRU [F x N, H, D]
                        full_traj: past trajectory [F, N, H+T, 2]
                        local_embed: [1, F x N, D]
                        global_embed: [H, F x N, D]
        """         
        sequence_length = full_traj.shape[2]
        full_traj_embed = self.MLP(full_traj) # [F, N, H+T, D]
        full_traj_embed = self.MLP_2(full_traj_embed) + full_traj_embed # [F, N, H+T, D]
        full_traj_embed = full_traj_embed.view(-1, full_traj.shape[2], self.hidden_size) # [F*N, (H+T), D]
        full_traj_embed = full_traj_embed.permute(1, 0, 2) # [(H+T), F*N, D]
        stage_two_out, _ = self.gru(full_traj_embed) 
        stage_two_out = stage_two_out.permute(1, 0, 2)  # [F x N, H+T, D]
        stage_two_out_pred = stage_two_out[:, int(sequence_length-self.args.future_frame_num):, :] # [F x N, H, D]
        pi = None
        global_embed = global_embed.permute(1, 0, 2) # [F*N, H, D]
        local_embed = local_embed.squeeze(0) # [F x N, D]
        local_embed = local_embed.expand(self.args.future_frame_num, *local_embed.shape)# [H, F x N, D]
        local_embed = local_embed.permute(1, 0, 2) # [F*N, H, D]
        loc_delta = self.loc_delta(torch.cat((stage_one_out, stage_two_out_pred, global_embed, local_embed), dim=-1))  # [F*N, H, 2]
        loc_delta = loc_delta.view(self.num_modes, -1, self.args.future_frame_num, 2)  # [F, N, H, 2]
    
        return (loc_delta, pi)  # [F, N, H, 2]