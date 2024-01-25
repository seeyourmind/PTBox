# -*- coding: utf-8 -*-
"""
@author: 123456
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from numpy.random import RandomState

from torch.distributions import uniform, normal, gumbel
    
    
class PTBox(nn.Module):
    def __init__(self, kg_cfg:dict, kg, cuda:bool=True, min_init_value=0., delta_init_value=1., tk=3, gumbel_beta=1., **kwargs) -> None:
        super().__init__()
        assert kg_cfg['mod'] in [0, 1], print('kg_cfg.mod must be set to 0 or 1.')
        self.kg = kg
        self.max_ent = kg_cfg['max_ent']
        self.max_rel = kg_cfg['max_rel']
        self.inp_dim = kg_cfg['inp_dim']
        self.inference_mod = True # kg_cfg['mod']
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.gumbel_beta = gumbel_beta
        self.min_init_value = min_init_value
        self.delta_init_value = delta_init_value
        self.euler_gamma = 0.57721566490153286060  # Euler's constant
        self.time_k = tk
        self.gpu = cuda
        
        # entity box embedding 
        min_embedding = self.__data_init(self.max_ent, self.inp_dim, self.min_init_value)
        delta_embedding = self.__data_init(self.max_ent, self.inp_dim, self.delta_init_value)
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        
        # time embedding
        time_embedding = self.__data_init(self.time_k, self.inp_dim)
        self.time_embedding = nn.Parameter(time_embedding)
        self.time_smooth = nn.Sequential(nn.Linear(1, self.time_k),
                                         nn.ReLU(),
                                         nn.Linear(self.time_k, self.time_k),
                                         nn.Sigmoid())  # Sigmoid
        
        # relation projection embedding
        rel_trans_for_head = torch.empty(self.max_rel, self.inp_dim)
        rel_scale_for_head = torch.empty(self.max_rel, self.inp_dim)
        torch.nn.init.normal_(rel_trans_for_head, mean=0, std=1e-4)  # 1e-4 before
        torch.nn.init.normal_(rel_scale_for_head, mean=1, std=0.2)  # 0.2 before

        rel_trans_for_tail = torch.empty(self.max_rel, self.inp_dim)
        rel_scale_for_tail = torch.empty(self.max_rel, self.inp_dim)
        torch.nn.init.normal_(rel_trans_for_tail, mean=0, std=1e-4)
        torch.nn.init.normal_(rel_scale_for_tail, mean=1, std=0.2)
        
        self.rel_trans_for_head, self.rel_trans_for_tail = nn.Parameter(rel_trans_for_head), nn.Parameter(rel_trans_for_tail)
        self.rel_scale_for_head, self.rel_scale_for_tail = nn.Parameter(rel_scale_for_head), nn.Parameter(rel_scale_for_tail)
        
        self.to(self.device)
        
    def __data_init(self, maxN, embed_dim, init_value=None):
        if init_value is None:
            distribution = normal.Normal(0., 1.)
        else:
            distribution = uniform.Uniform(init_value[0], init_value[1])
        embed = distribution.sample((maxN, embed_dim))
        return embed
    
    def __get_entity_boxes(self, ent_ids):
        min_rep = self.min_embedding[ent_ids]
        delta_rep = self.delta_embedding[ent_ids]
        max_rep = min_rep + torch.exp(delta_rep)
        boxes = self.Box(min_rep, max_rep)
        return boxes
    
    def __intersection(self, boxes1, boxes2):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(boxes1.min_embed, boxes2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(boxes1.max_embed, boxes2.max_embed)
        )

        intersection_box = self.Box(intersections_min, intersections_max)
        return intersection_box
    
    def log_volumes(self, boxes, temp=1., scale:float=1.):
        eps = torch.finfo(boxes.min_embed.dtype).tiny
        s = torch.tensor(scale)
        
        log_vol = torch.sum(torch.log(F.softplus(boxes.delta_embed - 2*self.euler_gamma*self.gumbel_beta, beta=temp).clamp_min(eps)), dim=-1) + torch.log(s)
        
        return log_vol.view(-1,1)
    
    def time_rel_transform(self, boxes, rel, time, flag=1):
        relu = nn.ReLU()
        if time.size(-1)!=self.time_k:
            time = time
            time_dec = self.time_smooth(1.0*time.unsqueeze(1))
        else:
            time_dec = time
        time = torch.einsum('bk,kd->bd',time_dec,self.time_embedding)
        
        if flag>0:
            translations = self.rel_trans_for_head[rel]
            scales = self.rel_scale_for_head[rel]
        else:
            translations = self.rel_trans_for_tail[rel]
            scales = self.rel_scale_for_tail[rel]
        # map translate
        triple_ele = translations
        inner_prod = torch.sum(triple_ele*time, dim=1).unsqueeze(1).repeat(1, self.inp_dim)
        translations = triple_ele - time*inner_prod
        # map scale
        triple_ele = scales
        inner_prod = torch.sum(triple_ele*time, dim=1).unsqueeze(1).repeat(1, self.inp_dim)
        scales = triple_ele - time*inner_prod
        # affine transformation
        boxes.min_embed += translations
        boxes.delta_embed *= scales
        boxes.max_embed = boxes.min_embed + boxes.delta_embed
        return boxes
    
    def normalize_time(self):
        normalization_weight = F.normalize(self.time_embedding.data, p=2, dim=1)
        self.time_embedding.data = normalization_weight
        
    def forward(self, samples):
        # get indexs
        heads = torch.from_numpy(samples[:,0].astype(np.int64)).to(self.device)
        tails = torch.from_numpy(samples[:,1].astype(np.int64)).to(self.device)
        rels = torch.from_numpy(samples[:,2].astype(np.int64)).to(self.device)
        if self.inference_mod:
            Ts = torch.from_numpy(samples[:,3].astype(np.int64)).to(self.device)
        else:
            Ts = torch.from_numpy(samples[:,3].astype(np.float32)).to(self.device)
        
        # get box embeddings
        h_boxes = self.__get_entity_boxes(heads)
        t_boxes = self.__get_entity_boxes(tails)

        h_boxes = self.time_rel_transform(h_boxes, rels, Ts, flag=1)
        t_boxes = self.time_rel_transform(t_boxes, rels, Ts, flag=-1)
        intersection_boxes = self.__intersection(h_boxes, t_boxes)
        
        log_intersection = self.log_volumes(intersection_boxes)
        log_pred = torch.cat([log_intersection - self.log_volumes(h_boxes), log_intersection - self.log_volumes(t_boxes)], dim=1).min(dim=1).values
        score = log_pred.squeeze()

        return score
    
    def get_log_loss(self, y_pos, y_neg, temp=1.0):
        p = F.softmax(temp*y_neg)
        loss_pos = torch.sum(F.softplus(y_pos))
        loss_neg = torch.sum(p*F.softplus(y_neg))
        loss = (loss_neg+loss_pos).mean()
        if self.gpu:
            loss = loss.cuda()
        return loss

    def get_probability_loss(self, y_pos, y_neg):
        M, N = y_pos.size(0), y_neg.size(0)
        target_pos = Variable(torch.from_numpy(np.ones(M, dtype=np.float32))).to(self.device)
        target_neg = Variable(torch.from_numpy(np.zeros(N, dtype=np.float32))).to(self.device)
        loss = nn.MSELoss(reduction='mean')
        loss = loss(torch.exp(y_pos), target_pos) + loss(torch.exp(y_neg), target_neg)
        return loss
    
    def get_rank_loss(self, y_pos, y_neg):
        target_pos = Variable(torch.ones_like(y_pos), requires_grad=False)
        target_neg = Variable(torch.zeros_like(y_neg), requires_grad=False)
        xs = torch.cat([y_pos, y_neg], dim=0)
        ys = torch.cat([target_pos, target_neg], dim=0)
        lfuc = nn.PoissonNLLLoss()
        loss = lfuc(xs, ys)
        return loss
    
    def get_loss(self, y_pos, y_neg, temp=1.0):
        loss = self.get_rank_loss(y_pos,y_neg)
        return loss
    
    def rank_left(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    i_score = torch.zeros(self.kg.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3],triple[4]]:
                        for i in range(0, self.kg.n_entity):
                            X_i[i, 0] = i
                            X_i[i, 1] = triple[1]
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        # i_score = i_score + -1*self.forward(X_i).view(-1)
                        if time_index == triple[3]:
                            i_score = -1*self.forward(X_i).view(-1,1)
                        else:
                            i_score = torch.cat([i_score, -1*self.forward(X_i).view(-1,1)], dim=-1)
                        if rev_set>0:
                            X_rev = np.ones([self.kg.n_entity,4])
                            for i in range(0, self.kg.n_entity):
                                X_rev[i, 0] = triple[1]
                                X_rev[i, 1] = i
                                X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                                X_rev[i, 3] = time_index
                            # i_score =  i_score + -1*self.forward(X_rev).view(-1)
                            i_score = torch.cat([i_score, -1*self.forward(X_rev).view(-1,1)], dim=-1)
                    i_score = torch.mean(i_score, dim=-1)
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]                            
                    target = i_score[int(triple[0])].clone()
                    i_score = i_score<target
                    i_score[filter_out] = False 
                    rank_triple=torch.sum(i_score.float()).cpu().item()+1
                    rank.append(rank_triple)
                        
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = -1*self.forward(X_i)
                    if rev_set>0:
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + -1*self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]
                    target = i_score[int(triple[0])].clone()
                    i_score = i_score<target
                    i_score[filter_out] = False 
                    rank_triple=torch.sum(i_score.float()).cpu().item()+1
                    rank.append(rank_triple)
        return rank

    def rank_right(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    i_score = torch.zeros(self.kg.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3],triple[4]]:
                        for i in range(0, self.kg.n_entity):
                            X_i[i, 0] = triple[0]
                            X_i[i, 1] = i
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        # i_score = i_score + -1*self.forward(X_i).view(-1)
                        if time_index == triple[3]:
                            i_score = -1*self.forward(X_i).view(-1,1)
                        else:
                            i_score = torch.cat([i_score, -1*self.forward(X_i).view(-1,1)], dim=-1)
                        if rev_set>0:
                            X_rev = np.ones([self.kg.n_entity,4])
                            for i in range(0, self.kg.n_entity):
                                X_rev[i, 0] = i
                                X_rev[i, 1] = triple[0]
                                X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                                X_rev[i, 3] = time_index
                            # i_score = i_score + -1*self.forward(X_rev).view(-1)
                            i_score = torch.cat([i_score, -1*self.forward(X_rev).view(-1,1)], dim=-1)
                    i_score = torch.mean(i_score, dim=-1)
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]
                    target = i_score[int(triple[1])].clone()
                    i_score = i_score<target
                    i_score[filter_out] = False 
                    rank_triple=torch.sum(i_score.float()).cpu().item()+1
                    rank.append(rank_triple)
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = -1*self.forward(X_i)
                    if rev_set>0: 
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + -1*self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]       
                    target = i_score[int(triple[1])].clone()
                    i_score = i_score<target
                    i_score[filter_out] = False 
                    rank_triple=torch.sum(i_score.float()).cpu().item()+1
                    rank.append(rank_triple)

        return rank
    
    class Box:
        # Gumbel-Box
        def __init__(self, min_embed, max_embed) -> None:
            self.min_embed = min_embed
            self.max_embed = max_embed
            self.delta_embed = max_embed - min_embed
