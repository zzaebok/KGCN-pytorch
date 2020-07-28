import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy

class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.aggregator = args.aggregator
        self.n_iter = args.n_iter
        self.neighbor_sample_size = args.neighbor_sample_size
        self.kg = kg
        self._gen_adj()
        if args.aggregator == 'concat':
            self.agg_weight = torch.nn.Linear(2 * args.dim, args.dim, bias=True)
        else:
            self.agg_weight = torch.nn.Linear(args.dim, args.dim, bias=True)        
            
        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = np.empty((self.num_ent, self.neighbor_sample_size))
        self.adj_rel = np.empty((self.num_ent, self.neighbor_sample_size))
        
        for e in self.kg:
            if len(self.kg[e]) >= self.neighbor_sample_size:
                neighbors = random.sample(self.kg[e], self.neighbor_sample_size)
            else:
                neighbors = random.choices(self.kg[e], k=self.neighbor_sample_size)
                
            self.adj_ent[e] = [ent for _, ent in neighbors]
            self.adj_rel[e] = [rel for rel, _ in neighbors]
        
    def forward(self, u, v):
        '''
        u, v as indices 'batch_size' tensor
        vector operation should be executed with embedding ex) self.usr[u]
        '''
        batch_size = u.size()[-1]
        v_u_list = []
        for i in range(batch_size):
            m = self._get_receptive(v[i])
            # e_u_dict { 3: tensor([1,2,3,4,5]), None, None]} / entity embedding updated
            e_u_dict = {int(x): [self.ent(x) if i == 0 else None for i in range(self.n_iter+1)] for x in m[0]}
            for h in range(1, self.n_iter+1):
                for e in m[h]:
                    e_u_neighbor = self._message_passing(u[i], e)
                    e_u_dict[int(e)][h] = self._aggregate(e_u_neighbor, e_u_dict[int(e)][h-1])
            v_u = e_u_dict[int(v[i])][self.n_iter]
            v_u_list.append(v_u)
        v_u_batched = torch.stack(v_u_list)
        return F.sigmoid((self.usr(u) * v_u_batched).sum(dim=1))
    
    def _get_receptive(self, v):
        '''
        get receptive field inwardly
        '''
        # batch_size x H x [](?)
        m = [[] for _ in range(self.n_iter+1)]
        m[self.n_iter].append(v)
        for h in range(self.n_iter-1, -1, -1): # from H-1 to 0
            m[h] = copy.deepcopy(m[h+1])
            for e in m[h+1]:
                m[h].extend(self._get_neighbors(e))
        # list of list which contains tensor
        return m
    
    def _get_neighbors(self, e):
        '''
        return neighbors and relations
        '''
        return torch.LongTensor(self.adj_ent[e])
        
    def _message_passing(self, u, e):
        # array of (relation, entity_tail)
        neighbors = self._get_neighbors(e)
        relations = torch.LongTensor(self.adj_rel[e])
        weights = self._get_weight(u, relations)
        return sum([weight * self.ent(entity) for entity, weight in zip(neighbors, weights)])
    
    def _get_weight(self, u, relations):
        pi_u_r = torch.Tensor([torch.dot(self.usr(u), self.rel(r)) for r in relations])
        pi_u_r = F.softmax(pi_u_r)
        return pi_u_r
    
    def _aggregate(self, v, v_u):
        '''
        Return v^u vector after aggregate v and v^u (sampled)
        Equation 4,5,6 in the paper
        '''
        if self.aggregator == 'sum':
            v_u = self.agg_weight(v + v_u)
        elif self.aggregator == 'concat':
            v_u = self.agg_weight(torch.cat((v, v_u)))
        else:
            v_u = self.agg_weight(v_u)
        return F.relu(v_u)