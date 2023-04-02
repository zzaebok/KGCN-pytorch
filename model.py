import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from aggregator import Aggregator

class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator, args.mixer)
        
        self._gen_adj()

        self.usr = torch.nn.Embedding(num_user, args.dim)  # Generate num_user embeddings in args.dim dimension
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        
    def _gen_adj(self):
        '''
        Generate adjacency entities and their corresponding relations.
        Specifically, for a given entity, this can produce a fixed number of its neighboring entities and connected relations.
        Only cares about fixed number of samples, self.n_neighbor.
        '''

        # torch.empty can play the same role as torch.zeros but sometimes it may cause IndexError
        # Suggestion: Do not change it.
        self.adj_ent = torch.zeros(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.zeros(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:

            # These two conditions assure a fixed number (i.e. self.n_neighbor) of sampling
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        
    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
        # [batch_size] -> [batch_size, 1]
        u = u.view((-1, 1))  # the size -1 is inferred from other dimensions
        v = v.view((-1, 1))
        
        # [batch_size, 1, dim] -> [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        
        entities, relations = self._get_neighbors(v)
        
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        
        scores = (user_embeddings * item_embeddings).sum(dim = 1)
            
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h].cpu()]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h].cpu()]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))