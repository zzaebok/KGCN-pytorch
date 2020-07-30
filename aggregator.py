import torch
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    
    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)        
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_mebeddings)
        
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.concat(self_vectors, neighbors_agg, dim = -1)
            output = output.view((-1, self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
        
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)
        
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)
        
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_score_normalized * neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated