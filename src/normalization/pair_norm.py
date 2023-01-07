import torch
import dgl
import dgl.nn.pytorch.glob

from src.utils.tensor import aggregate_with_index

class PairNorm(torch.nn.Module):
    """PairNorm from https://arxiv.org/abs/1909.12223"""
    def __init__(self, scale: float = 1., scale_individually: bool = False, eps: float = 1e-5):
        super().__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps
        self.avg = dgl.nn.pytorch.glob.AvgPooling()

    def forward(self, graph, x):
        scale = self.scale
        batch_size = len(graph.batch_num_nodes())

        mean = torch.repeat_interleave(self.avg(graph, x), graph.batch_num_nodes(), dim=0)
        x = x - mean

        if not self.scale_individually:
            index = torch.repeat_interleave(torch.arange(batch_size, device=graph.device), graph.batch_num_nodes(), dim=0)
            variance = torch.repeat_interleave(aggregate_with_index(x.pow(2).sum(-1), index, batch_size), graph.batch_num_nodes(), dim=0)
            return scale * x / (self.eps + variance).sqrt().reshape(-1, 1)
        else:
            raise NotImplementedError('has to be implemented')

    def __repr__(self):
        return f'{self.__class__.__name__}()'