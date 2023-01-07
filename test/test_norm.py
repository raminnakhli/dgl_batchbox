import unittest

import dgl
import torch

from src.normalization.pair_norm import PairNorm


def pytorch_pairnorm(x, scale=1, eps=1e-5):
    x = x - x.mean(dim=0, keepdim=True)
    return scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()


class TestPairNorm(unittest.TestCase):

    def test_validity(self):
        test_cases = [{
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1, 1], [1, 0, 1])), dgl.graph(([0, 0, 1, 1], [1, 0, 0, 2]))]),
                'features': [torch.Tensor([[1, 1, 1], [3, 3, 3]]), torch.Tensor([[0, 0, 0], [-1, -1, -1], [2, 2, 2]])]
            }
        }, 
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1, 1, 2, 3, 4], [1, 0, 1, 3, 4, 2])), dgl.graph(([0, 0, 1, 1], [1, 0, 0, 2]))]),
                'features': [torch.Tensor([[1, 1, 1], [3, 3, 3], [4, 5, 6], [0, 5, 7], [1, 8, 9]]), torch.Tensor([[0, 0, 0], [-1, -1, -1], [2, 2, 2]])]
            }
        }]

        norm = PairNorm()

        for case in test_cases:
            exp = torch.cat([pytorch_pairnorm(a) for a in case['input']['features']])
            features = norm(case['input']['graph'], torch.cat(case['input']['features']).type(torch.FloatTensor))
            assert torch.allclose(input=exp, other=features, ), f"exp: {exp}, got: {features}, diff: {exp - features}"


if  __name__ == '__main__':
    unittest.main()