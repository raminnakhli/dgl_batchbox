import unittest

import dgl
import torch

from src.pooling.sagpool import SAGPool


class TestSAGPooling(unittest.TestCase):

    def test_validity(self):
        test_cases = [{
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1, 1], [1, 0, 1])), dgl.graph(([0, 0, 1], [1, 0, 0]))]),
                'features': torch.Tensor([[1], [2], [0], [-1]])
            },
            'output': {
                'graph': dgl.graph(([0, 1], [0, 1])),
                'num_nodes': torch.tensor([1, 1]),
                'num_edges': torch.tensor([1, 1]),
                'features': torch.tensor([[4], [0]])
            }
        },
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 0, 1, 2, 4], [0, 1, 0, 3, 3])), dgl.graph(([0, 1, 1, 2, 2, 3], [1, 0, 1, 2, 3, 2]))]),
                'features': torch.tensor([[2], [-2], [3], [2], [1], [-1], [-2], [-3], [-3]])
            },
            'output': {
                'graph': dgl.graph(([0, 1, 3, 4, 4], [2, 1, 4, 3, 4])),
                'num_nodes': torch.tensor([3, 2]),
                'num_edges': torch.tensor([2, 3]),
                'features': torch.tensor([[9], [4], [4], [1], [4]])
            }
        }
        ]

        class Identity(torch.nn.Identity):

            def forward(self, graph, feature):
                return feature

        pooling = SAGPool(in_dim=0, conv_op=Identity, non_linearity=torch.nn.Identity())

        for case in test_cases:
            
            graph, feature, _ = pooling(case['input']['graph'], case['input']['features'].type(torch.FloatTensor))

            assert torch.equal(input=graph.edges()[0], other=case['output']['graph'].edges()[0]), graph.edges()[0]
            assert torch.equal(input=graph.edges()[1], other=case['output']['graph'].edges()[1]), graph.edges()[1]
            assert torch.equal(input=graph.nodes(), other=case['output']['graph'].nodes())
            assert torch.equal(input=graph.batch_num_nodes(), other=case['output']['num_nodes'])
            assert torch.equal(input=graph.batch_num_edges(), other=case['output']['num_edges'])
            assert torch.equal(input=feature, other=case['output']['features'].type(torch.FloatTensor))
            
if  __name__ == '__main__':
    unittest.main()