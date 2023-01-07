import unittest

import dgl
import torch

from src.graph_op.edge import add_self_loop, to_bidirected



class TestBatchInfo(unittest.TestCase):

    def test_add_self_loop(self):
        test_cases = [{
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1], [1, 0])), dgl.graph(([0, 1], [1, 0]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 0, 1, 1, 2, 2, 3, 3], [1, 0, 0, 1, 3, 2, 2, 3])),
                'num_nodes': torch.tensor([2, 2]),
                'num_edges': torch.tensor([4, 4]),
            }
        },
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 0, 1], [0, 1, 0])), dgl.graph(([0, 1, 1], [1, 0, 1]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 0, 1, 1, 2, 2, 3, 3], [1, 0, 0, 1, 3, 2, 2, 3])),
                'num_nodes': torch.tensor([2, 2]),
                'num_edges': torch.tensor([4, 4]),
            }
        },
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 0, 1, 2, 4], [0, 1, 0, 3, 3])), dgl.graph(([0, 1, 1, 2], [1, 0, 1, 2]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7], [1, 0, 0, 1, 3, 2, 3, 3, 4, 6, 5, 5, 6, 7])),
                'num_nodes': torch.tensor([5, 3]),
                'num_edges': torch.tensor([9, 5]),
            }
        }]

        for case in test_cases:
            graph = case['input']['graph']
            graph = add_self_loop(graph)
            assert torch.equal(input=graph.edges()[0], other=case['output']['graph'].edges()[0]), graph.edges()[0]
            assert torch.equal(input=graph.edges()[1], other=case['output']['graph'].edges()[1]), graph.edges()[1]
            assert torch.equal(input=graph.nodes(), other=case['output']['graph'].nodes())
            assert torch.equal(input=graph.batch_num_nodes(), other=case['output']['num_nodes'])
            assert torch.equal(input=graph.batch_num_edges(), other=case['output']['num_edges'])


class TestToBidirected(unittest.TestCase):

    def test_batch_info(self):
        test_cases = [{
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1], [1, 0])), dgl.graph(([0, 1, 1], [0, 0, 1]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 1, 2, 2, 3, 3], [1, 0, 2, 3, 2, 3])),
                'num_nodes': torch.tensor([2, 2]),
                'num_edges': torch.tensor([2, 4]),
            }
        },
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 1], [0, 0])), dgl.graph(([0, 1], [1, 1]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 0, 1, 2, 3, 3], [0, 1, 0, 3, 2, 3])),
                'num_nodes': torch.tensor([2, 2]),
                'num_edges': torch.tensor([3, 3]),
            }
        },
        {
            'input': {
                'graph': dgl.batch([dgl.graph(([0, 0, 2, 4], [0, 1, 3, 3])), dgl.graph(([0, 1, 2], [1, 1, 2]))]),
            },
            'output': {
                'graph': dgl.graph(([0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7], [0, 1, 0, 3, 2, 4, 3, 6, 5, 6, 7])),
                'num_nodes': torch.tensor([5, 3]),
                'num_edges': torch.tensor([7, 4]),
            }
        }
        ]

        for case in test_cases:
            graph = case['input']['graph']
            graph = to_bidirected(graph)
            assert torch.equal(input=graph.edges()[0], other=case['output']['graph'].edges()[0]), f"got: {graph.edges()[0]}, expected: {case['output']['graph'].edges()[0]}"
            assert torch.equal(input=graph.edges()[1], other=case['output']['graph'].edges()[1]), f"got: {graph.edges()[1]}, expected: {case['output']['graph'].edges()[1]}"
            assert torch.equal(input=graph.nodes(), other=case['output']['graph'].nodes()), f"got: {graph.nodes()}, expected: {case['output']['graph'].nodes()}"
            assert torch.equal(input=graph.batch_num_nodes(), other=case['output']['num_nodes']), f"got: {graph.batch_num_nodes()}, expected: {case['output']['num_nodes']}"
            assert torch.equal(input=graph.batch_num_edges(), other=case['output']['num_edges']), f"got: {graph.batch_num_edges()}, expected: {case['output']['num_edges']}"

if  __name__ == '__main__':
    unittest.main()