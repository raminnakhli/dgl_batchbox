import dgl
import torch

def add_self_loop(graph: dgl.DGLGraph):
    """Addes self-loop to the graph while keeping the batching information
    Args:
        graph (dgl.DGLGraph): Input graph to add the self-loop edges to
    Returns:
        (dgl.DGLGraph) the input graph with self-loop edges being added to it
    Note: This function changes the order of the edges
    """
    
    assert len(graph.canonical_etypes) == 1, "hetero graph is not supported for now"

    # remove the current self-loops
    graph = dgl.remove_self_loop(graph)

    # keep batch info
    next_batch_nodes = graph.batch_num_nodes()
    next_batch_edges = graph.batch_num_edges() + next_batch_nodes

    # add self-loops
    graph = dgl.add_self_loop(graph)

    # reorder based on node to make sure self-loops are added to their corresponding batch
    graph = dgl.reorder_graph(graph, edge_permute_algo='src', store_ids=False)

    # set batch info
    graph.set_batch_num_nodes(next_batch_nodes)
    graph.set_batch_num_edges(next_batch_edges)

    return graph

def to_bidirected(graph, copy_ndata=True):
    """Makes the graph bidirected while keeping the batching information
    Args:
        graph (dgl.DGLGraph): Input graph to add the bidirected edges to
        copy_ndata (bool, optional): Copy the node data (default: True)
    Returns:
        (dgl.DGLGraph) the input graph with bidirected edges being added to it
    Note: This function changes the order of the edges
    """

    assert len(graph.canonical_etypes) == 1, "hetero graph is not supported for now"

    # get batch info
    num_nodes = graph.batch_num_nodes()

    # to bidirected
    graph = dgl.to_bidirected(graph.cpu(), copy_ndata=copy_ndata).to(graph.device)

    # reorder based on node to make sure self-loops are added to their corresponding batch
    graph = dgl.reorder_graph(graph, edge_permute_algo='src', store_ids=False)

    # convert the edge numbers to node numbers
    batch_nodes_bin = torch.nn.functional.pad(torch.cumsum(num_nodes, dim=0), (1, 0), "constant", 0)
    batch_edge_count = torch.histogram(graph.edges()[0].cpu().type(torch.FloatTensor), batch_nodes_bin.cpu().type(torch.FloatTensor))[0]
    batch_edge_count = batch_edge_count.type(torch.LongTensor).to(graph.device)

    # set batch info
    graph.set_batch_num_nodes(num_nodes)
    graph.set_batch_num_edges(batch_edge_count)

    return graph