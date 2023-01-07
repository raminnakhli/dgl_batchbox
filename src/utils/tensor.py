import torch


def aggregate_with_index(feature, index, batch_size):
    index = index.type(torch.LongTensor).to(feature.device)
    output = sum_with_index(feature, index, batch_size)
    output = output / torch.bincount(index).reshape(-1, 1)
    return output.squeeze(dim=-1)


def sum_with_index(feature, index, batch_size):
    dtype = feature.dtype
    device = feature.device

    feature = feature.type(torch.double)
    index = index.type(torch.LongTensor).to(device)
    if len(feature.size()) == 1:
        feature = feature.reshape(-1, 1)

    correspondence = torch.zeros((batch_size, feature.size(0)))
    for i in range(correspondence.size(0)):
        correspondence[i] = (index == i).squeeze()
    correspondence = correspondence.type(torch.double).to(device)
    output = torch.matmul(correspondence, feature).type(dtype)
    return output